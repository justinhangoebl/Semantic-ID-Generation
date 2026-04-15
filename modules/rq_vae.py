import torch

from einops import rearrange
from modules.encoder import Encoder, Decoder
from modules.quantization import Quantization
from schemas.quantization import QuantizeForwardMode, QuantizeDistance
from huggingface_hub import PyTorchModelHubMixin
from typing import List
from torch import nn
from torch import Tensor
from schemas.rq_vae import RqVaeOutput, RqVaeComputedLosses
import torch.nn.functional as F

class RQ_VAE(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_sim_vq: bool = True,
        n_quantization_layers: int = 3,
        commitment_weight: float = 0.25,
        quantization_method: QuantizeForwardMode = QuantizeForwardMode.STE,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
        normalize_quantizer_inputs: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.codebook_size = codebook_size
        self.codebook_kmeans_init = codebook_kmeans_init
        self.codebook_sim_vq = codebook_sim_vq
        self.commitment_weight = commitment_weight
        self.quantization_method = quantization_method
        self.n_quantization_layers = n_quantization_layers
        self.distance_mode = distance_mode
        self.normalize_quantizer_inputs = normalize_quantizer_inputs

        self.quantization_layers = nn.ModuleList(modules=[
            Quantization(
                latent_dim=latent_dim,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight,
                do_kmeans_init=codebook_kmeans_init,
                sim_vq=codebook_sim_vq,
                forward_mode=quantization_method,
                distance_mode=distance_mode,
                normalize_inputs=normalize_quantizer_inputs,
            )
            for _ in range(n_quantization_layers)
        ])
        
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
        
        self.decoder = Decoder(
            output_dim=input_dim,
            hidden_dims=hidden_dims[::-1],
            latent_dim=latent_dim,
        )
        
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def set_quantization_method(self, method: QuantizeForwardMode) -> None:
        self.quantization_method = method
        for layer in self.quantization_layers:
            layer.set_quantization_method(method)

    @torch.no_grad()
    def kmeans_init_codebooks(self, data: Tensor, temperature: float = 1.0) -> None:
        x = self.encode(data.to(self.device).float())
        for layer in self.quantization_layers:
            layer._kmeans_init(x)
            emb = layer.get_item_embeddings(layer(x, temperature=temperature).ids)
            x = x - emb  # Always hard embeddings here since get_item_embeddings is direct lookup

    def get_semantic_ids(self, x: Tensor, temperature: float = 1.0) -> RqVaeOutput:
        res = self.encode(x)

        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []

        for layer in self.quantization_layers:
            residuals.append(res)
            quantized = layer(res, temperature=temperature)
            quantize_loss += quantized.loss
            emb, id = quantized.embeddings, quantized.ids
            res = res - quantized.hard_embeddings  # Residual uses hard (argmin) embedding
            sem_ids.append(id)
            embs.append(emb)

        return RqVaeOutput(
            embeddings=rearrange(embs, "h b d -> h d b"),
            residuals=rearrange(residuals, "h b d -> h d b"),
            sem_ids=rearrange(sem_ids, "h b -> b h"),
            quantize_loss=quantize_loss
        )

    def forward(self, x: Tensor, temperature: float = 1.0) -> "RqVaeComputedLosses":
        quantized = self.get_semantic_ids(x, temperature=temperature)
        embs = quantized.embeddings  # Shape: (h, d, b)
        x_hat = self.decode(embs.sum(dim=0).T)  # (h, d, b) -> (d, b) -> (b, d)
        x_hat = F.normalize(x_hat, p=2)

        reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')
        rqvae_loss = quantized.quantize_loss
        loss = (reconstruction_loss + rqvae_loss).mean()

        with torch.no_grad():
            embs_norm = embs.norm(dim=1).T  # (h, b) -> (b, h)
            p_unique_ids = (
                ~torch.triu(
                    (rearrange(quantized.sem_ids, "b d -> b 1 d") == rearrange(quantized.sem_ids, "b d -> 1 b d")).all(dim=-1),
                    diagonal=1,
                )
            ).all(dim=1).float().mean()

        from schemas.rq_vae import RqVaeComputedLosses
        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstruction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
            quantized=quantized,
        )