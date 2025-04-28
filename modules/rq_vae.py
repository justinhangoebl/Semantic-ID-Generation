import torch

from einops import rearrange
from modules.encoder import Encoder, Decoder
from modules.quantization import Quantization
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
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.codebook_size = codebook_size
        self.codebook_kmeans_init = codebook_kmeans_init
        self.codebook_sim_vq = codebook_sim_vq
        self.commitment_weight = commitment_weight
        
        self.quantization_layers = nn.ModuleList(modules=[
            Quantization(
                latent_dim=latent_dim,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight,
                do_kmeans_init=codebook_kmeans_init,
                sim_vq=codebook_sim_vq,
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
    
    @torch.no_grad()
    def kmeans_init_codebooks(self, data: Tensor) -> None:
        """
        Initializes all quantization layers using k-means with full (or large) dataset.
        Call this before training.
        """
        x = self.encode(data.to(self.device).float())
        for layer in self.quantization_layers:
            layer._kmeans_init(x)
            emb = layer.get_item_embeddings(layer(x).ids)
            x = x - emb
        
    def get_semantic_id_single(self, x: Tensor) -> Tensor:
        res = self.encode(x.unsqueeze(0))  # Add batch dim (1, ...)
        
        sem_ids = []
        for layer in self.quantization_layers:
            quantized = layer(res)
            id = quantized.ids.squeeze(0)  # Remove batch dim
            res = res - quantized.embeddings
            sem_ids.append(id)

        return torch.stack(sem_ids, dim=0)  # shape: (num_layers, semantic_id_dim)

    def get_semantic_ids(self, x: Tensor) -> RqVaeOutput:
        res = self.encode(x)

        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []

        for layer in self.quantization_layers:
            residuals.append(res)
            quantized = layer(res)
            quantize_loss += quantized.loss
            emb, id = quantized.embeddings, quantized.ids
            res = res - emb  # Update residuals
            sem_ids.append(id)
            embs.append(emb)

        return RqVaeOutput(
            embeddings=rearrange(embs, "b h d -> h d b"),
            residuals=rearrange(residuals, "b h d -> h d b"),
            sem_ids=rearrange(sem_ids, "b d -> d b"),
            quantize_loss=quantize_loss
        )
        
    def forward(self, x) -> RqVaeComputedLosses:
        quantized = self.get_semantic_ids(x)
        embs = quantized.embeddings
        x_hat = self.decode(embs.sum(axis=-1))
        x_hat = torch.nn.functional.normalize(x_hat, p=2)

        reconstuction_loss = F.mse_loss(x_hat, x, reduction='sum')  # Using sum as the loss to match the previous behavior
        rqvae_loss = quantized.quantize_loss
        loss = (reconstuction_loss + rqvae_loss).mean()
        
        with torch.no_grad():
            # Compute debug ID statistics
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (~torch.triu(
                (rearrange(quantized.sem_ids, "b d -> b 1 d") == rearrange(quantized.sem_ids, "b d -> 1 b d")).all(axis=-1), diagonal=1)
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstuction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids
        )