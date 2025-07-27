import torch
import torch.nn.functional as F

from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from typing import List
from torch import nn, Tensor

from modules.encoder import Encoder, Decoder
from modules.quantization import Quantization
from schemas.rq_vae import RqVaeOutput, RqVaeComputedLosses


class RQ_VAE(nn.Module, PyTorchModelHubMixin):
    """
    Residual Quantized Variational Autoencoder (RQ-VAE) for semantic ID generation.

    Supports both Straight-Through Estimation (STE) and Gumbel Softmax quantization
    methods for better gradient flow and joint training with language models.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        n_quantization_layers: int = 3,
        commitment_weight: float = 0.25,
        quantization_method: QuantizeForwardMode = QuantizeForwardMode.STE,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_quantization_layers = n_quantization_layers

        self.encoder = Encoder(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        self.decoder = Decoder(output_dim=input_dim, hidden_dims=hidden_dims[::-1], latent_dim=latent_dim)

        self.quantization_layers = nn.ModuleList([
            Quantization(latent_dim=latent_dim, codebook_size=codebook_size, commitment_weight=commitment_weight)
            for _ in range(n_quantization_layers)
        ])

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _quantize(self, x: Tensor):
        """Run residual quantization, return stacked embeddings/ids and total commitment loss."""
        res = x
        quantize_loss = torch.zeros(1, device=x.device)
        embs, residuals, sem_ids = [], [], []

        for layer in self.quantization_layers:
            residuals.append(res)
            out = layer(res)
            quantize_loss += out.loss
            res = res - out.hard_embeddings
            embs.append(out.embeddings)
            sem_ids.append(out.ids)

        return (
            rearrange(embs, 'h b d -> h d b'),
            rearrange(residuals, 'h b d -> h d b'),
            rearrange(sem_ids, 'h b -> b h'),
            quantize_loss,
        )

    @torch.no_grad()
    def get_semantic_ids(self, x: Tensor) -> RqVaeOutput:
        latent, _ = self.encoder(x)
        embs, residuals, sem_ids, quantize_loss = self._quantize(latent)
        return RqVaeOutput(embeddings=embs, residuals=residuals, sem_ids=sem_ids, quantize_loss=quantize_loss)

    def forward(self, x: Tensor) -> RqVaeComputedLosses:
        latent, x_norm = self.encoder(x)
        embs, residuals, sem_ids, quantize_loss = self._quantize(latent)

        # Reconstruct against the BatchNorm+L2-normalised input (GRID)
        x_hat = self.decoder(embs.sum(dim=0).T)   # (h, d, b) -> (d, b) -> (b, d)
        reconstruction_loss = F.mse_loss(x_hat, x_norm)
        rqvae_loss = quantize_loss.mean()
        loss = reconstruction_loss + rqvae_loss

        quantized = RqVaeOutput(embeddings=embs, residuals=residuals, sem_ids=sem_ids, quantize_loss=quantize_loss)

        with torch.no_grad():
            p_unique_ids = (
                ~torch.triu(
                    (rearrange(sem_ids, 'b d -> b 1 d') == rearrange(sem_ids, 'b d -> 1 b d')).all(dim=-1),
                    diagonal=1,
                )
            ).all(dim=1).sum() / sem_ids.shape[0]

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            rqvae_loss=rqvae_loss,
            p_unique_ids=p_unique_ids,
            quantized=quantized,
        )
