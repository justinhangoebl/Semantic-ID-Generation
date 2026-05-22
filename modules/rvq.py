import torch
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from torch import nn, Tensor

from modules.quantization import Quantization
from schemas.rq_vae import RqVaeOutput, RqVaeComputedLosses


class RVQ(nn.Module, PyTorchModelHubMixin):
    """
    Residual Vector Quantization without encoder/decoder.
    Codebooks operate directly in input space; only commitment loss is used.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        n_quantization_layers: int = 3,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.n_quantization_layers = n_quantization_layers

        self.quantization_layers = nn.ModuleList([
            Quantization(
                latent_dim=input_dim,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight,
            )
            for _ in range(n_quantization_layers)
        ])

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _quantize(self, x: Tensor):
        residual = x.float()
        quantize_loss = torch.zeros(1, device=x.device)
        embs, residuals, sem_ids = [], [], []
        for layer in self.quantization_layers:
            residuals.append(residual)
            out = layer(residual)
            quantize_loss += out.loss
            residual = residual - out.hard_embeddings
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
        embs, residuals, sem_ids, quantize_loss = self._quantize(x)
        return RqVaeOutput(embeddings=embs, residuals=residuals, sem_ids=sem_ids, quantize_loss=quantize_loss)

    def forward(self, x: Tensor) -> RqVaeComputedLosses:
        embs, residuals, sem_ids, quantize_loss = self._quantize(x)
        commitment_loss = quantize_loss.mean()

        with torch.no_grad():
            p_unique_ids = (
                ~torch.triu(
                    (rearrange(sem_ids, 'b d -> b 1 d') == rearrange(sem_ids, 'b d -> 1 b d')).all(dim=-1),
                    diagonal=1,
                )
            ).all(dim=1).sum() / sem_ids.shape[0]

        quantized = RqVaeOutput(
            embeddings=embs,
            residuals=residuals,
            sem_ids=sem_ids,
            quantize_loss=quantize_loss,
        )
        return RqVaeComputedLosses(
            loss=commitment_loss,
            reconstruction_loss=torch.zeros(1, device=self.device),
            rqvae_loss=commitment_loss,
            p_unique_ids=p_unique_ids,
            quantized=quantized,
        )
