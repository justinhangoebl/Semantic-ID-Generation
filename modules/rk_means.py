import torch

from einops import rearrange
from modules.quantization import Quantization
from schemas.quantization import QuantizeForwardMode, QuantizeDistance
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch import Tensor
from schemas.rq_vae import RqVaeOutput


class RKMeans(nn.Module, PyTorchModelHubMixin):
    """
    Residual K-Means tokenizer.

    Converts a continuous item embedding into a sequence of discrete token IDs
    via repeated nearest-centroid lookup + residual subtraction. No encoder,
    no decoder, no reconstruction loss, no commitment loss — codebooks are fit
    once offline via K-Means (kmeans_init_codebooks) and then frozen.

    Compared to RQ-VAE this removes:
      - encoder / decoder networks
      - hidden_dims / latent_dim projection  (input_dim IS the codebook dim)
      - reconstruction loss
      - commitment loss
      - any gradient flow through the codebooks
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        n_quantization_layers: int = 3,
        quantization_method: QuantizeForwardMode = QuantizeForwardMode.STE,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
        normalize_quantizer_inputs: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.n_quantization_layers = n_quantization_layers
        self.quantization_method = quantization_method
        self.distance_mode = distance_mode
        self.normalize_quantizer_inputs = normalize_quantizer_inputs

        # One codebook per hierarchy level; commitment_weight=0 → no loss
        self.quantization_layers = nn.ModuleList([
            Quantization(
                latent_dim=input_dim,
                codebook_size=codebook_size,
                commitment_weight=0.0,
                do_kmeans_init=True,
            )
            for _ in range(n_quantization_layers)
        ])

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Offline training: fit codebooks with K-Means, then freeze
    # ------------------------------------------------------------------

    @torch.no_grad()
    def kmeans_init_codebooks(self, data: Tensor) -> None:
        """
        Fit each codebook in sequence using K-Means on the residuals.
        Call this ONCE before any inference; codebooks are frozen afterwards.

        Args:
            data: item embeddings, shape (N, input_dim)
        """
        residual = data.to(self.device).float()
        for layer in self.quantization_layers:
            layer._kmeans_init(residual)
            # subtract hard centroid to get residual for the next level
            ids = layer(residual).ids
            centroids = layer.get_item_embeddings(ids)   # (N, input_dim)
            residual = residual - centroids

    # ------------------------------------------------------------------
    # Inference: residual lookup — no gradients, no loss
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x: Tensor) -> RqVaeOutput:
        """
        Encode embeddings into hierarchical semantic IDs.

        Args:
            x: item embeddings, shape (B, input_dim)

        Returns:
            RqVaeOutput with:
              .sem_ids   — shape (B, n_quantization_layers), integer token per level
              .embeddings — shape (n_quantization_layers, input_dim, B), chosen centroids
              .residuals  — shape (n_quantization_layers, input_dim, B), residual at each level
              .quantize_loss — 0.0 (no loss)
        """
        residual = x.float()
        embs, residuals, sem_ids = [], [], []

        for layer in self.quantization_layers:
            residuals.append(residual)                          # residual going IN to this level
            quantized = layer(residual)
            emb = quantized.hard_embeddings                     # nearest centroid, shape (B, D)
            residual = residual - emb                           # residual for next level
            sem_ids.append(quantized.ids)                       # integer token, shape (B,)
            embs.append(emb)

        return RqVaeOutput(
            embeddings=rearrange(embs, "h b d -> h d b"),
            residuals=rearrange(residuals, "h b d -> h d b"),
            sem_ids=rearrange(sem_ids, "h b -> b h"),
            quantize_loss=torch.tensor(0.0, device=self.device),
        )