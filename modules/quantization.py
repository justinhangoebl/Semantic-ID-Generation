import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.cluster import KMeans

from schemas.quantization import QuantizeOutput, QuantizeForwardMode, QuantizeDistance


class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query: Tensor, value: Tensor) -> Tensor:
        emb_loss = ((query.detach() - value)**2).sum(axis=[-1])
        query_loss = ((query - value.detach())**2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss

class Quantization(nn.Module):
    """
    Vector Quantization module supporting both Straight-Through Estimation (STE)
    and Gumbel Softmax quantization methods.

    Args:
        latent_dim: Dimension of the latent vectors
        codebook_size: Number of vectors in the codebook
        commitment_weight: Weight for the commitment loss
        do_kmeans_init: Whether to initialize codebook with k-means
        sim_vq: Whether to use similarity-based VQ with projection layer
        forward_mode: Quantization method to use (QuantizeForwardMode enum)
        distance_mode: Distance metric to use (QuantizeDistance enum)

    Note:
        For Gumbel Softmax quantization, temperature should be passed to the forward() method.
    """
    def __init__(
        self,
        latent_dim: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        do_kmeans_init: bool = True,
        sim_vq: bool = False,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
    ) -> None:
        super().__init__()
        self.embed_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False) if sim_vq else nn.Identity(),
        )
        self.quantize_loss = QuantizeLoss(commitment_weight)
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights uniformly."""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device



    def set_quantization_method(self, method: QuantizeForwardMode) -> None:
        """Change the quantization method."""
        self.forward_mode = method
    
    @torch.no_grad
    def _kmeans_init(self, x: Tensor):
        x = x.view(-1, self.embed_dim).cpu().numpy()
        kmeans = KMeans(n_clusters=self.codebook_size, n_init=10, max_iter=300)
        kmeans.fit(x)
        
        self.embedding.weight.copy_(torch.from_numpy(kmeans.cluster_centers_).to(self.device))
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids) -> Tensor:
        return self.out_proj(self.embedding(item_ids))
    
    def get_codebook(self) -> Tensor:
        return self.out_proj(self.embedding.weight)

    def _compute_distances(self, x: Tensor) -> Tensor:
        """Compute distances between input vectors and codebook entries."""
        codebook = self.get_codebook()

        if self.distance_mode == QuantizeDistance.L2:
            # Compute squared L2 distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x,c>
            dist = (
                (x**2).sum(dim=1, keepdim=True) +
                (codebook**2).sum(dim=1, keepdim=True).T -
                2 * x @ codebook.T
            )
        elif self.distance_mode == QuantizeDistance.COSINE:
            # Compute negative cosine similarity (so min distance = max similarity)
            x_norm = x / x.norm(dim=1, keepdim=True)
            codebook_norm = codebook / codebook.norm(dim=1, keepdim=True)
            dist = -(x_norm @ codebook_norm.T)
        else:
            raise ValueError(f"Unsupported distance mode: {self.distance_mode}")

        return dist



    def forward(self, x: Tensor, temperature: float = 1.0) -> QuantizeOutput:
        """
        Forward pass with unified quantization logic.

        Args:
            x: Input tensor to quantize
            temperature: Temperature for Gumbel Softmax (ignored for STE)
        """
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x)

        # Compute distances to codebook entries
        dist = self._compute_distances(x)

        # Get discrete assignments (used by both methods)
        _, ids = dist.detach().min(dim=1)

        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                # Gumbel Softmax quantization
                codebook = self.get_codebook()

                # Convert distances to logits (negative distances for higher probability)
                logits = -dist / temperature

                # Apply Gumbel Softmax
                soft_assignment = F.gumbel_softmax(logits, tau=temperature, hard=False)
                emb = soft_assignment @ codebook
                emb_out = emb

                # For loss, use the closest codebook entry (like STE) to encourage commitment
                closest_emb = self.get_item_embeddings(ids)
                loss = self.quantize_loss(query=x, value=closest_emb)

            elif self.forward_mode == QuantizeForwardMode.STE:
                # Straight-Through Estimation
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()

                # Use the quantized embedding for loss
                loss = self.quantize_loss(query=x, value=emb)

            else:
                raise ValueError(f"Unsupported forward mode: {self.forward_mode}")
        else:
            # Evaluation mode: use hard assignment for both methods
            emb_out = self.get_item_embeddings(ids)

            # Compute loss for compatibility
            loss = self.quantize_loss(query=x, value=emb_out)

        return QuantizeOutput(
            embeddings=emb_out,
            ids=ids,
            loss=loss,
        )