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


class CommitmentLoss(nn.Module):
    def __init__(self, beta: float = 0.25) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: Tensor, xq: Tensor) -> Tensor:
        # GRID BetaQuantizationLoss:
        #   codebook loss: pulls codebook toward encoder output
        #   commitment loss: pulls encoder toward codebook (weighted by beta)
        codebook_loss = F.mse_loss(x.detach(), xq, reduction='mean')
        commitment_loss = F.mse_loss(x, xq.detach(), reduction='mean')
        return codebook_loss + self.beta * commitment_loss


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
    ) -> None:
        super().__init__()
        self.embed_dim = latent_dim
        self.codebook_size = codebook_size
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.loss_fn = CommitmentLoss(beta=commitment_weight)
        nn.init.uniform_(self.embedding.weight)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def _kmeans_init(self, x: Tensor) -> None:
        flat = x.view(-1, self.embed_dim)
        n_unique = flat.unique(dim=0).shape[0]

        if n_unique >= self.codebook_size:
            kmeans = KMeans(n_clusters=self.codebook_size, init='k-means++', n_init=3, max_iter=300)
            kmeans.fit(flat.cpu().numpy())
            centers = torch.from_numpy(kmeans.cluster_centers_).float()
        else:
            # Too few distinct points for KMeans (collapsed residuals in deeper RQ layers).
            # Sample data points with replacement so each codebook entry starts at a real point
            # and diverges during training rather than staying as a single degenerate cluster.
            idx = torch.randint(0, flat.shape[0], (self.codebook_size,))
            centers = flat[idx].float().cpu()

        self.embedding.weight.copy_(centers.to(self.device))
        self.kmeans_initted = True

    def _l2_distances(self, x: Tensor) -> Tensor:
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x,c>
        codebook = self.embedding.weight
        return (
            (x ** 2).sum(dim=1, keepdim=True)
            + (codebook ** 2).sum(dim=1, keepdim=True).T
            - 2 * x @ codebook.T
        )

    def forward(self, x: Tensor) -> QuantizeOutput:
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x)

        dist = self._l2_distances(x)
        ids = dist.detach().argmin(dim=1)
        hard_emb = self.embedding(ids)

        # Straight-Through Estimator: forward uses hard embedding, backward flows through x
        emb_out = x + (hard_emb - x).detach()
        loss = self.loss_fn(x, hard_emb)

        return QuantizeOutput(embeddings=emb_out, hard_embeddings=hard_emb, ids=ids, loss=loss)
