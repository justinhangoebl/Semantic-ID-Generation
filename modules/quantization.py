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
    def __init__(
        self,
        latent_dim: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        do_kmeans_init: bool = True,
        sim_vq: bool = False,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
        revival_threshold: float = 1.0,
        revival_ema_decay: float = 0.97,
        normalize_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode
        self.revival_threshold = revival_threshold
        self.revival_ema_decay = revival_ema_decay
        self.normalize_inputs = normalize_inputs

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False) if sim_vq else nn.Identity(),
        )
        self.quantize_loss = QuantizeLoss(commitment_weight)
        # EMA usage counter - each entry starts at revival_threshold so we
        # don't evict anything immediately after k-means init.
        self.register_buffer(
            '_ema_usage', torch.full((codebook_size,), float(revival_threshold))
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_quantization_method(self, method: QuantizeForwardMode) -> None:
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
        codebook = self.get_codebook()

        if self.distance_mode == QuantizeDistance.L2:
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x,c>
            dist = (
                (x**2).sum(dim=1, keepdim=True) +
                (codebook**2).sum(dim=1, keepdim=True).T -
                2 * x @ codebook.T
            )
        elif self.distance_mode == QuantizeDistance.COSINE:
            x_norm = x / x.norm(dim=1, keepdim=True)
            codebook_norm = codebook / codebook.norm(dim=1, keepdim=True)
            dist = -(x_norm @ codebook_norm.T)
        else:
            raise ValueError(f"Unsupported distance mode: {self.distance_mode}")

        return dist

    def forward(self, x: Tensor, temperature: float = 1.0) -> QuantizeOutput:
        assert x.shape[-1] == self.embed_dim

        # Normalizing onto the unit sphere bounds L2 distances to [0, 2] and
        # makes them monotone with cosine similarity - essential for
        # high-dimensional inputs (e.g. Jukebox) where raw L2 is dominated by
        # magnitude variance and causes all vectors to collapse to one entry.
        if self.normalize_inputs:
            x = F.normalize(x, p=2, dim=-1)

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x)

        dist = self._compute_distances(x)
        _, ids = dist.detach().min(dim=1)

        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                codebook = self.get_codebook()
                logits = -dist / temperature
                soft_assignment = F.gumbel_softmax(logits, tau=temperature, hard=False)
                emb_out = soft_assignment @ codebook
                hard_emb = self.get_item_embeddings(ids)
                loss = self.quantize_loss(query=x, value=hard_emb)

            elif self.forward_mode == QuantizeForwardMode.STE:
                hard_emb = self.get_item_embeddings(ids)
                emb_out = x + (hard_emb - x).detach()
                loss = self.quantize_loss(query=x, value=hard_emb)

            else:
                raise ValueError(f"Unsupported forward mode: {self.forward_mode}")
        else:
            hard_emb = self.get_item_embeddings(ids)
            emb_out = hard_emb
            loss = self.quantize_loss(query=x, value=hard_emb)

        return QuantizeOutput(
            embeddings=emb_out,
            hard_embeddings=hard_emb if self.training else emb_out,
            ids=ids,
            loss=loss,
        )