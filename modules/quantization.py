import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.cluster import KMeans

from schemas.quantization import QuantizeOutput

class Quantization(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        do_kmeans_init: bool = True,
        sim_vq: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False) if sim_vq else nn.Identity(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
                
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
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

    def forward(self, x: Tensor) -> QuantizeOutput:
        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x)


        codebook = self.out_proj(self.embedding.weight)
        
        dist = (
            (x**2).sum(axis=1, keepdim=True) +
            (codebook.T**2).sum(axis=0, keepdim=True) -
            2 * x @ codebook.T
        )
        
        #probs = F.softmax(-dist, dim=1)
        #ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        _, ids = (dist.detach()).min(axis=1)
        emb = self.get_item_embeddings(ids)
        emb_out = x + (emb - x).detach()

        # Compute commitment loss

        emb_loss = ((x.detach() - emb)**2).sum(axis=[-1])
        query_loss = ((x - emb.detach())**2).sum(axis=[-1])
        loss = emb_loss + self.commitment_weight * query_loss
    
        return QuantizeOutput(
            embeddings=emb_out,
            ids=ids,
            loss=loss,
        )