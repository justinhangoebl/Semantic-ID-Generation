from typing import NamedTuple
from torch import Tensor


class RqVaeOutput(NamedTuple):
    embeddings: Tensor   # shape (n_layers, latent_dim, batch)
    residuals: Tensor    # shape (n_layers, latent_dim, batch)
    sem_ids: Tensor      # shape (batch, n_layers)
    quantize_loss: Tensor


class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    p_unique_ids: Tensor
    quantized: RqVaeOutput
