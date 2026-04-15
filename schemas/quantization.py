from typing import NamedTuple
from torch import Tensor
from enum import Enum

class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2

class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2

class QuantizeOutput(NamedTuple):
    embeddings: Tensor       # soft embedding (Gumbel) or STE embedding - used for decoder input
    hard_embeddings: Tensor  # hard (argmin) embedding - used for RQ residuals and commitment loss
    ids: Tensor
    loss: Tensor