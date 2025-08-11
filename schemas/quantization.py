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
    embeddings: Tensor
    ids: Tensor
    loss: Tensor