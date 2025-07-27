from enum import Enum
from typing import NamedTuple
from torch import Tensor
from enum import Enum

class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2

class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2


class QuantizeForwardMode(str, Enum):
    STE = "ste"


class QuantizeDistance(str, Enum):
    L2 = "l2"


class QuantizeOutput(NamedTuple):
    embeddings: Tensor       # STE embedding — used as decoder input
    hard_embeddings: Tensor  # argmin embedding — used for RQ residuals and commitment loss
    ids: Tensor
    loss: Tensor
