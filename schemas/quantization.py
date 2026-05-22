from enum import Enum
from typing import NamedTuple
from torch import Tensor


class QuantizeForwardMode(str, Enum):
    STE = "ste"


class QuantizeDistance(str, Enum):
    L2 = "l2"


class QuantizeOutput(NamedTuple):
    embeddings: Tensor       # STE embedding — used as decoder input
    hard_embeddings: Tensor  # argmin embedding — used for RQ residuals and commitment loss
    ids: Tensor
    loss: Tensor
