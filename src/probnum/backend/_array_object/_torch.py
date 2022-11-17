"""Array object in PyTorch."""
from typing import Tuple, Union

import numpy as np
import torch
from torch import (  # pylint: disable=redefined-builtin, unused-import, reimported
    Tensor as Array,
    Tensor as Scalar,
    device as Device,
)


def ndim(a: torch.Tensor):
    try:
        return a.ndim
    except AttributeError:
        return torch.as_tensor(a).ndim


def to_numpy(*arrays: torch.Tensor) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if len(arrays) == 1:
        return arrays[0].cpu().detach().numpy()

    return tuple(arr.cpu().detach().numpy() for arr in arrays)
