"""Sorting functions for torch tensors."""

import torch
from torch import (  # pylint: disable=redefined-builtin, unused-import, no-name-in-module
    isnan,
)


def sort(
    x: torch.Tensor, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> torch.Tensor:
    return torch.sort(x, dim=axis, descending=descending, stable=stable)[0]


def argsort(
    x: torch.Tensor, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> torch.Tensor:
    return torch.sort(x, dim=axis, descending=descending, stable=stable)[1]
