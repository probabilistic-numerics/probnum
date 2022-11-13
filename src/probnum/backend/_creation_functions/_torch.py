"""Torch tensor creation functions."""
from typing import Optional, Union

import torch
from torch import (  # pylint: disable=redefined-builtin, unused-import
    arange,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)


def asarray(
    obj: Union[
        torch.Tensor, bool, int, float, "NestedSequence", "SupportsBufferProtocol"
    ],
    /,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    copy: Optional[bool] = None,
) -> torch.Tensor:
    x = torch.as_tensor(obj, dtype=dtype, device=device)
    if copy is not None:
        if copy:
            return x.clone()
    return x


def tril(x: torch.Tensor, /, k: int = 0) -> torch.Tensor:
    return tril(x, diagonal=k)


def triu(x: torch.Tensor, /, k: int = 0) -> torch.Tensor:
    return triu(x, diagonal=k)
