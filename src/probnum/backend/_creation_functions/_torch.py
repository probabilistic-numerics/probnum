"""Torch tensor creation functions."""
from typing import Optional, Union

import torch


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
    return torch.tril(x, diagonal=k)


def triu(x: torch.Tensor, /, k: int = 0) -> torch.Tensor:
    return torch.triu(x, diagonal=k)
