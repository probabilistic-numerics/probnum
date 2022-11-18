"""Statistical functions implemented in PyTorch."""

from typing import Optional, Tuple, Union

try:
    import torch
except ModuleNotFoundError:
    pass


def max(
    x: "torch.Tensor",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> "torch.Tensor":
    return torch.max(x, dim=axis, keepdim=keepdims)


def min(
    x: "torch.Tensor",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> "torch.Tensor":
    return torch.min(x, dim=axis, keepdim=keepdims)


def mean(
    x: "torch.Tensor",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> "torch.Tensor":
    return torch.mean(x, dim=axis, keepdim=keepdims)


def prod(
    x: "torch.Tensor",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional["torch.dtype"] = None,
    keepdims: bool = False,
) -> "torch.Tensor":
    return torch.prod(x, dim=axis, dtype=dtype, keepdim=keepdims)


def sum(
    x: "torch.Tensor",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional["torch.dtype"] = None,
    keepdims: bool = False,
) -> "torch.Tensor":
    return torch.sum(x, dim=axis, dtype=dtype, keepdim=keepdims)


def std(
    x: "torch.Tensor",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> "torch.Tensor":
    if correction == 0.0:
        return torch.std(x, dim=axis, unbiased=False, keepdim=keepdims)
    elif correction == 1.0:
        return torch.std(x, dim=axis, unbiased=True, keepdim=keepdims)
    else:
        raise NotImplementedError("Only correction=0 or =1 implemented.")


def var(
    x: "torch.Tensor",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> "torch.Tensor":
    if correction == 0.0:
        return torch.var(x, dim=axis, unbiased=False, keepdim=keepdims)
    elif correction == 1.0:
        return torch.var(x, dim=axis, unbiased=True, keepdim=keepdims)
    else:
        raise NotImplementedError("Only correction=0 or =1 implemented.")
