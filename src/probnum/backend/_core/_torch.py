import torch

torch.set_default_dtype(torch.double)


def all(a: torch.Tensor, *, axis=None, keepdims: bool = False) -> torch.Tensor:
    if isinstance(axis, int):
        return torch.all(
            a,
            dim=axis,
            keepdim=keepdims,
        )

    axes = sorted(axis)

    res = a

    # If `keepdims is True`, this only works because axes is sorted!
    for axis in reversed(axes):
        res = torch.all(res, dim=axis, keepdims=keepdims)

    return res


def any(a: torch.Tensor, *, axis=None, keepdims: bool = False) -> torch.Tensor:
    if axis is None:
        return torch.any(a)

    if isinstance(axis, int):
        return torch.any(
            a,
            dim=axis,
            keepdim=keepdims,
        )

    axes = sorted(axis)

    res = a

    # If `keepdims is True`, this only works because axes is sorted!
    for axis in reversed(axes):
        res = torch.any(res, dim=axis, keepdims=keepdims)

    return res


def jit(f, *args, **kwargs):
    return f


def jit_method(f, *args, **kwargs):
    return f
