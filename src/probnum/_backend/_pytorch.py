import torch
from torch import as_tensor as asarray
from torch import atleast_1d, atleast_2d, broadcast_shapes
from torch import broadcast_tensors as broadcast_arrays
from torch import exp, sqrt


def array(object, dtype=None, *, copy=True):
    if copy:
        return torch.tensor(object, dtype=dtype)

    return asarray(object, dtype=dtype)


def grad(fun, argnums=0):
    def _grad_fn(*args, **kwargs):
        if isinstance(argnums, int):
            args[argnums].requires_grad_()

            return torch.autograd.grad(fun(*args, **kwargs), args[argnums])

        for argnum in argnums:
            args[argnum].requires_grad_()

        return torch.autograd.grad(
            fun(*args, **kwargs), tuple(args[argnum] for argnum in argnums)
        )

    return _grad_fn


def ndim(a):
    try:
        return a.ndim
    except AttributeError:
        return torch.as_tensor(a).ndim


def ones_like(a, dtype=None, *, shape=None):
    if shape is None:
        return torch.ones_like(input=a, dtype=dtype)

    return torch.ones(
        *shape,
        dtype=a.dtype if dtype is None else dtype,
        layout=a.layout,
        device=a.device,
    )


sum = lambda a, axis=None, dtype=None, keepdims=False: torch.sum(
    input=a, dim=axis, keepdim=keepdims, dtype=dtype
)


def zeros(shape, dtype=None):
    return torch.zeros(*shape, dtype=dtype)


def zeros_like(a, dtype=None, *, shape=None):
    if shape is None:
        return torch.zeros_like(input=a, dtype=dtype)

    return torch.zeros(
        *shape,
        dtype=a.dtype if dtype is None else dtype,
        layout=a.layout,
        device=a.device,
    )


def jit(f):
    return f
