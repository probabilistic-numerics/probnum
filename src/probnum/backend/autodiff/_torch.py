"""(Automatic) Differentiation in PyTorch."""

from typing import Callable, Sequence, Union

import torch


def grad(fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
    def _grad_fn(*args, **kwargs):
        args = list(args)
        if isinstance(argnums, int):
            args[argnums] = args[argnums].clone().detach().requires_grad_(True)

            return torch.autograd.grad(fun(*args, **kwargs), args[argnums])[0]

        for argnum in argnums:
            args[argnum] = args[argnum] = (
                args[argnum].clone().detach().requires_grad_(True)
            )

        return torch.autograd.grad(
            fun(*args, **kwargs), tuple(args[argnum] for argnum in argnums)
        )

    return _grad_fn
