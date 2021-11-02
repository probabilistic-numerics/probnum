import torch


def grad(fun, argnums=0):
    def _grad_fn(*args, **kwargs):
        if isinstance(argnums, int):
            args = list(args)
            args[argnums] = torch.tensor(args[argnums], requires_grad=True)

            return torch.autograd.grad(fun(*args, **kwargs), args[argnums])

        for argnum in argnums:
            args[argnum].requires_grad_()

        return torch.autograd.grad(
            fun(*args, **kwargs), tuple(args[argnum] for argnum in argnums)
        )

    return _grad_fn
