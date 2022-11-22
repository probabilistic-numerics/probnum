from typing import Callable

try:
    import torch
except ModuleNotFoundError:
    pass


def cond(pred: " torch.Tensor", true_fn: Callable, false_fn: Callable, *operands):
    pred = torch.as_tensor(pred)

    if pred.ndim != 0:
        raise ValueError("`pred` must be a scalar")

    if pred:
        return true_fn(*operands)

    return false_fn(*operands)
