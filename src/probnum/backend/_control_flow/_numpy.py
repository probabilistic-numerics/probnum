from typing import Callable, Union

import numpy as np


def cond(
    pred: Union[np.ndarray, np.generic],
    true_fn: Callable,
    false_fn: Callable,
    *operands
):
    if np.ndim(pred) != 0:
        raise ValueError("`pred` must be a scalar")

    if pred:
        return true_fn(*operands)

    return false_fn(*operands)
