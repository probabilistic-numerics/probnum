"""NumPy array creation functions."""
from typing import Optional, Union

import numpy as np
from numpy import tril, triu  # pylint: disable=redefined-builtin, unused-import


def asarray(
    obj: Union[
        np.ndarray, bool, int, float, "NestedSequence", "SupportsBufferProtocol"
    ],
    /,
    *,
    dtype: Optional["probnum.backend.Dtype"] = None,
    device: Optional["probnum.backend.Device"] = None,
    copy: Optional[bool] = None,
) -> np.ndarray:
    if copy is None:
        copy = False
    return np.array(obj, dtype=dtype, copy=copy)
