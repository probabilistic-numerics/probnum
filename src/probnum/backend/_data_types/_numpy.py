"""Data types in NumPy."""

from typing import Dict, Union

import numpy as np
from numpy import (  # pylint: disable=redefined-builtin, unused-import
    bool_ as bool,
    complex64,
    complex128,
    dtype as DType,
    float16,
    float32,
    float64,
    int32,
    int64,
)

from ..typing import DTypeLike


def asdtype(x: DTypeLike, /) -> DType:
    return np.dtype(x)


def cast(
    x: np.ndarray, dtype: DType, /, *, casting: str = "unsafe", copy: bool = True
) -> np.ndarray:
    return x.astype(dtype=dtype, casting=casting, copy=copy)


def can_cast(from_: Union[DType, np.ndarray], to: DType, /) -> bool:
    return np.can_cast(from_, to)


def finfo(type: Union[DType, np.ndarray], /) -> Dict:
    floating_info = np.finfo(type)
    return {
        "bits": floating_info.bits,
        "eps": floating_info.eps,
        "max": floating_info.max,
        "min": floating_info.min,
    }


def iinfo(type: Union[DType, np.ndarray], /) -> Dict:
    integer_info = np.iinfo(type)
    return {
        "bits": integer_info.bits,
        "max": integer_info.max,
        "min": integer_info.min,
    }


def is_floating_dtype(dtype: DType, /) -> bool:
    return np.issubdtype(dtype, np.floating)


def promote_types(type1: DType, type2: DType, /) -> DType:
    return np.promote_types(type1, type2)


def result_type(*arrays_and_dtypes: Union[np.ndarray, DType]) -> DType:
    return np.result_type(*arrays_and_dtypes)
