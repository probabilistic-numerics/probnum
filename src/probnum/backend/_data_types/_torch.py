"""Data types in PyTorch."""
from typing import Dict, Union

import numpy as np

try:
    import torch
    from torch import (  # pylint: disable=redefined-builtin, unused-import
        bool,
        complex64,
        complex128,
        dtype as DType,
        float16,
        float32,
        float64,
        int32,
        int64,
    )
except ModuleNotFoundError:
    pass

# from . import MachineLimitsFloatingPoint, MachineLimitsInteger
from ..typing import DTypeLike


def asdtype(x: DTypeLike, /) -> "DType":
    if isinstance(x, torch.dtype):
        return x

    return torch.as_tensor(
        np.empty(
            (),
            dtype=np.dtype(x),
        ),
    ).dtype


def cast(
    x: "torch.Tensor", dtype: "DType", /, *, casting: str = "unsafe", copy: bool = True
) -> "torch.Tensor":
    return x.to(dtype=dtype, copy=copy)


def can_cast(from_: Union["DType", "torch.Tensor"], to: "DType", /) -> bool:
    return torch.can_cast(from_, to)


def finfo(type: Union["DType", "torch.Tensor"], /) -> Dict:
    floating_info = torch.finfo(type)
    return {
        "bits": floating_info.bits,
        "eps": floating_info.eps,
        "max": floating_info.max,
        "min": floating_info.min,
    }


def iinfo(type: Union["DType", "torch.Tensor"], /) -> Dict:
    integer_info = torch.iinfo(type)
    return {
        "bits": integer_info.bits,
        "max": integer_info.max,
        "min": integer_info.min,
    }


def is_floating_dtype(dtype: "DType", /) -> bool:
    return torch.is_floating_point(torch.empty((), dtype=dtype))


def promote_types(type1: "DType", type2: "DType", /) -> "DType":
    return torch.promote_types(type1, type2)


def result_type(*arrays_and_dtypes: Union["torch.Tensor", "DType"]) -> "DType":
    return torch.result_type(*arrays_and_dtypes)
