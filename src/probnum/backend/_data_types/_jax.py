"""Data types in JAX."""

from typing import Dict, Union

import jax.numpy as jnp
from jax.numpy import (  # pylint: disable=redefined-builtin, unused-import
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
    return jnp.dtype(x)


def cast(
    x: jnp.ndarray, dtype: DType, /, *, casting: str = "unsafe", copy: bool = True
) -> jnp.ndarray:
    return x.astype(dtype=dtype)


def can_cast(from_: Union[DType, jnp.ndarray], to: DType, /) -> bool:
    return jnp.can_cast(from_, to)


def finfo(type: Union[DType, jnp.ndarray], /) -> Dict:
    floating_info = jnp.finfo(type)
    return {
        "bits": floating_info.bits,
        "eps": floating_info.eps,
        "max": floating_info.max,
        "min": floating_info.min,
    }


def iinfo(type: Union[DType, jnp.ndarray], /) -> Dict:
    integer_info = jnp.iinfo(type)
    return {
        "bits": integer_info.bits,
        "max": integer_info.max,
        "min": integer_info.min,
    }


def is_floating_dtype(dtype: DType, /) -> bool:
    return jnp.is_floating(jnp.empty((), dtype=dtype))


def promote_types(type1: DType, type2: DType, /) -> DType:
    return jnp.promote_types(type1, type2)


def result_type(*arrays_and_dtypes: Union[jnp.ndarray, DType]) -> DType:
    return jnp.result_type(*arrays_and_dtypes)
