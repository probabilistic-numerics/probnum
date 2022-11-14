"""JAX array creation functions."""
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
from jax.numpy import tril, triu  # pylint: disable=redefined-builtin, unused-import

from .. import Device, DType
from ..typing import ShapeType


def asarray(
    obj: Union[
        jnp.ndarray, bool, int, float, "NestedSequence", "SupportsBufferProtocol"
    ],
    /,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> jnp.ndarray:
    if copy is None:
        copy = True

    return jax.device_put(jnp.array(obj, dtype=dtype, copy=copy), device=device)


def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> jnp.ndarray:
    return jax.device_put(jnp.arange(start, stop, step, dtype=dtype), device=device)


def empty(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> jnp.ndarray:
    return jax.device_put(jnp.empty(shape, dtype=dtype), device=device)


def empty_like(
    x: jnp.ndarray,
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> jnp.ndarray:
    return jax.device_put(jnp.empty_like(x, shape=shape, dtype=dtype), device=device)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> jnp.ndarray:
    return jax.device_put(jnp.eye(n_rows, n_cols, k=k, dtype=dtype), device=device)


def full(
    shape: ShapeType,
    fill_value: Union[int, float],
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> jnp.ndarray:
    return jax.device_put(jnp.full(shape, fill_value, dtype=dtype), device=device)


def full_like(
    x: jnp.ndarray,
    /,
    fill_value: Union[int, float],
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> jnp.ndarray:
    return jax.device_put(
        jnp.full_like(x, fill_value=fill_value, shape=shape, dtype=dtype), device=device
    )


def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> jnp.ndarray:
    return jax.device_put(
        jnp.linspace(start, stop, num=num, dtype=dtype, endpoint=endpoint),
        device=device,
    )


def meshgrid(*arrays: jnp.ndarray, indexing: str = "xy") -> List[jnp.ndarray]:
    return jnp.meshgrid(*arrays, indexing=indexing)


def ones(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> jnp.ndarray:
    return jax.device_put(jnp.ones(shape, dtype=dtype), device=device)


def ones_like(
    x: jnp.ndarray,
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> jnp.ndarray:
    return jax.device_put(jnp.ones_like(x, shape=shape, dtype=dtype), device=device)


def zeros(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> jnp.ndarray:
    return jax.device_put(jnp.zeros(shape, dtype=dtype), device=device)


def zeros_like(
    x: jnp.ndarray,
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> jnp.ndarray:
    return jax.device_put(jnp.zeros_like(x, shape=shape, dtype=dtype), device=device)
