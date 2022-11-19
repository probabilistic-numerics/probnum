"""JAX array creation functions."""
from typing import List, Optional, Union

try:
    import jax
    import jax.numpy as jnp
    from jax.numpy import diag, tril, triu  # pylint: unused-import
except ModuleNotFoundError:
    pass

from .. import Device, DType
from .._data_types import is_floating_dtype
from ..typing import ShapeType

# pylint: disable=redefined-builtin


def asarray(
    obj: Union[
        "jax.Array", bool, int, float, "NestedSequence", "SupportsBufferProtocol"
    ],
    /,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> "jax.Array":
    if copy is None:
        copy = True

    out = jnp.array(obj, dtype=dtype, copy=copy)

    if is_floating_dtype(out.dtype):
        out = out.astype(config.default_floating_dtype, copy=False)

    return jax.device_put(out, device=device)


def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "jax.Array":
    return jax.device_put(jnp.arange(start, stop, step, dtype=dtype), device=device)


def empty(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "jax.Array":
    return jax.device_put(jnp.empty(shape, dtype=dtype), device=device)


def empty_like(
    x: "jax.Array",
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "jax.Array":
    return jax.device_put(jnp.empty_like(x, shape=shape, dtype=dtype), device=device)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "jax.Array":
    return jax.device_put(jnp.eye(n_rows, n_cols, k=k, dtype=dtype), device=device)


def full(
    shape: ShapeType,
    fill_value: Union[int, float],
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "jax.Array":
    return jax.device_put(jnp.full(shape, fill_value, dtype=dtype), device=device)


def full_like(
    x: "jax.Array",
    /,
    fill_value: Union[int, float],
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "jax.Array":
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
) -> "jax.Array":
    return jax.device_put(
        jnp.linspace(start, stop, num=num, dtype=dtype, endpoint=endpoint),
        device=device,
    )


def meshgrid(*arrays: "jax.Array", indexing: str = "xy") -> List["jax.Array"]:
    return jnp.meshgrid(*arrays, indexing=indexing)


def ones(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "jax.Array":
    return jax.device_put(jnp.ones(shape, dtype=dtype), device=device)


def ones_like(
    x: "jax.Array",
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "jax.Array":
    return jax.device_put(jnp.ones_like(x, shape=shape, dtype=dtype), device=device)


def zeros(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "jax.Array":
    return jax.device_put(jnp.zeros(shape, dtype=dtype), device=device)


def zeros_like(
    x: "jax.Array",
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "jax.Array":
    return jax.device_put(jnp.zeros_like(x, shape=shape, dtype=dtype), device=device)
