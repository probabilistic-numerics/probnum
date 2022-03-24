import hashlib
import numbers
from typing import Optional, Union

import numpy as np

from probnum import backend
from probnum.typing import ArrayType, DTypeLike, IntLike, SeedType, ShapeLike


def seed_from_sampling_args(
    *,
    base_seed: IntLike,
    shape: ShapeLike,
    dtype: Optional[DTypeLike] = None,
    **kwargs: Union[numbers.Number, np.ndarray, ArrayType],
) -> SeedType:
    """Diversify random seeds for deterministic testing.

    When writing a test relying on "random" input data generated from a fixed random
    seeds, a common pattern is to parametrize over seed and shape like so:

    >>> import pytest
    >>> from probnum.typing import ShapeType
    >>> @pytest.fixture(params=[42, 43])
    ... def seed(request) -> int:
    ...     return request.param

    >>> @pytest.fixture(params=((2,), (4,)))
    ... def shape(request) -> ShapeType:
    ...     return request.param

    >>> def test_function(seed: int, shape: ShapeType):
    ...     x = backend.random.uniform(
    ...         backend.random.seed(seed),
    ...         shape=shape,
    ...     )
    ...     ...  # Test something

    Unfortunately, when sampling from the same seed but with different shapes in NumPy
    and Jax, some sampling routines produce partially identical arrays.

    >>> np.random.default_rng(42).uniform(size=(2,))
    array([0.77395605, 0.43887844])
    >>> np.random.default_rng(42).uniform(size=(4,))
    array([0.77395605, 0.43887844, 0.85859792, 0.69736803])

    To diversify test data, while retaining test determinism (especially under the order
    of test execution!), `seed_from_sampling_args` provides a deterministic way to
    modify the base seed through other arguments passed to the sampling routine:

    >>> def test_data(seed: int, shape: ShapeType) -> ArrayType:
    ...     return backend.random.uniform(
    ...         seed_from_sampling_args(base_seed=seed, shape=shape),
    ...         shape=shape,
    ...     )

    >>> backend.all(test_data(42, shape=(2,)) != test_data(42, shape=(4,))[:2])
    True

    Parameters
    ----------
    base_seed
        Seed value common to all sample calls in a parametrized test.
    shape
        `shape` argument to the `backend.random.<sample_fn>` call.
    dtype
        `dtype` argument to the `backend.random.<sample_fn>` call.
    **kwargs
        Any other keyword argument passed to the `backend.random.<sample_fn>` call.

    Returns
    -------
    seed
        A seed object that is deterministically generated from the function's arguments
        using a cryptographic hash function.

    Raises
    ------
    ValueError
        If the `base_seed` is a negative number.
    TypeError
        If the type of any of the `kwargs` is not supported.
    """

    # Hash unique representations of the arguments into a 7-byte positive integer.
    # We choose 7 bytes, since an 8-byte positive integer could already overflow as an
    # int64.
    h = hashlib.blake2b(digest_size=7)

    # `base_seed`
    base_seed = int(base_seed)

    if base_seed < 0:
        raise ValueError("`base_seed` must be a non-negative `int`")

    h.update(hex(base_seed).encode())

    # `shape`
    shape = backend.as_shape(shape)

    h.update(b"(")

    for entry in shape:
        h.update(hex(entry).encode())

    h.update(b")")

    # `dtype`
    if dtype is not None:
        dtype = backend.asdtype(dtype)

        h.update(str(dtype).encode())

    # `kwargs`
    for key, value in kwargs.items():
        h.update(key.encode())

        if isinstance(value, numbers.Number) and (
            # NumPy doesn't handle `fractions.Fraction` too well
            not isinstance(value, numbers.Rational)
            or isinstance(value, numbers.Real)
        ):
            h.update(np.asarray(value).tobytes())
        elif isinstance(value, np.ndarray):
            h.update(value.tobytes(order="A"))
        elif backend.isarray(value):
            h.update(backend.to_numpy(value).tobytes(order="A"))
        else:
            raise TypeError(
                "Values passed by `kwargs` must be either numbers, `np.ndarray`s, or "
                f"`ArrayType`s, not {type(value)}."
            )

    # Convert hash to positive integer
    seed_int = abs(int(h.hexdigest(), base=16))

    return backend.random.seed(seed_int)
