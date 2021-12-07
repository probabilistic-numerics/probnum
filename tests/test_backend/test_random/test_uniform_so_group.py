import pytest_cases

from probnum import backend, compat
from probnum.typing import SeedLike, ShapeType


@pytest_cases.fixture
@pytest_cases.parametrize("seed", (234789, 7890))
@pytest_cases.parametrize("n", (1, 2, 5, 9))
@pytest_cases.parametrize("shape", ((), (1,), (2,), (3, 2)))
@pytest_cases.parametrize("dtype", (backend.single, backend.double))
def so_group_sample(
    seed: SeedLike, n: int, shape: ShapeType, dtype: backend.dtype
) -> backend.ndarray:
    return backend.random.uniform_so_group(
        seed=backend.random.seed(abs(seed + n + hash(shape) + hash(dtype))),
        n=n,
        shape=shape,
        dtype=dtype,
    )


def test_orthogonal(so_group_sample: backend.ndarray):
    n = so_group_sample.shape[-2]

    compat.testing.assert_allclose(
        so_group_sample @ backend.swapaxes(so_group_sample, -2, -1),
        backend.broadcast_arrays(backend.eye(n), so_group_sample)[0],
        atol=1e-6 if so_group_sample.dtype == backend.single else 1e-12,
    )
