import numpy as np

from probnum import backend, compat
from probnum.backend.typing import Seed, ShapeType

import pytest_cases
import tests.utils


@pytest_cases.fixture(scope="module")
@pytest_cases.parametrize("seed", (234789, 7890))
@pytest_cases.parametrize("n", (1, 2, 5, 9))
@pytest_cases.parametrize("shape", ((), (1,), (2,), (3, 2)))
@pytest_cases.parametrize("dtype", (backend.float32, backend.float64))
def so_group_sample(
    seed: Seed, n: int, shape: ShapeType, dtype: backend.Dtype
) -> backend.Array:
    return backend.random.uniform_so_group(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=seed, shape=shape, dtype=dtype, n=n
        ),
        n=n,
        shape=shape,
        dtype=dtype,
    )


def test_orthogonal(so_group_sample: backend.Array):
    n = so_group_sample.shape[-2]

    compat.testing.assert_allclose(
        so_group_sample @ backend.swapaxes(so_group_sample, -2, -1),
        backend.broadcast_arrays(backend.eye(n), so_group_sample)[0],
        atol=1e-6 if so_group_sample.dtype == backend.float32 else 1e-12,
    )


def test_determinant_1(so_group_sample: backend.Array):
    compat.testing.assert_allclose(
        np.linalg.det(compat.to_numpy(so_group_sample)),
        1.0,
        rtol=2e-6 if so_group_sample.dtype == backend.float32 else 1e-7,
    )
