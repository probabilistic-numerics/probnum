"""Test properties of normal random variables."""
import numpy as np
import scipy.stats
from pytest_cases import parametrize, parametrize_with_cases

from probnum import backend, randvars
from probnum.typing import SeedLike, ShapeType


@parametrize_with_cases("rv", cases=".cases", has_tag=["univariate"])
def test_entropy(rv: randvars.Normal):
    scipy_entropy = scipy.stats.norm.entropy(
        loc=backend.to_numpy(rv.mean),
        scale=backend.to_numpy(rv.std),
    )

    np.testing.assert_allclose(backend.to_numpy(rv.entropy), scipy_entropy)


@parametrize_with_cases("rv", cases=".cases", has_tag=["univariate"])
@parametrize("shape", ([(), (1,), (5,), (2, 3), (3, 1, 2)]))
@parametrize("seed", (91985,))
def test_pdf_univariate(rv: randvars.Normal, shape: ShapeType, seed: SeedLike):
    x = backend.random.standard_normal(
        backend.random.seed(seed),
        shape=shape,
        dtype=rv.dtype,
    )

    scipy_pdf = scipy.stats.norm.pdf(
        backend.to_numpy(x),
        loc=backend.to_numpy(rv.mean),
        scale=backend.to_numpy(rv.std),
    )

    np.testing.assert_allclose(backend.to_numpy(rv.pdf(x)), scipy_pdf)


@parametrize_with_cases("rv", cases=".cases", has_tag=["vectorvariate"])
@parametrize("shape", ((), (1,), (5,), (2, 3), (3, 1, 2)))
@parametrize("seed", (65465,))
def test_pdf_multivariate(rv: randvars.Normal, shape: ShapeType, seed: SeedLike):
    x = rv.sample(
        backend.random.seed(seed),
        sample_shape=shape,
    )

    scipy_pdf = scipy.stats.multivariate_normal.pdf(
        backend.to_numpy(x),
        mean=backend.to_numpy(rv.mean),
        cov=backend.to_numpy(rv.cov),
    )

    # There is a bug in scipy's implementation of the pdf for the multivariate normal:
    expected_shape = x.shape[: x.ndim - rv.ndim]

    if any(dim == 1 for dim in expected_shape):
        # scipy's implementation happily squeezes `1` dimensions out of the batch
        assert all(dim != 1 for dim in scipy_pdf.shape)

        scipy_pdf = scipy_pdf.reshape(expected_shape)

    np.testing.assert_allclose(backend.to_numpy(rv.pdf(x)), scipy_pdf)
