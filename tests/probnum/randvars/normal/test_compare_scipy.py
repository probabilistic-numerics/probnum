"""Test properties of normal random variables."""

import scipy.stats

from probnum import backend, compat, randvars
from probnum.backend.typing import ShapeType

import pytest
from pytest_cases import filters, parametrize, parametrize_with_cases
import tests.utils


@parametrize_with_cases(
    "rv",
    cases=".cases",
    filter=filters.has_tag("scalar") & ~filters.has_tag("degenerate"),
)
def test_entropy(rv: randvars.Normal):
    scipy_entropy = scipy.stats.norm.entropy(
        loc=backend.to_numpy(rv.mean),
        scale=backend.to_numpy(rv.std),
    )

    compat.testing.assert_allclose(rv.entropy, scipy_entropy)


@parametrize_with_cases(
    "rv",
    cases=".cases",
    filter=filters.has_tag("scalar") & ~filters.has_tag("degenerate"),
)
@parametrize("shape", ([(), (1,), (5,), (2, 3), (3, 1, 2)]))
def test_pdf_scalar(rv: randvars.Normal, shape: ShapeType):
    x = backend.random.standard_normal(
        tests.utils.random.seed_from_sampling_args(base_seed=245, shape=shape),
        shape=shape,
        dtype=rv.dtype,
    )

    scipy_pdf = scipy.stats.norm.pdf(
        backend.to_numpy(x),
        loc=backend.to_numpy(rv.mean),
        scale=backend.to_numpy(rv.std),
    )

    compat.testing.assert_allclose(rv.pdf(x), scipy_pdf)


@parametrize_with_cases(
    "rv",
    cases=".cases",
    filter=(
        (filters.has_tag("vector") | filters.has_tag("matrix"))
        & ~filters.has_tag("degenerate")
    ),
)
@parametrize("shape", ((), (1,), (5,), (2, 3), (3, 1, 2)))
def test_pdf_multivariate(rv: randvars.Normal, shape: ShapeType):
    x = rv.sample(
        tests.utils.random.seed_from_sampling_args(base_seed=65465, shape=shape),
        sample_shape=shape,
    )

    scipy_pdf = scipy.stats.multivariate_normal.pdf(
        backend.to_numpy(x.reshape(shape + (-1,))),
        mean=backend.to_numpy(rv.dense_mean.reshape(-1)),
        cov=backend.to_numpy(rv.dense_cov),
    )

    # There is a bug in scipy's implementation of the pdf for the multivariate normal:
    expected_shape = x.shape[: x.ndim - rv.ndim]

    if any(dim == 1 for dim in expected_shape):
        # scipy's implementation happily squeezes `1` dimensions out of the batch
        assert all(dim != 1 for dim in scipy_pdf.shape)

        scipy_pdf = scipy_pdf.reshape(expected_shape)

    compat.testing.assert_allclose(rv.pdf(x), scipy_pdf)


@pytest.mark.skipif_backend(backend.Backend.JAX)
@pytest.mark.skipif_backend(backend.Backend.TORCH)
@parametrize_with_cases(
    "rv",
    cases=".cases",
    filter=(
        (filters.has_tag("vector") | filters.has_tag("matrix"))
        & ~filters.has_tag("degenerate")
    ),
)
@parametrize("shape", ((), (1,), (5,), (2, 3), (3, 1, 2)))
def test_cdf_multivariate(rv: randvars.Normal, shape: ShapeType):
    scipy_rv = scipy.stats.multivariate_normal(
        mean=backend.to_numpy(rv.dense_mean.reshape(-1)),
        cov=backend.to_numpy(rv.dense_cov),
    )

    x = rv.sample(
        tests.utils.random.seed_from_sampling_args(base_seed=978134, shape=shape),
        sample_shape=shape,
    )

    cdf = rv.cdf(x)

    scipy_cdf = scipy_rv.cdf(backend.to_numpy(x.reshape(shape + (-1,))))

    # There is a bug in scipy's implementation of the pdf for the multivariate normal:
    expected_shape = x.shape[: x.ndim - rv.ndim]

    if any(dim == 1 for dim in expected_shape):
        # scipy's implementation happily squeezes `1` dimensions out of the batch
        assert all(dim != 1 for dim in scipy_cdf.shape)

        scipy_cdf = scipy_cdf.reshape(expected_shape)

    compat.testing.assert_allclose(
        cdf, scipy_cdf, atol=scipy_rv.abseps, rtol=scipy_rv.releps
    )
