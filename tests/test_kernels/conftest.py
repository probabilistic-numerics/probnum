"""Test fixtures for kernels."""

from typing import Optional

import numpy as np
import pytest

import probnum as pn


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in range(1)],
    name="rng",
)
def fixture_rng(request):
    """Random state(s) used for test parameterization."""
    return np.random.default_rng(seed=request.param)


@pytest.fixture(
    params=[
        pytest.param(num_data, id=f"ndata{num_data}") for num_data in [1, 2, 10, 100]
    ],
    name="num_data",
)
def fixture_num_data(request) -> int:
    """Size of the dataset."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(input_dim, id=f"indim{input_dim}") for input_dim in [1, 10, 100]
    ],
    name="input_dim",
)
def fixture_input_dim(request) -> int:
    """Input dimension of the covariance function."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(output_dim, id=f"outdim{output_dim}") for output_dim in [1, 2, 10]
    ]
)
def output_dim(request) -> int:
    """Output dimension of the covariance function."""
    return request.param


# Datasets
@pytest.fixture(name="x0")
def fixture_x0(num_data: int, input_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random data from a standard normal distribution."""
    return rng.normal(0, 1, size=(num_data, input_dim))


@pytest.fixture(
    params=[
        pytest.param(num_data, id=f"ndata{num_data}") for num_data in [None, 10, 2, 1]
    ],
    name="x1",
)
def fixture_x1(
    request, input_dim: int, rng: np.random.Generator
) -> Optional[np.ndarray]:
    """Random data from a standard normal distribution."""
    if request.param is None:
        return None

    return rng.normal(0, 1, size=(request.param, input_dim))


@pytest.fixture()
def x0_1d(input_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random 1D dataset."""
    return rng.normal(0, 1, size=(input_dim,))


# Kernel and kernel matrices


@pytest.fixture(
    params=[
        pytest.param(kerndef, id=kerndef[0].__name__)
        for kerndef in [
            (pn.kernels.Linear, {"constant": 1.0}),
            (pn.kernels.WhiteNoise, {"sigma": -1.0}),
            (pn.kernels.Polynomial, {"constant": 1.0, "exponent": 3}),
            (pn.kernels.ExpQuad, {"lengthscale": 1.5}),
            (pn.kernels.RatQuad, {"lengthscale": 0.5, "alpha": 2.0}),
            (pn.kernels.Matern, {"lengthscale": 0.5, "nu": 0.5}),
            (pn.kernels.Matern, {"lengthscale": 0.5, "nu": 1.5}),
            (pn.kernels.Matern, {"lengthscale": 1.5, "nu": 2.5}),
            (pn.kernels.Matern, {"lengthscale": 2.5, "nu": 7.0}),
            (pn.kernels.Matern, {"lengthscale": 3.0, "nu": np.inf}),
        ]
    ],
    name="kernel",
)
def fixture_kernel(request, input_dim: int) -> pn.kernels.Kernel:
    """Kernel / covariance function."""
    return request.param[0](**request.param[1], input_dim=input_dim)


@pytest.fixture(name="kernel_naive")
def fixture_kernel_naive(kernel: pn.kernels.Kernel):
    """Naive implementation of kernel broadcasting which applies the kernel function to
    scalar arguments while looping over the first dimensions of the inputs explicitly.
    Can be used as a reference implementation of kernel broadcasting."""
    kernel_vectorized = np.vectorize(
        kernel, signature="(d),(d)->(o,o)", excluded={"squeeze_output_dim"}
    )

    def _kernel_naive(
        x0: np.ndarray,
        x1: np.ndarray = None,
        squeeze_output_dim: bool = True,
    ):
        x0 = np.atleast_1d(x0)

        if x1 is None:
            x1 = x0
        else:
            x1 = np.atleast_1d(x1)

        K = kernel_vectorized(x0, x1, squeeze_output_dim=False)

        if kernel.output_dim == 1 and squeeze_output_dim:
            K = K.squeeze((-2, -1))

        return K

    return _kernel_naive


@pytest.fixture
def kernmat(
    kernel: pn.kernels.Kernel, x0: np.ndarray, x1: Optional[np.ndarray]
) -> np.ndarray:
    """Kernel evaluated at the data."""
    if x1 is None:
        return kernel(x0)

    return kernel(x0=x0[:, None, :], x1=x1[None, :, :])


@pytest.fixture
def kernmat_naive(
    kernel_naive: pn.kernels.Kernel, x0: np.ndarray, x1: Optional[np.ndarray]
) -> np.ndarray:
    """Kernel evaluated at the data."""
    if x1 is None:
        return kernel_naive(x0)

    return kernel_naive(x0=x0[:, None, :], x1=x1[None, :, :])
