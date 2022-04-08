"""Fixtures for random process tests."""

from typing import Any, Callable, Dict, Tuple, Type

from probnum import Function, LambdaFunction, backend, randprocs
from probnum.backend.typing import ShapeType
from probnum.randprocs import kernels, mean_fns

import pytest
import pytest_cases
import tests.utils


@pytest_cases.fixture(scope="package")
@pytest_cases.parametrize(
    "shape", [(), (1,), (10,), (100,)], idgen="input_shape{shape}"
)
def input_shape(shape: ShapeType) -> ShapeType:
    """Input dimension of the random process."""
    return shape


@pytest_cases.fixture(scope="package")
@pytest_cases.parametrize("shape", [()], idgen="output_shape{shape}")
def output_shape(shape: ShapeType) -> ShapeType:
    """Output dimension of the random process."""
    return shape


@pytest_cases.fixture(scope="package")
@pytest_cases.parametrize(
    "meanfndef",
    [
        ("Zero", mean_fns.Zero),
        (
            "Lambda",
            lambda input_shape, output_shape: LambdaFunction(
                lambda x: (
                    backend.full_like(x, 2.0, shape=output_shape)
                    * backend.sum(x, axis=tuple(range(-len(input_shape), 0)))
                    + 1.0
                ),
                input_shape=input_shape,
                output_shape=output_shape,
            ),
        ),
    ],
    idgen="{meanfndef[0]}",
)
def mean(
    meanfndef: Tuple[str, Callable[[ShapeType, ShapeType], Function]],
    input_shape: ShapeType,
    output_shape: ShapeType,
) -> Function:
    """Mean function of a random process."""
    return meanfndef[1](input_shape=input_shape, output_shape=output_shape)


@pytest_cases.fixture(scope="package")
@pytest_cases.parametrize(
    "kerndef",
    [
        (kernels.Polynomial, {"constant": 1.0, "exponent": 3}),
        (kernels.ExpQuad, {"lengthscale": 1.5}),
        (kernels.RatQuad, {"lengthscale": 0.5, "alpha": 2.0}),
        (kernels.Matern, {"lengthscale": 0.5, "nu": 1.5}),
    ],
    idgen="{kerndef[0].__name__}",
)
def cov(
    kerndef: Tuple[Type[kernels.Kernel], Dict[str, Any]],
    input_shape: ShapeType,
    output_shape: ShapeType,
) -> kernels.Kernel:
    """Covariance function."""

    if output_shape != ():
        pytest.skip()

    kernel_type, kwargs = kerndef

    return kernel_type(input_shape=input_shape, **kwargs)


@pytest_cases.fixture(scope="package")
@pytest_cases.parametrize(
    "randprocdef",
    [
        (
            "GP-Zero-Matern",
            lambda input_shape, output_shape: randprocs.GaussianProcess(
                mean=mean_fns.Zero(input_shape=input_shape),
                cov=kernels.Matern(input_shape=input_shape),
            ),
        ),
    ],
    idgen="{randprocdef[0]}",
)
def random_process(
    randprocdef: Tuple[str, Callable[[ShapeType, ShapeType], randprocs.RandomProcess]],
    input_shape: ShapeType,
    output_shape: ShapeType,
) -> randprocs.RandomProcess:
    """Random process."""
    return randprocdef[1](input_shape, output_shape)


@pytest_cases.fixture(scope="package")
def gaussian_process(mean: Function, cov: kernels.Kernel) -> randprocs.GaussianProcess:
    """Gaussian process."""
    return randprocs.GaussianProcess(mean=mean, cov=cov)


@pytest_cases.fixture(scope="session")
@pytest_cases.parametrize("shape", [(), (1,), (10,)], idgen="batch_shape{shape}")
def args0_batch_shape(shape: ShapeType) -> ShapeType:
    return shape


@pytest_cases.fixture(scope="package")
@pytest_cases.parametrize("seed", [0, 1, 2], idgen="seed{seed}")
def args0(
    random_process: randprocs.RandomProcess,
    seed: int,
    args0_batch_shape: ShapeType,
) -> backend.Array:
    """Input(s) to a random process."""
    args0_shape = args0_batch_shape + random_process.input_shape

    return backend.random.standard_normal(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=seed, shape=args0_shape
        ),
        shape=args0_shape,
    )
