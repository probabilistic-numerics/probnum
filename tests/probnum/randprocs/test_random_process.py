"""Tests for random processes."""

from probnum import backend, compat, randprocs, randvars
from probnum.backend.typing import ShapeType

import pytest
import tests.utils

# pylint: disable=invalid-name


def test_output_shape(
    random_process: randprocs.RandomProcess,
    args0: backend.Array,
    args0_batch_shape: ShapeType,
):
    """Test whether evaluations of the random process have the correct shape."""
    expected_shape = args0_batch_shape + random_process.output_shape
    assert random_process(args0).shape == expected_shape


def test_mean_shape(
    random_process: randprocs.RandomProcess,
    args0: backend.Array,
    args0_batch_shape: ShapeType,
):
    """Test whether the mean of the random process has the correct shape."""
    expected_shape = args0_batch_shape + random_process.output_shape
    assert random_process.mean(args0).shape == expected_shape


def test_var_shape(
    random_process: randprocs.RandomProcess,
    args0: backend.Array,
    args0_batch_shape: ShapeType,
):
    """Test whether the variance of the random process has the correct shape."""
    expected_shape = args0_batch_shape + random_process.output_shape
    assert random_process.var(args0).shape == expected_shape


def test_std_shape(
    random_process: randprocs.RandomProcess,
    args0: backend.Array,
    args0_batch_shape: ShapeType,
):
    """Test whether the standard deviation of the random process has the correct
    shape."""
    expected_shape = args0_batch_shape + random_process.output_shape
    assert random_process.std(args0).shape == expected_shape


def test_cov_shape(
    random_process: randprocs.RandomProcess,
    args0: backend.Array,
    args0_batch_shape: ShapeType,
):
    """Test whether the covariance of the random process has the correct shape."""
    expected_shape = 2 * args0_batch_shape + 2 * random_process.output_shape
    assert random_process.cov.matrix(args0).shape == expected_shape


def test_evaluated_random_process_is_random_variable(
    random_process: randprocs.RandomProcess,
):
    """Test whether evaluating a random process returns a random variable."""
    args0_shape = (10,) + random_process.input_shape
    args0 = backend.random.standard_normal(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=98332,
            shape=args0_shape,
        ),
        shape=args0_shape,
    )
    y0 = random_process(args0)

    assert isinstance(y0, randvars.RandomVariable), (
        f"Output of {repr(random_process)} is not a " f"random variable."
    )


@pytest.mark.xfail(reason="Not yet implemented for random processes.")
def test_samples_are_callables(random_process: randprocs.RandomProcess):
    """When not specifying inputs to the sample method it should return ``size`` number
    of callables."""
    assert callable(random_process.sample(rng_state=backend.random.rng_state(42)))


@pytest.mark.xfail(reason="Not yet implemented for random processes.")
def test_sample_paths_are_deterministic_functions(
    random_process: randprocs.RandomProcess, args0: backend.Array
):
    """When sampling paths from a random process, repeated evaluation of the sample path
    at the same inputs should return the same values."""
    sample_path = random_process.sample(rng_state=backend.random.rng_state(43))
    compat.testing.assert_array_equal(sample_path(args0), sample_path(args0))


def test_rp_mean_cov_evaluated_matches_rv_mean_cov(
    random_process: randprocs.RandomProcess,
):
    """Check whether the evaluated mean and covariance function of a random process is
    equivalent to the mean and covariance of the evaluated random process as a random
    variable."""
    x_shape = (10,) + random_process.input_shape
    x = backend.random.standard_normal(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=98332,
            shape=x_shape,
        ),
        shape=x_shape,
    )

    compat.testing.assert_allclose(
        random_process(x).mean,
        random_process.mean(x),
        err_msg=f"Mean of evaluated {repr(random_process)} does not match the "
        f"random process mean function evaluated.",
    )

    compat.testing.assert_allclose(
        random_process(x).cov,
        random_process.cov.matrix(x),
        err_msg=f"Covariance of evaluated {repr(random_process)} does not match the "
        f"random process mean function evaluated.",
    )


class DummyRandomProcess(randprocs.RandomProcess):
    def __call__(self, args):
        raise NotImplementedError


def test_invalid_mean_type_raises():
    with pytest.raises(TypeError):
        DummyRandomProcess(
            input_shape=(),
            output_shape=(),
            dtype=backend.float64,
            mean=backend.zeros_like,
        )


def test_invalid_cov_type_raises():
    with pytest.raises(TypeError):
        DummyRandomProcess(
            input_shape=(),
            output_shape=(3,),
            dtype=backend.float64,
            cov=lambda x: backend.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x,
                shape=x.shape + (3, 3),
            ),
        )


def test_inconsistent_mean_shape_errors():
    with pytest.raises(ValueError):
        DummyRandomProcess(
            input_shape=(42,),
            output_shape=(),
            dtype=backend.float64,
            mean=randprocs.mean_fns.Zero(
                input_shape=(3,),
                output_shape=(3,),
            ),
        )

    with pytest.raises(ValueError):
        DummyRandomProcess(
            input_shape=(),
            output_shape=(1,),
            dtype=backend.float64,
            mean=randprocs.mean_fns.Zero(
                input_shape=(),
                output_shape=(3,),
            ),
        )


def test_inconsistent_cov_shape_errors():
    with pytest.raises(ValueError):
        DummyRandomProcess(
            input_shape=(42,),
            output_shape=(),
            dtype=backend.float64,
            cov=randprocs.kernels.ExpQuad(
                input_shape=(3,),
            ),
        )

    with pytest.raises(ValueError):
        DummyRandomProcess(
            input_shape=(),
            output_shape=(1,),
            dtype=backend.float64,
            cov=randprocs.kernels.ExpQuad(
                input_shape=(),
            ),
        )
