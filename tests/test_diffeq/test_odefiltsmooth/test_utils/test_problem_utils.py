"""Test for utility functions for odefiltsmooth problem conversion."""


import numpy as np
import pytest

from probnum import diffeq, filtsmooth, problems, randprocs, randvars
from probnum.problems.zoo import diffeq as diffeq_zoo


@pytest.fixture
def locations():
    return np.linspace(0.0, 1.0, 20)


@pytest.fixture
def ode_information_operator():
    op = diffeq.odefiltsmooth.information_operators.ODEResidual(
        num_prior_derivatives=3, ode_dimension=2
    )
    return op


@pytest.mark.parametrize(
    "ivp", [diffeq_zoo.lotkavolterra(), diffeq_zoo.fitzhughnagumo()]
)
@pytest.mark.parametrize("ode_measurement_variance", [0.0, 1.0])
@pytest.mark.parametrize("exclude_initial_condition", [True, False])
@pytest.mark.parametrize(
    "approx_strategy",
    [
        None,
        diffeq.odefiltsmooth.approx_strategies.EK0(),
        diffeq.odefiltsmooth.approx_strategies.EK1(),
    ],
)
def test_ivp_to_regression_problem(
    ivp,
    locations,
    ode_information_operator,
    approx_strategy,
    ode_measurement_variance,
    exclude_initial_condition,
):
    """Test all possible parametrizations of ivp_to_regression_problem."""
    # Call function
    regprob = diffeq.odefiltsmooth.utils.ivp_to_regression_problem(
        ivp=ivp,
        locations=locations,
        ode_information_operator=ode_information_operator,
        approx_strategy=approx_strategy,
        ode_measurement_variance=ode_measurement_variance,
        exclude_initial_condition=exclude_initial_condition,
    )

    # Test basic properties
    assert isinstance(regprob, problems.TimeSeriesRegressionProblem)
    assert len(regprob.locations) == len(locations)
    assert len(regprob.locations) == len(locations)
    assert len(regprob.measurement_models) == len(locations)
    assert isinstance(
        regprob.measurement_models[1], randprocs.markov.discrete.DiscreteGaussian
    )
    assert isinstance(
        regprob.measurement_models[-1], randprocs.markov.discrete.DiscreteGaussian
    )

    # Depending on the desired exclusion of the initial condition,
    # the first element in the list of measurement models should
    # be LTIGaussian (for the initial condition) or DiscreteGaussian (for the ODE)
    if exclude_initial_condition:
        assert isinstance(
            regprob.measurement_models[0], randprocs.markov.discrete.DiscreteGaussian
        )
        assert isinstance(
            regprob.measurement_models[0], randprocs.markov.discrete.DiscreteGaussian
        )
    else:
        assert isinstance(
            regprob.measurement_models[0], randprocs.markov.discrete.DiscreteLTIGaussian
        )
        assert isinstance(
            regprob.measurement_models[0], randprocs.markov.discrete.DiscreteLTIGaussian
        )

    # If the ODE measurement variance is not None, i.e. not zero,
    # the process noise covariance matrices should be non-zero.
    if ode_measurement_variance > 0.0:
        cov = regprob.measurement_models[1].proc_noise_cov_mat_fun(locations[0])
        cov_cholesky = regprob.measurement_models[1].proc_noise_cov_cholesky_fun(
            locations[0]
        )
        assert np.linalg.norm(cov > 0.0)
        assert np.linalg.norm(cov_cholesky > 0.0)

    # If an approximation strategy is passed, the output should be an EKF component
    # which should suppoert forward_rv().
    # If not, the output is a generic DiscreteGaussian (which has been tested for above.
    # Recall that DiscreteEKFComponent implements DiscreteGaussian.)
    if approx_strategy is not None:
        assert isinstance(
            regprob.measurement_models[1],
            filtsmooth.gaussian.approx.DiscreteEKFComponent,
        )
        assert isinstance(
            regprob.measurement_models[-1],
            filtsmooth.gaussian.approx.DiscreteEKFComponent,
        )

        mm = regprob.measurement_models[1]  # should know forward_rv
        rv = randvars.Normal(np.zeros(mm.input_dim), cov=np.eye(mm.input_dim))
        new_rv, _ = mm.forward_rv(rv, t=locations[0])
        assert isinstance(new_rv, randvars.RandomVariable)
