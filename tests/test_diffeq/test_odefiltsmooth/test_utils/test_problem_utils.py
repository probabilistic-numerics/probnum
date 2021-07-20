"""Test for utility functions for odefiltsmooth problem conversion."""


import numpy as np
import pytest

from probnum import diffeq, problems, statespace
from probnum.problems.zoo import diffeq as diffeq_zoo


@pytest.fixture
def locations():
    return np.linspace(0.0, 1.0, 20)


@pytest.fixture
def ode_information_operator():
    op = diffeq.odefiltsmooth.information_operators.ODEResidual(
        prior_ordint=3, prior_spatialdim=2
    )
    return op


@pytest.mark.parametrize(
    "ivp", [diffeq_zoo.lotkavolterra(), diffeq_zoo.fitzhughnagumo()]
)
@pytest.mark.parametrize("ode_measurement_variance", [None, 1.0])
@pytest.mark.parametrize(
    "approx_strategy",
    [
        None,
        diffeq.odefiltsmooth.approx_strategies.EK0(),
        diffeq.odefiltsmooth.approx_strategies.EK1(),
    ],
)
def test_ivp_to_regression_problem(
    ivp, locations, ode_information_operator, approx_strategy, ode_measurement_variance
):

    ode_information_operator.incorporate_ode(ode=ivp)
    regprob = diffeq.odefiltsmooth.utils.ivp_to_regression_problem(
        ivp=ivp,
        locations=locations,
        ode_information_operator=ode_information_operator,
        approx_strategy=approx_strategy,
        ode_measurement_variance=ode_measurement_variance,
    )
    assert isinstance(regprob, problems.TimeSeriesRegressionProblem)
    assert len(regprob.locations) == len(locations)
    assert len(regprob.locations) == len(locations)
    assert len(regprob.measurement_models) == len(locations)
    assert isinstance(regprob.measurement_models[0], statespace.DiscreteLTIGaussian)
    assert isinstance(regprob.measurement_models[0], statespace.DiscreteLTIGaussian)
    assert isinstance(regprob.measurement_models[1], statespace.DiscreteGaussian)
    assert isinstance(regprob.measurement_models[-1], statespace.DiscreteGaussian)

    if ode_measurement_variance is not None:
        cov = regprob.measurement_models[1].proc_noise_cov_mat_fun(locations[0])
        cov_cholesky = regprob.measurement_models[1].proc_noise_cov_cholesky_fun(
            locations[0]
        )
        assert np.linalg.norm(cov > 0.0)
        assert np.linalg.norm(cov_cholesky > 0.0)
