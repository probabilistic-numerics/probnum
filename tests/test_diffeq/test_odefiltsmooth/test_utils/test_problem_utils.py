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
    op = diffeq.odefiltsmooth.information_operators.ExplicitODEResidual(
        prior_ordint=3, prior_spatialdim=2
    )
    return op


@pytest.mark.parametrize(
    "ivp", [diffeq_zoo.lotkavolterra(), diffeq_zoo.fitzhughnagumo()]
)
@pytest.mark.parametrize("ode_measurement_variance", [None, 1.0])
def test_ivp_to_regression_problem(
    ivp, locations, ode_information_operator, ode_measurement_variance
):

    ode_information_operator.incorporate_ode(ode=ivp)
    regprob = diffeq.odefiltsmooth.utils.ivp_to_regression_problem(
        ivp, locations, ode_information_operator, ode_measurement_variance
    )
    assert isinstance(regprob, problems.TimeSeriesRegressionProblem)
    assert len(regprob.locations) == len(locations)
    assert len(regprob.locations) == len(locations)
    assert len(regprob.measurement_models) == len(locations)
    assert isinstance(regprob.measurement_models[0], statespace.DiscreteLTIGaussian)
    assert isinstance(regprob.measurement_models[0], statespace.DiscreteLTIGaussian)
    assert isinstance(regprob.measurement_models[1], statespace.DiscreteGaussian)
    assert isinstance(regprob.measurement_models[-1], statespace.DiscreteGaussian)
