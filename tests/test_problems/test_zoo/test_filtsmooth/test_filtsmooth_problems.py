import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo

all_filtmooth_setups = pytest.mark.parametrize(
    "filtsmooth_setup",
    [
        filtsmooth_zoo.benes_daum(),
        filtsmooth_zoo.car_tracking(),
        filtsmooth_zoo.logistic_ode(),
        filtsmooth_zoo.ornstein_uhlenbeck(),
        filtsmooth_zoo.pendulum(),
    ],
)
