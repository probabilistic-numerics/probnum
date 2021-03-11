"""Specific types for filtering and smoothing."""

from typing import Union

import numpy as np

from probnum import statespace
from probnum.filtsmooth.gaussfiltsmooth.extendedkalman import (
    ContinuousEKFComponent,
    DiscreteEKFComponent,
)
from probnum.filtsmooth.gaussfiltsmooth.unscentedkalman import (
    ContinuousUKFComponent,
    DiscreteUKFComponent,
)
from probnum.type import FloatArgType

GaussMarkovPriorTransitionType = Union[
    statespace.DiscreteLinearGaussian,
    DiscreteEKFComponent,
    DiscreteUKFComponent,
    statespace.LinearSDE,
    ContinuousEKFComponent,
    ContinuousUKFComponent,
]

DenseOutputLocationArgType = Union[FloatArgType, np.ndarray]
