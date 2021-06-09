"""Interfaces for Bayesian filtering and smoothing."""

from abc import ABC
from typing import Dict, Optional, Tuple, Union

import numpy as np

from probnum import randprocs, randvars
from probnum.type import FloatArgType

from ._timeseriesposterior import TimeSeriesPosterior


class BayesFiltSmooth(ABC):
    """Bayesian filtering and smoothing."""

    def __init__(
        self,
        prior_process: randprocs.MarkovProcess,
    ):
        self.prior_process = prior_process

    # Interfaces for filter() and smooth() are gonna follow soon.
