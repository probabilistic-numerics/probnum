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

    def filter_step(
        self,
        start: FloatArgType,
        stop: FloatArgType,
        current_rv: randvars.RandomVariable,
        data: np.ndarray,
        _linearise_predict_at: Optional[randvars.RandomVariable] = None,
        _linearise_update_at: Optional[randvars.RandomVariable] = None,
        _diffusion: Union[FloatArgType, np.ndarray] = 1.0,
    ) -> Tuple[randvars.RandomVariable, Dict]:
        """Filter step.

        For Gaussian filters, this means a prediction step followed by
        an update step.
        """
        errormsg = (
            "filter_step(...) is not implemented for "
            + "the Bayesian filter {}.".format(type(self).__name__)
        )
        raise NotImplementedError(errormsg)

    def smooth(
        self,
        filter_posterior: TimeSeriesPosterior,
        _previous_posterior: Optional[TimeSeriesPosterior] = None,
    ) -> TimeSeriesPosterior:
        """Smoothing."""
        errormsg = (
            "smoother_step(...) is not implemented for "
            + "the Bayesian smoother {}.".format(type(self).__name__)
        )
        raise NotImplementedError(errormsg)
