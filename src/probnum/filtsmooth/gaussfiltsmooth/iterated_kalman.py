"""Iterated Gaussian filtering and smoothing."""
import numpy as np

import probnum.random_variables as pnrv

from .kalman import Kalman
from .stoppingcriterion import StoppingCriterion


class IteratedKalman(Kalman):
    """Iterated filter/smoother based on posterior linearisation."""

    def __init__(self, kalman, stoppingcriterion=None):
        self.kalman = kalman
        if stoppingcriterion is None:
            self.stoppingcriterion = StoppingCriterion()
        else:
            self.stoppingcriterion = stoppingcriterion

        self.current_results = []
        self.previous_results = []

        super().__init__(kalman.dynamics_model, kalman.measurement_model, kalman.initrv)

    def iterated_filtsmooth(
        self, dataset, times, _intermediate_step=None, _linearise_at=None
    ):
        posterior = self.filtsmooth(
            dataset=dataset,
            times=times,
            _intermediate_step=_intermediate_step,
            _linearise_at=None,
        )
        err = np.array(self.current_results) - np.array(self.previous_results)
        self.current_results = []
        self.previous_results = []
        while not self.stoppingcriterion.terminate(
            error=err, reference=self.current_results
        ):
            posterior = self.filtsmooth(
                dataset=dataset,
                times=times,
                _intermediate_step=_intermediate_step,
                _linearise_at=posterior,
            )
            err = np.array(self.current_results) - np.array(self.previous_results)
            self.current_results = []
            self.previous_results = []
        return posterior

    def filter_step(
        self,
        start,
        stop,
        current_rv,
        data,
        _intermediate_step=None,
        _linearise_at=None,
        _diffusion=1.0,
    ):
        if _linearise_at is None:
            raise RuntimeError("Posterior linearisation expected.")

        filt_rv, info = self.kalman.filter_step(
            start=start,
            stop=stop,
            current_rv=current_rv,
            data=data,
            _intermediate_step=_intermediate_step,
            _linearise_at=_linearise_at,
            _diffusion=_diffusion,
        )
        old_mean = (
            _linearise_at.mean
            if _linearise_at is not None
            else np.zeros(filt_rv.mean.shape)
        )
        new_mean = filt_rv.mean
        self.current_results.append(new_mean)
        self.previous_results.append(old_mean)
