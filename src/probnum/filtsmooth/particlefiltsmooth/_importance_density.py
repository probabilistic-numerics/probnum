"""Importance densities."""

from probnum import statespace


class KalmanImportanceDensity(statespace.Transition):
    """Wrap a Gaussian filter into a Transition so that it can be used as an importance
    density in a particle filter."""

    def __init__(self, kalman):
        self.kalman = kalman

    def forward_realization(
        self,
        realization,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        return self.kalman.filter_step

    pass
