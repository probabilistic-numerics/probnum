"""Square-root Gaussian filtering and smoothing."""

from .kalman import Kalman
from .utils import cholesky_update


class SquareRootKalman(Kalman):
    def __init__(self, dynamics_model, measurement_model, initrv):
        """Check that the initial distribution is Gaussian."""

        # check if LTI or discreteLTI for dynamics and measurements
        # or if at least EKF without diffusion

        super().__init__(dynamics_model, measurement_model, initrv)

    def predict(self, start, stop, randvar, intermediate_step=None):
        pass

    def measure(self, time, randvar):
        pass

    def update(self, time, randvar, data):
        pass

    def smooth_step(
        self, unsmoothed_rv, smoothed_rv, start, stop, intermediate_step=None
    ):
        pass
