"""Square-root Gaussian filtering and smoothing."""

from .kalman import Kalman


class SquareRootKalman(Kalman):
    def __init__(self, dynamics_model, measurement_model, initrv):
        """Check that the initial distribution is Gaussian."""

        super().__init__(dynamics_model, measurement_model, initrv)
