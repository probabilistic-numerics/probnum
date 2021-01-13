"""Square-root Gaussian filtering and smoothing."""

from .extendedkalman import DiscreteEKFComponent
from .kalman import Kalman
from .utils import cholesky_update


# This class re-implements predict and measure,
# because e.g. update MUST be in here and therefore it seems more
# didactic to do everything here.
class SquareRootKalman(Kalman):
    def __init__(self, dynamics_model, measurement_model, initrv):
        """Check that the models are linear(ised)."""

        if not isinstance(dynamics_model, (pnfss.LTISDE, pnfss.DiscreteLinearGaussian)):
            raise ValueError

        # EKF is acceptable, because of the Taylor linearisation.
        # UKF would need downdates, which is not supported at the moment.
        if not isinstance(
            measurement_model, (pnfss.DiscreteLinearGaussian, DiscreteEKFComponent)
        ):
            raise ValueError

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
