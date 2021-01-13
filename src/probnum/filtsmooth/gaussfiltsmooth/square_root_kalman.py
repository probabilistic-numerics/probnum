"""Square-root Gaussian filtering and smoothing."""

import numpy as np

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

    def predict(self, start, stop, randvar, **kwargs):
        if isinstance(self.dynamics_model, pnfss.LTISDE):
            disc_model = self.dynamics_model.discretise(stop - start)
            A = disc_model.driftmat
            s = disc_model.forcevec
            L_Q = disc_model.diffmat_cholesky
        else:
            A = self.dynamics_model.driftmatfun(start)
            s = self.dynamics_model.forcevecfun(start)
            L_Q = self.dynamics_model.diffmatfun_cholesky(start)

        old_mean = randvar.mean
        old_cov_cholesky = randvar.cov_cholesky

        new_mean = A @ old_mean + s
        new_cov_cholesky = cholesky_update(A @ old_cov_cholesky, L_Q)
        new_cov = new_cov_cholesky @ new_cov_cholesky.T
        return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky)

    def measure(self, time, randvar):
        if isinstance(self.measurement_model, DiscreteEKFComponent):
            self.measurement_model.linearize(at_this_rv=randvar)
            disc_model = self.linearized_model
        else:
            disc_model = self.measurement_model

        A = self.disc_model.driftmatfun(start)
        s = self.disc_model.forcevecfun(start)
        L_Q = self.disc_model.diffmatfun_cholesky(start)

        old_mean = randvar.mean
        old_cov_cholesky = randvar.cov_cholesky

        new_mean = A @ old_mean + s
        new_cov_cholesky = cholesky_update(A @ old_cov_cholesky, L_Q)
        new_cov = new_cov_cholesky @ new_cov_cholesky.T
        return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky)

    def update(self, time, randvar, data):

        pass

    def smooth_step(
        self, unsmoothed_rv, smoothed_rv, start, stop, intermediate_step=None
    ):
        pass


def square_root_update_matrices(L_R, H, L_C):

    zeros = np.zeros(H.T.shape)
    blockmat = np.block(((L_R, H @ L_C), (zeros, L_C))).T
    big_R = np.linalg.qr(blockmat, mode="r")

    size_small = len(H)

    L_S = big_R[:size_small, :size_small].T
    L_P = big_R[size_small:, size_small:].T
    K = np.linalg.solve(L_S, big_R[:size_small, size_small:]).T

    return L_S, K, L_P


def square_root_smoothing_matrices(L_P_unsmoothed_past, A, L_Q, L_P_smoothed_fut, G):

    zeros = np.zeros(A.shape)
    blockmat = np.block(
        (
            (L_P_unsmoothed_past.T @ A.T, L_P_unsmoothed_past.T),
            (L_Q.T, zeros),
            (zeros, L_P_smoothed_fut.T @ G.T),
        )
    )
    big_R = np.linalg.qr(blockmat, mode="r")

    size_small = len(A)

    L_P_smoothed = big_R[size_small : 2 * size_small, size_small:]

    return L_P_smoothed
