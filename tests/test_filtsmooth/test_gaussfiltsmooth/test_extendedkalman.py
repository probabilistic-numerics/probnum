import numpy as np
import scipy.linalg

import probnum.filtsmooth as pnfs

from .filtsmooth_testcases import (
    CarTrackingDDTestCase,
    OrnsteinUhlenbeckCDTestCase,
    PendulumNonlinearDDTestCase,
)

np.random.seed(5472)

VISUALISE = False  # show plots or not?
if VISUALISE is True:
    import matplotlib.pyplot as plt



class TestExtendedKalmanPendulum(PendulumNonlinearDDTestCase):
    """
    We test on the pendulum example 5.1 in BFaS.
    """

    def setUp(self):
        super().setup_pendulum()
        self.ekf_meas = pnfs.DiscreteEKF(self.measmod)
        self.ekf_dyna = pnfs.DiscreteEKF(self.dynamod)
        self.method = pnfs.Kalman(self.ekf_dyna, self.ekf_meas, self.initrv)

    def test_filtsmooth(self):
        filter_posterior = self.method.filter(self.obs, self.tms)
        filtms = filter_posterior.state_rvs.mean
        smooth_posterior = self.method.filtsmooth(self.obs, self.tms)
        smooms = smooth_posterior.state_rvs.mean

        comp = self.states[:, 0]
        normaliser = np.sqrt(comp.size)
        filtrmse = np.linalg.norm(filtms[:, 0] - comp) / normaliser
        smoormse = np.linalg.norm(smooms[:, 0] - comp) / normaliser
        obs_rmse = np.linalg.norm(self.obs[:, 0] - comp[1:]) / normaliser

        if VISUALISE is True:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(
                "Noisy pendulum model (%.2f " % smoormse
                + "< %.2f < %.2f?)" % (filtrmse, obs_rmse)
            )
            ax1.set_title("Horizontal position")
            ax1.plot(
                self.tms[1:], self.obs[:, 0], ".", alpha=0.25, label="Observations"
            )
            ax1.plot(
                self.tms[1:],
                np.sin(self.states)[1:, 0],
                "-",
                linewidth=4,
                alpha=0.5,
                label="Truth",
            )
            ax1.plot(self.tms[1:], np.sin(filtms)[1:, 0], "-", label="Filter")
            ax1.plot(self.tms[1:], np.sin(smooms)[1:, 0], "-", label="Smoother")
            ax1.set_xlabel("time")
            ax1.set_ylabel("horizontal pos. = sin(angular)")
            ax1.legend()

            ax2.set_title("Angular position")
            ax2.plot(
                self.tms[1:],
                self.states[1:, 0],
                "-",
                linewidth=4,
                alpha=0.5,
                label="Truth",
            )
            ax2.plot(self.tms[1:], filtms[1:, 0], "-", label="Filter")
            ax2.plot(self.tms[1:], smooms[1:, 0], "-", label="Smoother")
            ax2.set_xlabel("time")
            ax2.set_ylabel("angular pos.")
            ax2.legend()
            plt.show()

        self.assertLess(smoormse, filtrmse)
        self.assertLess(filtrmse, obs_rmse)
