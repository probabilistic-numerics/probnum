import numpy as np
import scipy.linalg

from probnum.filtsmooth.gaussfiltsmooth import ExtendedKalman

from .filtsmooth_testcases import (
    CarTrackingDDTestCase,
    OrnsteinUhlenbeckCDTestCase,
    PendulumNonlinearDDTestCase,
)

np.random.seed(5472)

VISUALISE = False  # show plots or not?
if VISUALISE is True:
    import matplotlib.pyplot as plt


class TestExtendedKalmanDiscDisc(CarTrackingDDTestCase):
    """
    Try Kalman filtering and smoothing on a discrete setting.
    """

    def setUp(self):
        super().setup_cartracking()
        self.method = ExtendedKalman(self.dynmod, self.measmod, self.initrv)

    def test_dynamicmodel(self):
        self.assertEqual(self.dynmod, self.method.dynamicmodel)

    def test_measurementmodel(self):
        self.assertEqual(self.measmod, self.method.measurementmodel)

    def test_initialdistribution(self):
        self.assertEqual(self.initrv, self.method.initialrandomvariable)

    def test_predict(self):
        pred, __ = self.method.predict(0.0, self.delta_t, self.initrv)
        self.assertEqual(pred.mean.ndim, 1)
        self.assertEqual(pred.mean.shape[0], 4)
        self.assertEqual(pred.cov.ndim, 2)
        self.assertEqual(pred.cov.shape[0], 4)
        self.assertEqual(pred.cov.shape[1], 4)

    def test_update(self):
        data = self.measmod.sample(0.0, self.initrv.mean)
        upd, __, __, __ = self.method.update(0.0, self.initrv, data)
        self.assertEqual(upd.mean.ndim, 1)
        self.assertEqual(upd.mean.shape[0], 4)
        self.assertEqual(upd.cov.ndim, 2)
        self.assertEqual(upd.cov.shape[0], 4)
        self.assertEqual(upd.cov.shape[1], 4)

    def test_filtsmooth(self):
        """
        RMSE of smoother smaller than rmse of filter smaller
        than of measurements?
        """
        filter_posterior = self.method.filter(self.obs, self.tms)
        filtms = filter_posterior.state_rvs.mean()
        filtcs = filter_posterior.state_rvs.cov()
        smooth_posterior = self.method.filtsmooth(self.obs, self.tms)
        smooms = smooth_posterior.state_rvs.mean()
        smoocs = smooth_posterior.state_rvs.cov()

        comp = self.states[1:, :2]
        normaliser = np.sqrt(comp.size)
        filtrmse = np.linalg.norm(filtms[1:, :2] - comp) / normaliser
        smoormse = np.linalg.norm(smooms[1:, :2] - comp) / normaliser
        obs_rmse = np.linalg.norm(self.obs[:, :2] - comp) / normaliser

        if VISUALISE is True:
            plt.title(
                "Car tracking trajectory (%.2f " % smoormse
                + "< %.2f < %.2f?)" % (filtrmse, obs_rmse)
            )
            plt.plot(
                self.obs[:, 0], self.obs[:, 1], ".", label="Observations", alpha=0.5
            )
            plt.plot(filtms[:, 0], filtms[:, 1], "-", label="Filter guess")
            plt.plot(smooms[:, 0], smooms[:, 1], "-", label="Smoother guess")
            plt.plot(
                self.states[:, 0],
                self.states[:, 1],
                "-",
                linewidth=6,
                alpha=0.25,
                label="Truth",
            )
            plt.legend()
            plt.show()
        self.assertLess(smoormse, filtrmse)
        self.assertLess(filtrmse, obs_rmse)


class TestExtendedKalmanContDisc(OrnsteinUhlenbeckCDTestCase):
    """
    Try Kalman filtering on a continuous-discrete setting.

    Try OU process.
    """

    def setUp(self):
        super().setup_ornsteinuhlenbeck()
        self.method = ExtendedKalman(self.dynmod, self.measmod, self.initrv)

    def test_dynamicmodel(self):
        self.assertEqual(self.dynmod, self.method.dynamicmodel)

    def test_measurementmodel(self):
        self.assertEqual(self.measmod, self.method.measurementmodel)

    def test_initialdistribution(self):
        self.assertEqual(self.initrv, self.method.initialrandomvariable)

    def test_predict_shape(self):
        pred, __ = self.method.predict(0.0, self.delta_t, self.initrv)
        self.assertEqual(pred.mean.shape, (1,))
        self.assertEqual(pred.cov.shape, (1, 1))

    def test_predict_value(self):
        pred, __ = self.method.predict(0.0, self.delta_t, self.initrv)
        ah = scipy.linalg.expm(self.delta_t * self.drift)
        qh = (
            self.q
            / (2 * self.lam)
            * (1 - scipy.linalg.expm(2 * self.drift * self.delta_t))
        )
        expectedmean = np.squeeze(ah @ (self.initrv.mean * np.ones(1)))
        expectedcov = np.squeeze(ah @ (self.initrv.cov * np.eye(1)) @ ah.T + qh)
        self.assertApproxEqual(expectedmean, pred.mean)
        self.assertApproxEqual(expectedcov, pred.cov)

    def test_update(self):
        data = self.measmod.sample(0.0, self.initrv.mean * np.ones(1))
        upd, __, __, __ = self.method.update(0.0, self.initrv, data)
        self.assertEqual(upd.mean.shape, (1,))
        self.assertEqual(upd.cov.shape, (1, 1))

    def test_smoother(self):
        """
        RMSE of filter smaller than rmse of measurements?
        """
        filter_posterior = self.method.filter(self.obs, self.tms)
        filtms = filter_posterior.state_rvs.mean()
        filtcs = filter_posterior.state_rvs.cov()
        smooth_posterior = self.method.filtsmooth(self.obs, self.tms)
        smooms = smooth_posterior.state_rvs.mean()
        smoocs = smooth_posterior.state_rvs.cov()

        comp = self.states[1:]
        normaliser = np.sqrt(comp.size)
        filtrmse = np.linalg.norm(filtms[1:] - comp) / normaliser
        smoormse = np.linalg.norm(smooms[1:] - comp) / normaliser
        obs_rmse = np.linalg.norm(self.obs - comp) / normaliser

        if VISUALISE is True:
            plt.title(
                "Ornstein Uhlenbeck (%.2f < " % smoormse
                + "%.2f < %.2f?)" % (filtrmse, obs_rmse)
            )
            plt.plot(self.tms[1:], self.obs[:, 0], ".", label="Observations", alpha=0.5)
            plt.plot(self.tms, filtms, "-", label="Filter guess")
            plt.plot(self.tms, smooms, "-", label="Smoother guess")
            plt.plot(self.tms, self.states, "-", linewidth=6, alpha=0.25, label="Truth")
            plt.legend()
            plt.show()
        self.assertLess(smoormse, filtrmse)
        self.assertLess(filtrmse, obs_rmse)


class TestExtendedKalmanPendulum(PendulumNonlinearDDTestCase):
    """
    We test on the pendulum example 5.1 in BFaS.
    """

    def setUp(self):
        super().setup_pendulum()
        self.method = ExtendedKalman(self.dynamod, self.measmod, self.initrv)

    def test_filtsmooth(self):
        filter_posterior = self.method.filter(self.obs, self.tms)
        filtms = filter_posterior.state_rvs.mean()
        filtcs = filter_posterior.state_rvs.cov()
        smooth_posterior = self.method.filtsmooth(self.obs, self.tms)
        smooms = smooth_posterior.state_rvs.mean()
        smoocs = smooth_posterior.state_rvs.cov()

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
