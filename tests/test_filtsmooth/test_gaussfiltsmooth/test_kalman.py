import numpy as np

from probnum.filtsmooth.gaussfiltsmooth import Kalman, IteratedKalman, StoppingCriterion

from .filtsmooth_testcases import CarTrackingDDTestCase, OrnsteinUhlenbeckCDTestCase

np.random.seed(5472)
VISUALISE = False  # show plots or not?

if VISUALISE is True:
    import matplotlib.pyplot as plt


class TestKalmanDiscreteDiscrete(CarTrackingDDTestCase):
    """
    Kalman filtering and smoothing on a discrete setting,
    i.e. the car tracking problem.

    By comparing filtering and smoothing RMSEs on the test problem,
    all methods in Kalman() are called.
    """

    def setUp(self):
        super().setup_cartracking()
        self.method = Kalman(self.dynmod, self.measmod, self.initrv)

    def test_filtsmooth(self):
        """
        RMSE of smoother smaller than rmse of filter smaller
        than of measurements?
        """
        filter_posterior = self.method.filter(self.obs, self.tms)
        filtms = filter_posterior.state_rvs.mean
        smooth_posterior = self.method.filtsmooth(self.obs, self.tms)
        smooms = smooth_posterior.state_rvs.mean

        normaliser = np.sqrt(self.states[1:, :2].size)
        filtrmse = np.linalg.norm(filtms[1:, :2] - self.states[1:, :2]) / normaliser
        smoormse = np.linalg.norm(smooms[1:, :2] - self.states[1:, :2]) / normaliser
        obs_rmse = np.linalg.norm(self.obs - self.states[1:, :2]) / normaliser

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


class TestKalmanContinuousDiscrete(OrnsteinUhlenbeckCDTestCase):
    """
    Try Kalman filtering on a continuous-discrete setting.

    Try OU process.
    """

    def setUp(self):
        super().setup_ornsteinuhlenbeck()
        self.method = Kalman(self.dynmod, self.measmod, self.initrv)

    def test_filtsmooth(self):
        """
        RMSE of smoother smaller than rmse of filter smaller
        than of measurements?
        """
        filter_posterior = self.method.filter(self.obs, self.tms)
        filtms = filter_posterior.state_rvs.mean
        smooth_posterior = self.method.filtsmooth(self.obs, self.tms)
        smooms = smooth_posterior.state_rvs.mean

        self.assertEqual(filtms[1:].shape, self.states[1:].shape)
        self.assertEqual(smooms[1:].shape, self.states[1:].shape)
        self.assertEqual(self.obs.shape, self.states[1:].shape)

        normaliser = np.sqrt(self.states[1:].size)
        filtrmse = np.linalg.norm(filtms[1:] - self.states[1:]) / normaliser
        smoormse = np.linalg.norm(smooms[1:] - self.states[1:]) / normaliser
        obs_rmse = np.linalg.norm(self.obs - self.states[1:]) / normaliser

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


class MockStoppingCriterion(StoppingCriterion):

    def __init__(self, max_num_updates=5):
        self.num_filter_updates = 0
        # self.num_filtsmooth_updates = 0
        self.max_num_updates = max_num_updates

    def continue_filter_updates(self, predrv=None, info_pred=None, filtrv=None, meas_rv=None, info_upd=None, **kwargs):
        """
        When do we stop iterating the filter steps. Default is true.
        If, e.g. IEKF is wanted, overwrite with something that does not always return True.
        """
        if self.num_filter_updates < self.max_num_updates:
            self.num_filter_updates += 1
            return True
        else:
            return False


    def continue_filtsmooth_updates(self, **kwargs):
        """If implemented, iterated_filtsmooth() is unlocked."""
        if self.num_filtsmooth_updates < self.max_num_updates:
            self.num_filtsmooth_updates += 1
            return True
        else:
            return False


class TestIteratedKalman(OrnsteinUhlenbeckCDTestCase):

    def setUp(self):
        super().setup_ornsteinuhlenbeck()
        kalman = Kalman(self.dynmod, self.measmod, self.initrv)
        self.max_updates = 5
        self.method = IteratedKalman(kalman, stoppingcriterion=MockStoppingCriterion(max_num_updates=self.max_updates))

    def test_filter_step(self):
        self.assertEqual(self.method.stoppingcriterion.num_filter_updates, 0)
        self.method.filter_step(start=0., stop=1., current_rv=self.initrv, data=0.)
        self.assertEqual(self.method.stoppingcriterion.num_filter_updates, 5)

    # def test_filtsmooth_step(self):
    #     self.assertEqual(self.method.stoppingcriterion.num_filter_updates, 0)
    #     self.method.iterated_filtsmooth(current_rv=self.initrv, data=0.)
    #     self.assertEqual(self.method.stoppingcriterion.num_filter_updates, 5)
