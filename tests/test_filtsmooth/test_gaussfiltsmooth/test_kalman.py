""""""
import scipy.linalg

from probnum.filtsmooth.gaussfiltsmooth import *

from .filtsmooth_testcases import *

np.random.seed(5472)
VISUALISE = False  # show plots or not?

if VISUALISE is True:
    import matplotlib.pyplot as plt


class TestKalmanDiscreteDiscrete(CarTrackingDDTestCase):
    """
    Try Kalman filtering and smoothing on a discrete setting.
    """

    def setUp(self):
        """
        """
        super().setup_cartracking()
        self.filt = KalmanFilter(self.dynmod, self.measmod, self.initrv)
        self.smoo = KalmanSmoother(self.dynmod, self.measmod, self.initrv)

    def test_dynamicmodel(self):
        """
        """
        self.assertEqual(self.dynmod, self.filt.dynamicmodel)
        self.assertEqual(self.dynmod, self.smoo.dynamicmodel)

    def test_measurementmodel(self):
        """
        """
        self.assertEqual(self.measmod, self.filt.measurementmodel)
        self.assertEqual(self.measmod, self.smoo.measurementmodel)

    def test_initialdistribution(self):
        """
        """
        self.assertEqual(self.initrv, self.filt.initialrandomvariable)
        self.assertEqual(self.initrv, self.smoo.initialrandomvariable)

    def test_predict(self):
        """
        """
        pred, __ = self.filt.predict(0., self.delta_t, self.initrv)
        self.assertEqual(pred.mean().ndim, 1)
        self.assertEqual(pred.mean().shape[0], 4)
        self.assertEqual(pred.cov().ndim, 2)
        self.assertEqual(pred.cov().shape[0], 4)
        self.assertEqual(pred.cov().shape[1], 4)

    def test_update(self):
        """
        """
        data = self.measmod.sample(0., self.initrv.mean())
        upd, __, __, __ = self.filt.update(0., self.initrv, data)
        self.assertEqual(upd.mean().ndim, 1)
        self.assertEqual(upd.mean().shape[0], 4)
        self.assertEqual(upd.cov().ndim, 2)
        self.assertEqual(upd.cov().shape[0], 4)
        self.assertEqual(upd.cov().shape[1], 4)

    def test_filtsmooth(self):
        """
        RMSE of smoother smaller than rmse of filter smaller
        than of measurements?
        """
        filtms, filtcs, filtts = self.filt.filter_set(self.obs, self.tms)
        smooms, smoocs, smoots = self.smoo.smooth(self.obs, self.tms)

        normaliser = np.sqrt(self.states[1:, :2].size)
        filtrmse = np.linalg.norm(filtms[1:, :2] - self.states[1:, :2]) / normaliser
        smoormse = np.linalg.norm(smooms[1:, :2] - self.states[1:, :2]) / normaliser
        obs_rmse = np.linalg.norm(self.obs - self.states[1:, :2]) / normaliser

        if VISUALISE is True:
            plt.title("Car tracking trajectory (%.2f " % smoormse
                      + "< %.2f < %.2f?)" % (filtrmse, obs_rmse))
            plt.plot(self.obs[:, 0], self.obs[:, 1], '.',
                     label="Observations", alpha=0.5)
            plt.plot(filtms[:, 0], filtms[:, 1], '-', label="Filter guess")
            plt.plot(smooms[:, 0], smooms[:, 1], '-', label="Smoother guess")
            plt.plot(self.states[:, 0], self.states[:, 1], '-',
                     linewidth=6, alpha=0.25, label="Truth")
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
        """ """
        super().setup_ornsteinuhlenbeck()
        self.smoo = KalmanSmoother(self.dynmod, self.measmod, self.initrv)
        self.filt = KalmanFilter(self.dynmod, self.measmod, self.initrv)

    def test_dynamicmodel(self):
        """
        """
        self.assertEqual(self.dynmod, self.smoo.dynamicmodel)
        self.assertEqual(self.dynmod, self.filt.dynamicmodel)

    def test_measurementmodel(self):
        """
        """
        self.assertEqual(self.measmod, self.smoo.measurementmodel)
        self.assertEqual(self.measmod, self.filt.measurementmodel)

    def test_initialdistribution(self):
        """
        """
        self.assertEqual(self.initrv, self.smoo.initialrandomvariable)
        self.assertEqual(self.initrv, self.filt.initialrandomvariable)


    def test_predict_shape(self):
        """
        """
        pred, __ = self.filt.predict(0., self.delta_t, self.initrv)
        self.assertEqual(np.isscalar(pred.mean()), True)
        self.assertEqual(np.isscalar(pred.cov()), True)

    def test_predict_value(self):
        """
        """
        pred, __ = self.filt.predict(0., self.delta_t, self.initrv)
        ah = scipy.linalg.expm(self.delta_t * self.drift)
        qh = self.q / (2 * self.lam) \
             * (1 - scipy.linalg.expm(2 * self.drift * self.delta_t))
        expectedmean = np.squeeze(ah @ (self.initrv.mean()*np.ones(1)))
        expectedcov = np.squeeze(ah @ (self.initrv.cov()*np.eye(1)) @ ah.T + qh)
        self.assertAlmostEqual(float(expectedmean), pred.mean())
        self.assertAlmostEqual(float(expectedcov), pred.cov())

    def test_update(self):
        """
        """
        data = np.array([self.measmod.sample(0., self.initrv.mean()*np.ones(1))])
        upd, __, __, __ = self.filt.update(0., self.initrv, data)
        self.assertEqual(np.isscalar(upd.mean()), True)
        self.assertEqual(np.isscalar(upd.cov()), True)

    def test_smoother(self):
        """
        RMSE of filter smaller than rmse of measurements?
        """
        filtms, filtcs, filtts = self.filt.filter_set(self.obs, self.tms)
        smooms, smoocs, smoots = self.smoo.smooth(self.obs, self.tms)

        normaliser = np.sqrt(self.states[1:].size)
        filtrmse = np.linalg.norm(filtms[1:] - self.states[1:, 0]) / normaliser
        smoormse = np.linalg.norm(smooms[1:] - self.states[1:, 0]) / normaliser
        obs_rmse = np.linalg.norm(self.obs - self.states[1:]) / normaliser

        if VISUALISE is True:
            plt.title("Ornstein Uhlenbeck (%.2f < " % smoormse
                      + "%.2f < %.2f?)" % (filtrmse, obs_rmse))
            plt.plot(self.tms[1:], self.obs[:, 0], '.',
                     label="Observations", alpha=0.5)
            plt.plot(filtts, filtms, '-', label="Filter guess")
            plt.plot(smoots, smooms, '-', label="Smoother guess")
            plt.plot(self.tms, self.states, '-',
                     linewidth=6, alpha=0.25, label="Truth")
            plt.legend()
            plt.show()
        self.assertLess(smoormse, filtrmse)
        self.assertLess(filtrmse, obs_rmse)