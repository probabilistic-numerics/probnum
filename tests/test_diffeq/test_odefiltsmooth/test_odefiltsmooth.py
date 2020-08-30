"""
We test on two test-problems:
    * logistic ODE (because it has a closed form sol.)
        -> make sure error converges to zero (even with rate q?)
        -> Check if iterates match the closed-form solutions in
        Schober et al.
    * Lotka-Volterra (because it provides meaningful uncertainty estimates,
    if e.g. EKF-based ODE filter is implemented correctly)
        -> error estimates from adaptive step sizes are roughly satsified
        (for the ibm1-kf combo, the other ones do not apply.)
"""

import unittest

import numpy as np

from probnum.diffeq.odefiltsmooth import probsolve_ivp
from probnum.diffeq import ode
from probnum.random_variables import Dirac

from tests.testing import NumpyAssertions


class TestConvergenceOnLogisticODE(unittest.TestCase):
    """
    We test whether the convergence rates roughly hold true.
    """

    def setUp(self):
        """Setup odesolver and solve a scalar ode"""
        initrv = Dirac(0.1 * np.ones(1))
        self.ivp = ode.logistic([0.0, 1.5], initrv)
        self.stps = [0.2, 0.1]

    def test_error_ibm1(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, which_prior="ibm1")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, which_prior="ibm1")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 2
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)

    def test_error_ibm2(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, which_prior="ibm2")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, which_prior="ibm2")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 3
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)

    def test_error_ibm3(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, which_prior="ibm3")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, which_prior="ibm3")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 4
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)

    def test_error_ioup1(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, which_prior="ioup1")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, which_prior="ioup1")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 2
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)

    def test_error_ioup2(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, which_prior="ioup2")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, which_prior="ioup2")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 3
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)

    def test_error_ioup3(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, which_prior="ioup3")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, which_prior="ioup3")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 4
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)


class TestFirstIterations(unittest.TestCase, NumpyAssertions):
    """
    Test whether first few means and covariances coincide with Prop. 1
    in Schober et al., 2019.
    """

    def setUp(self):
        initrv = Dirac(0.1 * np.ones(1))
        self.ivp = ode.logistic([0.0, 1.5], initrv)
        self.step = 0.5
        sol = probsolve_ivp(
            self.ivp, step=self.step, initrv=initrv, diffconst=1.0, which_prior="ibm1"
        )
        state_rvs = sol._state_rvs
        self.ms, self.cs = state_rvs.mean(), state_rvs.cov()

    def test_t0(self):
        exp_mean = np.array(
            [self.ivp.initrv.mean, self.ivp.rhs(0, self.ivp.initrv.mean)]
        )

        self.assertAllClose(self.ms[0], exp_mean[:, 0], rtol=1e-14)
        self.assertAllClose(self.cs[0], np.zeros((2, 2)), rtol=1e-14)

    def test_t1(self):
        """
        The kernels does not coincide exactly because of the
        uncertainty calibration that takes place in
        GaussianIVPFilter.solve()
        and not in Prop. 1 of Schober et al., 2019.
        """
        y0 = self.ivp.initrv.mean
        z0 = self.ivp.rhs(0, y0)
        z1 = self.ivp.rhs(0, y0 + self.step * z0)
        exp_mean = np.array([y0 + 0.5 * self.step * (z0 + z1), z1])
        self.assertAllClose(self.ms[1], exp_mean[:, 0], rtol=1e-14)


class TestAdaptivityOnLotkaVolterra(unittest.TestCase):
    """
    Only test on "ekf0" with IBM(1) prior, since every other combination
    seems to dislike the adaptive scheme based on the whitened residual
    as an error estimate.
    """

    def setUp(self):
        """Setup odesolver and solve a scalar ode"""
        initrv = Dirac(20 * np.ones(2))
        self.ivp = ode.lotkavolterra([0.0, 0.5], initrv)
        self.tol = 1e-2

    def test_kf_ibm1_stdev(self):
        """
        Standard deviation at end point roughly equal to tolerance.
        """
        sol = probsolve_ivp(self.ivp, tol=self.tol, which_prior="ibm1", method="ekf0")
        self.assertLess(np.sqrt(sol.y.cov()[-1, 0, 0]), 10 * self.tol)
        self.assertLess(0.1 * self.tol, np.sqrt(sol.y.cov()[-1, 0, 0]))

    def test_kf_ibm1(self):
        """
        Tests whether resulting steps are not evenly distributed.
        """
        sol = probsolve_ivp(self.ivp, tol=self.tol, which_prior="ibm1", method="ekf0")
        steps = np.diff(sol.t)
        self.assertLess(np.amin(steps) / np.amax(steps), 0.8)


class TestLotkaVolterraOtherPriors(unittest.TestCase):
    """
    We only test whether all the prior-filter-adaptivity combinations
    finish.
    """

    def setUp(self):
        """Setup odesolver and Lotka-Volterra IVP"""
        initrv = Dirac(20 * np.ones(2))
        self.ivp = ode.lotkavolterra([0.0, 0.5], initrv)
        self.tol = 1e-1
        self.step = 0.1

    def test_filter_ivp_ioup1_kf(self):
        probsolve_ivp(self.ivp, tol=self.tol, which_prior="ioup1", method="ekf0")

    def test_filter_ivp_ioup2_ekf(self):
        probsolve_ivp(self.ivp, tol=self.tol, which_prior="ioup2", method="ekf1")

    def test_filter_ivp_ioup3_ukf(self):
        """
        UKF requires some evaluation-variance to have a positive definite
        innovation matrix, apparently.
        """
        probsolve_ivp(
            self.ivp, tol=self.tol, evlvar=0.01, which_prior="ioup3", method="ukf"
        )

    def test_filter_ivp_h_ioup1_ekf(self):
        probsolve_ivp(self.ivp, step=self.step, which_prior="ioup1", method="ekf1")

    def test_filter_ivp_h_ioup2_ukf(self):
        """
        UKF requires some evaluation-variance to have a positive definite
        innovation matrix, apparently.
        """
        probsolve_ivp(
            self.ivp, step=self.step, evlvar=0.01, which_prior="ioup2", method="ukf"
        )

    def test_filter_ivp_h_ioup3_kf(self):
        probsolve_ivp(self.ivp, step=self.step, which_prior="ioup3", method="ekf0")

    def test_filter_ivp_mat32_kf(self):
        probsolve_ivp(self.ivp, tol=self.tol, which_prior="matern32", method="ekf0")

    def test_filter_ivp_mat52_ekf(self):
        probsolve_ivp(self.ivp, tol=self.tol, which_prior="matern52", method="ekf1")

    def test_filter_ivp_mat72_ukf(self):
        """
        UKF requires some evaluation-variance to have a positive definite
        innovation matrix, apparently.
        """
        probsolve_ivp(
            self.ivp, tol=self.tol, evlvar=0.01, which_prior="matern72", method="ukf"
        )

    def test_filter_ivp_h_mat32_ekf(self):
        probsolve_ivp(self.ivp, step=self.step, which_prior="matern32", method="ekf1")

    def test_filter_ivp_h_mat52_ukf(self):
        """
        UKF requires some evaluation-variance to have a positive definite
        innovation matrix, apparently.
        """
        probsolve_ivp(
            self.ivp, step=self.step, evlvar=0.01, which_prior="matern52", method="ukf"
        )

    def test_filter_ivp_h_mat72_kf(self):
        probsolve_ivp(self.ivp, step=self.step, which_prior="matern72", method="ekf0")


class TestConvergenceOnLogisticODESmoother(unittest.TestCase):
    """
    We test whether the convergence rates roughly hold true.
    """

    def setUp(self):
        """Setup odesolver and solve a scalar ode"""
        initrv = Dirac(0.1 * np.ones(1))
        self.ivp = ode.logistic([0.0, 1.5], initrv)
        self.stps = [0.2, 0.1]

    def test_error_ibm1(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, method="eks0", which_prior="ibm1")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, method="eks0", which_prior="ibm1")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 2
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)

    def test_error_ibm2(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, method="eks0", which_prior="ibm2")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, method="eks0", which_prior="ibm2")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 3
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)

    def test_error_ibm3(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, method="eks0", which_prior="ibm3")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, method="eks0", which_prior="ibm3")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 4
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)

    def test_error_ioup1(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, method="eks0", which_prior="ioup1")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, method="eks0", which_prior="ioup1")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 2
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)

    def test_error_ioup2(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, method="eks0", which_prior="ioup2")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, method="eks0", which_prior="ioup2")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 3
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)

    def test_error_ioup3(self):
        """Expect error rate q+1 """
        stp1, stp2 = self.stps
        sol = probsolve_ivp(self.ivp, step=stp1, method="eks0", which_prior="ioup3")
        means1 = sol.y.mean()
        sols1 = np.array([self.ivp.solution(t) for t in sol.t])
        err1 = np.amax(np.abs(sols1 - means1))
        sol = probsolve_ivp(self.ivp, step=stp2, method="eks0", which_prior="ioup3")
        means2 = sol.y.mean()
        sols2 = np.array([self.ivp.solution(t) for t in sol.t])
        err2 = np.amax(np.abs(sols2 - means2))
        exp_decay = (stp2 / stp1) ** 4
        diff = np.abs(exp_decay * err1 - err2) / np.abs(err2)
        self.assertLess(diff, 1.0)


class TestAdaptivityOnLotkaVolterraSmoother(unittest.TestCase):
    """
    Only test on "ekf0" with IBM(1) prior, since every other combination
    seems to dislike the adaptive scheme based on the whitened residual
    as an error estimate.
    """

    def setUp(self):
        """Setup odesolver and solve a scalar ode"""
        initrv = Dirac(20 * np.ones(2))
        self.ivp = ode.lotkavolterra([0.0, 0.5], initrv)
        self.tol = 1e-2

    def test_kf_ibm1_stdev(self):
        """
        Standard deviation at end point roughly equal to tolerance.
        """
        sol = probsolve_ivp(self.ivp, tol=self.tol, which_prior="ibm1", method="eks0")
        self.assertLess(np.sqrt(sol.y.cov()[-1, 0, 0]), 10 * self.tol)
        self.assertLess(0.1 * self.tol, np.sqrt(sol.y.cov()[-1, 0, 0]))

    def test_kf_ibm1(self):
        """
        Tests whether resulting steps are not evenly distributed.
        """
        sol = probsolve_ivp(self.ivp, tol=self.tol, which_prior="ibm1", method="eks0")
        steps = np.diff(sol.t)
        self.assertLess(np.amin(steps) / np.amax(steps), 0.8)


class TestLotkaVolterraOtherPriorsSmoother(unittest.TestCase):
    """
    We only test whether all the prior-filter-adaptivity combinations
    finish.
    """

    def setUp(self):
        """Setup odesolver and Lotka-Volterra IVP"""
        initdist = Dirac(20 * np.ones(2))
        self.ivp = ode.lotkavolterra([0.0, 0.5], initdist)
        self.tol = 1e-1
        self.step = 0.1

    def test_filter_ivp_ioup1_kf(self):
        probsolve_ivp(self.ivp, tol=self.tol, which_prior="ioup1", method="eks0")

    def test_filter_ivp_ioup2_ekf(self):
        probsolve_ivp(self.ivp, tol=self.tol, which_prior="ioup2", method="eks1")

    def test_filter_ivp_ioup3_ukf(self):
        """
        UKF requires some evaluation-variance to have a positive definite
        innovation matrix, apparently.
        """
        probsolve_ivp(
            self.ivp, tol=self.tol, evlvar=0.01, which_prior="ioup3", method="uks"
        )

    def test_filter_ivp_h_ioup1_ekf(self):
        probsolve_ivp(self.ivp, step=self.step, which_prior="ioup1", method="eks1")

    def test_filter_ivp_h_ioup2_ukf(self):
        """
        UKF requires some evaluation-variance to have a positive definite
        innovation matrix, apparently.
        """
        probsolve_ivp(
            self.ivp, step=self.step, evlvar=0.01, which_prior="ioup2", method="uks"
        )

    def test_filter_ivp_h_ioup3_kf(self):
        probsolve_ivp(self.ivp, step=self.step, which_prior="ioup3", method="eks0")

    def test_filter_ivp_mat32_kf(self):
        probsolve_ivp(self.ivp, tol=self.tol, which_prior="matern32", method="eks0")

    def test_filter_ivp_mat52_ekf(self):
        probsolve_ivp(self.ivp, tol=self.tol, which_prior="matern52", method="eks1")

    def test_filter_ivp_mat72_ukf(self):
        """
        UKF requires some evaluation-variance to have a positive definite
        innovation matrix, apparently.
        """
        probsolve_ivp(
            self.ivp, tol=self.tol, evlvar=0.01, which_prior="matern72", method="uks"
        )

    def test_filter_ivp_h_mat32_ekf(self):
        probsolve_ivp(self.ivp, step=self.step, which_prior="matern32", method="eks1")

    def test_filter_ivp_h_mat52_ukf(self):
        """
        UKF requires some evaluation-variance to have a positive definite
        innovation matrix, apparently.
        """
        probsolve_ivp(
            self.ivp, step=self.step, evlvar=0.01, which_prior="matern52", method="uks"
        )

    def test_filter_ivp_h_mat72_kf(self):
        probsolve_ivp(self.ivp, step=self.step, which_prior="matern72", method="eks0")


class TestPreconditioning(unittest.TestCase):
    """
    Solver with high order and small stepsize should work up to a point where
    step**order is below machine precision.
    """

    def setUp(self):
        initdist = Dirac(20 * np.ones(2))
        self.ivp = ode.lotkavolterra([0.0, 1e-4], initdist)
        self.step = 1e-5
        self.prior = "ibm3"

    def test_small_step_feasible(self):
        """
        With the 'old' preconditioner, this is impossible because step**(2*order + 1) is too small.
        With the 'new' preconditioner, the smallest value that appears in the solver code is step**order
        """
        probsolve_ivp(self.ivp, step=self.step, which_prior=self.prior, method="eks0")
