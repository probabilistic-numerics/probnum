"""
We test on two test-problems:
    * logistic ODE (because it has a closed form sol.)
        -> compare error to standard deviation of solvers
        -> make sure error converges to zero (even with rate q?)
        -> Check if iterates match the closed-form solutions in
        Schober et al.
    * Lotka-Volterra (because it provides meaningful uncertainty estimates,
    if e.g. EKF-based ODE filter is implemented correctly)
        -> Uncertainty is larger around peaks than around valleys.

    * OPTIONAL: Benchmark tests: Solutions to FHN (easy) and Res2Bod (hard)
    are expected to cross a certain benchmark point

Todo
----
Adaptive step size tests and some more cleverness in the current tests...
"""

import unittest

import numpy as np

from probnum.diffeq import steprule, ode
from probnum.diffeq.odefilter import prior, odefilter, ivptofilter
from probnum.prob import RandomVariable
from probnum.prob.distributions import Dirac

VISUALISE = True

if VISUALISE is True:
    import matplotlib.pyplot as plt


class TestErrorLogisticQ1(unittest.TestCase):
    """
    Test whether the mean and covariance output of
    applying the solver to a SCALAR ODE coincide
    with the ones in the proof of Proposition 1 in [1]; see p. 108.
    """

    def setUp(self):
        """Setup odesolver and solve a scalar ode"""
        initdist = RandomVariable(distribution=Dirac(0.1 * np.ones(1)))
        self.ivp = ode.logistic([0.0, 1.5], initdist)

    def test_visualise(self):
        """
        """
        maxerr, maxstd = [], []
        stps = np.array([0.5 ** i for i in range(3, 10)])
        for step in stps:
            ms, cs, ts = odefilter.filter_ivp_h(self.ivp, step, which_prior="ibm1")
            means = ms[:, 0]
            sols = np.array([self.ivp.solution(t) for t in ts])
            maxerr.append(np.amax(np.abs(sols[:, 0] - means)))
            maxstd.append(np.amax(np.sqrt(np.abs(cs[:, 0, 0]))))

        if VISUALISE is True:
            plt.loglog(stps, maxerr, "x-", label="Error")
            plt.loglog(stps, stps ** 2, "--", label="O(h^2)", alpha=0.5)
            plt.xlabel("Stepsize h")
            plt.ylabel("Error")
            plt.title("Max. Pointwise error: IBM(1)")
            plt.grid()
            plt.legend()
            plt.show()


class TestErrorLogisticQ2(unittest.TestCase):
    """
    Test whether the mean and covariance output of
    applying the solver to a SCALAR ODE coincide
    with the ones in the proof of Proposition 1 in [1]; see p. 108.
    """

    def setUp(self):
        """Setup odesolver and solve a scalar ode"""
        initdist = RandomVariable(distribution=Dirac(0.1 * np.ones(1)))
        self.ivp = ode.logistic([0.0, 1.5], initdist)

    def test_visualise(self):
        """
        """
        maxerr, maxstd = [], []
        stps = np.array([0.5 ** i for i in range(3, 10)])
        for step in stps:
            ms, cs, ts = odefilter.filter_ivp_h(self.ivp, step, which_prior="ibm2")
            means = ms[:, 0]
            sols = np.array([self.ivp.solution(t) for t in ts])
            maxerr.append(np.amax(np.abs(sols[:, 0] - means)))
            maxstd.append(np.amax(np.sqrt(np.abs(cs[:, 0, 0]))))

        if VISUALISE is True:
            plt.loglog(stps, maxerr, "x-", label="Error")
            plt.loglog(stps, stps ** 3, "--", label="O(h^3)", alpha=0.5)
            plt.xlabel("Stepsize h")
            plt.ylabel("Error")
            plt.title("Max. Pointwise error: IBM(2)")
            plt.grid()
            plt.legend()
            plt.show()


class TestErrorLogisticQ3(unittest.TestCase):
    """
    Test whether the mean and covariance output of
    applying the solver to a SCALAR ODE coincide
    with the ones in the proof of Proposition 1 in [1]; see p. 108.
    """

    def setUp(self):
        """Setup odesolver and solve a scalar ode"""
        initdist = RandomVariable(distribution=Dirac(0.1 * np.ones(1)))
        self.ivp = ode.logistic([0.0, 1.5], initdist)

    def test_visualise(self):
        """
        """
        maxerr, maxstd = [], []
        stps = np.array([0.5 ** i for i in range(3, 10)])
        for step in stps:
            ms, cs, ts = odefilter.filter_ivp_h(self.ivp, step, which_prior="ibm3")
            means = ms[:, 0]
            sols = np.array([self.ivp.solution(t) for t in ts])
            maxerr.append(np.amax(np.abs(sols[:, 0] - means)))
            maxstd.append(np.amax(np.sqrt(np.abs(cs[:, 0, 0]))))

        if VISUALISE is True:
            plt.loglog(stps, maxerr, "x-", label="Error")
            plt.loglog(stps, stps ** 4, "--", label="O(h^4)", alpha=0.5)
            plt.xlabel("Stepsize h")
            plt.ylabel("Error")
            plt.title("Max. Pointwise error: IBM(3)")
            plt.grid()
            plt.legend()
            plt.show()


class TestErrorLogisticQ3_IOUP(unittest.TestCase):
    """
    Test whether the mean and covariance output of
    applying the solver to a SCALAR ODE coincide
    with the ones in the proof of Proposition 1 in [1]; see p. 108.
    """

    def setUp(self):
        """Setup odesolver and solve a scalar ode"""
        initdist = RandomVariable(distribution=Dirac(0.1 * np.ones(1)))
        self.ivp = ode.logistic([0.0, 1.5], initdist)

    def test_visualise(self):
        """
        """
        maxerr, maxstd = [], []
        stps = np.array([0.5 ** i for i in range(3, 9)])
        for step in stps:
            ms, cs, ts = odefilter.filter_ivp_h(self.ivp, step, which_prior="ioup3")
            means = ms[:, 0]
            sols = np.array([self.ivp.solution(t) for t in ts])
            maxerr.append(np.amax(np.abs(sols[:, 0] - means)))
            maxstd.append(np.amax(np.sqrt(np.abs(cs[:, 0, 0]))))

        if VISUALISE is True:
            plt.loglog(stps, maxerr, "x-", label="Error")
            plt.loglog(stps, stps ** 4, "--", label="O(h^4)", alpha=0.5)
            plt.xlabel("Stepsize h")
            plt.ylabel("Error")
            plt.title("Max. Pointwise error: IOUP(3)")
            plt.grid()
            plt.legend()
            plt.show()

#
#
#
#
#
#
#
#
#
#
#
#
#
# class TestErrorLogistic2(unittest.TestCase):
#     """
#     Test whether the mean and covariance output of
#     applying the solver to a SCALAR ODE coincide
#     with the ones in the proof of Proposition 1 in [1]; see p. 108.
#     """
#
#     def setUp(self):
#         """Setup odesolver and solve a scalar ode"""
#         ibm = prior.IBM(1, 1, 1.0)
#         stepsize = 0.1
#         initdist = dirac.Dirac(0.1 * np.ones(1))
#         self.ivp = ode.logistic([0.0, 2.0], initdist)
#         kfilt = ivp_to_kf(self.ivp, ibm, 0.0)
#         stprl = steprule.ConstantSteps(stepsize)
#         ofi = odefilter.GaussianODEFilter(self.ivp, kfilt, stprl)
#         self.means, self.covars, self.times = ofi.solve(stepsize)
#
#     def test_visualise(self):
#         """
#         """
#         means = self.means[:, 0]
#         sols = np.array([self.ivp.solution(t) for t in self.times])
#         avgerr = np.mean(np.abs(sols[:, 0] - means))
#         avgstd = np.mean(np.sqrt(self.covars[:, 0, 0]))
#         if VISUALISE is True:
#             plt.plot(self.times, means)
#             plt.title(
#                 "Logistic; avgerr: %.1e; avgstd: %.1e" % (avgerr, avgstd))
#             plt.plot(self.times, sols, color="black")
#             plt.fill_between(self.times, means, alpha=0.25)
#             plt.show()
#
#     #
# def test_first_few_iterations(self):
#     """
#     Test whether first few means and covariances coincide with Prop. 1.
#     """
#     self.check_mean_t0()
#     self.check_stdevs_t0()
#     self.check_mean_t1()
#     self.check_stdevs_t1()
#
# def check_mean_t0(self):
#     """Expect: m(t0) = (y0, z0) where z0=f(y0)"""
#     y0 = self.ode.initval
#     z0 = self.ode.modeval(t=0.0, x=y0)
#     mean_at_t0 = self.means[0][0]
#     self.assertEqual(mean_at_t0[0], y0)
#     self.assertEqual(mean_at_t0[1], z0)
#
# def check_stdevs_t0(self):
#     """Expect: C(t0) = 0, hence stdevs equal to zero"""
#     stdev_at_t0 = self.stdevs[0][0]
#     self.assertEqual(stdev_at_t0[0], 0.0)
#     self.assertEqual(stdev_at_t0[1], 0.0)
#
# def check_mean_t1(self):
#     """Expect: m(t0) = (y0 + h/2*(z0 + z1), z1)"""
#     y0 = self.ode.initval
#     z0 = self.ode.modeval(t=0.0, x=y0)
#     z1 = self.ode.modeval(t=0.0, x=(y0 + self.h * z0))
#     mean_at_t1 = self.means[1][0]
#     self.assertEqual(mean_at_t1[0], y0 + 0.5 * self.h * (z0 + z1))
#     self.assertEqual(mean_at_t1[1], z1)
#
# def check_stdevs_t1(self):
#     """Expect: C(t1) = (sigma**2 h**3/12, 0; 0, 0)"""
#     stdev_at_t1 = self.stdevs[1][0]
#     sigmasquared = self.solver.filt.ssm.diffconst       # digging deep
#     self.assertAlmostEqual(stdev_at_t1[0],
#                            np.sqrt(self.h**3 * sigmasquared / 12.0),
#                            places=12)
#     self.assertEqual(stdev_at_t1[1], 0.0)
