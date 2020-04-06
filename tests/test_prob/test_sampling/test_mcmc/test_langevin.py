"""
DONT FORGET TO TEST THE PRECONDITIONED VERSIONS!!!
"""

import unittest

import numpy as np
import scipy.stats  # for true CDFs

from probnum.optim import objective
from probnum.prob.sampling.mcmc import langevin, hamiltonian

VISUALISE = False

if VISUALISE is True:
    import matplotlib.pyplot as plt


class TestMALA(unittest.TestCase):
    """
    We test whether
        * proposals in region with higher probability are always accepted
        * the qq-plot of a Gaussian and a Laplace is a straight line
        * the qq-plot of an exponential distribution to capture the corner
          case of when a PDF evaluates to zero in some ranges
    """

    def setUp(self):
        """
        """

        def obj(x):
            return x.T @ x / 2.0

        def der(x):
            return x

        self.logpdf = objective.Objective(obj, der)
        self.mala = langevin.MetropolisAdjustedLangevinAlgorithm(self.logpdf)

    def test_higher_prob_accepted(self):
        """
        Proposals in region with higher probability are always accepted!
        """
        currstate = self.logpdf.evaluate(np.random.rand(1))
        pwidth = np.random.rand()
        for __ in range(5000):
            proposal, cfact = self.mala.generate_proposal(currstate, pwidth)
            __, acc = self.mala.accept_or_reject(currstate, proposal, cfact)
            if np.exp(-proposal.fx) > np.exp(-currstate.fx):
                self.assertEqual(acc, True)

    def test_qq_gaussian(self):
        """
        """

        def obj(x):  # adapter
            return x.T @ x / 2.0

        def der(x):
            return x

        mh_norm = langevin.MetropolisAdjustedLangevinAlgorithm(objective.Objective(obj, der))
        samples, __, __ = mh_norm.sample_nd(2500, np.zeros(1), 1.75)
        sortsamps, quants = _compute_qqvals(samples[:, 0],
                                            scipy.stats.norm.ppf)
        avgdiff = np.linalg.norm(sortsamps - quants) / len(sortsamps)
        if VISUALISE is True:
            _visualise(samples, scipy.stats.norm.pdf, sortsamps, quants,
                       "Normal")
        self.assertLess(avgdiff, 1e-2)

    def test_qq_laplace(self):
        """
        """

        def obj(x):  # adapter
            return np.linalg.norm(x)

        def der(x):
            return np.sign(x)

        mh_norm = langevin.MetropolisAdjustedLangevinAlgorithm(objective.Objective(obj, der))
        samples, __, __ = mh_norm.sample_nd(2500, np.ones(1), 2.25)
        sortsamps, quants = _compute_qqvals(samples[:, 0],
                                            scipy.stats.laplace.ppf)
        avgdiff = np.linalg.norm(sortsamps - quants) / len(sortsamps)
        if VISUALISE is True:
            _visualise(samples, scipy.stats.laplace.pdf, sortsamps, quants,
                       "Laplace")
        self.assertLess(avgdiff, 1e-2)

    def test_qq_expon(self):
        """
        """

        def obj(x):  # adapter
            if x[0] >= 0:
                return x[0]
            else:
                return np.inf  # make negative values impossible

        def der(x):
            if x[0] >= 0:
                return 1
            else:
                return np.inf

        mh_norm = langevin.MetropolisAdjustedLangevinAlgorithm(objective.Objective(obj, der))
        samples, __, __ = mh_norm.sample_nd(2500, np.ones(1), 0.75)
        sortsamps, quants = _compute_qqvals(samples[:, 0],
                                            scipy.stats.expon.ppf)
        avgdiff = np.linalg.norm(sortsamps - quants) / len(sortsamps)
        if VISUALISE is True:
            _visualise(samples, scipy.stats.expon.pdf, sortsamps, quants,
                       "Exponential")
        self.assertLess(avgdiff, 1e-2)


class TestPMALA(unittest.TestCase):
    """
    We only do the Gaussians because Laplace and Exponential do not have
    well-defined, positive definite Hessians of the negative log-likelihood.
    """

    def setUp(self):
        """
        """

        def obj(x):
            return x.T @ x / 2.0

        def der(x):
            return x

        def hess(x):
            return np.ones((len(x), len(x)))

        self.logpdf = objective.Objective(obj, der, hess)
        self.pmala = langevin.PreconditionedMetropolisAdjustedLangevinAlgorithm(self.logpdf)

    def test_higher_prob_accepted(self):
        """
        Proposals in region with higher probability are always accepted!
        """
        currstate = self.logpdf.evaluate(np.random.rand(1))
        pwidth = np.random.rand()
        for __ in range(5000):
            proposal, cfact = self.pmala.generate_proposal(currstate, pwidth)
            __, acc = self.pmala.accept_or_reject(currstate, proposal, cfact)
            if np.exp(-proposal.fx) > np.exp(-currstate.fx):
                self.assertEqual(acc, True)

    def test_qq_gaussian(self):
        """
        """

        samples, __, __ = self.pmala.sample_nd(1250, np.zeros(1), 1.725)
        sortsamps, quants = _compute_qqvals(samples[:, 0],
                                            scipy.stats.norm.ppf)
        avgdiff = np.linalg.norm(sortsamps - quants) / len(sortsamps)
        if VISUALISE is True:
            _visualise(samples, scipy.stats.norm.pdf, sortsamps, quants,
                       "Normal")
        self.assertLess(avgdiff, 1e-2)

    def test_coinced_hamiltonian(self):
        """
        We use the exponential because it seems to be the "hardest".
        """

        def obj(x):
            return x.T @ x / 2.0

        def der(x):
            return x

        def hess(x):
            return np.ones((len(x), len(x)))

        # deriv = autodiff.asautodiff(der, hess)
        logpdf = objective.Objective(obj, der, hess)

        pham = hamiltonian.PreconditionedHamiltonianMonteCarlo(logpdf, nsteps=1)
        plang = langevin.PreconditionedMetropolisAdjustedLangevinAlgorithm(logpdf)

        np.random.seed(1)
        samples_ham, __, __ = pham.sample_nd(250, 0.5 * np.ones(1), 1.8)
        np.random.seed(1)
        samples_lang, __, __ = plang.sample_nd(250, 0.5 * np.ones(1),
                                               1.8 ** 2 / 2.0)
        avgdiff = np.linalg.norm(samples_ham - samples_lang) / len(
            samples_lang)
        self.assertLess(avgdiff, 1e-14)


def _compute_qqvals(samples, ppf, *args, **kwargs):
    """
    """
    sortedsamps = np.sort(samples)
    enums = np.arange(1, 1 + len(sortedsamps)) / (len(sortedsamps) + 1)
    quantiles = ppf(enums, *args, **kwargs)
    return sortedsamps, quantiles


def _visualise(samples, pdf, sortsamps, quants, title, *args, **kwargs):
    """
    """
    _show_histplot(samples, pdf, title, *args, **kwargs)
    _show_qqplot(sortsamps, quants, title)


def _show_histplot(samples, pdf, title, *args, **kwargs):
    """
    """
    xvals = np.linspace(1.1 * np.amin(samples), 1.1 * np.amax(samples))
    yvals = pdf(xvals, *args, **kwargs)
    plt.hist(samples, density=True, bins=75, alpha=0.5)
    plt.plot(xvals, yvals)
    plt.title(title)
    plt.show()


def _show_qqplot(sortsamps, quants, title):
    """
    """
    minval = np.amin(sortsamps) - 0.25
    maxval = np.amax(sortsamps) + 0.25
    xvals = np.linspace(minval, maxval)
    plt.plot(sortsamps, quants, "x")
    plt.plot(xvals, xvals, alpha=0.5)
    plt.xlabel("Empirical quantiles")
    plt.ylabel("True quantiles")
    plt.xlim((minval, maxval))
    plt.ylim((minval, maxval))
    plt.title(title)
    plt.show()
