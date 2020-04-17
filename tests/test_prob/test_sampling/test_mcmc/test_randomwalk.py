"""
"""

import unittest

import numpy as np
import scipy.stats  # for true CDFs

from probnum.optim import objective
from probnum.prob.sampling.mcmc import randomwalk

VISUALISE = False

if VISUALISE is True:
    import matplotlib.pyplot as plt


class TestRandomWalkMH(unittest.TestCase):
    """
    We test whether
        * proposals in region with higher probability are always accepted
        * the qq-plot of a Gaussian and a Laplace is a straight line
    """

    def setUp(self):
        """
        """

        def obj(x):
            return x.T @ x

        self.logpdf = objective.Objective(obj)
        self.rw = randomwalk.RandomWalkMH(self.logpdf)

    def test_higher_prob_accepted(self):
        """
        Proposals in region with higher probability are always accepted!
        """
        currstate = self.logpdf.evaluate(np.random.rand(1))
        pwidth = np.random.rand()
        for __ in range(75):
            proposal, cfact = self.rw.generate_proposal(currstate, pwidth)
            __, acc = self.rw.accept_or_reject(currstate, proposal, cfact)
            if np.exp(-proposal.fx) >= np.exp(-currstate.fx):
                self.assertEqual(acc, True)

    def test_qq_gaussian(self):
        """
        Tests whether the qq plot of a 1d-Gaussian is a straight line.
        We use scipy.stats
        """

        def obj(x):  # adapter
            return -scipy.stats.norm.logpdf(x[0])

        mh_norm = randomwalk.RandomWalkMH(objective.Objective(obj))
        samples, __, __ = mh_norm.sample_nd(2500, np.zeros(1), 21.0)
        sortsamps, quants = _compute_qqvals(samples[:, 0],
                                            scipy.stats.norm.ppf)
        avgdiff = np.linalg.norm(sortsamps - quants) / len(sortsamps)
        self.assertLess(avgdiff, 1e-2)
        if VISUALISE is True:
            _visualise(samples, scipy.stats.norm.pdf, sortsamps, quants,
                       "Normal")

    def test_qq_laplace(self):
        """
        Tests whether the qq plot of a 1d-Gaussian is a straight line.
        We use scipy.stats
        """

        def obj(x):  # adapter
            return -scipy.stats.laplace.logpdf(x[0])

        mh_norm = randomwalk.RandomWalkMH(objective.Objective(obj))
        samples, __, __ = mh_norm.sample_nd(2500, np.zeros(1), 27.0)
        sortsamps, quants = _compute_qqvals(samples[:, 0],
                                            scipy.stats.laplace.ppf)
        avgdiff = np.linalg.norm(sortsamps - quants) / len(sortsamps)
        self.assertLess(avgdiff, 1e-2)
        if VISUALISE is True:
            _visualise(samples, scipy.stats.laplace.pdf, sortsamps, quants,
                       "Laplace")

    def test_qq_invgamma(self):
        """
        Tests whether the qq plot of a 1d-Gaussian is a straight line.
        We use scipy.stats
        """

        def obj(x):  # adapter
            return -scipy.stats.invgamma.logpdf(x[0], a=4.07)

        mh_norm = randomwalk.RandomWalkMH(objective.Objective(obj))
        samples, __, __ = mh_norm.sample_nd(5000, np.ones(1), 0.45)
        sortsamps, quants = _compute_qqvals(samples[:, 0],
                                            scipy.stats.invgamma.ppf, a=4.07)
        avgdiff = np.linalg.norm(sortsamps - quants) / len(sortsamps)
        self.assertLess(avgdiff, 1e-2)
        if VISUALISE is True:
            _visualise(samples, scipy.stats.invgamma.pdf, sortsamps, quants,
                       "Inv.-Gamma", a=4.07)


def _compute_qqvals(samples, ppf, **kwargs):
    """
    """
    sortedsamps = np.sort(samples)
    enums = np.arange(1, 1 + len(sortedsamps)) / (len(sortedsamps) + 1)
    quantiles = ppf(enums, **kwargs)
    return sortedsamps, quantiles


def _visualise(samples, pdf, sortsamps, quants, title, **kwargs):
    """
    """
    _show_histplot(samples, pdf, title, **kwargs)
    _show_qqplot(sortsamps, quants, title)


def _show_histplot(samples, pdf, title, **kwargs):
    """
    """
    xvals = np.linspace(1.1 * np.amin(samples), 1.1 * np.amax(samples))
    yvals = pdf(xvals, **kwargs)
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
