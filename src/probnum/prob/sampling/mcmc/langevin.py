"""
sampling.py
We sample Metropolis-Hastings:
    * Random walk proposals
    * Langevin proposals
    * Langevin proposals with preconditioning
    * Hamiltonian MC
    * Hamiltonian MC with preconditioning
NOTE:
    The functionality of this module is restricted to log-densities,
    i.e. densities of the form p(s) = exp(-E(s)). We work with E(s) only.
    The reason is that in Bayesian inference, evaluations of exp(-E(s))
    are too instable in a numerical sense. 
"""

import numpy as np

from probnum.prob.sampling.mcmc.metropolishastings import MetropolisHastings


class MetropolisAdjustedLangevinAlgorithm(MetropolisHastings):
    """
    Optimal acceptance ratio seems to be 0.574

    https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00123
    """

    good_acc_ratio = [0.5, 0.65]

    def generate_proposal(self, currstate, pwidth, *pars, **namedpars):
        """
        """
        newloc = self._sample_langevin(currstate, pwidth)
        proposal = self.logdens.evaluate(newloc)
        corrfact = self._compute_corrfact_langevin(currstate, proposal, pwidth)
        return proposal, corrfact

    def _sample_langevin(self, currstate, pwidth):
        """
        """
        noise = np.random.randn(len(currstate.x))
        return currstate.x - pwidth * currstate.dfx + np.sqrt(
            2 * pwidth) * noise

    def _compute_corrfact_langevin(self, currstate, proposal, pwidth):
        """
        """
        lognomin = self._kernel_langevin(currstate, proposal, pwidth)
        logdenom = self._kernel_langevin(proposal, currstate, pwidth)
        return (lognomin - logdenom)

    def _kernel_langevin(self, state1, state2, pwidth):
        """
        """
        state2_dyn = state2.x - pwidth * state2.dfx
        dist = np.linalg.norm(state1.x - state2_dyn) ** 2
        return 0.5 * dist / (2 * pwidth)


class PreconditionedMetropolisAdjustedLangevinAlgorithm(MetropolisHastings):
    """
    Preconditioning with (inverse) Hessian.

    Optimal acceptance ratio seems to be 0.574

    https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00123
    """

    good_acc_ratio = [0.5, 0.65]

    def generate_proposal(self, currstate, pwidth, *pars, **namedpars):
        """
        """
        newloc = self.sample_langevin(currstate, pwidth)
        proposal = self.logdens.evaluate(newloc)
        corrfact = self.compute_corrfact_langevin(currstate, proposal, pwidth)
        return proposal, corrfact

    def sample_langevin(self, currstate, pwidth):
        """
        """
        noise = np.random.multivariate_normal(np.zeros(len(currstate.dfx)),
                                              np.linalg.inv(currstate.ddfx))
        prec_dyn = np.linalg.solve(currstate.ddfx, currstate.dfx)
        return currstate.x - pwidth * prec_dyn + np.sqrt(2 * pwidth) * noise

    def compute_corrfact_langevin(self, currstate, proposal, pwidth):
        """
        """
        lognomin = self.kernel_langevin(currstate, proposal, pwidth)
        logdenom = self.kernel_langevin(proposal, currstate, pwidth)
        return lognomin - logdenom

    def kernel_langevin(self, state1, state2, pwidth):
        """
        """
        prec_dyn = np.linalg.solve(state2.ddfx, state2.dfx)
        state2_dyn = state2.x - pwidth * prec_dyn
        difference = state1.x - state2_dyn
        return 0.5 * difference.dot(
            np.linalg.solve(state2.ddfx, difference)) / (2 * pwidth)
