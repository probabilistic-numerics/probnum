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

from probnum.prob.sampling.mcmc import metropolishastings


class RandomWalkMH(metropolishastings.MetropolisHastings):
    """
    Optimal acceptance ratio seems to be 0.234

    http://probability.ca/jeff/ftpdir/mylene2.pdf

    https://arxiv.org/pdf/1206.1901.pdf (p. 29)
    """

    good_acc_ratio = [.15, .30]

    def generate_proposal(self, currstate, pwidth, *args, **kwargs):
        """
        """
        newloc = self._sample_randomwalk(currstate.x, pwidth)
        proposal = self.logdens.evaluate(newloc)
        corrfact = 0
        return proposal, corrfact

    def _sample_randomwalk(self, mean, var):
        """
        """
        return mean + np.sqrt(var) * np.random.randn(len(mean))
