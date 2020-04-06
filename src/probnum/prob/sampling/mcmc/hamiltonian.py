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


class HMC(metropolishastings.MetropolisHastings):
    """
    Optimal acceptance ratio seems to be 0.651 (at least in high dim.)

    http://www.people.fas.harvard.edu/~pillai/Publications_files/hmcbookchapter.pdf

    https://arxiv.org/pdf/1206.1901.pdf (p. 29)
    """

    good_acc_ratio = [0.575, 0.725]

    def __init__(self, logdens, nsteps):
        """
        """
        metropolishastings.MetropolisHastings.__init__(self, logdens)
        self.nsteps = nsteps

    def generate_proposal(self, currstate, pwidth, *args, **kwargs):
        """
        pwidth is used as stepsize for self.nsteps leapfrog steps.
        The correction factor is the quotient of the hamiltonian terms.
        """
        momentum = np.random.randn(len(currstate.x))
        momentum_new, proposal = self._leapfrog_dynamics(momentum, currstate,
                                                         pwidth)
        corrfact = self._get_corrfact(momentum, momentum_new)
        return proposal, corrfact

    def _leapfrog_dynamics(self, momentum, currstate, pwidth):
        """
        """
        proposal = currstate
        for __ in range(self.nsteps):
            momentum, proposal = self._compute_next_lfstep(momentum, proposal,
                                                           pwidth)
        return momentum, proposal

    def _compute_next_lfstep(self, momentum, proposal, pwidth):
        """
        """
        momentum = momentum - 0.5 * pwidth * proposal.dfx
        pstate = proposal.x + pwidth * momentum
        proposal = self.logdens.evaluate(pstate)
        momentum = momentum - 0.5 * pwidth * proposal.dfx
        return momentum, proposal

    def _get_corrfact(self, mom, mom_new):
        """
        """
        return 0.5 * (mom_new.T @ mom_new - mom.T @ mom)


class PHMC(metropolishastings.MetropolisHastings):
    """
    In fact, the true name would be either
        * Riemannian-Gaussian HMC: if the preconditioner depends on the state
        * Euclidean-Gaussian HMC: if the preconditioner is constant
        
    See Girolami and Calderhead, 2011; Betancourt, 2018.

    Optimal acceptance ratio seems to be 0.651 (at least in high dim.)

    http://www.people.fas.harvard.edu/~pillai/Publications_files/hmcbookchapter.pdf

    https://arxiv.org/pdf/1206.1901.pdf (p. 29)
    """

    good_acc_ratio = [0.575, 0.725]

    def __init__(self, logdens, nsteps):
        """
        evalprecond returns M (and not M^{-1}) as used in Cald&Gir.
        M is the Hessian
        """
        metropolishastings.MetropolisHastings.__init__(self, logdens)
        self.nsteps = nsteps

    def generate_proposal(self, currstate, pwidth, *args, **kwargs):
        """
        pwidth is used as stepsize for self.nsteps leapfrog steps.
        The correction factor is the quotient of the hamiltonian terms.

        This is NOT a duplicate from HamiltonianMC.generate_proposal(...)!
        """
        momentum = np.random.multivariate_normal(np.zeros(len(currstate.x)),
                                                 currstate.ddfx)
        momentum_new, proposal = self.leapfrog_dynamics(momentum, currstate,
                                                        pwidth)
        corrfact = self.get_corrfact(momentum, momentum_new, currstate,
                                     proposal)
        return proposal, corrfact

    def leapfrog_dynamics(self, momentum, currstate, pwidth):
        """
        This is a duplicate from HamiltonianMC.generate_proposal(...).
        """
        proposal = currstate
        for idx in range(self.nsteps):
            momentum, proposal = self.compute_next_lfstep(momentum, proposal,
                                                          pwidth)
        return momentum, proposal

    def compute_next_lfstep(self, momentum, proposal, pwidth):
        """
        This is NOT a duplicate from HamiltonianMC.generate_proposal(...)!
        """
        momentum = momentum - 0.5 * pwidth * proposal.dfx
        pstate = proposal.x + pwidth * np.linalg.solve(proposal.ddfx, momentum)
        proposal = self.logdens.evaluate(pstate)
        momentum = momentum - 0.5 * pwidth * proposal.dfx
        return momentum, proposal

    def get_corrfact(self, mom, mom_new, currstate, proposal):
        """
        This is NOT a duplicate from HamiltonianMC.generate_proposal(...)!
        """
        return 0.5 * (mom_new.T @ np.linalg.solve(proposal.ddfx, mom_new) \
                      + np.log(np.linalg.det(proposal.ddfx)) \
                      - mom.T @ np.linalg.solve(currstate.ddfx, mom)
                      - np.log(np.linalg.det(currstate.ddfx)))
