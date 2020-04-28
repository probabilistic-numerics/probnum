"""
Metropolis-Hastings sampling in N dimensions.

We sample Metropolis-Hastings:
    * Random walk proposals
    * Langevin proposals
    * Langevin proposals with preconditioning
    * Hamiltonian MC
    * Hamiltonian MC with preconditioning

Note
----
The functionality of this module is restricted to log-densities,
i.e. densities of the form p(s) = exp(-E(s)) with a support on the entire
real line. We work with E(s) = -log(p(s)) only.
The reason is that in Bayesian inference, evaluations of exp(-E(s))
can too instable in a numerical sense.
"""

from abc import ABC, abstractmethod

import numpy as np

from probnum.optim import objective
from probnum import utils


class MetropolisHastings(ABC):
    """
    Abstract Metropolis-Hastings class. Contains everything but the
    proposal kernels.
    """

    good_acc_ratio = None

    def __init__(self, logdens):
        """
        """
        if not isinstance(logdens, objective.Objective):
            raise ValueError("Please initialise with Objective "
                             "(function) instance.")
        self.logdens = logdens

    def __repr__(self):
        """
        """
        return "MetropolisHastings() object"

    def sample_nd(self, nsamps, init_state, pwidth, **kwargs):
        """
        """
        self._assert_inputs_compatible(init_state)
        states, logprobs = np.zeros((nsamps, len(init_state))), np.zeros(
            nsamps)
        currstate = self.logdens.evaluate(init_state)
        states[0], logprobs[0], accepted = currstate.x, currstate.fx, 1
        for idx in range(1, nsamps):
            proposal, corrfact = self.generate_proposal(currstate, pwidth,
                                                        **kwargs)
            currstate, is_accept = self.accept_or_reject(currstate, proposal,
                                                         corrfact)
            states[idx], logprobs[idx] = currstate.x, currstate.fx
            accepted = accepted + int(is_accept)
        acc_ratio = accepted / nsamps
        self._check_acc_ratio(acc_ratio)
        return states, logprobs, acc_ratio

    def _assert_inputs_compatible(self, init_state):
        """
        """
        utils.assert_is_1d_ndarray(init_state)
        utils.assert_evaluates_to_scalar(self.logdens.objective, init_state)

    def _check_acc_ratio(self, ratio):
        """
        Checks---if available---if acceptance ratio is within given
        interval of "good values". If not, it prints a warning.
        """
        if self.good_acc_ratio is not None:
            if ratio < self.good_acc_ratio[0] or ratio > self.good_acc_ratio[
                1]:
                print("!!! Careful: acc_ratio is not near optimality")
                print("!!! Desired: %s, got: %s" % (
                    str(self.good_acc_ratio), str(ratio)))

    def accept_or_reject(self, currstate, proposal, corrfact):
        """
        """
        logaccprob = self._get_logaccprob(currstate, proposal, corrfact)
        ran = np.random.rand()
        if logaccprob < 0 or logaccprob < -np.log(ran):
            state = proposal
            is_accept = True
        else:
            state = currstate
            is_accept = False
        return state, is_accept

    def _get_logaccprob(self, currstate, proposal, corrfact):
        """
        Returns NEGATIVE log acceptance probability, i.e.
            corrected proposal - corrected currstate
        """
        return corrfact + (proposal.fx - currstate.fx)

    @abstractmethod
    def generate_proposal(self, currstate, pwidth, **kwargs):
        """
        """
        pass
