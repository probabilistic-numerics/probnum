"""Interface for tests of initialization routines of ODE filters."""

import abc

import numpy as np

from probnum import randprocs, randvars, statespace


class InterfaceInitializationRoutineTest(abc.ABC):
    """Interface for tests of initialization routines of ODE filters."""

    @abc.abstractmethod
    def test_call(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_is_exact(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_requires_jax(self):
        raise NotImplementedError

    def _construct_prior_process(self, order, spatialdim, t0):
        """Construct a prior process of appropriate size."""
        prior_transition = statespace.IBM(
            ordint=order,
            spatialdim=spatialdim,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
        initrv = randvars.Normal(
            np.zeros(prior_transition.dimension), np.eye(prior_transition.dimension)
        )
        prior_process = randprocs.MarkovProcess(
            transition=prior_transition, initrv=initrv, initarg=t0
        )
        return prior_process
