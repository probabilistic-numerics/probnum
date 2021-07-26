"""Interface for tests of initialization routines of ODE filters."""

import abc

from probnum import randprocs


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
        prior_process = randprocs.markov.integrator.IntegratedWienerProcess(
            initarg=t0,
            num_derivatives=order,
            wiener_process_dimension=spatialdim,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
        return prior_process
