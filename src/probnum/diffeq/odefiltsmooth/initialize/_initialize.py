"""Interface for ODE filter initialization."""

import abc

from probnum import problems, randprocs, randvars


class InitializationRoutine(abc.ABC):
    """Interface for initialization routines for a filtering-based ODE solver."""

    @abc.abstractmethod
    def __call__(
        self, ivp: problems.InitialValueProblem, prior_process: randprocs.MarkovProcess
    ) -> randvars.RandomVariable:
        pass
