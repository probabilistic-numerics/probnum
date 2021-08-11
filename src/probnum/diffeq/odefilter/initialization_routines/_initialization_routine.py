"""Interface for ODE filter initialization."""

import abc

from probnum import problems, randprocs, randvars


class InitializationRoutine(abc.ABC):
    """Interface for initialization routines for a filtering-based ODE solver.

    One crucial factor for stable implementation of probabilistic ODE solvers is
    starting with a good approximation of the derivatives of the initial condition [1]_.
    (This is common in all Nordsieck-like ODE solvers.)
    For this reason, efficient methods of initialization need to be devised.
    All initialization routines in ProbNum implement the interface :class:`InitializationRoutine`.

    References
    ----------
    .. [1] Krämer, N. and Hennig, P., Stable implementation of probabilistic ODE solvers,
       *arXiv:2012.10106*, 2020.
    """

    def __init__(self, is_exact: bool, requires_jax: bool):
        self._is_exact = is_exact
        self._requires_jax = requires_jax

    @abc.abstractmethod
    def __call__(
        self,
        ivp: problems.InitialValueProblem,
        prior_process: randprocs.markov.MarkovProcess,
    ) -> randvars.RandomVariable:
        raise NotImplementedError

    @property
    def is_exact(self) -> bool:
        """Exactness of the computed initial values.

        Some initialization routines yield the exact initial derivatives, some others
        only yield approximations.
        """
        return self._is_exact

    @property
    def requires_jax(self) -> bool:
        """Whether the implementation of the routine relies on JAX."""
        return self._requires_jax
