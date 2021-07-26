"""Interface for information operators."""

import abc
from typing import Callable, Optional

import numpy as np

from probnum import problems, randprocs
from probnum.typing import FloatArgType, IntArgType

__all__ = ["InformationOperator", "ODEInformationOperator"]


class InformationOperator(abc.ABC):
    r"""Information operators used in probabilistic ODE solvers.

    ODE solver-related information operators gather information about whether a state or function solves an ODE.
    More specifically, an information operator maps a sample from the prior distribution
    **that is also an ODE solution** to the zero function.

    Consider the following example. For an ODE

    .. math:: \dot y(t) - f(t, y(t)) = 0,

    and a :math:`\nu` times integrated Wiener process prior,
    the information operator maps

    .. math:: \mathcal{Z}: [t, (Y_0, Y_1, ..., Y_\nu)] \mapsto Y_1(t) - f(t, Y_0(t)).

    (Recall that :math:`Y_j` models the `j` th derivative of `Y_0` for given prior.)
    If :math:`Y_0` solves the ODE, :math:`\mathcal{Z}(Y)(t)` is zero for all :math:`t`.

    Information operators are used to condition prior distributions on solving a numerical problem.
    This happens by conditioning the prior distribution :math:`Y` on :math:`\mathcal{Z}(Y)(t_n)=0`
    on time-points :math:`t_1, ..., t_n, ..., t_N` (:math:`N` is usually large).
    Therefore, they are one important component in a probabilistic ODE solver.
    """

    def __init__(self, input_dim: IntArgType, output_dim: IntArgType):
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abc.abstractmethod
    def __call__(self, t: FloatArgType, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, t: FloatArgType, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def as_transition(
        self,
        measurement_cov_fun: Optional[Callable[[FloatArgType], np.ndarray]] = None,
        measurement_cov_cholesky_fun: Optional[
            Callable[[FloatArgType], np.ndarray]
        ] = None,
    ):

        if measurement_cov_fun is None:
            if measurement_cov_cholesky_fun is not None:
                raise ValueError(
                    "If a Cholesky function is provided, a covariance function must be provided as well."
                )
            return randprocs.markov.discrete.DiscreteGaussian.from_callable(
                state_trans_fun=self.__call__,
                jacob_state_trans_fun=self.jacobian,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
            )

        return randprocs.markov.discrete.DiscreteGaussian(
            state_trans_fun=self.__call__,
            jacob_state_trans_fun=self.jacobian,
            proc_noise_cov_mat_fun=measurement_cov_fun,
            proc_noise_cov_cholesky_fun=measurement_cov_cholesky_fun,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )


class ODEInformationOperator(InformationOperator):
    """Information operators that depend on an ODE function.

    Other than :class:`InformationOperator`s, :class:`ODEInformationOperators` depend explicitly on an
    :class:`InitialValueProblem`. Not all information operators that are used in ODE solvers do.
    """

    def __init__(self, input_dim: IntArgType, output_dim: IntArgType):
        super().__init__(input_dim=input_dim, output_dim=output_dim)

        # Initialized once the ODE can be seen
        self.ode = None

    def incorporate_ode(self, ode: problems.InitialValueProblem):
        """Incorporate the ODE into the operator."""
        if self.ode_has_been_incorporated:
            raise ValueError("An ODE has been incorporated already.")
        else:
            self.ode = ode

    @property
    def ode_has_been_incorporated(self) -> bool:
        return self.ode is not None

    def as_transition(
        self,
        measurement_cov_fun: Optional[Callable[[FloatArgType], np.ndarray]] = None,
        measurement_cov_cholesky_fun: Optional[
            Callable[[FloatArgType], np.ndarray]
        ] = None,
    ):
        if not self.ode_has_been_incorporated:
            raise ValueError("An ODE has not been incorporated yet.")
        return super().as_transition(
            measurement_cov_fun=measurement_cov_fun,
            measurement_cov_cholesky_fun=measurement_cov_cholesky_fun,
        )
