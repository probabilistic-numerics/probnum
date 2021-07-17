"""Interface for information operators."""

import abc

from probnum import filtsmooth, statespace

__all__ = ["InformationOperator", "ODEInformationOperator"]


class InformationOperator(abc.ABC):
    r"""ODE information operators used in probabilistic ODE solvers.

    ODE information operators gather information about whether a state or function solves an ODE.
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

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abc.abstractmethod
    def __call__(self, t, x):
        raise NotImplementedError

    def jacobian(self, t, x):
        raise NotImplementedError

    def as_transition(
        self, measurement_cov_fun=None, measurement_cov_cholesky_fun=None
    ):
        if measurement_cov_fun is None:
            if measurement_cov_cholesky_fun is not None:
                raise ValueError
            return statespace.DiscreteGaussian.from_callable(
                state_trans_fun=self.__call__,
                jacob_state_trans_fun=self.jacobian,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
            )

        return statespace.DiscreteGaussian(
            state_trans_fun=self.__call__,
            jacob_state_trans_fun=self.jacobian,
            proc_noise_cov_mat_fun=measurement_cov_fun,
            proc_noise_cov_cholesky_fun=measurement_cov_cholesky_fun,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )

    def as_ekf_component(
        self, forward_implementation="sqrt", backward_implementation="sqrt"
    ):
        return filtsmooth.gaussian.approx.DiscreteEKFComponent(
            non_linear_model=self.as_transition(),
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )


class ODEInformationOperator(InformationOperator):
    """Information operators that depend on an ODE function."""

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim=input_dim, output_dim=output_dim)

        # Initialized once the ODE can be seen
        self.ode = None

    def incorporate_ode(self, ode):
        """Incorporate the ODE into the operator."""
        if self.ode_has_been_incorporated:
            raise ValueError
        else:
            self.ode = ode

    @property
    def ode_has_been_incorporated(self):
        return self.ode is not None
