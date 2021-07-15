"""Interface for information operators."""

import abc

from probnum import filtsmooth, statespace


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

        # Initialized once the IVP can be seen
        self.ivp = None

    def set_ivp(self, ivp):
        """Set the initial value problem."""
        if self.ivp_has_been_set:
            raise ValueError
        else:
            self.ivp = ivp

    @property
    def ivp_has_been_set(self):
        return self.ivp is not None

    @abc.abstractmethod
    def __call__(self, t, x):
        raise NotImplementedError

    def jacobian(self, t, x):
        raise NotImplementedError

    def as_transition(self):
        return statespace.DiscreteGaussian.from_callable(
            state_trans_fun=self.__call__,
            jacob_state_trans_fun=self.jacobian,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )

    def as_ekf_component(
        self, forward_implementation="sqrt", backward_implementation="sqrt"
    ):
        return filtsmooth.DiscreteEKFComponent(
            non_linear_model=self.as_transition(),
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )


# Implicit ODE Residuals might be done somewhere else?!
# The advantage would be that the EK0 could throw errors as soon
# as the to-be-linearized information operator is not the residual of an explicit ODE.
class ODEResidualOperator(InformationOperator):
    """Information operator that measures the residual of an ODE."""

    def __init__(self, prior_ordint, prior_spatialdim):
        integrator_dimension = prior_spatialdim * (prior_ordint + 1)
        expected_ode_dimension = prior_spatialdim
        super().__init__(
            input_dim=integrator_dimension, output_dim=expected_ode_dimension
        )

        # Prepare caching the projection matrices
        self.projection_matrices = None
        self.prior_ordint = prior_ordint
        self.prior_spatialdim = prior_spatialdim

        self._residual = None
        self._residual_jacobian = None

    def set_ivp(self, ivp):
        if self.ivp_has_been_set:
            raise ValueError
        else:
            self.ivp = ivp

        # Cache the projection matrices and match the implementation to the ODE
        dummy_integrator = statespace.Integrator(
            ordint=self.prior_ordint, spatialdim=self.prior_spatialdim
        )
        ode_order = 1  # currently everything we can do
        self.projection_matrices = [
            dummy_integrator.proj2coord(coord=deriv) for deriv in range(ode_order + 1)
        ]
        self._residual = self._residual_first_order_ode
        self._residual_jacobian = self._residual_first_order_ode_jacobian

        # For higher order IVPs, do something along the lines of
        # self.proj_matrices = [dummy_integrator.proj2coord(coord=deriv) for deriv in range(ivp.order + 1)]
        # self._residual, self._residual_jacobian = self._match_residual_and_jacobian(ode_order=ode_order)

    def _match_residual_and_jacobian(self, ode_order):
        choose_implementation = {
            1: (self._residual_first_order_ode, self._residual_first_order_ode_jacobian)
        }
        return choose_implementation[ode_order]

    def __call__(self, t, x):
        return self._residual(t, x)

    def jacobian(self, t, x):
        return self._residual_jacobian(t, x)

    # Implementation of different residuals

    def _residual_first_order_ode(self, t, x):
        h0, h1 = self.projection_matrices
        return h1 @ x - self.ivp.f(t, h0 @ x)

    def _residual_first_order_ode_jacobian(self, t, x):
        h0, h1 = self.projection_matrices
        return h1 - self.ivp.df(t, h0 @ x) @ h0

    # Implementation of the residuals for higher order ODEs:
    #
    # def _residual_second_order_ode(self, t, x):
    #     h0, h1, h2 = self.projection_matrices
    #     return h2 @ x - self.ivp.f(t, h0 @ x, h1 @ x)
    #
    # def _residual_second_order_ode_jacobian(self, t, x):
    #     h0, h1, h2 = self.projection_matrices
    #     df_dx0, df_dx1 = self.ivp.df
    #     return h1 - df_dx0(t, h0 @ x, h1 @ x) @ h0 - df_dx1(t, h0 @ x, h1 @ x) @ h1
    #
    # This way, the EK0 can jump right into setting all the Jacobians to zero
    # and suddenly works out-of-the-box for higher order ODEs!
