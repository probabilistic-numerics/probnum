"""ODE residual information operators."""


from probnum import statespace
from probnum.diffeq.odefiltsmooth.information_operators import _information_operator

__all__ = ["ODEResidual"]

# Implicit ODE Residuals might be done somewhere else?!
# The advantage would be that the EK0 could throw errors as soon
# as the to-be-linearized information operator is not the residual of an explicit ODE.
class ODEResidual(_information_operator.ODEInformationOperator):
    """Information operator that measures the residual of an explicit ODE."""

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

    def incorporate_ode(self, ode):
        if self.ode_has_been_incorporated:
            raise ValueError
        else:
            self.ode = ode

        # Cache the projection matrices and match the implementation to the ODE
        dummy_integrator = statespace.Integrator(
            ordint=self.prior_ordint, spatialdim=self.prior_spatialdim
        )
        ode_order = 1  # currently everything we can do
        self.projection_matrices = [
            dummy_integrator.proj2coord(coord=deriv) for deriv in range(ode_order + 1)
        ]
        res, res_jac = self._match_residual_and_jacobian_to_ode_order(
            ode_order=ode_order
        )
        self._residual, self._residual_jacobian = res, res_jac

        # For higher order IVPs, do something along the lines of
        # self.proj_matrices = [dummy_integrator.proj2coord(coord=deriv) for deriv in range(ivp.order + 1)]
        # self._residual, self._residual_jacobian = self._match_residual_and_jacobian_to_ode_order(ode_order=ode_order)

    def _match_residual_and_jacobian_to_ode_order(self, ode_order):
        choose_implementation = {
            1: (self._residual_first_order_ode, self._residual_first_order_ode_jacobian)
        }
        return choose_implementation[ode_order]

    def __call__(self, t, x):
        if not self.ode_has_been_incorporated:
            raise ValueError
        return self._residual(t, x)

    def jacobian(self, t, x):
        return self._residual_jacobian(t, x)

    # Implementation of different residuals

    def _residual_first_order_ode(self, t, x):
        h0, h1 = self.projection_matrices
        return h1 @ x - self.ode.f(t, h0 @ x)

    def _residual_first_order_ode_jacobian(self, t, x):
        h0, h1 = self.projection_matrices
        return h1 - self.ode.df(t, h0 @ x) @ h0

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
