"""ODE residual information operators."""

from typing import Callable, Tuple

import numpy as np

from probnum import problems, randprocs
from probnum.diffeq.odefilter.information_operators import _information_operator
from probnum.typing import FloatLike, IntLike

__all__ = ["ODEResidual"]


class ODEResidual(_information_operator.ODEInformationOperator):
    """Information operator that measures the residual of an explicit ODE."""

    def __init__(self, num_prior_derivatives: IntLike, ode_dimension: IntLike):
        integrator_dimension = ode_dimension * (num_prior_derivatives + 1)
        super().__init__(input_dim=integrator_dimension, output_dim=ode_dimension)
        # Store remaining attributes
        self.num_prior_derivatives = num_prior_derivatives
        self.ode_dimension = ode_dimension

        # Prepare caching the projection matrices
        self.projection_matrices = None

        # These will be assigned once the ODE has been seen
        self._residual = None
        self._residual_jacobian = None

    def incorporate_ode(self, ode: problems.InitialValueProblem):
        """Incorporate the ODE and cache the required projection matrices."""
        super().incorporate_ode(ode=ode)

        # Cache the projection matrices and match the implementation to the ODE
        dummy_integrator = randprocs.markov.integrator.IntegratorTransition(
            num_derivatives=self.num_prior_derivatives,
            wiener_process_dimension=self.ode_dimension,
        )
        ode_order = 1  # currently everything we can do
        self.projection_matrices = [
            dummy_integrator.proj2coord(coord=deriv) for deriv in range(ode_order + 1)
        ]
        res, res_jac = self._match_residual_and_jacobian_to_ode_order(
            ode_order=ode_order
        )
        self._residual, self._residual_jacobian = res, res_jac

    def _match_residual_and_jacobian_to_ode_order(
        self, ode_order: IntLike
    ) -> Tuple[Callable, Callable]:
        """Choose the correct residual (and Jacobian) implementation based on the order
        of the ODE."""
        choose_implementation = {
            1: (self._residual_first_order_ode, self._residual_first_order_ode_jacobian)
        }
        return choose_implementation[ode_order]

    def __call__(self, t: FloatLike, x: np.ndarray) -> np.ndarray:
        return self._residual(t, x)

    def jacobian(self, t: FloatLike, x: np.ndarray) -> np.ndarray:
        return self._residual_jacobian(t, x)

    # Implementation of different residuals

    def _residual_first_order_ode(self, t: FloatLike, x: np.ndarray) -> np.ndarray:
        h0, h1 = self.projection_matrices
        return h1 @ x - np.asarray(self.ode.f(t, h0 @ x))

    def _residual_first_order_ode_jacobian(
        self, t: FloatLike, x: np.ndarray
    ) -> np.ndarray:
        h0, h1 = self.projection_matrices
        return h1 - np.asarray(self.ode.df(t, h0 @ x) @ h0)
