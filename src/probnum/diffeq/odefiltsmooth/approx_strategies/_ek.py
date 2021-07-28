"""Extended Kalman filter-based approximation strategies.

Make an intractable information operator tractable with local linearization.
"""
from typing import Optional

import numpy as np

from probnum import problems
from probnum.diffeq.odefiltsmooth import information_operators
from probnum.diffeq.odefiltsmooth.approx_strategies import _approx_strategy


class EK1(_approx_strategy.ApproximationStrategy):
    """Make inference with an (ODE-)information operator tractable using a first-order
    linearization of the ODE vector-field."""

    def __init__(
        self,
        forward_implementation: Optional[str] = "sqrt",
        backward_implementation: Optional[str] = "sqrt",
    ):
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def __call__(
        self, information_operator: information_operators.InformationOperator
    ) -> information_operators.LocallyLinearizedInformationOperator:

        return information_operators.LocallyLinearizedInformationOperator(
            information_operator=information_operator,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
        )


class EK0(_approx_strategy.ApproximationStrategy):
    """Make inference with the information operator tractable using a zeroth-order
    linearization of the ODE vector-field.

    This only applies to standard (explicit) ODEs. Implicit ODEs must use the EK1.
    """

    def __init__(
        self,
        forward_implementation: Optional[str] = "sqrt",
        backward_implementation: Optional[str] = "sqrt",
    ):
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def __call__(
        self, information_operator: information_operators.ODEResidual
    ) -> information_operators.LocallyLinearizedInformationOperator:

        if not information_operator.ode_has_been_incorporated:
            raise ValueError(
                "ODE has not been incorporated into the ODE information operator."
            )

        # The following EK0 implementation generalizes to higher-order ODEs in the sense
        # that for higher order ODEs, the attribute `df` is a list of Jacobians,
        # and in this case we can loop over "all" Jacobians and set them to the
        # custom (zero) linearization.
        ode = information_operator.ode
        custom_linearization = lambda t, x: np.zeros((len(x), len(x)))
        new_ivp = problems.InitialValueProblem(
            f=ode.f,
            df=custom_linearization,
            y0=ode.y0,
            t0=ode.t0,
            tmax=ode.tmax,
            solution=ode.solution,
        )

        # From here onwards, mimic the EK1 implementation.
        ek0_information_operator = information_operators.ODEResidual(
            num_prior_derivatives=information_operator.num_prior_derivatives,
            ode_dimension=information_operator.ode_dimension,
        )
        ek0_information_operator.incorporate_ode(ode=new_ivp)
        return information_operators.LocallyLinearizedInformationOperator(
            information_operator=ek0_information_operator,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
        )
