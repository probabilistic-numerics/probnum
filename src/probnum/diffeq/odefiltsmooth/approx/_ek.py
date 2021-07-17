"""Extended Kalman information.

Make an intractable information operator tractable with local
linearization.
"""
import numpy as np

from probnum import problems, statespace
from probnum.diffeq.odefiltsmooth import information_operators
from probnum.diffeq.odefiltsmooth.approx import _approx


class EK1(_approx.ApproximationStrategy):
    """Make inference with the information operator tractable using a first-order
    linearization of the ODE vector-field."""

    def __init__(self, forward_implementation="sqrt", backward_implementation="sqrt"):
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def __call__(
        self, information_operator: information_operators.InformationOperator
    ) -> statespace.DiscreteGaussian:
        if not information_operator.ode_has_been_incorporated:
            raise ValueError

        return information_operator.as_ekf_component()


class EK0(_approx.ApproximationStrategy):
    """Make inference with the information operator tractable using a zeroth-order
    linearization of the ODE vector-field.

    This only applies to standard (explicit) ODEs. Implicit ODEs must
    use the EK1.
    """

    def __init__(self, forward_implementation="sqrt", backward_implementation="sqrt"):
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def __call__(
        self, information_operator: information_operators.ExplicitODEResidual
    ) -> statespace.DiscreteGaussian:

        if not information_operator.ode_has_been_incorporated:
            raise ValueError

        ode = information_operator.ode
        custom_linearisation = lambda t, x: np.zeros((len(x), len(x)))

        new_ivp = problems.InitialValueProblem(
            f=ode.f,
            df=custom_linearisation,
            y0=ode.y0,
            t0=ode.t0,
            tmax=ode.tmax,
            solution=ode.solution,
        )

        ek0_information_operator = information_operators.ExplicitODEResidual(
            prior_ordint=information_operator.prior_ordint,
            prior_spatialdim=information_operator.prior_spatialdim,
        )
        ek0_information_operator.incorporate_ode(new_ivp)
        return ek0_information_operator.as_ekf_component()
