"""Extended Kalman information.

Make an intractable information operator tractable with local
linearization.
"""
import numpy as np

from probnum import problems, statespace
from probnum.diffeq.odefiltsmooth import information_operators
from probnum.diffeq.odefiltsmooth.approx import _approx


class EK0(_approx.ODEInformationApproximationStrategy):
    """Make inference with the information operator tractable using a zeroth-order
    linearization of the ODE vector-field.

    This only applies to standard (explicit) ODEs. Implicit ODEs must
    use the EK1.
    """

    def __init__(self, forward_implementation="sqrt", backward_implementation="sqrt"):
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def __call__(
        self, information_operator: information_operators.ODEResidualOperator
    ) -> statespace.DiscreteGaussian:

        if not information_operator.ivp_has_been_set:
            raise ValueError

        ivp = information_operator.ivp
        custom_linearisation = lambda t, x: np.zeros((len(x), len(x)))

        new_ivp = problems.InitialValueProblem(
            f=ivp.f,
            df=custom_linearisation,
            y0=ivp.y0,
            t0=ivp.t0,
            tmax=ivp.tmax,
            solution=ivp.solution,
        )

        ek0_information_operator = information_operators.ODEResidualOperator(
            prior_ordint=information_operator.prior_ordint,
            prior_spatialdim=information_operator.prior_spatialdim,
        )
        ek0_information_operator.set_ivp(new_ivp)
        return ek0_information_operator.as_ekf_component()


class EK1(_approx.ODEInformationApproximationStrategy):
    """Make inference with the information operator tractable using a first-order
    linearization of the ODE vector-field."""

    def __init__(self, forward_implementation="sqrt", backward_implementation="sqrt"):
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def __call__(
        self, information_operator: information_operators.InformationOperator
    ) -> statespace.DiscreteGaussian:
        return information_operator.as_ekf_component()
