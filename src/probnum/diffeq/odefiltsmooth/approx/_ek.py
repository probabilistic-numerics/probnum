"""Extended Kalman information.

Make an intractable information operator tractable with local
linearization.
"""
from probnum import filtsmooth, statespace
from probnum.diffeq.odefiltsmooth import information_operators
from probnum.diffeq.odefiltsmooth.approx import _approx


class EK0(_approx.ODEInformationApproximationStrategy):
    def __init__(self, forward_implementation="sqrt", backward_implementation="sqrt"):
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def __call__(
        self, ode_info: information_operators.ODEInformation
    ) -> statespace.DiscreteGaussian:
        def ek0_jacobian(t, x):
            return ode_info.h1

        ek0_model = statespace.DiscreteGaussian.from_callable(
            input_dim=ode_info.information_model.input_dim,
            output_dim=ode_info.information_model.output_dim,
            state_trans_fun=ode_info.information_model.state_trans_fun,
            jacob_state_trans_fun=ek0_jacobian,
        )
        return filtsmooth.DiscreteEKFComponent(
            ek0_model,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
        )


class EK1(_approx.ODEInformationApproximationStrategy):
    def __init__(self, forward_implementation="sqrt", backward_implementation="sqrt"):
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def __call__(
        self, ode_info: information_operators.ODEInformation
    ) -> statespace.DiscreteGaussian:

        return filtsmooth.DiscreteEKFComponent(
            ode_info.information_model,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
        )
