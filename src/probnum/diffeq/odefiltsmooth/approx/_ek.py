"""Extended Kalman information.

Turn an intractable information operator into a tractable one.
"""
from probnum import filtsmooth, statespace
from probnum.diffeq.odefiltsmooth import information_operators


def ek0(
    info: information_operators.ODEInformation,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
) -> statespace.DiscreteGaussian:
    def ek0_jacobian(t, x):
        return info.h1

    ek0_model = statespace.DiscreteGaussian.from_callable(
        input_dim=info.information_model.input_dim,
        output_dim=info.information_model.output_dim,
        state_trans_fun=info.information_model.state_trans_fun,
        jacob_state_trans_fun=ek0_jacobian,
    )
    return filtsmooth.DiscreteEKFComponent(
        ek0_model,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )


def ek1(
    info: information_operators.ODEInformation,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
) -> statespace.DiscreteGaussian:
    return filtsmooth.DiscreteEKFComponent(
        info.information_model,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )


#
# def uk(info: information_operators.ODEInformation) -> statespace.DiscreteGaussian:
#     return filtsmooth.gaussian.approx.DiscreteUKFComponent(info.information_model)
