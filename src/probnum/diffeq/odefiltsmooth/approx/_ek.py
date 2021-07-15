"""Extended Kalman information.

Turn an intractable information operator into a tractable one.
"""
from probnum import filtsmooth, statespace
from probnum.diffeq.odefiltsmooth import information_operators


def ek0(info: information_operators.ODEInformation) -> statespace.DiscreteGaussian:
    def ek0_jacobian(t, x):
        return h1

    ek0_model = statespace.DiscreteGaussian.from_callable(
        input_dim=info.information_model.input_dim,
        output_dim=info.information_model.output_dim,
        state_trans_fun=info.information_model.dyna,
        jacob_state_trans_fun=ek0_jacobian,
    )
    return filtsmooth.gaussian.approx.DiscreteEKFComponent(ek0_model)


def ek1(info: information_operators.ODEInformation) -> statespace.DiscreteGaussian:
    return filtsmooth.gaussian.approx.DiscreteEKFComponent(info.information_model)


#
# def uk(info: information_operators.ODEInformation) -> statespace.DiscreteGaussian:
#     return filtsmooth.gaussian.approx.DiscreteUKFComponent(info.information_model)
