"""Approximate information operators."""

from probnum import filtsmooth
from probnum.diffeq.odefiltsmooth.information_operators import _information_operator

__all__ = ["ApproximateInformationOperator"]


class ApproximateInformationOperator(_information_operator.InformationOperator):
    """Approximate information operators."""

    def __init__(
        self,
        information_operator,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    ):
        self.information_operator = information_operator
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def __call__(self, t, x):
        return self.information_operator(t, x)

    def jacobian(self, t, x):
        return self.information_operator.jacobian(t, x)

    def as_transition(
        self,
        measurement_cov_fun=None,
        measurement_cov_cholesky_fun=None,
    ):
        """Return an approximate transition.

        In this case, an EKF component.
        """
        transition = self.information_operator.as_transition(
            measurement_cov_fun=measurement_cov_fun,
            measurement_cov_cholesky_fun=measurement_cov_cholesky_fun,
        )
        return filtsmooth.gaussian.approx.DiscreteEKFComponent(
            non_linear_model=transition,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
        )
