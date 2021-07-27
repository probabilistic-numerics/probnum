"""Approximate information operators."""

import abc
from typing import Optional

from probnum import filtsmooth
from probnum.diffeq.odefiltsmooth.information_operators import _information_operator

__all__ = ["ApproximateInformationOperator"]


class ApproximateInformationOperator(
    _information_operator.InformationOperator, abc.ABC
):
    """Approximate information operators.

    An approximate information operator is a version of an information operator that
    differs from its non-approximated operator in two ways:

    1) When it is transformed into a transition, the output is an approximate transition such as an EKF component.
    2) The Jacobian might be different to the Jacobian of the original version.

    Approximate information operators are returned by approximation strategies such as EK0 and EK1.
    For instance, the EK0 changes the Jacobian of the information operator
    (in the sense that it sets the Jacobian of the ODE vector field to zero).
    """

    def __init__(
        self,
        information_operator: _information_operator.InformationOperator,
    ):
        super().__init__(
            input_dim=information_operator.input_dim,
            output_dim=information_operator.output_dim,
        )
        self.information_operator = information_operator

    def __call__(self, t, x):
        return self.information_operator(t, x)

    def jacobian(self, t, x):
        return self.information_operator.jacobian(t, x)

    @abc.abstractmethod
    def as_transition(
        self,
        measurement_cov_fun=None,
        measurement_cov_cholesky_fun=None,
    ):
        raise NotImplementedError


class LocallyLinearizedInformationOperator(ApproximateInformationOperator):
    """Approximate information operators based on local linearization."""

    def __init__(
        self,
        information_operator: _information_operator.InformationOperator,
        forward_implementation: Optional[str] = "sqrt",
        backward_implementation: Optional[str] = "sqrt",
    ):
        super().__init__(
            information_operator=information_operator,
        )
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

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
