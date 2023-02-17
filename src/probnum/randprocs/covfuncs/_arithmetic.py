"""Covariance function arithmetic."""
from ._arithmetic_fallbacks import SumCovarianceFunction, _mul_fallback
from ._covariance_function import BinaryOperandType, CovarianceFunction


# pylint: disable=missing-param-doc
def add(op1: BinaryOperandType, op2: BinaryOperandType) -> CovarianceFunction:
    """Covariance function summation."""
    return SumCovarianceFunction(op1, op2)


def mul(op1: BinaryOperandType, op2: BinaryOperandType) -> CovarianceFunction:
    """Covariance function multiplication."""
    return _mul_fallback(op1, op2)


# pylint: enable=missing-param-doc
