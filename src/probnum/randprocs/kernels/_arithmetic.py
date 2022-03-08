"""Kernel arithmetic."""
from ._arithmetic_fallbacks import SumKernel, _mul_fallback
from ._kernel import BinaryOperandType, Kernel


def add(op1: BinaryOperandType, op2: BinaryOperandType) -> Kernel:
    """Kernel summation."""
    return SumKernel(op1, op2)


def mul(op1: BinaryOperandType, op2: BinaryOperandType) -> Kernel:
    """Kernel multiplication."""
    return _mul_fallback(op1, op2)
