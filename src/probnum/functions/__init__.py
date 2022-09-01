"""Callables with in- and output shape information supporting algebraic operations."""

from . import _algebra
from ._algebra_fallbacks import ScaledFunction, SumFunction
from ._function import Function, LambdaFunction
from ._zero import Zero
