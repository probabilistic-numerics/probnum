"""Kernel arithmetic."""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from probnum import utils
from probnum.typing import NotImplementedType

from ._arithmetic_fallbacks import SumKernel, _mul_fallback
from ._kernel import BinaryOperandType, Kernel


def add(op1: BinaryOperandType, op2: BinaryOperandType) -> Kernel:
    return _apply(_add_fns, op1, op2, fallback_operator=SumKernel)


def mul(op1: BinaryOperandType, op2: BinaryOperandType) -> Kernel:
    return _apply(_mul_fns, op1, op2, fallback_operator=_mul_fallback)


########################################################################################
# Operator registry
########################################################################################

_BinaryOperatorType = Callable[[Kernel, Kernel], Union[Kernel, NotImplementedType]]
_BinaryOperatorRegistryType = Dict[Tuple[type, type], _BinaryOperatorType]


_add_fns: _BinaryOperatorRegistryType = {}
_mul_fns: _BinaryOperatorRegistryType = {}

########################################################################################
# Fill Arithmetics Registries
########################################################################################


########################################################################################
# Apply
########################################################################################


def _apply(
    op_registry: _BinaryOperatorRegistryType,
    op1: Kernel,
    op2: Kernel,
    fallback_operator: Optional[
        Callable[
            [Kernel, Kernel],
            Union[Kernel, NotImplementedType],
        ]
    ] = None,
) -> Union[Kernel, NotImplementedType]:
    if np.ndim(op1) == 0:
        key1 = np.number
        op1 = utils.as_numpy_scalar(op1)
    else:
        key1 = type(op1)

    if np.ndim(op2) == 0:
        key2 = np.number
        op2 = utils.as_numpy_scalar(op2)
    else:
        key2 = type(op2)

    key = (key1, key2)

    if key in op_registry:
        res = op_registry[key](op1, op2)
    else:
        res = NotImplemented

    if res is NotImplemented and fallback_operator is not None:
        res = fallback_operator(op1, op2)

    return res
