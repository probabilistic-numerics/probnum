import operator
from typing import Any, Callable, Tuple, Type, Union

from probnum import backend, compat, randvars
from probnum.backend.typing import ShapeType

from .operand_generators import (
    GeneratorFnType,
    array_generator,
    constant_generator,
    normal_generator,
)

import pytest
from pytest_cases import fixture, parametrize


@fixture(scope="package")
@parametrize(
    shapes_=[
        ((), ()),
        ((1,), (1,)),
        ((4,), (4,)),
        ((2, 3), (2, 3)),
        ((2, 3, 2), (2, 3, 2)),
        # ((3,), ()),  # This is broken if the `Normal` random variable has fewer
        # entries.
        # ((3, 1), (1, 4)),  # This is broken if `Normal`s are involved
    ]
)
def shapes(shapes_: Tuple[ShapeType, ShapeType]) -> Tuple[ShapeType, ShapeType]:
    return shapes_


OperandType = Union[randvars.RandomVariable, backend.Array]


@fixture(scope="package")
@parametrize(
    operator_operands_and_expected_result_type_=[
        (operator.add, constant_generator, constant_generator, randvars.Constant),
        (operator.sub, constant_generator, constant_generator, randvars.Constant),
        (operator.add, constant_generator, array_generator, randvars.Constant),
        (operator.sub, constant_generator, array_generator, randvars.Constant),
        (operator.add, array_generator, constant_generator, randvars.Constant),
        (operator.sub, array_generator, constant_generator, randvars.Constant),
        (operator.add, normal_generator, normal_generator, randvars.Normal),
        (operator.sub, normal_generator, normal_generator, randvars.Normal),
        (operator.add, normal_generator, constant_generator, randvars.Normal),
        (operator.sub, normal_generator, constant_generator, randvars.Normal),
        (operator.add, constant_generator, normal_generator, randvars.Normal),
        (operator.sub, constant_generator, normal_generator, randvars.Normal),
        (operator.add, normal_generator, array_generator, randvars.Normal),
        (operator.sub, normal_generator, array_generator, randvars.Normal),
        (operator.add, array_generator, normal_generator, randvars.Normal),
        (operator.sub, array_generator, normal_generator, randvars.Normal),
    ],
)
def operator_operands_and_expected_result_type(
    shapes: Tuple[ShapeType, ShapeType],
    operator_operands_and_expected_result_type_: Tuple[
        Callable[[Any, Any], Any],
        GeneratorFnType,
        GeneratorFnType,
        Type[randvars.RandomVariable],
    ],
) -> Tuple[
    Callable[[Any, Any], Any],
    OperandType,
    OperandType,
    Type[randvars.RandomVariable],
]:
    shape0, shape1 = shapes

    (
        operator,
        generator0,
        generator1,
        expected_result_type,
    ) = operator_operands_and_expected_result_type_

    return operator, generator0(shape0), generator1(shape1), expected_result_type


@fixture(scope="package")
def operator(
    operator_operands_and_expected_result_type: Tuple[
        Callable[[Any, Any], Any],
        OperandType,
        OperandType,
        Type[randvars.RandomVariable],
    ]
) -> Callable[[Any, Any], Any]:
    return operator_operands_and_expected_result_type[0]


@fixture(scope="package")
def operand0(
    operator_operands_and_expected_result_type: Tuple[
        Callable[[Any, Any], Any],
        OperandType,
        OperandType,
        Type[randvars.RandomVariable],
    ]
) -> OperandType:
    return operator_operands_and_expected_result_type[1]


@fixture(scope="package")
def operand1(
    operator_operands_and_expected_result_type: Tuple[
        Callable[[Any, Any], Any],
        OperandType,
        OperandType,
        Type[randvars.RandomVariable],
    ]
) -> OperandType:
    return operator_operands_and_expected_result_type[2]


@fixture(scope="package")
def expected_result_type(
    operator_operands_and_expected_result_type: Tuple[
        Callable[[Any, Any], Any],
        OperandType,
        OperandType,
        Type[randvars.RandomVariable],
    ]
) -> Type[randvars.RandomVariable]:
    return operator_operands_and_expected_result_type[3]


@fixture(scope="package")
def result(
    operator: Callable[[Any, Any], Any],
    operand0: OperandType,
    operand1: OperandType,
) -> randvars.RandomVariable:
    return operator(operand0, operand1)


def test_type(
    result: randvars.RandomVariable, expected_result_type: Callable[[Any, Any], Any]
):
    assert isinstance(result, expected_result_type)


def test_shape(
    operand0: OperandType,
    operand1: OperandType,
    result: randvars.RandomVariable,
):
    if not isinstance(operand0, randvars.RandomVariable):
        operand0 = randvars.asrandvar(operand0)

    if not isinstance(operand1, randvars.RandomVariable):
        operand1 = randvars.asrandvar(operand1)

    expected_shape = backend.broadcast_shapes(operand0.shape, operand1.shape)
    assert result.shape == expected_shape


def test_mean(
    operator: Callable[[Any, Any], Any],
    operand0: OperandType,
    operand1: OperandType,
    result: randvars.RandomVariable,
):
    if not isinstance(operand0, randvars.RandomVariable):
        operand0 = randvars.asrandvar(operand0)

    if not isinstance(operand1, randvars.RandomVariable):
        operand1 = randvars.asrandvar(operand1)

    try:
        mean0 = operand0.mean
        mean1 = operand1.mean
    except NotImplementedError:
        pytest.skip()

    compat.testing.assert_allclose(result.mean, operator(mean0, mean1))


def test_cov(
    operand0: OperandType,
    operand1: OperandType,
    result: randvars.RandomVariable,
):
    if not isinstance(operand0, randvars.RandomVariable):
        operand0 = randvars.asrandvar(operand0)

    if not isinstance(operand1, randvars.RandomVariable):
        operand1 = randvars.asrandvar(operand1)

    try:
        cov0 = operand0.cov
        cov1 = operand1.cov
    except NotImplementedError:
        pytest.skip()

    expected_cov = (
        cov0.reshape(operand0.shape + operand0.shape)
        + cov1.reshape(operand1.shape + operand1.shape)
    ).reshape(result.cov.shape)

    compat.testing.assert_allclose(result.cov, expected_cov)
