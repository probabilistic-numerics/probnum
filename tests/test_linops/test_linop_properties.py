import pathlib

import numpy as np
import pytest
import pytest_cases

import probnum as pn

case_modules = [
    ".test_linops_cases." + path.stem
    for path in (pathlib.Path(__file__).parent / "test_linops_cases").glob("*_cases.py")
]


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_symmetric_linop_must_be_square(
    linop: pn.linops.LinearOperator, matrix: np.ndarray
):
    if not linop.is_square:
        with pytest.raises(ValueError):
            linop.is_symmetric = True


@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_positive_definite_linop_must_be_symmetric(
    linop: pn.linops.LinearOperator, matrix: np.ndarray
):
    if not linop.is_symmetric:
        with pytest.raises(ValueError):
            linop.is_positive_definite = True
