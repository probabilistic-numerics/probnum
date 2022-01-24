import pathlib

import numpy as np
import pytest
import pytest_cases
import scipy.linalg
from pytest_cases import filters

import probnum as pn

case_modules = [
    ".test_linops_cases." + path.stem
    for path in (pathlib.Path(__file__).parent / "test_linops_cases").glob("*_cases.py")
]


@pytest_cases.parametrize_with_cases(
    "linop,matrix",
    cases=case_modules,
    has_tag=(
        "symmetric",
        "positive-definite",
    ),
)
@pytest_cases.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_cholesky(linop: pn.linops.LinearOperator, matrix: np.ndarray, lower: bool):
    """Tests whether the lower (upper) Cholesky factor of a ``LinearOperator`` is a
    lower-triangular (upper-triangular) matrix square-root with positive diagonal."""

    linop_cho = linop.cholesky(lower=lower)

    np.testing.assert_allclose(
        (linop_cho @ linop_cho.T if lower else linop_cho.T @ linop_cho).todense(),
        matrix,
        err_msg="Not a valid matrix square root",
    )

    np.testing.assert_array_equal(
        np.diag(linop_cho.todense()) > 0,
        True,
        err_msg="Diagonal of the Cholesky factor is not positive.",
    )

    assert linop_cho.is_lower_triangular if lower else linop_cho.is_upper_triangular

    np.testing.assert_array_equal(
        (
            np.triu(linop_cho.todense(), k=1)
            if lower
            else np.tril(linop_cho.todense(), k=-1)
        ),
        0.0,
        err_msg=f"Cholesky factor is not {'lower' if lower else 'upper'} triangular",
    )

    # Test cacheing
    assert linop.cholesky(lower=lower) is linop_cho

    # Access transpose of the cached Cholesky factor
    np.testing.assert_array_equal(
        linop.cholesky(lower=not lower).T.todense(), linop_cho.todense()
    )


@pytest_cases.parametrize_with_cases(
    "linop,matrix",
    cases=case_modules,
)
@pytest_cases.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_cholesky_is_symmetric_not_true(
    linop: pn.linops.LinearOperator, matrix: np.ndarray, lower: bool
):  # pylint: disable=unused-argument
    """Tests whether computing the Cholesky decomposition of a ``LinearOperator``
    whose ``is_symmetric`` property is not set to ``True`` results in an error."""

    if linop.is_symmetric is not True:
        with pytest.raises(np.linalg.LinAlgError):
            linop.cholesky(lower=lower)


@pytest_cases.parametrize_with_cases(
    "linop,matrix",
    cases=case_modules,
)
@pytest_cases.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_cholesky_is_positive_definite_false(
    linop: pn.linops.LinearOperator, matrix: np.ndarray, lower: bool
):  # pylint: disable=unused-argument
    """Tests whether computing the Cholesky decomposition of a ``LinearOperator``
    whose ``is_symmetric`` property is not set to ``True`` results in an error."""

    if linop.is_positive_definite is False:
        with pytest.raises(np.linalg.LinAlgError):
            linop.cholesky(lower=lower)


@pytest_cases.parametrize_with_cases(
    "linop,matrix",
    cases=case_modules,
    filter=(
        filters.has_tag("symmetric")
        & (
            filters.has_tag("singular")
            | filters.has_tag("negative-definite")
            | filters.has_tag("indefinite")
        )
    ),
)
@pytest_cases.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_cholesky_not_positive_definite(
    linop: pn.linops.LinearOperator, matrix: np.ndarray, lower: bool
):
    """Tests whether computing the Cholesky decomposition of a symmetric, but not
    positive definite matrix results in an error"""

    expected_exception = None

    try:
        scipy.linalg.cholesky(matrix, lower=lower)
    except Exception as e:  # pylint: disable=broad-except
        expected_exception = e

    with pytest.raises(type(expected_exception)):
        linop.cholesky(lower=lower)
