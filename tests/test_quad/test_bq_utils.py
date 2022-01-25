"""Basic tests for bq utils."""

import numpy as np
import pytest

from probnum.quad import IntegrationMeasure


# fmt: off
@pytest.mark.parametrize(
    "dom, in_dim",
    [
        ((0, 1), -2),  # negative dimension
        ((np.zeros(2), np.ones(2)), 3),  # length of bounds does not match dimension
        ((np.zeros(2), np.ones(3)), None),  # lower and upper bounds not equal lengths
        ((np.array([0, 0]), np.array([1, 0])), None),  # integration domain is empty
        ((np.zeros([2, 1]), np.ones([2, 1])), None),  # bounds have too many dimensions
        ((np.zeros([2, 1]), np.ones([2, 1])), 2),  # bounds have too many dimensions
    ]
)
def test_as_domain_wrong_input(dom, in_dim):
    with pytest.raises(ValueError):
        IntegrationMeasure.as_domain(dom, in_dim)


@pytest.mark.parametrize(
    "dom, in_dim",
    [
        ((0, 1), 1),  # convert bounds to 1D array
        ((0, 1), 3),  # expand bounds to 3D array
        ((np.zeros(3), np.ones(3)), 3)  # bounds already expanded
    ]
)
def test_as_domain_returns_correct_shape(dom, in_dim):
    as_domain, as_input_dim = IntegrationMeasure.as_domain(dom, in_dim)
    assert len(as_domain) == 2
    assert as_domain[0].ndim == 1 and as_domain[0].ndim == 1
    assert as_domain[0].shape[0] == in_dim and as_domain[1].shape[0] == in_dim


@pytest.mark.parametrize(
    "dom, in_dim",
    [
        ((0, 1), None),  # convert to 1D array
        ((0, 1), 1),  # expand to 1D array
        ((0, 1), 3),  # expand to 3D array
        ((np.zeros(3), np.ones(3)), None),  # already expanded
        ((np.zeros(3), np.ones(3)), 3),  # already expanded
    ]
)
def test_as_domain_returns_correct_type(dom, in_dim):
    as_domain, as_input_dim = IntegrationMeasure.as_domain(dom, in_dim)
    assert isinstance(as_input_dim, int) and isinstance(as_domain, tuple)
    assert isinstance(as_domain[0], np.ndarray) and isinstance(as_domain[1], np.ndarray)


def test_as_domain_returns_correct_none():
    as_domain, as_input_dim = IntegrationMeasure.as_domain(None, None)
    assert as_domain is None and as_input_dim is None

    as_domain, as_input_dim = IntegrationMeasure.as_domain(None, float(1.0))
    assert as_domain is None and isinstance(as_input_dim, int)


def test_as_domain_correct_values():
    in_dim, lb, ub = 3, 0.0, 1.5
    as_domain, as_input_dim = IntegrationMeasure.as_domain((lb, ub), in_dim)

    lb_expanded = lb * np.ones(in_dim)
    ub_expanded = ub * np.ones(in_dim)
    np.testing.assert_allclose(as_domain[0], lb_expanded, atol=0.0, rtol=1e-12)
    np.testing.assert_allclose(as_domain[1], ub_expanded, atol=0.0, rtol=1e-12)
# fmt: on
