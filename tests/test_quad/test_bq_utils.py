"""Basic tests for bq utils."""

import numpy as np
import pytest

from probnum.quad._utils import as_domain


# fmt: off
@pytest.mark.parametrize(
    "dom, in_dim",
    [
        ((0, 1), 0),  # zero dimension
        ((0, 1), -2),  # negative dimension
        ((np.zeros(2), np.ones(2)), 3),  # length of bounds does not match dimension
        ((np.zeros(2), np.ones(3)), None),  # lower and upper bounds not equal lengths
        ((np.array([0, 0]), np.array([1, 0])), None),  # integration domain is empty
        ((np.zeros([2, 1]), np.ones([2, 1])), None),  # bounds have too many dimensions
        ((np.zeros([2, 1]), np.ones([2, 1])), 2),  # bounds have too many dimensions
        ((0, 1, 2), 2),  # domain has too many elements
        ((-np.ones(2), np.zeros(2), np.ones(2)), 2),  # domain has too many elements
    ]
)
def test_as_domain_wrong_input(dom, in_dim):
    with pytest.raises(ValueError):
        as_domain(dom, in_dim)


@pytest.mark.parametrize(
    "dom, in_dim",
    [
        ((0, 1), 1),  # convert bounds to 1D array
        ((0, 1), 3),  # expand bounds to 3D array
        ((np.zeros(3), np.ones(3)), 3)  # bounds already expanded
    ]
)
def test_as_domain_returns_correct_shape(dom, in_dim):
    domain, _ = as_domain(dom, in_dim)
    assert len(domain) == 2
    assert domain[0].ndim == 1 and domain[0].ndim == 1
    assert domain[0].shape[0] == in_dim and domain[1].shape[0] == in_dim


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
    domain, input_dim = as_domain(dom, in_dim)
    assert isinstance(input_dim, int) and isinstance(domain, tuple)
    assert isinstance(domain[0], np.ndarray) and isinstance(domain[1], np.ndarray)


def test_as_domain_correct_values():
    in_dim, lb, ub = 3, 0.0, 1.5
    domain, input_dim = as_domain((lb, ub), in_dim)

    lb_expanded = lb * np.ones(in_dim)
    ub_expanded = ub * np.ones(in_dim)
    np.testing.assert_allclose(domain[0], lb_expanded, atol=0.0, rtol=1e-12)
    np.testing.assert_allclose(domain[1], ub_expanded, atol=0.0, rtol=1e-12)
# fmt: on
