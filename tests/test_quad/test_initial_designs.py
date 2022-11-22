"""Test cases for initial designs."""

import pytest

from probnum.quad.integration_measures import GaussianMeasure, LebesgueMeasure
from probnum.quad.solvers.initial_designs import InitialDesign, LatinDesign, MCDesign


@pytest.fixture
def input_dim():
    return 2


@pytest.fixture
def num_nodes():
    return 5


@pytest.fixture
def lebesgue_measure(input_dim):
    return LebesgueMeasure(domain=(0, 1), input_dim=input_dim)


@pytest.fixture
def gaussian_measure(input_dim):
    return GaussianMeasure(mean=0.0, cov=1.0, input_dim=input_dim)


@pytest.fixture(
    params=[
        pytest.param(sc, id=sc[0].__name__)
        for sc in [
            (MCDesign, "lebesgue_measure"),
            (MCDesign, "gaussian_measure"),
            (LatinDesign, "lebesgue_measure"),
        ]
    ],
    name="design",
)
def fixture_stopping_criterion(request, num_nodes) -> InitialDesign:
    measure = request.getfixturevalue(request.param[1])
    return request.param[0](**dict(measure=measure, num_nodes=num_nodes))


def test_initial_design_values(design, num_nodes):
    assert design.num_nodes == num_nodes


def test_initial_design_shapes(design, rng, input_dim, num_nodes):
    res = design(rng=rng)  # get nodes
    assert res.shape == (num_nodes, input_dim)


def test_latin_design_wrong_domain(gaussian_measure, num_nodes):
    # Latin design requires finite domain but Gaussian measure has infinite domain)
    with pytest.raises(ValueError):
        LatinDesign(measure=gaussian_measure, num_nodes=num_nodes)
