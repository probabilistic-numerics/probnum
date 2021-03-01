# """Test for utility functions."""
#
#
# import numpy as np
# import pytest
#
# import probnum.diffeq as pnde
# import probnum.filtsmooth.statespace as pnss
#
# from ._known_initial_derivatives import THREEBODY_INITS
#
#
# @pytest.fixture
# def order():
#     return 3
#
# @pytest.fixture
# def ode_dim():
#     return 4
#
# @pytest.fixture
# def threebody_inits(order, ode_dim):
#     return THREEBODY_INITS[: ode_dim * (order+1)]
#
#
# @pytest.fixture
# def in_out_pair(threebody_inits, order):
#     """Initial states for the three-body initial values.
#
#     Returns (derivwise, coordwise)
#     """
#     return threebody_inits, threebody_inits.reshape((-1, order + 1)).T.flatten()
#
#
# def test_in_out_pair_is_not_identical(in_out_pair):
#     # A little sanity check to assert that these are actually different, so the conversion is non-trivial.
#     derivwise, coordwise = in_out_pair
#     assert np.linalg.norm(derivwise - coordwise) > 10.0
#
#
# def test_convert_coordwise_to_derivwise(in_out_pair, order, ode_dim):
#     derivwise, coordwise = in_out_pair
#
#     coordwise_as_derivwise = pnss.convert_coordwise_to_derivwise(coordwise, order, ode_dim)
#     np.testing.assert_allclose(coordwise_as_derivwise, derivwise)
#
#
# def test_convert_derivwise_to_coordwise(in_out_pair, order, ode_dim):
#     derivwise, coordwise = in_out_pair
#     derivwise_as_coordwise = pnss.convert_derivwise_to_coordwise(derivwise, order, ode_dim)
#     np.testing.assert_allclose(derivwise_as_coordwise, coordwise)
#
#
# def test_conversion_pairwise_inverse(in_out_pair, order):
#     derivwise, coordwise = in_out_pair
#     as_coord = pnde.convert_derivwise_to_coordwise(derivwise, order)
#     as_deriv_again = pnde.convert_coordwise_to_derivwise(as_coord, order)
#     np.testing.assert_allclose(as_deriv_again, derivwise)
