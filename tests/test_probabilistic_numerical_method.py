"""Tests for an implementation of a probabilistic numerical method."""

import unittest

from probnum import ProbabilisticNumericalMethod


class PNMethodTestCase(unittest.TestCase):
    """Test case for a generic implementation of a probabilistic numerical method."""

    def test_cannot_instantiate(self):
        """Tests whether the base class cannot be instantiated."""
        with self.assertRaises(TypeError):
            _ = ProbabilisticNumericalMethod(
                prior=None,
                action_rule=None,
                observe=None,
                update_belief=None,
            )
