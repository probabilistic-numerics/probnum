import unittest

import numpy as np

from probnum.utils import randomutils
from tests.testing import NumpyAssertions


class RandomUtilsTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for utility functions handling objects dealing with randomness."""

    def setUp(self) -> None:
        self.seed = 42
        self.random_state = np.random.RandomState(seed=self.seed)
        self.random_generator = np.random.default_rng(seed=self.seed)
        self.random_generator_list = [np.random.default_rng(seed=s) for s in range(5)]

    def test_derive_random_seed_invariant_random_state(self):
        """Test whether deriving a random seed leaves the original random states
        untouched."""
        # Original random states
        rs_state = self.random_state.get_state()[1]
        rng_state = self.random_generator.bit_generator.state["state"]["state"]
        rng_list_states = [
            rng.bit_generator.state["state"]["state"]
            for rng in self.random_generator_list
        ]

        # Combine RandomState and Generator object
        _ = randomutils.derive_random_seed(self.random_state, self.random_generator)
        self.assertArrayEqual(rs_state, self.random_state.get_state()[1])
        self.assertEqual(
            rng_state, self.random_generator.bit_generator.state["state"]["state"]
        )

        # Combine list of generators
        _ = randomutils.derive_random_seed(*self.random_generator_list)
        self.assertArrayEqual(
            rng_list_states,
            [
                rng.bit_generator.state["state"]["state"]
                for rng in self.random_generator_list
            ],
        )
