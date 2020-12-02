import unittest

import numpy as np

from probnum.diffeq import steprule

random_state = np.random.mtrand.RandomState(seed=1234)


class TestConstantStep(unittest.TestCase):
    """Check that step is always the same and that is_accepted is always true."""

    def setUp(self):
        self.step = random_state.rand()
        self.sr = steprule.ConstantSteps(self.step)

    def test_suggest(self):
        stp = self.sr.suggest(self.step, np.nan)
        self.assertEqual(stp, self.step)

    def test_is_accepted(self):
        isacc = self.sr.is_accepted(np.inf, np.nan)
        self.assertEqual(isacc, True)


# The tests below will need reconsideration
# since Adaptive Steps are different now...
class TestAdaptiveStep(unittest.TestCase):
    """We pretend that we have a solver of local error rate three and see if steps are
    proposed accordingly."""

    def setUp(self):
        """Set up imaginative solver of convergence rate 3."""
        self.asr = steprule.AdaptiveSteps(firststep=1.0)

    def test_is_accepted(self):
        errorest = 0.5  # < 1, should be accepted
        suggstep = np.nan  # value should not matter
        localconvrate = None  # should not matter either
        self.assertEqual(
            self.asr.is_accepted(suggstep, errorest, localconvrate=localconvrate), True
        )

    def test_suggest(self):
        """If errorest <1, the next step should be larger."""
        step = 0.55 * random_state.rand()
        errorest = 0.75
        sugg = self.asr.suggest(step, errorest, localconvrate=3)
        self.assertGreater(sugg, step)

    def test_errorest_to_internalnorm_1d(self):
        errorest = 0.5
        proposed_state = np.array(1.0)
        current_state = np.array(2.0)
        atol = 0.1
        rtol = 0.01
        expected = errorest / (atol + rtol * np.maximum(proposed_state, current_state))
        received = self.asr.errorest_to_internalnorm(
            errorest, proposed_state, current_state, atol, rtol
        )
        self.assertAlmostEqual(expected, received)

    def test_errorest_to_internalnorm_2d(self):
        errorest = np.array([0.1, 0.2])
        proposed_state = np.array([1.0, 3.0])
        current_state = np.array([2.0, 3.0])
        atol = 0.1
        rtol = 0.01
        expected = np.linalg.norm(
            errorest / (atol + rtol * np.maximum(proposed_state, current_state))
        ) / np.sqrt(2)
        received = self.asr.errorest_to_internalnorm(
            errorest, proposed_state, current_state, atol, rtol
        )
        self.assertAlmostEqual(expected, received)
