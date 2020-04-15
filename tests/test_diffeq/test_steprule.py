"""

"""
from probnum.diffeq import steprule
import unittest
import numpy as np

np.random.seed(75468)


class TestConstantStep(unittest.TestCase):
    """
    Check that step is always the same
    and that is_accepted is always true.
    """
    def setUp(self):
        """
        """
        self.step = np.random.rand()
        self.sr = steprule.ConstantSteps(self.step)

    def test_suggest(self):
        """
        """
        stp = self.sr.suggest(self.step, np.nan)
        self.assertEqual(stp, self.step)

    def test_is_accepted(self):
        """
        """
        isacc = self.sr.is_accepted(np.inf, np.nan)
        self.assertEqual(isacc, True)


class TestAdaptiveStep(unittest.TestCase):
    """
    We pretend that we have a solver of local error rate
    three and see if steps are proposed accordingly.
    """
    def setUp(self):
        """
        Set up imaginative solver of convergence rate 3.
        That is,
        """
        self.tol = 1e-4
        self.asr = steprule.AdaptiveSteps(self.tol, 3)

    def test_is_accepted(self):
        """
        """
        suggstep = np.random.rand()
        errorest = suggstep**3 / 3
        self.assertEqual(self.asr.is_accepted(suggstep, errorest), False)

    def test_propose(self):
        """
        """
        step = 0.25 * np.random.rand()
        errorest = step
        sugg = self.asr.suggest(step, errorest)
        err = sugg**3 / 3
        self.assertEqual(self.asr.is_accepted(sugg, err), True)
