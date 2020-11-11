import unittest

import probnum.filtsmooth as pnfs
import probnum.random_variables as pnrv


class TestDefaultStoppingCriterion(unittest.TestCase):
    """The default stoppingcriterion should make sure that no filter updates are repeated but also
    make sure that whenever iterated filtsmooth is attempted, an exception is thrown.
    """

    def setUp(self):
        self.stopcrit = pnfs.StoppingCriterion()

    def test_continue_filter_updates(self):
        self.assertFalse(self.stopcrit.continue_filter_updates())

    def test_continue_filtsmooth_updates(self):
        with self.assertRaises(NotImplementedError):
            self.assertTrue(self.stopcrit.continue_filtsmooth_updates())


class TestFixedPointIteration(unittest.TestCase):
    def setUp(self):
        self.stopcrit = pnfs.FixedPointStopping(atol=1e-4, rtol=1e-4, max_num_filter_updates=10)

    def test_continue_filter_updates(self):
        self.assertEqual(self.stopcrit.num_filter_updates, 0)
        x0 = 1.0
        while self.stopcrit.continue_filter_updates(filt_rv=pnrv.Constant(x0)):
            x0 *= 0.1
        self.assertGreaterEqual(self.stopcrit.num_filter_updates, 1)

    def test_continue_filter_updates_exception(self):
        """No improvement at all raises error eventually"""
        worsening = 0.1
        value = 0.
        with self.assertRaises(RuntimeError):
            while self.stopcrit.continue_filter_updates(filt_rv=pnrv.Constant(value)):
                value += worsening


if __name__ == "__main__":
    unittest.main()
