import unittest

import probnum.filtsmooth as pnfs


class TestDefaultStoppingCriterion(unittest.TestCase):
    """The default stoppingcriterion should make sure that no filter updates are repeated but also
    make sure that whenever iterated filtsmooth is attempted, an exception is thrown.
    """

    def setUp(self):
        """"""
        self.stopcrit = pnfs.StoppingCriterion()

    def test_continue_filter_updates(self):
        self.assertFalse(self.stopcrit.continue_filter_updates())

    def test_continue_filtsmooth_updates(self):
        with self.assertRaises(NotImplementedError):
            self.assertTrue(self.stopcrit.continue_filtsmooth_updates())


if __name__ == "__main__":
    unittest.main()
