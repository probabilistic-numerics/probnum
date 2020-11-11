import unittest

import probnum.filtsmooth as pnfs

class MyTestCase(unittest.TestCase):
    def test_something(self):
        pnfs.StoppingCriterion()
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
