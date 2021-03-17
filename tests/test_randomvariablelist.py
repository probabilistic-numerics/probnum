import unittest

import numpy as np

from probnum._randomvariablelist import _RandomVariableList
from probnum.randvars import Constant


class TestRandomVariableList(unittest.TestCase):
    def setUp(self):
        self.rv_list = _RandomVariableList([Constant(0.1), Constant(0.2)])

    def test_inputs(self):
        """Inputs rejected or accepted according to expected types."""
        numpy_array = np.ones(3) * Constant(0.1)
        constant_list = [Constant(0.1), Constant(0.4)]
        number_list = [0.5, 0.41]
        inputs = [numpy_array, constant_list, number_list]
        inputs_acceptable = [False, True, False]

        for inputs, is_acceptable in zip(inputs, inputs_acceptable):
            with self.subTest(input=inputs, is_acceptable=is_acceptable):

                if is_acceptable:
                    _RandomVariableList(inputs)
                else:
                    with self.assertRaises(TypeError):
                        _RandomVariableList(inputs)

    def test_mean(self):
        mean = self.rv_list.mean
        self.assertEqual(mean.shape, (2,))

    def test_cov(self):
        cov = self.rv_list.cov
        self.assertEqual(cov.shape, (2,))

    def test_var(self):
        var = self.rv_list.var
        self.assertEqual(var.shape, (2,))

    def test_std(self):
        std = self.rv_list.std
        self.assertEqual(std.shape, (2,))

    def test_getitem(self):
        item = self.rv_list[0]
        self.assertIsInstance(item, Constant)

    def test_shape(self):
        self.assertEqual(self.rv_list.shape, (2,))


class TestEmptyRandomVariableList(unittest.TestCase):
    """Passing an empty list does not screw things up."""

    def setUp(self):
        self.rv_list = _RandomVariableList([])

    def test_mean(self):
        mean = self.rv_list.mean
        self.assertEqual(mean.shape, (0,))

    def test_cov(self):
        cov = self.rv_list.cov
        self.assertEqual(cov.shape, (0,))

    def test_var(self):
        var = self.rv_list.var
        self.assertEqual(var.shape, (0,))

    def test_std(self):
        std = self.rv_list.std
        self.assertEqual(std.shape, (0,))

    def test_getitem(self):
        with self.assertRaises(IndexError):
            item = self.rv_list[0]


if __name__ == "__main__":
    unittest.main()
