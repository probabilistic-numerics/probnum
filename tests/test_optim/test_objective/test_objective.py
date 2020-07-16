"""
"""

import unittest

import numpy as np

from probnum.optim import Objective, Eval


class TestObjective(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """

        def fct(x):
            return 5.225 * x.T @ x - 4.231

        def grad(x):
            return 5.225 * 2.0 * x

        def hess(x):
            return 5.225 * 2.0 * np.ones((len(x), len(x)))

        self.obj = Objective(fct, grad, hess)

    def test_evaluate(self):
        """
        Successful evaluation and creation of an Eval(...) object.
        """
        xval = np.random.rand(2)
        evl = self.obj.evaluate(xval)
        self.assertEqual(issubclass(type(evl), Eval), True)

    def test_consistency(self):
        """
        Entries of the Eval() object are the evaluations of grad, etc.
        """
        xval = np.random.rand(2)
        evl = self.obj.evaluate(xval)
        diff_fx = np.linalg.norm(evl.fx - self.obj.objective(xval))
        diff_dfx = np.linalg.norm(evl.dfx - self.obj.gradient(xval))
        diff_ddfx = np.linalg.norm(evl.ddfx - self.obj.hessian(xval))
        self.assertLess(diff_fx, 1e-14)
        self.assertLess(diff_dfx, 1e-14)
        self.assertLess(diff_ddfx, 1e-14)

    def test_call_only_obj(self):
        """
        One may initialise Objective with only a function and no grad
        nor hess.
        """

        def fct(x):
            return 5.225 * x.T @ x - 4.231

        obj = Objective(fct)
        obj.evaluate(np.random.rand(2))
        obj.objective(np.random.rand(2))
        with self.assertRaises(NotImplementedError):
            obj.gradient(np.random.rand(2))
        with self.assertRaises(NotImplementedError):
            obj.hessian(np.random.rand(2))

    def test_call_only_obj_and_grad(self):
        """
        One may initialise Objective with only a function
        and grad, no hess.
        """

        def fct(x):
            return 5.225 * x.T @ x - 4.231

        def grad(x):
            return 5.225 * 2.0 * x

        obj = Objective(fct, grad)
        obj.evaluate(np.random.rand(2))
        obj.objective(np.random.rand(2))
        obj.gradient(np.random.rand(2))
        with self.assertRaises(NotImplementedError):
            obj.hessian(np.random.rand(2))


class TestEval(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        TestCase: evaluation of f_1(x) = 5.225*||x||^2 - 4.231
        at random x
        """

        def fct(x):
            return 5.225 * x.T @ x - 4.231

        def grad(x):
            return 5.225 * 2.0 * x

        def hess(x):
            return 5.225 * 2.0 * np.ones((len(x), len(x)))

        xval = np.random.rand(2)
        self.obj = Objective(fct, grad, hess)
        self.eval = self.obj.evaluate(xval)

    def test_scalaradd_right(self):
        """
        """
        scal = np.random.rand()
        evalsum = self.eval + scal
        self.assertEqual(np.linalg.norm(self.eval.x - evalsum.x), 0.0)
        self.assertLess(np.linalg.norm(self.eval.fx + scal - evalsum.fx),
                        1e-14)
        self.assertLess(np.linalg.norm(self.eval.dfx - evalsum.dfx), 1e-14)
        self.assertLess(np.linalg.norm(self.eval.ddfx - evalsum.ddfx), 1e-14)

    def test_scalaradd_left(self):
        """
        """
        scal = np.random.rand()
        evalsum = scal + self.eval
        self.assertEqual(np.linalg.norm(self.eval.x - evalsum.x), 0.0)
        self.assertLess(np.linalg.norm(self.eval.fx + scal - evalsum.fx),
                        1e-14)
        self.assertLess(np.linalg.norm(self.eval.dfx - evalsum.dfx), 1e-14)
        self.assertLess(np.linalg.norm(self.eval.ddfx - evalsum.ddfx), 1e-14)

    def test_add_obj(self):
        """
        """
        scal = np.random.rand()  # make it less predictable
        evalsum = self.eval + scal * self.eval
        self.assertEqual(np.linalg.norm(self.eval.x - evalsum.x), 0.0)
        self.assertLess(
            np.linalg.norm(self.eval.fx + scal * self.eval.fx - evalsum.fx),
            1e-14)
        self.assertLess(
            np.linalg.norm(self.eval.dfx + scal * self.eval.dfx - evalsum.dfx),
            1e-14)
        self.assertLess(np.linalg.norm(
            self.eval.ddfx + scal * self.eval.ddfx - evalsum.ddfx), 1e-14)

    def test_scalarmul_left(self):
        """
        """
        scal = np.random.rand()
        evalsum = scal * self.eval
        self.assertEqual(np.linalg.norm(self.eval.x - evalsum.x), 0.0)
        self.assertLess(np.linalg.norm(scal * self.eval.fx - evalsum.fx),
                        1e-14)
        self.assertLess(np.linalg.norm(scal * self.eval.dfx - evalsum.dfx),
                        1e-14)
        self.assertLess(np.linalg.norm(scal * self.eval.ddfx - evalsum.ddfx),
                        1e-14)

    def test_scalarmul_right(self):
        """
        """
        scal = np.random.rand()
        evalsum = self.eval * scal
        self.assertEqual(np.linalg.norm(self.eval.x - evalsum.x), 0.0)
        self.assertLess(np.linalg.norm(self.eval.fx * scal - evalsum.fx),
                        1e-14)
        self.assertLess(np.linalg.norm(self.eval.dfx * scal - evalsum.dfx),
                        1e-14)
        self.assertLess(np.linalg.norm(self.eval.ddfx * scal - evalsum.ddfx),
                        1e-14)

    def test_mul_right(self):
        """
        """
        scal = np.random.rand()
        otherobj = self.eval * scal
        evalsum = self.eval * otherobj
        diff_obj = np.linalg.norm(self.eval.fx * otherobj.fx - evalsum.fx)
        diff_grad = np.linalg.norm(self.eval.fx * otherobj.dfx \
                                   + self.eval.dfx * otherobj.fx \
                                   - evalsum.dfx)
        diff_hess = np.linalg.norm(self.eval.fx * otherobj.ddfx \
                                   + 2 * self.eval.dfx @ otherobj.dfx.T \
                                   + self.eval.ddfx * otherobj.fx \
                                   - evalsum.ddfx)
        self.assertLess(diff_obj, 1e-13)
        self.assertLess(diff_grad, 1e-13)
        self.assertLess(diff_hess, 1e-13)

    def test_truediv(self):
        """
        CAREFUL
        -------
        Test whether f(x) / (a * f(x)) yields (1/a) in terms
        of Eval(...)
        This test may be a bit lazy, however, something more elaborate
        seems to be a waste of time to me. 
        """
        scal = 0.1 * np.random.rand() + 0.5
        otherobj = self.eval * scal
        evalsum = self.eval / otherobj
        diff_obj = np.linalg.norm(1.0 / scal - evalsum.fx)
        diff_grad = np.linalg.norm(0.0 - evalsum.dfx)
        diff_hess = np.linalg.norm(0.0 - evalsum.ddfx)
        self.assertLess(diff_obj, 1e-10)
        self.assertLess(diff_grad, 1e-10)
        self.assertLess(diff_hess, 1e-10)

    def test_diff_x(self):
        """
        """
        with self.assertRaises(TypeError):
            self.eval + self.obj.evaluate(np.random.rand(2))

        with self.assertRaises(TypeError):
            self.obj.evaluate(np.random.rand(2)) + self.eval

        with self.assertRaises(TypeError):
            self.eval * self.obj.evaluate(np.random.rand(2))

        with self.assertRaises(TypeError):
            self.obj.evaluate(np.random.rand(2)) * self.eval

        with self.assertRaises(TypeError):
            self.eval / self.obj.evaluate(np.random.rand(2))

        with self.assertRaises(TypeError):
            self.obj.evaluate(np.random.rand(2)) / self.eval
