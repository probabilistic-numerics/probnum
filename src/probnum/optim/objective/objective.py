"""
Objective function class (functions with gradients and Hessians)
as well as a corresponding evaluation data structure.

The evaluation data structure is used for Metropolis-Hastings
sampling, too.
"""

from collections import namedtuple

import numpy as np


class Objective:
    """
    Functions with vector-valued input and scalar output
    as well as their derivatives.

    Not sure why, but I think implementing dunder methods
    here is tedious (hence a waste of time). I will postpone
    this until it is actually important!

    There is no law in the world (i.e. this module) that
    dictates that the gradient and hessian have to be
    the "proper" gradients and hessians. They may be anything,
    and I'm especially thinking about approximate gradients. 
    """

    def __init__(self, obj, grad=None, hess=None):
        """
        """
        self._obj = obj
        self._grad = grad
        self._hess = hess

    def evaluate(self, evalhere, *args, **kwargs):
        """
        Evaluates all and returns an Eval(...) object.
        """
        objval = self._obj(evalhere, *args, **kwargs)
        if self._grad is not None:
            gradval = self._grad(evalhere, *args, **kwargs)
        else:
            gradval = None
        if self._hess is not None:
            hessval = self._hess(evalhere, *args, **kwargs)
        else:
            hessval = None
        return Eval(evalhere, objval, gradval, hessval)

    def objective(self, evalhere, *args, **kwargs):
        """
        Evaluates just the objective function.
        """
        return self._obj(evalhere, *args, **kwargs)

    def gradient(self, evalhere, *args, **kwargs):
        """
        Evaluates just the gradient.
        """
        if self._grad is not None:
            return self._grad(evalhere, *args, **kwargs)
        else:
            raise NotImplementedError("Gradient not available.")

    def hessian(self, evalhere, *args, **kwargs):
        """
        Evaluates just the hessian.
        """
        if self._hess is not None:
            return self._hess(evalhere, *args, **kwargs)
        else:
            raise NotImplementedError("Hessian not available.")


class Eval(namedtuple('ObjEval', 'x fx dfx ddfx')):
    """
    Evaluations of scalar-valued functions with n-dimensional inputs.

    We want states to be glued together with evalautions of e.g.
    probability densitiy functions and gradients.
    This data structure is used almost everywhere, especially in the metropolishastings
    and optimiser modules.

    We provide _add__, __mult__ and other fundamental methods
    for the namedtuples-based evaluation data structure, however,
    this is not used in the toolkit implementation.

    If you don't use the dunder methods, you will get
    away with ignoring the attribute-documentation below.
    However, this is how we expect it to be used:

    Attributes
    ----------
    x: np.ndarray, shape (m,)
        Evaluation f(x)
    fx: float
        Evaluation f(x)
    dfx: np.ndarray, shape (m,)
        Evaluation of the gradient, \\nabla f(x)
    ddfx: np.ndarray, shape (m, m)
        Evaluation of the Hessian, \\nabla^2 f(x)
    """

    def __add__(left, right):
        """
        Only supported if f_1 and f_2 are evaluated at the same x.
        """
        if np.isscalar(right):
            return Eval(left.x, left.fx + right, left.dfx, left.ddfx)
        elif issubclass(type(right), Eval):
            if np.linalg.norm(left.x - right.x) != 0:
                raise TypeError("Operation not supported")
            return Eval(left.x,
                        left.fx + right.fx,
                        left.dfx + right.dfx,
                        left.ddfx + right.ddfx)
        else:
            raise TypeError("Operation not supported")

    def __mul__(left, right):
        """
        Scalar optimiser can be done via:
        d * Eval(a, b, c) = Eval(d, 0, 0) * Eval(a, b, c)
                          = Eval(d*a, d+b, d*c)
        """
        if np.isscalar(right):
            return left * Eval(left.x, right, 0.0 * left.dfx, 0.0 * left.ddfx)
        elif issubclass(type(right), Eval):
            if np.linalg.norm(left.x - right.x) != 0:
                raise TypeError("Operation not supported")
            return Eval(left.x,
                        left.fx * right.fx,
                        left.dfx * right.fx + left.fx * right.dfx,
                        left.ddfx * right.fx + 2 * left.dfx @ (right.dfx).T \
                        + left.fx * right.ddfx)
        else:
            raise TypeError("Operation not supported")

    __radd__ = __add__
    __rmul__ = __mul__

    def __truediv__(left, right):
        """
        """
        if not issubclass(type(right), Eval):
            raise TypeError("Operation not supported")
        if np.linalg.norm(left.x - right.x) != 0:
            raise TypeError("Operation not supported")
        return Eval(left.x,
                    left.fx / right.fx,
                    (left.dfx * right.fx - left.fx * right.dfx) / (
                            right.fx ** 2),
                    (right.fx ** 2 * left.ddfx - right.fx \
                     * (2 * left.dfx @ (right.dfx).T + left.fx * right.ddfx) \
                     + 2 * left.fx * right.dfx @ (right.dfx).T) \
                    / (right.fx ** 3))
