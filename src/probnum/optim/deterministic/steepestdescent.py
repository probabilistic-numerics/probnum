"""
0th, 1st and 2nd order local (!!!!) minimisation.


Functionalities
---------------
* Random Search with constant learning rate" and fixed number of steps
* GD, Newton, damped Newton (levenberg marquardt)
    * with stopping criterions: absolute tolerance, relative tolerance, maxit
    * with learning rates: constant, backtracking linesearch


Note
----
Out iterations are x -> x + lrate * direction
And directions are e.g. negative gradients.

Consider
--------
Make states ndimensional, to be able to cleanly call len(), .ndim, etc.
"""

from abc import abstractmethod

import numpy as np

from probnum.optim.optimizer import Optimizer


class SteepestDescent(Optimizer):
    """
    """

    def iterate(self, curriter, objec, *args, **kwargs):
        """
        """
        direction = self.compute_direction(curriter, *args, **kwargs)
        lrate = self.lsearch.next_lrate(curriter, objec, direction)
        new_iterate = curriter.x + lrate * direction
        return objec.evaluate(new_iterate)

    @abstractmethod
    def compute_direction(self, curriter, *args, **kwargs):
        """
        """
        raise NotImplementedError


class GradientDescent(SteepestDescent):
    """
    """

    def compute_direction(self, curriter, *args, **kwargs):
        """
        """
        if curriter.dfx is None:
            raise ValueError("No gradient available.")
        return -curriter.dfx


class NewtonMethod(SteepestDescent):
    """
    """

    def compute_direction(self, curriter, *args, **kwargs):
        """
        """
        if curriter.dfx is None:
            raise ValueError("No gradient available.")
        if curriter.ddfx is None:
            raise ValueError("No Hessian available.")
        grad, hess = curriter.dfx, curriter.ddfx
        return -np.linalg.solve(hess, grad)


class LevenbergMarquardt(SteepestDescent):
    """
    Levenberg-Marquardt with constant damping!

    For nonconstant damping, change dampingpar into a DampingRule
    class which computes the damping parameters. 
    Only do this if it is relevant, otherwise it will be unnecessary
    functionality.
    """

    def __init__(self, dampingpar, lsearch, stopcrit, maxit):
        """
        lsearch: LineSearch instance
        """
        if dampingpar < 0:
            raise ValueError("Damping parameter cannot be negative")
        elif dampingpar == 0:
            raise ValueError("dampingpar=0 is NewtonMethod(). Please use it.")
        self.dampingpar = dampingpar
        SteepestDescent.__init__(self, lsearch, stopcrit, maxit)

    def compute_direction(self, curriter, *args, **kwargs):
        """
        """
        if curriter.dfx is None:
            raise ValueError("No gradient available.")
        if curriter.ddfx is None:
            raise ValueError("No Hessian available.")
        grad, hess = curriter.dfx, curriter.ddfx
        damphess = hess + self.dampingpar * np.eye(hess.shape[0],
                                                   hess.shape[1])
        return -np.linalg.solve(damphess, grad)
