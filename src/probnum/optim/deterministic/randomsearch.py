"""
0th, 1st and 2nd order local (!!!!) minimisation.


Functionalities
---------------
* Random Search with constant learning rate and fixed number of steps
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

import numpy as np

from probnum.optim import objective, linesearch, optimizer


class RandomSearch(optimizer.Optimizer):
    """
    RandomSearch requires constant learning rate.

    Recommended: use RelativeTolerance(0.0), i.e. constant number of iterations,
    as proper RelativeTolerance might take ages.
    """

    def __init__(self, lsearch, stopcrit, maxit):
        """
        objective: Objective instance
        lsearch: ConstantLearningRate instance
        """
        if not issubclass(type(lsearch), linesearch.ConstantLearningRate):
            raise AttributeError("RandomSearch requires constant "
                                 "learning rates.")
        optimizer.Optimizer.__init__(self, lsearch, stopcrit, maxit)

    def iterate(self, curriter, objec):
        """
        """
        if curriter.x is None:
            raise ValueError("No state variable available")
        if curriter.fx is None:
            raise ValueError("No fct evaluation available")
        this_loc, objval = curriter.x, curriter.fx
        direction = np.random.randn(len(curriter.x))
        lrate = self.lsearch.next_lrate(curriter, objective, direction)
        newloc = this_loc + lrate * direction / np.linalg.norm(direction)
        newobjval = objec.objective(newloc)
        if newobjval <= objval:
            return objective.Eval(newloc, newobjval, 0., 0.)
        else:
            return curriter
