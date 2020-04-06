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
Function parameters must be numpy.ndarrays. 
"""

from abc import ABC, abstractmethod

import numpy as np

from probnum import utils


class Optimizer(ABC):
    """
    Abstract optimiser class.

    An optimiser is an object
    with a minimise() method and a LineSearch attribute.
    """

    def __init__(self, lsearch, stopcrit, maxit):
        """
        lsearch: LineSearch instance
        """
        if maxit <= 0:
            raise ValueError("Maxit needs to be a positive integer")
        self.lsearch = lsearch
        self.stopcrit = stopcrit
        self.maxit = maxit

    def minimise_nd(self, objective, initval, *args, **kwargs):
        """
        objective: AutoDiff instance
        initval: ndarray

        Note
        ----
        Restrict to the nd case for algorithmic simplicity
        """
        utils.assert_is_1d_ndarray(initval)
        utils.assert_evaluates_to_scalar(objective.objective, initval)
        curriter = objective.evaluate(initval)
        lastiter = self.stopcrit.create_unfulfilled(curriter, *args, **kwargs)
        traj, objvals = self._make_traj_objvals(initval)
        count, traj[0], objvals[0] = 0, curriter.x, curriter.fx
        while not self._stop_now(curriter, lastiter, count):
            count = count + 1
            curriter = self.iterate(curriter, objective, *args, **kwargs)
            traj[count], objvals[count] = curriter.x, curriter.fx
        return traj[0:count + 1], objvals[0:count + 1]

    def _make_traj_objvals(self, initval):
        """
        Allocates storage for trajectory and realisations.
        """
        traj = np.zeros((self.maxit, len(initval)))  # quicker than .append()
        objvals = np.zeros(self.maxit)
        return traj, objvals

    def _stop_now(self, curriter, lastiter, count):
        """
        """
        stopcrit_yes = self.stopcrit.fulfilled(curriter, lastiter)
        maxit_yes = count >= self.maxit - 1
        return stopcrit_yes or maxit_yes

    @abstractmethod
    def iterate(self, curriter, objective, *args, **kwargs):
        """
        Needs to be implemented for minimise() to be complete.
        """
        raise NotImplementedError
