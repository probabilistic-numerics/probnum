"""
If other line searches are created (wolfe conditions?),
this has to be redesigned. 

Note to devs
------------
LineSearch.next_lrate has a signature different to
solver.stepsize.suggest() which is due to different requirements of
optimizer and odesolvers. Both of them sharing a common interface would
be pointless as the interface would essentially be empty, only allowing
constant stepsizes which then would suffer from being called with
different signatures.
"""

from abc import ABC, abstractmethod

import numpy as np


class LineSearch(ABC):
    """
    """

    def __repr__(self):
        """
        """
        return "LineSearch instance"

    @abstractmethod
    def next_lrate(self, curriter, objective, direction, *pars, **namedpars):
        """
        Must return a learning rate!
        """
        raise NotImplementedError


class ConstantLearningRate(LineSearch):
    """
    """

    def __init__(self, lrate):
        """
        """
        if lrate <= 0:
            raise ValueError("Please enter a positive lrate.")
        self._lrate = lrate

    def next_lrate(self, curriter, objective, direction, *pars, **namedpars):
        """
        """
        return self._lrate


class BacktrackingLineSearch(LineSearch):
    """
    Slide 10 of:
    https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf
    """

    def __init__(self, initguess=1.0, reduction=0.75):
        """
        """
        if initguess <= 0.0:
            raise ValueError("initguess must be positive")
        if reduction <= 0.0 or reduction >= 1.0:
            raise ValueError("reduction must be in (0, 1)")
        self._initguess = initguess
        self._reduction = reduction

    def next_lrate(self, curriter, objective, direction, *pars, **namedpars):
        """
        curriter is an AutoDiffEval object (a namedtuple)
        objective is an AutoDiff object.
        direction is an np.ndarray object
        """
        if np.linalg.norm(direction) < 1e-20:
            raise ValueError("direction cannot have norm == 0")
        lrate_guess = self._initguess
        while self._condition(curriter, objective, lrate_guess,
                              direction) is False:
            lrate_guess = self._reduction * lrate_guess
        return lrate_guess

    def _condition(self, curriter, objective, lrate_guess, direction):
        """
        curriter is an AutoDiffEval object (a named tuple)
        """
        new_eval = objective.objective(curriter.x + lrate_guess * direction)
        thresh = curriter.fx + 0.5 * lrate_guess * curriter.dfx @ direction
        if new_eval > thresh:
            return False
        else:
            return True
