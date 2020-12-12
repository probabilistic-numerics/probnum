"""Probabilistic Linear Solvers.

Iterative probabilistic numerical methods solving linear systems :math:`Ax_* = b`.
"""

import numpy as np

from probnum import ProbabilisticNumericalMethod


class ProbabilisticLinearSolver(ProbabilisticNumericalMethod):
    def __init__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def solve(self, linear_system: LinearSystem):
        raise NotImplementedError
