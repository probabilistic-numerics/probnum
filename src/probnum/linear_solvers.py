from abc import ABC, abstractmethod


class ProbabilisticLinearSolver(ABC):
    """
    Probabilistic linear solvers infer solutions to linear systems in a Bayesian framework.

    Probabilistic numerical linear solvers infer solutions to problems of the form

    .. math:: Ax^*=b,
    where :math:`A \\in \\mathbb{R}^{m \\times n}` and :math:`b \\in \\mathbb{R}^{m}`. They output a probability measure
    which quantifies uncertainty in the solution. 
    """

    @abstractmethod
    def solve(self, A, b, **kwargs):
        pass


