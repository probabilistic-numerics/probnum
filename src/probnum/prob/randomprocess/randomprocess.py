"""
Random / Stochastic processes.

This module implements classes and functions representing random processes, i.e. families of random variables.
"""


class RandomProcess:
    """
    Random process.

    Random processes are collections of random variables, often representing numerical values of some system randomly
    changing over time.

    Parameters
    ----------
    """

    def __init__(self):
        """Create a new random process."""

    def mean(self):
        """
        Mean (function) of the random process.
        """
        raise NotImplementedError

    def cov(self):
        """
        Covariance (function) of the random process, sometimes known as kernel.
        """
        raise NotImplementedError
