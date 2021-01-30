"""Stopping criteria for iterated filtering and smoothing."""
import numpy as np


class StoppingCriterion:
    """Stop iteration if absolute and relative tolerance are reached."""

    def __init__(self, atol=1e-3, rtol=1e-6, maxit=1000):
        self.atol = atol
        self.rtol = rtol
        self.maxit = maxit
        self.iterations = 0

    def do_not_terminate_yet(self, error, reference):
        if self.iterations > self.maxit:
            errormsg = f"Maximum number of iterations (N={self.maxit}) reached."
            raise RuntimeError(errormsg)
        magnitude = self.evaluate_error(error=error, reference=reference)
        if magnitude > 1:
            self.iterations += 1
            return True
        else:
            self.iterations = 0
            return False

    def evaluate_error(self, error, reference):
        normalisation = self.atol + self.rtol * reference
        magnitude = np.sqrt(np.mean((error / normalisation) ** 2))
        return magnitude
