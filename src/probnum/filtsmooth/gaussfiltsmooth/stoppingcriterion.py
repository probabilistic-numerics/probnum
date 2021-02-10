"""Stopping criteria for iterated filtering and smoothing."""
import numpy as np


class StoppingCriterion:
    """Stop iteration if absolute and relative tolerance are reached."""

    def __init__(self, atol=1e-3, rtol=1e-6, maxit=1000):
        self.atol = atol
        self.rtol = rtol
        self.maxit = maxit
        self.iterations = 0

    def terminate(self, error, reference):
        """Decide whether the stopping criterion is satisfied, which implies terminating
        of the iteration.

        If the error is sufficiently small (with respect to atol, rtol
        and the reference), return True. Else, return False. Throw a
        runtime error if the maximum number of iterations is reached.
        """
        if self.iterations > self.maxit:
            errormsg = f"Maximum number of iterations (N={self.maxit}) reached."
            raise RuntimeError(errormsg)
        magnitude = self.evaluate_error(error=error, reference=reference)
        if magnitude > 1:
            self.iterations += 1
            return False
        else:
            self.iterations = 0
            return True

    def evaluate_error(self, error, reference):
        """Compute the normalised error."""
        normalisation = self.atol + self.rtol * reference
        magnitude = np.sqrt(np.mean((error / normalisation) ** 2))
        return magnitude
