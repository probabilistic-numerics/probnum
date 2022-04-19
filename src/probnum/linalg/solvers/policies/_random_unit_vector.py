"""Policy returning randomly drawn standard unit vectors."""

from typing import Optional

import numpy as np

import probnum  # pylint: disable="unused-import"

from . import _linear_solver_policy


class RandomUnitVectorPolicy(_linear_solver_policy.LinearSolverPolicy):
    r"""Policy returning randomly drawn standard unit vectors.

    Draw a standard unit vector :math:`e_i` at random and return it. This policy
    corresponds to selecting columns of the matrix as observations.

    Parameters
    ----------
    probabilities
        Probabilities with which to choose different columns. Either `uniform`
        or `rownorm`.
    replace
        Whether to sample with or without replacement.
    """

    def __init__(self, probabilities: str = "uniform", replace=True) -> None:
        if probabilities not in ["uniform", "rownorm"]:
            raise ValueError(
                f"Option '{probabilities}' for sampling probabilities not available."
            )
        self.probabilities = probabilities
        self.replace = replace

    def __call__(
        self,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:

        nrows = solver_state.problem.A.shape[0]

        # Initial sampling probabilities
        if not "row_sample_probs" in solver_state.cache.keys():
            if self.probabilities == "uniform":
                solver_state.cache["row_sample_probs"] = np.ones(nrows) / nrows
            elif self.probabilities == "rownorm":
                sampling_probs = (solver_state.problem.A**2).T @ np.ones(shape=nrows)
                solver_state.cache["row_sample_probs"] = (
                    sampling_probs / sampling_probs.sum()
                )
            else:
                raise NotImplementedError

        # Sample unit vector
        idx = rng.choice(
            a=nrows,
            size=1,
            p=solver_state.cache["row_sample_probs"],
        )
        action = np.zeros(nrows)
        action[idx] = 1.0

        # Handle replacement
        if not self.replace:
            solver_state.cache["row_sample_probs"][idx] = 0.0
            solver_state.cache["row_sample_probs"] /= solver_state.cache[
                "row_sample_probs"
            ].sum()

        return action
