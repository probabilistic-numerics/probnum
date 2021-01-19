"""Tests for the noisy linear system belief."""

import numpy as np

from probnum.linalg.solvers.beliefs import NoisySymmetricNormalLinearSystemBelief
from probnum.problems import LinearSystem


def test_from_solution_noisy_system(
    linsys_noise: LinearSystem, random_state: np.random.RandomState
):
    """Test whether a belief can be constructed from a solution and a noisy linear
    system."""
    x0 = random_state.normal(size=linsys_noise.A.shape[1])
    NoisySymmetricNormalLinearSystemBelief.from_solution(x0=x0, problem=linsys_noise)
