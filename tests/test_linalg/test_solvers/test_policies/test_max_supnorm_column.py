"""Tests for the maximum column supremum norm policy."""

import numpy as np

from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.linalg.solvers.policies import Policy
from probnum.problems import LinearSystem


def test_selects_max_supnorm_column(
    linsys_spd: LinearSystem, prior: LinearSystemBelief, maxsupnormcol_policy: Policy
):
    """Test whether the policy selects the column with maximum absolute value."""

    maxcol_idx = np.argmax(np.amax(np.absolute(linsys_spd.A), axis=0))
    action = maxsupnormcol_policy(problem=linsys_spd, belief=prior)

    assert np.argwhere((action.actA != 0.0) & (action.actA == 1.0))[0][0] == maxcol_idx
