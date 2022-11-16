"""Test cases for the BQ belief updater."""

import pytest

from probnum.quad.solvers.belief_updates import BQBeliefUpdate


def test_belief_update_raises():
    # negative jitter is not allowed
    wrong_jitter = -1.0
    with pytest.raises(ValueError):
        BQBeliefUpdate(jitter=wrong_jitter)
