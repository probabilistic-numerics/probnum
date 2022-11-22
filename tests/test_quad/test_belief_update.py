"""Test cases for the BQ belief updater."""

from probnum.quad.solvers.belief_updates import BQStandardBeliefUpdate

import pytest


def test_belief_update_raises():
    # negative jitter is not allowed
    wrong_jitter = -1.0
    with pytest.raises(ValueError):
        BQStandardBeliefUpdate(jitter=wrong_jitter, scale_estimation="mle")
