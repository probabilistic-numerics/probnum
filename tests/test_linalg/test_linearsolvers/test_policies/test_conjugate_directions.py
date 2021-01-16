"""Tests for the conjugate directions policy."""

import pytest

from probnum.linalg.linearsolvers.policies import ConjugateDirections


def test_directions_are_conjugate(policy):
    """Test whether the actions given by the ConjugateDirections policy are
    A-conjugate."""
    # TODO: use ProbabilisticLinearSolver's solve_iter function to test this
