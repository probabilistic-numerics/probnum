"""Test cases for linear system beliefs."""

import numpy as np

from probnum.linalg.linearsolvers.beliefs import (
    LinearSystemBelief,
    WeakMeanCorrespondence,
)
from probnum.problems import LinearSystem
from tests.testing import NumpyAssertions

# pylint: disable="invalid-name"


class LinearSystemBeliefTestCase(NumpyAssertions):
    """General test case for linear system beliefs."""

    def setUp(self) -> None:
        """Test resources for linear system beliefs."""
