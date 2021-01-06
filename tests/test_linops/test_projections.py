"""Tests for projection operators."""

import unittest

import numpy as np

from probnum import linops
from tests.testing import NumpyAssertions


class OrthogonalProjectionTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for orthogonal projection operators."""
