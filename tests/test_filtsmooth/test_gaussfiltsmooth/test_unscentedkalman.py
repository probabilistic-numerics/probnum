import unittest

import probnum.filtsmooth as pnfs

from . import filtsmooth_testcases as cases
import functools


class TestContinuousUKFComponent(unittest.TestCase):
    """Implementation incomplete, hence check that an error is raised."""

    def test_notimplementederror(self):
        sde = pnfs.statespace.SDE(None, None, None)  # content is irrelevant.
        dummy_for_dimension = 1
        with self.assertRaises(NotImplementedError):
            pnfs.ContinuousUKFComponent(sde, dimension=dummy_for_dimension)


class TestDiscreteUKFComponent(cases.LinearisedDiscreteTransitionTestCase):
    def setUp(self):
        self.linearising_component_pendulum = functools.partial(
            pnfs.DiscreteUKFComponent, dimension=2
        )
        self.linearising_component_car = functools.partial(
            pnfs.DiscreteUKFComponent, dimension=4
        )
        self.visualise = False
