import unittest

import probnum.filtsmooth as pnfs
import probnum.statespace as pnss

from . import filtsmooth_testcases as cases


class TestContinuousUKFComponent(unittest.TestCase):
    """Implementation incomplete, hence check that an error is raised."""

    def test_notimplementederror(self):
        sde = pnss.SDE(1, None, None, None)  # content is irrelevant.
        with self.assertRaises(NotImplementedError):
            pnfs.ContinuousUKFComponent(sde)


class TestDiscreteUKFComponent(cases.LinearisedDiscreteTransitionTestCase):
    def setUp(self):
        self.linearising_component_pendulum = pnfs.DiscreteUKFComponent
        self.linearising_component_car = pnfs.DiscreteUKFComponent
        self.visualise = False
