import functools
import unittest

import probnum.filtsmooth as pnfs

from . import filtsmooth_testcases as cases


class TestContinuousEKFComponent(cases.LinearisedContinuousTransitionTestCase):
    """Implementation incomplete, hence check that an error is raised."""

    def setUp(self):
        ekf_component = functools.partial(pnfs.ContinuousEKFComponent, num_steps=10)
        self.linearising_component_benes_daum = ekf_component
        self.visualise = False


class TestDiscreteEKFComponent(cases.LinearisedDiscreteTransitionTestCase):
    def setUp(self):
        self.linearising_component_pendulum = pnfs.DiscreteEKFComponent
        self.linearising_component_car = pnfs.DiscreteEKFComponent
        self.visualise = False
