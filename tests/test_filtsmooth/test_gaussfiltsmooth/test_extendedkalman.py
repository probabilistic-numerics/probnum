import probnum.filtsmooth as pnfs

from . import filtsmooth_testcases as cases


class TestContinuousEKFComponent(cases.LinearisedContinuousTransitionTestCase):
    """Implementation incomplete, hence check that an error is raised."""

    def setUp(self):
        self.linearising_component_benes_daum = pnfs.ContinuousEKFComponent
        self.visualise = False


class TestDiscreteEKFComponent(cases.LinearisedDiscreteTransitionTestCase):
    def setUp(self):
        self.linearising_component_pendulum = pnfs.DiscreteEKFComponent
        self.linearising_component_car = pnfs.DiscreteEKFComponent
        self.visualise = False
