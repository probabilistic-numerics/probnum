"""
State space models.

Graphical models and Continuous-discrete state spaces.
"""

from probnum.prob.randomprocess import Transition, SDE


class StateSpaceModel:
    """
    State space models.

    Some dynamics (a random process) plus some observation process.
    If a measurement model is not specified, fully observable systems
    are assumed.
    """
    def __init__(self, randproc, measmodel=None):
        """ """
        self._randproc = randproc
        # make a measurement model with exact measurements
        if measmodel is None:
            measmodel = _ExactMeasurements(randproc)
        self._measmodel = measmodel

    # ... todo: some template methods: generate data, etc..


class GraphicalModel(StateSpaceModel):
    """
    Discrete-discrete state space models.

    State space models with an additional type check at initialization.
    """
    def __init__(self, randproc, measmodel=None):
        """ """
        # todo: type check if randproc.transition has discrete support
        if measmodel is not None:
            # todo: assert that both supports are identical
            pass
        super().__init__(randproc, measmodel)


class ContinuousDiscreteModel:
    """
    Continuous-discrete state space models.

    State space models with an additional type check at initialization.
    """
    def __init__(self, randproc, measmodel=None):
        """ """
        if not issubclass(type(randproc.transition), SDE):
            raise ValueError("Continuous dynamics need (continuous) "
                             "SDE transitions.")
        super().__init__(randproc, measmodel)


class _ExactMeasurements(Transition):
    """
    Exact measurements.
    """

    def __init__(self, randproc):
        super().__init__(support=randproc.transition.support)

    def forward(self, start, stop, value, **kwargs):
        """ """
        return value

    def condition(self, start, stop, randvar, **kwargs):
        """ """
        return randvar

