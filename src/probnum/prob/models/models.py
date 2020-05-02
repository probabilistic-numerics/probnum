"""
State space models.

Graphical models and Continuous-discrete state spaces.
"""

from probnum.prob.models.transitions import Transition, SDE


class ProbabilisticStateSpace:
    """
    Probabilistic state space models.

    Some dynamics (a transition rule, represented by a
    :class:`Transition` object) plus some observation model
    (another :class:`Transition` object).
    If an observation model is not specified, fully observable systems
    are assumed.

    Examples include graphical models, Markov random fields, SDE-driven
    dynamics and more.
    """
    def __init__(self, transition, initrv, measmodel=None, support=None):
        """ """
        # todo: some type checking, etc.
        if measmodel is None:
            measmodel = _ExactMeasurements()
        self._measmodel = measmodel
        self._transition = transition
        self._initrv = initrv
        self._support = support

    # Properties, getters and setters ##################################
    # Transition and measmodel are set, ################################
    # the rest can be altered. #########################################

    @property
    def measmodel(self):
        return self._measmodel

    @property
    def transition(self):
        return self._transition

    @property
    def initrv(self):
        return self._initrv

    @initrv.setter
    def initrv(self, randvar):
        # todo: some type checking
        self._initrv = randvar

    @property
    def support(self):
        return self._support

    @support.setter
    def support(self, support):
        # todo: some type checking
        self._support = support

    # Data generation ##################################################

    def gendata(self, *args):
        """Generate artificial data."""
        # todo: fill in arguments and stuff
        raise NotImplementedError


class _ExactMeasurements(Transition):
    """
    Exact measurements.
    """

    def forward(self, start, stop, value, **kwargs):
        """ """
        return value

    def condition(self, start, stop, randvar, **kwargs):
        """ """
        return randvar

