"""
State space models.

Graphical models and Continuous-discrete state spaces.
"""

class StateSpaceModel:
    """
    Interface for state space models.

    Some dynamics (a random process) plus some observation process.
    If a measurement model is not specified, fully observable systems
    are assumed.
    """
    def __init__(self, dynamics, measmodel=None):
        """ """
        self._dynamics = dynamics
        # make a measurement model with exact measurements
        self._measmodel = measmodel

    # ... some template methods: generate data, etc..
    # Good choice would be to make them abstractmethods I think...


class GraphicalModel(StateSpaceModel):
    """
    Graphical models.

    Discrete dynamics and discrete measurement transitions.
    """
    def __init__(self, dynamics, measmodel=None):
        """
        dynamics is a DiscreteProcess
        """
        super().__init__(dynamics, measmodel)


class ContinuousModel(StateSpaceModel):
    """
    Continuous-discrete state space models.

    Continuous dynamics and discrete measurement transitions.
    """
    def __init__(self, dynamics, measmodel=None):
        """
        dynamics is a ContinuousProcess
        """
        super().__init__(dynamics, measmodel)

