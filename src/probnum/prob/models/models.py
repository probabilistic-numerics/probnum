"""
State space models.

Graphical models and Continuous-discrete state spaces.
"""

class StateSpaceModel:
    """
    Interface for state space models.

    Some dynamics (a random process) plus some observation process
    """
    def __init__(self, dynamics, measmodel):
        """ """
        self._dynamics = dynamics
        self._measmodel = measmodel

    # ... some template methods: generate data, etc..
    # Good choice would be to make them abstractmethods I think...

class GraphicalModel(StateSpaceModel):
    """
    Graphical models.

    Discrete dynamics and discrete measurement transitions.
    """
    def __init__(self, dynamics, measmodel):
        """
        dynamics is a DiscreteProcess
        """
        super().__init__(dynamics, measmodel)


class ContinuousModel(StateSpaceModel):
    """
    Continuous-discrete state space models.

    Continuous dynamics and discrete measurement transitions.
    """
    def __init__(self, dynamics, measmodel):
        """
        dynamics is a ContinuousProcess
        """
        super().__init__(dynamics, measmodel)

