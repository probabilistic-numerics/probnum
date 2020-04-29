
import numpy as np
import scipy.spatial as scs

from probnum.prob.randomprocess.covariance import covariance
from probnum.utils import *


__all__ = ["IntegratedBrownianMotion", "BrownianMotion"]


class IntegratedBrownianMotion(covariance.Covariance):
    """
    """
    def __new__(cls, ordint, diffconst):
        """
        If q=0, return BrownianMotion object instead
        of IntegratedBrownianMotion.
        """
        if cls is IntegratedBrownianMotion:
            if ordint == 0:
                return BrownianMotion(diffconst=diffconst)
        else:
            return super().__new__(cls)

    def __init__(self, ordint, diffconst):
        """
        """
        # Coming soon
        pass


class BrownianMotion(covariance.Covariance):
    """
    """

    def __init__(self, diffconst):
        """
        """
        # coming soon
        pass
