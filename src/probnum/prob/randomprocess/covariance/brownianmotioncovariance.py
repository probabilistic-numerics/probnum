
import numpy as np
import scipy.spatial as scs

from probnum.prob.randomprocess.covariance import covariance
from probnum.utils import *


__all__ = ["IntegratedBrownianMotionCovariance", "BrownianMotionCovariance"]


class IntegratedBrownianMotionCovariance(covariance.Covariance):
    """
    """
    def __new__(cls, ordint, diffconst):
        """
        If q=0, return BrownianMotion object instead
        of IntegratedBrownianMotion.
        """
        if cls is IntegratedBrownianMotionCovariance:
            if ordint == 0:
                return BrownianMotionCovariance(diffconst=diffconst)
        else:
            return super().__new__(cls)

    def __init__(self, ordint, diffconst):
        """
        """
        # Coming soon
        pass


class BrownianMotionCovariance(covariance.Covariance):
    """
    """

    def __init__(self, diffconst):
        """
        """
        # coming soon
        pass
