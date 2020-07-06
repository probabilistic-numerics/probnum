"""
"""

from probnum.prob.randomprocess.kernels import kernels


class IntegratedBrownianMotionKernel(kernels.Kernel):
    """
    """
    def __new__(cls, ordint, diffconst):
        """
        If q=0, return BrownianMotion object instead
        of IntegratedBrownianMotion.
        """
        if cls is IntegratedBrownianMotionKernel:
            if ordint == 0:
                return BrownianMotionKernel(diffconst=diffconst)
        else:
            return super().__new__(cls)

    def __init__(self, ordint, diffconst):
        """
        """
        # Coming soon
        pass


class BrownianMotionKernel(kernels.Kernel):
    """
    """

    def __init__(self, diffconst):
        """
        """
        # coming soon
        pass
