"""
Quadrature rules based on interpolating functions.

Class of quadrature rules derived by constructing interpolating functions which are simple to integrate.
"""

from probnum.quad.quadrature import Quadrature
from probnum import utils


class InterpolationQuadrature(Quadrature):
    """
    Quadrature rule based on interpolating functions.

    An interpolation quadrature rule is given by a collection of nodes, the roots of the
    interpolating function and a set of corresponding weights.

    Parameters
    ----------
    nodes : ndarray, shape=(n,d)
        Integration nodes.
    weights : ndarray, shape=(n,)
        Integration weights.
    bounds : ndarray, shape=(d, 2)
        Integration bounds.

    See Also
    --------
    Clenshaw-Curtis : Clenshaw-Curtis quadrature rule.
    """

    def __init__(self, nodes, weights, bounds):
        """
        Create an instance of a quadrature method.
        """
        utils.assert_is_2d_ndarray(nodes)
        utils.assert_is_1d_ndarray(weights)
        utils.assert_is_2d_ndarray(bounds)
        if len(nodes) != len(weights) or len(nodes.T) != len(bounds):
            raise ValueError("Either nodes and weights or nodes and bounds are incompatible.")
        self.nodes = nodes
        self.weights = weights
        self.bounds = bounds
        super().__init__()

    def integrate(self, func, isvectorized=False):
        """
        Numerically integrate the function ``func``.

        Parameters
        ----------
        func : function
            Function to be integrated.
        isvectorized : bool
            Whether integrand allows vectorised evaluation (i.e. evaluation of all nodes at once).
        """
        if isvectorized is False:
            output = 0.0
            for (node, weight) in zip(self.nodes, self.weights):
                output = output + weight * func(node)
        else:
            output = self.weights @ func(self.nodes)
        return output
