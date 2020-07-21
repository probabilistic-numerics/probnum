"""
Quadrature rules based on polynomial functions.

Class of quadrature rules derived by constructing polynomial functions which are simple
to integrate.
"""

from probnum.quad.quadrature import Quadrature
from probnum import utils


class PolynomialQuadrature(Quadrature):
    """
    Quadrature rule based on polynomial functions.

    A polynomial quadrature rule is given by a collection of nodes, the roots of the
    polynomial function and a set of corresponding weights.

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
            raise ValueError(
                "Either nodes and weights or nodes and bounds are incompatible."
            )
        self.nodes = nodes
        self.weights = weights
        self.bounds = bounds
        super().__init__()

    def integrate(self, fun, isvectorized=False, **kwargs):
        """
        Numerically integrate the function ``fun``.

        Parameters
        ----------
        fun : callable
            Function to be integrated. Signature is ``fun(x, **kwargs)``
            where ``x`` is either a float or an ``ndarray`` with
            shape (d,). If ``fun`` can be evaluated vectorized, the
            implementation expects signature ``fun(X, **kwargs)`` where
            ``X`` is is an ``ndarray`` of shape (n, d). Making use of
            vectorization is recommended wherever possible for improved
            speed of computation.
        isvectorized : bool
            Whether integrand allows vectorised evaluation (i.e. evaluation of all nodes
            at once).
        kwargs
            Key-word arguments being passed down to ``fun`` at each
            evaluation. For example (hyper)parameters.
        """
        if isvectorized is False:
            output = 0.0
            for (node, weight) in zip(self.nodes, self.weights):
                output = output + weight * fun(node, **kwargs)
        else:
            output = self.weights @ fun(self.nodes, **kwargs)
        return output
