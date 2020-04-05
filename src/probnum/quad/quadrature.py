"""
Interface for quadrature rules,
i.e. computation of nodes and weights.
"""

from probnum import utils


__all__ = ["Quadrature"]


class Quadrature:
    """
    A quadrature rule is a collection of nodes and weights,
    both need to correspond to each other.
    It also knows about current integration bounds.
    """

    def __init__(self, nodes, weights, ilbds):
        """
        nodes are a (n, d) shaped numpy array!
        weights are a (n,) shaped numpy array!
        """
        utils.assert_is_2d_ndarray(nodes)
        utils.assert_is_1d_ndarray(weights)
        utils.assert_is_2d_ndarray(ilbds)
        if len(nodes) != len(weights) or len(nodes.T) != len(ilbds):
            raise TypeError("Nodes and weights and ilbds are incompatible")
        self.nodes = nodes
        self.weights = weights
        self.ilbds = ilbds

    def compute(self, integrand, vect=False):
        """
        Computes integral approximation
        vect is a bool, indicating whether integrand allows vectorised evaluation
        (i.e. evaluation of all nodes at once)
        """
        if vect is False:
            output = 0.0
            for (node, weight) in zip(self.nodes, self.weights):
                output = output + weight * integrand(node)
        else:
            output = self.weights @ integrand(self.nodes)
        return output
