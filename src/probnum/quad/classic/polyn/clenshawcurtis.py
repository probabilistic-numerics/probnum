"""
Clenshaw-Curtis quadrature formula.

Formula for nodes and weights:
    [1] Sparse Grid Quadrature in High Dimensions with Applications in Finance and Insurance
        Holtz, M., Springer, 2010(, Chapter 3, p. 42ff)
URL:
    https://books.google.de/books?id=XOfMm-4ZM9AC&pg=PA42&lpg=PA42&dq=filippi+formu
    la+clenshaw+curtis&source=bl&ots=gkhNu9F1fp&sig=ACfU3U3zdH-OHx0PqqB_KAXb1mM5iXI
    ojw&hl=de&sa=X&ved=2ahUKEwijovCC1O7kAhUKr6QKHaVWCwQQ6AEwD3oECAgQAQ#v=onepage&q=
    filippi%20formula%20clenshaw%20curtis&f=false

This is a lot of old code. It is well tested, but the _compute_*(...)
functions are really hard to read (they are efficient, though).
"""

import numpy as np

from probnum.quad import quadrature
from probnum import utils


__all__ = ["ClenshawCurtis"]


class ClenshawCurtis(quadrature.Quadrature):
    """
    """

    def __init__(self, npts_per_dim, ndim, ilbds):
        """
        """
        utils.assert_is_2d_ndarray(ilbds)
        weights = _compute_weights(npts_per_dim, ndim, ilbds)
        nodes = _compute_nodes(npts_per_dim, ndim, ilbds)
        quadrature.Quadrature.__init__(self, nodes, weights, ilbds)


def _compute_weights(npts, ndim, ilbds):
    """
    """
    if npts ** ndim * ndim >= 1e9:
        raise TypeError("Tensor-mesh too large for memory.")
    num_tiles = np.arange(ndim)
    num_reps = ndim - np.arange(ndim) - 1
    weights = _compute_weights_1d(npts, ndim, ilbds[0])
    prodweights = np.repeat(weights, npts ** (num_reps[0]))
    for i in range(1, ndim):
        weights = _compute_weights_1d(npts, ndim, ilbds[i])
        column = np.repeat(np.tile(weights, int(npts ** i)),
                           int(npts ** (ndim - 1 - i)))
        prodweights *= column
    return prodweights


def _compute_weights_1d(npts, ndim, ilbds1d):
    """
    """
    if npts % 2 == 0:
        raise TypeError("Please enter odd npts")
    nhalfpts = int((npts + 1.0) / 2.0)
    ind_j = 2.0 * np.arange(1, nhalfpts + 1) - 1.0
    ind_i = np.arange(1, npts + 1)
    arr1 = 2.0 / (npts + 1.0) * np.sin(ind_i * np.pi / (npts + 1.0))
    arr2 = 1.0 / ind_j
    arr3 = np.sin(np.outer(ind_j, ind_i) * np.pi / (npts + 1.0))
    weights = arr1 * (arr2 @ arr3)
    return (ilbds1d[1] - ilbds1d[0]) * weights


def _compute_nodes(npts, ndim, ilbds):
    """
    """
    if npts ** ndim * ndim >= 1e9:
        raise TypeError("Tensor-mesh too large for memory.")
    nodes = _compute_nodes_1d(npts, ilbds[0])
    productmesh = np.repeat(nodes, npts ** (ndim - 1))
    for i in range(1, ndim):
        nodes = _compute_nodes_1d(npts, ilbds[i])
        column = np.repeat(np.tile(nodes, int(npts ** i)),
                           int(npts ** (ndim - 1 - i)))
        productmesh = np.vstack((productmesh.T, column)).T
    if ndim == 1:
        return productmesh.reshape((npts, 1))
    else:
        return productmesh


def _compute_nodes_1d(npts, ilbds1d):
    """
    Computes Clenshaw-Curtis nodes in 1d.

    Parameters
    ----------
    npts : int
        number of 1d quadrature points
    ilbds1d : np.ndarray of shape (2,)
        integration bounds of the form [lower, upper]

    Raises
    ------
    TypeError
        if  'npts' is even (CC needs odd number of points for reasons)

    Returns
    -------
    np.ndarray, shape (npts,)
        1d CC nodes in ilbds1d
    """
    if npts % 2 == 0:
        raise TypeError("Please enter odd npts")
    ind = np.arange(1, npts + 1)
    nodes = 0.5 * (1 - np.cos(np.pi * ind / (npts + 1)))
    return nodes * (ilbds1d[1] - ilbds1d[0]) + ilbds1d[0]
