"""
Clenshaw-Curtis quadrature rule.

This module implements the Clenshaw-Curtis quadrature rule and associated functions.

Formula for nodes and weights:
    [1] Sparse Grid Quadrature in High Dimensions with Applications in Finance and
        Insurance Holtz, M., Springer, 2010(, Chapter 3, p. 42ff)
URL:
    https://books.google.de/books?id=XOfMm-4ZM9AC&pg=PA42&lpg=PA42&dq=filippi+formu
    la+clenshaw+curtis&source=bl&ots=gkhNu9F1fp&sig=ACfU3U3zdH-OHx0PqqB_KAXb1mM5iXI
    ojw&hl=de&sa=X&ved=2ahUKEwijovCC1O7kAhUKr6QKHaVWCwQQ6AEwD3oECAgQAQ#v=onepage&q=
    filippi%20formula%20clenshaw%20curtis&f=false

Note
----
This is a lot of old code. It is well tested, but the _compute_*(...)
functions are really hard to read (they are efficient, though).
"""

import numpy as np

from probnum.quad.polynomial.polynomialquadrature import PolynomialQuadrature
from probnum import utils


class ClenshawCurtis(PolynomialQuadrature):
    """
    Clenshaw-Curtis quadrature rule.

    Method of numerical integration based on an expansion of the
    integrand in terms of a discrete cosine transform.


    The nodes of the Clenshaw-Curtis rule are the roots of the Chebyshev
    polynomials. The :math:`i^\\text{th}` root is

    .. math:: x_i = \\frac{1}{2} \\left(1 - \\cos\\left( \\frac{i \\pi}{n+1} \\right) \\right)

    for :math:`i=1, ..., n`. The :math:`i^\\text{th}` weight is given by

    .. math:: w_i = \\frac{2}{n+1} \\sin\\left(\\frac{i \\pi}{n+1}\\right)\\sum_{j=1}^{(n+1)/2} \\frac{1}{2j-1}\\sin\\left(\\frac{(2j-1)i \\pi}{n+1}\\right).

    These formulas can be found in [1]_. For an :math:`r`-times
    differentiable integrand, the Clenshaw-Curtis approximation error is
    proportional to :math:`\\mathcal{O}(n^{-r})`. It integrates
    polynomials of degree :math:`\\leq n+1` exactly.

    Parameters
    ----------
    npts_per_dim : int
        Number of evaluation points per dimension. The resulting mesh
        will have `npts_per_dim**ndim` elements.
    ndim : int
        Number of dimensions.
    bounds : ndarray, shape=(n, 2)
        Integration bounds.

    See Also
    --------
    PolynomialQuadrature : Quadrature rule based on polynomial functions.

    References
    ----------
    .. [1] Holtz, M., Sparse Grid Quadrature in High Dimensions with Applications in
       Finance and Insurance, Springer, 2010


    Examples
    --------
    >>> cc = ClenshawCurtis(npts_per_dim=3, ndim=2, bounds=np.array([[0, 1], [0, 0.1]]))
    >>> print(cc.nodes)
    [[0.14644661 0.01464466]
     [0.14644661 0.05      ]
     [0.14644661 0.08535534]
     [0.5        0.01464466]
     [0.5        0.05      ]
     [0.5        0.08535534]
     [0.85355339 0.01464466]
     [0.85355339 0.05      ]
     [0.85355339 0.08535534]]
    >>> print(cc.weights)
    [0.01111111 0.01111111 0.01111111 0.01111111 0.01111111 0.01111111
     0.01111111 0.01111111 0.01111111]
    >>> print(cc.integrate(lambda x: x[0] + x[1]))
    0.05500000000000001

    >>> cc = ClenshawCurtis(npts_per_dim=7, ndim=1, bounds=np.array([[0, 1]]))
    >>> print(cc.weights)
    [0.08898234 0.12380952 0.19673195 0.18095238 0.19673195 0.12380952
     0.08898234]
    >>> print(cc.nodes)
    [[0.03806023]
     [0.14644661]
     [0.30865828]
     [0.5       ]
     [0.69134172]
     [0.85355339]
     [0.96193977]]
    >>> print(cc.integrate(lambda x: np.sin(x)))
    [0.45969769]

    """  # pylint: disable=line-too-long

    def __init__(self, npts_per_dim, ndim, bounds):
        utils.assert_is_2d_ndarray(bounds)
        weights = _compute_weights(npts_per_dim, ndim, bounds)
        nodes = _compute_nodes(npts_per_dim, ndim, bounds)
        PolynomialQuadrature.__init__(self, nodes, weights, bounds)


def _compute_weights(npts, ndim, ilbds):
    """
    Computes 1D Clenshaw-Curtis weights and aligns them in
    correspondence to the computed nodes. Since the resulting mesh is of
    size (n**d, d), the weight array is of size (n**d,).
    """
    if npts ** ndim * ndim >= 1e9:
        raise MemoryError("Weights for tensor-mesh too large for memory.")
    # num_tiles = np.arange(ndim)
    num_reps = ndim - np.arange(ndim) - 1
    weights = _compute_weights_1d(npts, ndim, ilbds[0])
    prodweights = np.repeat(weights, npts ** (num_reps[0]))
    for i in range(1, ndim):
        weights = _compute_weights_1d(npts, ndim, ilbds[i])
        column = np.repeat(
            np.tile(weights, int(npts ** i)), int(npts ** (ndim - 1 - i))
        )
        prodweights *= column
    return prodweights


def _compute_weights_1d(npts, ndim, ilbds1d):
    """
    Computes weights of Clenshaw-Curtis formula.

    The :math:`i^\textrm{th}` weight is given by

    .. math:: w_i = \\frac{2}{n+1} \\sin\\left(\\frac{i \\pi}{n+1}\\right)\\sum_{j=1}^{(n+1)/2} \\frac{1}{2j-1}\\sin\\left(\\frac{(2j-1)i \\pi}{n+1}\\right).
    """  # pylint: disable=line-too-long
    if npts % 2 == 0:
        raise ValueError("Please enter odd npts")
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
    Computes 1D Clenshaw-Curtis nodes and aligns them in order to create
    a tensor mesh: each point is aligned with each point to create a
    mesh of size (n^d, d).
    """
    if npts ** ndim * ndim >= 1e9:
        raise ValueError("Tensor-mesh too large for memory.")
    nodes = _compute_nodes_1d(npts, ilbds[0])
    productmesh = np.repeat(nodes, npts ** (ndim - 1))
    for i in range(1, ndim):
        nodes = _compute_nodes_1d(npts, ilbds[i])
        column = np.repeat(np.tile(nodes, int(npts ** i)), int(npts ** (ndim - 1 - i)))
        productmesh = np.vstack((productmesh.T, column)).T
    if ndim == 1:
        return productmesh.reshape((npts, 1))
    else:
        return productmesh


def _compute_nodes_1d(npts, ilbds1d):
    """
    Computes Clenshaw-Curtis nodes in 1d.

    The :math:`i^\\text{th}` root is

    .. math:: x_i = \\frac{1}{2} \\left(1 - \\cos\\left( \\frac{i \\pi}{n+1} \\right) \\right)

    Parameters
    ----------
    npts : int
        number of 1d quadrature points
    ilbds1d : np.ndarray of shape (2,)
        integration bounds of the form [lower, upper]

    Raises
    ------
    ValueError
        if  'npts' is even (CC needs odd number of points for reasons)

    Returns
    -------
    np.ndarray, shape (npts,)
        1d CC nodes in ilbds1d
    """  # pylint: disable=line-too-long
    if npts % 2 == 0:
        raise ValueError("Please enter odd npts")
    ind = np.arange(1, npts + 1)
    nodes = 0.5 * (1 - np.cos(np.pi * ind / (npts + 1)))
    return nodes * (ilbds1d[1] - ilbds1d[0]) + ilbds1d[0]
