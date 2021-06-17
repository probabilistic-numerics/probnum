.. _api_docs:

=================
API Documentation
=================


ProbNum implements probabilistic numerical methods in Python. Such methods solve
numerical problems from linear algebra, optimization, quadrature and differential
equations using probabilistic inference. This approach captures uncertainty arising from
finite computational resources and stochastic input.

.. table::

    +----------------------------------+--------------------------------------------------------------+
    | **Subpackage**                   | **Description**                                              |
    +----------------------------------+--------------------------------------------------------------+
    | :mod:`~probnum.diffeq`           | Probabilistic solvers for ordinary differential equations.   |
    +----------------------------------+--------------------------------------------------------------+
    | :mod:`~probnum.filtsmooth`       | Bayesian filtering and smoothing.                            |
    +----------------------------------+--------------------------------------------------------------+
    | :mod:`~probnum.kernels`          | Kernels / covariance functions.                              |
    +----------------------------------+--------------------------------------------------------------+
    | :mod:`~probnum.linalg`           | Probabilistic numerical linear algebra.                      |
    +----------------------------------+--------------------------------------------------------------+
    | :mod:`~probnum.linops`           | Finite-dimensional linear operators.                         |
    +----------------------------------+--------------------------------------------------------------+
    | :mod:`~probnum.problems`         | Definitions and collection of problems solved by PN methods. |
    +----------------------------------+--------------------------------------------------------------+
    | :mod:`~probnum.quad`             | Bayesian quadrature / numerical integration.                 |
    +----------------------------------+--------------------------------------------------------------+
    | :mod:`~probnum.randprocs`        | Random processes representing uncertain functions.           |
    +----------------------------------+--------------------------------------------------------------+
    | :mod:`~probnum.randvars`         | Random variables representing uncertain values.              |
    +----------------------------------+--------------------------------------------------------------+
    | :mod:`~probnum.statespace`       | Probabilistic state space models.                            |
    +----------------------------------+--------------------------------------------------------------+
    | :mod:`~probnum.utils`            | Utility functions.                                           |
    +----------------------------------+--------------------------------------------------------------+

.. toctree::
    :maxdepth: 2
    :caption: API Documentation
    :hidden:

    public_api/probnum
    public_api/diffeq
    public_api/filtsmooth
    public_api/kernels
    public_api/linalg
    public_api/linops
    public_api/problems
    public_api/quad
    public_api/randprocs
    public_api/randvars
    public_api/statespace
    public_api/utils
