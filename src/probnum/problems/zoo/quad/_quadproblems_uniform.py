import itertools

import numpy as np
from scipy.stats import norm

from probnum.typing import FloatArgType, IntArgType

__all__ = [
    "genz_continuous",
    "genz_cornerpeak",
    "genz_discontinuous",
    "genz_gaussian",
    "genz_oscillatory",
    "genz_productpeak",
    "bratley1992",
    "roos_arnold",
    "gfunction",
    "morokoff_caflisch_1",
    "morokoff_caflisch_2",
]


def genz_continuous(
    dim: int, a: np.ndarray = None, u: np.ndarray = None
) -> QuadratureProblem:
    """Genz 'continuous' test function on [0,1]^d.

    .. math::  f(x) = \exp(- \sum_{i=1}^d a_i |x_i - u_i|).
    Parameters
    ----------
        dim
            Dimension of the domain
        a
            First set of parameters affecting the difficulty of the integration problem.
        u
            Second set of parameters affecting the difficulty of the integration problem.
            All entries should be in [0,1].
    References
    ----------
        https://www.sfu.ca/~ssurjano/cont.html
    """

    # Specify default values of parameters a and u
    if a is None:
        a = np.broadcast_to(5.0, dim)

    if u is None:
        u = np.broadcast_to(0.5, dim)

    # Check that the parameters have valid values and shape
    if a.shape != (dim,):
        raise ValueError(
            f"Invalid shape {a.shape} for parameter `a`. Expected {(dim,)}."
        )

    if u.shape != (dim,):
        raise ValueError(
            f"Invalid shape {u.shape} for parameter `u`. Expected {(dim,)}."
        )

    if np.any(u < 0.0) or np.any(u > 1):
        raise ValueError(f"The parameters `u` must lie in the interval [0.0, 1.0].")

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        nonlocal a, u
        n = x.shape[0]

        # Check that the input points have valid values and shape
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Reshape u into an (n,dim) array with identical rows
        u_reshaped = np.repeat(u.reshape([1, dim]), n, axis=0)

        # Compute function values
        f = np.exp(-np.sum(a * np.abs(x - u_reshaped), axis=1))

        return f.reshape((n, 1))

    solution = np.prod((2.0 - np.exp(-a * u) - np.exp(a * (u - 1))) / a)

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(0.0, dim),
        upper_bd=np.broadcast_to(1.0, dim),
        output_dim=None,
        solution=solution,
    )


def genz_cornerpeak(
    dim: int, a: np.ndarray = None, u: np.ndarray = None
) -> QuadratureProblem:

    """Genz 'corner peak' test function on [0,1]^d.

    .. math:: f(x) = (1+\sum_{i=1}^d a_i x_i)^{-d+1}

    Parameters
    ----------
        dim
            Dimension of the domain
        a
            First set of parameters affecting the difficulty of the integration problem.
        u
            Second set of parameters affecting the difficulty of the integration problem.
            All entries should be in [0,1].

    References
    ----------
        https://www.sfu.ca/~ssurjano/copeak.html
    """

    # Specify default values of parameters a and u
    if a is None:
        a = np.broadcast_to(5.0, dim)

    if u is None:
        u = np.broadcast_to(0.5, dim)

    # Check that the parameters have valid values and shape
    if a.shape != (dim,):
        raise ValueError(
            f"Invalid shape {a.shape} for parameter `a`. Expected {(dim,)}."
        )

    if u.shape != (dim,):
        raise ValueError(
            f"Invalid shape {u.shape} for parameter `u`. Expected {(dim,)}."
        )

    if np.any(u < 0.0) or np.any(u > 1):
        raise ValueError(f"The parameters `u` must lie in the interval [0.0, 1.0].")

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        nonlocal a, u
        n = x.shape[0]

        # Check that the input points have valid values
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Compute function values
        f = (1.0 + np.sum(a * x, axis=1)) ** (-dim - 1)

        return f.reshape((n, 1))

    # Calculate closed-form solution of the integral
    solution = 0.0
    for k in range(0, dim + 1):
        subsets_k = list(itertools.combinations(range(dim), k))
        for subset_ind in range(len(subsets_k)):
            a_subset = a[np.asarray(subsets_k[subset_ind], dtype=int)]
            solution = solution + ((-1.0) ** (k + dim)) * (
                1.0 + np.sum(a) - np.sum(a_subset)
            ) ** (-1)
    solution = solution / (np.prod(a) * np.math.factorial(dim))

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(0.0, dim),
        upper_bd=np.broadcast_to(1.0, dim),
        output_dim=None,
        solution=solution,
    )


def genz_discontinuous(
    dim: int, a: np.ndarray = None, u: np.ndarray = None
) -> QuadratureProblem:
    """Genz 'discontinuous' test function on [0,1]^d.

    :math:'f(x) = 0' if any :math:'x_i > u_i' and :math:'f(x) = \exp(\sum_{i=1}^d a_i x_i)' otherwise.


    Parameters
    ----------
        dim
            Dimension of the domain
        a
            First set of parameters affecting the difficulty of the integration problem.
        u
            Second set of parameters affecting the difficulty of the integration problem.
            All entries should be in [0,1].

    References
    ----------
        https://www.sfu.ca/~ssurjano/disc.html
    """

    # Specify default values of parameters a and u
    if a is None:
        a = np.broadcast_to(5.0, dim)

    if u is None:
        u = np.broadcast_to(0.5, dim)

    # Check that the parameters have valid values and shape
    if a.shape != (dim,):
        raise ValueError(
            f"Invalid shape {a.shape} for parameter `a`. Expected {(dim,)}."
        )

    if u.shape != (dim,):
        raise ValueError(
            f"Invalid shape {u.shape} for parameter `u`. Expected {(dim,)}."
        )

    if np.any(u < 0.0) or np.any(u > 1):
        raise ValueError(f"The parameters `u` must lie in the interval [0.0, 1.0].")

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        nonlocal a, u
        n = x.shape[0]

        # Check that the input points have valid values
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Compute function values
        f = np.exp(np.sum(a * x, axis=1))
        # Set function to zero whenever x_i > u_i for i =1,..,min(2,d)
        f[np.any(x - u > 0, axis=1)] = 0

        return f.reshape(n, 1)

    if dim == 1:
        solution = (np.exp(a * u) - 1.0) / a
    if dim > 1:
        solution = np.prod((np.exp(a * np.minimum(u, 1.0)) - 1.0) / a)

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(0.0, dim),
        upper_bd=np.broadcast_to(1.0, dim),
        output_dim=None,
        solution=solution,
    )


def genz_gaussian(
    dim: int, a: np.ndarray = None, u: np.ndarray = None
) -> QuadratureProblem:
    """Genz 'Gaussian' test function on [0,1]^d.

    .. math::  f(x) = \exp(- \sum_{i=1}^d a_i^2 (x_i - u_i)^2).

    Parameters
    ----------
        dim
            Dimension of the domain
        a
            First set of parameters affecting the difficulty of the integration problem.
        u
            Second set of parameters affecting the difficulty of the integration problem.
            All entries should be in [0,1].

    References
    ----------
        https://www.sfu.ca/~ssurjano/gaussian.html
    """

    # Specify default values of parameters a and u
    if a is None:
        a = np.broadcast_to(5.0, dim)

    if u is None:
        u = np.broadcast_to(0.5, dim)

    # Check that the parameters have valid values and shape
    if a.shape != (dim,):
        raise ValueError(
            f"Invalid shape {a.shape} for parameter `a`. Expected {(dim,)}."
        )

    if u.shape != (dim,):
        raise ValueError(
            f"Invalid shape {u.shape} for parameter `u`. Expected {(dim,)}."
        )

    if np.any(u < 0.0) or np.any(u > 1):
        raise ValueError(f"The parameters `u` must lie in the interval [0.0, 1.0].")

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        nonlocal a, u
        n = x.shape[0]

        # Check that the input points have valid values
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Reshape u into an (n,dim) array with identical rows
        u_reshaped = np.repeat(u.reshape([1, dim]), n, axis=0)

        # Compute function values
        f = np.exp(-np.sum((a * (x - u_reshaped)) ** 2, axis=1))

        return f.reshape((n, 1))

    solution = np.pi ** (dim / 2) * np.prod(
        (norm.cdf(np.sqrt(2) * a * (1.0 - u)) - norm.cdf(-np.sqrt(2) * a * u)) / a
    )

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(0.0, dim),
        upper_bd=np.broadcast_to(1.0, dim),
        output_dim=None,
        solution=solution,
    )


def genz_oscillatory(
    dim: int, a: np.ndarray = None, u: np.ndarray = None
) -> QuadratureProblem:
    """Genz 'oscillatory' test function on [0,1]^d.

    .. math::  f(x) = \cos( 2 \pi u_1 + \sum_{i=1}^d a_i x_i).


    Parameters
    ----------
        dim
            Dimension of the domain
        a
            First set of parameters affecting the difficulty of the integration problem.
        u
            Second set of parameters affecting the difficulty of the integration problem.
            All entries should be in [0,1].

    References
    ----------
        https://www.sfu.ca/~ssurjano/oscil.html
    """

    # Specify default values of parameters a and u
    if a is None:
        a = np.broadcast_to(5.0, dim)

    if u is None:
        u = np.broadcast_to(0.5, dim)

    # Check that the parameters have valid values and shape
    if a.shape != (dim,):
        raise ValueError(
            f"Invalid shape {a.shape} for parameter `a`. Expected {(dim,)}."
        )

    if u.shape != (dim,):
        raise ValueError(
            f"Invalid shape {u.shape} for parameter `u`. Expected {(dim,)}."
        )

    if np.any(u < 0.0) or np.any(u > 1):
        raise ValueError(f"The parameters `u` must lie in the interval [0.0, 1.0].")

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        nonlocal a, u
        n = x.shape[0]

        # Check that the input points have valid values
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Compute function values
        f = np.cos(2.0 * np.pi * u[0] + np.sum(a * x, axis=1))

        return f.reshape((n, 1))

    # Calculate closed-form solution to the integral
    dim_modulo4 = np.remainder(dim, 4)

    def hfunc(x):
        if dim_modulo4 == 1:
            return np.sin(x)
        if dim_modulo4 == 2:
            return -np.cos(x)
        if dim_modulo4 == 3:
            return -np.sin(x)
        if dim_modulo4 == 0:
            return np.cos(x)

    solution = 0.0
    for k in range(0, dim + 1):
        subsets_k = list(itertools.combinations(range(dim), k))
        for subset_ind in range(len(subsets_k)):
            a_subset = a[np.asarray(subsets_k[subset_ind], dtype=int)]
            solution = solution + ((-1.0) ** k) * hfunc(
                (2.0 * np.pi * u[0]) + np.sum(a) - np.sum(a_subset)
            )

    solution = solution / np.prod(a)

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(0.0, dim),
        upper_bd=np.broadcast_to(1.0, dim),
        output_dim=None,
        solution=solution,
    )


def genz_productpeak(
    dim: int, a: np.ndarray = None, u: np.ndarray = None
) -> QuadratureProblem:
    """Genz 'Product Peak' test function on [0,1]^d.

    .. math::  f(x) = \prod_{i=1}^d ( a_i^{-2} + (x_i-u_i)^2)^{-1}.


    Parameters
    ----------
        dim
            Dimension of the domain
        a
            First set of parameters affecting the difficulty of the integration problem.
        u
            Second set of parameters affecting the difficulty of the integration problem.
            All entries should be in [0,1].

    References
    ----------
        https://www.sfu.ca/~ssurjano/prpeak.html
    """

    # Specify default values of parameters a and u
    if a is None:
        a = np.broadcast_to(5.0, dim)

    if u is None:
        u = np.broadcast_to(0.5, dim)

    # Check that the parameters have valid values and shape
    if a.shape != (dim,):
        raise ValueError(
            f"Invalid shape {a.shape} for parameter `a`. Expected {(dim,)}."
        )

    if u.shape != (dim,):
        raise ValueError(
            f"Invalid shape {u.shape} for parameter `u`. Expected {(dim,)}."
        )

    if np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError(f"The parameters `u` must lie in the interval [0.0, 1.0].")

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        nonlocal a, u
        n = x.shape[0]

        # Check that the input points have valid values
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Reshape u into an (n,dim) array with identical rows
        u_reshaped = np.repeat(u.reshape([1, dim]), n, axis=0)

        # Compute function values
        f = np.prod((a ** (-2) + (x - u_reshaped) ** 2) ** (-1), axis=1)

        return f.reshape((n, 1))

    solution = np.prod(a * (np.arctan(a * (1.0 - u)) - np.arctan(-a * u)))

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(0.0, dim),
        upper_bd=np.broadcast_to(1.0, dim),
        output_dim=None,
        solution=solution,
    )


def bratley1992(dim: int) -> QuadratureProblem:

    """'Bratley 1992' test function on [0,1]^d.

    .. math::  f(x) = \sum_{i=1}^d (-1)^i \prod_{j=1}^i x_j.


    https://www.sfu.ca/~ssurjano/bratleyetal92.html

    Parameters
    ----------
        dim
            Dimension of the domain

    References
    ----------
        Bratley, P., Fox, B. L., & Niederreiter, H. (1992). Implementation and tests of low-discrepancy sequences. ACM Transactions on Modeling and Computer Simulation (TOMACS), 2(3), 195-213.
    """

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        n = x.shape[0]

        # Check that the input points have valid values
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Compute function values
        f = np.sum(
            ((-1.0) ** np.arange(1, dim + 1)) * np.cumprod(x, axis=1),
            axis=1,
        )

        return f.reshape((n, 1))

    solution = -(1.0 / 3) * (1.0 - ((-0.5) ** dim))

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(0.0, dim),
        upper_bd=np.broadcast_to(1.0, dim),
        output_dim=None,
        solution=solution,
    )


def roos_arnold(dim: int) -> QuadratureProblem:
    """'Roos & Arnold 1963' test function on [0,1]^d.

    .. math::  f(x) = \prod_{i=1}^d |4 x_i - 2 |.


    https://www.sfu.ca/~ssurjano/roosarn63.html

    Parameters
    ----------
        dim
            Dimension of the domain

    References
    ----------
        Roos, P., & Arnold, L. (1963). Numerische experimente zur mehrdimensionalen quadratur. Springer.
    """

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        n = x.shape[0]

        # Check that the input points have valid values
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Compute function values
        f = np.prod(np.abs(4.0 * x - 2.0), axis=1)

        return f.reshape((n, 1))

    solution = 1.0

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(0.0, dim),
        upper_bd=np.broadcast_to(1.0, dim),
        output_dim=None,
        solution=solution,
    )


def gfunction(dim: int) -> QuadratureProblem:
    """'G-function' test function on [0,1]^d.

    .. math::  f(x) = \prod_{i=1}^d \frac{|4 x_i - 2 |+a_i}{1+a_i} \text{ where } a_i = \frac{i-2}{2} \forall i = 1, \ldots,d

    https://www.sfu.ca/~ssurjano/gfunc.html


    Parameters
    ----------
        dim
            Dimension of the domain

    References
    ----------
        Marrel, A., Iooss, B., Laurent, B., & Roustant, O. (2009). Calculations of sobol indices for the gaussian process metamodel. Reliability Engineering & System Safety, 94(3), 742-751.
    """

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        n = x.shape[0]

        # Check that the input points have valid values
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Compute function values
        a = np.atleast_2d(((np.arange(dim) + 1.0) - 2.0) / 2.0)
        f = np.prod((np.abs(4.0 * x - 2.0) + a) / (1.0 + a), axis=1)

        return f.reshape((n, 1))

    solution = 1.0

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(0.0, dim),
        upper_bd=np.broadcast_to(1.0, dim),
        output_dim=None,
        solution=solution,
    )


def morokoff_caflisch_1(dim: int) -> QuadratureProblem:
    """'Morokoff & Caflisch 1995' test function number 1 on [0,1]^d.

    .. math::  f(x) = (1+1/d)^d \prod_{i=1}^d x_i^{1/d}

    https://www.sfu.ca/~ssurjano/morcaf95a.html

    Parameters
    ----------
        dim
            Dimension of the domain

    References
    ----------
        Morokoff, W. J., & Caflisch, R. E. (1995). Quasi-monte carlo integration. Journal of computational physics, 122(2), 218-230.
        Gerstner, T., & Griebel, M. (1998). Numerical integration using sparse grids. Numerical algorithms, 18(3-4), 209-232.
    """

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        n = x.shape[0]

        # Check that the input points have valid values
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Compute function values
        f = ((1.0 + 1.0 / dim) ** (dim)) * np.prod(x ** (1.0 / dim), axis=1)

        return f.reshape((n, 1))

    solution = 1.0

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(0.0, dim),
        upper_bd=np.broadcast_to(1.0, dim),
        output_dim=None,
        solution=solution,
    )


def morokoff_caflisch_2(dim: int) -> QuadratureProblem:
    """'Morokoff & Caflisch 1995' test function number 2 on [0,1]^d.

    .. math::  f(x) = \frac{1}{(d-0.5)^d} \prod_{i=1}^d (d-x_i)


    https://www.sfu.ca/~ssurjano/morcaf95b.html

    Parameters
    ----------
        dim
            Dimension of the domain

    References
    ----------
        Morokoff, W. J., & Caflisch, R. E. (1995). Quasi-monte carlo integration. Journal of computational physics, 122(2), 218-230.
    """

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        n = x.shape[0]

        # Check that the input points have valid values
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Compute function values
        f = (1.0 / ((dim - 0.5) ** dim)) * np.prod(dim - x, axis=1)

        return f.reshape((n, 1))

    solution = 1.0

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(0.0, dim),
        upper_bd=np.broadcast_to(1.0, dim),
        output_dim=None,
        solution=solution,
    )
