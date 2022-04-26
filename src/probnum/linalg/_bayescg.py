"""A Bayesian conjugate gradient method."""

from typing import Callable, Optional


def bayescg(
    A,
    b,
    x0=None,
    maxiter: Optional[int] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    callback: Optional[Callable] = None,
):
    r"""Bayesian Conjugate Gradient Method.

    In the setting where :math:`A` is a symmetric positive-definite matrix, this solver
    takes prior information on the solution and outputs a posterior belief over
    :math:`x`. This code implements the method described in Cockayne et al. [1]_.

    Note that the solution-based view of BayesCG and the matrix-based view of
    :meth:`problinsolve` correspond [2]_.

    Parameters
    ----------
    A
        *shape=(n, n)* -- A symmetric positive definite matrix (or linear operator).
        Only matrix-vector products :math:`Av` are used internally.
    b
        *shape=(n, )* -- Right-hand side vector.
    x0
        *shape=(n, )* -- Prior belief for the solution of the linear system.
    maxiter
        Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is the
        dimension of :math:`A`.
    atol
        Absolute residual tolerance. If :math:`\lVert r_i \rVert = \lVert b - Ax_i
        \rVert < \text{atol}`, the iteration terminates.
    rtol
        Relative residual tolerance. If :math:`\lVert r_i \rVert  < \text{rtol}
        \lVert b \rVert`, the iteration terminates.
    callback
        User-supplied function called after each iteration of the linear solver. It is
        called as ``callback(xk, sk, yk, alphak, resid, **kwargs)`` and can be used to
        return quantities from the iteration. Note that depending on the function
        supplied, this can slow down the solver.

    Returns
    -------

    References
    ----------
    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian
       Analysis*, 2019, 14, 937-1012
    .. [2] Bartels, S. et al., Probabilistic Linear Solvers: A Unifying View,
       *Statistics and Computing*, 2019

    See Also
    --------
    problinsolve : Solve linear systems in a Bayesian framework.
    """
    raise NotImplementedError
