"""Interface for the Bayesian conjugate gradient method."""

from typing import Optional

from probnum.type import MatrixArgType


def bayescg(
    A: MatrixArgType,
    b: MatrixArgType,
    x0: MatrixArgType = None,
    maxiter: Optional[int] = None,
    atol: float = 10 ** -6,
    rtol: float = 10 ** -6,
):
    """Bayesian conjugate gradient method.

    In the setting where :math:`A` is a symmetric positive-definite matrix, this solver
    takes prior information on the solution and outputs a posterior belief about
    :math:`x`. This code implements the method described in Cockayne et al. [1]_.

    Parameters
    ----------
    A :
        *shape=(n, n)* -- A square linear operator (or matrix). Only matrix-vector
        products :math:`v \\mapsto Av` are used internally.
    b :
        *shape=(n, ) or (n, nrhs)* -- Right-hand side vector in :math:`A x = b`.
    x0 :
        *shape=(n, ) or (n, nrhs)* -- Prior belief for the solution of the linear
        system.
    maxiter :
        Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is the
        dimension of :math:`A`.
    atol :
        Absolute residual tolerance. If :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b
        \\rVert < \\text{atol}`, the iteration terminates.
    rtol :
        Relative residual tolerance. If :math:`\\lVert r_i \\rVert  < \\text{rtol}
        \\lVert b \\rVert`, the iteration terminates.

    Notes
    -----
    The solution-based view of BayesCG and the matrix-based view of
    :meth:`.problinsolve` correspond [2]_.

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
