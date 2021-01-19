"""Linear system belief.

Base class defining a belief about the quantities of interest of a
linear system such as its solution, the matrix inverse.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import probnum
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.problems import LinearSystem
from probnum.type import MatrixArgType

# Public classes and functions. Order is reflected in documentation.
__all__ = ["LinearSystemBelief"]

# pylint: disable="invalid-name"


class LinearSystemBelief:
    """Belief over quantities of interest of a linear system.

    Random variables :math:`(\\mathsf{x}, \\mathsf{A}, \\mathsf{H}, \\mathsf{b})`
    modelling the solution :math:`x`, the system matrix :math:`A`, its (pseudo-)inverse
    :math:`H=A^{-1}` and the right hand side :math:`b` of a linear system :math:`Ax=b`.

    Parameters
    ----------
    x :
        Belief over the solution.
    A :
        Belief over the system matrix.
    Ainv :
        Belief over the (pseudo-)inverse of the system matrix.
    b :
        Belief over the right hand side.

    Examples
    --------
    Construct a linear system belief from a preconditioner.

    >>> import numpy as np
    >>> from probnum.problems import LinearSystem
    >>> from probnum.linalg.solvers.beliefs import LinearSystemBelief
    >>> # Linear System Ax=b
    >>> linsys = LinearSystem(
    ...     A=np.array([[7.5, 2.0, 1.0], [2.0, 2.0, 0.5], [1.0, 0.5, 5.5]]),
    ...     b=np.array([1.0, 2.0, -3.0]),
    ... )
    >>> # Preconditioner, i.e. approximate inverse of A
    >>> Ainv_approx = np.array(
    ...     [[ 0.2,   -0.18, -0.015],
    ...      [-0.18,   0.7,  -0.03],
    ...      [-0.015, -0.03,  0.20]]
    ... )
    >>> # Prior belief induced by approximate inverse
    >>> prior = LinearSystemBelief.from_inverse(Ainv0=Ainv_approx, problem=linsys)
    >>> # Initial residual Ax0 - b
    >>> residual = linsys.A @ prior.x.mean - linsys.b
    >>> residual
    array([[ 0.0825],
           [ 0.0525],
           [-0.1725]])
    """

    def __init__(
        self,
        x: rvs.RandomVariable,
        A: rvs.RandomVariable,
        Ainv: rvs.RandomVariable,
        b: rvs.RandomVariable,
    ):

        x, A, Ainv, b = self._reshape_2d(x=x, A=A, Ainv=Ainv, b=b)
        self._check_shape_mismatch(x=x, A=A, Ainv=Ainv, b=b)
        self._x = x
        self._A = A
        self._Ainv = Ainv
        self._b = b

    @staticmethod
    def _reshape_2d(
        x: rvs.RandomVariable,
        A: rvs.RandomVariable,
        Ainv: rvs.RandomVariable,
        b: rvs.RandomVariable,
    ) -> Tuple[
        rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable
    ]:
        """Reshape inputs to 2d matrices."""
        if b.ndim == 1:
            b = b.reshape((-1, 1))
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        if x.ndim != 2:
            raise ValueError("Belief over solution must be two-dimensional.")
        if A.ndim != 2 or Ainv.ndim != 2 or b.ndim != 2:
            raise ValueError("Beliefs over system components must be two-dimensional.")
        return x, A, Ainv, b

    @staticmethod
    def _check_shape_mismatch(
        x: rvs.RandomVariable,
        A: rvs.RandomVariable,
        Ainv: rvs.RandomVariable,
        b: rvs.RandomVariable,
    ) -> None:
        """Check whether shapes of arguments match."""

        def dim_mismatch_error(arg0, arg1, arg0_name, arg1_name):
            return ValueError(
                f"Dimension mismatch. The shapes of {arg0_name} : {arg0.shape} "
                f"and {arg1_name} : {arg1.shape} must match."
            )

        if A.shape[0] != b.shape[0]:
            raise dim_mismatch_error(A, b, "A", "b")

        if A.shape[1] != x.shape[0]:
            raise dim_mismatch_error(A, x, "A", "x")

        if x.shape[1] != b.shape[1]:
            raise dim_mismatch_error(x, b, "x", "b")

        if A.shape != Ainv.shape:
            raise dim_mismatch_error(A, Ainv, "A", "Ainv")

    @cached_property
    def x(self) -> rvs.RandomVariable:
        """Belief over the solution."""
        if self._x is None:
            return self._induced_solution_belief()
        elif isinstance(self._x, np.ndarray):
            return rvs.Normal(mean=self._x, cov=self._induced_solution_cov())
        else:
            return self._x

    @property
    def A(self) -> rvs.RandomVariable:
        """Belief over the system matrix."""
        return self._A

    @property
    def Ainv(self) -> rvs.RandomVariable:
        """Belief over the (pseudo-)inverse of the system matrix."""
        return self._Ainv

    @property
    def b(self) -> rvs.RandomVariable:
        """Belief over the right hand side."""
        return self._b

    @classmethod
    def from_solution(
        cls,
        x0: np.ndarray,
        problem: LinearSystem,
        check_for_better_x0: bool = True,
    ) -> "LinearSystemBelief":
        r"""Construct a belief about the linear system from an approximate solution.

        Constructs a matrix-variate prior mean for :math:`H` from an initial
        guess of the solution :math:`x0` and the right hand side :math:`b` such
        that :math:`H_0b = x_0`, :math:`H_0` symmetric positive definite and
        :math:`A_0 = H_0^{-1}`. If :code:`check_for_better_x0=True` and
        :math:`x_0^\top b \leq 0` the belief is initialized with a better approximate
        solution :math:`x_1` with lower error :math:`\lVert x_1 \rVert_A < \lVert x_0
        \rVert_A`. [#]_

        Parameters
        ----------
        x0 :
            Initial guess for the solution of the linear system.
        problem :
            Linear system to solve.
        check_for_better_x0 :
            Choose a better initial guess for the solution if possible.

        References
        ----------
        .. [#] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for
               Machine Learning, *Advances in Neural Information Processing Systems (
               NeurIPS)*, 2020
        """
        x0, Ainv0, A0, b0 = cls._belief_means_from_solution(
            x0=x0, problem=problem, check_for_better_x0=check_for_better_x0
        )
        return cls(
            x=rvs.asrandvar(x0),
            Ainv=rvs.asrandvar(Ainv0),
            A=rvs.asrandvar(A0),
            b=rvs.asrandvar(b0),
        )

    @staticmethod
    def _belief_means_from_solution(
        x0: np.ndarray,
        problem: LinearSystem,
        check_for_better_x0: bool = True,
    ) -> Tuple[
        np.ndarray,
        Union[np.ndarray, linops.LinearOperator],
        Union[np.ndarray, linops.LinearOperator],
        np.ndarray,
    ]:
        """Construct means for the belief from an approximate solution.

        Constructs matrices :math:`H_0` and :math:`A_0` such
        that :math:`H_0b = x_0`, :math:`H_0` symmetric positive definite and
        :math:`A_0 = H_0^{-1}`. If :code:`check_for_better_x0=True` and
        :math:`x_0^\top b \leq 0` the construction is done for a better approximate
        solution :math:`x_1` with lower error :math:`\lVert x_1 \rVert_A < \lVert x_0
        \rVert_A`.

        Parameters
        ----------
        x0 :
            Initial guess for the solution of the linear system.
        problem :
            Linear system to solve.
        check_for_better_x0 :
            Choose a better initial guess for the solution if possible.

        Returns
        -------
        x0 :
            Approximate solution of the linear system.
        Ainv0 :
            Approximate system matrix inverse.
        A0 :
            Approximate system matrix.
        b0 :
            Approximate right hand side
        """
        if x0.ndim < 2:
            x0 = x0.reshape((-1, 1))

        # Instantiate belief over right hand side via sample in stochastic case
        if isinstance(problem.b, rvs.RandomVariable):
            b0 = problem.b.sample()
        else:
            b0 = problem.b

        # If b = 0, set x0 = 0
        if check_for_better_x0 and np.all(b0 == np.zeros_like(b0)):
            x0 = b0
            A0 = linops.Identity(shape=problem.A.shape)
            Ainv0 = A0

        else:
            bx0 = (b0.T @ x0).item()
            bb = np.linalg.norm(b0) ** 2
            # If inner product <x0, b> is non-positive, choose better initialization.
            if check_for_better_x0 and bx0 < -100 * np.finfo(float).eps:
                # <x0, b> < 0
                x0 = -x0
                bx0 = -bx0
            elif check_for_better_x0 and np.abs(bx0) < 100 * np.finfo(float).eps:
                # <x0, b> = 0, b != 0
                if not isinstance(problem.A, rvs.RandomVariable):
                    bAb = (b0.T @ (problem.A @ b0)).item()
                    x0 = bb / bAb * b0
                    bx0 = bb ** 2 / bAb

            # Construct prior mean of A and Ainv
            alpha = 0.5 * bx0 / bb

            def _mv(v):
                return (x0 - alpha * b0) * (x0 - alpha * b0).T @ v

            def _mm(M):
                return (x0 - alpha * b0) @ (x0 - alpha * b0).T @ M

            Ainv0 = linops.ScalarMult(
                scalar=alpha, shape=problem.A.shape
            ) + 2 / bx0 * linops.LinearOperator(
                matvec=_mv, matmat=_mm, shape=problem.A.shape
            )

            A0 = linops.ScalarMult(scalar=1 / alpha, shape=problem.A.shape) - 1 / (
                alpha * np.squeeze((x0 - alpha * b0).T @ x0)
            ) * linops.LinearOperator(matvec=_mv, matmat=_mm, shape=problem.A.shape)

        return x0, Ainv0, A0, b0

    @classmethod
    def from_inverse(
        cls,
        Ainv0: MatrixArgType,
        problem: LinearSystem,
    ) -> "LinearSystemBelief":
        r"""Construct a belief about the linear system from an approximate inverse.

        Returns a belief about the linear system from an approximate inverse
        :math:`H_0\approx A^{-1}` such as a preconditioner.

        Parameters
        ----------
        Ainv0 :
            Approximate inverse of the system matrix.
        problem :
            Linear system to solve.
        """
        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=rvs.asrandvar(Ainv0),
            A=rvs.asrandvar(problem.A),
            b=rvs.asrandvar(problem.b),
        )

    @classmethod
    def from_matrix(
        cls,
        A0: MatrixArgType,
        problem: LinearSystem,
    ) -> "LinearSystemBelief":
        r"""Construct a belief about the linear system from an approximate system matrix.

        Returns a belief about the linear system from an approximation of
        the system matrix :math:`A_0\approx A`.

        Parameters
        ----------
        A0 :
            Approximate system matrix.
        problem :
            Linear system to solve.
        """
        Ainv0 = linops.Identity(shape=A0.shape)

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=rvs.asrandvar(Ainv0),
            A=rvs.asrandvar(A0),
            b=rvs.asrandvar(problem.b),
        )

    @classmethod
    def from_matrices(
        cls,
        A0: MatrixArgType,
        Ainv0: MatrixArgType,
        problem: LinearSystem,
    ) -> "LinearSystemBelief":
        r"""Construct a belief from an approximate system matrix and
        corresponding inverse.

        Returns a belief about the linear system from an approximation of
        the system matrix :math:`A_0\approx A` and an approximate inverse
        :math:`H_0\approx A^{-1}`.

        Parameters
        ----------
        A0 :
            Approximate system matrix.
        Ainv0 :
            Approximate inverse of the system matrix.
        problem :
            Linear system to solve.
        """
        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=rvs.asrandvar(Ainv0),
            A=rvs.asrandvar(A0),
            b=rvs.asrandvar(problem.b),
        )

    @classmethod
    def from_scalar(
        cls,
        scalar: float,
        problem: LinearSystem,
    ) -> "LinearSystemBelief":
        r"""Construct a belief about the linear system from a scalar.

        Returns a belief about the linear system assuming scalar prior means
        :math:`A_0 = H_0^{-1} = \alpha I` for the system matrix and inverse model.

        Parameters
        ----------
        scalar :
            Scalar parameter defining prior mean(s) of matrix models.
        problem :
            Linear system to solve.
        """
        A0 = linops.ScalarMult(scalar=scalar, shape=problem.A.shape)
        Ainv0 = linops.ScalarMult(scalar=1 / scalar, shape=problem.A.shape)
        return cls.from_matrices(A0=A0, Ainv0=Ainv0, problem=problem)

    def _induced_solution_belief(self) -> rvs.RandomVariable:
        r"""Induced belief about the solution from a belief about the inverse.

        Computes the induced belief about the solution given by (an approximation
        to) the random variable :math:`x=Hb`. This assumes independence between
        :math:`H` and :math:`b`.
        """
        return self.Ainv @ self.b

    def _induced_solution_cov(self) -> Union[np.ndarray, linops.LinearOperator]:
        r"""Induced covariance of the belief about the solution.

        Approximates the covariance of the induced random variable :math:`x=Hb`. This
        assumes independence between :math:`H` and :math:`b`.
        """
        raise NotImplementedError
