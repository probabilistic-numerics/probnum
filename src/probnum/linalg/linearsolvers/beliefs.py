"""Linear system beliefs.

Classes representing probabilistic (prior) beliefs over the quantities
of interest of a linear system such as its solution, the matrix inverse
or spectral information.
"""

from typing import Tuple, Union

import numpy as np
import scipy.linalg

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSystemBelief",
    "WeakMeanCorrespondenceBelief",
    "NoisyLinearSystemBelief",
]

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
    >>> from probnum.linalg.linearsolvers.beliefs import LinearSystemBelief
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
        self._x = rvs.asrandvar(x)
        self._A = rvs.asrandvar(A)
        self._Ainv = rvs.asrandvar(Ainv)
        self._b = rvs.asrandvar(b)

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
        if self._x is None and self.Ainv is not None and self.b is not None:
            return self._induced_solution_belief(Ainv=self.Ainv, b=self.b)
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
        x0: Union[np.ndarray, rvs.RandomVariable],
        problem: LinearSystem,
    ) -> "LinearSystemBelief":
        """Construct a belief over the linear system from an approximate solution.

        Constructs a matrix-variate prior mean for :math:`H` from an initial
        guess of the solution :math:`x0` and the right hand side :math:`b` such
        that :math:`H_0b = x_0`, :math:`H_0` symmetric positive definite and
        :math:`A_0 = H_0^{-1}`. If

        For a detailed construction see Proposition S5 of Wenger and Hennig,
        2020. [#]_

        Parameters
        ----------
        x0 :
            Initial guess for the solution of the linear system.
        problem :
            Linear system to solve.

        References
        ----------
        .. [#] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for
               Machine Learning, *Advances in Neural Information Processing Systems (
               NeurIPS)*, 2020
        """
        if x0.ndim < 2:
            x0 = x0.reshape((-1, 1))

        # If b = 0, set x0 = 0
        if np.all(problem.b == np.zeros_like(problem.b)):
            print("Right-hand-side is zero. Initializing with solution x0 = 0.")
            x0 = problem.b
            A0 = linops.Identity(shape=problem.A.shape)
            A = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))
            Ainv = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))

            return cls(
                x=rvs.Normal(
                    mean=x0,
                    cov=linops.ScalarMult(
                        scalar=np.finfo(float).eps, shape=problem.A.shape
                    ),
                ),
                Ainv=Ainv,
                A=A,
                b=rvs.Constant(support=problem.b),
            )
        else:
            bx0 = (problem.b.T @ x0).item()
            bb = np.linalg.norm(problem.b) ** 2
            # If inner product <x0, b> is non-positive, choose better initialization.
            if bx0 < -100 * np.finfo(float).eps:
                x0 = -x0
                bx0 = -bx0
                print("Better initialization found, setting x0 = - x0.")
            elif np.abs(bx0) < 100 * np.finfo(float).eps:
                print("Better initialization found, setting x0 = (b'b/b'Ab) * b.")
                bAb = np.squeeze(problem.b.T @ (problem.A @ problem.b))
                x0 = bb / bAb * problem.b
                bx0 = bb ** 2 / bAb

            # Construct prior mean of A and Ainv
            alpha = 0.5 * bx0 / bb

            def _mv(v):
                return (x0 - alpha * problem.b) * (x0 - alpha * problem.b).T @ v

            def _mm(M):
                return (x0 - alpha * problem.b) @ (x0 - alpha * problem.b).T @ M

            Ainv_mean = linops.ScalarMult(
                scalar=alpha, shape=problem.A.shape
            ) + 2 / bx0 * linops.LinearOperator(
                matvec=_mv, matmat=_mm, shape=problem.A.shape
            )
            Ainv_cov = linops.SymmetricKronecker(
                A=linops.Identity(shape=problem.A.shape)
            )
            Ainv = rvs.Normal(mean=Ainv_mean, cov=Ainv_cov)

            A_mean = linops.ScalarMult(scalar=1 / alpha, shape=problem.A.shape) - 1 / (
                alpha * np.squeeze((x0 - alpha * problem.b).T @ x0)
            ) * linops.LinearOperator(matvec=_mv, matmat=_mm, shape=problem.A.shape)
            A_cov = linops.SymmetricKronecker(A=linops.Identity(shape=problem.A.shape))
            A = rvs.Normal(mean=A_mean, cov=A_cov)

            return cls(
                x=rvs.Normal(
                    mean=x0, cov=cls._induced_solution_cov(Ainv=Ainv, b=problem.b)
                ),
                Ainv=Ainv,
                A=A,
                b=rvs.Constant(support=problem.b),
            )

    @classmethod
    def from_inverse(
        cls,
        Ainv0: Union[np.ndarray, rvs.RandomVariable, linops.LinearOperator],
        problem: LinearSystem,
    ) -> "LinearSystemBelief":
        r"""Construct a belief over the linear system from an approximate inverse.

        Returns a belief over the linear system from an approximate inverse
        :math:`H_0\approx A^{-1}` such as a preconditioner.

        Parameters
        ----------
        Ainv0 :
            Approximate inverse of the system matrix.
        problem :
            Linear system to solve.
        """
        if isinstance(Ainv0, (np.ndarray, linops.LinearOperator)):
            Ainv0 = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))

        A0 = rvs.Normal(mean=problem.A, cov=linops.SymmetricKronecker(A=problem.A))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=Ainv0,
            A=A0,
            b=problem.b,
        )

    @classmethod
    def from_matrix(
        cls,
        A0: Union[np.ndarray, rvs.RandomVariable],
        problem: LinearSystem,
    ) -> "LinearSystemBelief":
        r"""Construct a belief over the linear system from an approximate system matrix.

        Returns a belief over the linear system from an approximation of
        the system matrix :math:`A_0\approx A`.

        Parameters
        ----------
        A0 :
            Approximate system matrix.
        problem :
            Linear system to solve.
        """
        if isinstance(A0, (np.ndarray, linops.LinearOperator)):
            A0 = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))

        Ainv0_mean = linops.Identity(shape=A0.shape)
        Ainv0 = rvs.Normal(mean=Ainv0_mean, cov=linops.SymmetricKronecker(A=Ainv0_mean))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=Ainv0,
            A=A0,
            b=problem.b,
        )

    @classmethod
    def from_matrices(
        cls,
        A0: Union[np.ndarray, rvs.RandomVariable],
        Ainv0: Union[np.ndarray, rvs.RandomVariable],
        problem: LinearSystem,
    ) -> "LinearSystemBelief":
        r"""Construct a belief from an approximate system matrix and
        corresponding inverse.

        Returns a belief over the linear system from an approximation of
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
        if isinstance(A0, (np.ndarray, linops.LinearOperator)):
            A0 = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))
        if isinstance(Ainv0, (np.ndarray, linops.LinearOperator)):
            Ainv0 = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=Ainv0,
            A=A0,
            b=problem.b,
        )

    @staticmethod
    def _induced_solution_cov(
        Ainv: rvs.Normal, b: rvs.RandomVariable
    ) -> linops.LinearOperator:
        r"""Induced covariance of the belief over the solution.

        Approximates the covariance :math:`\Sigma` of the induced random variable
        :math:`x=Hb` for :math:`H \sim \mathcal{N}(H_0, W \otimes_s W)` such that
        :math:`\Sigma=\frac{1}{2}(Wb^\top Wb + Wb b^\top W)`.

        Parameters
        ----------
        Ainv :
            Belief over the (pseudo-)inverse of the system matrix.
        b :
            Belief over the right hand side
        """
        b = rvs.asrandvar(b)
        Wb = Ainv.cov.A @ b.mean
        bWb = (Wb.T @ b.mean).item()

        def _mv(x):
            return 0.5 * (bWb * Ainv.cov.A @ x + Wb @ (Wb.T @ x))

        return linops.LinearOperator(
            shape=Ainv.shape, dtype=float, matvec=_mv, matmat=_mv
        )

    @staticmethod
    def _induced_solution_belief(Ainv: rvs.Normal, b: rvs.RandomVariable) -> rvs.Normal:
        r"""Induced belief over the solution from a belief over the inverse.

        Approximates the induced random variable :math:`x=Hb` for :math:`H \sim
        \mathcal{N}(H_0, W \otimes_s W)`, such that :math:`x \sim \mathcal{N}(\mu,
        \Sigma)` with :math:`\mu=\mathbb{E}[H]\mathbb{E}[b]` and :math:`\Sigma=\frac{
        1}{2}(Wb^\top Wb + Wb b^\top W)`.

        Parameters
        ----------
        Ainv :
            Belief over the (pseudo-)inverse of the system matrix.
        b :
            Belief over the right hand side
        """
        b = rvs.asrandvar(b)
        return rvs.Normal(
            mean=Ainv.mean @ b.mean,
            cov=LinearSystemBelief._induced_solution_cov(Ainv=Ainv, b=b),
        )


class WeakMeanCorrespondenceBelief(LinearSystemBelief):
    r"""Belief enforcing (weak) mean correspondence.

    Belief over the linear system such that the means over the matrix and its inverse
    correspond and the covariance symmetric Kronecker factors act like :math:`A` and the
    approximate inverse :math:`H_0` on the spaces spanned by the actions and
    observations. On the respective orthogonal spaces the uncertainty over the matrix
    and its inverse is determined by scaling parameters.

    For a scalar prior mean :math:`A_0 = H_0^{-1} = \alpha I`, when paired with a
    :class:`~probnum.linalg.linearsolvers.policies.ConjugateDirections`
    policy and linear observations, this (prior) belief recovers the *method of
    conjugate gradients*. [1]_

    For more details, see Wenger and Hennig, 2020. [1]_

    Parameters
    ----------
    A :
        System matrix :math:`A`.
    Ainv0 :
        Approximate matrix inverse :math:`H_0 \approx A^{-1}`.
    actions :
        Actions to probe the linear system with.
    observations :
        Observations of the linear system for the given actions.
    unc_scale_A :
        Uncertainty scaling :math:`\Phi` of the belief over the matrix in the unexplored
        space :math:`\operatorname{span}(s_1, \dots, s_k)^\perp`.
    unc_scale_A :
        Uncertainty scaling :math:`\Psi` of the belief over the inverse in the
        unexplored space :math:`\operatorname{span}(y_1, \dots, y_k)^\perp`.

    Notes
    -----
    This construction fulfills *weak posterior correspondence* [1]_ meaning on the space
    spanned by the observations :math:`y_i` it holds that :math:`\mathbb{
    E}[A]^{-1}y = \mathbb{E}[H]y` for all :math:`y \in \operatorname{span}(y_1,
    \dots, y_k)`.

    References
    ----------
    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning,
       *Advances in Neural Information Processing Systems (NeurIPS)*, 2020

    See Also
    --------
    LinearSystemBelief : Belief over quantities of interest of a linear system.

    Examples
    --------
    """

    def __init__(
        self,
        A,
        Ainv0,
        actions=None,
        observations=None,
        unc_scale_A: float = 0.0,
        unc_scale_Ainv: float = 0.0,
    ):
        self.A0 = A
        self.Ainv0 = Ainv0
        self.actions = (actions,)
        self.observations = (observations,)
        self.unc_scale_A = unc_scale_A
        self.unc_scale_Ainv = unc_scale_Ainv

        # Construct beliefs

        super().__init__(x=x, A=A, Ainv=Ainv, b=b)

    @cached_property
    def _pseudo_inverse_actions(self) -> np.ndarray:
        raise NotImplementedError

    @cached_property
    def _pseudo_inverse_observations(self) -> np.ndarray:
        raise NotImplementedError

    def _matrix_covariance_factor(self):
        """"""
        # Ensure prior covariance class only acts in span(S) like A
        def _matvec(x):
            # First term of calibration covariance class: AS(S'AS)^{-1}S'A
            return (Y * sy ** -1) @ (Y.T @ x.ravel())

        A_covfactor0 = linops.LinearOperator(shape=self.A0.shape, matvec=_matvec)
        return rvs.Normal(mean=self.A0, cov=linops.SymmetricKronecker(A=A_covfactor0))

    def _inverse_covariance_factor(self):
        """"""

        def _matvec(x):
            # Term in covariance class: A_0^{-1}Y(Y'A_0^{-1}Y)^{-1}Y'A_0^{-1}
            # TODO: for efficiency ensure that we dont have to compute
            #   (Y.T Y)^{-1} two times! For a scalar mean this is the same as in
            #   the null space projection
            #   Use cached pseudo inverse to do this if A0 /Ainv0 is scalar
            YAinv0Y_inv_YAinv0x = np.linalg.solve(
                Y.T @ (self.Ainv0 @ Y), Y.T @ (self.Ainv0 @ x)
            )
            return self.Ainv0 @ (Y @ YAinv0Y_inv_YAinv0x)

        Ainv_covfactor0 = linops.LinearOperator(shape=self.Ainv0.shape, matvec=_matvec)

    def _scaled_null_space_projection(
        self, M: np.ndarray, scale: float
    ) -> Union[np.ndarray, linops.LinearOperator]:
        r"""Compute a scaled null space projection.

        Returns a linear operator which maps to the null space of :math:`M \in
        \mathbb{R}^{n \times k}` and applies a given scaling.
        """
        # Null space basis via SVD
        # Complexity O(n^2k) for k <= n (Matrix Computations - Golub, Van Loan)
        null_space_basis = scipy.linalg.null_space(M)
        # TODO cache this using cached_property

        def scaled_projection(x: np.ndarray):
            return scale * null_space_basis @ null_space_basis.T @ x

        return linops.LinearOperator(
            shape=(M.shape[0], M.shape[0]), matvec=scaled_projection, dtype=float
        )

    @classmethod
    def from_inverse(
        cls,
        Ainv0: Union[np.ndarray, rvs.RandomVariable, linops.LinearOperator],
        problem: LinearSystem,
    ) -> "WeakMeanCorrespondenceBelief":
        r"""Construct a belief over the linear system from an approximate inverse.

        Returns a belief over the linear system from an approximate inverse
        :math:`H_0\approx A^{-1}` such as a preconditioner. This internally inverts
        (the prior mean of) :math:`H_0`, which may be computationally costly.

        Parameters
        ----------
        Ainv0 :
            Approximate inverse of the system matrix.
        problem :
            Linear system to solve.
        """
        if isinstance(Ainv0, (np.ndarray, linops.LinearOperator)):
            Ainv0 = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))

        # Weak (symmetric) mean correspondence
        try:
            A0_mean = Ainv0.mean.inv()
        except AttributeError as exc:
            raise AttributeError(
                "Cannot efficiently invert (prior mean of) Ainv. "
                "Additionally, specify a prior (mean) of A"
                "instead."
            ) from exc
        A0 = rvs.Normal(mean=A0_mean, cov=linops.SymmetricKronecker(A=problem.A))

        return super().__init__(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=Ainv0,
            A=A0,
            b=problem.b,
        )

    @classmethod
    def from_matrix(
        cls,
        A0: Union[np.ndarray, rvs.RandomVariable],
        problem: LinearSystem,
    ) -> "WeakMeanCorrespondenceBelief":
        r"""Construct a belief over the linear system from an approximate system matrix.

        Returns a belief over the linear system from an approximation of
        the system matrix :math:`A_0\approx A`. This internally inverts (the prior mean
        of) :math:`A_0`, which may be computationally costly.

        Parameters
        ----------
        A0 :
            Approximate system matrix.
        problem :
            Linear system to solve.
        """
        if isinstance(A0, (np.ndarray, linops.LinearOperator)):
            A0 = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))

        # Weak (symmetric) mean correspondence
        try:
            Ainv0_mean = A0.mean.inv()
        except AttributeError as exc:
            raise AttributeError(
                "Cannot efficiently invert (prior mean of) A. "
                "Additionally, specify an inverse prior (mean) "
                "instead."
            ) from exc
        Ainv0 = rvs.Normal(mean=Ainv0_mean, cov=linops.SymmetricKronecker(A=Ainv0_mean))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=Ainv0,
            A=A0,
            b=problem.b,
        )


class NoisyLinearSystemBelief(LinearSystemBelief):
    r"""Belief over a noise-corrupted linear system.

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
    noise_scale :
        Estimate for the scale of the noise on the system matrix.

    """

    def __init__(
        self,
        x: rvs.RandomVariable,
        A: rvs.RandomVariable,
        Ainv: rvs.RandomVariable,
        b: rvs.RandomVariable,
        noise_scale: float = None,
    ):
        self._noise_scale = noise_scale
        super().__init__(x=x, A=A, Ainv=Ainv, b=b)

    @property
    def noise_scale(self) -> float:
        """Estimate for the scale of the noise on the system matrix."""
        return self._noise_scale
