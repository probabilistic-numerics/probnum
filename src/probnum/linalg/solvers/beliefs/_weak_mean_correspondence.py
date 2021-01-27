"""Belief assuming weak correspondence between the means of the matrix models."""

from typing import List, Optional, Union

import numpy as np
import scipy.sparse

import probnum
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.solvers.beliefs._symmetric_normal_linear_system import (
    SymmetricNormalLinearSystemBelief,
)
from probnum.linalg.solvers.data import LinearSolverData
from probnum.linalg.solvers.hyperparams import UncertaintyUnexploredSpace
from probnum.problems import LinearSystem
from probnum.type import MatrixArgType

# pylint: disable="invalid-name"

# Public classes and functions. Order is reflected in documentation.
__all__ = ["WeakMeanCorrespondenceBelief"]


class WeakMeanCorrespondenceBelief(SymmetricNormalLinearSystemBelief):
    r"""Symmetric Gaussian belief enforcing weak mean correspondence.

    Belief over the linear system such that the means over the matrix and its inverse
    correspond and the covariance symmetric Kronecker factors act like :math:`A` and the
    approximate inverse :math:`H_0` on the spaces spanned by the actions and
    observations. On the respective orthogonal spaces the uncertainty over the matrix
    and its inverse is determined by scaling parameters.

    For a scalar prior mean :math:`A_0 = H_0^{-1} = \alpha I` with :math:`\alpha > 0`,
    a :class:`~probnum.linalg.solvers.policies.ConjugateDirections`
    policy and linear observations, this (prior) belief recovers the *method of
    conjugate gradients*.

    For more details, see Wenger and Hennig, 2020. [1]_

    Parameters
    ----------
    A0 :
        Approximate system matrix :math:`A_0 \approx A`.
    Ainv0 :
        Approximate matrix inverse :math:`H_0 \approx A^{-1}`.
    uncertainty_scales :
        Uncertainty scales of the system matrix and inverse model in the respective
        null spaces of the actions and observations.
    data :
        Actions and observations of the linear system.
    action_obs_innerprods :
        Inner product(s) :math:`(S^\top Y)_{ij} = s_i^\top y_j` of actions
        and observations. If a vector is passed, actions are assumed to be
        :math:`A`-conjugate, i.e. :math:`s_i^\top A s_j =0` for :math:`i \neq j`.

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
    Belief recovering the method of conjugate gradients.

    >>> import numpy as np
    >>> from probnum.linalg.solvers.beliefs import WeakMeanCorrespondenceBelief
    >>> from probnum.linops import ScalarMult
    >>> from probnum.problems import LinearSystem
    >>> from probnum.problems.zoo.linalg import random_spd_matrix
    >>> # Linear system
    >>> np.random.seed(1)
    >>> linsys = LinearSystem.from_matrix(random_spd_matrix(dim=5))
    >>> # Prior belief
    >>> prior = WeakMeanCorrespondenceBelief.from_scalar(scalar=2.5, problem=linsys)
    >>> # Initial residual / gradient: r0 = A x0 - b
    >>> residual = linsys.A @ prior.x.mean - linsys.b
    >>> residual
    array([[-72.19081319],
           [-55.35996205],
           [ 57.51106765],
           [-61.94224112],
           [ 29.72231145]])
    """

    def __init__(
        self,
        A0: MatrixArgType,
        Ainv0: MatrixArgType,
        b: Union[rvs.Constant, np.ndarray],
        uncertainty_scales: UncertaintyUnexploredSpace = UncertaintyUnexploredSpace(
            Phi=1.0, Psi=1.0
        ),
        data: Optional[LinearSolverData] = None,
        action_obs_innerprods: Optional[np.ndarray] = None,
    ):
        self.A0 = A0
        self.Ainv0 = Ainv0
        self._A_covfactor_update_op = None
        self._Ainv_covfactor_update_op = None
        self.data = data

        cov_factor_A = self._cov_factor_matrix(
            uncertainty_scales=uncertainty_scales,
            action_obs_innerprods=action_obs_innerprods,
        )
        cov_factor_Ainv = self._cov_factor_inverse(
            uncertainty_scales=uncertainty_scales
        )

        A = rvs.Normal(
            mean=self.A0,
            cov=linops.SymmetricKronecker(A=cov_factor_A),
        )
        Ainv = rvs.Normal(
            mean=self.Ainv0,
            cov=linops.SymmetricKronecker(A=cov_factor_Ainv),
        )

        super().__init__(x=None, Ainv=Ainv, A=A, b=b, hyperparams=uncertainty_scales)

    def _cov_factor_matrix(
        self,
        uncertainty_scales: UncertaintyUnexploredSpace,
        action_obs_innerprods: Optional[np.ndarray] = None,
    ) -> linops.LinearOperator:
        r"""Covariance factor of the system matrix model.

        Computes the covariance factor :math:`W_0^A = Y(S^\top Y)^{-1}Y_^\top + P_{
        S^\perp}\Phi P_{S^\perp}` as given in eqn. (3) of Wenger and Hennig, 2020.

        Parameters
        ----------
        uncertainty_scales :
            Uncertainty scales of the system matrix and inverse model in the respective
            null spaces of the actions and observations.
        action_obs_innerprods :
            Inner product(s) :math:`(S^\top Y)_{ij} = s_i^\top y_j` of actions
            and observations. If a vector is passed, actions and observations are
            assumed to be conjugate, i.e. :math:`s_i^\top y_j =0` for :math:`i \neq j`.
        """
        if self.data is None:
            # For no actions taken, the uncertainty scales determine the overall
            # uncertainty, since :math:`{0}^\perp=\mathbb{R}^n`.
            return linops.ScalarMult(scalar=uncertainty_scales.Phi, shape=self.A0.shape)
        else:
            if action_obs_innerprods is None:
                action_obs_innerprods = np.squeeze(
                    self.data.actions_arr.actA.T @ self.data.observations_arr.obsA
                )

            action_proj = linops.OrthogonalProjection(
                subspace_basis=self.data.actions_arr.actA
            )

            if np.squeeze(action_obs_innerprods).ndim in (0, 1):

                def _matvec(x):
                    """Conjugate actions implying :math:`S^{\top} Y` is a diagonal
                    matrix."""
                    return (
                        self.data.observations_arr.obsA * action_obs_innerprods ** -1
                    ) @ (self.data.observations_arr.obsA.T @ x)

            else:

                def _matvec(x):
                    return self.data.observations_arr.obsA @ np.linalg.solve(
                        action_obs_innerprods,
                        self.data.observations_arr.obsA.T @ x,
                    )

            action_space_op = linops.LinearOperator(
                shape=self.A0.shape, matvec=_matvec, matmat=_matvec, dtype=float
            )

            orthogonal_space_op = uncertainty_scales.Phi * (
                linops.Identity(shape=self.A0.shape) - action_proj
            )
            return action_space_op + orthogonal_space_op

    def _cov_factor_inverse(
        self, uncertainty_scales: UncertaintyUnexploredSpace
    ) -> linops.LinearOperator:
        r"""Covariance factor of the inverse model.

        Computes the covariance factor :math:`W_0^H = A_0^{-1}Y(Y^\top A_0^{-1}Y)^{
        -1}Y_^\top A_0^{-1} + P_{Y^\perp}\Psi P_{Y^\perp}` as given in eqn. (3) of
        Wenger and Hennig, 2020.

        Parameters
        ----------
        uncertainty_scales :
            Uncertainty scales of the system matrix and inverse model in the respective
            null spaces of the actions and observations.
        """
        if self.data is None:
            # For no actions taken, the uncertainty scales determine the overall
            # uncertainty, since :math:`{0}^\perp=\mathbb{R}^n`.
            return linops.ScalarMult(
                scalar=uncertainty_scales.Psi, shape=self.Ainv0.shape
            )
        else:
            observation_proj = linops.OrthogonalProjection(
                subspace_basis=self.data.observations_arr.obsA
            )

            if isinstance(self.Ainv0, linops.ScalarMult):
                observation_space_op = self.Ainv0.scalar * observation_proj
            else:

                def _matvec(x):
                    return self.Ainv0 @ (
                        linops.OrthogonalProjection(
                            subspace_basis=self.data.observations_arr.obsA,
                            is_orthonormal=False,
                            innerprod_matrix=self.Ainv0,
                        )
                        @ x
                    )

                observation_space_op = linops.LinearOperator(
                    shape=self.A0.shape, matvec=_matvec, matmat=_matvec, dtype=float
                )

            orthogonal_space_op = uncertainty_scales.Psi * (
                linops.Identity(shape=self.A0.shape) - observation_proj
            )

            return observation_space_op + orthogonal_space_op

    @property
    def hyperparams(self) -> Optional[UncertaintyUnexploredSpace]:
        r"""Uncertainty scales :math:`\Phi` and :math:`\Psi` in the respective
        unexplored spaces."""
        return self._hyperparams

    @classmethod
    def from_solution(
        cls,
        x0: np.ndarray,
        problem: LinearSystem,
        check_for_better_x0: bool = True,
    ) -> "WeakMeanCorrespondenceBelief":
        _, Ainv0, A0, b0 = cls._belief_means_from_solution(
            x0=x0, problem=problem, check_for_better_x0=check_for_better_x0
        )
        return cls(
            Ainv0=Ainv0,
            A0=A0,
            b=rvs.asrandvar(b0),
        )

    @classmethod
    def from_inverse(
        cls,
        Ainv0: MatrixArgType,
        problem: LinearSystem,
        data: Optional[LinearSolverData] = None,
    ) -> "WeakMeanCorrespondenceBelief":
        r"""Construct a belief about the linear system from an approximate inverse.

        Returns a belief about the linear system from an approximate inverse
        :math:`H_0\approx A^{-1}` such as a preconditioner. By default this internally
        inverts (the prior mean of) :math:`H_0`, which may be computationally costly.

        Parameters
        ----------
        Ainv0 :
            Approximate inverse of the system matrix.
        problem :
            Linear system to solve.
        data :
            Actions and observations of the linear system.
        """
        try:
            A0 = Ainv0.inv()
        except AttributeError as exc:
            raise TypeError(
                "Cannot efficiently invert (prior mean of) Ainv. "
                "Additionally, specify a prior (mean) of A instead or wrap into "
                "a linear operator with an .inv() function."
            ) from exc

        return cls(
            A0=A0,  # Ensure (weak) mean correspondence
            Ainv0=Ainv0,
            b=rvs.asrandvar(problem.b),
            data=data,
        )

    @classmethod
    def from_matrix(
        cls,
        A0: MatrixArgType,
        problem: LinearSystem,
        data: Optional[LinearSolverData] = None,
    ) -> "WeakMeanCorrespondenceBelief":
        r"""Construct a belief about the linear system from an approximate system matrix.

        Returns a belief about the linear system from an approximation of
        the system matrix :math:`A_0\approx A`. This internally inverts (the prior mean
        of) :math:`A_0`, which may be computationally costly.

        Parameters
        ----------
        A0 :
            Approximate system matrix.
        problem :
            Linear system to solve.
        data :
            Actions and observations of the linear system.
        """
        try:
            Ainv0 = A0.inv()
        except AttributeError as exc:
            raise TypeError(
                "Cannot efficiently invert (prior mean of) A. "
                "Additionally, specify an inverse prior (mean) instead or wrap into "
                "a linear operator with an .inv() function."
            ) from exc
        return cls(
            A0=A0,
            Ainv0=Ainv0,  # Ensure (weak) mean correspondence
            b=rvs.asrandvar(problem.b),
            data=data,
        )

    @classmethod
    def from_matrices(
        cls,
        A0: MatrixArgType,
        Ainv0: MatrixArgType,
        problem: LinearSystem,
        data: Optional[LinearSolverData] = None,
    ) -> "WeakMeanCorrespondenceBelief":
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
        data :
            Actions and observations of the linear system.
        """
        return cls(A0=A0, Ainv0=Ainv0, b=rvs.asrandvar(problem.b), data=data)

    @classmethod
    def from_scalar(
        cls,
        scalar: float,
        problem: LinearSystem,
    ) -> "WeakMeanCorrespondenceBelief":
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
        if scalar <= 0.0:
            raise ValueError(f"Scalar parameter alpha={scalar:.4f} must be positive.")
        A0 = linops.ScalarMult(scalar=scalar, shape=problem.A.shape)
        Ainv0 = linops.ScalarMult(scalar=1 / scalar, shape=problem.A.shape)
        return cls.from_matrices(
            A0=A0,
            Ainv0=Ainv0,
            problem=problem,
        )
