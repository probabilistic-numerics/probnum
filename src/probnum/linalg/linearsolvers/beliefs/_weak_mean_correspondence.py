"""Belief assuming weak correspondence between the means of the matrix models."""

from typing import List, Optional, Union

import numpy as np

import probnum
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers.beliefs._linear_system_belief import (
    LinearSystemBelief,
)
from probnum.linalg.linearsolvers.hyperparam_optim import UncertaintyCalibration
from probnum.problems import LinearSystem


class WeakMeanCorrespondenceBelief(LinearSystemBelief):
    r"""Belief enforcing (weak) mean correspondence.

    Belief over the linear system such that the means over the matrix and its inverse
    correspond and the covariance symmetric Kronecker factors act like :math:`A` and the
    approximate inverse :math:`H_0` on the spaces spanned by the actions and
    observations. On the respective orthogonal spaces the uncertainty over the matrix
    and its inverse is determined by scaling parameters.

    For a scalar prior mean :math:`A_0 = H_0^{-1} = \alpha I` with :math:`\alpha > 0`,
    a :class:`~probnum.linalg.linearsolvers.policies.ConjugateDirections`
    policy and linear observations, this (prior) belief recovers the *method of
    conjugate gradients*.

    For more details, see Wenger and Hennig, 2020. [1]_

    Parameters
    ----------
    A :
        Approximate system matrix :math:`A_0 \approx A`.
    Ainv0 :
        Approximate matrix inverse :math:`H_0 \approx A^{-1}`.
    phi :
        Uncertainty scaling :math:`\Phi` of the belief over the matrix in the unexplored
        action space :math:`\operatorname{span}(s_1, \dots, s_k)^\perp`.
    psi :
        Uncertainty scaling :math:`\Psi` of the belief over the inverse in the
        unexplored observation space :math:`\operatorname{span}(y_1, \dots, y_k)^\perp`.
    actions :
        Actions to probe the linear system with.
    observations :
        Observations of the linear system for the given actions.
    action_projection_innerprods :
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

    >>> from probnum.linalg.linearsolvers.beliefs import WeakMeanCorrespondenceBelief
    >>> from probnum.linops import ScalarMult
    >>> from probnum.problems.zoo.linalg import random_spd_matrix
    >>> # Linear system
    >>> np.random.seed(1)
    >>> linsys = LinearSystem.from_matrix(random_spd_matrix(dim=5))
    >>> # Prior belief
    >>> prior = WeakMeanCorrespondenceBelief.from_scalar(alpha=2.5, problem=linsys)
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
        A0: Union[np.ndarray, linops.LinearOperator],
        Ainv0: Union[np.ndarray, linops.LinearOperator],
        b: rvs.RandomVariable,
        phi: float = 1.0,
        psi: float = 1.0,
        actions: Optional[Union[List, np.ndarray]] = None,
        observations: Optional[Union[List, np.ndarray]] = None,
        action_projection_innerprods: Optional[np.ndarray] = None,
        calibration_method: Optional[UncertaintyCalibration] = None,
    ):
        self.A0 = A0
        self.Ainv0 = Ainv0
        self.unc_scale_A = phi
        self.unc_scale_Ainv = psi
        self.calibration_method = calibration_method
        if actions is None or observations is None:
            self.actions = None
            self.observations = None
            # For no actions taken, the uncertainty scales determine the overall
            # uncertainty, since :math:`{0}^\perp=\mathbb{R}^n`.
            A = rvs.Normal(
                mean=A0,
                cov=linops.SymmetricKronecker(
                    linops.ScalarMult(scalar=phi, shape=A0.shape)
                ),
            )
            Ainv = rvs.Normal(
                mean=Ainv0,
                cov=linops.SymmetricKronecker(
                    linops.ScalarMult(scalar=psi, shape=Ainv0.shape)
                ),
            )
        else:
            self.actions = (
                actions if isinstance(actions, np.ndarray) else np.hstack(actions)
            )
            self.observations = (
                observations
                if isinstance(observations, np.ndarray)
                else np.hstack(observations)
            )
            if action_projection_innerprods is not None:
                self.action_projection_innerprods = action_projection_innerprods
            else:
                self.action_projection_innerprods = self.actions.T @ self.observations
            A = rvs.Normal(
                mean=A0, cov=linops.SymmetricKronecker(self._cov_factor_matrix())
            )
            Ainv = rvs.Normal(
                mean=Ainv0,
                cov=linops.SymmetricKronecker(self._cov_factor_inverse()),
            )

        super().__init__(
            x=super()._induced_solution_belief(Ainv=Ainv, b=b), Ainv=Ainv, A=A, b=b
        )

    def _cov_factor_matrix(self) -> linops.LinearOperator:
        r"""Covariance factor of the system matrix model.

        Computes the covariance factor :math:`W_0^A = Y(S^\top Y)^{-1}Y_^\top + P_{
        S^\perp}\Phi P_{S^\perp}` as given in eqn. (3) of Wenger and Hennig, 2020.
        """
        action_proj = linops.OrthogonalProjection(subspace_basis=self.actions)

        if self.action_projection_innerprods.ndim == 1:

            def _matvec(x):
                """Conjugate actions implying :math:`S^{\top} Y` is a diagonal
                matrix."""
                return (self.observations * self.action_projection_innerprods ** -1) @ (
                    self.observations.T @ x
                )

        else:

            def _matvec(x):
                return self.observations @ np.linalg.solve(
                    self.action_projection_innerprods,
                    self.observations.T @ x,
                )

        action_space_op = linops.LinearOperator(
            shape=self.A0.shape, matvec=_matvec, matmat=_matvec, dtype=float
        )

        orthogonal_space_op = self.unc_scale_A * (
            linops.Identity(shape=self.A0.shape) - action_proj
        )

        return action_space_op + orthogonal_space_op

    def _cov_factor_inverse(self) -> linops.LinearOperator:
        r"""Covariance factor of the inverse model.

        Computes the covariance factor :math:`W_0^H = A_0^{-1}Y(Y^\top A_0^{-1}Y)^{
        -1}Y_^\top A_0^{-1} + P_{Y^\perp}\Psi P_{Y^\perp}` as given in eqn. (3) of
        Wenger and Hennig, 2020.
        """
        observation_proj = linops.OrthogonalProjection(subspace_basis=self.observations)

        if isinstance(self.Ainv0, linops.ScalarMult):
            observation_space_op = self.Ainv0.scalar * observation_proj
        else:

            def _matvec(x):
                return self.Ainv0 @ (
                    linops.OrthogonalProjection(
                        subspace_basis=self.observations,
                        is_orthonormal=False,
                        innerprod_matrix=self.Ainv0,
                    )
                    @ x
                )

            observation_space_op = linops.LinearOperator(
                shape=self.A0.shape, matvec=_matvec, matmat=_matvec, dtype=float
            )

        orthogonal_space_op = self.unc_scale_Ainv * (
            linops.Identity(shape=self.A0.shape) - observation_proj
        )

        return observation_space_op + orthogonal_space_op

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
        try:
            return cls(
                A0=Ainv0.inv(),  # Ensure (weak) mean correspondence
                Ainv0=Ainv0,
                b=problem.b,
            )
        except AttributeError as exc:
            raise TypeError(
                "Cannot efficiently invert (prior mean of) Ainv. "
                "Additionally, specify a prior (mean) of A instead or wrap into"
                "a linear operator with an .inv() function."
            ) from exc

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
        try:
            return cls(
                A0=A0,
                Ainv0=A0.inv(),  # Ensure (weak) mean correspondence
                b=problem.b,
            )
        except AttributeError as exc:
            raise TypeError(
                "Cannot efficiently invert (prior mean of) A. "
                "Additionally, specify an inverse prior (mean) instead or wrap into"
                "a linear operator with an .inv() function."
            ) from exc

    @classmethod
    def from_matrices(
        cls,
        A0: Union[np.ndarray, rvs.RandomVariable],
        Ainv0: Union[np.ndarray, rvs.RandomVariable],
        problem: LinearSystem,
    ) -> "WeakMeanCorrespondenceBelief":
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
        return cls(A0=A0, Ainv0=Ainv0, b=problem.b)

    @classmethod
    def from_scalar(
        cls, alpha: float, problem: LinearSystem
    ) -> "WeakMeanCorrespondenceBelief":
        r"""Construct a belief over the linear system from a scalar.

        Returns a belief over the linear system assuming scalar prior means
        :math:`A_0 = H_0^{-1} = \alpha I` for the system matrix and inverse model.

        Parameters
        ----------
        alpha :
            Scalar parameter defining prior mean(s) of matrix models.
        problem :
            Linear system to solve.
        """
        if alpha <= 0.0:
            raise ValueError(f"Scalar parameter alpha={alpha:.4f} must be positive.")
        A0 = linops.ScalarMult(scalar=alpha, shape=problem.A.shape)
        Ainv0 = linops.ScalarMult(scalar=1 / alpha, shape=problem.A.shape)
        return cls.from_matrices(A0=A0, Ainv0=Ainv0, problem=problem)

    def optimize_hyperparams(
        self,
        problem: LinearSystem,
        actions: List[np.ndarray],
        observations: List[np.ndarray],
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Optional["probnum.linalg.linearsolvers.LinearSolverState"]:
        """Calibrate the uncertainty scales :math:`Phi` and :math:`Psi`."""
        if self.calibration_method is not None:
            (phi, psi), solver_state = self.calibration_method(
                problem=problem,
                belief=self,
                actions=actions,
                observations=observations,
                solver_state=solver_state,
            )
            self.unc_scale_A = phi
            self.unc_scale_Ainv = psi
            return solver_state
        else:
            raise NotImplementedError
