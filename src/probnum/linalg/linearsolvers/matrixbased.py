"""
Matrix-based probabilistic linear solvers.

Implementations of matrix-based linear solvers which perform inference on the matrix or its inverse given linear
observations.
"""
import warnings
import abc

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import GPy

from probnum import prob
from probnum.linalg import linops


class ProbabilisticLinearSolver(abc.ABC):
    """
    An abstract base class for probabilistic linear solvers.

    This class is designed to be subclassed with new (probabilistic) linear solvers, which implement a ``.solve()``
    method. Objects of this type are instantiated in wrapper functions such as :meth:``problinsolve``.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        A square matrix or linear operator. A prior distribution can be provided as a
        :class:`~probnum.prob.RandomVariable`. If an array or linear operator is given, a prior distribution is
        chosen automatically.
    b : RandomVariable, shape=(n,) or (n, nrhs)
        Right-hand side vector, matrix or RandomVariable of :math:`A x = b`.
    """

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.n = A.shape[1]

    def has_converged(self, iter, maxiter, **kwargs):
        """
        Check convergence of a linear solver.

        Evaluates a set of convergence criteria based on its input arguments to decide whether the iteration has converged.

        Parameters
        ----------
        iter : int
            Current iteration of solver.
        maxiter : int
            Maximum number of iterations

        Returns
        -------
        has_converged : bool
            True if the method has converged.
        convergence_criterion : str
            Convergence criterion which caused termination.
        """
        # maximum iterations
        if iter >= maxiter:
            warnings.warn(message="Iteration terminated. Solver reached the maximum number of iterations.")
            return True, "maxiter"
        else:
            return False, ""

    def solve(self, callback=None, **kwargs):
        """
        Solve the linear system :math:`Ax=b`.

        Parameters
        ----------
        callback : function, optional
            User-supplied function called after each iteration of the linear solver. It is called as
            ``callback(xk, Ak, Ainvk, sk, yk, alphak, resid, **kwargs)`` and can be used to return quantities from the
            iteration. Note that depending on the function supplied, this can slow down the solver.
        kwargs
            Key-word arguments adjusting the behaviour of the ``solve`` iteration. These are usually convergence
            criteria.

        Returns
        -------
        x : RandomVariable, shape=(n,) or (n, nrhs)
            Approximate solution :math:`x` to the linear system. Shape of the return matches the shape of ``b``.
        A : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator.
        Ainv : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
        info : dict
            Information on convergence of the solver.

        """
        raise NotImplementedError


class MatrixBasedSolver(ProbabilisticLinearSolver, abc.ABC):
    """
    Abstract class for matrix-based probabilistic linear solvers.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        A square matrix or linear operator. A prior distribution can be provided as a
        :class:`~probnum.prob.RandomVariable`. If an array or linear operator is given, a prior distribution is
        chosen automatically.
    b : RandomVariable, shape=(n,) or (n, nrhs)
        Right-hand side vector, matrix or RandomVariable of :math:`A x = b`.
    x0 : array-like, shape=(n,) or (n, nrhs)
        Optional. Guess for the solution of the linear system.
    """

    def __init__(self, A, b, x0=None):
        self.x0 = x0
        super().__init__(A=A, b=b)

    def _get_prior_params(self, A0, Ainv0, x0, b):
        """
        Parameters
        ----------
        A0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
            A square matrix, linear operator or random variable representing the prior belief over the linear operator
            :math:`A`. If an array or linear operator is given, a prior distribution is chosen automatically.
        Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
            A square matrix, linear operator or random variable representing the prior belief over the inverse
            :math:`H=A^{-1}`. This can be viewed as taking the form of a pre-conditioner. If an array or linear operator is
            given, a prior distribution is chosen automatically.
        x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
            Optional. Prior belief for the solution of the linear system. Will be ignored if ``A0`` or ``Ainv0`` is
            given.
        b : array_like, shape=(n,) or (n, nrhs)
            Right-hand side vector or matrix in :math:`A x = b`.
        """
        raise NotImplementedError

    def _construct_symmetric_matrix_prior_means(self, x0, b):
        """
        Create matrix prior means from an initial guess for the solution of the linear system.

        Constructs a matrix-variate prior mean for H from ``x0`` and ``b`` such that :math:`H_0b = x_0`, :math:`H_0`
        symmetric positive definite and :math:`A_0 = H_0^{-1}`.

        Parameters
        ----------
        x0 : array-like, shape=(n,) or (n, nrhs)
            Optional. Guess for the solution of the linear system.
        b : array_like, shape=(n,) or (n, nrhs)
            Right-hand side vector or matrix in :math:`A x = b`.

        Returns
        -------
        A0_mean : linops.LinearOperator
            Mean of the matrix-variate prior distribution on the system matrix :math:`A`.
        Ainv0_mean : linops.LinearOperator
            Mean of the matrix-variate prior distribution on the inverse of the system matrix :math:`H = A^{-1}`.
        """
        # Check inner product between x0 and b; if negative or zero, choose better initialization
        bx0 = np.squeeze(b.T @ x0)
        bb = np.linalg.norm(b) ** 2
        if bx0 < 0:
            x0 = -x0
            bx0 = - bx0
            print("Better initialization found, setting x0 = - x0.")
        elif bx0 == 0:
            if np.all(b == np.zeros_like(b)):
                print("Right-hand-side is zero. Initializing with solution x0 = 0.")
                x0 = b
            else:
                print("Better initialization found, setting x0 = (b'b/b'Ab) * b.")
                bAb = np.squeeze(b.T @ (self.A @ b))
                x0 = bb / bAb * b
                bx0 = bb ** 2 / bAb

        # Construct prior mean of A and H
        alpha = 0.5 * bx0 / bb

        def _mv(v):
            return (x0 - alpha * b) * (x0 - alpha * b).T @ v

        def _mm(M):
            return (x0 - alpha * b) @ (x0 - alpha * b).T @ M

        Ainv0_mean = linops.ScalarMult(scalar=alpha, shape=(self.n, self.n)) + 2 / bx0 * linops.LinearOperator(
            matvec=_mv,
            matmat=_mm,
            shape=(self.n, self.n))
        A0_mean = linops.ScalarMult(scalar=1 / alpha, shape=(self.n, self.n)) - 1 / (
                alpha * np.squeeze((x0 - alpha * b).T @ x0)) * linops.LinearOperator(matvec=_mv,
                                                                                     matmat=_mm,
                                                                                     shape=(
                                                                                         self.n, self.n))
        return A0_mean, Ainv0_mean

    def has_converged(self, iter, maxiter, **kwargs):
        raise NotImplementedError

    def solve(self, callback=None, maxiter=None, atol=None):
        raise NotImplementedError


class AsymmetricMatrixBasedSolver(ProbabilisticLinearSolver):
    """
    Asymmetric matrix-based probabilistic linear solver.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        The square matrix or linear operator of the linear system.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    """

    def __init__(self, A, b, x0):
        self.x0 = x0
        super().__init__(A=A, b=b)

    def has_converged(self, iter, maxiter, **kwargs):
        raise NotImplementedError

    def solve(self, callback=None, maxiter=None, atol=None):
        raise NotImplementedError


class SymmetricMatrixBasedSolver(MatrixBasedSolver):
    """
    Symmetric matrix-based probabilistic linear solver.

    Implements the solve iteration of the symmetric matrix-based probabilistic linear solver described in [1]_ and [2]_.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        The square matrix or linear operator of the linear system.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    A0 : array-like or LinearOperator or RandomVariable, shape=(n, n), optional
        A square matrix, linear operator or random variable representing the prior belief over the linear operator
        :math:`A`. If an array or linear operator is given, a prior distribution is chosen automatically.
    Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
        A square matrix, linear operator or random variable representing the prior belief over the inverse
        :math:`H=A^{-1}`. This can be viewed as taking the form of a pre-conditioner. If an array or linear operator is
        given, a prior distribution is chosen automatically.
    x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
        Optional. Prior belief for the solution of the linear system. Will be ignored if ``Ainv0`` is given.

    Returns
    -------
    A : RandomVariable
        Posterior belief over the linear operator.
    Ainv : RandomVariable
        Posterior belief over the inverse linear operator.
    x : RandomVariable
        Posterior belief over the solution of the linear system.
    info : dict
        Information about convergence and the solution.

    References
    ----------
    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning, 2020
    .. [2] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260

    See Also
    --------
    NoisySymmetricMatrixBasedSolver : Class implementing the noisy symmetric probabilistic linear solver.
    """

    def __init__(self, A, b, A0=None, Ainv0=None, x0=None):

        # Assume constant right hand side
        if isinstance(b, prob.RandomVariable):
            _b = b.sample(size=1)
        else:
            _b = b

        super().__init__(A=A, b=_b, x0=x0)

        # Get or construct prior parameters
        A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor = self._get_prior_params(A0=A0, Ainv0=Ainv0, x0=self.x0,
                                                                                    b=self.b)

        # Initialize prior parameters
        self.A_mean = A0_mean
        self.A_covfactor = A0_covfactor
        self.Ainv_mean = Ainv0_mean
        self.Ainv_covfactor = Ainv0_covfactor
        if isinstance(x0, np.ndarray):
            self.x = x0
        elif x0 is None:
            self.x = Ainv0_mean @ self.b
        else:
            raise NotImplementedError

        # Computed search directions and observations
        self.search_dir_list = []
        self.obs_list = []
        self.sy = []

    def _get_prior_params(self, A0, Ainv0, x0, b):
        """
        Get the parameters of the matrix priors on A and H.

        Retrieves and / or initializes prior parameters of ``A0`` and ``Ainv0``.

        Parameters
        ----------
        A0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
            A square matrix, linear operator or random variable representing the prior belief over the linear operator
            :math:`A`. If an array or linear operator is given, a prior distribution is chosen automatically.
        Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
            A square matrix, linear operator or random variable representing the prior belief over the inverse
            :math:`H=A^{-1}`. This can be viewed as taking the form of a pre-conditioner. If an array or linear operator is
            given, a prior distribution is chosen automatically.
        x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
            Optional. Prior belief for the solution of the linear system. Will be ignored if ``A0`` or ``Ainv0`` is
            given.
        b : array_like, shape=(n,) or (n, nrhs)
            Right-hand side vector or matrix in :math:`A x = b`.

        Returns
        -------
        A0_mean : array-like or LinearOperator, shape=(n,n)
            Prior mean of the linear operator :math:`A`.
        A0_covfactor : array-like or LinearOperator, shape=(n,n)
            Factor :math:`W^A` of the symmetric Kronecker product prior covariance :math:`W^A \\otimes_s W^A` of
            :math:`A`.
        Ainv0_mean : array-like or LinearOperator, shape=(n,n)
            Prior mean of the linear operator :math:`H`.
        Ainv0_covfactor : array-like or LinearOperator, shape=(n,n)
            Factor :math:`W^H` of the symmetric Kronecker product prior covariance :math:`W^H \\otimes_s W^H` of
            :math:`H`.
        """
        # No matrix priors specified
        if A0 is None and Ainv0 is None:
            # No prior information given
            if x0 is None:
                Ainv0_mean = linops.Identity(shape=self.n)
                Ainv0_covfactor = linops.Identity(shape=self.n)
                # Symmetric posterior correspondence
                A0_mean = linops.Identity(shape=self.n)
                A0_covfactor = self.A
                return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor
            # Construct matrix priors from initial guess x0
            elif isinstance(x0, np.ndarray):
                A0_mean, Ainv0_mean = self._construct_symmetric_matrix_prior_means(x0=x0, b=b)
                Ainv0_covfactor = Ainv0_mean
                # Symmetric posterior correspondence
                A0_covfactor = self.A
                return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor
            elif isinstance(x0, prob.RandomVariable):
                raise NotImplementedError

        # Only prior on Ainv specified
        if not isinstance(A0, prob.RandomVariable) and Ainv0 is not None:
            if isinstance(Ainv0, prob.RandomVariable):
                Ainv0_mean = Ainv0.mean()
                Ainv0_covfactor = Ainv0.cov().A
            else:
                Ainv0_mean = Ainv0
                Ainv0_covfactor = Ainv0  # Symmetric posterior correspondence
            try:
                if A0 is not None:
                    A0_mean = A0
                elif isinstance(Ainv0, prob.RandomVariable):
                    A0_mean = Ainv0.mean().inv()
                else:
                    A0_mean = Ainv0.inv()
            except AttributeError:
                warnings.warn(message="Prior specified only for Ainv. Inverting prior mean naively. " +
                                      "This operation is computationally costly! Specify an inverse prior (mean) instead.")
                A0_mean = np.linalg.inv(Ainv0.mean())
            except NotImplementedError:
                A0_mean = linops.Identity(self.n)
                warnings.warn(
                    message="Prior specified only for Ainv. Automatic prior mean inversion not implemented, "
                            + "falling back to standard normal prior.")
            # Symmetric posterior correspondence
            A0_covfactor = self.A
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor

        # Only prior on A specified
        elif A0 is not None and not isinstance(Ainv0, prob.RandomVariable):
            if isinstance(A0, prob.RandomVariable):
                A0_mean = A0.mean()
                A0_covfactor = A0.cov().A
            else:
                A0_mean = A0
                A0_covfactor = A0  # Symmetric posterior correspondence
            try:
                if Ainv0 is not None:
                    Ainv0_mean = Ainv0
                elif isinstance(A0, prob.RandomVariable):
                    Ainv0_mean = A0.mean().inv()
                else:
                    Ainv0_mean = A0.inv()
            except AttributeError:
                warnings.warn(message="Prior specified only for A. Inverting prior mean naively. " +
                                      "This operation is computationally costly! Specify an inverse prior (mean) instead.")
                Ainv0_mean = np.linalg.inv(A0.mean())
            except NotImplementedError:
                Ainv0_mean = linops.Identity(self.n)
                warnings.warn(message="Prior specified only for A. " +
                                      "Automatic prior mean inversion failed, falling back to standard normal prior.")
            # Symmetric posterior correspondence
            Ainv0_covfactor = Ainv0_mean
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor
        # Both matrix priors on A and H specified
        elif isinstance(A0, prob.RandomVariable) and isinstance(Ainv0, prob.RandomVariable):
            A0_mean = A0.mean()
            A0_covfactor = A0.cov().A
            Ainv0_mean = Ainv0.mean()
            Ainv0_covfactor = Ainv0.cov().A
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor
        else:
            raise NotImplementedError

    def has_converged(self, iter, maxiter, resid=None, atol=None, rtol=None):
        """
        Check convergence of a linear solver.

        Evaluates a set of convergence criteria based on its input arguments to decide whether the iteration has converged.

        Parameters
        ----------
        iter : int
            Current iteration of solver.
        maxiter : int
            Maximum number of iterations
        resid : array-like
            Residual vector :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert` of the current iteration.
        atol : float
            Absolute residual tolerance. Stops if :math:`\\lVert r_i \\rVert \\leq \\text{atol}`.
        rtol : float
            Relative residual tolerance. Stops if :math:`\\lVert r_i \\rVert \\leq \\text{rtol} \\lVert b \\rVert`.

        Returns
        -------
        has_converged : bool
            True if the method has converged.
        convergence_criterion : str
            Convergence criterion which caused termination.
        """
        # maximum iterations
        if iter >= maxiter:
            warnings.warn(message="Iteration terminated. Solver reached the maximum number of iterations.")
            return True, "maxiter"
        # residual below error tolerance
        elif np.linalg.norm(resid) <= atol:
            return True, "resid_atol"
        elif np.linalg.norm(resid) <= rtol * np.linalg.norm(self.b):
            return True, "resid_rtol"
        # uncertainty-based
        # todo: based on posterior contraction
        else:
            return False, ""

    def _calibrate_uncertainty(self):
        """
        Calibrate uncertainty based on the Rayleigh coefficients

        A regression model for the log-Rayleigh coefficient is built based on the collected observations. The degrees of
        freedom in the covariance of A and H are set according to the predicted log-Rayleigh coefficient for the
        remaining unexplored dimensions.
        """
        # Transform to arrays
        _sy = np.squeeze(np.array(self.sy))
        _S = np.squeeze(np.array(self.search_dir_list)).T
        _Y = np.squeeze(np.array(self.obs_list)).T

        if self.iter_ > 5:  # only calibrate if enough iterations for a regression model have been performed
            # Rayleigh quotient
            iters = np.arange(self.iter_)
            logR = np.log(_sy) - np.log(np.einsum('ij,ij->j', _S, _S))

            # Least-squares fit for y intercept
            x_mean = np.mean(iters)
            y_mean = np.mean(logR)
            beta1 = np.sum((iters - x_mean) * (logR - y_mean)) / np.sum((iters - x_mean) ** 2)
            beta0 = y_mean - beta1 * x_mean

            # Log-Rayleigh quotient regression
            mf = GPy.mappings.linear.Linear(1, 1)
            k = GPy.kern.RBF(input_dim=1, lengthscale=1, variance=1)
            m = GPy.models.GPRegression(iters[:, None], (logR - beta0)[:, None], kernel=k, mean_function=mf)
            m.optimize(messages=False)

            # Predict Rayleigh quotient
            remaining_dims = np.arange(self.iter_, self.A.shape[0])[:, None]
            GP_pred = m.predict(remaining_dims)
            R_pred = np.exp(GP_pred[0].ravel() + beta0)

            # Set scale
            Phi = linops.ScalarMult(shape=self.A.shape, scalar=np.asscalar(np.mean(R_pred)))
            Psi = linops.ScalarMult(shape=self.A.shape, scalar=np.asscalar(np.mean(1 / R_pred)))

        else:
            Phi = None
            Psi = None

        return Phi, Psi

    def _get_output_randvars(self, S=None, Y=None, Phi=None, Psi=None):
        """Return output random variables x, A, Ainv from their means and covariances."""

        _A_covfactor = self.A_covfactor
        _Ainv_covfactor = self.Ainv_covfactor

        # Set degrees of freedom based on uncertainty calibration in unexplored space
        if Phi is not None:
            def _mv(x):
                def _I_S_fun(x):
                    return x - S @ np.linalg.solve(S.T @ S, S.T @ x)

                return _I_S_fun(Phi @ _I_S_fun(x))

            I_S_Phi_I_S_op = linops.LinearOperator(shape=self.A.shape, matvec=_mv)
            _A_covfactor = self.A_covfactor + I_S_Phi_I_S_op

        if Psi is not None:
            def _mv(x):
                def _I_Y_fun(x):
                    return x - Y @ np.linalg.solve(Y.T @ Y, Y.T @ x)

                return _I_Y_fun(Psi @ _I_Y_fun(x))

            I_Y_Psi_I_Y_op = linops.LinearOperator(shape=self.A.shape, matvec=_mv)
            _Ainv_covfactor = self.Ainv_covfactor + I_Y_Psi_I_Y_op

        # Create output random variables
        A = prob.RandomVariable(shape=self.A_mean.shape,
                                dtype=float,
                                distribution=prob.Normal(mean=self.A_mean,
                                                         cov=linops.SymmetricKronecker(
                                                             A=_A_covfactor)))
        cov_Ainv = linops.SymmetricKronecker(A=_Ainv_covfactor)
        Ainv = prob.RandomVariable(shape=self.Ainv_mean.shape,
                                   dtype=float,
                                   distribution=prob.Normal(mean=self.Ainv_mean, cov=cov_Ainv))
        # Induced distribution on x via Ainv
        # Exp(x) = A^-1 b, Cov(x) = 1/2 (W b'Wb + Wbb'W)
        Wb = _Ainv_covfactor @ self.b
        bWb = np.squeeze(Wb.T @ self.b)

        def _mv(x):
            return 0.5 * (bWb * _Ainv_covfactor @ x + Wb @ (Wb.T @ x))

        cov_op = linops.LinearOperator(shape=np.shape(_Ainv_covfactor), dtype=float,
                                       matvec=_mv, matmat=_mv)

        x = prob.RandomVariable(shape=(self.A_mean.shape[0],),
                                dtype=float,
                                distribution=prob.Normal(mean=self.x.ravel(), cov=cov_op))
        return x, A, Ainv

    def _mean_update(self, u, v):
        """Linear operator implementing the symmetric rank 2 mean update (+= uv' + vu')."""

        def mv(x):
            return u @ (v.T @ x) + v @ (u.T @ x)

        return linops.LinearOperator(shape=self.A_mean.shape, matvec=mv, matmat=mv)

    def _covariance_update(self, u, Ws):
        """Linear operator implementing the symmetric rank 2 covariance update (-= Ws u^T)."""

        def mv(x):
            return u @ (Ws.T @ x)

        return linops.LinearOperator(shape=self.A_mean.shape, matvec=mv, matmat=mv)

    def solve(self, callback=None, maxiter=None, atol=None, rtol=None, calibrate=True):
        """
        Solve the linear system :math:`Ax=b`.

        Parameters
        ----------
        callback : function, optional
            User-supplied function called after each iteration of the linear solver. It is called as
            ``callback(xk, Ak, Ainvk, sk, yk, alphak, resid)`` and can be used to return quantities from the
            iteration. Note that depending on the function supplied, this can slow down the solver.
        maxiter : int
            Maximum number of iterations
        atol : float
            Absolute residual tolerance. Stops if :math:`\\lVert r_i \\rVert \\leq \\text{atol}`.
        rtol : float
            Relative residual tolerance. Stops if :math:`\\lVert r_i \\rVert \\leq \\text{rtol} \\lVert b \\rVert`.
        calibrate : bool, default=True
            Should the posterior covariances be calibrated based on a Rayleigh quotient regression?

        Returns
        -------
        x : RandomVariable, shape=(n,) or (n, nrhs)
            Approximate solution :math:`x` to the linear system. Shape of the return matches the shape of ``b``.
        A : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator.
        Ainv : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
        info : dict
            Information on convergence of the solver.
        """
        # Initialization
        self.iter_ = 0
        resid = self.A @ self.x - self.b

        # Iteration with stopping criteria
        while True:
            # Check convergence
            _has_converged, _conv_crit = self.has_converged(iter=self.iter_, maxiter=maxiter,
                                                            resid=resid, atol=atol, rtol=rtol)
            if _has_converged:
                break

            # Compute search direction (with implicit reorthogonalization) via policy
            search_dir = - self.Ainv_mean @ resid
            self.search_dir_list.append(search_dir)

            # Perform action and observe
            obs = self.A @ search_dir
            self.obs_list.append(obs)

            # Compute step size
            sy = search_dir.T @ obs
            step_size = - (search_dir.T @ resid) / sy
            self.sy.append(sy)

            # Step and residual update
            self.x = self.x + step_size * search_dir
            resid = resid + step_size * obs

            # (Symmetric) mean and covariance updates
            Vs = self.A_covfactor @ search_dir
            delta_A = obs - self.A_mean @ search_dir
            u_A = Vs / (search_dir.T @ Vs)
            v_A = delta_A - 0.5 * (search_dir.T @ delta_A) * u_A

            Wy = self.Ainv_covfactor @ obs
            delta_Ainv = search_dir - self.Ainv_mean @ obs
            u_Ainv = Wy / (obs.T @ Wy)
            v_Ainv = delta_Ainv - 0.5 * (obs.T @ delta_Ainv) * u_Ainv

            # Rank 2 mean updates (+= uv' + vu')
            # TODO: Operator form may cause stack size issues for too many iterations
            self.A_mean = linops.aslinop(self.A_mean) + self._mean_update(u=u_A, v=v_A)
            self.Ainv_mean = linops.aslinop(self.Ainv_mean) + self._mean_update(u=u_Ainv, v=v_Ainv)

            # Rank 1 covariance kronecker factor update (-= u_A(Vs)' and -= u_Ainv(Wy)')
            self.A_covfactor = linops.aslinop(self.A_covfactor) - self._covariance_update(u=u_A, Ws=Vs)
            self.Ainv_covfactor = linops.aslinop(self.Ainv_covfactor) - self._covariance_update(u=u_Ainv, Ws=Wy)

            # Iteration increment
            self.iter_ += 1

            # Callback function used to extract quantities from iteration
            if callback is not None:
                # Phi, Psi = self._calibrate_uncertainty()
                xk, Ak, Ainvk = self._get_output_randvars(S=np.squeeze(np.array(self.search_dir_list)).T,
                                                          Y=np.squeeze(np.array(self.obs_list)).T,
                                                          Phi=None,  # Phi,
                                                          Psi=None)  # Psi)
                callback(xk=xk, Ak=Ak, Ainvk=Ainvk, sk=search_dir, yk=obs, alphak=step_size, resid=resid)

        # Calibrate uncertainty
        if calibrate:
            Phi, Psi = self._calibrate_uncertainty()
        else:
            Phi = None
            Psi = None

        # Create output random variables
        x, A, Ainv = self._get_output_randvars(S=np.squeeze(np.array(self.search_dir_list)).T,
                                               Y=np.squeeze(np.array(self.obs_list)).T,
                                               Phi=Phi,
                                               Psi=Psi)

        # Log information on solution
        info = {
            "iter": self.iter_,
            "maxiter": maxiter,
            "resid_l2norm": np.linalg.norm(resid, ord=2),
            "conv_crit": _conv_crit,
            "rel_cond": None  # TODO: matrix condition from solver (see scipy solvers)
        }

        return x, A, Ainv, info


class NoisySymmetricMatrixBasedSolver(MatrixBasedSolver):
    """
    Solver iteration of the noisy symmetric probabilistic linear solver.

    Implements the solve iteration of the symmetric matrix-based probabilistic linear solver taking into account noisy
    matrix-vector products :math:`y_k = (A + E_k)s_k` as described in [1]_ and [2]_.

    Parameters
    ----------
    A : LinearOperator or RandomVariable, shape=(n,n)
        The square matrix or linear operator of the linear system.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    A0 : array-like or LinearOperator or RandomVariable, shape=(n, n), optional
        A square matrix, linear operator or random variable representing the prior belief over the linear operator
        :math:`A`. If an array or linear operator is given, a prior distribution is chosen automatically.
    Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
        A square matrix, linear operator or random variable representing the prior belief over the inverse
        :math:`H=A^{-1}`. This can be viewed as taking the form of a pre-conditioner. If an array or linear operator is
        given, a prior distribution is chosen automatically.
    x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
        Optional. Prior belief for the solution of the linear system. Will be ignored if ``Ainv0`` is given.

    Returns
    -------
    A : RandomVariable
        Posterior belief over the linear operator.
    Ainv : RandomVariable
        Posterior belief over the inverse linear operator.
    x : RandomVariable
        Posterior belief over the solution of the linear system.
    info : dict
        Information about convergence and the solution.

    References
    ----------
    .. [1] Wenger, J., de Roos, F. and Hennig, P., Probabilistic Solution of Noisy Linear Systems, 2020
    .. [2] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260

    See Also
    --------
    SymmetricMatrixBasedSolver : Class implementing the symmetric probabilistic linear solver.
    """

    def __init__(self, A, b, A0=None, Ainv0=None, x0=None):

        # Transform system to linear operator
        A_preproc = A
        if isinstance(A, prob.RandomVariable):
            def mv(x):
                return (A @ x).sample(size=1)

            A_preproc = linops.LinearOperator(matvec=mv, matmat=mv, shape=A.shape, dtype=A.dtype)

        # Transform right hand side to random variable
        if not isinstance(b, prob.RandomVariable):
            _b = prob.asrandvar(b)
        else:
            _b = b

        super().__init__(A=A_preproc, b=_b, x0=x0)

        # Get or initialize prior parameters
        A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor, b_mean = self._get_prior_params(A0=A0, Ainv0=Ainv0, x0=x0,
                                                                                            b=_b)

        # Matrix prior parameters
        self.A0_mean = A0_mean
        self.A_mean = A0_mean
        self.A0_covfactor = A0_covfactor
        self.Ainv0_mean = Ainv0_mean
        self.Ainv_mean = Ainv0_mean
        self.Ainv0_covfactor = Ainv0_covfactor
        self.b_mean = b_mean

        # Induced distribution on x via Ainv
        # Exp = x = A^-1 b, Cov = 1/2 (W b'Wb + Wbb'W)
        Wb = Ainv0_covfactor @ self.b_mean
        bWb = np.squeeze(Wb.T @ self.b_mean)

        def _mv(x):
            return 0.5 * (bWb * Ainv0_covfactor @ x + Wb @ (Wb.T @ x))

        self.x_cov = linops.LinearOperator(shape=np.shape(Ainv0_covfactor), dtype=float, matvec=_mv, matmat=_mv)
        if isinstance(x0, np.ndarray):
            self.x_mean = x0
        elif x0 is None:
            self.x_mean = Ainv0_mean @ self.b_mean
        else:
            raise NotImplementedError
        self.x0 = self.x_mean

    def _get_prior_params(self, A0, Ainv0, x0, b):
        """
        Get the parameters of the matrix priors on A and H.

        Retrieves and / or initializes prior parameters of ``A0`` and ``Ainv0``.

        Parameters
        ----------
        A0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
            A square matrix, linear operator or random variable representing the prior belief over the linear operator
            :math:`A`. If an array or linear operator is given, a prior distribution is chosen automatically.
        Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
            A square matrix, linear operator or random variable representing the prior belief over the inverse
            :math:`H=A^{-1}`. This can be viewed as taking the form of a pre-conditioner. If an array or linear operator is
            given, a prior distribution is chosen automatically.
        x0 : array-like, or RandomVariable, shape=(n,)
            Optional. Prior belief for the solution of the linear system. Will be ignored if ``A0`` or ``Ainv0`` is
            given.
        b : RandomVariable, shape=(n,) or (n, nrhs)
            Right-hand side random variable `b` in :math:`A x = b`.

        Returns
        -------
        A0_mean : array-like or LinearOperator, shape=(n,n)
            Prior mean of the linear operator :math:`A`.
        A0_covfactor : array-like or LinearOperator, shape=(n,n)
            Factor :math:`W^A` of the symmetric Kronecker product prior covariance :math:`W^A \\otimes_s W^A` of
            :math:`A`.
        Ainv0_mean : array-like or LinearOperator, shape=(n,n)
            Prior mean of the linear operator :math:`H`.
        Ainv0_covfactor : array-like or LinearOperator, shape=(n,n)
            Factor :math:`W^H` of the symmetric Kronecker product prior covariance :math:`W^H \\otimes_s W^H` of
            :math:`H`.
        b_mean : array-like, shape=(n,nrhs)
            Prior mean of the right hand side :math:`b`.
        """
        # Right hand side mean
        b_mean = b.sample(1)  # TODO: build prior model for rhs and change to b.mean()

        # No matrix priors specified
        if A0 is None and Ainv0 is None:
            # No prior information given
            if x0 is None:
                Ainv0_mean = linops.Identity(shape=self.n)
                Ainv0_covfactor = linops.Identity(shape=self.n)
                # Standard normal covariance
                A0_mean = linops.Identity(shape=self.n)
                A0_covfactor = linops.Identity(shape=self.n)
                return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor, b_mean
            # Construct matrix priors from initial guess x0
            elif isinstance(x0, np.ndarray):
                A0_mean, Ainv0_mean = self._construct_symmetric_matrix_prior_means(x0=x0, b=b_mean)
                Ainv0_covfactor = Ainv0_mean
                # Standard normal covariance
                A0_covfactor = linops.Identity(shape=self.n)
                return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor, b_mean
            elif isinstance(x0, prob.RandomVariable):
                raise NotImplementedError

        # Only prior on Ainv specified
        if not isinstance(A0, prob.RandomVariable) and Ainv0 is not None:
            if isinstance(Ainv0, prob.RandomVariable):
                Ainv0_mean = Ainv0.mean()
                Ainv0_covfactor = Ainv0.cov().A
            else:
                Ainv0_mean = Ainv0
                Ainv0_covfactor = linops.Identity(shape=self.n)  # Standard normal covariance
            try:
                if A0 is not None:
                    A0_mean = A0
                elif isinstance(Ainv0, prob.RandomVariable):
                    A0_mean = Ainv0.mean().inv()
                else:
                    A0_mean = Ainv0.inv()
            except AttributeError:
                warnings.warn(message="Prior specified only for Ainv. Inverting prior mean naively. " +
                                      "This operation is computationally costly! Specify an inverse prior (mean) instead.")
                A0_mean = np.linalg.inv(Ainv0.mean())
            except NotImplementedError:
                A0_mean = linops.Identity(self.n)
                warnings.warn(
                    message="Prior specified only for Ainv. Automatic prior mean inversion not implemented, "
                            + "falling back to standard normal prior.")
            # Standard normal covariance
            A0_covfactor = linops.Identity(shape=self.n)
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor, b_mean

        # Only prior on A specified
        elif A0 is not None and not isinstance(Ainv0, prob.RandomVariable):
            if isinstance(A0, prob.RandomVariable):
                A0_mean = A0.mean()
                A0_covfactor = A0.cov().A
            else:
                A0_mean = A0
                A0_covfactor = linops.Identity(shape=self.n)  # Standard normal covariance
            try:
                if Ainv0 is not None:
                    Ainv0_mean = Ainv0
                elif isinstance(A0, prob.RandomVariable):
                    Ainv0_mean = A0.mean().inv()
                else:
                    Ainv0_mean = A0.inv()
            except AttributeError:
                warnings.warn(message="Prior specified only for A. Inverting prior mean naively. " +
                                      "This operation is computationally costly! Specify an inverse prior (mean) instead.")
                Ainv0_mean = np.linalg.inv(A0.mean())
            except NotImplementedError:
                Ainv0_mean = linops.Identity(self.n)
                warnings.warn(message="Prior specified only for A. " +
                                      "Automatic prior mean inversion failed, falling back to standard normal prior.")
            # Standard normal covariance
            Ainv0_covfactor = linops.Identity(shape=self.n)
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor, b_mean
        # Both matrix priors on A and H specified
        elif isinstance(A0, prob.RandomVariable) and isinstance(Ainv0, prob.RandomVariable):
            A0_mean = A0.mean()
            A0_covfactor = A0.cov().A
            Ainv0_mean = Ainv0.mean()
            Ainv0_covfactor = Ainv0.cov().A
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor, b_mean
        else:
            raise NotImplementedError

    def has_converged(self, iter, maxiter, ctol=None):
        """
        Check convergence of a linear solver.

        Evaluates a set of convergence criteria based on its input arguments to decide whether the iteration has converged.

        Parameters
        ----------
        iter : int
            Current iteration of solver.
        maxiter : int
            Maximum number of iterations
        ctol : float
            Tolerance for the uncertainty about the solution estimate. Stops if
            :math:`\\text{tr}(\\Sigma) \\leq \\text{ctol}`, where :math:`\\Sigma` is the covariance of the solution
            ``x``.

        Returns
        -------
        has_converged : bool
            True if the method has converged.
        convergence_criterion : str
            Convergence criterion which caused termination.
        """
        # maximum iterations
        if iter >= maxiter:
            warnings.warn(message="Iteration terminated. Solver reached the maximum number of iterations.")
            return True, "maxiter"
        # uncertainty-based
        if isinstance(self.x_cov, linops.LinearOperator):
            tracecov = self.x_cov.trace()
        else:
            tracecov = np.trace(self.x_cov)
        if tracecov <= ctol:
            return True, "covariance"
        else:
            return False, ""

    def _set_optimal_noise_scale(self, searchdirs, observations):
        """Computes the optimal noise scale maximizing the log-marginal likelihood (evidence)."""
        # Construct matrices of search directions and observations
        S = np.array(searchdirs).reshape(self.n, -1)
        Y = np.array(observations).reshape(self.n, -1)

        # Compute intermediate quantities
        Delta0 = Y - self.A0_mean @ S
        SW0S = S.T @ (self.A0_covfactor @ S)
        try:
            SW0SinvSDelta0 = scipy.linalg.solve(SW0S, S.T @ Delta0, assume_a="pos")  # solves k x k system k times: O(k^3)
            linop_rhs = Delta0.T @ (2 * self.A0_covfactor.inv() @ Delta0 - S @ SW0SinvSDelta0)
            linop_tracearg = scipy.linalg.solve(SW0S, linop_rhs, assume_a="pos")  # solves k x k system k times: O(k^3)

            # Optimal noise scale with respect to the evidence
            self.noise_scale = 1 / (self.n * (self.iter_ + 1)) * linop_tracearg.trace() - 1
            assert self.noise_scale > 0, "Noise scale must be positive."
        except scipy.linalg.LinAlgError:
            warnings.warn("Matrix S'W_0S not invertible. Noise scale estimate may be inaccurate.")

    def _mean_update(self, u, v):
        """Linear operator implementing the symmetric rank 2 mean update (+= uv' + vu')."""

        def matvec(x):
            return u @ (v.T @ x) + v @ (u.T @ x)

        return linops.LinearOperator(shape=self.A_mean.shape, matvec=matvec, matmat=matvec)

    def _covariance_update(self, u, Ws):
        """Linear operator implementing the symmetric rank 2 covariance update (-= Ws u^T)."""

        def matvec(x):
            return Ws * np.squeeze(u.T @ x)

        return linops.LinearOperator(shape=self.A_mean.shape, matvec=matvec, matmat=matvec)

    def _solution_covariance(self):
        """Covariance of the solution estimate induced by x = Hb."""
        W1b = self.Ainv_covfactor1 @ self.b_mean
        bW1b = np.squeeze(W1b.T @ self.b_mean)
        W2b = self.Ainv_covfactor2 @ self.b_mean
        bW2b = np.squeeze(W2b.T @ self.b_mean)

        def matvec(x):
            return 0.5 * (bW1b * self.Ainv_covfactor1 @ x + W1b @ (
                    W1b.T @ x) + bW2b * self.Ainv_covfactor2 @ x + W2b @ (W2b.T @ x))

        return linops.LinearOperator(shape=(self.n, self.n), dtype=float, matvec=matvec, matmat=matvec)

    def _get_output_randvars(self):
        """Return output random variables x, A, Ainv from their means and covariances."""

        # Estimate of matrix A
        cov_A = linops.SymmetricKronecker(A=self.A_covfactor1) + linops.SymmetricKronecker(A=self.A_covfactor2)
        A = prob.RandomVariable(shape=(self.n, self.n),
                                dtype=self.b_mean.dtype,
                                distribution=prob.Normal(mean=self.A_mean, cov=cov_A))

        # Estimate of inverse Ainv
        cov_Ainv = linops.SymmetricKronecker(A=self.Ainv_covfactor1) + linops.SymmetricKronecker(A=self.Ainv_covfactor2)
        Ainv = prob.RandomVariable(shape=(self.n, self.n),
                                   dtype=self.b_mean.dtype,
                                   distribution=prob.Normal(mean=self.Ainv_mean, cov=cov_Ainv))

        # Estimate of solution x
        x = prob.RandomVariable(shape=(self.n,),
                                dtype=self.b_mean.dtype,
                                distribution=prob.Normal(mean=self.x_mean.ravel(), cov=self.x_cov))
        return x, A, Ainv

    def solve(self, callback=None, maxiter=None, ctol=10 ** -6, noise_scale=None, **kwargs):
        """
        Solve the linear system :math:`Ax=b`.

        Parameters
        ----------
        callback : function, optional
            User-supplied function called after each iteration of the linear solver. It is called as
            ``callback(xk, Ak, Ainvk, sk, yk, alphak, resid, noise_scale)`` and can be used to return quantities from the
            iteration. Note that depending on the function supplied, this can slow down the solver.
        maxiter : int
            Maximum number of iterations
        ctol : float
            Tolerance for the uncertainty about the solution estimate. Stops if
            :math:`\\text{tr}(\\Sigma) \\leq \\text{ctol}`, where :math:`\\Sigma` is the covariance of the estimated
            solution ``x``.
        noise_scale : float
            Assumed (initial) noise scale :math:`\\varepsilon^2`.

        Returns
        -------
        x : RandomVariable, shape=(n,) or (n, nrhs)
            Approximate solution :math:`x` to the linear system. Shape of the return matches the shape of ``b``.
        A : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator.
        Ainv : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
        info : dict
            Information on convergence of the solver.
        """
        # Initialization
        self.iter_ = 0
        has_custom_noise_scale = False
        if noise_scale is not None:
            has_custom_noise_scale = True
            self.noise_scale = noise_scale
            if noise_scale < 0:
                raise ValueError("Noise scale must be non-negative.")

        A_mean_update = linops.ScalarMult(shape=(self.n, self.n), scalar=0)
        Ainv_mean_update = linops.ScalarMult(shape=(self.n, self.n), scalar=0)
        A_covfactor_update = linops.ScalarMult(shape=(self.n, self.n), scalar=0)
        Ainv_covfactor_update = linops.ScalarMult(shape=(self.n, self.n), scalar=0)
        search_dir_list = []
        obs_list = []

        # Iteration with stopping criteria
        while True:
            # Check convergence
            _has_converged, _conv_crit = self.has_converged(iter=self.iter_, maxiter=maxiter, ctol=ctol)
            if _has_converged:
                break

            # Noisy matrix-vector product for current iteration
            if isinstance(self.A, prob.RandomVariable):
                A_iter = self.A.sample()  # Sample new system matrix for this iteration
            else:
                A_iter = self.A

            # Right hand side estimate update
            self.b_mean = ((self.iter_ + 1) * self.b_mean + self.b.sample(size=1)) / (self.iter_ + 2)

            # Compute search direction via policy
            resid = A_iter @ self.x_mean - self.b_mean
            search_dir = - self.Ainv_mean @ resid
            search_dir_list.append(search_dir)

            # Perform action and observe
            obs = A_iter @ search_dir
            obs_list.append(obs)

            # Optimal noise scale with respect to log-marginal likelihood
            if not has_custom_noise_scale:
                self._set_optimal_noise_scale(searchdirs=search_dir_list, observations=obs_list)

            # Inference updates of system matrix, inverse and solution
            Vs = (self.A0_covfactor - A_covfactor_update) @ search_dir
            delta_A = obs - self.A_mean @ search_dir
            u_A = Vs / (search_dir.T @ Vs)
            v_A = delta_A - 0.5 * (search_dir.T @ delta_A) * u_A

            Wy = (self.Ainv0_covfactor - Ainv_covfactor_update) @ obs
            delta_Ainv = search_dir - self.Ainv_mean @ obs
            u_Ainv = Wy / (obs.T @ Wy)
            v_Ainv = delta_Ainv - 0.5 * (obs.T @ delta_Ainv) * u_Ainv

            # Mean updates
            A_mean_update += self._mean_update(u=u_A, v=v_A)
            self.A_mean = self.A0_mean + 1 / (1 + self.noise_scale) * A_mean_update
            Ainv_mean_update += self._mean_update(u=u_Ainv, v=v_Ainv)
            self.Ainv_mean = self.Ainv0_mean + 1 / (1 + self.noise_scale) * Ainv_mean_update

            # Covariance update(s)
            A_covfactor_update = self._covariance_update(u=u_A, Ws=Vs)
            self.A_covfactor1 = self.A0_covfactor - 1 / (1 + self.noise_scale) * A_covfactor_update
            self.A_covfactor2 = np.sqrt(self.noise_scale) / (1 + self.noise_scale) * A_covfactor_update

            Ainv_covfactor_update = self._covariance_update(u=u_Ainv, Ws=Wy)
            self.Ainv_covfactor1 = self.Ainv0_covfactor - 1 / (1 + self.noise_scale) * Ainv_covfactor_update
            self.Ainv_covfactor2 = np.sqrt(self.noise_scale) / (1 + self.noise_scale) * Ainv_covfactor_update

            # Solution update
            self.x_mean = self.Ainv_mean @ self.b_mean
            self.x_cov = self._solution_covariance()

            # Callback function used to extract quantities from iteration
            if callback is not None:
                xk, Ak, Ainvk = self._get_output_randvars()
                callback(xk=xk, Ak=Ak, Ainvk=Ainvk, sk=search_dir, yk=obs, alphak=None, resid=None,
                         noise_scale=noise_scale)

            # Iteration increment
            self.iter_ += 1

        # Create output random variables
        x, A, Ainv = self._get_output_randvars()

        # Log information on solution
        info = {
            "iter": self.iter_,
            "maxiter": maxiter,
            "trace_cov_sol": x.cov().trace(),
            "conv_crit": _conv_crit,
            "rel_cond": None,  # TODO: relative matrix condition from solver (see scipy solvers)
            "noise_scale": self.noise_scale
        }

        return x, A, Ainv, info
