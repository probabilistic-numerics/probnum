"""
Matrix-based probabilistic linear solvers.

Implementations of matrix-based linear solvers which perform inference on the matrix or
its inverse given linear observations.
"""
import warnings
import abc

import numpy as np
import GPy

import probnum
from probnum import random_variables as rvs
from probnum.linalg import linops


class ProbabilisticLinearSolver(abc.ABC):
    """
    An abstract base class for probabilistic linear solvers.

    This class is designed to be subclassed with new (probabilistic) linear solvers,
    which implement a ``.solve()`` method. Objects of this type are instantiated in
    wrapper functions such as :meth:``problinsolve``.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        A square matrix or linear operator. A prior distribution can be provided as a
        :class:`~probnum.RandomVariable`. If an array or linear operator is given,
        a prior distribution is chosen automatically.
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

        Evaluates a set of convergence criteria based on its input arguments to decide
        whether the iteration has converged.

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
            warnings.warn(
                "Iteration terminated. Solver reached the maximum number of iterations."
            )
            return True, "maxiter"
        else:
            return False, ""

    def solve(self, callback=None, **kwargs):
        """
        Solve the linear system :math:`Ax=b`.

        Parameters
        ----------
        callback : function, optional
            User-supplied function called after each iteration of the linear solver. It
            is called as ``callback(xk, Ak, Ainvk, sk, yk, alphak, resid, **kwargs)``
            and can be used to return quantities from the iteration. Note that depending
            on the function supplied, this can slow down the solver.
        kwargs
            Key-word arguments adjusting the behaviour of the ``solve`` iteration. These
            are usually convergence criteria.

        Returns
        -------
        x : RandomVariable, shape=(n,) or (n, nrhs)
            Approximate solution :math:`x` to the linear system. Shape of the return
            matches the shape of ``b``.
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
        :class:`~probnum.RandomVariable`. If an array or linear operator is given,
        a prior distribution is
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
            A square matrix, linear operator or random variable representing the prior
            belief over the linear operator :math:`A`. If an array or linear operator is
            given, a prior distribution is chosen automatically.
        Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
            A square matrix, linear operator or random variable representing the prior
            belief over the inverse :math:`H=A^{-1}`. This can be viewed as taking the
            form of a pre-conditioner. If an array or linear operator is given, a prior
            distribution is chosen automatically.
        x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
            Optional. Prior belief for the solution of the linear system. Will be
            ignored if ``A0`` or ``Ainv0`` is given.
        b : array_like, shape=(n,) or (n, nrhs)
            Right-hand side vector or matrix in :math:`A x = b`.
        """
        raise NotImplementedError

    def _construct_symmetric_matrix_prior_means(self, A, x0, b):
        """
        Create matrix prior means from an initial guess for the solution of the linear
        system.

        Constructs a matrix-variate prior mean for H from ``x0`` and ``b`` such that
        :math:`H_0b = x_0`, :math:`H_0` symmetric positive definite and
        :math:`A_0 = H_0^{-1}`.

        Parameters
        ----------
        A : array-like or LinearOperator, shape=(n,n)
            System matrix assumed to be square.
        x0 : array-like, shape=(n,) or (n, nrhs)
            Optional. Guess for the solution of the linear system.
        b : array_like, shape=(n,) or (n, nrhs)
            Right-hand side vector or matrix in :math:`A x = b`.

        Returns
        -------
        A0_mean : linops.LinearOperator
            Mean of the matrix-variate prior distribution on the system matrix
            :math:`A`.
        Ainv0_mean : linops.LinearOperator
            Mean of the matrix-variate prior distribution on the inverse of the system
            matrix :math:`H = A^{-1}`.
        """
        # Check inner product between x0 and b; if negative or zero, choose better
        # initialization
        bx0 = np.squeeze(b.T @ x0)
        bb = np.linalg.norm(b) ** 2
        if bx0 < 0:
            x0 = -x0
            bx0 = -bx0
            print("Better initialization found, setting x0 = - x0.")
        elif bx0 == 0:
            if np.all(b == np.zeros_like(b)):
                print("Right-hand-side is zero. Initializing with solution x0 = 0.")
                x0 = b
            else:
                print("Better initialization found, setting x0 = (b'b/b'Ab) * b.")
                bAb = np.squeeze(b.T @ (A @ b))
                x0 = bb / bAb * b
                bx0 = bb ** 2 / bAb

        # Construct prior mean of A and H
        alpha = 0.5 * bx0 / bb

        def _mv(v):
            return (x0 - alpha * b) * (x0 - alpha * b).T @ v

        def _mm(M):
            return (x0 - alpha * b) @ (x0 - alpha * b).T @ M

        Ainv0_mean = linops.ScalarMult(
            scalar=alpha, shape=(self.n, self.n)
        ) + 2 / bx0 * linops.LinearOperator(
            matvec=_mv, matmat=_mm, shape=(self.n, self.n)
        )
        A0_mean = linops.ScalarMult(scalar=1 / alpha, shape=(self.n, self.n)) - 1 / (
            alpha * np.squeeze((x0 - alpha * b).T @ x0)
        ) * linops.LinearOperator(matvec=_mv, matmat=_mm, shape=(self.n, self.n))
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

    Implements the solve iteration of the symmetric matrix-based probabilistic linear
    solver described in [1]_ and [2]_.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        The square matrix or linear operator of the linear system.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    A0 : array-like or LinearOperator or RandomVariable, shape=(n, n), optional
        A square matrix, linear operator or random variable representing the prior
        belief over the linear operator :math:`A`. If an array or linear operator is
        given, a prior distribution is chosen automatically.
    Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
        A square matrix, linear operator or random variable representing the prior
        belief over the inverse :math:`H=A^{-1}`. This can be viewed as taking the form
        of a pre-conditioner. If an array or linear operator is given, a prior
        distribution is chosen automatically.
    x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
        Optional. Prior belief for the solution of the linear system. Will be ignored if
        ``Ainv0`` is given.

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
    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning,
       2020
    .. [2] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on
       Optimization*, 2015, 25, 234-260

    See Also
    --------
    NoisySymmetricMatrixBasedSolver :
        Class implementing the noisy symmetric probabilistic linear solver.
    """

    def __init__(self, A, b, A0=None, Ainv0=None, x0=None):

        # Assume constant right hand side
        if isinstance(b, probnum.RandomVariable):
            _b = b.sample(size=1)
        else:
            _b = b

        super().__init__(A=A, b=_b, x0=x0)

        # Get or construct prior parameters
        A_mean0, A_covfactor0, Ainv_mean0, Ainv_covfactor0 = self._get_prior_params(
            A0=A0, Ainv0=Ainv0, x0=self.x0, b=self.b
        )

        # Initialize prior parameters
        self.A_mean = A_mean0
        self.A_mean0 = A_mean0
        self.A_covfactor = A_covfactor0
        self.A_covfactor0 = A_covfactor0
        self.Ainv_mean = Ainv_mean0
        self.Ainv_mean0 = Ainv_mean0
        self.Ainv_covfactor = Ainv_covfactor0
        self.Ainv_covfactor0 = Ainv_covfactor0
        if isinstance(x0, np.ndarray):
            self.x_mean = x0
        elif x0 is None:
            self.x_mean = Ainv_mean0 @ self.b
        else:
            raise NotImplementedError
        self.x0 = self.x_mean

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
            A square matrix, linear operator or random variable representing the prior
            belief over the linear operator :math:`A`. If an array or linear operator is
            given, a prior distribution is chosen automatically. Ainv0 : array-like or
            LinearOperator or RandomVariable, shape=(n,n), optional A square matrix,
            linear operator or random variable representing the prior belief over the
            inverse :math:`H=A^{-1}`. This can be viewed as taking the form of a
            pre-conditioner. If an array or linear operator is given, a prior
            distribution is chosen automatically.
        x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
            Optional. Prior belief for the solution of the linear system. Will be
            ignored if ``A0`` or ``Ainv0`` is given.
        b : array_like, shape=(n,) or (n, nrhs)
            Right-hand side vector or matrix in :math:`A x = b`.

        Returns
        -------
        A0_mean : array-like or LinearOperator, shape=(n,n)
            Prior mean of the linear operator :math:`A`.
        A0_covfactor : array-like or LinearOperator, shape=(n,n)
            Factor :math:`W^A` of the symmetric Kronecker product prior covariance
            :math:`W^A \\otimes_s W^A` of :math:`A`.
        Ainv0_mean : array-like or LinearOperator, shape=(n,n)
            Prior mean of the linear operator :math:`H`.
        Ainv0_covfactor : array-like or LinearOperator, shape=(n,n)
            Factor :math:`W^H` of the symmetric Kronecker product prior covariance
            :math:`W^H \\otimes_s W^H` of :math:`H`.
        """
        self.is_calib_covclass = False
        # No matrix priors specified
        if A0 is None and Ainv0 is None:
            self.is_calib_covclass = True
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
                A0_mean, Ainv0_mean = self._construct_symmetric_matrix_prior_means(
                    A=self.A, x0=x0, b=b
                )
                Ainv0_covfactor = Ainv0_mean
                # Symmetric posterior correspondence
                A0_covfactor = self.A
                return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor
            elif isinstance(x0, probnum.RandomVariable):
                raise NotImplementedError

        # Prior on Ainv specified
        if not isinstance(A0, probnum.RandomVariable) and Ainv0 is not None:
            if isinstance(Ainv0, probnum.RandomVariable):
                Ainv0_mean = Ainv0.mean
                Ainv0_covfactor = Ainv0.cov.A
            else:
                self.is_calib_covclass = True
                Ainv0_mean = Ainv0
                Ainv0_covfactor = Ainv0  # Symmetric posterior correspondence
            try:
                if A0 is not None:
                    A0_mean = A0
                elif isinstance(Ainv0, probnum.RandomVariable):
                    A0_mean = Ainv0.mean.inv()
                else:
                    A0_mean = Ainv0.inv()
            except AttributeError:
                warnings.warn(
                    "Prior specified only for Ainv. Inverting prior mean naively. "
                    "This operation is computationally costly! Specify an inverse "
                    "prior (mean) instead."
                )
                A0_mean = np.linalg.inv(Ainv0.mean)
            except NotImplementedError:
                A0_mean = linops.Identity(self.n)
                warnings.warn(
                    "Prior specified only for Ainv. Automatic prior mean inversion "
                    "not implemented, falling back to standard normal prior."
                )
            # Symmetric posterior correspondence
            A0_covfactor = self.A
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor

        # Prior on A specified
        elif A0 is not None and not isinstance(Ainv0, probnum.RandomVariable):
            if isinstance(A0, probnum.RandomVariable):
                A0_mean = A0.mean
                A0_covfactor = A0.cov.A
            else:
                self.is_calib_covclass = True
                A0_mean = A0
                A0_covfactor = A0  # Symmetric posterior correspondence
            try:
                if Ainv0 is not None:
                    Ainv0_mean = Ainv0
                elif isinstance(A0, probnum.RandomVariable):
                    Ainv0_mean = A0.mean.inv()
                else:
                    Ainv0_mean = A0.inv()
            except AttributeError:
                warnings.warn(
                    "Prior specified only for A. Inverting prior mean naively. "
                    "This operation is computationally costly! "
                    "Specify an inverse prior (mean)."
                )
                Ainv0_mean = np.linalg.inv(A0.mean)
            except NotImplementedError:
                Ainv0_mean = linops.Identity(self.n)
                warnings.warn(
                    "Prior specified only for A. Automatic prior mean inversion "
                    "failed, falling back to standard normal prior."
                )
            # Symmetric posterior correspondence
            Ainv0_covfactor = Ainv0_mean
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor
        # Both matrix priors on A and H specified via random variables
        elif isinstance(A0, probnum.RandomVariable) and isinstance(
            Ainv0, probnum.RandomVariable
        ):
            A0_mean = A0.mean
            A0_covfactor = A0.cov.A
            Ainv0_mean = Ainv0.mean
            Ainv0_covfactor = Ainv0.cov.A
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor
        else:
            raise NotImplementedError

    def _compute_trace_Ainv_covfactor0(self, Y, unc_scale):
        """
        Computes the trace of the prior covariance factor for the inverse view.

        Parameters
        ----------
        Y : np.ndarray, shape=(n,k)
            Observations.
        unc_scale : float
            Uncertainty scale :math:`\\psi` of the inverse view.

        Returns
        -------
        trace_Ainv_covfactor0 : float
            Trace of prior covariance factor.
        """
        # Initialization
        if Y is not None:
            k = Y.shape[1]
        else:
            k = 0
        if unc_scale is None:
            unc_scale = 0

        if isinstance(self.Ainv_covfactor0, linops.ScalarMult):
            # Scalar prior mean
            if self.is_calib_covclass and k > 0 and unc_scale != 0:
                _trace = self.Ainv_covfactor0.scalar * k
            else:
                _trace = self.Ainv_covfactor0.trace()
        else:
            # General prior mean
            if self.is_calib_covclass and k > 0 and unc_scale != 0:
                # General prior mean with calibration covariance class
                _trace = np.trace(
                    np.linalg.solve(
                        Y.T @ self.Ainv_mean0 @ Y,
                        Y.T @ self.Ainv_mean0 @ self.Ainv_mean0 @ Y,
                    )
                )
            else:
                _trace = self.Ainv_covfactor0.trace()
        if self.is_calib_covclass:
            # Additive term from uncertainty calibration
            _trace += unc_scale * (self.n - k)
        return _trace

    def _compute_trace_solution_covariance(self, bWb, Wb):
        """
        Computes the trace of the solution covariance
        :math:`\\tr(\\operatorname{Cov}[x])`

        Parameters
        ----------
        bWb : float
            Inner product of right hand side and the inverse covariance factor.
        Wb : np.ndarray
            Matrix-vector product between the inverse covariance factor and the right
            hand side.

        Returns
        -------
        trace_x_cov : float
            Trace of solution covariance.
        """
        # Trace of inverse covariance factor after k iterations
        return 0.5 * (bWb * self.trace_Ainv_covfactor + np.linalg.norm(Wb, ord=2) ** 2)

    def has_converged(self, iter, maxiter, resid=None, atol=None, rtol=None):
        """
        Check convergence of a linear solver.

        Evaluates a set of convergence criteria based on its input arguments to decide
        whether the iteration has converged.

        Parameters
        ----------
        iter : int
            Current iteration of solver.
        maxiter : int
            Maximum number of iterations
        resid : array-like
            Residual vector :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert` of
            the current iteration.
        atol : float
            Absolute residual tolerance. Stops if
            :math:`\\min(\\lVert r_i \\rVert, \\sqrt{\\operatorname{tr}(\\operatorname{Cov}(x))}) \\leq \\text{atol}`.
        rtol : float
            Relative residual tolerance. Stops if
            :math:`\\min(\\lVert r_i \\rVert, \\sqrt{\\operatorname{tr}(\\operatorname{Cov}(x))}) \\leq \\text{rtol} \\lVert b \\rVert`.

        Returns
        -------
        has_converged : bool
            True if the method has converged.
        convergence_criterion : str
            Convergence criterion which caused termination.
        """  # pylint: disable=line-too-long
        # maximum iterations
        if iter >= maxiter:
            warnings.warn(
                "Iteration terminated. Solver reached the maximum number of iterations."
            )
            return True, "maxiter"
        # residual below error tolerance
        resid_norm = np.linalg.norm(resid)
        b_norm = np.linalg.norm(self.b)
        if resid_norm <= atol:
            return True, "resid_atol"
        elif resid_norm <= rtol * b_norm:
            return True, "resid_rtol"
        # uncertainty-based
        if np.sqrt(self.trace_sol_cov) <= atol:
            return True, "tracecov_atol"
        elif np.sqrt(self.trace_sol_cov) <= rtol * b_norm:
            return True, "tracecov_rtol"
        else:
            return False, ""

    def _calibrate_uncertainty(self, S, sy, method):
        """
        Calibrate uncertainty based on the Rayleigh coefficients

        A regression model for the log-Rayleigh coefficient is built based on the
        collected observations. The degrees of freedom in the kernels of A and H are set
        according to the predicted log-Rayleigh coefficient for the remaining unexplored
        dimensions.

        Parameters
        ----------
        S : np.ndarray, shape=(n, k)
            Array of search directions
        sy : np.ndarray
            Array of inner products ``s_i'As_i``
        method : str
            Type of calibration method to use based on the Rayleigh quotient. Available
            calibration procedures are
            ====================================  ==================
             Most recent Rayleigh quotient         ``adhoc``
             Running (weighted) mean               ``weightedmean``
             GP regression for kernel matrices     ``gpkern``
            ====================================  ==================

        Returns
        -------
        phi : float
            Uncertainty scale of the null space of span(S) for the A view
        psi : float
            Uncertainty scale of the null space of span(Y) for the Ainv view
        """

        # Rayleigh quotient
        iters = np.arange(self.iter_ + 1)
        logR = np.log(sy) - np.log(np.einsum("nk,nk->k", S, S))

        # only calibrate if enough iterations for a regression model have been performed
        if self.iter_ > 1:
            if method == "adhoc":
                logR_pred = logR[-1]
            elif method == "weightedmean":
                deprecation_rate = 0.9
                logR_pred = logR * np.repeat(
                    deprecation_rate, self.iter_ + 1
                ) ** np.arange(self.iter_ + 1)
            elif method == "gpkern":
                # GP mean function via Weyl's result on spectra of Gram matrices for
                # differentiable kernels
                # ln(sigma(n)) ~= theta_0 - theta_1 ln(n)
                lnmap = GPy.core.Mapping(1, 1)
                lnmap.f = lambda n: np.log(n + 10 ** -16)
                lnmap.update_gradients = lambda a, b: None
                mf = GPy.mappings.Additive(
                    GPy.mappings.Constant(1, 1, value=0),
                    GPy.mappings.Compound(lnmap, GPy.mappings.Linear(1, 1)),
                )
                k = GPy.kern.RBF(input_dim=1, lengthscale=1, variance=1)
                m = GPy.models.GPRegression(
                    iters[:, None] + 1, logR[:, None], kernel=k, mean_function=mf
                )
                m.optimize(messages=False)

                # Predict Rayleigh quotient
                remaining_dims = np.arange(self.iter_, self.A.shape[0])[:, None]
                logR_pred = m.predict(remaining_dims + 1)[0].ravel()
            else:
                raise ValueError("Calibration method not recognized.")

            # Set uncertainty scale (degrees of freedom in calibration covariance class)
            Phi = (np.exp(np.mean(logR_pred))).item()
            Psi = (np.exp(-np.mean(logR_pred))).item()
        else:
            # For too few iterations take the most recent Rayleigh quotient
            Phi = np.exp(logR[-1])
            Psi = 1 / Phi

        return Phi, Psi

    def _get_calibration_covariance_update_terms(self, phi=None, psi=None):
        """
        For the calibration covariance class set the calibration update terms of the
        covariance in the null spaces of span(S) and span(Y) based on the degrees of
        freedom.
        """
        # Search directions and observations as arrays
        S = np.hstack(self.search_dir_list)
        Y = np.hstack(self.obs_list)

        def get_null_space_map(V, unc_scale):
            """
            Returns a function mapping to the null space of span(V), scaling with a
            single degree of freedom and mapping back.
            """

            def null_space_proj(x):
                try:
                    VVinvVx = np.linalg.solve(V.T @ V, V.T @ x)
                    return x - V @ VVinvVx
                except np.linalg.LinAlgError:
                    return np.zeros_like(x)

            # For a scalar uncertainty scale projecting to the null space twice is
            # equivalent to projecting once
            return lambda y: unc_scale * null_space_proj(y)

        # Compute calibration term in the A view as a linear operator with scaling from
        # degrees of freedom
        calibration_term_A = linops.LinearOperator(
            shape=(self.n, self.n), matvec=get_null_space_map(V=S, unc_scale=phi)
        )

        # Compute calibration term in the Ainv view as a linear operator with scaling
        # from degrees of freedom
        calibration_term_Ainv = linops.LinearOperator(
            shape=(self.n, self.n), matvec=get_null_space_map(V=Y, unc_scale=psi)
        )

        return calibration_term_A, calibration_term_Ainv

    def _get_output_randvars(self, Y_list, sy_list, phi=None, psi=None):
        """
        Return output random variables x, A, Ainv from their means and covariances.
        """

        if self.iter_ > 0:
            # Observations and inner products in A-space between actions
            Y = np.hstack(Y_list)
            sy = np.vstack(sy_list).ravel()

            # Posterior covariance factors
            if self.is_calib_covclass and (not phi is None) and (not psi is None):
                # Ensure prior covariance class only acts in span(S) like A
                def _matvec(x):
                    # First term of calibration covariance class: AS(S'AS)^{-1}S'A
                    return (Y * sy ** -1) @ (Y.T @ x.ravel())

                _A_covfactor0 = linops.LinearOperator(
                    shape=(self.n, self.n), matvec=_matvec
                )

                def _matvec(x):
                    # Term in covariance class: A_0^{-1}Y(Y'A_0^{-1}Y)^{-1}Y'A_0^{-1}
                    # TODO: for efficiency ensure that we dont have to compute
                    # (Y.T Y)^{-1} two times! For a scalar mean this is the same as in
                    # the null space projection
                    YAinv0Y_inv_YAinv0x = np.linalg.solve(
                        Y.T @ (self.Ainv_mean0 @ Y), Y.T @ (self.Ainv_mean0 @ x)
                    )
                    return self.Ainv_mean0 @ (Y @ YAinv0Y_inv_YAinv0x)

                _Ainv_covfactor0 = linops.LinearOperator(
                    shape=(self.n, self.n), matvec=_matvec
                )

                # Set degrees of freedom based on uncertainty calibration in unexplored
                # space
                (
                    calibration_term_A,
                    calibration_term_Ainv,
                ) = self._get_calibration_covariance_update_terms(phi=phi, psi=psi)

                _A_covfactor = (
                    _A_covfactor0 - self._A_covfactor_update_term + calibration_term_A
                )
                _Ainv_covfactor = (
                    _Ainv_covfactor0
                    - self._Ainv_covfactor_update_term
                    + calibration_term_Ainv
                )
            else:
                # No calibration
                _A_covfactor = self.A_covfactor
                _Ainv_covfactor = self.Ainv_covfactor
        else:
            # Converged before making any observations
            _A_covfactor = self.A_covfactor0
            _Ainv_covfactor = self.Ainv_covfactor0

        # Create output random variables
        A = rvs.Normal(mean=self.A_mean, cov=linops.SymmetricKronecker(A=_A_covfactor))

        Ainv = rvs.Normal(
            mean=self.Ainv_mean, cov=linops.SymmetricKronecker(A=_Ainv_covfactor)
        )
        # Induced distribution on x via Ainv
        # Exp(x) = Ainv b, Cov(x) = 1/2 (W b'Wb + Wbb'W)
        Wb = _Ainv_covfactor @ self.b
        bWb = np.squeeze(Wb.T @ self.b)

        def _mv(x):
            return 0.5 * (bWb * _Ainv_covfactor @ x + Wb @ (Wb.T @ x))

        cov_op = linops.LinearOperator(
            shape=(self.n, self.n), dtype=float, matvec=_mv, matmat=_mv
        )

        x = rvs.Normal(mean=self.x_mean.ravel(), cov=cov_op)

        # Compute trace of solution covariance: tr(Cov(x))
        self.trace_sol_cov = np.real_if_close(
            self._compute_trace_solution_covariance(bWb=bWb, Wb=Wb)
        ).item()

        return x, A, Ainv

    def _mean_update(self, u, v):
        """
        Linear operator implementing the symmetric rank 2 mean update (+= uv' + vu').
        """

        def mv(x):
            return u @ (v.T @ x) + v @ (u.T @ x)

        return linops.LinearOperator(shape=(self.n, self.n), matvec=mv, matmat=mv)

    def _covariance_update(self, u, Ws):
        """
        Linear operator implementing the symmetric rank 2 kernels update (-= Ws u^T).
        """

        def mv(x):
            return Ws @ (u.T @ x)

        return linops.LinearOperator(shape=(self.n, self.n), matvec=mv, matmat=mv)

    def solve(
        self, callback=None, maxiter=None, atol=None, rtol=None, calibration=None
    ):
        """
        Solve the linear system :math:`Ax=b`.

        Parameters
        ----------
        callback : function, optional
            User-supplied function called after each iteration of the linear solver. It
            is called as ``callback(xk, Ak, Ainvk, sk, yk, alphak, resid)`` and can be
            used to return quantities from the iteration. Note that depending on the
            function supplied, this can slow down the solver.
        maxiter : int
            Maximum number of iterations
        atol : float
            Absolute residual tolerance. Stops if
            :math:`\\min(\\lVert r_i \\rVert, \\sqrt{\\operatorname{tr}(\\operatorname{Cov}(x))}) \\leq \\text{atol}`.
        rtol : float
            Relative residual tolerance. Stops if
            :math:`\\min(\\lVert r_i \\rVert, \\sqrt{\\operatorname{tr}(\\operatorname{Cov}(x))}) \\leq \\text{rtol} \\lVert b \\rVert`.
        calibration : str or float, default=False
            If supplied calibrates the output via the given procedure or uncertainty
            scale. Available calibration procedures / choices are

            ====================================  ================
             No calibration                       None
             Provided scale                       float
             Most recent Rayleigh quotient        ``adhoc``
             Running (weighted) mean              ``weightedmean``
             GP regression for kernel matrices    ``gpkern``
            ====================================  ================

        Returns
        -------
        x : RandomVariable, shape=(n,) or (n, nrhs)
            Approximate solution :math:`x` to the linear system. Shape of the return
            matches the shape of ``b``.
        A : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator.calibrate
        Ainv : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
        info : dict
            Information on convergence of the solver.
        """  # pylint: disable=line-too-long
        # Initialization
        self.iter_ = 0
        resid = self.A @ self.x_mean - self.b

        # Initialize uncertainty calibration
        phi = None
        psi = None
        if calibration is None:
            pass
        elif (
            calibration is not None
            or calibration is not False
            and not self.is_calib_covclass
        ):
            warnings.warn(
                message="Cannot use calibration without a compatible covariance class."
            )
        elif isinstance(calibration, str) and self.is_calib_covclass:
            pass
        elif self.is_calib_covclass:
            if calibration < 0:
                raise ValueError("Calibration scale must be non-negative.")
            elif calibration == 0.0:
                pass
            else:
                phi = calibration
                psi = 1 / calibration

        # Trace of solution covariance
        _trace_Ainv_covfactor_update = 0
        self.trace_Ainv_covfactor = self._compute_trace_Ainv_covfactor0(
            Y=None, unc_scale=psi
        )

        # Create output random variables
        x, A, Ainv = self._get_output_randvars(
            Y_list=self.obs_list, sy_list=self.sy, phi=phi, psi=psi
        )

        # Iteration with stopping criteria
        while True:
            # Check convergence
            _has_converged, _conv_crit = self.has_converged(
                iter=self.iter_, maxiter=maxiter, resid=resid, atol=atol, rtol=rtol
            )
            if _has_converged:
                break

            # Compute search direction (with implicit reorthogonalization) via policy
            search_dir = -self.Ainv_mean @ resid
            self.search_dir_list.append(search_dir)

            # Perform action and observe
            obs = self.A @ search_dir
            self.obs_list.append(obs)

            # Compute step size
            sy = search_dir.T @ obs
            step_size = -np.squeeze((search_dir.T @ resid) / sy)
            self.sy.append(sy)

            # Step and residual update
            self.x_mean = self.x_mean + step_size * search_dir
            resid = resid + step_size * obs

            # (Symmetric) mean and covariance updates
            Vs = self.A_covfactor @ search_dir
            delta_A = obs - self.A_mean @ search_dir
            u_A = Vs / (search_dir.T @ Vs)
            v_A = delta_A - 0.5 * (search_dir.T @ delta_A) * u_A

            Wy = self.Ainv_covfactor @ obs
            delta_Ainv = search_dir - self.Ainv_mean @ obs
            yWy = np.squeeze(obs.T @ Wy)
            u_Ainv = Wy / yWy
            v_Ainv = delta_Ainv - 0.5 * (obs.T @ delta_Ainv) * u_Ainv

            # Rank 2 mean updates (+= uv' + vu')
            self.A_mean = linops.aslinop(self.A_mean) + self._mean_update(u=u_A, v=v_A)
            self.Ainv_mean = linops.aslinop(self.Ainv_mean) + self._mean_update(
                u=u_Ainv, v=v_Ainv
            )

            # Rank 1 covariance Kronecker factor update (-= u_A(Vs)' and -= u_Ainv(Wy)')
            if self.iter_ == 0:
                self._A_covfactor_update_term = self._covariance_update(u=u_A, Ws=Vs)
                self._Ainv_covfactor_update_term = self._covariance_update(
                    u=u_Ainv, Ws=Wy
                )
            else:
                self._A_covfactor_update_term = (
                    self._A_covfactor_update_term
                    + self._covariance_update(u=u_A, Ws=Vs)
                )
                self._Ainv_covfactor_update_term = (
                    self._Ainv_covfactor_update_term
                    + self._covariance_update(u=u_Ainv, Ws=Wy)
                )
            self.A_covfactor = (
                linops.aslinop(self.A_covfactor0) - self._A_covfactor_update_term
            )
            self.Ainv_covfactor = (
                linops.aslinop(self.Ainv_covfactor0) - self._Ainv_covfactor_update_term
            )

            # Calibrate uncertainty based on Rayleigh quotient
            if isinstance(calibration, str) and self.is_calib_covclass:
                phi, psi = self._calibrate_uncertainty(
                    S=np.hstack(self.search_dir_list),
                    sy=np.vstack(self.sy).ravel(),
                    method=calibration,
                )

            # Update trace of solution covariance: tr(Cov(Hb))
            _trace_Ainv_covfactor_update += 1 / yWy * np.squeeze(Wy.T @ Wy)
            self.trace_Ainv_covfactor = np.real_if_close(
                self._compute_trace_Ainv_covfactor0(
                    Y=np.hstack(self.obs_list), unc_scale=psi
                )
                - _trace_Ainv_covfactor_update
            ).item()

            # Create output random variables
            x, A, Ainv = self._get_output_randvars(
                Y_list=self.obs_list, sy_list=self.sy, phi=phi, psi=psi
            )

            # Callback function used to extract quantities from iteration
            if callback is not None:
                callback(
                    xk=x,
                    Ak=A,
                    Ainvk=Ainv,
                    sk=search_dir,
                    yk=obs,
                    alphak=step_size,
                    resid=resid,
                )

            # Iteration increment
            self.iter_ += 1

        # Log information on solution
        info = {
            "iter": self.iter_,
            "maxiter": maxiter,
            "resid_l2norm": np.linalg.norm(resid, ord=2),
            "trace_sol_cov": self.trace_sol_cov,
            "conv_crit": _conv_crit,
            "rel_cond": None,  # TODO: matrix condition from solver (see scipy solvers)
        }

        return x, A, Ainv, info


class NoisySymmetricMatrixBasedSolver(MatrixBasedSolver):
    """
    Solver iteration of the noisy symmetric probabilistic linear solver.

    Implements the solve iteration of the symmetric matrix-based probabilistic linear
    solver taking into account noisy matrix-vector products :math:`y_k = (A + E_k)s_k`
    as described in [1]_ and [2]_.

    Parameters
    ----------
    A : LinearOperator or RandomVariable, shape=(n,n)
        The square matrix or linear operator of the linear system.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    A0 : array-like or LinearOperator or RandomVariable, shape=(n, n), optional
        A square matrix, linear operator or random variable representing the prior
        belief over the linear operator :math:`A`. If an array or linear operator is
        given, a prior distribution is chosen automatically.
    Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
        A square matrix, linear operator or random variable representing the prior
        belief over the inverse :math:`H=A^{-1}`. This can be viewed as taking the form
        of a pre-conditioner. If an array or linear operator is given, a prior
        distribution is chosen automatically.
    x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
        Optional. Prior belief for the solution of the linear system. Will be ignored if
        ``Ainv0`` is given.

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
    .. [1] Wenger, J., de Roos, F. and Hennig, P., Probabilistic Solution of Noisy
       Linear Systems, 2020
    .. [2] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on
       Optimization*, 2015, 25, 234-260

    See Also
    --------
    SymmetricMatrixBasedSolver :
        Class implementing the symmetric probabilistic linear solver.
    """

    def __init__(self, A, b, A0=None, Ainv0=None, x0=None):

        # Transform right hand side to random variable
        if not isinstance(b, probnum.RandomVariable):
            _b = probnum.asrandvar(b)
        else:
            _b = b

        super().__init__(A=A, b=_b, x0=x0)

        # Get or initialize prior parameters
        (
            A0_mean,
            A0_covfactor,
            Ainv0_mean,
            Ainv0_covfactor,
            b_mean,
        ) = self._get_prior_params(A0=A0, Ainv0=Ainv0, x0=x0, b=_b)

        # Matrix prior parameters
        self.A0_mean = linops.aslinop(A0_mean)
        self.A_mean = linops.aslinop(A0_mean)
        self.A0_covfactor = A0_covfactor
        self.Ainv0_mean = linops.aslinop(Ainv0_mean)
        self.Ainv_mean = linops.aslinop(Ainv0_mean)
        self.Ainv0_covfactor = Ainv0_covfactor
        self.b_mean = b_mean

        # Induced distribution on x via Ainv
        # Exp = x = A^-1 b, Cov = 1/2 (W b'Wb + Wbb'W)
        Wb = Ainv0_covfactor @ self.b_mean
        bWb = np.squeeze(Wb.T @ self.b_mean)

        def _mv(x):
            return 0.5 * (bWb * Ainv0_covfactor @ x + Wb @ (Wb.T @ x))

        self.x_cov = linops.LinearOperator(
            shape=(self.n, self.n), dtype=float, matvec=_mv, matmat=_mv
        )
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
            A square matrix, linear operator or random variable representing the prior
            belief over the linear operator :math:`A`. If an array or linear operator is
            given, a prior distribution is chosen automatically.
        Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
            A square matrix, linear operator or random variable representing the prior
            belief over the inverse :math:`H=A^{-1}`. This can be viewed as taking the
            form of a pre-conditioner. If an array or linear operator is given, a prior
            distribution is chosen automatically.
        x0 : array-like, or RandomVariable, shape=(n,)
            Optional. Prior belief for the solution of the linear system. Will be
            ignored if ``A0`` or ``Ainv0`` is given.
        b : RandomVariable, shape=(n,) or (n, nrhs)
            Right-hand side random variable `b` in :math:`A x = b`.

        Returns
        -------
        A0_mean : array-like or LinearOperator, shape=(n,n)
            Prior mean of the linear operator :math:`A`.
        A0_covfactor : array-like or LinearOperator, shape=(n,n)
            Factor :math:`W^A` of the symmetric Kronecker product prior covariance
            :math:`W^A \\otimes_s W^A` of :math:`A`.
        Ainv0_mean : array-like or LinearOperator, shape=(n,n)
            Prior mean of the linear operator :math:`H`.
        Ainv0_covfactor : array-like or LinearOperator, shape=(n,n)
            Factor :math:`W^H` of the symmetric Kronecker product prior covariance
            :math:`W^H \\otimes_s W^H` of :math:`H`.
        b_mean : array-like, shape=(n,nrhs)
            Prior mean of the right hand side :math:`b`.
        """

        # Right hand side mean
        b_mean = b.sample(1)  # TODO: build prior model for rhs and change to b.mean

        # No matrix priors specified
        if A0 is None and Ainv0 is None:
            # No prior information given
            if x0 is None:
                Ainv0_mean = linops.Identity(shape=self.n)
                Ainv0_covfactor = linops.Identity(shape=self.n)
                # Standard normal covariance
                A0_mean = linops.Identity(shape=self.n)
                A0_covfactor = linops.Identity(shape=self.n)
                # TODO: should this be a sample from A to achieve symm. posterior
                # correspondence?
                return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor, b_mean
            # Construct matrix priors from initial guess x0
            elif isinstance(x0, np.ndarray):
                # Sample from linear operator for prior construction
                if isinstance(self.A, probnum.RandomVariable):
                    _A = self.A.sample([1])[0]
                else:
                    _A = self.A
                A0_mean, Ainv0_mean = self._construct_symmetric_matrix_prior_means(
                    A=_A, x0=x0, b=b_mean
                )
                Ainv0_covfactor = Ainv0_mean
                # Standard normal covariance
                A0_covfactor = linops.Identity(shape=self.n)
                # TODO: should this be a sample from A to achieve symm. posterior
                # correspondence?
                return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor, b_mean
            elif isinstance(x0, probnum.RandomVariable):
                raise NotImplementedError

        # Prior on Ainv specified
        if not isinstance(A0, probnum.RandomVariable) and Ainv0 is not None:
            if isinstance(Ainv0, probnum.RandomVariable):
                Ainv0_mean = Ainv0.mean
                Ainv0_covfactor = Ainv0.cov.A
            else:
                Ainv0_mean = Ainv0
                Ainv0_covfactor = Ainv0  # Symmetric posterior correspondence
            try:
                if A0 is not None:
                    A0_mean = A0
                elif isinstance(Ainv0, probnum.RandomVariable):
                    A0_mean = Ainv0.mean.inv()
                else:
                    A0_mean = Ainv0.inv()
            except AttributeError:
                warnings.warn(
                    "Prior specified only for Ainv. Inverting prior mean naively. "
                    "This operation is computationally costly! Specify an inverse "
                    "prior (mean) instead."
                )
                A0_mean = np.linalg.inv(Ainv0.mean)
            except NotImplementedError:
                A0_mean = linops.Identity(self.n)
                warnings.warn(
                    "Prior specified only for Ainv. Automatic prior mean inversion "
                    "not implemented, falling back to standard normal prior."
                )
            # Standard normal covariance
            A0_covfactor = linops.Identity(shape=self.n)
            # TODO: should this be a sample from A to achieve symm. posterior
            # correspondence?
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor, b_mean

        # Prior on A specified
        elif A0 is not None and not isinstance(Ainv0, probnum.RandomVariable):
            if isinstance(A0, probnum.RandomVariable):
                A0_mean = A0.mean
                A0_covfactor = A0.cov.A
            else:
                A0_mean = A0
                A0_covfactor = A0  # Symmetric posterior correspondence
            try:
                if Ainv0 is not None:
                    Ainv0_mean = Ainv0
                elif isinstance(A0, probnum.RandomVariable):
                    Ainv0_mean = A0.mean.inv()
                else:
                    Ainv0_mean = A0.inv()
            except AttributeError:
                warnings.warn(
                    "Prior specified only for A. Inverting prior mean naively. "
                    "This operation is computationally costly! Specify an inverse "
                    "prior (mean) instead."
                )
                Ainv0_mean = np.linalg.inv(A0.mean)
            except NotImplementedError:
                Ainv0_mean = linops.Identity(self.n)
                warnings.warn(
                    "Prior specified only for A. Automatic prior mean inversion "
                    "failed, falling back to standard normal prior."
                )
            # Symmetric posterior correspondence
            Ainv0_covfactor = Ainv0_mean
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor, b_mean
        # Both matrix priors on A and H specified via random variables
        elif isinstance(A0, probnum.RandomVariable) and isinstance(
            Ainv0, probnum.RandomVariable
        ):
            A0_mean = A0.mean
            A0_covfactor = A0.cov.A
            Ainv0_mean = Ainv0.mean
            Ainv0_covfactor = Ainv0.cov.A
            return A0_mean, A0_covfactor, Ainv0_mean, Ainv0_covfactor, b_mean
        else:
            raise NotImplementedError

    def has_converged(self, iter, maxiter, atol=None, rtol=None):
        """
        Check convergence of a linear solver.

        Evaluates a set of convergence criteria based on its input arguments to decide
        whether the iteration has converged.

        Parameters
        ----------
        iter : int
            Current iteration of solver.
        maxiter : int
            Maximum number of iterations
        atol : float
            Absolute tolerance for the uncertainty about the solution estimate. Stops if
            :math:`\\sqrt{\\text{tr}(\\Sigma)}  \\leq \\text{atol}`, where
            :math:`\\Sigma` is the covariance of the solution :math:`x`.
        rtol : float
            Relative tolerance for the uncertainty about the solution estimate. Stops if
            :math:`\\sqrt{\\text{tr}(\\Sigma)} \\leq \\text{rtol} \\lVert x_i \\rVert`,
            where :math:`\\Sigma` is the covariance of the solution :math`x` and
            :math:`x_i` its mean.

        Returns
        -------
        has_converged : bool
            True if the method has converged.
        convergence_criterion : str
            Convergence criterion which caused termination.
        """
        # maximum iterations
        if iter >= maxiter:
            warnings.warn(
                "Iteration terminated. Solver reached the maximum number of iterations."
            )
            return True, "maxiter"
        # uncertainty-based
        if isinstance(self.x_cov, linops.LinearOperator):
            sqrttracecov = np.sqrt(self.x_cov.trace())
        else:
            sqrttracecov = np.sqrt(np.trace(self.x_cov))
        if sqrttracecov <= atol:
            return True, "covar_atol"
        elif sqrttracecov <= rtol * np.linalg.norm(self.x_mean):
            return True, "covar_rtol"
        else:
            return False, ""

    def solve(
        self,
        callback=None,
        maxiter=None,
        atol=10 ** -6,
        rtol=10 ** -6,
        noise_scale=None,
        **kwargs
    ):
        """
        Solve the linear system :math:`Ax=b`.

        Parameters
        ----------
        callback : function, optional
            User-supplied function called after each iteration of the linear solver. It
            is called as ``callback(xk, Ak, Ainvk, sk, yk, alphak, resid, noise_scale)``
            and can be used to return quantities from the iteration. Note that depending
            on the function supplied, this can slow down the solver.
        maxiter : int
            Maximum number of iterations
        atol : float
            Absolute tolerance for the uncertainty about the solution estimate. Stops if
            :math:`\\sqrt{\\text{tr}(\\Sigma)}  \\leq \\text{atol}`, where
            :math:`\\Sigma` is the covariance of the solution :math:`x`.
        rtol : float
            Relative tolerance for the uncertainty about the solution estimate. Stops if
            :math:`\\sqrt{\\text{tr}(\\Sigma)} \\leq \\text{rtol} \\lVert x_i \\rVert`,
            where :math:`\\Sigma` is the covariance of the solution :math`x` and
            :math:`x_i` its mean.
        noise_scale : float
            Assumed (initial) noise scale :math:`\\varepsilon^2`.

        Returns
        -------
        x : RandomVariable, shape=(n,) or (n, nrhs)
            Approximate solution :math:`x` to the linear system. Shape of the return
            matches the shape of ``b``.
        A : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator.
        Ainv : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
        info : dict
            Information on convergence of the solver.
        """
        raise NotImplementedError
