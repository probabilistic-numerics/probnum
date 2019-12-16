"""Probabilistic numerical methods for solving linear systems."""

import abc
import numpy as np
import scipy.sparse.linalg


class ProbabilisticLinearSolver(abc.ABC):
    """
    Probabilistic linear solvers infer solutions to linear systems in a Bayesian framework.

    Probabilistic numerical linear solvers infer solutions to problems of the form

    .. math:: Ax^*=b,

    where :math:`A \\in \\mathbb{R}^{m \\times n}` and :math:`b \\in \\mathbb{R}^{m}`. They output a probability measure
    which quantifies uncertainty in the solution.
    """

    @abc.abstractmethod
    def solve(self, A, b, maxiter=None):
        """
        Solve the given linear system.

        Parameters
        ----------
        A : array-like
            The matrix of the linear system.
        b : array-like
            The right-hand-side of the linear system.
        maxiter : int
            Maximum number of iterations.

        Returns
        -------

        """
        pass


def _check_linear_system(A, b):
    """
    Check linear system dimensions.

    Raises an exception if the input arguments are not compatible.

    Parameters
    ----------
    A : array-like
        Matrix.
    b : array-like
        Right-hand side vector.

    """
    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimension mismatch.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")


class MatrixBasedLinearSolver(ProbabilisticLinearSolver):
    """
    Probabilistic linear solver using prior information on the matrix inverse.

    In the setting where :math:`A` is a symmetric positive-definite matrix, this solver takes prior information either
    on the matrix inverse :math:`H=A^{-1}` and outputs a posterior belief over :math:`H`. For a specific prior choice
    this recovers the iterates of the conjugate gradient method. This code implements the method described in [1]_ and
    [2]_.

    .. [1] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260
    .. [2] Hennig, P. et al., Probabilistic Numerics, 2020

    Attributes
    ----------
    H_mean : array-like
        Mean of the Gaussian prior on :math:`H=A^{-1}`.
    H_cov_kronfac : array-like
        Kronecker factor of the covariance :math:`W_H \\otimes W_H` of the Gaussian prior on :math:`H=A^{-1}`.
    """

    def __init__(self, H_mean=None, H_cov_kronfac=None):
        # todo: refactor as dimension of linear operator is not known at this point
        if H_mean is not None:
            self.H_mean = scipy.sparse.linalg.aslinearoperator(H_mean)
        else:
            self.H_mean = None
        if H_cov_kronfac is not None:
            self.H_cov_kronfac = scipy.sparse.linalg.aslinearoperator(H_cov_kronfac)
        else:
            self.H_cov_kronfac = None

    def _has_converged(self, iter, maxiter, resid, resid_tol):
        """

        Parameters
        ----------
        iter : int
            Current iteration of solver.
        maxiter : int
            Maximum number of iterations
        resid : array-like
            Residual vector :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert` of the current iteration.
        resid_tol : float
            Residual tolerance.

        Returns
        -------
        has_converged : bool
        """
        # maximum iterations
        if iter >= maxiter:
            return True
        # residual below error tolerance
        # todo: add / replace with relative tolerance
        elif np.linalg.norm(resid) < resid_tol:
            return True
        # uncertainty-based
        # todo: based on posterior contraction
        else:
            return False

    def _mean_update_operator(self, u, v, shape):
        """
        Implements the rank 2 update term for the posterior mean of :math:`H`.

        Parameters
        ----------
        u : array-like
            Update vector :math:`u_i=\\frac{W_iy_i}{y_i^{\\top}W_iy_i}`
        v : array-like
            Update vector :math:`v_i=\\Delta - \\frac{1}{2} u_iy_i^{\\top}\\Delta`
        shape : tuple
            Shape of the resulting update operator.

        Returns
        -------
        update : LinearOperator
        """

        def mv(x):
            return np.dot(v, x) * u + np.dot(u, x) * v

        def mm(M):
            return np.outer(u, np.matmul(M, v)) + np.outer(v, np.matmul(u, M))

        return scipy.sparse.linalg.LinearOperator(
            shape=shape,
            matvec=mv,
            rmatvec=mv,
            matmat=mm,
            dtype=u.dtype
        )

    def _cov_kron_fac_update_operator(self, u, Wy, shape):
        """
        Implements the rank 1 update term for the posterior covariance Kronecker factor :math:`W`.

        Parameters
        ----------
        u : array-like
            Update vector :math:`u_i=\\frac{W_iy_i}{y_i^{\\top}W_iy_i}`
        Wy : array-like
            Update vector :math:`W_iy_i`
        shape : tuple
            Shape of the resulting update operator.

        Returns
        -------
        update : LinearOperator
        """

        def mv(x):
            return np.dot(Wy, x) * u

        def mm(M):
            return np.outer(u, np.matmul(M, Wy))

        return scipy.sparse.linalg.LinearOperator(
            shape=shape,
            matvec=mv,
            rmatvec=mv,
            matmat=mm,
            dtype=u.dtype
        )

    def solve(self, A, b, maxiter=None, resid_tol=10 ** -6):
        """
        Solve the given linear system for symmetric, positive-definite :math:`A`.

        Parameters
        ----------
        A : array-like
            The matrix of the linear system.
        b : array-like
            The right-hand-side of the linear system.
        maxiter : int, default=len(b)*10
            Maximum number of iterations.
        resid_tol : float
            Residual tolerance. If :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert < \\text{tol}`, the iteration
            terminates.

        Returns
        -------
        x : array-like
            Approximate solution :math:`Ax \\approx b` to the linear system.
        """

        # make linear system
        A = scipy.sparse.linalg.interface.aslinearoperator(A)
        b = np.asanyarray(b)

        # check linear system dimensionality
        _check_linear_system(A=A, b=b)

        # setup
        n = len(b)
        if maxiter is None:
            maxiter = n * 10
        if self.H_mean is None:
            # todo: use / implement better operator classes (PyLops?) that can perform .T, .todense() for H_mean as well
            self.H_mean = scipy.sparse.linalg.interface.IdentityOperator(shape=(n, n))
        if self.H_cov_kronfac is None:
            self.H_cov_kronfac = scipy.sparse.linalg.interface.IdentityOperator(shape=(n, n))

        # initialization
        iter = 0
        x = self.H_mean.matvec(b)
        resid = A.matvec(x) - b

        # iteration with stopping criteria
        # todo: extract iteration and make into iterator
        while not self._has_converged(iter=iter, maxiter=maxiter, resid=resid, resid_tol=resid_tol):

            # compute search direction (with implicit reorthogonalization)
            search_dir = - self.H_mean.matvec(resid)

            # perform action and observe
            obs = A.matvec(search_dir)

            # compute step size
            step_size = - np.dot(search_dir, resid) / np.dot(search_dir, obs)
            # step and residual update
            x = x + step_size * search_dir
            resid = resid + step_size * obs

            # (symmetric) mean and covariance updates
            Wy = self.H_cov_kronfac.matvec(obs)
            delta = search_dir - self.H_mean.matvec(obs)
            u = Wy / np.dot(obs, Wy)
            v = delta - 0.5 * np.dot(obs, delta) * u

            # rank 2 mean update operator (+= uv' + vu')
            # todo: speedup: implement full update as operator and do not rely on +?
            self.H_mean = self.H_mean + self._mean_update_operator(u=u, v=v, shape=(n, n))

            # rank 1 covariance kronecker factor update operator (-= u(Wy)')
            self.H_cov_kronfac = self.H_cov_kronfac - self._cov_kron_fac_update_operator(u=u, Wy=Wy, shape=(n, n))

            # iteration increment
            iter += 1

        # information about convergence
        info = {
            "n_iter": iter,
            "resid_norm": np.linalg.norm(resid)
        }
        print(info)
        # todo: return solution, some general distribution class and dict with
        #  convergence info (iter, resid, convergence criterion)
        return x, info


class SolutionBasedConjugateGradients(ProbabilisticLinearSolver):
    """
    Conjugate Gradients using prior information on the solution of the linear system.

    In the setting where :math:`A` is a symmetric positive-definite matrix, this solver takes prior information
    on the solution and outputs a posterior belief over :math:`x`. This code implements the
    method described in [1]_.

    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian Analysis*, 2019, 14, 937-1012

    Attributes
    ----------
    x_mean : array-like
        Mean of the Gaussian prior on :math:`x`.
    x_cov : array-like
        Covariance :math:`\\Sigma` of the Gaussian prior on :math:`x`.
    """

    def __init__(self, x_mean=None, x_cov=None):
        self.x_mean = x_mean
        self.x_cov = x_cov

    def solve(self, A, b, maxiter=None):
        """
        Solve the given linear system.

        Parameters
        ----------
        A : array-like
            The matrix of the linear system.
        b : array-like
            The right-hand-side of the linear system.
        maxiter : int
            Maximum number of iterations.

        Returns
        -------

        """

        # check linear system dimensionality
        _check_linear_system(A=A, b=b)

        raise NotImplementedError("Not yet implemented.")
