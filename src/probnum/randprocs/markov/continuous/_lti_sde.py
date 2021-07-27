"""LTI SDE models as transitions."""
import functools
from typing import Callable, Optional

import numpy as np
import scipy.integrate
import scipy.linalg

from probnum import randvars
from probnum.randprocs.markov import _transition, discrete
from probnum.randprocs.markov.continuous import _linear_sde, _mfd
from probnum.typing import FloatArgType, IntArgType
from probnum.utils.linalg import tril_to_positive_tril


class LTISDE(_linear_sde.LinearSDE):
    """Linear time-invariant continuous Markov models of the form.

    .. math:: d x(t) = [G x(t) + v] d t + L d w(t).

    In the language of dynamic models,

    x(t) : state process
    G : drift matrix
    v : force term/vector
    L : dispersion matrix.
    w(t) : Wiener process with unit diffusion.

    Parameters
    ----------
    drift_matrix :
        This is F. It is the drift matrix of the SDE.
    force_vector :
        This is U. It is the force vector of the SDE.
    dispersion_matrix :
        This is L. It is the dispersion matrix of the SDE.
    """

    def __init__(
        self,
        drift_matrix: np.ndarray,
        force_vector: np.ndarray,
        dispersion_matrix: np.ndarray,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        # Assert all shapes match
        _check_initial_state_dimensions(drift_matrix, force_vector, dispersion_matrix)

        # Convert input into super() compatible format and initialize super()
        state_dimension = drift_matrix.shape[0]
        wiener_process_dimension = dispersion_matrix.shape[1]

        def drift_matrix_function(t):
            return drift_matrix

        def force_vector_function(t):
            return force_vector

        def dispersion_matrix_function(t):
            return dispersion_matrix

        super().__init__(
            state_dimension=state_dimension,
            wiener_process_dimension=wiener_process_dimension,
            drift_matrix_function=drift_matrix_function,
            dispersion_matrix_function=dispersion_matrix_function,
            force_vector_function=force_vector_function,
        )

        # Initialize remaining attributes
        self._drift_matrix = drift_matrix
        self._force_vector = force_vector
        self._dispersion_matrix = dispersion_matrix
        self._forward_implementation_string = forward_implementation
        self._backward_implementation_string = backward_implementation

    @property
    def drift_matrix(self):
        return self._drift_matrix

    @property
    def force_vector(self):
        return self._force_vector

    @property
    def dispersion_matrix(self):
        return self._dispersion_matrix

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )
        discretised_model = self.discretise(dt=dt)
        return discretised_model.forward_rv(
            rv, t, compute_gain=compute_gain, _diffusion=_diffusion
        )

    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )
        discretised_model = self.discretise(dt=dt)
        return discretised_model.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            _diffusion=_diffusion,
        )

    @functools.lru_cache(maxsize=None)
    def discretise(self, dt):
        """Return a discrete transition model (i.e. mild solution to SDE) using matrix
        fraction decomposition.

        That is, matrices A(h) and Q(h) and vector s(h) such
        that the transition is

        .. math:: x | x_\\text{old} \\sim \\mathcal{N}(A(h) x_\\text{old} + s(h), Q(h)) ,

        which is the transition of the mild solution to the LTI SDE.
        """

        if np.linalg.norm(self.force_vector) > 0:
            zeros = np.zeros((self.state_dimension, self.state_dimension))
            eye = np.eye(self.state_dimension)
            drift_matrix = np.block([[self.drift_matrix, eye], [zeros, zeros]])
            dispersion_matrix = np.concatenate(
                (self.dispersion_matrix, np.zeros(self.dispersion_matrix.shape))
            )
            ah_stack, qh_stack, _ = _mfd.matrix_fraction_decomposition(
                drift_matrix, dispersion_matrix, dt
            )
            proj = np.eye(self.state_dimension, 2 * self.state_dimension)
            proj_rev = np.flip(proj, axis=1)
            ah = proj @ ah_stack @ proj.T
            sh = proj @ ah_stack @ proj_rev.T @ self.force_vector
            qh = proj @ qh_stack @ proj.T
        else:
            ah, qh, _ = _mfd.matrix_fraction_decomposition(
                self.drift_matrix, self.dispersion_matrix, dt
            )
            sh = np.zeros(len(ah))
        return discrete.DiscreteLTIGaussian(
            ah,
            sh,
            qh,
            forward_implementation=self._forward_implementation_string,
            backward_implementation=self._backward_implementation_string,
        )


def _check_initial_state_dimensions(drift_matrix, force_vector, dispersion_matrix):
    """Checks that the matrices all align and are of proper shape.

    Parameters
    ----------
    drift_matrix : np.ndarray, shape=(n, n)
    force_vector : np.ndarray, shape=(n,)
    dispersion_matrix : np.ndarray, shape=(n, s)
    """
    if drift_matrix.ndim != 2 or drift_matrix.shape[0] != drift_matrix.shape[1]:
        raise ValueError("drift_matrixrix not of shape (n, n)")
    if force_vector.ndim != 1:
        raise ValueError("force not of shape (n,)")
    if force_vector.shape[0] != drift_matrix.shape[1]:
        raise ValueError(
            "force not of shape (n,) or drift_matrixrix not of shape (n, n)"
        )
    if dispersion_matrix.ndim != 2:
        raise ValueError("dispersion not of shape (n, s)")
