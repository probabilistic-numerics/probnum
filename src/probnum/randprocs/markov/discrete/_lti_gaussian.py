"""Discrete, linear, time-invariant Gaussian transitions."""


from typing import Optional

import numpy as np

from probnum import randvars
from probnum.randprocs.markov.discrete import _linear_gaussian

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property  # pylint: disable=ungrouped-imports
except ImportError:

    from cached_property import cached_property


class LTIGaussian(_linear_gaussian.LinearGaussian):
    """Discrete, linear, time-invariant Gaussian transition models of the form.

    .. math:: x_{i+1} \\sim \\mathcal{N}(G x_i + v, S)

    for some dynamics matrix :math:`G`, force vector :math:`v`,
    and diffusion matrix :math:`S`.

    Parameters
    ----------
    state_trans_mat :
        State transition matrix :math:`G`.
    shift_vec :
        Shift vector :math:`v`.
    proc_noise_cov_mat :
        Process noise covariance matrix :math:`S`.

    Raises
    ------
    TypeError
        If state_trans_mat, shift_vec and proc_noise_cov_mat have incompatible shapes.

    See Also
    --------
    :class:`DiscreteModel`
    :class:`NonlinearGaussianLinearModel`
    """

    def __init__(
        self,
        *,
        state_trans_mat: np.ndarray,
        process_noise: randvars.RandomVariable,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        _check_dimensions(state_trans_mat, process_noise.mean, process_noise.cov)
        output_dim, input_dim = state_trans_mat.shape

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            state_trans_mat_fun=lambda t: state_trans_mat,
            process_noise_fun=lambda t: process_noise,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation
        self.state_trans_mat = state_trans_mat
        self.process_noise = process_noise

    @classmethod
    def from_linop(
        cls,
        state_trans_mat: np.ndarray,
        shift_vec: np.ndarray,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        """Turn a linear operator (or numpy array) into a deterministic transition."""
        # Currently, this is only a numpy array.
        # In the future, once linops are more widely adopted here, this will become a linop.
        if state_trans_mat.ndim != 2:
            raise ValueError
        return cls(
            state_trans_mat=state_trans_mat,
            process_noise=randvars.Constant(shift_vec),
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )


def _check_dimensions(state_trans_mat, shift_vec, proc_noise_cov_mat):
    """LTI SDE model needs matrices which are compatible with each other in size."""
    if state_trans_mat.ndim != 2:
        raise TypeError(
            f"dynamat.ndim=2 expected. dynamat.ndim={state_trans_mat.ndim} received."
        )
    if shift_vec.ndim != 1:
        raise TypeError(
            f"shift_vec.ndim=1 expected. shift_vec.ndim={shift_vec.ndim} received."
        )
    if proc_noise_cov_mat.ndim != 2:
        raise TypeError(
            f"proc_noise_cov_mat.ndim=2 expected. proc_noise_cov_mat.ndim={proc_noise_cov_mat.ndim} received."
        )
    if (
        state_trans_mat.shape[0] != shift_vec.shape[0]
        or shift_vec.shape[0] != proc_noise_cov_mat.shape[0]
        or proc_noise_cov_mat.shape[0] != proc_noise_cov_mat.shape[1]
    ):
        raise TypeError(
            f"Dimension of dynamat, force_vector and diffmat do not align. "
            f"Expected: dynamat.shape=(N,*), force_vector.shape=(N,), diffmat.shape=(N, N).     "
            f"Received: dynamat.shape={state_trans_mat.shape}, force_vector.shape={shift_vec.shape}, "
            f"proc_noise_cov_mat.shape={proc_noise_cov_mat.shape}."
        )
