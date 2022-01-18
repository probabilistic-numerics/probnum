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
        transition_matrix: np.ndarray,
        process_noise: randvars.RandomVariable,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        _assert_shapes_match(transition_matrix, process_noise)
        output_dim, input_dim = transition_matrix.shape

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            state_trans_mat_fun=lambda t: transition_matrix,
            process_noise_fun=lambda t: process_noise,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )
        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

        self._transition_matrix = transition_matrix
        self._process_noise = process_noise

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def process_noise(self):
        return self._process_noise

    @classmethod
    def from_linop(
        cls,
        transition_matrix: np.ndarray,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        """Turn a linear operator (or numpy array) into a deterministic transition."""
        # Currently, this is only a numpy array.
        # In the future, once linops are more widely adopted here, this will become a linop.
        if transition_matrix.ndim != 2:
            raise ValueError
        return cls(
            state_trans_mat=transition_matrix,
            process_noise=randvars.Constant(np.zeros(transition_matrix.shape[0])),
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )


def _assert_shapes_match(transition_matrix, process_noise):
    if transition_matrix.ndim != 2:
        raise TypeError(
            f"transition_matrix.ndim = 2 expected. "
            f"transition_matrix.ndim = {transition_matrix.ndim} received."
        )
    if process_noise.ndim != 1:
        raise TypeError(
            f"process_noise.ndim = 1 expected. "
            f"process_noise.ndim = {process_noise.ndim} received."
        )
    if transition_matrix.shape[0] != process_noise.shape[0]:
        raise TypeError(
            f"Dimension of transition_matrix and process_noise do not align."
        )
