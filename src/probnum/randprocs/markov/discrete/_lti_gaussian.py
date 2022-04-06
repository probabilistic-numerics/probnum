"""Discrete, linear, time-invariant Gaussian transitions."""


from probnum import randvars
from probnum.randprocs.markov.discrete import _linear_gaussian
from probnum.typing import ArrayLike, LinearOperatorLike


class LTIGaussian(_linear_gaussian.LinearGaussian):
    r"""Discrete, linear, time-invariant transitions with additive, Gaussian noise.

    .. math:: y = G x + v, \quad v \sim \mathcal{N}(m, C)

    for some transition matrix :math:`G` and Noise :math:`v`.

    Parameters
    ----------
    transition_matrix
        Transition matrix :math:`G`.
    noise
        Noise :math:`v`.
    forward_implementation
        A string indicating the choice of forward implementation.
    backward_implementation
        A string indicating the choice of backward implementation.

    Raises
    ------
    TypeError
        If ``transition_matrix`` and ``noise`` have incompatible shapes.
    """

    def __init__(
        self,
        *,
        transition_matrix: LinearOperatorLike,
        noise: randvars.RandomVariable,
        forward_implementation: str = "classic",
        backward_implementation: str = "classic",
    ):
        _assert_shapes_match(transition_matrix, noise)

        output_dim, input_dim = transition_matrix.shape
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            transition_matrix_fun=lambda t: transition_matrix,
            noise_fun=lambda t: noise,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

        self._transition_matrix = transition_matrix
        self._noise = noise

    @property
    def transition_matrix(self) -> LinearOperatorLike:
        return self._transition_matrix

    @property
    def noise(self) -> randvars.RandomVariable:
        return self._noise

    @classmethod
    def from_linop(
        cls,
        transition_matrix: LinearOperatorLike,
        noise_mean: ArrayLike,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        """Turn a linear operator (or numpy array) into a noise-free transition."""

        # Currently, this is only a numpy array.
        # In the future, once linops are more widely adopted here, this will become
        # a linop.
        if transition_matrix.ndim != 2:
            raise ValueError
        return cls(
            transition_matrix=transition_matrix,
            noise=randvars.Constant(noise_mean),
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )


def _assert_shapes_match(transition_matrix, noise):
    if transition_matrix.ndim != 2:
        raise TypeError(
            f"transition_matrix.ndim = 2 expected. "
            f"transition_matrix.ndim = {transition_matrix.ndim} received."
        )
    if noise.ndim != 1:
        raise TypeError(
            f"noise.ndim = 1 expected. " f"noise.ndim = {noise.ndim} received."
        )
    if transition_matrix.shape[0] != noise.shape[0]:
        raise TypeError("Dimension of transition_matrix and noise do not align.")
