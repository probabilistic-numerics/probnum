"""Integrated processes such as the integrated Wiener process or the Matern process."""

import numpy as np

from probnum.randprocs.markov.integrator import _preconditioner

__all__ = ["IntegratorMixIn"]


class IntegratorMixIn:
    r"""MixIn for integrators: joint models of a state and a number of its derivatives.

    An integrator is a special kind of random process
    that models a stack of a state and its first :math:`\nu` time-derivatives.
    For instances, this includes integrated Wiener processes or Matern processes.

    In ProbNum, integrators are always driven by :math:`d` dimensional Wiener processes.

    The present MixIn provides the following additional functionality:
        * Projection to the coordinate of interest (the zeroth derivative)
        * Availability of preconditioning
        * Reordering of states: derivative-wise to coordinate-wise
    """

    def __init__(self, num_derivatives, state_ordering="coordinate"):

        if state_ordering != "coordinate":
            raise ValueError("ProbNum only supports coordinate-wise ordering")

        self._num_derivatives = num_derivatives
        self._precon = _preconditioner.NordsieckLikeCoordinates.from_order(
            num_derivatives, self.wiener_process_dimension
        )
        self._state_ordering = state_ordering

    @property
    def state_ordering(self):
        return self._state_ordering

    @property
    def num_derivatives(self):
        return self._num_derivatives

    @property
    def precon(self):
        return self._precon

    def select_derivative(self, state, derivative):

        # Once we allow changed orderings, extend this functionality here.
        # Due to the behaviour in __init__,
        # non-coordinate-representation should be impossible
        assert self.state_ordering == "coordinate"

        derivative_indices = np.arange(
            start=derivative, stop=self.state_dimension, step=(self.num_derivatives + 1)
        )
        return np.take(state, indices=derivative_indices)

    def derivative_selection_operator(self, derivative):

        selection_unit_vector = np.eye(self.num_derivatives + 1)[:, derivative]
        selection_unit_vector_as_matrix = selection_unit_vector.reshape(
            (1, self.num_derivatives + 1)
        )
        selection_matrix = np.kron(
            np.eye(self.wiener_process_dimension), selection_unit_vector_as_matrix
        )
        return selection_matrix

    def reorder_state(self, state, current_ordering, target_ordering):
        """Change e.g. coordinate-wise ordering to derivative-wise ordering."""
        reorder_function = {
            (
                "coordinate",
                "derivative",
            ): self._reorder_state_from_coordinate_to_derivative,
            (
                "derivative",
                "coordinate",
            ): self._reorder_state_from_derivative_to_coordinate,
        }
        try:
            return reorder_function[(current_ordering, target_ordering)](state=state)
        except KeyError:
            msg1 = "Reordering is not supported for given keys"
            msg2 = f" '{current_ordering}' and '{target_ordering}'. "
            msg3 = "Only combinations of 'derivative' and 'coordinate' are supported."
            raise ValueError(msg1 + msg2 + msg3)

    def _reorder_state_from_derivative_to_coordinate(self, state):
        d, dim = self.wiener_process_dimension, self.state_dimension
        stride = lambda i: np.arange(start=i, stop=dim, step=d).reshape((-1, 1))
        indices = np.vstack([stride(i) for i in range(d)])[:, 0]
        return state[indices]

    def _reorder_state_from_coordinate_to_derivative(self, state):
        n, dim = self.num_derivatives, self.state_dimension
        stride = lambda i: np.arange(start=i, stop=dim, step=(n + 1)).reshape((-1, 1))
        indices = np.vstack([stride(i) for i in range(n + 1)])[:, 0]
        return state[indices]

    @property
    def _derivwise2coordwise_projmat(self) -> np.ndarray:
        r"""Projection matrix to change the ordering of the state representation in an :class:`Integrator` from coordinate-wise to derivative-wise representation.

        A coordinate-wise ordering is

        .. math:: (y_1, \dot y_1, \ddot y_1, y_2, \dot y_2, ..., y_d^{(\nu)})

        and a derivative-wise ordering is

        .. math:: (y_1, y_2, ..., y_d, \dot y_1, \dot y_2, ..., \dot y_d, \ddot y_1, ..., y_d^{(\nu)}).

        Default representation in an :class:`Integrator` is coordinate-wise ordering, but sometimes, derivative-wise ordering is more convenient.

        See Also
        --------
        :attr:`Integrator._convert_coordwise_to_derivwise`
        :attr:`Integrator._convert_derivwise_to_coordwise`

        """
        dim = (self.num_derivatives + 1) * self.wiener_process_dimension
        projmat = np.zeros((dim, dim))
        E = np.eye(dim)
        for q in range(self.num_derivatives + 1):

            projmat[q :: (self.num_derivatives + 1)] = E[
                q
                * self.wiener_process_dimension : (q + 1)
                * self.wiener_process_dimension
            ]
        return projmat

    @property
    def _coordwise2derivwise_projmat(self) -> np.ndarray:
        r"""Projection matrix to change the ordering of the state representation in an :class:`Integrator` from derivative-wise to coordinate-wise representation.

        A coordinate-wise ordering is

        .. math:: (y_1, \dot y_1, \ddot y_1, y_2, \dot y_2, ..., y_d^{(\nu)})

        and a derivative-wise ordering is

        .. math:: (y_1, y_2, ..., y_d, \dot y_1, \dot y_2, ..., \dot y_d, \ddot y_1, ..., y_d^{(\nu)}).

        Default representation in an :class:`Integrator` is coordinate-wise ordering, but sometimes, derivative-wise ordering is more convenient.

        See Also
        --------
        :attr:`Integrator._convert_coordwise_to_derivwise`
        :attr:`Integrator._convert_derivwise_to_coordwise`

        """
        return self._derivwise2coordwise_projmat.T
