"""Van der Corput points for integration on 1D intervals."""

from typing import Optional

import numpy as np

from probnum.quad.solvers.bq_state import BQState

from ._policy import Policy

# pylint: disable=invalid-name


class VanDerCorputPolicy(Policy):
    r"""Pick nodes from the van der Corput sequence [1]_

    .. math:: 0.5, 0.25, 0.75, 0.125, 0.625, \ldots

    that is linearly mapped to a one-dimensional finite integration domain.

    Parameters
    ----------
    domain_a
        Starting point of the interval. Must be finite.
    domain_b
        End point of the interval. Must be finite.
    batch_size
        Size of batch of nodes when calling the policy once.

    References
    --------
    .. [1] https://en.wikipedia.org/wiki/Van_der_Corput_sequence
    """

    def __init__(self, domain_a: float, domain_b: float, batch_size: int) -> None:
        super().__init__(batch_size=batch_size)
        self.domain_a = domain_a
        self.domain_b = domain_b

    def __call__(self, bq_state: BQState) -> np.ndarray:
        n_nodes = bq_state.nodes.shape[0]
        vdc_seq = VanDerCorputPolicy.van_der_corput_sequence(
            n_nodes + 1, n_nodes + 1 + self.batch_size
        )
        transformed_vdc_seq = vdc_seq * (self.domain_b - self.domain_a) + self.domain_a
        return transformed_vdc_seq.reshape((self.batch_size, 1))

    @staticmethod
    def van_der_corput_sequence(n_start: int, n_end: Optional[int] = None):
        """Returns elements n_start, n_start + 1, ..., n_end - 1 in the van der
        Corput sequence.

            0.5, 0.25, 0.75, 0.125, 0.625, ...

        If no ``n_end`` is given, only a single element in the
        sequence is returned."""

        if n_end is None:
            n_end = n_start + 1
        vdc_seq = np.zeros((n_end - n_start,))
        ind = 0
        for m in range(n_start, n_end):
            q = 0.0
            base_inv = 0.5
            n = m
            while n != 0:
                q = q + (n % 2) * base_inv
                n = n // 2
                base_inv = base_inv / 2.0
            vdc_seq[ind] = q
            ind += 1
        return vdc_seq
