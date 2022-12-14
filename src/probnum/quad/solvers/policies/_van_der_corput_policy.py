"""Van der Corput points for integration on 1D intervals."""

from __future__ import annotations

from typing import Optional

import numpy as np

from probnum.quad.integration_measures import IntegrationMeasure
from probnum.quad.solvers._bq_state import BQState
from probnum.typing import IntLike

from ._policy import Policy


class VanDerCorputPolicy(Policy):
    r"""Pick nodes from the van der Corput sequence.

    The van der Corput sequence [1]_ is

    .. math:: 0.5, 0.25, 0.75, 0.125, 0.625, \ldots

    If the integration domain is not [0, 1], the van der Corput sequence is linearly
    mapped to the domain. The domain must be finite.

    Parameters
    ----------
    batch_size
        Size of batch of nodes when calling the policy once.
    measure
        The integration measure with finite domain.

    Raises
    ------
    ValueError
        If input dimension is not 1.
    ValueError
        If measure domain is not bounded.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Van_der_Corput_sequence
    """

    def __init__(self, batch_size: IntLike, measure: IntegrationMeasure) -> None:
        super().__init__(batch_size=batch_size)

        if int(measure.input_dim) > 1:
            raise ValueError("Policy 'vdc' works only when the input dimension is one.")

        domain_a = measure.domain[0]
        domain_b = measure.domain[1]
        if np.Inf in np.hstack([abs(domain_a), abs(domain_b)]):
            raise ValueError("Policy 'vdc' works only for bounded domains.")

        self.domain_a = domain_a
        self.domain_b = domain_b

    @property
    def requires_rng(self) -> bool:
        return False

    def __call__(
        self, bq_state: BQState, rng: Optional[np.random.Generator]
    ) -> np.ndarray:
        n_nodes = bq_state.nodes.shape[0]
        vdc_seq = VanDerCorputPolicy.van_der_corput_sequence(
            n_nodes + 1, n_nodes + 1 + self.batch_size
        )
        transformed_vdc_seq = vdc_seq * (self.domain_b - self.domain_a) + self.domain_a
        return transformed_vdc_seq.reshape((self.batch_size, 1))

    @staticmethod
    def van_der_corput_sequence(
        n_start: int, n_end: Optional[int] = None
    ) -> np.ndarray:
        r"""Returns elements ``n_start``, ``n_start + 1``, ..., ``n_end - 1`` in the van
        der Corput sequence.

        .. math:: 0.5, 0.25, 0.75, 0.125, 0.625, \ldots

        If no ``n_end`` is given, only a single element in the
        sequence is returned.

        Parameters
        ----------
        n_start
            First element of the van der Corput to be included (inclusive).
        n_end
            Last element of the van der Corput to be included (exclusive). If not given,
            only the ``n_start`` element is returned.

        Returns
        -------
        vdc_seq
            Array containing elements from ``n_start`` to ``n_end - 1`` of the van der
            Corput sequence.
        """

        # pylint: disable=invalid-name
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
