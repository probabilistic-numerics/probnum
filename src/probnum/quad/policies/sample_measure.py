"""Randomly draw nodes from the measure to use for integration."""

import numpy as np

from probnum.quad.bq_methods.bq_state import BQState


def sample_from_measure(batch_size: int, bq_state: BQState, **kwargs) -> np.ndarray:
    return bq_state.measure.sample(batch_size)
