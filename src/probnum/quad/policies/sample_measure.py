"""Randomly draw nodes from the measure to use for integration."""

import numpy as np

from probnum.quad._integration_measures import IntegrationMeasure


def sample_from_measure(nevals: int, measure: IntegrationMeasure) -> np.ndarray:
    r"""Acquisition policy: Draw random samples from the integration measure.

    Parameters
    ----------
    nevals : int
        Number of function evaluations.

    measure :
        The integration measure :math:`\mu`.

    Returns
    -------
    x : np.ndarray
        Nodes where the integrand will be evaluated.
    """
    return measure.sample(nevals).reshape(nevals, -1)
