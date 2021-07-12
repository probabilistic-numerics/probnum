"""Randomly draw nodes from the measure to use for integration."""

import numpy as np

from probnum.quad._integration_measures import IntegrationMeasure


def sample_from_measure(
    rng: np.random.Generator, nevals: int, measure: IntegrationMeasure
) -> np.ndarray:
    r"""Acquisition policy: Draw random samples from the integration measure.

    Parameters
    ----------
    rng :
        Random number generator.

    nevals :
        Number of function evaluations.

    measure :
        The integration measure :math:`\mu`.

    Returns
    -------
    x : np.ndarray
        Nodes where the integrand will be evaluated.
    """
    return measure.sample(rng=rng, n_sample=nevals).reshape(nevals, -1)
