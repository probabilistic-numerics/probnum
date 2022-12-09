"""Helper functions for the quad package"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from probnum.typing import IntLike

from .typing import DomainLike, DomainType


def as_domain(
    domain: DomainLike, input_dim: Optional[IntLike]
) -> Tuple[DomainType, int]:
    """Static method that converts the integration domain and input dimension to
    the correct types.

    If no ``input_dim`` is given, the dimension is inferred from the size of
    domain limits ``domain[0]`` and ``domain[1]``. These must be either scalars
    or arrays of equal length.

    If ``input_dim`` is given, the domain limits must be either scalars or arrays.
    If they are arrays, their lengths must equal ``input_dim``. If they are scalars,
    the domain is taken to be the hypercube
    ``[domain[0], domain[1]] x .... x [domain[0], domain[1]]``
    of dimension ``input_dim``.

    Parameters
    ----------
    domain
        The integration domain as supplied.
    input_dim
        The input dimensionality as supplied.

    Returns
    -------
    converted_domain :
        The integration domain.
    converted_input_dim :
        The input dimensionality.

    Raises
    ------
    ValueError
        If ``input_dim`` is not positive.
        If domain has too many or too little elements.
        If the bounds of the domain have differing sizes.
        If ``input_dim`` is incompatible with domain bounds.
        If bounds have wrong shape.
        If integration domain is empty.
    """
    if input_dim is not None and input_dim < 1:
        raise ValueError(
            f"Input dimension must be positive. Current value is ({input_dim})."
        )

    if len(domain) != 2:
        raise ValueError(f"'domain' must be of length 2 ({len(domain)}).")

    # Domain limits must have equal dimensions
    if np.size(domain[0]) != np.size(domain[1]):
        raise ValueError(
            f"Domain limits must be given either as scalars or arrays "
            f"of equal dimension. Current sizes are ({np.size(domain[0])}) "
            f"and ({np.size(domain[1])})."
        )

    domain_dim = int(np.size(domain[0]))

    # Input dimension not given, infer it from the domain.
    if input_dim is None:
        input_dim = domain_dim
        domain_a, domain_b = domain

    # Bounds are given as scalars: Expand domain limits.
    elif domain_dim == 1:
        domain_a = np.full((input_dim,), domain[0])
        domain_b = np.full((input_dim,), domain[1])

    # Bounds are given as arrays
    else:
        # Size of domain and input dimension do not match
        if input_dim != domain_dim:
            raise ValueError(
                f"If domain limits are not scalars, their lengths "
                f"must match the input dimension ({input_dim})."
            )
        domain_a = domain[0]
        domain_b = domain[1]

    # convert bounds to 1D arrays if necessary
    domain_a = np.atleast_1d(domain_a)
    domain_b = np.atleast_1d(domain_b)
    if domain_a.ndim > 1 or domain_b.ndim > 1:
        raise ValueError(
            f"Upper ({domain_b.ndim}) or lower ({domain_a.ndim}) "
            f"bounds have too many dimensions."
        )

    # Make sure the domain is non-empty
    if not np.all(domain_a < domain_b):
        raise ValueError(
            "Integration domain must be non-empty. For example, some "
            "lower bounds may be larger than their corresponding "
            "upper bounds."
        )

    domain = (domain_a, domain_b)
    return domain, input_dim
