"""This module implements some test statistics that are used in multiple test suites."""


import numpy as np

__all__ = ["chi_squared_statistic"]


def chi_squared_statistic(realisations, means, covs):
    """Compute the multivariate chi-squared test statistic for a set of realisations of
    a random variable.

    For :math:`N`, :math:`d`-dimensional realisations :math:`x_1, ..., x_N`
    with (assumed) means :math:`m_1, ..., m_N` and covariances
    :math:`C_1, ..., C_N`, compute the value

    .. math:`\\chi^2 = \\frac{1}{Nd} \\sum_{n=1}^N (x_n - m_n)^\\top C_n^{-1}(x_n - m_n).`

    If it is roughly equal to 1, the samples are likely to correspond to given mean and covariance.

    Parameters
    ----------
    realisations : array_like
        :math:`N` realisations of a :math:`d`-dimensional random variable. Shape (N, d).
    means : array_like
        :math:`N`, :math:`d`-dimensional (assumed) means of a random variable. Shape (N, d).
    realisations : array_like
        :math:`N`, :math:`d \\times d`-dimensional (assumed) covariances of a random variable. Shape (N, d, d).
    """
    if not realisations.shape == means.shape == covs.shape[:-1]:
        print(realisations.shape, means.shape, covs.shape)
        raise TypeError("Inputs do not align")
    centered_realisations = realisations - means
    centered_2 = np.linalg.solve(covs, centered_realisations)
    return _dot_along_last_axis(centered_realisations, centered_2).mean()


def _dot_along_last_axis(a, b):
    """Dot product of (N, K) and (N, K) into (N,).

    Extracted, because otherwise I keep having to look up einsum...
    """
    return np.einsum("...j, ...j->...", a, b)
