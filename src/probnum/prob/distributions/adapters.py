"""
Probability distribution adapters.

Makes e.g. scipy.stats distributions consistent with the Distribution
class and hence usable within RandomVariable objects.
"""

from probnum.prob.distributions.distribution import Distribution


def scipy2dist(rvcont):
    """
    Transform a scipy.stats.rv_continuous object into a Distribution.

    Parameters
    ----------
    rvcont : scipy.stats.rv_continuous
        Random variable object to be transformed.

    Returns
    -------
    Distribution
        scipy.stats.rv_continuous wrapped by a Distribution.
    """
    try:
        cov = rvcont.cov
    except AttributeError:
        cov = None
    parameters = {"var": rvcont.var}
    return Distribution(parameters=parameters,
                        pdf=rvcont.pdf, logpdf=rvcont.logpdf,
                        cdf=rvcont.cdf, logcdf=rvcont.logcdf,
                        sample=rvcont.rvs, mean=rvcont.mean,
                        cov=cov, dtype=None,
                        random_state=rvcont.random_state)


def pyro2dist(pyrodist):
    """
    Transforms a pyro.Distribution object into a Distribution.

    Parameters
    ----------
    pyrodist : pyro.Distribution
        Distribution object to be transformed.

    Returns
    -------
    Distribution
        pyro.Distribution wrapped by a Distribution.
    """
    raise NotImplementedError
