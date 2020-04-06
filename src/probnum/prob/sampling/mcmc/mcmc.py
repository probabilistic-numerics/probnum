"""
Convenience functions for Metropolis-Hastings sampling.

Functionalities
---------------
* Random walk proposals
* Langevin proposals
* Langevin proposals with preconditioning
* Hamiltonian MC
* Hamiltonian MC with preconditioning

Access
------
The functions in this module can be accessed either
via diffeq.mcmc.conveniencefunctions.rwmh(...)
or viia diffeq.mcmc.rwmh(...) due to the
import in the __init__ file.

Note
----
The functionality of this module is restricted to log-densities,
i.e. densities of the form p(s) = exp(-E(s)). We work with E(s) only.
The reason is that in Bayesian inference, evaluations of exp(-E(s))
are too instable in a numerical sense.
"""

from probnum.optim import objective
from probnum.prob.sampling.mcmc import randomwalk, langevin, hamiltonian


def rwmh(logpdf, nsamps, initstate, pwidth):
    """
    Convenience function for Metropolis-Hastings sampling with
    random walk proposal kernel.

    Examples
    --------


    Sampling from a Gaussian distribution.


    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> np.random.seed(1)
    >>>
    >>> def logpdf(x):
    ...     return x.T @ x / 2.0
    ...
    >>> xval = np.linspace(-4, 4, 200)
    >>> yval = np.exp(-xval**2 / 2.0)/np.sqrt(2*np.pi)
    >>> __ = plt.plot(xval, yval)
    >>>
    >>> a, b, c = rwmh(logpdf, 7500, np.array([.5]), pwidth=18.0)
    >>> __ = plt.hist(a[:, 0], bins=50, density=True, alpha=0.5)
    >>> __ = plt.title("RW Samples")
    >>> plt.show()


    Raising a warning about bad acceptance ratios.


    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> np.random.seed(1)
    >>>
    >>> def logpdf(x):
    ...     return x.T @ x / 2.0
    ...
    >>> a, b, c = rwmh(logpdf, 100, np.array([.5]), pwidth=0.1)
    !!! Careful: acc_ratio is not near optimality
    !!! Desired: [0.15, 0.3], got: 0.9
    """
    logdens = objective.Objective(logpdf)
    rwmh = randomwalk.RandomWalkMH(logdens)
    return rwmh.sample_nd(nsamps, initstate, pwidth)


def mala(logpdf, loggrad, nsamps, initstate, pwidth):
    """
    Convenience function for Metropolis-Hastings sampling with
    Langevin dynamics proposal kernel.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> np.random.seed(1)
    >>>
    >>> def logpdf(x):
    ...     return x.T @ x / 2.0
    ...
    >>> def logder(x):
    ...     return x
    ...
    >>> xval = np.linspace(-4, 4, 200)
    >>> yval = np.exp(-xval**2 / 2.0)/np.sqrt(2*np.pi)
    >>> __ = plt.plot(xval, yval)
    >>>
    >>> a, b, c = mala(logpdf, logder, 2500, np.array([.5]), pwidth=1.5)
    >>> __ = plt.hist(a[:, 0], bins=50, density=True, alpha=0.5)
    >>> __ = plt.title("MALA Samples")
    >>> plt.show()
    """
    logdens = objective.Objective(logpdf, loggrad)
    langmh = langevin.MALA(logdens)
    return langmh.sample_nd(nsamps, initstate, pwidth)


def pmala(logpdf, loggrad, loghess, nsamps, initstate, pwidth):
    """
    Convenience function for Metropolis-Hastings sampling with
    Riemannian (preconditioned) Langevin dynamics proposal kernel.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> np.random.seed(1)
    >>>
    >>> def logpdf(x):
    ...     return x.T @ x
    ...
    >>> def logder(x):
    ...     return 2*x
    ...
    >>> def loghess(x):
    ...     return 2 * np.ones((len(x), len(x)))
    ...
    >>> xval = np.linspace(-4, 4, 200)
    >>> yval = np.exp(-xval**2)/np.sqrt(2*np.pi*0.5)
    >>> __ = plt.plot(xval, yval)
    >>>
    >>> a, b, c = pmala(logpdf, logder, loghess, 2500, np.array([.5]), pwidth=1.5)
    >>> __ = plt.hist(a[:, 0], bins=50, density=True, alpha=0.5)
    >>> __ = plt.title("PMALA Samples")
    >>> plt.show()
    """
    logdens = objective.Objective(logpdf, loggrad, loghess)
    plangmh = langevin.PMALA(logdens)
    return plangmh.sample_nd(nsamps, initstate, pwidth)


def hmc(logpdf, loggrad, nsamps, initstate, stepsize, nsteps):
    """
    Convenience function for Hamiltonian MCMC.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> np.random.seed(1)
    >>>
    >>> def logpdf(x):
    ...     return x.T @ x / 2.0
    ...
    >>> def logder(x):
    ...     return x
    ...
    >>> xval = np.linspace(-4, 4, 200)
    >>> yval = np.exp(-xval**2 / 2.0)/np.sqrt(2*np.pi)
    >>> __ = plt.plot(xval, yval)
    >>>
    >>> a, b, c = hmc(logpdf, logder, 2500, np.array([.5]), stepsize=1.75, nsteps=5)
    >>> __ = plt.hist(a[:, 0], bins=50, density=True, alpha=0.5)
    >>> __ = plt.title("HMC Samples")
    >>> plt.show()
    """
    logdens = objective.Objective(logpdf, loggrad)
    hmc = hamiltonian.HMC(logdens, nsteps)
    return hmc.sample_nd(nsamps, initstate, stepsize)


def phmc(logpdf, logder, loghess, nsamps, initstate, stepsize, nsteps):
    """
    Convenience function for preconditioned Hamiltonian MCMC.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> np.random.seed(1)
    >>>
    >>> def logpdf(x):
    ...     return x.T @ x
    ...
    >>> def logder(x):
    ...     return 2*x
    ...
    >>> def loghess(x):
    ...     return 2 * np.ones((len(x), len(x)))
    ...
    >>> xval = np.linspace(-4, 4, 200)
    >>> yval = np.exp(-xval**2)/np.sqrt(2*np.pi*0.5)
    >>> __ = plt.plot(xval, yval)
    >>>
    >>> a, b, c = phmc(logpdf, logder, loghess, 2500, np.array([.5]), stepsize=1.75, nsteps=5)
    >>> __ = plt.hist(a[:, 0], bins=50, density=True, alpha=0.5)
    >>> __ = plt.title("PHMC Samples")
    >>> plt.show()
    """
    logdens = objective.Objective(logpdf, logder, loghess)
    phmc = hamiltonian.PHMC(logdens, nsteps)
    return phmc.sample_nd(nsamps, initstate, stepsize)
