"""ODE solutions returned by Gaussian ODE filtering."""

from typing import Optional

import numpy as np

from probnum import _randomvariablelist, filtsmooth, randvars, utils
from probnum.diffeq import _odesolution
from probnum.filtsmooth._timeseriesposterior import DenseOutputLocationArgType
from probnum.typing import FloatArgType, IntArgType, ShapeArgType


class KalmanODESolution(_odesolution.ODESolution):
    """Probabilistic ODE solution corresponding to the :class:`GaussianIVPFilter`.

    Recall that in ProbNum, Gaussian filtering and smoothing is generally named "Kalman".

    Parameters
    ----------
    kalman_posterior
        Gauss-Markov posterior over the ODE solver state space model.
        Therefore, it assumes that the dynamics model is an :class:`Integrator`.

    See Also
    --------
    GaussianIVPFilter : ODE solver that behaves like a Gaussian filter.
    KalmanPosterior : Posterior over states after Gaussian filtering/smoothing.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.diffeq import probsolve_ivp
    >>> from probnum import randvars
    >>>
    >>> def f(t, x):
    ...     return 4*x*(1-x)
    >>>
    >>> y0 = np.array([0.15])
    >>> t0, tmax = 0., 1.5
    >>> solution = probsolve_ivp(f, t0, tmax, y0, step=0.1, adaptive=False)
    >>> # Mean of the discrete-time solution
    >>> print(np.round(solution.states.mean, 2))
    [[0.15]
     [0.21]
     [0.28]
     [0.37]
     [0.47]
     [0.57]
     [0.66]
     [0.74]
     [0.81]
     [0.87]
     [0.91]
     [0.94]
     [0.96]
     [0.97]
     [0.98]
     [0.99]]

    >>> # Times of the discrete-time solution
    >>> print(solution.locations)
    [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.  1.1 1.2 1.3 1.4 1.5]
    >>> # Individual entries of the discrete-time solution can be accessed with
    >>> print(solution[5])
    <Normal with shape=(1,), dtype=float64>
    >>> print(np.round(solution[5].mean, 2))
    [0.56]
    >>> # Evaluate the continuous-time solution at a new time point t=0.65
    >>> print(np.round(solution(0.65).mean, 2))
    [0.70]
    """

    def __init__(self, kalman_posterior: filtsmooth.gaussian.KalmanPosterior):
        self.kalman_posterior = kalman_posterior

        # Pre-compute projection matrices.
        # The prior must be an integrator, if not, an error is thrown in 'GaussianIVPFilter'.
        self.proj_to_y = self.kalman_posterior.transition.proj2coord(coord=0)
        self.proj_to_dy = self.kalman_posterior.transition.proj2coord(coord=1)

        states = _randomvariablelist._RandomVariableList(
            [_project_rv(self.proj_to_y, rv) for rv in self.kalman_posterior.states]
        )
        derivatives = _randomvariablelist._RandomVariableList(
            [_project_rv(self.proj_to_dy, rv) for rv in self.kalman_posterior.states]
        )
        super().__init__(
            locations=kalman_posterior.locations, states=states, derivatives=derivatives
        )

    def interpolate(
        self,
        t: FloatArgType,
        previous_index: Optional[IntArgType] = None,
        next_index: Optional[IntArgType] = None,
    ) -> randvars.RandomVariable:
        out_rv = self.kalman_posterior.interpolate(
            t, previous_index=previous_index, next_index=next_index
        )
        return _project_rv(self.proj_to_y, out_rv)

    def sample(
        self,
        rng: np.random.Generator,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
    ) -> np.ndarray:

        samples = self.kalman_posterior.sample(rng=rng, t=t, size=size)
        # Project the samples down to the "true" KalmanODESolution dimensions
        # (which are a subset of the KalmanPosterior dimensions)
        ode_samples = np.einsum("dq,...q->...d", self.proj_to_y, samples)

        return ode_samples

    def transform_base_measure_realizations(
        self,
        base_measure_realizations: np.ndarray,
        t: DenseOutputLocationArgType = None,
    ) -> np.ndarray:
        errormsg = (
            "The KalmanODESolution does not implement transformation of realizations of a base measure."
            "Try `KalmanODESolution.kalman_posterior.transform_base_measure_realizations` instead."
        )

        raise NotImplementedError(errormsg)

    @property
    def filtering_solution(self):

        if isinstance(self.kalman_posterior, filtsmooth.gaussian.FilteringPosterior):
            return self

        # else: self.kalman_posterior is a SmoothingPosterior object, which has the field filter_posterior.
        return KalmanODESolution(
            kalman_posterior=self.kalman_posterior.filtering_posterior
        )


def _project_rv(projmat, rv):
    # There is no way of checking whether `rv` has its Cholesky factor computed already or not.
    # Therefore, since we need to update the Cholesky factor for square-root filtering,
    # we also update the Cholesky factor for non-square-root algorithms here,
    # which implies additional cost.
    # See Issues #319 and #329.
    # When they are resolved, this function here will hopefully be superfluous.

    new_mean = projmat @ rv.mean
    new_cov = projmat @ rv.cov @ projmat.T
    new_cov_cholesky = utils.linalg.cholesky_update(projmat @ rv.cov_cholesky)
    return randvars.Normal(new_mean, new_cov, cov_cholesky=new_cov_cholesky)
