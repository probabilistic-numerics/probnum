"""Prediction, update, and smoothing implementations for the Kalman filter/smoother."""

import functools as ft
import typing

import numpy as np

import probnum.random_variables as pnrv

########################################################################################################################
# Prediction choices
########################################################################################################################


def predict_via_transition(
    dynamics_model,
    start,
    stop,
    rv,
    _intermediate_step=None,
    _linearise_at=None,
    _diffusion=1.0,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute the prediction under the assumption that the transition is available in
    closed form."""
    return dynamics_model.transition_rv(
        rv=rv,
        start=start,
        stop=stop,
        step=_intermediate_step,
        _linearise_at=_linearise_at,
        _diffusion=_diffusion,
    )


########################################################################################################################
# Measure choices
########################################################################################################################


def measure_via_transition(
    measurement_model,
    rv,
    time,
    _linearise_at=None,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute the measurement under the assumption that the transition is available in
    closed form."""
    return measurement_model.transition_rv(
        rv=rv, start=time, _linearise_at=_linearise_at
    )


########################################################################################################################
# Update choices
########################################################################################################################


def update_classic(
    measurement_model, rv, time, data, _linearise_at=None
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute a classic Kalman update.

    This is a combination of a call to measure_*() and a Gaussian
    update.
    """
    meas_rv, info = measure_via_transition(
        measurement_model=measurement_model,
        rv=rv,
        time=time,
        _linearise_at=_linearise_at,
    )
    crosscov = info["crosscov"]
    updated_rv = condition_state_on_measurement(
        pred_rv=rv, meas_rv=meas_rv, crosscov=crosscov, data=data
    )
    return updated_rv, meas_rv, {}


def condition_state_on_measurement(pred_rv, meas_rv, crosscov, data):
    new_mean = pred_rv.mean + crosscov @ np.linalg.solve(
        meas_rv.cov, data - meas_rv.mean
    )
    new_cov = pred_rv.cov - crosscov @ np.linalg.solve(meas_rv.cov, crosscov.T)
    updated_rv = pnrv.Normal(new_mean, new_cov)
    return updated_rv


# Basically a decorator (but not quite?).
def iterate_update(update_fun, stopcrit):
    """"""

    def new_update_fun(*args, **kwargs):
        return _iterated_update(update_fun, stopcrit, *args, **kwargs)

    return new_update_fun


def _iterated_update(
    update_fun, stopcrit, measurement_model, rv, time, data, _linearise_at=None
):
    """Turn an update_*() function into an iterated update.

    This iteration is continued until it reaches a fixed-point (as measured with atol and rtol).
    Using this inside `Kalman` yields the iterated (extended/unscented/...) Kalman filter.

    Examples
    --------
    >>> import functools as ft
    >>>
    >>> stopcrit = StoppingCriterion(atol=1e-2, rtol=1e-4, maxit=1000)
    >>> iterated_update_classic = ft.partial(iterated_update, update_fun=update_classic, stopcrit=stopcrit)
    """
    current_rv, meas_rv, info = update_fun(
        measurement_model=measurement_model,
        rv=rv,
        time=time,
        data=data,
        _linearise_at=_linearise_at,
    )
    new_mean = current_rv.mean
    old_mean = np.inf * np.ones(current_rv.mean.shape)
    while not stopcrit.terminate(error=new_mean - old_mean, reference=new_mean):
        old_mean = new_mean
        current_rv, meas_rv, info = update_fun(
            measurement_model=measurement_model,
            rv=current_rv,
            time=time,
            data=data,
        )
        new_mean = current_rv.mean
    return current_rv, meas_rv, info


########################################################################################################################
# Smoothing choices
########################################################################################################################

# Maybe this can be done more cleanly with a decorator.
# For the time being I think this is sufficiently clean though.
def rts_smooth_step_with_precon(
    smooth_step_fun,
    precon,
    precon_inv,
    unsmoothed_rv,
    predicted_rv,
    smoothed_rv,
    smoothing_gain,
    dynamics_model=None,
    start=None,
    stop=None,
):
    """Execute a smoothing step with preconditioning.

    Examples
    --------
    >>> import functools as ft
    >>> rts_smooth_step_classic_with_precon = ft.partial(rts_smooth_step_with_precon, smooth_step_fun=rts_smooth_step_classic)
    >>> rts_smooth_step_joseph_with_precon = ft.partial(rts_smooth_step_with_precon, smooth_step_fun=rts_smooth_step_joseph)
    >>> rts_smooth_step_sqrt_with_precon = ft.partial(rts_smooth_step_with_precon, smooth_step_fun=rts_smooth_step_sqrt)
    """
    # Assemble preconditioners
    dt = stop - start
    precon = dynamics_model.precon(dt)
    precon_inv = dynamics_model.precon.inverse(dt)

    # Pull RVs/matrices into preconditioned space
    unsmoothed_rv = precon_inv @ unsmoothed_rv
    predicted_rv = precon_inv @ predicted_rv
    smoothed_rv = precon_inv @ smoothed_rv
    smoothing_gain = np.nan
    print("What needs to happen to the smoothing gain???")
    assert True is False

    # Undo preconditioning
    updated_rv = smooth_step_fun(
        unsmoothed_rv,
        predicted_rv,
        smoothed_rv,
        smoothing_gain,
        dynamics_model=None,
        start=None,
        stop=None,
    )
    new_rv = precon @ updated_rv
    return new_rv, {}


def rts_smooth_step_classic(
    unsmoothed_rv,
    predicted_rv,
    smoothed_rv,
    smoothing_gain,
    dynamics_model=None,
    start=None,
    stop=None,
) -> (pnrv.RandomVariable, typing.Dict):
    new_mean = unsmoothed_rv.mean + smoothing_gain @ (
        smoothed_rv.mean - predicted_rv.mean
    )
    new_cov = (
        unsmoothed_rv.cov
        + smoothing_gain @ (smoothed_rv.cov - predicted_rv.cov) @ smoothing_gain.T
    )
    return pnrv.Normal(new_mean, new_cov), {}
