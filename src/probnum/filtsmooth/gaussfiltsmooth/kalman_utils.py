"""Prediction, update, and smoothing implementations for the Kalman filter/smoother."""

import typing

import numpy as np

import probnum.random_variables as pnrv

# Prediction choices


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
        rv,
        start,
        stop,
        step=_intermediate_step,
        _linearise_at=None,
        _diffusion=_diffusion,
    )


def predict_sqrt(
    dynamics_model,
    start,
    stop,
    rv,
    _intermediate_step=None,
    _linearise_at=None,
    _diffusion=1.0,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute prediction in square-root form under the assumption that the system is
    linear."""
    raise NotImplementedError("TBD.")


# Measure choices


def measure_via_transition(
    measurement_model,
    rv,
    time,
    _linearise_at=None,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute the measurement under the assumption that the transition is available in
    closed form."""
    return measurement_model.transition_rv(rv, start=time, _linearise_at=_linearise_at)


def measure_sqrt(
    measurement_model,
    rv,
    time,
    _linearise_at=None,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute measurement in square-root form under the assumption that the system is
    linear."""
    raise NotImplementedError("TBD.")


# Update choices


def update_classic(
    measurement_model, rv, time, data, _linearise_at=None
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute a classic Kalman update.

    This is a combination of a call to measure_*() and a Gaussian
    update.
    """
    meas_rv, info = measure_via_transition(
        measurement_model, rv, time, _linearise_at=_linearise_at
    )
    crosscov = info["crosscov"]
    new_mean = rv.mean + crosscov @ np.linalg.solve(meas_rv.cov, data - meas_rv.mean)
    new_cov = rv.cov - crosscov @ np.linalg.solve(meas_rv.cov, crosscov.T)
    filt_rv = pnrv.Normal(new_mean, new_cov)
    return filt_rv


def update_joseph(
    measurement_model, rv, time, data
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute a classic Kalman update in Joseph form."""
    raise NotImplementedError("TBD.")


def update_sqrt(
    measurement_model, rv, time, data
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute a classic Kalman update in square-root form."""
    raise NotImplementedError("TBD.")


# Maybe this can be done more cleanly with a decorator.
# For the time being I think this is sufficiently clean though.
def iterated_update(update_fun, atol, rtol, measurement_model, rv, time, data):
    """Turn an update_*() function into an iterated update.

    This iteration is continued until it reaches a fixed-point (as measured with atol and rtol).
    Using this inside `Kalman` yields the iterated (extended/unscented/...) Kalman filter.

    Examples
    --------
    >>> import functools as ft
    >>> # All the below can be used inside a "Kalman" object.
    >>> iterated_update_classic = ft.partial(iterated_update, update_fun=update_classic)
    >>> iterated_update_joseph = ft.partial(iterated_update, update_fun=update_joseph)
    >>> iterated_update_sqrt = ft.partial(iterated_update, update_fun=update_sqrt)
    """
    pass


# Smoothing choices

# Maybe this can be done more cleanly with a decorator.
# For the time being I think this is sufficiently clean though.
def rts_smooth_step_with_precon(
    smooth_step_fun,
    dynamics_model,
    unsmoothed_rv,
    predicted_rv,
    smoothed_rv,
    smoothing_gain,
    start,
    stop,
    _diffusion=1.0,
):
    """Execute a smoothing step with preconditioning.

    Examples
    --------
    >>> import functools as ft
    >>> rts_smooth_step_classic_with_precon = ft.partial(rts_smooth_step_with_precon, smooth_step_fun=rts_smooth_step_classic)
    >>> rts_smooth_step_joseph_with_precon = ft.partial(rts_smooth_step_with_precon, smooth_step_fun=rts_smooth_step_joseph)
    >>> rts_smooth_step_sqrt_with_precon = ft.partial(rts_smooth_step_with_precon, smooth_step_fun=rts_smooth_step_sqrt)
    """
    pass


def rts_smooth_step_classic_with_precon(
    dynamics_model,
    unsmoothed_rv,
    predicted_rv,
    smoothed_rv,
    smoothing_gain,
    start,
    stop,
    _diffusion=1.0,
) -> (pnrv.RandomVariable, typing.Dict):
    precon_inv = dynamics_model.precon.inverse(stop - start)
    unsmoothed_rv = precon_inv @ unsmoothed_rv
    smoothed_rv = precon_inv @ smoothed_rv

    # call other function here
    pass


def rts_smooth_step_classic(
    dynamics_model,
    unsmoothed_rv,
    predicted_rv,
    smoothed_rv,
    smoothing_gain,
    start,
    stop,
    _diffusion=1.0,
) -> (pnrv.RandomVariable, typing.Dict):
    new_mean = unsmoothed_rv.mean + smoothing_gain @ (
        smoothed_rv.mean - predicted_rv.mean
    )
    new_cov = (
        unsmoothed_rv.cov
        + smoothing_gain @ (smoothed_rv.cov - predicted_rv.cov) @ smoothing_gain.T
    )
    return pnrv.Normal(new_mean, new_cov)


def rts_smooth_step_joseph(
    dynamics_model,
    unsmoothed_rv,
    predicted_rv,
    smoothed_rv,
    smoothing_gain,
    start,
    stop,
    _diffusion=1.0,
) -> (pnrv.RandomVariable, typing.Dict):
    raise NotImplementedError("TBD.")


def rts_smooth_step_sqrt(
    dynamics_model,
    unsmoothed_rv,
    predicted_rv,
    smoothed_rv,
    smoothing_gain,
    start,
    stop,
    _diffusion=1.0,
) -> (pnrv.RandomVariable, typing.Dict):
    raise NotImplementedError("TBD.")
