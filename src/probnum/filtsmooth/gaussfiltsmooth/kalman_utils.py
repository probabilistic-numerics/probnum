"""Prediction, update, and smoothing implementations for the Kalman filter/smoother."""

import typing

import numpy as np

import probnum.random_variables as pnrv
from probnum.filtsmooth import statespace

from .extendedkalman import DiscreteEKFComponent
from .stoppingcriterion import StoppingCriterion

########################################################################################################################
# Prediction choices
########################################################################################################################


def predict_via_transition(
    dynamics_model,
    rv,
    start,
    stop,
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


def predict_sqrt(
    dynamics_model,
    rv,
    start,
    stop,
    _intermediate_step=None,
    _linearise_at=None,
    _diffusion=1.0,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute the prediction in square-root form."""
    H, SR, shift = linear_system_matrices(
        dynamics_model, rv, start, stop, _linearise_at
    )

    new_mean = H @ rv.mean + shift
    new_cov_cholesky = cholesky_update(H @ rv.cov_cholesky, np.sqrt(_diffusion) * SR)
    new_cov = new_cov_cholesky @ new_cov_cholesky.T
    crosscov = rv.cov @ H.T
    return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky), {
        "crosscov": crosscov
    }


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


def measure_sqrt(
    measurement_model,
    rv,
    time,
    _linearise_at=None,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute the prediction in square-root form."""
    H, SR, shift = linear_system_matrices(
        measurement_model, rv, time, None, _linearise_at
    )

    new_mean = H @ rv.mean + shift
    new_cov_cholesky = cholesky_update(H @ rv.cov_cholesky, SR)
    new_cov = new_cov_cholesky @ new_cov_cholesky.T
    crosscov = rv.cov @ H.T
    return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky), {
        "crosscov": crosscov
    }


########################################################################################################################
# Update choices
########################################################################################################################


def update_classic(
    measurement_model, data, rv, time, _linearise_at=None
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
    return updated_rv, {"meas_rv": meas_rv}


def condition_state_on_measurement(pred_rv, meas_rv, crosscov, data):
    """Condition a Gaussian random variable on an observation."""
    new_mean = pred_rv.mean + crosscov @ np.linalg.solve(
        meas_rv.cov, data - meas_rv.mean
    )
    new_cov = pred_rv.cov - crosscov @ np.linalg.solve(meas_rv.cov, crosscov.T)
    updated_rv = pnrv.Normal(new_mean, new_cov)
    return updated_rv


def update_sqrt(
    measurement_model, data, rv, time, _linearise_at=None
) -> (pnrv.RandomVariable, typing.Dict):
    r"""Compute the Kalman update in square-root form.

    Assumes a measurement model of the form

        .. math::  x \mapsto N(H x, R)

    and acts only on the square-root of the predicted covariance.

    See Eq. 48 in
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.1059&rep=rep1&type=pdf.
    """
    H, SR, shift = linear_system_matrices(
        measurement_model, rv, time, None, _linearise_at
    )
    SC = rv.cov_cholesky

    zeros = np.zeros(H.T.shape)
    blockmat = np.block([[SR, H @ SC], [zeros, SC]]).T
    big_triu = np.linalg.qr(blockmat, mode="r")
    ndim_measurements = len(H)

    measured_triu = big_triu[:ndim_measurements, :ndim_measurements]
    measured_cholesky = triu_to_positive_tril(measured_triu)

    postcov_triu = big_triu[ndim_measurements:, ndim_measurements:]
    postcov_cholesky = triu_to_positive_tril(postcov_triu)
    kalman_gain = big_triu[:ndim_measurements, ndim_measurements:].T @ np.linalg.inv(
        measured_triu.T
    )

    meas_mean = H @ rv.mean + shift
    meas_cov = measured_cholesky @ measured_cholesky
    meas_rv = pnrv.Normal(meas_mean, cov=meas_cov, cov_cholesky=measured_cholesky)

    new_mean = rv.mean + kalman_gain @ (data - meas_mean)
    new_cov = postcov_cholesky @ postcov_cholesky.T
    new_rv = pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=postcov_cholesky)
    return new_rv, {"meas_rv": meas_rv}


def iterate_update(update_fun, stopcrit=None):
    """Iterated update decorator.

    Examples
    --------
    >>> import functools as ft
    >>>
    >>> # default stopping
    >>> iter_upd_default = iterate_update(update_classic)
    >>>
    >>> # Custom stopping
    >>> stopcrit = StoppingCriterion(atol=1e-12, rtol=1e-14, maxit=1000)
    >>> iter_upd_custom = iterate_update(update_fun=update_classic, stopcrit=stopcrit)
    """
    if stopcrit is None:
        stopcrit = StoppingCriterion()

    def new_update_fun(*args, **kwargs):
        return _iterated_update(update_fun, stopcrit, *args, **kwargs)

    return new_update_fun


def _iterated_update(
    update_fun, stopcrit, measurement_model, data, rv, time, _linearise_at=None
):
    """Turn an update_*() function into an iterated update.

    This iteration is continued until it reaches a fixed-point (as
    measured with atol and rtol). Using this inside `Kalman` yields the
    iterated (extended/unscented/...) Kalman filter.
    """
    current_rv, info = update_fun(
        measurement_model=measurement_model,
        data=data,
        rv=rv,
        time=time,
        _linearise_at=_linearise_at,
    )
    new_mean = current_rv.mean
    old_mean = np.inf * np.ones(current_rv.mean.shape)
    while not stopcrit.terminate(error=new_mean - old_mean, reference=new_mean):
        old_mean = new_mean
        current_rv, info = update_fun(
            measurement_model=measurement_model,
            data=data,
            rv=current_rv,
            time=time,
        )
        new_mean = current_rv.mean
    return current_rv, info


########################################################################################################################
# Smoothing choices
########################################################################################################################


def rts_add_precon(smooth_step_fun):
    """Make a RTS smoothing step respect preconditioning.

    This is only available for Integrators.

    Examples
    --------
    >>> step_with_precon = rts_add_precon(rts_smooth_step_classic)
    """

    def new_smoothing_function(*args, **kwargs):
        return _rts_smooth_step_with_precon(smooth_step_fun, *args, **kwargs)

    return new_smoothing_function


def _rts_smooth_step_with_precon(
    smooth_step_fun,
    unsmoothed_rv,
    predicted_rv,
    smoothed_rv,
    crosscov,
    dynamics_model=None,
    start=None,
    stop=None,
):
    """Execute a smoothing step with preconditioning."""

    # Assemble preconditioners
    dt = stop - start
    precon = dynamics_model.precon(dt)
    precon_inv = dynamics_model.precon.inverse(dt)

    # Pull RVs/matrices into preconditioned space
    unsmoothed_rv = precon_inv @ unsmoothed_rv
    predicted_rv = precon_inv @ predicted_rv
    smoothed_rv = precon_inv @ smoothed_rv
    crosscov = precon_inv @ crosscov @ precon_inv.T

    # Carry out the smoothing step
    updated_rv, _ = smooth_step_fun(
        unsmoothed_rv,
        predicted_rv,
        smoothed_rv,
        crosscov,
        dynamics_model=dynamics_model,
        start=start,
        stop=stop,
    )
    new_rv = precon @ updated_rv
    return new_rv, {}


def rts_smooth_step_classic(
    unsmoothed_rv,
    predicted_rv,
    smoothed_rv,
    crosscov,
    dynamics_model=None,
    start=None,
    stop=None,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute a classical Rauch-Tung-Striebel smoothing step."""
    smoothing_gain = crosscov @ np.linalg.inv(predicted_rv.cov)
    new_mean = unsmoothed_rv.mean + smoothing_gain @ (
        smoothed_rv.mean - predicted_rv.mean
    )
    new_cov = (
        unsmoothed_rv.cov
        + smoothing_gain @ (smoothed_rv.cov - predicted_rv.cov) @ smoothing_gain.T
    )
    return pnrv.Normal(new_mean, new_cov), {}


def rts_smooth_step_sqrt(
    unsmoothed_rv,
    predicted_rv,
    smoothed_rv,
    crosscov,
    dynamics_model=None,
    start=None,
    stop=None,
) -> (pnrv.RandomVariable, typing.Dict):
    r"""Smoothing step in square-root form.

    Assumes a prior dynamic model of the form

        .. math:: x \\mapsto N(A x, Q).

    For the mathematical justification of this step, see Eq. 45 in
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.1059&rep=rep1&type=pdf.
    """
    A, SQ, shift = linear_system_matrices(dynamics_model, None, start, stop, None)
    SC_past = unsmoothed_rv.cov_cholesky
    SC_futu = smoothed_rv.cov_cholesky
    G = crosscov @ np.linalg.inv(predicted_rv.cov)

    dim = len(A)
    zeros = np.zeros((dim, dim))
    blockmat = np.block(
        [
            [SC_past.T @ A.T, SC_past.T],
            [SQ.T, zeros],
            [zeros, SC_futu.T @ G.T],
        ]
    )
    big_triu = np.linalg.qr(blockmat, mode="r")
    SC = big_triu[dim : 2 * dim, dim:]

    new_cov_cholesky = triu_to_positive_tril(SC)
    new_cov = new_cov_cholesky @ new_cov_cholesky.T
    new_mean = unsmoothed_rv.mean + G @ (smoothed_rv.mean - predicted_rv.mean)
    return pnrv.Normal(new_mean, new_cov, cov_cholesky=new_cov_cholesky), {}


########################################################################################################################
# Helper functions
########################################################################################################################


def linear_system_matrices(model, rv, start, stop, _linearise_at):
    """Extract the linear system matrices from Transition objects in order to apply
    square-root steps."""
    if isinstance(model, statespace.LTISDE):
        model = model.discretise(stop - start)
    elif isinstance(model, DiscreteEKFComponent):
        compute_jacobian_at = _linearise_at if _linearise_at is not None else rv
        model.linearize(compute_jacobian_at)
        model = model.linearized_model
    H = model.state_trans_mat_fun(start)
    SR = model.proc_noise_cov_cholesky_fun(start)
    shift = model.shift_vec_fun(start)
    return H, SR, shift


# used for predict() and measure(), but more general than that,
# so it has a more general name than the functions below.
def cholesky_update(
    S1: np.ndarray, S2: typing.Optional[np.ndarray] = None
) -> np.ndarray:
    r"""Compute Cholesky update/factorization :math:`C C^\top = S_1 S_1^\top + S_2 S_2^\top`.

    This can be used in various ways.
    For example, :math:`S_1` and :math:`S_2` do not need to be Cholesky factors; any matrix square-root is sufficient.
    As long as :math:`C C^\top = S_1 S_1^\top + S_2 S_2^\top` is well-defined (and admits a Cholesky-decomposition),
    :math:`S_1` and :math:`S_2` do not even have to be square.
    """
    # doc might need a bit more explanation in the future
    # perhaps some doctest or so?
    if S2 is not None:
        stacked_up = np.vstack((S1.T, S2.T))
    else:
        stacked_up = np.vstack(S1.T)
    upper_sqrtm = np.linalg.qr(stacked_up, mode="r")
    return triu_to_positive_tril(upper_sqrtm)


def triu_to_positive_tril(triu_mat: np.ndarray) -> np.ndarray:
    r"""Change an upper triangular matrix into a valid lower Cholesky factor.

    Transpose, and change the sign of the diagonals to '+' if necessary.
    """
    tril_mat = triu_mat.T
    with_pos_diag = tril_mat @ np.diag(np.sign(np.diag(tril_mat)))
    return with_pos_diag
