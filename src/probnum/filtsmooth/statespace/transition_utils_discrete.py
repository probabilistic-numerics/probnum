"""Discrete-time transition implementations.

All sorts of implementations for implementing discrete-time updates,
including predictions, updates, smoothings, and more.
"""


########################################################################################################################
# Forward implementations (think: predictions)
# All sorts of ways of computing m = A m + z; C = A C At + Q
########################################################################################################################


def forward_rv_classic(
    discrete_transition,
    rv,
    time,
    with_gain=False,
    _linearise_at=None,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute the forward propagation in square-root form."""
    H = discrete_transition.state_trans_mat_fun(time)
    R = discrete_transition.proc_noise_cov_mat_fun(time)
    shift = discrete_transition.shift_vec_fun(time)

    new_mean = H @ rv.mean + shift
    crosscov = rv.cov @ H.T
    new_cov = H @ crosscov + R
    info = {"crosscov": crosscov}
    if with_gain:
        gain = crosscov @ np.linalg.inv(new_cov)
        info["gain"] = gain
    return pnrv.Normal(new_mean, cov=new_cov), info


def forward_rv_sqrt(
    discrete_transition,
    rv,
    time,
    with_gain=False,
    _linearise_at=None,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute the forward propagation in square-root form."""
    H = discrete_transition.state_trans_mat_fun(time)
    SR = discrete_transition.proc_noise_cov_cholesky_fun(time)
    shift = discrete_transition.shift_vec_fun(time)

    new_mean = H @ rv.mean + shift
    new_cov_cholesky = cholesky_update(H @ rv.cov_cholesky, SR)
    new_cov = new_cov_cholesky @ new_cov_cholesky.T
    crosscov = rv.cov @ H.T
    info = {"crosscov": crosscov}
    if with_gain:
        gain = crosscov @ np.linalg.inv(new_cov)
        info["gain"] = gain
    return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky), info


########################################################################################################################
# Backward implementations (think: updates and smoothing)
########################################################################################################################


def backward_realization(realization, rv_future, rv_past, gain):
    new_mean = rv_past.mean + gain @ (realization - rv_future.mean)
    new_cov = rv_past.cov - gain @ rv_future @ gain.T
    updated_rv = pnrv.Normal(new_mean, new_cov)
    return updated_rv


def backward_rv(rv_attained, rv_future, rv_past, crosscov_past_future):
    # Plain smoothing update
    smoothing_gain = crosscov_past_future @ np.linalg.inv(rv_future.cov)
    new_mean = rv_past.mean + smoothing_gain @ (rv_attained.mean - rv_future.mean)
    new_cov = (
        rv_past.cov
        + smoothing_gain @ (rv_attained.cov - rv_future.cov) @ smoothing_gain.T
    )
    return pnrv.Normal(new_mean, new_cov), {}


########################################################################################################################
# Forward-backward combos (think: full smoothing updates)
########################################################################################################################


def forward_rv_and_backward_realization_sqrt(
    discrete_transition, realization, rv, time
):

    # Square-root Kalman update: measure and condition in one.
    # Computes its own gain.

    # Read off matrices
    H = discrete_transition.state_trans_mat_fun(time)
    SR = discrete_transition.proc_noise_cov_cholesky_fun(time)
    shift = discrete_transition.shift_vec_fun(time)
    SC = rv.cov_cholesky

    # QR-decomposition
    zeros = np.zeros(H.T.shape)
    blockmat = np.block([[SR, H @ SC], [zeros, SC]]).T
    big_triu = np.linalg.qr(blockmat, mode="r")

    # Extract relevant info
    ndim_measurements = len(H)
    measured_triu = big_triu[:ndim_measurements, :ndim_measurements]
    measured_cholesky = triu_to_positive_tril(measured_triu)
    postcov_triu = big_triu[ndim_measurements:, ndim_measurements:]
    postcov_cholesky = triu_to_positive_tril(postcov_triu)
    gain = big_triu[:ndim_measurements, ndim_measurements:].T @ np.linalg.inv(
        measured_triu.T
    )

    # Compute "measured" RV
    meas_mean = H @ rv.mean + shift
    meas_cov = measured_cholesky @ measured_cholesky
    meas_rv = pnrv.Normal(meas_mean, cov=meas_cov, cov_cholesky=measured_cholesky)
    info = {"meas_rv": meas_rv}

    # Update RV
    new_mean = rv.mean + gain @ (realization - meas_mean)
    new_cov = postcov_cholesky @ postcov_cholesky.T
    new_rv = pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=postcov_cholesky)
    return new_rv, info


def forward_rv_and_backward_rv_sqrt(
    discrete_transition, rv_attained, rv_past, time, gain
):
    # Like *_and_backward_realization, but since we need to incorporate
    # the rv_attained.cov, the gain must be provided, too.
    # Does its own forward_rv in principle (except for the gain-info just discussed).

    H = discrete_transition.state_trans_mat_fun(time)
    SR = discrete_transition.proc_noise_cov_cholesky_fun(time)
    shift = discrete_transition.shift_vec_fun(time)

    SC_past = rv_past.cov_cholesky
    SC_attained = rv_attained.cov_cholesky

    dim = len(A)
    zeros = np.zeros((dim, dim))
    blockmat = np.block(
        [
            [SC_past.T @ H.T, SC_past.T],
            [SR.T, zeros],
            [zeros, SC_attained.T @ gain.T],
        ]
    )
    big_triu = np.linalg.qr(blockmat, mode="r")
    SC = big_triu[dim : 2 * dim, dim:]

    #  Perhaps extract meas_rv here, and provide it in info?

    new_mean = unsmoothed_rv.mean + gain @ (
        smoothed_rv.mean - H @ unsmoothed_rv.mean - shift
    )
    new_cov_cholesky = triu_to_positive_tril(SC)
    new_cov = new_cov_cholesky @ new_cov_cholesky.T
    return pnrv.Normal(new_mean, new_cov, cov_cholesky=new_cov_cholesky), {}


########################################################################################################################
# Helper functions
########################################################################################################################


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
