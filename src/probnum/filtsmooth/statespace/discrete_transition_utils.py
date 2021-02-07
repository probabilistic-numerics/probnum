"""Discrete-time transition implementations (for linear transitions).

All sorts of implementations for implementing discrete-time updates,
including predictions, updates, smoothing steps, and more.

Implementations happen on a random variable level. That means that
only `forward_rv_*` and `backward_rv_*` implementations are provided.
Their respective realisation implementations are obtained by turning
the realisation into a Normal RV with zero covariance.
"""

import typing

import numpy as np

import probnum.random_variables as pnrv

########################################################################################################################
#
# Forward implementations (think: predictions)
#
# All sorts of ways of computing m = A m + z; C = A C At + Q
# The signature of a forward method is
#
#   forward_rv_*(discrete_transition, rv, t, compute_gain=False, _diffusion=1.0) -> (RV, dict)
#
########################################################################################################################


def forward_rv_classic(
    discrete_transition,
    rv,
    t,
    compute_gain=False,
    _diffusion=1.0,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute the forward propagation in square-root form."""
    H = discrete_transition.state_trans_mat_fun(t)
    R = discrete_transition.proc_noise_cov_mat_fun(t)
    shift = discrete_transition.shift_vec_fun(t)

    new_mean = H @ rv.mean + shift
    crosscov = rv.cov @ H.T
    new_cov = H @ crosscov + _diffusion * R
    info = {"crosscov": crosscov}
    if compute_gain:
        gain = crosscov @ np.linalg.inv(new_cov)
        info["gain"] = gain
    return pnrv.Normal(new_mean, cov=new_cov), info


def forward_rv_sqrt(
    discrete_transition,
    rv,
    t,
    compute_gain=False,
    _diffusion=1.0,
) -> (pnrv.RandomVariable, typing.Dict):
    """Compute the forward propagation in square-root form."""
    H = discrete_transition.state_trans_mat_fun(t)
    SR = discrete_transition.proc_noise_cov_cholesky_fun(t)
    shift = discrete_transition.shift_vec_fun(t)

    new_mean = H @ rv.mean + shift
    new_cov_cholesky = cholesky_update(H @ rv.cov_cholesky, np.sqrt(_diffusion) * SR)
    new_cov = new_cov_cholesky @ new_cov_cholesky.T
    crosscov = rv.cov @ H.T
    info = {"crosscov": crosscov}
    if compute_gain:
        gain = crosscov @ np.linalg.inv(new_cov)
        info["gain"] = gain
    return pnrv.Normal(new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky), info


# #######################################################################################################################
#
# Backward implementations (think: updates)
#
# The signature of a backward method is
#
#   backward_rv_*(attained_rv, rv, forwarded_rv=None, gain=None, discrete_transition=None, t=None, _diffusion=None) -> (RV, dict)
#
# where as much as possible out of `forwarded_rv` and `gain` are reused,
# and if either one is not provided, they are computed via discrete_transition.forward_rv(rv, t, _diffusion).
# #######################################################################################################################


def backward_rv_classic(
    attained_rv,
    rv,
    forwarded_rv=None,
    gain=None,
    discrete_transition=None,
    t=None,
    _diffusion=None,
):

    if forwarded_rv is None or gain is None:
        forwarded_rv, info = discrete_transition.forward_rv(
            rv, t=t, compute_gain=True, _diffusion=_diffusion
        )
        gain = info["gain"]

    new_mean = rv.mean + gain @ (attained_rv.mean - forwarded_rv.mean)
    new_cov = rv.cov + gain @ (attained_rv.cov - forwarded_rv.cov) @ gain.T
    return pnrv.Normal(new_mean, new_cov), {}


def backward_rv_sqrt(
    attained_rv,
    rv,
    forwarded_rv=None,
    gain=None,
    discrete_transition=None,
    t=None,
    dt=None,
    _diffusion=None,
):  # forwarded_rv is ignored in square-root smoothing.

    # Smoothing updates need the gain from the beginning on
    if np.linalg.norm(rv.cov) > 0 and gain is None:
        _, info = discrete_transition.forward_rv(
            rv, t=t, dt=dt, compute_gain=True, _diffusion=_diffusion
        )
        gain = info["gain"]

    H = discrete_transition.state_trans_mat_fun(time)
    SR = discrete_transition.proc_noise_cov_cholesky_fun(time)
    shift = discrete_transition.shift_vec_fun(time)

    SC_past = rv.cov_cholesky
    SC_attained = attained_rv.cov_cholesky

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

    if gain is None:
        gain = big_triu[:dim, dim:].T @ np.linalg.inv(big_triu[:dim, :dim].T)

    new_mean = rv.mean + gain @ (attained_rv.mean - H @ rv.mean - shift)
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
    The name of the function is leaned on `np.triu` and `np.tril`.
    """
    tril_mat = triu_mat.T
    with_pos_diag = tril_mat @ np.diag(np.sign(np.diag(tril_mat)))
    return with_pos_diag
