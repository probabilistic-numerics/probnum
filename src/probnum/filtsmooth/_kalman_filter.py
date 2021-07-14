"""Convenience functions for filtering and smoothing."""

from probnum import problems, randprocs, statespace

__all__ = ["kalman_filter", "rauch_tung_striebel_smoother"]


def kalman_filter(observations, locations, F, L, H, R, m0, C0, prior_model="discrete"):
    regression_problem = _setup_regression_problem(
        H=H, R=R, observations=observations, locations=locations
    )
    prior_process = _setup_prior_process(
        F=F, L=L, m0=m0, C0=C0, prior_model=prior_model
    )
    kalman = gaussian.Kalman(prior_process)
    return kalman.filter(regression_problem)


def rauch_tung_striebel_smoother(
    observations, locations, F, L, H, R, prior_model="discrete"
):
    regression_problem = _setup_regression_problem(
        H=H, R=R, observations=observations, locations=locations
    )
    prior_process = _setup_prior_process(
        F=F, L=L, m0=m0, C0=C0, prior_model=prior_model
    )
    kalman = gaussian.Kalman(prior_process)
    return kalman.filtsmooth(regression_problem)


def _setup_prior_process(F, L, m0, c0, prior_model):
    zero_shift_prior = np.zeros(F.shape[1])
    if prior_model == "discrete":
        prior = statespace.DiscreteLTIGaussian(
            state_trans_mat=F, shift_vec=zero_shift_prior, proc_noise_cov_mat=L
        )
    elif prior_model == "continuous":
        prior = statespace.LTISDE(driftmat=F, forcevec=zero_shift_prior, dispmat=L)
    else:
        raise ValueError
    initrv = randvars.Normal(m0, C0)
    initarg = locations[0]
    prior_process = randprocs.MarkovProcess(
        transition=prior, initrv=initrv, initarg=initarg
    )
    return prior_process


def _setup_regression_problem(H, R, observations, locations):

    zero_shift_mm = np.zeros(H.shape[1])
    measmod = statespace.DiscreteLTIGaussian(
        state_trans_mat=H, shift_vec=zero_shift_mm, proc_noise_cov_mat=R
    )
    measurement_models = [measmod] * len(locations)
    regression_problem = problems.TimeSeriesRegressionProblem(
        observations=observations,
        locations=locations,
        measurement_models=measurement_models,
    )
    return regression_problem
