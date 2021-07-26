import numpy as np

from probnum import randvars


def condition_state_on_measurement(measurement, forwarded_rv, rv, gain):
    zero_mat = np.zeros((len(measurement), len(measurement)))

    meas_as_rv = randvars.Normal(mean=measurement, cov=zero_mat, cov_cholesky=zero_mat)
    return condition_state_on_rv(meas_as_rv, forwarded_rv, rv, gain)


def condition_state_on_rv(attained_rv, forwarded_rv, rv, gain):
    new_mean = rv.mean + gain @ (attained_rv.mean - forwarded_rv.mean)
    new_cov = rv.cov + gain @ (attained_rv.cov - forwarded_rv.cov) @ gain.T

    return randvars.Normal(new_mean, new_cov)
