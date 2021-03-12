import numpy as np

import probnum.random_variables as pnrv


def condition_state_on_measurement(measurement, forwarded_rv, rv, gain):
    zero_mat = np.zeros((len(measurement), len(measurement)))

    meas_as_rv = pnrv.Normal(mean=measurement, cov=zero_mat, cov_cholesky=zero_mat)
    return condition_state_on_rv(meas_as_rv, forwarded_rv, rv, gain)


def condition_state_on_rv(attained_rv, forwarded_rv, rv, gain):
    new_mean = rv.mean + gain @ (attained_rv.mean - forwarded_rv.mean)
    new_cov = rv.cov + gain @ (attained_rv.cov - forwarded_rv.cov) @ gain.T

    return pnrv.Normal(new_mean, new_cov)
