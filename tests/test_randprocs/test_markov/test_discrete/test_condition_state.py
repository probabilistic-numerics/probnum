import numpy as np

from probnum import randprocs


def test_condition_state_on_rv(some_normal_rv1, some_normal_rv2):
    """If rv_attained == rv_forwarded, the conditioned rv is the prior rv.

    This function is indirectly tested so many times, we really don't need to be fancy
    here.
    """
    gain = np.random.rand(len(some_normal_rv1.mean), len(some_normal_rv1.mean))

    out = randprocs.markov.discrete.condition_state_on_rv(
        some_normal_rv2, some_normal_rv2, some_normal_rv1, gain
    )
    np.testing.assert_allclose(out.mean, some_normal_rv1.mean)
    np.testing.assert_allclose(out.cov, some_normal_rv1.cov)


def test_condition_state_on_measurement(some_normal_rv1, some_normal_rv2):
    """If rv_attained == rv_forwarded, the conditioned rv is the prior rv.

    This function is indirectly tested so many times, we really don't need to be fancy
    here.
    """
    gain = np.random.rand(len(some_normal_rv1.mean), len(some_normal_rv1.mean))

    out = randprocs.markov.discrete.condition_state_on_measurement(
        some_normal_rv2.mean, some_normal_rv2, some_normal_rv1, gain
    )

    # Only shape tests
    np.testing.assert_allclose(out.mean.shape, some_normal_rv1.mean.shape)
    np.testing.assert_allclose(out.cov.shape, some_normal_rv1.cov.shape)
