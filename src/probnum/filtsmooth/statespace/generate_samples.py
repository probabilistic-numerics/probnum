import numpy as np


def generate_samples(dynmod, measmod, initrv, times):
    """Samples true states and observations at pre-determined timesteps "times" for a
    state space model.

    Parameters
    ----------
    dynmod : statespace.Transition
        Transition model describing the prior dynamics.
    measmod : statespace.Transition
        Transition model describing the measurement model.
    initrv : probnum.RandomVariable object
        Random variable according to initial distribution
    times : np.ndarray, shape (n,)
        Timesteps on which the states are to be sampled.
    num_steps : int
        Number of steps to be taken for numerical integration
        of the continuous prior model. Optional. Default is 5.
        Irrelevant for time-invariant or discrete models.

    Returns
    -------
    states : np.ndarray; shape (len(times), dynmod.dimension)
        True states according to dynamic model.
    obs : np.ndarray; shape (len(times), measmod.dimension)
        Observations according to measurement model.
    """
    states = np.zeros((len(times), measmod.input_dim))
    obs = np.zeros((len(times), measmod.output_dim))

    # initial observation point
    states[0] = initrv.sample()
    next_obs_rv, _ = measmod.forward_realization(realization=states[0], t=times[0])
    obs[0] = next_obs_rv.sample()

    # all future points
    for idx in range(1, len(times)):
        t, dt = times[idx - 1], times[idx] - times[idx - 1]
        next_state_rv, _ = dynmod.forward_realization(
            realization=states[idx - 1], t=t, dt=dt
        )
        states[idx] = next_state_rv.sample()
        next_obs_rv, _ = measmod.forward_realization(realization=states[idx], t=t)
        obs[idx] = next_obs_rv.sample()
    return states, obs
