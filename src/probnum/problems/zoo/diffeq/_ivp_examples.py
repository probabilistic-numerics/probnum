import numpy as np

from probnum.problems import InitialValueProblem

__all__ = ["threebody_jax"]


def threebody_jax(tmax=17.0652165601579625588917206249):
    try:
        import jax
        import jax.numpy as jnp
        from jax.config import config
        from jax.experimental.jet import jet

        config.update("jax_enable_x64", True)

    except ImportError:
        raise ImportError("Initialisation requires jax. Sorry :( ")

    def threebody_rhs(Y):
        # defining the ODE:
        # assume Y = [y1,y2,y1',y2']
        mu = 0.012277471  # a constant (standardised moon mass)
        mp = 1 - mu
        D1 = ((Y[0] + mu) ** 2 + Y[1] ** 2) ** (3 / 2)
        D2 = ((Y[0] - mp) ** 2 + Y[1] ** 2) ** (3 / 2)
        y1p = Y[0] + 2 * Y[3] - mp * (Y[0] + mu) / D1 - mu * (Y[0] - mp) / D2
        y2p = Y[1] - 2 * Y[2] - mp * Y[1] / D1 - mu * Y[1] / D2
        return jnp.array([Y[2], Y[3], y1p, y2p])

    df = jax.jit(jax.jacfwd(threebody_rhs))
    ddf = jax.jit(jax.jacrev(df))

    def rhs(t, y):
        return threebody_rhs(Y=y)

    def jac(t, y):
        return df(y)

    def hess(t, y):
        return ddf(y)

    y0 = np.array([0.994, 0, 0, -2.00158510637908252240537862224])
    t0, tmax = 0.0, tmax
    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac, ddf=hess)
