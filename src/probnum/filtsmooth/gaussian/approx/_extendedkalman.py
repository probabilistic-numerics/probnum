"""Extended Kalman filtering."""

from probnum import randprocs, randvars

from ._interface import _LinearizationInterface


# Order of inheritance matters, because forward and backward
# are defined in EKFComponent, and must not be inherited from SDE.
class ContinuousEKFComponent(_LinearizationInterface, randprocs.markov.continuous.SDE):
    """Continuous-time extended Kalman filter transition.

    Parameters
    ----------
    non_linear_model
        Non-linear continuous-time model (:class:`SDE`)
        that is approximated with the EKF.
    mde_atol
        Absolute tolerance passed to the solver of the
        moment differential equations (MDEs). Optional.
    mde_rtol
        Relative tolerance passed to the solver of the
        moment differential equations (MDEs). Optional.
    mde_solver
        Method that is chosen in `scipy.integrate.solve_ivp`.
        Any string that is compatible with
        ``solve_ivp(..., method=mde_solve,...)`` works here.
        Usual candidates are ``[RK45, LSODA, Radau, BDF, RK23, DOP853]``.
        Optional. Default is LSODA.
    """

    def __init__(
        self,
        non_linear_model,
        mde_atol=1e-5,
        mde_rtol=1e-5,
        mde_solver="RK45",
        forward_implementation="classic",
    ) -> None:

        randprocs.markov.continuous.SDE.__init__(
            self,
            state_dimension=non_linear_model.state_dimension,
            wiener_process_dimension=non_linear_model.wiener_process_dimension,
            drift_function=non_linear_model.drift_function,
            dispersion_function=non_linear_model.dispersion_function,
            drift_jacobian=non_linear_model.drift_jacobian,
        )
        _LinearizationInterface.__init__(self, non_linear_model=non_linear_model)

        self.mde_atol = mde_atol
        self.mde_rtol = mde_rtol
        self.mde_solver = mde_solver

        self.forward_implementation = forward_implementation

    def linearize(self, t, at_this_rv: randvars.Normal):
        """Linearize the drift function with a first order Taylor expansion."""

        g = self.non_linear_model.drift_function
        dg = self.non_linear_model.drift_jacobian
        l = self.non_linear_model.dispersion_function

        x0 = at_this_rv.mean

        def force_vector_function(t):
            return g(t, x0) - dg(t, x0) @ x0

        def drift_matrix_function(t):
            return dg(t, x0)

        def dispersion_matrix_function(t):
            return l(t, x0)

        return randprocs.markov.continuous.LinearSDE(
            state_dimension=self.non_linear_model.state_dimension,
            wiener_process_dimension=self.non_linear_model.wiener_process_dimension,
            drift_matrix_function=drift_matrix_function,
            force_vector_function=force_vector_function,
            dispersion_matrix_function=dispersion_matrix_function,
            mde_atol=self.mde_atol,
            mde_rtol=self.mde_rtol,
            mde_solver=self.mde_solver,
            forward_implementation=self.forward_implementation,
        )


class DiscreteEKFComponent(
    _LinearizationInterface, randprocs.markov.discrete.NonlinearGaussian
):
    """Discrete extended Kalman filter transition."""

    def __init__(
        self,
        non_linear_model,
        forward_implementation="classic",
        backward_implementation="classic",
    ) -> None:

        randprocs.markov.discrete.NonlinearGaussian.__init__(
            self,
            input_dim=non_linear_model.input_dim,
            output_dim=non_linear_model.output_dim,
            transition_fun=non_linear_model.transition_fun,
            noise_fun=non_linear_model.noise_fun,
            transition_fun_jacobian=non_linear_model.transition_fun_jacobian,
        )
        _LinearizationInterface.__init__(self, non_linear_model=non_linear_model)

        self.forward_implementation = forward_implementation
        self.backward_implementation = backward_implementation

    def linearize(self, t, at_this_rv: randvars.Normal):
        """Linearize the dynamics function with a first order Taylor expansion."""

        g = self.non_linear_model.transition_fun
        dg = self.non_linear_model.transition_fun_jacobian

        x0 = at_this_rv.mean

        def transition_matrix_fun(t):
            return dg(t, x0)

        def noise_fun(t):
            pnoise = self.non_linear_model.noise_fun(t)
            m = g(t, x0) - dg(t, x0) @ x0
            return m + pnoise

        return randprocs.markov.discrete.LinearGaussian(
            input_dim=self.non_linear_model.input_dim,
            output_dim=self.non_linear_model.output_dim,
            transition_matrix_fun=transition_matrix_fun,
            noise_fun=noise_fun,
            forward_implementation=self.forward_implementation,
            backward_implementation=self.backward_implementation,
        )
