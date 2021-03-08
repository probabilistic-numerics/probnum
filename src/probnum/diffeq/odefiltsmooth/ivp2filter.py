# """Adapter methods: from initial value problems + state space model to filters.
#
# Soon the be replaced by initialisation methods. The adapter is taken
# care of elsewhere.
# """
#
# import numpy as np
#
# import probnum.filtsmooth as pnfs
# import probnum.random_variables as pnrv
#
# from .initialize import initialize_odefilter_with_rk
#
#
# def ivp2ekf0(ivp, prior, evlvar):
#     """Computes measurement model and initial distribution for KF based on IVP and
#     prior.
#
#     **Initialdistribution:**
#
#     Conditions the initial distribution of the Gaussian filter
#     onto the initial values.
#
#     - If preconditioning is set to ``False``, it conditions
#       the initial distribution :math:`\\mathcal{N}(0, I)`
#       on the initial values :math:`(x_0, f(t_0, x_0), ...)` using
#       as many available deri    vatives as possible.
#
#     - If preconditioning is set to ``True``, it conditions
#       the initial distribution :math:`\\mathcal{N}(0, P P^\\top)`
#       on the initial values :math:`(x_0, f(t_0, x_0), ...)` using
#       as many available derivatives as possible.
#       Note that the projection matrices :math:`H_0` and :math:`H_1`
#       become :math:`H_0 P^{-1}` and :math:`H_1 P^{-1}` which has
#       to be taken into account during the preconditioning.
#
#     **Measurement model:**
#
#     Returns a measurement model :math:`\\mathcal{N}(g(m), R)`
#     involving computing the discrepancy
#
#     .. math:: g(m) = H_1 m(t) - f(t, H_0 m(t)).
#
#     Then it returns either type of Gaussian filter, each with a
#     different interpretation of the Jacobian :math:`J_g`:
#
#     - EKF0 thinks :math:`J_g(m) = H_1`
#     - EKF1 thinks :math:`J_g(m) = H_1 - J_f(t, H_0 m(t)) H_0^\\top`
#     - UKF thinks: ''What is a Jacobian?''
#
#     Note that, again, in the case of a preconditioned state space
#     model, :math:`H_0` and :math:`H_1`
#     become :math:`H_0 P^{-1}` and :math:`H_1 P^{-1}` which has
#     to be taken into account. In this case,
#
#     - EKF0 thinks :math:`J_g(m) = H_1 P^{-1}`
#     - EKF1 thinks :math:`J_g(m) = H_1 P^{-1} - J_f(t, H_0  P^{-1} m(t)) (H_0 P^{-1})^\\top`
#     - UKF again thinks: ''What is a Jacobian?''
#
#     **Note:**
#     The choice between :math:`H_i` and :math:`H_i P^{-1}` is taken care
#     of within the Prior.
#
#     Returns ExtendedKalmanFilter object that is compatible with
#     the GaussianIVPFilter.
#
#     evlvar : float,
#         measurement variance; in the literature, this is "R"
#     """  # pylint: disable=line-too-long
#     ekf_mod = pnfs.DiscreteEKFComponent.from_ode(
#         ivp,
#         prior,
#         evlvar,
#         ek0_or_ek1=0,
#         forward_implementation="sqrt",
#         backward_implementation="sqrt",
#     )
#     initrv = _initial_random_variable(ivp, prior)
#     return pnfs.Kalman(prior, ekf_mod, initrv)
#
#
# def ivp2ekf1(ivp, prior, evlvar):
#     """Computes measurement model and initial distribution for EKF based on IVP and
#     prior.
#
#     Returns ExtendedKalmanFilter object.
#
#     evlvar : float, (this is "R")
#     """
#     ekf_mod = pnfs.DiscreteEKFComponent.from_ode(
#         ivp,
#         prior,
#         evlvar,
#         ek0_or_ek1=1,
#         forward_implementation="sqrt",
#         backward_implementation="sqrt",
#     )
#     initrv = _initial_random_variable(ivp, prior)
#     return pnfs.Kalman(prior, ekf_mod, initrv)
#
#
# def ivp2ukf(ivp, prior, evlvar):
#     """Computes measurement model and initial distribution for EKF based on IVP and
#     prior.
#
#     Returns ExtendedKalmanFilter object.
#
#     evlvar : float, (this is "R")
#     """
#     ukf_mod = pnfs.DiscreteUKFComponent.from_ode(ivp, prior, evlvar)
#     initrv = _initial_random_variable(ivp, prior)
#     return pnfs.Kalman(prior, ukf_mod, initrv)
#
#
# def _initial_random_variable(ivp, prior):
#     """Initialize derivatives of the initial value with a Runge-Kutta method."""
#
#     dim = prior.dimension
#     return pnrv.Normal(np.zeros(dim), np.eye(dim), cov_cholesky=np.eye(dim))
#
#     m0, s0 = initialize_odefilter_with_rk(
#         ivp.rhs, ivp.initrv.mean, ivp.t0, order=prior.ordint, df=ivp.jacobian
#     )
#     return pnrv.Normal(m0, np.diag(s0 ** 2), cov_cholesky=np.diag(s0))
