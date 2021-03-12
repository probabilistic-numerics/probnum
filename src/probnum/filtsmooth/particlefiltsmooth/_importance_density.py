# """Importance densities."""
#
# from probnum import statespace
#
#
# def astransition(kalman):
#
#     pass
#
#
# class _WrappedTransition(statespace.Transition):
#
#     def __init__(self, kalman):
#         self.kalman = kalman
#
#     def forward_realization(
#         self,
#         realization,
#         t,
#         dt=None,
#         compute_gain=False,
#         _diffusion=1.0,
#         _linearise_at=None,
#     ):
#         return self.kalman.filter_step
#     pass
#
#
