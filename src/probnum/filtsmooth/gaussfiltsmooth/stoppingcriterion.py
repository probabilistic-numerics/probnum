"""Stopping criteria for iterated filtering and smoothing."""


class StoppingCriterion:
    """
    Stopping criteria for iterated filters/smoothers.


    By default this stopping criterion is defined in a way
    that iterated filters behave like normal filters.
    Though, to unlock `Kalman.iterated_filtsmooth`, implement `self.stop_filtsmooth_updates`.
    For iterated filtering (not iterated smoothing!), implement `self.stop_filter_updates`.
    """

    def continue_filter_updates(self, predrv=None, info_pred=None, filtrv=None, meas_rv=None, info_upd=None, **kwargs):
        """
        When do we stop iterating the filter steps. Default is true.
        If, e.g. IEKF is wanted, overwrite with something that does not always return True.
        """
        return False

    def continue_filtsmooth_updates(self, **kwargs):
        """If implemented, iterated_filtsmooth() is unlocked."""
        raise NotImplementedError
