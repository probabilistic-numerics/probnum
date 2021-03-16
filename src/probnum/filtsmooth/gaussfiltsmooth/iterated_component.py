import numpy as np

import probnum.statespace as pnss

from .stoppingcriterion import StoppingCriterion


class IteratedDiscreteComponent(pnss.Transition):
    """Iterated updates.

    Examples
    --------
    >>> from probnum.filtsmooth import DiscreteEKFComponent, StoppingCriterion
    >>> from probnum.diffeq import logistic
    >>> from probnum.statespace import IBM
    >>> from probnum.randvars import Constant
    >>> from numpy import array
    >>>

    Set up an iterated component.

    >>> prior = IBM(ordint=2, spatialdim=1)
    >>> ekf = DiscreteEKFComponent.from_ode(logistic((0., 1.), initrv=Constant(array([0.1]))), prior, 0.)
    >>> comp = IteratedDiscreteComponent(ekf, StoppingCriterion())

    Generate some random variables and pseudo observations.

    >>> some_array = array([0.1, 1., 2.])
    >>> some_rv = Constant(some_array)
    >>> rv, _ = prior.forward_realization(some_array , t=0., dt=0.1)
    >>> rv_observed, _ =  comp.forward_rv(rv, t=0.2)
    >>> rv_observed *= 0.01  # mitigate zero data

    Its attributes are inherited from the component that is passed through.

    >>> print(comp.input_dim)
    3
    >>> out, info = comp.forward_realization(some_array,some_rv,)
    >>> print(out.mean)
    [0.73]

    But its backward values are different, because of the iteration.

    >>> out_ekf, _ = ekf.backward_rv(rv_observed, rv)
    >>> print(out_ekf.mean)
    [ 0.18392711  0.504723   -8.429155  ]
    >>> out_iterated, _ = comp.backward_rv(rv_observed, rv)
    >>> print(out_iterated.mean)
    [ 0.18201288  0.45367681 -9.1948478 ]
    """

    def __init__(
        self,
        component,
        stopcrit=None,
    ):
        self._component = component
        self.stopcrit = StoppingCriterion() if stopcrit is None else stopcrit
        super().__init__(input_dim=component.input_dim, output_dim=component.output_dim)

    # Iterated filtering implementation

    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        current_rv, info = self._component.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            t=t,
            dt=dt,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

        new_mean = current_rv.mean.copy()
        old_mean = np.inf * np.ones(current_rv.mean.shape)
        while not self.stopcrit.terminate(
            error=new_mean - old_mean, reference=new_mean
        ):
            old_mean = new_mean.copy()
            current_rv, info = self._component.backward_rv(
                rv_obtained=rv_obtained,
                rv=rv,
                t=t,
                dt=dt,
                _diffusion=_diffusion,
                _linearise_at=current_rv,
            )
            new_mean = current_rv.mean.copy()
        return current_rv, info

    def backward_realization(
        self,
        real_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        return self._backward_realization_via_backward_rv(
            real_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            dt=dt,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    # These need to be re-implemented here, because otherwise this class
    # cannot be instantiated (abc things)

    def forward_rv(
        self, rv, t, dt=None, compute_gain=False, _diffusion=1.0, _linearise_at=None
    ):
        return self._component.forward_rv(
            rv,
            t,
            dt=dt,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    def forward_realization(
        self, real, t, dt=None, compute_gain=False, _diffusion=1.0, _linearise_at=None
    ):
        return self._component.forward_realization(
            real,
            t,
            dt=dt,
            compute_gain=compute_gain,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

    # Pass on all the rest to the EKF/UKF component

    def __getattr__(self, attr):

        if attr in [
            "backward_rv",
            "backward_realization",
        ]:
            return self.attr
        else:
            return getattr(self._component, attr)
