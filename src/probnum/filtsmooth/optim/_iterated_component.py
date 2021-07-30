import numpy as np

from probnum import randprocs
from probnum.filtsmooth.optim import _stoppingcriterion


class IteratedDiscreteComponent(randprocs.markov.Transition):
    """Iterated updates.

    Examples
    --------
    >>> from probnum.filtsmooth.optim import StoppingCriterion
    >>> from probnum.filtsmooth.gaussian.approx import DiscreteEKFComponent
    >>> from probnum.problems.zoo.diffeq import logistic
    >>> from probnum.randprocs.markov.integrator import IntegratedWienerProcess
    >>> from probnum.randprocs.markov.discrete import DiscreteGaussian
    >>> from probnum.randvars import Constant
    >>> import numpy as np
    >>>

    Set up an iterated component.

    >>> iwp = IntegratedWienerProcess(initarg=0., num_derivatives=2, wiener_process_dimension=1)
    >>> H0, H1 = iwp.transition.proj2coord(coord=0), iwp.transition.proj2coord(coord=1)
    >>> call = lambda t, x: H1 @ x - H0 @ x * (1 - H0 @ x)
    >>> jacob = lambda t, x: H1 - (1 - 2*(H0 @ x)) @ H0
    >>> nonlinear_model = DiscreteGaussian.from_callable(3, 1, call, jacob)
    >>> ekf = DiscreteEKFComponent(nonlinear_model)
    >>> comp = IteratedDiscreteComponent(ekf, StoppingCriterion())

    Generate some random variables and pseudo observations.

    >>> some_array = np.array([0.1, 1., 2.])
    >>> some_rv = Constant(some_array)
    >>> rv, _ = iwp.transition.forward_realization(some_array , t=0., dt=0.1)
    >>> rv_observed, _ =  comp.forward_rv(rv, t=0.2)
    >>> rv_observed *= 0.01  # mitigate zero data

    Its attributes are inherited from the component that is passed through.

    >>> print(comp.input_dim)
    3
    >>> out, info = comp.forward_realization(some_array,some_rv,)
    >>> print(out.mean)
    [0.91]

    But its backward values are different, because of the iteration.

    >>> out_ekf, _ = ekf.backward_rv(rv_observed, rv)
    >>> print(out_ekf.mean)
    [  0.17081493   0.15351366 -13.73607367]
    >>> out_iterated, _ = comp.backward_rv(rv_observed, rv)
    >>> print(out_iterated.mean)
    [  0.17076427   0.15194483 -13.76505168]
    """

    def __init__(
        self,
        component,
        stopcrit=None,
    ):
        self._component = component
        self.stopcrit = (
            _stoppingcriterion.StoppingCriterion() if stopcrit is None else stopcrit
        )
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
