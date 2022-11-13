"""Markovian processes."""

from typing import Optional

from probnum import backend, functions, randvars
from probnum.backend.random import RNGState
from probnum.backend.typing import ArrayLike, ShapeLike
from probnum.randprocs import _random_process, kernels
from probnum.randprocs.markov import _transition, continuous, discrete


class _MarkovBase(_random_process.RandomProcess):
    def __init__(
        self,
        *,
        initrv: randvars.RandomVariable,
        transition: _transition.Transition,
        input_shape: ShapeLike = (),
    ):
        self.initrv = initrv
        self.transition = transition

        output_shape = initrv.shape

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=backend.float64,
            mean=functions.LambdaFunction(
                lambda x: self.__call__(args=x).mean,
                input_shape=input_shape,
                output_shape=output_shape,
            ),
            cov=_MarkovBase.Kernel(
                self.__call__,
                input_shape=input_shape,
                output_shape=2 * output_shape,
            ),
        )

    def __call__(self, args: ArrayLike) -> randvars.RandomVariable:
        raise NotImplementedError

    def _sample_at_input(
        self,
        rng_state: RNGState,
        args: ArrayLike,
        sample_shape: ShapeLike = (),
    ) -> backend.Array:

        sample_shape = backend.asshape(sample_shape)
        args = backend.atleast_1d(args)
        if args.ndim > 1:
            raise ValueError(f"Invalid args shape {args.shape}")

        base_measure_realizations = backend.random.standard_normal(
            rng_state=rng_state,
            shape=(sample_shape + args.shape + self.initrv.shape),
        )

        if sample_shape == ():
            return backend.asarray(
                self.transition.jointly_transform_base_measure_realization_list_forward(
                    base_measure_realizations=base_measure_realizations,
                    t=args,
                    initrv=self.initrv,
                    _diffusion_list=backend.ones_like(args[:-1]),
                )
            )

        return backend.stack(
            [
                self.transition.jointly_transform_base_measure_realization_list_forward(
                    base_measure_realizations=base_real,
                    t=args,
                    initrv=self.initrv,
                    _diffusion_list=backend.ones_like(args[:-1]),
                )
                for base_real in base_measure_realizations
            ]
        )

    class Kernel(kernels.Kernel):
        def __init__(
            self, markov_proc_call, input_shape: ShapeLike, output_shape: ShapeLike
        ):
            self._markov_proc_call = markov_proc_call

            super().__init__(
                input_shape=input_shape,
                output_shape=output_shape,
            )

        def _evaluate(
            self, x0: backend.Array, x1: Optional[backend.Array]
        ) -> backend.Array:
            if x1 is None:
                return self._markov_proc_call(args=x0).cov

            raise NotImplementedError


class MarkovProcess(_MarkovBase):
    r"""Random processes with the Markov property.

    A Markov process is a random process with the additional property that
    conditioned on the present state of the system its future and past states are
    independent. This is known as the Markov property or as the process being
    memoryless. A Markov process can be fully defined via an initial state and a
    state transition.

    Parameters
    ----------
    initarg
        Initial starting input of the process.
    initrv
        Random variable describing the initial state.
    transition
        State transition of the system.

    See Also
    --------
    RandomProcess : Random processes.
    GaussianProcess : Gaussian processes.
    """

    def __init__(
        self,
        *,
        initarg: backend.Array,
        initrv: randvars.RandomVariable,
        transition: continuous.SDE,
    ):
        if not isinstance(transition, continuous.SDE):  # pragma: no cover
            msg = "The transition is not continuous. Did you mean 'MarkovSequence'?"
            raise TypeError(msg)

        super().__init__(
            initrv=initrv,
            transition=transition,
            input_shape=backend.asarray(initarg).shape,
        )
        self.initarg = initarg


class MarkovSequence(_MarkovBase):
    """Discrete-time Markov processes."""

    def __init__(
        self,
        *,
        initarg: backend.Array,
        initrv: randvars.RandomVariable,
        transition: continuous.SDE,
    ):
        if not isinstance(transition, discrete.NonlinearGaussian):  # pragma: no cover
            msg = "The transition is not discrete. Did you mean 'MarkovProcess'?"
            raise TypeError(msg)

        super().__init__(
            initrv=initrv,
            transition=transition,
            input_shape=backend.asarray(initarg).shape,
        )
        self.initarg = initarg
