"""Markovian processes."""

from typing import Optional

from probnum import _function, backend, randvars
from probnum.randprocs import _random_process, kernels
from probnum.randprocs.markov import _transition
from probnum.typing import ArrayLike, SeedLike, ShapeLike


class MarkovProcess(_random_process.RandomProcess[ArrayLike, backend.ndarray]):
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
        initarg: backend.ndarray,
        initrv: randvars.RandomVariable,
        transition: _transition.Transition,
    ):
        self.initarg = initarg
        self.initrv = initrv
        self.transition = transition

        input_shape = initarg.shape
        output_shape = initrv.shape

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=backend.dtype(backend.double),
            mean=_function.LambdaFunction(
                lambda x: self.__call__(args=x).mean,
                input_shape=input_shape,
                output_shape=output_shape,
            ),
            cov=MarkovProcess.Kernel(
                self.__call__,
                input_shape=input_shape,
                output_shape=2 * output_shape,
            ),
        )

    def __call__(self, args: ArrayLike) -> randvars.RandomVariable:
        raise NotImplementedError

    def _sample_at_input(
        self,
        seed: SeedLike,
        args: ArrayLike,
        sample_shape: ShapeLike = (),
    ) -> backend.ndarray:

        size = backend.as_shape(size)
        args = backend.atleast_1d(args)
        if args.ndim > 1:
            raise ValueError(f"Invalid args shape {args.shape}")

        base_measure_realizations = backend.random.standard_normal(
            seed=backend.random.seed(seed),
            shape=(sample_shape + args.shape + self.initrv.shape),
        )

        if size == ():
            return backend.array(
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
            self, x0: backend.ndarray, x1: Optional[backend.ndarray]
        ) -> backend.ndarray:
            if x1 is None:
                return self._markov_proc_call(args=x0).cov

            raise NotImplementedError
