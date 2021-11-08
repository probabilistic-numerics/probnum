"""Stein kernels."""

from typing import Callable, Optional, Union

import numpy as np

from probnum import backend
from probnum.typing import ArrayLike, IntArgType, ShapeArgType

from ._kernel import Kernel


class LangevinSteinKernel(Kernel):
    """Langevin Stein kernel.

    Parameters
    ----------
    kernel :
        Base kernel.
    grad_log_density :
        Gradient of the log-density.
    input_dim :
        Input dimension
    shape :
        Shape.
    kernel_mean :
        Desired kernel mean. Will be set automatically if not provided.
    """

    def __init__(
        self,
        base_kernel: Kernel,
        grad_log_density: Callable,
        input_dim: IntArgType,
        shape: ShapeArgType = ...,
        kernel_mean: Optional[ArrayLike] = None,
    ):
        self.base_kernel = base_kernel
        self.grad_log_density = grad_log_density
        self.stein_operator = LangevinSteinOperator(
            grad_log_density=self._grad_log_density
        )
        if kernel_mean is None:
            kernel_mean = 0.0
        self.kernel_mean = kernel_mean
        super().__init__(input_dim, shape=shape)

    def _evaluate(
        self, x0: ArrayLike, x1: Optional[ArrayLike]
    ) -> Union[np.ndarray, np.float_]:

        return self.stein_operator(x0=x0, x1=x1) + self._kernel_mean


class LangevinSteinOperator:
    r"""Langevin Stein operator.

    Stein operator defined via a Langevin diffusion of the first kind given by

    .. math ::
        \mathcal{L}_\pi[u] = \nabla_x \log \pi(x) \cdot u(x) + \nabla \cdot u(x)


    Parameters
    ----------
    fun
        Function to which to apply the Stein operator to.
    """

    def __init__(self, grad_log_density: Callable[[np.ndarray], np.ndarray]) -> None:
        self.grad_log_density = grad_log_density

    def __call__(
        self, fun: Callable[[np.ndarray], float]
    ) -> Callable[[np.ndarray], float]:
        raise NotImplementedError

    def apply_to_kernel(
        self, kernel: Kernel
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Applies the Langevin Stein operator to a given kernel."""

        def _stein_kernel_matrix(x0: np.ndarray, x1: np.ndarray):
            gradx0_logp = self._grad_log_density(x0)
            gradx1_logp = self._grad_log_density(x1)
            gradx0_base_kernel = backend.grad(kernel, argnums=0)
            gradx1_base_kernel = backend.grad(kernel, argnums=1)
            gradx0_gradx1_base_kernel = backend.grad(kernel, argnums=(0, 1))

            score_prod = gradx0_logp @ gradx1_logp.T

            return (
                score_prod * kernel(x0=x0, x1=x1)
                + np.einsum(
                    "md,dmn->mn", gradx0_logp, gradx1_base_kernel
                )  # TODO: double check order of indices after grad
                + np.einsum("nd,dmn->mn", gradx1_logp, gradx0_base_kernel)
                + np.einsum("ddmn->mn", gradx0_gradx1_base_kernel)
            )

        return _stein_kernel_matrix
