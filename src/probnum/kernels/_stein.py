from re import X
from typing import Callable, Optional, Union

import numpy as np

from probnum import _backend
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
        self._base_kernel = base_kernel
        self._grad_log_density = grad_log_density
        if kernel_mean is None:
            kernel_mean = 0.0
        self._kernel_mean = kernel_mean
        super().__init__(input_dim, shape=shape)

    def _evaluate(
        self, x0: ArrayLike, x1: Optional[ArrayLike]
    ) -> Union[np.ndarray, np.float_]:

        return self._langevin_operator_first_kind(x0=x0, x1=x1) + self._kernel_mean

    def _langevin_operator_of_the_first_kind(
        self,
        x0: ArrayLike,
        x1: ArrayLike,
    ) -> Union[np.ndarray, np.float_]:
        r"""Evaluates the Langevin operator of the first kind.

        Computes the Langevin operator of the first kind given by

        .. math ::
            \mathcal{L}_\pi[u] = \nabla_x \log \pi(x) \cdot u(x) + \nabla \cdot u(x)

        Parameters
        ----------

        Returns
        -------

        """
        gradx0_logp = self._grad_log_density(x0)
        gradx1_logp = self._grad_log_density(x1)
        gradx0_base_kernel = _backend.grad(self._base_kernel, argnums=0)
        gradx1_base_kernel = _backend.grad(self._base_kernel, argnums=1)
        gradx0_gradx1_base_kernel = _backend.grad(self._base_kernel, argnums=(0, 1))

        score_prod = gradx0_logp @ gradx1_logp.T

        return (
            score_prod * self._base_kernel(x0=x0, x1=x1)
            + np.einsum(
                "md,dmn->mn", gradx0_logp, gradx1_base_kernel
            )  # TODO: double check order of indices after grad
            + np.einsum("nd,dmn->mn", gradx1_logp, gradx0_base_kernel)
            + np.einsum("ddmn->mn", gradx0_gradx1_base_kernel)
        )
