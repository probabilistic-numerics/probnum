import numpy as np

from probnum import backend, linops
from probnum.typing import SeedType, ShapeType

from . import _normal


class SymmetricMatrixNormal(_normal.Normal):
    def __init__(
        self,
        mean: linops.LinearOperatorLike,
        cov: linops.SymmetricKronecker,
    ) -> None:
        if not isinstance(cov, linops.SymmetricKronecker):
            raise ValueError(
                "The covariance operator must have type `SymmetricKronecker`."
            )

        m, n = mean.shape

        if m != n or n != cov.A.shape[0] or n != cov.B.shape[1]:
            raise ValueError(
                "Normal distributions with symmetric Kronecker structured "
                "kernels must have square mean and square kernels factors with "
                "matching dimensions."
            )

        super().__init__(mean=linops.aslinop(mean), cov=cov)

    def _sample(self, seed: SeedType, sample_shape: ShapeType = ()) -> np.ndarray:
        assert (
            isinstance(self.cov, linops.SymmetricKronecker)
            and self.cov.identical_factors
        )

        # TODO (#xyz): Implement correct sampling routine

        n = self.mean.shape[1]

        # Draw standard normal samples
        stdnormal_samples = backend.random.standard_normal(
            seed,
            shape=sample_shape + (n * n, 1),
            dtype=self.dtype,
        )

        # Appendix E: Bartels, S., Probabilistic Linear Algebra, PhD Thesis 2019
        samples_scaled = linops.Symmetrize(n) @ (self.cov_cholesky @ stdnormal_samples)

        # TODO: can we avoid todense here and just return operator samples?
        return self.dense_mean[None, :, :] + samples_scaled.reshape(-1, n, n)