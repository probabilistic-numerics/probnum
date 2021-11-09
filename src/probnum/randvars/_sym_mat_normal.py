import numpy as np

from probnum import linops
from probnum.typing import FloatArgType

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

    def _symmetric_kronecker_identical_factors_sample(
        self, rng: np.random.Generator, size: ShapeType = ()
    ) -> np.ndarray:
        assert (
            isinstance(self.cov, linops.SymmetricKronecker)
            and self.cov.identical_factors
        )

        n = self.mean.shape[1]

        # Draw standard normal samples
        size_sample = (n * n,) + size

        stdnormal_samples = scipy.stats.norm.rvs(size=size_sample, random_state=rng)

        # Appendix E: Bartels, S., Probabilistic Linear Algebra, PhD Thesis 2019
        samples_scaled = linops.Symmetrize(n) @ (self.cov_cholesky @ stdnormal_samples)

        # TODO: can we avoid todense here and just return operator samples?
        return self.dense_mean[None, :, :] + samples_scaled.T.reshape(-1, n, n)
