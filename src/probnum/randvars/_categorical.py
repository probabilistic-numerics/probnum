"""Categorical random variables."""
from typing import Optional

import numpy as np

from probnum import BACKEND, Backend, backend
from probnum.backend.random import RNGState
from probnum.backend.typing import ArrayLike, SeedType, ShapeLike

from ._random_variable import DiscreteRandomVariable


class Categorical(DiscreteRandomVariable):
    """Categorical random variable.

    Parameters
    ----------
    probabilities
        Probabilities of the events.
    support
        Support of the categorical distribution. Optional. Default is None,
        in which case the support is chosen as :math:`(0, ..., K-1)` where
        :math:`K` is the number of elements in `probabilities`.
    """

    def __init__(
        self,
        probabilities: ArrayLike,
        support: Optional[backend.Array] = None,
    ):

        # The set of events is named "support" to be aligned with the method
        # DiscreteRandomVariable.in_support().

        self._probabilities = backend.asarray(probabilities)
        num_categories = len(probabilities)
        self._support = (
            backend.asarray(support)
            if support is not None
            else backend.arange(num_categories)
        )

        parameters = {
            "support": self._support,
            "probabilities": self._probabilities,
            "num_categories": num_categories,
        }

        def _sample_categorical(rng_state: RNGState, sample_shape: ShapeLike = ()):
            """Sample from a categorical distribution.

            While on first sight, one might think that this implementation can be
            replaced by `np.random.choice(self.support, sample_shape,
            self.probabilities)`, this is not true, because `np.random.choice` cannot
            handle arrays with `ndim > 1`, but `self.support` can be just that. This
            detour via the `mask` avoids this problem.
            """
            sample_shape = backend.asshape(sample_shape)
            indices = backend.random.choice(
                rng_state,
                np.arange(len(self.support)),
                shape=sample_shape,
                p=self.probabilities,
            ).reshape(sample_shape)
            return self.support[indices]

        def _pmf_categorical(x: ArrayLike):
            """PMF of a categorical distribution."""

            # This implementation is defense against cryptic warnings such as:
            # https://stackoverflow.com/questions/45020217/numpy-where-function-throws-a-futurewarning-returns-scalar-instead-of-list
            x = backend.asarray(x)
            if x.dtype != self.dtype:
                raise ValueError(
                    "The data type of x does not match with the data type of the "
                    "support."
                )

            mask = (x == self.support).nonzero()[0]
            return self.probabilities[mask][0] if len(mask) > 0 else 0.0

        def _mode_categorical():
            mask = backend.argmax(self.probabilities)
            return self.support[mask]

        super().__init__(
            shape=self._support[0].shape,
            dtype=self._support[0].dtype,
            parameters=parameters,
            sample=_sample_categorical,
            pmf=_pmf_categorical,
            mode=_mode_categorical,
        )

    @property
    def probabilities(self) -> backend.Array:
        """Event probabilities of the categorical distribution."""
        return self._probabilities

    @property
    def support(self) -> backend.Array:
        """Support of the categorical distribution."""
        return self._support

    def resample(self, rng_state: RNGState) -> "Categorical":
        """Resample the support of the categorical random variable.

        Return a new categorical random variable (RV), where the support
        is randomly chosen from the elements in the current support with
        probabilities given by the current event probabilities. The
        probabilities of the resulting categorical RV are all equal.

        Parameters
        ----------
        rng_state
            Random number generator state.

        Returns
        -------
        Categorical
            Categorical random variable with resampled support
            (according to ``self.probabilities``).
        """
        num_events = len(self.support)
        new_support = self.sample(rng_state, sample_shape=num_events)
        new_probabilities = backend.ones(self.probabilities.shape) / num_events
        return Categorical(
            support=new_support,
            probabilities=new_probabilities,
        )
