import numpy as np

from probnum import utils

from ._random_variable import DiscreteRandomVariable


class Categorical(DiscreteRandomVariable):
    """Categorical random variable.

    Parameters
    ----------
    support :
        Support of the categorical distribution.
    event_probabilities :
        Probabilities that each event in the support happens.
    """

    def __init__(self, support, event_probabilities=None):

        # support = utils.atleast_1d(support)
        # event_probabilities = utils.atleast_1d(event_probabilities)

        self._event_probabilities = (
            np.ones(len(support)) / len(support)
            if event_probabilities is None
            else np.asarray(event_probabilities)
        )
        self._support = np.asarray(support)

        parameters = {
            "support": self._support,
            "event_probabilities": self._event_probabilities,
            "num_categories": len(support),
        }

        def _sample_categorical(size=()):
            raise NotImplementedError

        def _pmf_categorical(x):
            idx = np.where(x == self._support)[0]
            return self._event_probabilities[idx] if len(idx) > 0 else 0.0

        super().__init__(
            shape=(),
            dtype=self._support.dtype,
            parameters=parameters,
            sample=_sample_categorical,
            pmf=_pmf_categorical,
        )

    @property
    def event_probabilities(self):
        return self._event_probabilities

    @property
    def support(self):
        return self._support
