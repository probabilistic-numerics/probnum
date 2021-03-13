import numpy as np

from probnum import utils

from ._random_variable import DiscreteRandomVariable


class Categorical(DiscreteRandomVariable):
    """Categorical random variable.

    Parameters
    ----------
    event_probabilities :
        Probabilities of the events.
    support :
        Support of the categorical distribution. Optional. Default is None,
        in which case the support is chosen as :math:`(0, ..., K-1)` where
        :math:`K` is the number of elements in `event_probabilities`.
    """

    def __init__(self, event_probabilities, support=None, random_state=None):

        num_categories = len(event_probabilities)
        self._event_probabilities = np.asarray(event_probabilities)
        self._support = (
            np.asarray(support) if support is not None else np.arange(num_categories)
        )

        parameters = {
            "support": self._support,
            "event_probabilities": self._event_probabilities,
            "num_categories": num_categories,
        }

        def _sample_categorical(size=()):
            mask = np.random.choice(
                np.arange(len(self.support)), size=size, p=self.event_probabilities
            )
            return self.support[mask]

        def _pmf_categorical(x):
            idx = np.where(x == self._support)[0]
            return self._event_probabilities[idx] if len(idx) > 0 else 0.0

        def _mode_categorical():
            mask = np.argmax(self.event_probabilities)
            return self.support[mask]

        super().__init__(
            shape=self._support[0].shape,
            dtype=self._support.dtype,
            random_state=random_state,
            parameters=parameters,
            sample=_sample_categorical,
            pmf=_pmf_categorical,
            mode=_mode_categorical,
        )

    @property
    def event_probabilities(self):
        """Event probabilities of the categorical distribution."""
        return self._event_probabilities

    @event_probabilities.setter
    def event_probabilities(self, event_probabilities):
        self._event_probabilities = np.asarray(event_probabilities)

    @property
    def support(self):
        """Support of the categorical distribution."""
        return self._support

    @support.setter
    def support(self, support):
        self._support = np.asarray(support)
