import numpy as np

from probnum import utils

from ._random_variable import DiscreteRandomVariable


class Categorical(DiscreteRandomVariable):
    """Categorical random variable.

    Parameters
    ----------
    probabilities :
        Probabilities of the events.
    support :
        Support of the categorical distribution. Optional. Default is None,
        in which case the support is chosen as :math:`(0, ..., K-1)` where
        :math:`K` is the number of elements in `event_probabilities`.
    """

    def __init__(self, probabilities, support=None, random_state=None):
        # The set of events is names "support" to be aligned with the method
        # DiscreteRandomVariable.in_support().

        num_categories = len(probabilities)
        self._probabilities = np.asarray(probabilities)
        self._support = (
            np.asarray(support) if support is not None else np.arange(num_categories)
        )

        parameters = {
            "support": self._support,
            "event_probabilities": self._probabilities,
            "num_categories": num_categories,
        }

        def _sample_categorical(size=()):
            mask = np.random.choice(
                np.arange(len(self.support)), size=size, p=self.probabilities
            ).reshape(size)
            return self.support[mask]

        def _pmf_categorical(x):

            # Defense against cryptic warnings such as:
            # https://stackoverflow.com/questions/45020217/numpy-where-function-throws-a-futurewarning-returns-scalar-instead-of-list
            x = np.asarray(x)
            if x.dtype != self.dtype:
                raise ValueError(
                    "The data type of x does not match with the data type of the support."
                )

            mask = (x == self.support).nonzero()[0]
            return self.probabilities[mask][0] if len(mask) > 0 else 0.0

        def _mode_categorical():
            mask = np.argmax(self.probabilities)
            return self.support[mask]

        super().__init__(
            shape=self._support[0].shape,
            dtype=self._support[0].dtype,
            random_state=random_state,
            parameters=parameters,
            sample=_sample_categorical,
            pmf=_pmf_categorical,
            mode=_mode_categorical,
        )

    @property
    def probabilities(self):
        """Event probabilities of the categorical distribution."""
        return self._probabilities

    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = np.asarray(probabilities)

    @property
    def support(self):
        """Support of the categorical distribution."""
        return self._support

    @support.setter
    def support(self, support):
        self._support = np.asarray(support)
