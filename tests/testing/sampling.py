from typing import Optional

import numpy as np
import scipy.stats


def random_spd_matrix(
    n: int,
    spectrum: np.ndarray = None,
    spectrum_shape: float = 10.0,
    spectrum_scale: float = 1.0,
    spectrum_offset: float = 0.0,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    Q = scipy.stats.special_ortho_group.rvs(n, random_state=random_state)

    if spectrum is None:
        spectrum = scipy.stats.gamma.rvs(
            spectrum_shape,
            loc=spectrum_offset,
            scale=spectrum_scale,
            size=n,
            random_state=random_state,
        )

        # TODO: Sort the spectrum?

    return Q @ np.diag(spectrum) @ Q.T
