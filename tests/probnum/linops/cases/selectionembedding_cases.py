import numpy as np

import probnum as pn


def case_embedding():
    return (
        pn.linops.Embedding(
            take_indices=(0, 1, 2), put_indices=(1, 0, 3), shape=(4, 3)
        ),
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def case_selection():
    return pn.linops.Selection(indices=(1, 0, 3), shape=(3, 4)), np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
