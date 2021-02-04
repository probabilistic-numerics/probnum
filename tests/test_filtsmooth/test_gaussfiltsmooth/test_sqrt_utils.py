# import numpy as np
# import pytest  # pylint: disable=import-error
#
# import probnum.filtsmooth as pnfs
# from probnum.problems.zoo.linalg import random_spd_matrix
#
# SPDMAT = random_spd_matrix(3)
# RANDMAT = np.random.rand(3, 3)
#
#
# @pytest.mark.parametrize("S1", [SPDMAT, RANDMAT @ SPDMAT])
# @pytest.mark.parametrize("S2", [SPDMAT, RANDMAT @ SPDMAT])
# def test_cholesky_update(S1, S2):
#     """Test whether cholesky_update() yields valid Cholesky factors."""
#     S3 = pnfs.cholesky_update(S1, S2)
#     np.testing.assert_allclose(S3 @ S3.T, S1 @ S1.T + S2 @ S2.T)
#     np.testing.assert_allclose(np.tril(S3), S3)
#     np.testing.assert_allclose(np.diag(S3), np.abs(np.diag(S3)))
