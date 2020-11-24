import unittest

import numpy as np

import probnum as pn
import tests.testing


class CholeskyUpdateTestCase(unittest.TestCase, tests.testing.NumpyAssertions):
    def setUp(self):
        self.random_seeds = list(range(5))
        self.random_states = {
            seed: np.random.RandomState(seed=seed) for seed in self.random_seeds
        }

        self.dimensions = [2, 3, 5, 10, 100]

        # Matrices to be updated
        self.matrices = {
            (n, seed): tests.testing.random_spd_matrix(
                n,
                spectrum_shape=10.0,
                spectrum_offset=1.0,
                random_state=self.random_states[seed],
            )
            for seed in self.random_seeds
            for n in self.dimensions
        }

        # Update vectors
        self.vs = {
            (n, seed): self.random_states[seed].normal(scale=10, size=n)
            for n, seed in self.matrices.keys()
        }

        # Updated matrices
        self.updated_matrices = {
            key: matrix + np.outer(self.vs[key], self.vs[key])
            for key, matrix in self.matrices.items()
        }

        # Lower Cholesky factors
        self.Ls = {}

        for (n, seed), matrix in self.matrices.items():
            L = np.linalg.cholesky(matrix)

            L_row_major = np.array(L, copy=False, order="C")
            L_column_major = np.array(L, copy=False, order="F")

            assert L_row_major.flags.owndata or L_column_major.flags.owndata
            assert L_row_major.flags.c_contiguous
            assert L_column_major.flags.f_contiguous

            self.Ls[(n, seed, "C")] = L_row_major
            self.Ls[(n, seed, "F")] = L_column_major

        # Updated upper Cholesky factors
        self.updated_Ls = {
            (n, seed, order): pn.utils.cholesky_rank_1_update(
                L=L.copy(order="K"),
                v=self.vs[(n, seed)].copy(),
                overwrite_L=True,
                overwrite_v=True,
            )
            for (n, seed, order), L in self.Ls.items()
        }

    def test_valid_matrix_square_root(self):
        for key, updated_L in self.updated_Ls.items():
            n, seed, order = key

            with self.subTest(f"n = {n}, random seed = {seed}, memory order: {order}"):
                self.assertAllClose(
                    updated_L @ updated_L.T, self.updated_matrices[(n, seed)]
                )

    def test_positive_diagonal(self):
        for key, updated_L in self.updated_Ls.items():
            n, seed, order = key

            with self.subTest(f"n = {n}, random seed = {seed}, memory order: {order}"):
                diag_updated_L = np.diag(updated_L)

                self.assertArrayLess(np.zeros_like(diag_updated_L), diag_updated_L)

    def test_memory_order(self):
        for key, updated_L in self.updated_Ls.items():
            n, seed, order = key

            with self.subTest(f"n = {n}, random seed = {seed}, memory order: {order}"):
                if order == "C":
                    self.assertTrue(updated_L.flags.c_contiguous)
                else:
                    assert order == "F"

                    self.assertTrue(updated_L.flags.f_contiguous)

    def test_upper_triangular_part_not_accessed(self):
        for key, L in self.Ls.items():
            n, seed, order = key

            with self.subTest(f"n = {n}, random seed = {seed}, memory order: {order}"):
                # Modify the lower triangular part of L_T to check that the same result
                # is produced and that the lower triangular part is not modified.
                mod_triu_L = L.copy(order="K")
                mod_triu_L[np.triu_indices(n, k=1)] = np.random.rand()

                updated_mod_triu_L = pn.utils.cholesky_rank_1_update(
                    L=mod_triu_L, v=self.vs[(n, seed)],
                )

                self.assertArrayEqual(
                    np.triu(mod_triu_L, k=1),
                    np.triu(updated_mod_triu_L, k=1),
                    msg=(
                        "The rank-1 update modified the upper triangular part of the "
                        "Cholesky factor"
                    ),
                )

                self.assertArrayEqual(
                    np.tril(updated_mod_triu_L),
                    self.updated_Ls[key],
                    msg=(
                        "The rank-1 update did not ignore the upper triangular part of "
                        "the original Cholesky factor"
                    ),
                )

    def test_raise_on_non_square_cholesky_factor(self):
        for shape in [(3, 2), (3,), (1, 3, 3)]:
            with self.subTest(msg=f"shape = {shape}"):
                with self.assertRaises(ValueError):
                    pn.utils.cholesky_rank_1_update(
                        L=np.zeros(shape), v=np.zeros(shape[-1])
                    )

    def test_raise_on_vector_dimension_mismatch(self):
        for key, L in self.Ls.items():
            n, seed, order = key

            with self.subTest(f"n = {n}, random seed = {seed}, memory order: {order}"):
                with self.assertRaises(ValueError):
                    pn.utils.cholesky_rank_1_update(
                        L=L, v=np.random.rand(n + np.random.randint(1, 10)),
                    )

    def test_raise_on_wrong_dtype(self):
        dtypes = [np.float64, np.float32, np.float16, np.complex64, np.int64]

        for L_dtype in dtypes:
            for v_dtype in dtypes:
                if L_dtype is np.float64 and v_dtype is np.float64:
                    continue

                with self.subTest(
                    f"L dtype = {np.dtype(L_dtype).name}, "
                    f"v dtype = {np.dtype(v_dtype).name}"
                ):
                    with self.assertRaises(TypeError):
                        pn.utils.cholesky_rank_1_update(
                            L=np.eye(5, dtype=L_dtype), v=np.zeros(5, dtype=v_dtype),
                        )

    def test_no_input_mutation(self):
        for key, L in self.Ls.items():
            n, seed, order = key

            v = self.vs[(n, seed)]

            with self.subTest(f"n = {n}, random seed = {seed}, memory order: {order}"):
                L_copy = L.copy(order="K")
                v_copy = v.copy()

                pn.utils.cholesky_rank_1_update(
                    L_copy, v_copy, overwrite_L=False, overwrite_v=False
                )

                self.assertArrayEqual(L_copy, L)
                self.assertArrayEqual(v_copy, v)
