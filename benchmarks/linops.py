import numpy as np
import scipy.sparse

import probnum as pn

ALL_LINOPS = (
    "matrix",
    "matrix_sparse",
    "identity",
    "isotropic_scaling",
    "anisotropic_scaling",
    "kronecker",
    "symmetric_kronecker_distinct_factors",
    "symmetric_kronecker_identical_factors",
)

NO_RANK = (
    "matrix",
    "matrix_sparse",
    "symmetric_kronecker_distinct_factors",
)
NO_EIGVALS = (
    "matrix",
    "matrix_sparse",
    "kronecker",
    "symmetric_kronecker_distinct_factors",
    "symmetric_kronecker_identical_factors",
)
NO_COND = {
    2: (
        "matrix",
        "matrix_sparse",
        "symmetric_kronecker_distinct_factors",
    ),
    "fro": (
        "matrix",
        "matrix_sparse",
        "symmetric_kronecker_distinct_factors",
    ),
    1: (
        "matrix",
        "matrix_sparse",
        "symmetric_kronecker_distinct_factors",
    ),
    np.inf: (
        "matrix",
        "matrix_sparse",
        "symmetric_kronecker_distinct_factors",
    ),
}
NO_DET = (
    "matrix",
    "matrix_sparse",
    "symmetric_kronecker_distinct_factors",
)
NO_LOGABSDET = (
    "matrix",
    "matrix_sparse",
    "symmetric_kronecker_distinct_factors",
)
NO_TRACE = ()


def get_linear_operator(name: str) -> pn.linops.LinearOperator:
    if name == "matrix":
        linop = pn.linops.aslinop(np.random.normal(size=(10000, 10000)))
    elif name == "matrix_sparse":
        linop = pn.linops.aslinop(scipy.sparse.random(10000, 10000, format="csr"))
    elif name == "identity":
        linop = pn.linops.Identity(10000)
    elif name == "isotropic_scaling":
        linop = pn.linops.Scaling(3.0, shape=(10000, 10000))
    elif name == "anisotropic_scaling":
        linop = pn.linops.Scaling(np.random.normal(size=(10000,)))
    elif name == "kronecker":
        linop = pn.linops.Kronecker(
            np.random.normal(size=(100, 100)),
            np.random.normal(size=(100, 100)),
        )
    elif name == "symmetric_kronecker_distinct_factors":
        linop = pn.linops.SymmetricKronecker(
            np.random.normal(size=(100, 100)),
            np.random.normal(size=(100, 100)),
        )
    elif name == "symmetric_kronecker_identical_factors":
        linop = pn.linops.SymmetricKronecker(
            np.random.normal(size=(100, 100)),
        )
    else:
        raise NotImplementedError()

    return linop


class Construction:
    param_names = ("operator",)
    params = (ALL_LINOPS,)

    def setup(self, operator: str):
        np.random.seed(42)

    def time_construction(self, operator: str):
        get_linear_operator(operator)

    def peakmem_matvec(self, operator: str):
        get_linear_operator(operator)


class MatVec:
    param_names = ("operator",)
    params = (ALL_LINOPS,)

    def setup(self, operator: str):
        np.random.seed(42)

        self.linop = get_linear_operator(operator)
        self.vec = np.ones(self.linop.shape[1], dtype=self.linop.dtype)

    def time_matvec(self, operator: str):
        _ = self.linop @ self.vec

    def peakmem_matvec(self, operator: str):
        _ = self.linop @ self.vec


class MatMat:
    param_names = ("operator",)
    params = (ALL_LINOPS,)

    def setup(self, operator: str):
        np.random.seed(42)

        self.linop = get_linear_operator(operator)
        self.mat = np.ones(
            (self.linop.shape[1], 1000), dtype=self.linop.dtype, order="F"
        )

    def time_matmat(self, operator: str):
        _ = self.linop @ self.mat

    def peakmem_matmat(self, operator: str):
        _ = self.linop @ self.mat


# TODO: RMatVec, RMatMat, Transpose, Adjoint, Inverse


class Rank:
    param_names = ("operator",)
    params = (ALL_LINOPS,)

    def setup(self, operator: str):
        np.random.seed(42)

        if operator in NO_RANK:
            raise NotImplementedError()

        self.linop = get_linear_operator(operator)

    def time_rank(self, operator: str):
        self.linop.rank()

    def peakmem_rank(self, operator: str):
        self.linop.rank()


class Eigvals:
    param_names = ("operator",)
    params = (ALL_LINOPS,)

    def setup(self, operator: str):
        np.random.seed(42)

        if operator in NO_EIGVALS:
            raise NotImplementedError()

        self.linop = get_linear_operator(operator)

    def time_eigvals(self, operator: str):
        self.linop.eigvals()

    def peakmem_eigvals(self, operator: str):
        self.linop.eigvals()


class Cond:
    param_names = ("operator", "p")
    params = (ALL_LINOPS, [2, "fro", 1, np.inf])

    def setup(self, operator: str, p):
        np.random.seed(42)

        if operator in NO_COND[p]:
            raise NotImplementedError()

        self.linop = get_linear_operator(operator)

    def time_cond(self, operator: str, p):
        self.linop.cond(p=p)

    def peakmem_cond(self, operator: str, p):
        self.linop.cond(p=p)


class Det:
    param_names = ("operator",)
    params = (ALL_LINOPS,)

    def setup(self, operator: str):
        np.random.seed(42)

        if operator in NO_DET:
            raise NotImplementedError()

        self.linop = get_linear_operator(operator)

    def time_det(self, operator: str):
        self.linop.det()

    def peakmem_det(self, operator: str):
        self.linop.det()


class LogAbsDet:
    param_names = ("operator",)
    params = (ALL_LINOPS,)

    def setup(self, operator: str):
        np.random.seed(42)

        if operator in NO_LOGABSDET:
            raise NotImplementedError()

        self.linop = get_linear_operator(operator)

    def time_logabsdet(self, operator: str):
        self.linop.logabsdet()

    def peakmem_logabsdet(self, operator: str):
        self.linop.logabsdet()


class Trace:
    param_names = ("operator",)
    params = (ALL_LINOPS,)

    def setup(self, operator: str):
        np.random.seed(42)

        if operator in NO_TRACE:
            raise NotImplementedError()

        self.linop = get_linear_operator(operator)

    def time_trace(self, operator: str):
        self.linop.trace()

    def peakmem_trace(self, operator: str):
        self.linop.trace()
