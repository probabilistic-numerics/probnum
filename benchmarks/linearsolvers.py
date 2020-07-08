"""
Benchmarks for linear solvers.
"""

import numpy as np

from probnum.linalg import problinsolve


class SystemSolve:
    """
    Benchmark solving a linear system.
    """

    def setup(self):
        pass

    def time_sparse(self):
        pass

    def time_dense(self):
        pass

    def time_largescale(self):
        pass

    def peakmem_largescale(self):
        pass

    def mem_largescale(self):
        pass