"""Linear system belief.

Class defining a belief about the quantities of interest of a linear
system such as its solution or the matrix inverse and any associated
hyperparameters.
"""

from typing import Mapping, Optional

from probnum import randvars

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

# pylint: disable="invalid-name"


class LinearSystemBelief:
    r"""Belief about quantities of interest of a linear system.

    Random variables :math:`(\mathsf{x}, \mathsf{A}, \mathsf{H}, \mathsf{b})`
    modelling the solution :math:`x`, the system matrix :math:`A`, its (pseudo-)inverse
    :math:`H=A^{-1}` and the right hand side :math:`b` of a linear system :math:`Ax=b`, as well as any associated hyperparameters.

    Parameters
    ----------
    x :
        Belief about the solution.
    A :
        Belief about the system matrix.
    Ainv :
        Belief about the (pseudo-)inverse of the system matrix.
    b :
        Belief about the right hand side.
    hyperparams :
        Hyperparameters of the belief.
    """

    def __init__(
        self,
        A: randvars.RandomVariable,
        Ainv: randvars.RandomVariable,
        b: randvars.RandomVariable,
        x: randvars.RandomVariable,
        hyperparams: Optional[Mapping[str, randvars.RandomVariable]] = None,
    ):

        # Check shapes
        if A.ndim != 2 or b.ndim > 2 or b.ndim < 1:
            raise ValueError(
                "Beliefs over system components must be at most two-dimensional."
            )

        def dim_mismatch_error(arg0, arg1, arg0_name, arg1_name):
            return ValueError(
                f"Dimension mismatch. The shapes of {arg0_name} : {arg0.shape} "
                f"and {arg1_name} : {arg1.shape} must match."
            )

        if A.shape[0] != b.shape[0]:
            raise dim_mismatch_error(A, b, "A", "b")

        if x is not None:
            if x.ndim > 2 or x.ndim < 1:
                raise ValueError("Belief over solution must be one or two-dimensional.")
            if A.shape[1] != x.shape[0]:
                raise dim_mismatch_error(A, x, "A", "x")

            if x.ndim > 1:
                if x.shape[1] != b.shape[1]:
                    raise dim_mismatch_error(x, b, "x", "b")
            else:
                if b.ndim > 1:
                    raise dim_mismatch_error(x, b, "x", "b")

        if Ainv is not None:
            if A.shape != Ainv.shape:
                raise dim_mismatch_error(A, Ainv, "A", "Ainv")

        self._x = x
        self._A = A
        self._Ainv = Ainv
        self._b = b
        self._hyperparams = hyperparams

    @property
    def hyperparams(
        self,
    ) -> Optional[Mapping[str, randvars.RandomVariable]]:
        """Hyperparameters of the linear system belief."""
        return self._hyperparams

    @cached_property
    def x(self) -> randvars.RandomVariable:
        """Belief about the solution."""
        if self._x is None:
            return self._induced_x()
        else:
            return self._x

    @property
    def A(self) -> randvars.RandomVariable:
        """Belief about the system matrix."""
        return self._A

    @property
    def Ainv(self) -> randvars.RandomVariable:
        """Belief about the (pseudo-)inverse of the system matrix."""
        return self._Ainv

    @property
    def b(self) -> randvars.RandomVariable:
        """Belief about the right hand side."""
        return self._b

    def _induced_x(self) -> randvars.RandomVariable:
        r"""Induced belief about the solution from a belief about the inverse.

        Computes the induced belief about the solution given by (an approximation
        to) the random variable :math:`x=Hb`. This assumes independence between
        :math:`H` and :math:`b`.
        """
        return self.Ainv @ self.b
