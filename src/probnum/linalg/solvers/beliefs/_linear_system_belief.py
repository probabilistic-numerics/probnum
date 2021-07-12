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

    For instantiation either a belief about the solution or the inverse and right hand side must be provided. Note that if both are specified, their consistency is not checked and depending on the algorithm either may be used.

    Parameters
    ----------
    x :
        Belief about the solution.
    Ainv :
        Belief about the (pseudo-)inverse of the system matrix.
    A :
        Belief about the system matrix.
    b :
        Belief about the right hand side.
    hyperparams :
        Hyperparameters of the belief.
    """

    def __init__(
        self,
        x: Optional[randvars.RandomVariable] = None,
        Ainv: Optional[randvars.RandomVariable] = None,
        A: Optional[randvars.RandomVariable] = None,
        b: Optional[randvars.RandomVariable] = None,
        hyperparams: Optional[Mapping[str, randvars.RandomVariable]] = None,
    ):

        if x is None and Ainv is None:
            raise TypeError(
                "Belief over the solution x and the inverse Ainv cannot both be None."
            )

        # Check shapes and their compatibility
        def dim_mismatch_error(**kwargs):
            argnames = list(kwargs.keys())
            return ValueError(
                f"Dimension mismatch. The shapes of {argnames[0]} : {kwargs[argnames[0]].shape} "
                f"and {argnames[1]} : {kwargs[argnames[1]].shape} must match."
            )

        if x is not None:
            if x.ndim > 2 or x.ndim < 1:
                raise ValueError(
                    f"Belief over solution must have either one or two dimensions, but has {x.ndim}."
                )
            if A is not None:
                if A.shape[1] != x.shape[0]:
                    raise dim_mismatch_error(A=A, x=x)

            if x.ndim > 1:
                if x.shape[1] != b.shape[1]:
                    raise dim_mismatch_error(x=x, b=b)
            elif b is not None:
                if b.ndim > 1:
                    raise dim_mismatch_error(x=x, b=b)

        if Ainv is not None:
            if Ainv.ndim != 2:
                raise ValueError(
                    f"Belief over the inverse system matrix may have at most two dimensions, but has {A.ndim}."
                )
            if A is not None:
                if A.shape != Ainv.shape:
                    raise dim_mismatch_error(A=A, Ainv=Ainv)

        if A is not None:
            if A.ndim != 2:
                raise ValueError(
                    f"Belief over the system matrix may have at most two dimensions, but has {A.ndim}."
                )
            if b is not None:
                if A.shape[0] != b.shape[0]:
                    raise dim_mismatch_error(A=A, b=b)

        if b is not None:
            if b.ndim > 2 or b.ndim < 1:
                raise ValueError(
                    f"Belief over right-hand-side may have either one or two dimensions but has {b.ndim}."
                )

        self._x = x
        self._A = A
        self._Ainv = Ainv
        self._b = b
        if hyperparams is None:
            hyperparams = {}
        self._hyperparams = hyperparams

    def hyperparameter(self, key: str) -> randvars.RandomVariable:
        """Hyperparameter of the linear system belief.

        Parameters
        ----------
        key :
            Hyperparameter key.
        """
        return self._hyperparams[key]

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
    def Ainv(self) -> Optional[randvars.RandomVariable]:
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
