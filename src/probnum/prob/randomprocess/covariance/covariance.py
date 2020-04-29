"""

"""


class Covariance:
    """
    Covariance function interface.

    Functions of the form :math:`k = k(x, y)`
    and, if available, their partial derivatives.

    Parameters
    ----------
    covfun : callable, signature=``(x, y)``
    xderiv : callable or Covariance
        If callable, a new Covariance object is created without
        derivatives.
    yderiv : callable or Covariance
        If callable, a new Covariance object is created without
        derivatives.

    Examples
    --------

    To initialize a Covariance object just pass a function with
    signature ``(x, y)`` to the ``covfun`` key.

    >>> from probnum.prob.randomprocess import Covariance
    >>> cov = Covariance(covfun=(lambda x, y: 1))

    It can be evaluated with ``self.covfun(x, y)``

    >>> print(cov.covfun(3, 4))
    1

    But it is also possible to call the object directly

    >>> print(cov(3, 4))
    1

    which works because ``__call__(self, x, y)``
    calls ``self.covfun(x, y)``.

    One thing that this object can do that generic functions cannot
    do is know derivatives

    >>> cov2 = Covariance(covfun=(lambda x, y: x**2 + y**2), xderiv=(lambda x, y: 2*x), yderiv=(lambda x, y: 2*y))
    >>> cov2.covfun(3, 4)
    25
    >>> cov2(3, 4)
    25
    >>> cov2.xderiv(3, 4)
    6
    >>> cov2.yderiv(3, 4)
    8

    TODO
    ----
    Implement dunder methods to create new covariance kernels out of
    old ones.
    """

    def __init__(self, covfun=None, xderiv=None, yderiv=None):
        """
        """
        self._covfun = covfun
        if xderiv is not None and not issubclass(type(xderiv), Covariance):
            xderiv = Covariance(covfun=xderiv)
        if yderiv is not None and not issubclass(type(yderiv), Covariance):
            yderiv = Covariance(covfun=yderiv)
        # maybe even check whether the derivatives are precise,
        # or provide a method that checks.
        self._xderiv = xderiv
        self._yderiv = yderiv

    def __call__(self, x, y):
        """
        Proxy for self.covfun(x, y)
        """
        return self.covfun(x, y)

    def covfun(self, x, y):
        """
        Evaluates covariance kernel :math:`k = k(x, y)`.
        """
        return self._covfun(x, y)

    @property
    def xderiv(self):
        """
        Return :math:`\\partial_x k(x, y)` as a Covariance object.

        To evaluate it, call ``self.xderiv.covfun(x, y)``
        or ``self.xderiv(x, y)``.
        """
        return self._xderiv

    @property
    def yderiv(self):
        """
        Return :math:`\\partial_y k(x, y)` as a Covariance object.

        To evaluate it, call ``self.yderiv.covfun(x, y)``.
        or ``self.yderiv(x, y)``.
        """
        return self._yderiv

    def check_xderiv(self, x, fdstep=None, tol=None):
        """
        Checks if the partial derivative w.r.t. :math:`x` is accurate.

        Computes central finite difference approximation around point y
        with size fdstep (finite-difference step)
        and asserts that the error is less than tol.
        """
        raise NotImplementedError

    def check_yderiv(self, y, fdstep=None, tol=None):
        """
        Checks if the partial derivative w.r.t. :math:`x` is accurate.

        Computes central finite difference approximation around point y
        with size fdstep (finite-difference step)
        and asserts that the error is less than tol.
        """
        raise NotImplementedError

    # Some arithmetics (incomplete!) Allow everything that makes #######
    # a new kernel out of old ones. ####################################
    # Be careful with the derivatives ##################################

    def __add__(self, other):
        """
        """
        return NotImplemented

    def __mul__(self, other):
        """
        """
        return NotImplemented
