"""

"""


class Covariance:
    """
    (Parameterized) covariance functions.

    Functions of the form :math:`k = k(x, y)[=k_\\theta(x, y)]`
    and, if available, their derivatives with respect to
    spatial variables :math:`x, y` and the parameters `\\theta`.

    Parameters
    ----------
    covfun : callable, signature=``(x, y)``
    params : list
        Parameters of the covariance. If not empty, calls to the
        evaluation of the covariance objects are done via
        ``self._covfun(x, y, self._params)``, i.e. in this case
        the object expects covariance function signature of
        ``(x, y, params)``.
    parderiv : callable or Covariance
        If callable, a new Covariance object is created without
        derivatives.
    xderiv : callable or Covariance
        (Partial) derivative of the :math:`x` coordinate. If the
        covariance acts on higher dimensions, it is understood as
        the gradient; that is, a function that returns a list of
        evaluations at each evaluation. It can be taken as partial
        derivative w.r.t. the :math:`y_j` coordinate if the ` yidx`
        parameter is set and `yderiv` evalautes to a scalar.
        If callable, a new Covariance object is created without
        derivatives.
    xidx : int, optional.
        Index of the derivative stored in `xderiv`. Only important
        in higher dimensions and if the derivatives are partial derivatives,
        not full gradients. Relevant for implementation of
        covariance arithmetics.
    yderiv : callable or Covariance
        (Partial) derivative of the :math:`y` coordinate. If the
        covariance acts on higher dimensions, it is understood as
        the gradient; that is, a function that returns a list of
        evaluations at each evaluation. It can be taken as partial
        derivative w.r.t. the :math:`y_j` coordinate if the ` yidx`
        parameter is set and `yderiv` evalautes to a scalar.
        If callable, a new Covariance object is created without
        derivatives.
    yidx : int, optional.
        Index of the derivative stored in `yderiv`. Only important
        in higher dimensions and if the derivatives are partial derivatives,
        not full gradients. Relevant for implementation of
        covariance arithmetics.

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

    or that know explicit parameterizations

    >>> cov3 = Covariance(covfun=(lambda x, y, p: p*(x**2 + y)), params=3, parderiv=(lambda x, y, p: x**2 + y))
    >>> cov3(3, 4)
    39
    >>> cov3.parderiv(3, 4)
    13

    >>> cov4 = Covariance(covfun=(lambda x, y, p: p[0]*(x**2 + p[1]*y)), params=[1, 2], parderiv=(lambda x, y, p: [x**2 + p[1]*y, p[0]*y]))
    >>> cov4(3, 4)
    17
    >>> cov4.parderiv(3, 4)
    [17, 4]

    The second example also shows how to implement vector-valued
    covariances which can be used for computing full gradients.

    TODO
    ----
    Implement dunder methods to create new covariance kernels out of
    old ones.
    """

    def __init__(self, covfun=None, params=None, parderiv=None,
                 xderiv=None, xidx=None, yderiv=None, yidx=None):
        """
        """
        self._covfun = covfun
        if params is None:
            params = []
        if parderiv is not None and not issubclass(type(parderiv), Covariance):
            parderiv = Covariance(covfun=parderiv, params=params)
        if xderiv is not None and not issubclass(type(xderiv), Covariance):
            xderiv = Covariance(covfun=xderiv, params=params)
        if yderiv is not None and not issubclass(type(yderiv), Covariance):
            yderiv = Covariance(covfun=yderiv, params=params)
        self._params = params
        self._parderiv = parderiv
        self._xderiv = xderiv
        self._xidx = xidx
        self._yderiv = yderiv
        self._yidx = yidx

    def __call__(self, x, y):
        """
        Proxy for self.covfun(x, y)
        """
        return self.covfun(x, y)

    def covfun(self, x, y):
        """
        Evaluates covariance kernel :math:`k = k(x, y)`.
        """
        if not self._params:
            return self._covfun(x, y)
        else:
            return self._covfun(x, y, self._params)

    @property
    def params(self):
        """
        Return parameters :math:`\\theta` of the covariance.
        """
        return self._params

    @property
    def parderiv(self):
        """
        Return :math:`\\partial_{\\theta} k_\\theta(x, y)`.

        To evaluate it, call ``self.parderiv.covfun(x, y)``
        or ``self.parderiv(x, y)``.
        """
        return self._parderiv

    @property
    def xderiv(self):
        """
        Return :math:`\\partial_{x_i} k(x, y)` as a Covariance object.

        To evaluate it, call ``self.xderiv.covfun(x, y)``
        or ``self.xderiv(x, y)``.
        """
        return self._xderiv

    @property
    def xidx(self):
        """
        Return index :math:`i` of :math:`\\partial_{x_i} k(x, y)`.
        """
        return self._xidx

    @property
    def yidx(self):
        """
        Return index :math:`j` of :math:`\\partial_{y_j} k(x, y)`.
        """
        return self._yidx

    @property
    def yderiv(self):
        """
        Return :math:`\\partial_{y_j} k(x, y)` as a Covariance object.

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
