"""

"""


class Kernel:
    """
    (Parameterized) kernel functions.

    Functions of the form :math:`k = k(x, y)[=k_p(x, y)]`
    and, if available, their derivatives with respect to
    spatial variables :math:`x, y` and the parameters :math:`p`.

    Parameters
    ----------
    kernfun : callable, signature=``(x, y)``
    params : list
        Parameters of the kernels. If not empty, calls to the
        evaluation of the kernels objects are done via
        ``self._kernfun(x, y, self._params)``, i.e. in this case
        the object expects kernels function signature of
        ``(x, y, params)``.
    parderiv : callable or Kernel
        If callable, a new Kernel object is created without
        derivatives.
    xderiv : callable or Kernel
        (Partial) derivative of the :math:`x` coordinate. If the
        kernels acts on higher dimensions, it is understood as
        the gradient; that is, a function that returns a list of
        evaluations at each evaluation. It can be taken as partial
        derivative w.r.t. the :math:`y_j` coordinate if the ` yidx`
        parameter is set and `yderiv` evalautes to a scalar.
        If callable, a new Kernel object is created without
        derivatives.
    xidx : int, optional.
        Index of the derivative stored in `xderiv`. Only important
        in higher dimensions and if the derivatives are partial derivatives,
        not full gradients. Relevant for implementation of
        kernels arithmetics.
    yderiv : callable or Kernel
        (Partial) derivative of the :math:`y` coordinate. If the
        kernels acts on higher dimensions, it is understood as
        the gradient; that is, a function that returns a list of
        evaluations at each evaluation. It can be taken as partial
        derivative w.r.t. the :math:`y_j` coordinate if the ` yidx`
        parameter is set and `yderiv` evalautes to a scalar.
        If callable, a new Kernel object is created without
        derivatives.
    yidx : int, optional.
        Index of the derivative stored in `yderiv`. Only important
        in higher dimensions and if the derivatives are partial derivatives,
        not full gradients. Relevant for implementation of
        kernels arithmetics.

    Examples
    --------

    To initialize a Kernel object just pass a function with
    signature ``(x, y)`` to the ``kernfun`` key.

    >>> from probnum.prob.randomprocess import Kernel
    >>> kern = Kernel(kernfun=(lambda x, y: 1))

    It can be evaluated with ``self.kernfun(x, y)``

    >>> print(kern.kernfun(3, 4))
    1

    But it is also possible to call the object directly

    >>> print(kern(3, 4))
    1

    which works because ``__call__(self, x, y)``
    calls ``self.kernfun(x, y)``.

    One thing that this object can do that generic functions cannot
    do is know derivatives

    >>> kern2 = Kernel(kernfun=(lambda x, y: x**2 + y**2),
    ...                xderiv=(lambda x, y: 2*x),
    ...                yderiv=(lambda x, y: 2*y))
    >>> kern2.kernfun(3, 4)
    25
    >>> kern2(3, 4)
    25
    >>> kern2.xderiv(3, 4)
    6
    >>> kern2.yderiv(3, 4)
    8

    or that know explicit parameterizations

    >>> kern3 = Kernel(kernfun=(lambda x, y, p: p*(x**2 + y)), params=3,
    ...                parderiv=(lambda x, y, p: x**2 + y))
    >>> kern3(3, 4)
    39
    >>> kern3.parderiv(3, 4)
    13

    >>> kern4 = Kernel(kernfun=(lambda x, y, p: p[0]*(x**2 + p[1]*y)),
    ...                params=[1, 2],
    ...                parderiv=(lambda x, y, p: [x**2 + p[1]*y, p[0]*y]))
    >>> kern4(3, 4)
    17
    >>> kern4.parderiv(3, 4)
    [17, 4]

    The second example also shows how to implement vector-valued
    kernels which can be used for computing full gradients.

    TODO
    ----
    Implement dunder methods to create new kernels kernels out of
    old ones.
    """

    def __init__(self, kernfun=None, params=None, parderiv=None,
                 xderiv=None, xidx=None, yderiv=None, yidx=None):
        """
        """
        self._kernfun = kernfun
        if params is None:
            params = []
        if parderiv is not None and not issubclass(type(parderiv), Kernel):
            parderiv = Kernel(kernfun=parderiv, params=params)
        if xderiv is not None and not issubclass(type(xderiv), Kernel):
            xderiv = Kernel(kernfun=xderiv, params=params)
        if yderiv is not None and not issubclass(type(yderiv), Kernel):
            yderiv = Kernel(kernfun=yderiv, params=params)
        self._params = params
        self._parderiv = parderiv
        self._xderiv = xderiv
        self._xidx = xidx
        self._yderiv = yderiv
        self._yidx = yidx

    def __call__(self, x, y):
        """
        Proxy for self.kernfun(x, y)
        """
        return self.kernfun(x, y)

    def kernfun(self, x, y):
        """
        Evaluates kernels kernel :math:`k = k(x, y)`.
        """
        if not self._params:
            return self._kernfun(x, y)
        else:
            return self._kernfun(x, y, self._params)

    @property
    def params(self):
        """
        Return parameters :math:`\\theta` of the kernels.
        """
        return self._params

    @property
    def parderiv(self):
        """
        Return :math:`\\partial_{\\theta} k_\\theta(x, y)`.

        To evaluate it, call ``self.parderiv.kernfun(x, y)``
        or ``self.parderiv(x, y)``.
        """
        return self._parderiv

    @property
    def xderiv(self):
        """
        Return :math:`\\partial_{x_i} k(x, y)` as a Kernel object.

        To evaluate it, call ``self.xderiv.kernfun(x, y)``
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
        Return :math:`\\partial_{y_j} k(x, y)` as a Kernel object.

        To evaluate it, call ``self.yderiv.kernfun(x, y)``.
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
    # Be careful with the derivatives. #################################

    def __add__(self, other):
        """ """
        return NotImplemented

    def __sub__(self, other):
        """ """
        return NotImplemented

    def __mul__(self, other):
        """ """
        return NotImplemented

    def __radd__(self, other):
        """ """
        return NotImplemented

    def __rsub__(self, other):
        """ """
        return NotImplemented

    def __rmul__(self, other):
        """ """
        return NotImplemented

    # todo: how does concatenation work? k(x, y) = k1(x, k2(x, y))