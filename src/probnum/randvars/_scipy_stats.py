"""Wrapper classes for SciPy random variables."""

from typing import Any, Dict, Union

import numpy as np
import scipy.stats

from probnum import utils as _utils

from . import _normal, _random_variable

_ValueType = Union[np.generic, np.ndarray]

# pylint: disable=protected-access


class _SciPyRandomVariableMixin:
    """Mix-in class for SciPy random variable wrappers."""

    @property
    def scipy_rv(self):
        """SciPy random variable."""
        return self._scipy_rv


class WrappedSciPyRandomVariable(
    _SciPyRandomVariableMixin, _random_variable.RandomVariable[_ValueType]
):
    """Wrapper for SciPy random variable objects.

    Parameters
    ----------
    scipy_rv
        SciPy random variable.
    """

    def __init__(
        self,
        scipy_rv: Union[
            scipy.stats._distn_infrastructure.rv_frozen,
            scipy.stats._multivariate.multi_rv_frozen,
        ],
    ):
        self._scipy_rv = scipy_rv

        super().__init__(**_rv_init_kwargs_from_scipy_rv(scipy_rv))


class WrappedSciPyDiscreteRandomVariable(
    _SciPyRandomVariableMixin, _random_variable.DiscreteRandomVariable[_ValueType]
):
    """Wrapper for discrete SciPy random variable objects.

    Parameters
    ----------
    scipy_rv
        Discrete SciPy random variable.
    """

    def __init__(
        self,
        scipy_rv: Union[
            scipy.stats._distn_infrastructure.rv_frozen,
            scipy.stats._multivariate.multi_rv_frozen,
        ],
    ):
        if isinstance(scipy_rv, scipy.stats._distn_infrastructure.rv_frozen):
            if not isinstance(scipy_rv.dist, scipy.stats.rv_discrete):
                raise ValueError("The given SciPy random variable is not discrete.")

        self._scipy_rv = scipy_rv

        rv_kwargs = _rv_init_kwargs_from_scipy_rv(scipy_rv)

        rv_kwargs["pmf"] = _return_numpy(
            getattr(scipy_rv, "pmf", None),
            dtype=np.float_,
        )

        rv_kwargs["logpmf"] = _return_numpy(
            getattr(scipy_rv, "logpmf", None),
            dtype=np.float_,
        )

        super().__init__(**rv_kwargs)


class WrappedSciPyContinuousRandomVariable(
    _SciPyRandomVariableMixin, _random_variable.ContinuousRandomVariable[_ValueType]
):
    """Wrapper for continuous SciPy random variable objects.

    Parameters
    ----------
    scipy_rv
        Continuous SciPy random variable.
    """

    def __init__(
        self,
        scipy_rv: Union[
            scipy.stats._distn_infrastructure.rv_frozen,
            scipy.stats._multivariate.multi_rv_frozen,
        ],
    ):
        if isinstance(scipy_rv, scipy.stats._distn_infrastructure.rv_frozen):
            if not isinstance(scipy_rv.dist, scipy.stats.rv_continuous):
                raise ValueError("The given SciPy random variable is not continuous.")

        self._scipy_rv = scipy_rv

        rv_kwargs = _rv_init_kwargs_from_scipy_rv(scipy_rv)

        rv_kwargs["pdf"] = _return_numpy(
            getattr(scipy_rv, "pdf", None),
            dtype=np.float_,
        )

        rv_kwargs["logpdf"] = _return_numpy(
            getattr(scipy_rv, "logpdf", None),
            dtype=np.float_,
        )

        super().__init__(**rv_kwargs)


def wrap_scipy_rv(
    scipy_rv: Union[
        scipy.stats._distn_infrastructure.rv_frozen,
        scipy.stats._multivariate.multi_rv_frozen,
    ]
) -> _random_variable.RandomVariable:
    """Transform SciPy distributions to ProbNum :class:`RandomVariable`s.

    Parameters
    ----------
    scipy_rv :
        SciPy random variable.
    """

    # pylint: disable=too-many-return-statements

    # Random variables with concrete implementations in ProbNum
    if isinstance(scipy_rv, scipy.stats._distn_infrastructure.rv_frozen):
        # Univariate distributions
        if scipy_rv.dist.name == "norm":
            # Normal distribution
            return _normal.Normal(
                mean=scipy_rv.mean(),
                cov=scipy_rv.var(),
            )
    elif isinstance(scipy_rv, scipy.stats._multivariate.multi_rv_frozen):
        # Multivariate distributions
        if scipy_rv.__class__.__name__ == "multivariate_normal_frozen":
            # Multivariate normal distribution
            return _normal.Normal(
                mean=scipy_rv.mean,
                cov=scipy_rv.cov,
            )

    # Generic random variables
    if isinstance(scipy_rv, scipy.stats._distn_infrastructure.rv_frozen):
        if isinstance(scipy_rv.dist, scipy.stats.rv_discrete):
            return WrappedSciPyDiscreteRandomVariable(scipy_rv)
        elif isinstance(scipy_rv.dist, scipy.stats.rv_continuous):
            return WrappedSciPyContinuousRandomVariable(scipy_rv)
        else:
            assert isinstance(scipy_rv.dist, scipy.stats.rv_generic)

            return WrappedSciPyRandomVariable(scipy_rv)
    elif isinstance(scipy_rv, scipy.stats._multivariate.multi_rv_frozen):
        has_pmf = hasattr(scipy_rv, "pmf") or hasattr(scipy_rv, "logpmf")
        has_pdf = hasattr(scipy_rv, "pdf") or hasattr(scipy_rv, "logpdf")

        if has_pdf and has_pmf:
            return WrappedSciPyRandomVariable(scipy_rv)
        elif has_pmf:
            return WrappedSciPyDiscreteRandomVariable(scipy_rv)
        elif has_pdf:
            return WrappedSciPyContinuousRandomVariable(scipy_rv)
        else:
            assert not has_pmf and not has_pdf

            return WrappedSciPyRandomVariable(scipy_rv)

    raise ValueError(f"Unsupported argument type {type(scipy_rv)}")


def _rv_init_kwargs_from_scipy_rv(
    scipy_rv: Union[
        scipy.stats._distn_infrastructure.rv_frozen,
        scipy.stats._multivariate.multi_rv_frozen,
    ],
) -> Dict[str, Any]:
    """Create dictionary of random variable properties from a Scipy random variable.

    Parameters
    ----------
    scipy_rv
        SciPy random variable.
    """
    # Infer shape and dtype
    sample = _return_numpy(scipy_rv.rvs)()

    shape = sample.shape
    dtype = sample.dtype

    median_dtype = np.promote_types(dtype, np.float_)
    moments_dtype = np.promote_types(dtype, np.float_)

    # Support of univariate random variables
    if isinstance(scipy_rv, scipy.stats._distn_infrastructure.rv_frozen):

        def in_support(x):
            low, high = scipy_rv.support()

            return bool(low <= x <= high)

    else:
        in_support = None

    def sample_from_scipy_rv(rng, size):
        return scipy_rv.rvs(size=size, random_state=rng)

    if hasattr(scipy_rv, "rvs"):
        sample_wrapper = sample_from_scipy_rv
    else:
        sample_wrapper = None

    return {
        "shape": shape,
        "dtype": dtype,
        "sample": _return_numpy(sample_wrapper, dtype),
        "in_support": in_support,
        "cdf": _return_numpy(getattr(scipy_rv, "cdf", None), np.float_),
        "logcdf": _return_numpy(getattr(scipy_rv, "logcdf", None), np.float_),
        "quantile": _return_numpy(getattr(scipy_rv, "ppf", None), dtype),
        "mode": None,  # not offered by scipy.stats
        "median": _return_numpy(getattr(scipy_rv, "median", None), median_dtype),
        "mean": _return_numpy(getattr(scipy_rv, "mean", None), moments_dtype),
        "cov": _return_numpy(getattr(scipy_rv, "cov", None), moments_dtype),
        "var": _return_numpy(getattr(scipy_rv, "var", None), moments_dtype),
        "std": _return_numpy(getattr(scipy_rv, "std", None), moments_dtype),
        "entropy": _return_numpy(getattr(scipy_rv, "entropy", None), np.float_),
    }


def _return_numpy(fun, dtype=None):
    if fun is None:
        return None

    def _wrapper(*args, **kwargs):
        res = fun(*args, **kwargs)

        if np.isscalar(res):
            return _utils.as_numpy_scalar(res, dtype=dtype)

        return np.asarray(res, dtype=dtype)

    return _wrapper
