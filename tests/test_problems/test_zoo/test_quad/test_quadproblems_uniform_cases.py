import numpy as np
import pytest

from probnum.problems.zoo.quad import (
    bratley1992,
    genz_continuous,
    genz_cornerpeak,
    genz_discontinuous,
    genz_gaussian,
    genz_oscillatory,
    genz_productpeak,
    gfunction,
    morokoff_caflisch_1,
    morokoff_caflisch_2,
    roos_arnold,
)

genz_params = [
    (None, None, 3),
    (5, 0.5, 1),
    (5.0, 1.0, 1),
    (2.0, 0.8, 1),
    (5, 0.5, 4),
    (5.0, 1.0, 4),
    (2.0, 0.8, 4),
]


class GenzUniformCases:
    @pytest.mark.parametrize("a, u, dim", genz_params)
    def case_genz_continuous(self, dim, a, u):
        a_vec = np.repeat(a, dim) if a is not None else None
        u_vec = np.repeat(u, dim) if u is not None else None
        quadprob = genz_continuous(dim=dim, a=a_vec, u=u_vec)
        rtol = 5e-2
        return quadprob, rtol

    @pytest.mark.parametrize("a, u, dim", genz_params)
    def case_genz_cornerpeak(self, dim, a, u):
        a_vec = np.repeat(a, dim) if a is not None else None
        u_vec = np.repeat(u, dim) if u is not None else None
        quadprob = genz_cornerpeak(dim=dim, a=a_vec, u=u_vec)
        rtol = 5e-2
        return quadprob, rtol

    @pytest.mark.parametrize("a, u, dim", genz_params)
    def case_genz_discontinuous(self, dim, a, u):
        a_vec = np.repeat(a, dim) if a is not None else None
        u_vec = np.repeat(u, dim) if u is not None else None
        quadprob = genz_discontinuous(dim=dim, a=a_vec, u=u_vec)
        rtol = 5e-2
        return quadprob, rtol

    @pytest.mark.parametrize("a, u, dim", genz_params)
    def case_genz_gaussian(self, dim, a, u):
        a_vec = np.repeat(a, dim) if a is not None else None
        u_vec = np.repeat(u, dim) if u is not None else None
        quadprob = genz_gaussian(dim=dim, a=a_vec, u=u_vec)
        rtol = 9e-2
        return quadprob, rtol

    @pytest.mark.parametrize("a, u, dim", genz_params)
    def case_genz_oscillatory(self, dim, a, u):
        a_vec = np.repeat(a, dim) if a is not None else None
        u_vec = np.repeat(u, dim) if u is not None else None
        quadprob = genz_oscillatory(dim=dim, a=a_vec, u=u_vec)
        rtol = 5e-1
        return quadprob, rtol

    @pytest.mark.parametrize("a, u, dim", genz_params)
    def case_genz_productpeak(self, dim, a, u):
        a_vec = np.repeat(a, dim) if a is not None else None
        u_vec = np.repeat(u, dim) if u is not None else None
        quadprob = genz_productpeak(dim=dim, a=a_vec, u=u_vec)
        rtol = 5e-2
        return quadprob, rtol


dim_params = [1, 2, 10]


class OtherIntegrandsUniformCases:
    @pytest.mark.parametrize("dim", dim_params)
    def case_bratley1992(self, dim):
        quadprob = bratley1992(dim=dim)
        rtol = 1e-2
        return quadprob, rtol

    @pytest.mark.parametrize("dim", dim_params)
    def case_roos_arnold(self, dim):
        quadprob = roos_arnold(dim=dim)
        rtol = 1e-2
        return quadprob, rtol

    @pytest.mark.parametrize("dim", dim_params)
    def case_gfunction(self, dim):
        quadprob = gfunction(dim=dim)
        rtol = 1e-2
        return quadprob, rtol

    @pytest.mark.parametrize("dim", dim_params)
    def case_morokoff_caflisch_1(self, dim):
        quadprob = morokoff_caflisch_1(dim=dim)
        rtol = 1e-2
        return quadprob, rtol

    @pytest.mark.parametrize("dim", dim_params)
    def case_morokoff_caflisch_2(self, dim):
        quadprob = morokoff_caflisch_2(dim=dim)
        rtol = 1e-2
        return quadprob, rtol
