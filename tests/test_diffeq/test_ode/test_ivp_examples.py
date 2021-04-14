import unittest

import numpy as np

from probnum.diffeq.ode import ivp, ivp_examples
from probnum.randvars import Constant
from tests.testing import NumpyAssertions


class TestConvenienceFunction(unittest.TestCase):
    """Test case for correct object initialization."""

    def setUp(self):
        self.tspan = (0.0, 4.212)

    def test_logistic(self):
        """Test the logistic ODE convenience function."""
        rv = Constant(0.1)
        lg1 = ivp_examples.logistic(self.tspan, rv)
        lg2 = ivp_examples.logistic(self.tspan, rv, params=(1.0, 1.0))

        self.assertIsInstance(lg1, ivp.IVP)
        self.assertIsInstance(lg2, ivp.IVP)

    def test_fitzhughnagumo(self):
        """Test the FHN IVP convenience function."""
        rv = Constant(np.ones(2))
        lg1 = ivp_examples.fitzhughnagumo(self.tspan, rv)
        lg2 = ivp_examples.fitzhughnagumo(self.tspan, rv, params=(1.0, 1.0, 1.0, 1.0))

        self.assertIsInstance(lg1, ivp.IVP)
        self.assertIsInstance(lg2, ivp.IVP)

    def test_lotkavolterra(self):
        """Test the LV ODE convenience function."""
        rv = Constant(np.ones(2))
        lg1 = ivp_examples.lotkavolterra(self.tspan, rv)
        lg2 = ivp_examples.lotkavolterra(self.tspan, rv, params=(1.0, 1.0, 1.0, 1.0))

        self.assertIsInstance(lg1, ivp.IVP)
        self.assertIsInstance(lg2, ivp.IVP)

    def test_seir(self):
        """Test the SEIR ODE convenience function."""
        rv = Constant(np.array([1.0, 0.0, 0.0, 0.0]))
        lg1 = ivp_examples.seir(self.tspan, rv)
        lg2 = ivp_examples.seir(self.tspan, rv, params=(1.0, 1.0, 1.0, 1.0))

        self.assertIsInstance(lg1, ivp.IVP)
        self.assertIsInstance(lg2, ivp.IVP)

    def test_lorenz(self):
        """Test the Lorenz model ODE convenience function."""
        rv = Constant(np.array([1.0, 1.0, 1.0]))
        lg1 = ivp_examples.lorenz(self.tspan, rv)
        lg2 = ivp_examples.lorenz(
            self.tspan,
            rv,
            params=(
                10.0,
                28.0,
                8.0 / 3.0,
            ),
        )

        self.assertIsInstance(lg1, ivp.IVP)
        self.assertIsInstance(lg2, ivp.IVP)


class TestRHSEvaluation(unittest.TestCase, NumpyAssertions):
    """Test cases that check the evaluation of IVP vector fields."""

    def setUp(self):
        self.tspan = (0.0, 4.212)

    def test_logistic_rhs(self):
        rv = Constant(0.1)
        lg1 = ivp_examples.logistic(self.tspan, rv)

        self.assertEqual(lg1.rhs(0.1, rv).shape, rv.shape)

    def test_fitzhughnagumo_rhs(self):
        rv = Constant(np.ones(2))
        lg1 = ivp_examples.fitzhughnagumo(self.tspan, rv)

        self.assertEqual(lg1.rhs(0.1, rv).shape, rv.shape)

    def test_lotkavolterra_rhs(self):
        rv = Constant(np.ones(2))
        lg1 = ivp_examples.lotkavolterra(self.tspan, rv)

        self.assertEqual(lg1.rhs(0.1, rv).shape, rv.shape)

    def test_seir_rhs(self):
        rv = Constant(np.ones(4))
        lg1 = ivp_examples.seir(self.tspan, rv)

        self.assertEqual(lg1.rhs(0.1, rv).shape, rv.shape)

    def test_lorenz_rhs(self):
        rv = Constant(np.ones(3))
        lg1 = ivp_examples.lorenz(self.tspan, rv)

        self.assertEqual(lg1.rhs(0.1, rv).shape, rv.shape)


class TestJacobianEvaluation(unittest.TestCase, NumpyAssertions):
    """Test cases that check Jacobians of IVPs against finite differences."""

    def setUp(self):
        self.tspan = (0.0, 4.212)

        self.dt = 1e-6  # should lead to errors ~1e-8 in the present tests
        self.rtol = 1e-4

    def test_logistic_jacobian(self):
        rv = Constant(0.1)
        lg1 = ivp_examples.logistic(self.tspan, rv)
        random_direction = 1 + 0.1 * np.random.rand(lg1.dimension)
        random_point = 1 + np.random.rand(lg1.dimension)
        fd_approx = (
            0.5
            * 1.0
            / self.dt
            * (
                lg1(0.1, random_point + self.dt * random_direction)
                - lg1(0.1, random_point - self.dt * random_direction)
            )
        )
        self.assertAllClose(
            lg1.jacobian(0.1, random_point) @ random_direction,
            fd_approx,
            rtol=self.rtol,
        )

    def test_fitzhughnagumo_jacobian(self):
        rv = Constant(np.ones(2))
        lg1 = ivp_examples.fitzhughnagumo(self.tspan, rv)
        random_direction = 1 + 0.1 * np.random.rand(lg1.dimension)
        random_point = 1 + np.random.rand(lg1.dimension)
        fd_approx = (
            0.5
            * 1.0
            / self.dt
            * (
                lg1(0.1, random_point + self.dt * random_direction)
                - lg1(0.1, random_point - self.dt * random_direction)
            )
        )
        self.assertAllClose(
            lg1.jacobian(0.1, random_point) @ random_direction,
            fd_approx,
            rtol=self.rtol,
        )

    def test_lotkavolterra_jacobian(self):
        rv = Constant(np.ones(2))
        lg1 = ivp_examples.lotkavolterra(self.tspan, rv)
        random_direction = 1 + 0.1 * np.random.rand(lg1.dimension)
        random_point = 1 + np.random.rand(lg1.dimension)
        fd_approx = (
            0.5
            * 1.0
            / self.dt
            * (
                lg1(0.1, random_point + self.dt * random_direction)
                - lg1(0.1, random_point - self.dt * random_direction)
            )
        )
        self.assertAllClose(
            lg1.jacobian(0.1, random_point) @ random_direction,
            fd_approx,
            rtol=self.rtol,
        )

    def test_seir_jacobian(self):
        rv = Constant(np.ones(4))
        lg1 = ivp_examples.seir(self.tspan, rv)
        random_direction = 1 + 0.1 * np.random.rand(lg1.dimension)
        random_point = 1 + np.random.rand(lg1.dimension)
        fd_approx = (
            0.5
            * 1.0
            / self.dt
            * (
                lg1(0.1, random_point + self.dt * random_direction)
                - lg1(0.1, random_point - self.dt * random_direction)
            )
        )
        self.assertAllClose(
            lg1.jacobian(0.1, random_point) @ random_direction,
            fd_approx,
            rtol=self.rtol,
        )

    def test_lorenz_jacobian(self):
        rv = Constant(np.array([1.0, 1.0, 1.0]))
        lg1 = ivp_examples.lorenz(self.tspan, rv)
        random_direction = 1 + 0.1 * np.random.rand(lg1.dimension)
        random_point = 1 + np.random.rand(lg1.dimension)
        fd_approx = (
            0.5
            * 1.0
            / self.dt
            * (
                lg1(0.1, random_point + self.dt * random_direction)
                - lg1(0.1, random_point - self.dt * random_direction)
            )
        )
        self.assertAllClose(
            lg1.jacobian(0.1, random_point) @ random_direction,
            fd_approx,
            rtol=self.rtol,
        )
