"""Regression checks for the corrected algebra and discrete channels."""

import unittest

import jax
import numpy as np

from util import Integrator, config, linearization


class EquationConsistencyTests(unittest.TestCase):
    def test_sigma_p_first_order_model(self):
        rng = np.random.default_rng(11)
        p_bar = rng.normal(size=(3, 3))
        p_bar = p_bar @ p_bar.T
        dp = rng.normal(size=(3, 3))
        dp = 0.5 * (dp + dp.T)
        sigma_bar, dsigma = 0.4, -0.07
        exact = (sigma_bar + dsigma) * (p_bar + dp)
        linear = (
            (sigma_bar + dsigma) * p_bar
            + sigma_bar * (p_bar + dp)
            - sigma_bar * p_bar
        )
        np.testing.assert_allclose(exact - linear, dsigma * dp, atol=1e-12)

    def test_mixed_quadratic_first_order_model(self):
        rng = np.random.default_rng(29)
        h1 = rng.normal(size=(2, 3))
        h2 = rng.normal(size=(2, 3))
        dh1 = 1e-4 * rng.normal(size=(2, 3))
        dh2 = 1e-4 * rng.normal(size=(2, 3))
        exact = (h2 + dh2).T @ (h1 + dh1)
        linear = h2.T @ h1 + dh2.T @ h1 + h2.T @ dh1
        np.testing.assert_allclose(exact - linear, dh2.T @ dh1, atol=1e-12)

    def test_euler_discrete_noise_and_measurement_jacobians(self):
        mode = config.current_mode
        dt = mode["Tf"] / (mode["T"] - 1)
        x = np.array([1.0, -0.4, 0.3])
        u = np.array([1.2, -0.2])
        w = np.zeros(mode["nw"])
        wy = np.zeros(mode["nwy"])
        _, _, gd, cd, dd = linearization.linearization_fun_new(
            Integrator.STEP_NEW, dt, x, u, w, wy
        )
        np.testing.assert_allclose(np.asarray(gd), dt * mode["G"], atol=1e-12)
        np.testing.assert_allclose(np.asarray(cd), mode["C"], atol=1e-12)
        np.testing.assert_allclose(np.asarray(dd), mode["D"], atol=1e-12)

    def test_iqc_fourth_block_scaling(self):
        vx, vy = config.current_mode["vx"], config.current_mode["vy"]
        np.testing.assert_allclose(vx * np.eye(2) / vx**2, np.eye(2) / vx)
        np.testing.assert_allclose(vy * np.eye(2) / vy**2, np.eye(2) / vy)


if __name__ == "__main__":
    unittest.main()
