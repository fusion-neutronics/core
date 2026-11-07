"""The stellarator shape, which yani ships as yamc does.

The Rust core tests the fit to VMEC coefficients and yamc's suite the rest of
the Python surface; this checks that yani's copy fits a VMEC equilibrium too.
"""

import math

import pytest

yani = pytest.importorskip("yani", reason="standalone yani wheel not installed")

R0, A, E, NFP, NS = 600.0, 150.0, 50.0, 5, 21


def test_a_rotating_ellipse_from_vmec_is_fitted_exactly():
    # R = R0 + a rho cos(t) + e rho cos(t - N z), Z = a rho sin(t) - e rho
    # sin(t - N z) on the surfaces s = rho**2 = j / (NS - 1), in cm.
    rho = [math.sqrt(j / (NS - 1)) for j in range(NS)]
    shape = yani.sources.StellaratorShape.from_vmec(
        num_field_periods=NFP,
        xm=[0.0, 1.0, 1.0],
        xn=[0.0, 0.0, float(NFP)],
        rmnc=[[R0, A * r, E * r] for r in rho],
        zmns=[[0.0, A * r, -E * r] for r in rho],
    )
    assert shape.fit_error < 1e-9 * A
    assert "fit_error=" in repr(shape)
    theta, zeta = 0.7, 0.2
    r, z = shape.evaluate(0.6, theta, zeta)
    assert r == pytest.approx(R0 + 0.6 * (A * math.cos(theta) + E * math.cos(theta - NFP * zeta)))
    assert z == pytest.approx(0.6 * (A * math.sin(theta) - E * math.sin(theta - NFP * zeta)))
