"""Stellarator plasma source.

Builds a neutron source from a stellarator shape, written as a Fourier-Zernike
series in DESC's conventions, plus the same density and temperature profiles
as the tokamak source, here in rho (the square root of the normalised
toroidal flux) rather than in cm.

No nuclear data is needed: this only samples the source. Drop the returned
list into ``yamc.Model(source=...)`` to transport it.
"""
import math

import yamc

# A rotating ellipse with five field periods, R0 = 600 cm:
#   R = R0 + a rho cos(theta) + e rho cos(theta - N zeta)
#   Z =      a rho sin(theta) - e rho sin(theta - N zeta)
# Each (l, m, n) is a Zernike mode in (rho, theta) times a Fourier mode in
# zeta; m < 0 and n < 0 select sines. A DESC equilibrium's R_basis.modes /
# R_lmn (and Z) can be passed in the same way, times 100 for cm.
a, e = 150.0, 50.0
shape = yamc.sources.StellaratorShape(
    num_field_periods=5,
    r_modes=[(0, 0, 0), (1, 1, 0), (1, 1, 1), (1, -1, -1)],
    r_lmn=[600.0, a, e, e],
    z_modes=[(1, -1, 0), (1, -1, 1), (1, 1, -1)],
    z_lmn=[a, -e, e],
)
print(f"plasma volume: {shape.volume() / 1e6:.2f} m^3")

# The boundary cross-section at the start of a field period and half way
# through it, where the ellipse has turned a quarter turn.
thetas = [2 * math.pi * i / 8 for i in range(8)]
for zeta in (0.0, math.pi / 5):
    radii, heights = shape.evaluate(1.0, thetas, zeta)
    points = ", ".join(f"({r:.0f}, {z:.0f})" for r, z in zip(radii, heights))
    print(f"boundary at zeta = {zeta:.3f}: {points}")

sources = yamc.sources.stellarator_source(
    shape=shape,
    mode="H",
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9e3,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_beta=6.0,
    ion_temperature_pedestal=6.09e3,
    ion_temperature_separatrix=0.1e3,
    pedestal_rho=0.8,
)
print(f"\n{len(sources)} flux-shell sources, strengths summing to "
      f"{sum(source.strength for source in sources):.6f}")

dd_yield = sum(source.strength for source in sources if source.energy.mean < 5e6)
print(f"D-D share of the yield: {dd_yield:.3%}")

# Half the neutrons are born inside this rho, weighted by source strength.
cumulative = 0.0
for source in sorted(sources, key=lambda s: s.position.rho[1]):
    cumulative += source.strength
    if cumulative >= 0.5:
        print(f"half the yield comes from inside rho = {source.position.rho[1]:.2f}")
        break

# Hand the list straight to a model:
#     model = yamc.Model(geometry=geometry, source=sources, tallies=tallies)
