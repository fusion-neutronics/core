"""
Manual mesh-tally heatmap with geometry outline overlay using matplotlib.

Shows how to:
1. Extract a 2D mesh tally slice (MeshSliceData with edges and labels)
2. Sample geometry for smooth outlines (GeometrySliceData with edges())
3. Overlay the outline on the tally heatmap

This is the manual equivalent of tally.plot().
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yamc

yamc.set_cross_section_data_entry('fendl-3.2d')

# ── Build geometry ───────────────────────────────────────────────────────
sphere = yamc.Sphere(radius=50.0, boundary="vacuum")
material = yamc.Material(
    composition={"H1": 1.0},
    density=0.001,
    name="hydrogen")
cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
geometry = yamc.Geometry([cell])

# ── Source + settings ────────────────────────────────────────────────────
source = [
    yamc.NeutronSource(
        energy=14.06e6,
        position=(-20, 0, 0),
        strength=1),
]
# ── Mesh tally ───────────────────────────────────────────────────────────
mesh = yamc.RegularRectangularMesh.from_domain(geometry, shape=1000)
tally = yamc.Tally(scores=["flux"], mesh=mesh, name="flux")
model = yamc.Model(geometry=geometry, tallies=[tally], source=source)

print("Running simulation...")
model.simulate_transport(total_particles=100000, seed=42)
print("Done.\n")

# ═══════════════════════════════════════════════════════════════════════════
# MANUAL PLOTTING STARTS HERE
# ═══════════════════════════════════════════════════════════════════════════

basis = "xy"

# ── Step 1: Extract tally slice -- h_edges, v_edges, labels included ─────
mean = tally.extract_mesh_slice(basis=basis, value="mean")
std = tally.extract_mesh_slice(basis=basis, value="standard_deviation")
mean_2d = np.array(mean)
std_2d = np.array(std)
print(f"Slice shape: {mean_2d.shape}")

# ── Step 2: Sample geometry for smooth outlines ─────────────────────────
geom_slice = geometry.sample_slice(basis=basis, resolution=(400, 400))
edge_mask = np.array(geom_slice.edges("material"))

# ── Step 3: Plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: mean flux
ax = axes[0]
vmin = mean_2d[mean_2d > 0].min() if np.any(mean_2d > 0) else 1e-20
vmax = mean_2d.max()
im = ax.pcolormesh(mean.h_edges, mean.v_edges, mean_2d,
                   cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax), shading="flat")
outline_rgba = np.zeros((*edge_mask.shape, 4))
outline_rgba[edge_mask] = [1, 1, 1, 0.8]
ax.imshow(outline_rgba, extent=geom_slice.extent, origin="lower",
          aspect="auto", interpolation="nearest", zorder=2)
cb = fig.colorbar(im, ax=ax)
cb.set_label("Flux [n/cm\u00b2/src]")
ax.set_xlabel(f"{mean.h_label} [cm]")
ax.set_ylabel(f"{mean.v_label} [cm]")
ax.set_title(f"Mean flux \u2014 {basis.upper()}")
ax.set_aspect("equal")

# Right: relative error
rel_err = np.divide(std_2d, mean_2d, out=np.zeros_like(mean_2d), where=mean_2d > 0)
ax = axes[1]
im2 = ax.pcolormesh(std.h_edges, std.v_edges, rel_err,
                    cmap="RdYlGn_r", vmin=0, vmax=1, shading="flat")
outline_rgba2 = np.zeros((*edge_mask.shape, 4))
outline_rgba2[edge_mask] = [0, 0, 0, 0.8]
ax.imshow(outline_rgba2, extent=geom_slice.extent, origin="lower",
          aspect="auto", interpolation="nearest", zorder=2)
cb2 = fig.colorbar(im2, ax=ax)
cb2.set_label("Relative error")
ax.set_xlabel(f"{std.h_label} [cm]")
ax.set_ylabel(f"{std.v_label} [cm]")
ax.set_title(f"Relative error \u2014 {basis.upper()}")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("manual_mesh_tally_with_geometry.png", dpi=150)
print("Saved manual_mesh_tally_with_geometry.png")
plt.show()
