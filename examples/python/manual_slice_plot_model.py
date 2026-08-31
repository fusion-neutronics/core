"""
Manual model slice plot using matplotlib.

Shows how to sample a model (geometry + source + settings) on a 2D grid and
produce a matplotlib plot with cell/material colour fills, outlines, and the
source location -- the same data that model.plot() uses internally, but giving
the user full control over the figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import yamc

yamc.set_cross_section_data_entry('fendl-3.2d')

# ── Build a simple geometry ──────────────────────────────────────────────
sphere_inner = yamc.Sphere(radius=20.0)
sphere_outer = yamc.Sphere(radius=40.0, boundary="vacuum")

mat_steel = yamc.Material(
    composition={"Fe": 1.0},
    density=7.8,
    name="steel")

mat_water = yamc.Material(
    composition={"H": 2.0, "O": 1.0},
    density=1.0,
    name="water")

cell_inner = yamc.Cell(name="inner", region=sphere_inner.below, material=mat_water)
cell_outer = yamc.Cell(name="outer", region=sphere_inner.above & sphere_outer.below, material=mat_steel)
geometry = yamc.Geometry([cell_inner, cell_outer])

# ── Source + settings ────────────────────────────────────────────────────
source = yamc.NeutronSource(
    energy=14.06e6,
    position=(10, 5, 0),
    strength=1)
model = yamc.Model(geometry=geometry, tallies=[], source=source)

# ── Sample geometry ──────────────────────────────────────────────────────
basis = "xy"
result = geometry.sample_slice(basis=basis, resolution=(300, 300))

# ── Source position projected onto the slice basis ───────────────────────
source_pos = [10, 5, 0]

# ── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: material colour fill with material outlines
unique_mats = sorted(set(np.array(result.material_ids).flat) - {-1})
cmap = ListedColormap(plt.cm.Set2.colors[: len(unique_mats)])
mat_plot = np.where(np.array(result.material_ids) >= 0, result.material_ids, np.nan)
ax = axes[0]
im = ax.pcolormesh(result.h_edges, result.v_edges, mat_plot, cmap=cmap, shading="flat")
outline = np.zeros((*np.array(result.edges("material")).shape, 4))
outline[result.edges("material")] = [0, 0, 0, 1]
ax.imshow(outline, extent=result.extent, origin="lower", aspect="auto",
          interpolation="nearest", zorder=2)
ax.plot(source_pos[0], source_pos[1], marker="*", color="red", markersize=14,
        markeredgecolor="black", markeredgewidth=0.5, zorder=3, label="Source")
ax.legend(loc="upper right")
ax.set_xlabel(f"{result.h_label} [cm]")
ax.set_ylabel(f"{result.v_label} [cm]")
ax.set_title("Materials with outlines")
ax.set_aspect("equal")

# Right: cell ID fill with cell outlines
unique_cells = sorted(set(np.array(result.cell_ids).flat) - {-1})
cmap2 = ListedColormap(plt.cm.tab10.colors[: len(unique_cells)])
cell_plot = np.where(np.array(result.cell_ids) >= 0, result.cell_ids, np.nan)
ax = axes[1]
ax.pcolormesh(result.h_edges, result.v_edges, cell_plot, cmap=cmap2, shading="flat")
outline2 = np.zeros((*np.array(result.edges("cell")).shape, 4))
outline2[result.edges("cell")] = [1, 0, 0, 1]
ax.imshow(outline2, extent=result.extent, origin="lower", aspect="auto",
          interpolation="nearest", zorder=2)
ax.plot(source_pos[0], source_pos[1], marker="*", color="red", markersize=14,
        markeredgecolor="black", markeredgewidth=0.5, zorder=3, label="Source")
ax.legend(loc="upper right")
ax.set_xlabel(f"{result.h_label} [cm]")
ax.set_ylabel(f"{result.v_label} [cm]")
ax.set_title("Cells with outlines")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("manual_model_slice.png", dpi=150)
print("Saved manual_model_slice.png")
plt.show()
