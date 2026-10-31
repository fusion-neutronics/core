#!/usr/bin/env python3
"""Generate a large benchmark Arrow mesh with many nested sphere shells.

Creates 14 concentric sphere shells, all tet-meshed at high resolution,
producing a ~100-200MB Arrow file suitable for benchmarking read/init
performance in yamt.

Expected output (with default settings):
  ~495K vertices, ~611K triangles, ~1.66M tetrahedra, ~194MB Arrow file

Usage (from the yamc repo root):
    python crates/yamt/tests/generate_benchmark_nested_spheres.py [output_path]

Requirements:
    pip install cadquery yamc[cad]
"""

import os
import sys
import time

import cadquery as cq
from yamc.cad import CadToYamc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_SHELLS = 14
RADII = [float(r) for r in range(1, NUM_SHELLS + 2)]  # [1.0, 2.0, ..., 15.0]
TARGET_EDGE_LENGTH = 0.8  # fine resolution for a large mesh
TOLERANCE = 0.05  # surface mesh chordal deflection
ANGULAR_TOLERANCE = 0.05  # surface mesh angular deflection (radians)

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "data", "nested_spheres_benchmark.arrow"
)


def build_nested_spheres():
    """Build a CadQuery assembly of 14 concentric hollow sphere shells."""
    spheres = [cq.Workplane("XY").sphere(r) for r in RADII]

    # Build hollow shells: shell_i = sphere_{i+1} - sphere_i
    shells = []
    names = []
    for i in range(NUM_SHELLS):
        shell = spheres[i + 1].cut(spheres[i])
        shells.append(shell)
        names.append(f"shell_{i}")

    assy = cq.Assembly()
    for shell, name in zip(shells, names):
        assy.add(shell, name=name)

    return assy, names


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT

    print(f"Building {NUM_SHELLS} nested sphere shells (radii {RADII[0]}-{RADII[-1]} cm)")
    print(f"Target edge length: {TARGET_EDGE_LENGTH}")
    print()

    t0 = time.time()
    assy, names = build_nested_spheres()
    t_cad = time.time() - t0
    print(f"CAD geometry built in {t_cad:.1f}s")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags="assembly_names")

    t1 = time.time()
    mesh = c2y.mesh(
        tet_volumes=names,  # tet-mesh ALL shells
        target_edge_length=TARGET_EDGE_LENGTH,
        tolerance=TOLERANCE,
        angular_tolerance=ANGULAR_TOLERANCE,
    )
    t_mesh = time.time() - t1
    print(f"Meshing completed in {t_mesh:.1f}s")
    print(f"  Vertices:  {len(mesh.vertices)}")
    print(f"  Triangles: {len(mesh.triangles)}")

    n_tets = sum(len(tt) for _, (_, tt) in c2y._tet_data.items())
    print(f"  Tetrahedra: {n_tets}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    t2 = time.time()
    c2y.to_arrow(output_path)
    t_export = time.time() - t2
    print(f"Arrow export in {t_export:.1f}s")

    file_size = os.path.getsize(output_path)
    print(f"\nWrote {output_path}")
    print(f"  File size: {file_size / 1e6:.1f} MB")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
