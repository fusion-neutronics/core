"""CSG-vs-mesh transport agreement (mesh-independence).

A solid meshed via the yamm scene mesher must transport equivalently to the
analytic CSG solid. The sphere is the key guard: it meshes via the
structured-grid fallback, whose seam/pole vertices are coincident but distinct.
Without welding them into shared indices the surface leaks in ray-fire (~66% of
the flux escaped before the fix); this test pins that the mesh sphere now agrees
with CSG to within statistics.
"""

import os
import tempfile

import numpy as np
import pytest
import yamc

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc  # noqa: E402


def _material():
    m = yamc.Material(
        composition={"Be9": 1.0}, density=1.85, name="beryllium", temperature=294
    )
    m.read_nuclear_data({"Be9": "tests/Be9.arrow"})
    return m


def _total_flux(geometry, source):
    tally = yamc.Tally(scores=["flux"], name="flux")
    model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
    results = model.simulate_transport(total_particles=50_000, seed=42)
    return float(np.sum(results[tally].mean))


def test_sphere_mesh_matches_csg():
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(0, 0, 0)
    )

    # Analytic CSG sphere.
    s = yamc.Sphere(radius=10, boundary="vacuum")
    csg = yamc.Geometry([yamc.Cell(name="s", region=s.below, material=_material())])
    f_csg = _total_flux(csg, source)

    # Same sphere via CadQuery -> yamm scene mesher -> Arrow -> MeshGeometry.
    asm = cq.Assembly()
    asm.add(cq.Workplane().sphere(10), name="beryllium")
    c2y = CadToYamc()
    c2y.add_cadquery_object(asm, ["beryllium"])
    c2y.mesh()
    path = os.path.join(tempfile.mkdtemp(), "sphere.arrow")
    c2y.to_arrow(path)
    f_mesh = _total_flux(
        yamc.MeshGeometry(path, {"beryllium": _material()}), source
    )

    assert f_csg > 0.0 and f_mesh > 0.0
    rel = abs(f_mesh - f_csg) / f_csg
    assert rel < 0.05, (
        f"mesh sphere flux {f_mesh:.4e} differs from CSG {f_csg:.4e} "
        f"by {rel * 100:.1f}% (surface not watertight for ray-fire?)"
    )
