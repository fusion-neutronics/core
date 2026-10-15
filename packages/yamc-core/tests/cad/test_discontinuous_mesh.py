"""Meshing without imprinting, and transporting the result.

``CadToYamc.mesh(imprint=False)`` meshes every solid on its own, so two solids
that touch meet along two separately faceted surfaces. Conformal tracking
cannot cross such an interface (the far side of every surface reads as the
implicit complement): on nested cylinders it lost 99 percent of the middle
shell's flux without losing a single particle. The file records that it was
not imprinted, which makes :class:`yamc.MeshGeometry` track it
discontinuously. These tests pin that the record is written and read, and that
the transported answer matches the imprinted mesh of the same CAD.
"""

import math
import os

import pytest
import yamc
from yamc._core import mesh_to_arrow

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc  # noqa: E402

TOLERANCE = 0.1


def _nested_cylinders(rotate_middle=7.0, rotate_outer=13.0):
    """Cylinder inside two annuli, all 20 tall. Rotating the annuli moves
    their seams, so each shared cylindrical face is faceted differently on
    either side and the two facetings cross (overlap and gap slivers)."""
    inner = cq.Workplane().circle(5).extrude(20).translate((0, 0, -10))
    middle = (
        cq.Workplane().circle(10).circle(5).extrude(20).translate((0, 0, -10))
        .rotate((0, 0, 0), (0, 0, 1), rotate_middle)
    )
    outer = (
        cq.Workplane().circle(15).circle(10).extrude(20).translate((0, 0, -10))
        .rotate((0, 0, 0), (0, 0, 1), rotate_outer)
    )
    assembly = cq.Assembly()
    assembly.add(inner, name="a")
    assembly.add(middle, name="b")
    assembly.add(outer, name="c")
    return assembly


def _materials():
    out = {}
    for tag, (nuclide, density) in {
        "a": ("Be9", 1.85),
        "b": ("Fe56", 7.874),
        "c": ("Pb208", 11.35),
    }.items():
        m = yamc.Material(
            composition={nuclide: 1.0}, density=density, name=tag, temperature=294
        )
        m.read_nuclear_data({nuclide: f"tests/{nuclide}.arrow"})
        out[tag] = m
    return out


def _export(tmp_path, imprint, name):
    c2y = CadToYamc()
    c2y.add_cadquery_object(_nested_cylinders(), material_tags=["a", "b", "c"])
    mesh = c2y.mesh(tolerance=TOLERANCE, imprint=imprint)
    path = str(tmp_path / f"{name}.arrow")
    c2y.to_arrow(path)
    return c2y, mesh, path


def _fluxes(path, seed, particles=20_000):
    mats = _materials()
    geometry = yamc.MeshGeometry(path, mats, graveyard_offset=1.0)
    tallies = {t: yamc.Tally(scores=["flux"], materials=m, name=t) for t, m in mats.items()}
    model = yamc.Model(
        geometry=geometry,
        tallies=list(tallies.values()),
        source=yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(0.0, 0.0, 0.0)
        ),
    )
    results = model.simulate_transport(total_particles=particles, seed=seed)
    return geometry, model, {
        t: (float(results[tally].mean[0]), float(results[tally].standard_deviation[0]))
        for t, tally in tallies.items()
    }


def test_imprint_false_shares_no_faces(tmp_path):
    _, imprinted, _ = _export(tmp_path, True, "imp")
    _, separate, _ = _export(tmp_path, False, "sep")
    assert imprinted.shared_faces, "the imprinted cylinders share their interfaces"
    assert not separate.shared_faces, "without imprinting no face is shared"
    assert separate.num_solids == imprinted.num_solids == 3


def test_the_file_selects_discontinuous_tracking(tmp_path):
    _, _, imp = _export(tmp_path, True, "imp")
    _, _, sep = _export(tmp_path, False, "sep")
    mats = _materials()

    conformal = yamc.MeshGeometry(imp, mats)
    assert not conformal.discontinuous
    assert conformal.contact_tolerance is None

    discontinuous = yamc.MeshGeometry(sep, mats, graveyard_offset=1.0)
    assert discontinuous.discontinuous
    assert discontinuous.contact_tolerance == pytest.approx(2 * TOLERANCE)


def test_transport_matches_the_imprinted_mesh(tmp_path):
    """Same CAD, imprinted or not, independent seeds: same fluxes."""
    _, _, imp = _export(tmp_path, True, "imp")
    _, _, sep = _export(tmp_path, False, "sep")
    _, _, reference = _fluxes(imp, seed=1)
    _, model, flux = _fluxes(sep, seed=2)

    assert len(model.lost_particles) == 0
    for tag, (mean, std) in flux.items():
        ref_mean, ref_std = reference[tag]
        z = (mean - ref_mean) / math.hypot(std, ref_std)
        assert abs(z) < 4.0, (
            f"{tag}: {mean:.5e} vs imprinted {ref_mean:.5e} ({z:+.1f} sigma)"
        )


def test_exterior_surface_ids_refuses_a_non_imprinted_mesh(tmp_path):
    c2y, _, _ = _export(tmp_path, False, "sep")
    with pytest.raises(RuntimeError, match="graveyard_offset"):
        c2y.exterior_surface_ids()


def _box(x0, x1, y0, y1, z0, z1):
    v = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 4, 7, 3), (1, 2, 6, 5)]
    t = []
    for a, b, c, d in quads:
        t += [[a, b, c], [a, c, d]]
    return v, t


def _write_touching_boxes(path, **provenance):
    """Two 10 cm cubes meeting at x = 10, each with its own surface."""
    va, ta = _box(0, 10, 0, 10, 0, 10)
    vb, tb = _box(10, 20, 0, 10, 0, 10)
    vertices = [c for v in va + vb for c in v]
    triangles = [i for t in ta for i in t] + [i + 8 for t in tb for i in t]
    n = len(ta)
    mesh_to_arrow(
        path,
        vertices,
        triangles,
        [1] * n + [2] * n,
        [1] * n + [2] * n,
        physical_groups_json=(
            '{"1": {"name": "mat:a", "dim": 3}, "2": {"name": "mat:b", "dim": 3}}'
        ),
        surface_volumes_json="[[0, null], [1, null]]",
        **provenance,
    )


def _two_box_fluxes(path, **kwargs):
    mats = {}
    for tag in "ab":
        m = yamc.Material(composition={"Be9": 1.0}, density=1.85, name=tag, temperature=294)
        m.read_nuclear_data({"Be9": "tests/Be9.arrow"})
        mats[tag] = m
    geometry = yamc.MeshGeometry(path, mats, graveyard_offset=1.0, **kwargs)
    tallies = [yamc.Tally(scores=["flux"], materials=m, name=t) for t, m in mats.items()]
    model = yamc.Model(
        geometry=geometry,
        tallies=tallies,
        source=yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(5.0, 5.0, 5.0)
        ),
    )
    results = model.simulate_transport(total_particles=5_000, seed=3)
    return geometry, [float(results[t].mean[0]) for t in tallies]


def test_contact_tolerance_tracks_a_mesh_from_another_writer(tmp_path):
    """A file that does not record its provenance is tracked conformally,
    which cannot reach cube B through the unshared interface; passing a
    contact tolerance tracks it discontinuously and B is reached."""
    path = str(tmp_path / "boxes.arrow")
    _write_touching_boxes(path)

    conformal, (_, b_conformal) = _two_box_fluxes(path)
    assert not conformal.discontinuous

    forced, (a, b) = _two_box_fluxes(path, contact_tolerance=1e-3)
    assert forced.discontinuous
    assert forced.contact_tolerance == pytest.approx(1e-3)
    assert b > 10 * b_conformal, (b, b_conformal)

    recorded = str(tmp_path / "recorded.arrow")
    _write_touching_boxes(recorded, imprinted=False, contact_tolerance=1e-3)
    from_file, (a2, b2) = _two_box_fluxes(recorded)
    assert from_file.discontinuous
    # Same tracking either way; only the threaded tally reduction may round
    # differently in the last bit.
    assert (a2, b2) == (pytest.approx(a, rel=1e-12), pytest.approx(b, rel=1e-12))


def test_contact_tolerance_needs_a_non_imprinted_mesh(tmp_path):
    with pytest.raises(ValueError, match="imprinted=False"):
        _write_touching_boxes(str(tmp_path / "x.arrow"), contact_tolerance=1e-3)
    with pytest.raises(ValueError, match="non-negative"):
        _write_touching_boxes(
            str(tmp_path / "y.arrow"), imprinted=False, contact_tolerance=-1.0
        )
