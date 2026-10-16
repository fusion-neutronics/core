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

    with pytest.warns(UserWarning, match="contact_tolerance"):
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


def _assembly(**parts):
    assembly = cq.Assembly()
    for name, shape in parts.items():
        assembly.add(shape, name=name)
    return assembly


def test_an_embedded_solid_is_refused():
    """A cylinder inside a box that was never cut touches the box nowhere, so
    a particle reaching it from the box would never enter it. Refused at
    meshing time by default; overlaps="ignore" skips the check."""
    from yamc.cad import OverlappingSolidsError

    def embedded():
        return _assembly(
            a=cq.Workplane().circle(4).extrude(20).translate((0, 0, -10)),
            b=cq.Workplane().box(30, 30, 30),
        )

    c2y = CadToYamc()
    c2y.add_cadquery_object(embedded(), material_tags=["a", "b"])
    with pytest.raises(OverlappingSolidsError, match="'a'.*'b'"):
        c2y.mesh(imprint=False)

    c2y = CadToYamc()
    c2y.add_cadquery_object(embedded(), material_tags=["a", "b"])
    c2y.mesh(imprint=False, overlaps="ignore")


def test_contact_tolerance_admits_interference_and_is_recorded(tmp_path):
    """Two cubes interfering by 0.5: an overlap at the default contact
    tolerance (twice the 0.1 deflection), modelling slack once the tolerance
    covers it, and the tolerance travels to the transport in the file."""
    from yamc.cad import OverlappingSolidsError

    def interfering():
        return _assembly(
            a=cq.Workplane().box(10, 10, 10),
            b=cq.Workplane().box(10, 10, 10).translate((9.5, 0, 0)),
        )

    c2y = CadToYamc()
    c2y.add_cadquery_object(interfering(), material_tags=["a", "b"])
    with pytest.raises(OverlappingSolidsError, match="0.5"):
        c2y.mesh(imprint=False)

    c2y = CadToYamc()
    c2y.add_cadquery_object(interfering(), material_tags=["a", "b"])
    c2y.mesh(imprint=False, contact_tolerance=0.6)
    path = str(tmp_path / "interfering.arrow")
    c2y.to_arrow(path)
    geometry = yamc.MeshGeometry(path, _materials(), graveyard_offset=1.0)
    assert geometry.discontinuous
    assert geometry.contact_tolerance == pytest.approx(0.6)


def test_contact_tolerance_needs_imprint_false():
    c2y = CadToYamc()
    c2y.add_cadquery_object(_nested_cylinders(), material_tags=["a", "b", "c"])
    with pytest.raises(ValueError, match="imprint=False"):
        c2y.mesh(contact_tolerance=0.1)


def _embedded(cut):
    box = cq.Workplane().box(30, 30, 30)
    cyl = cq.Workplane().circle(4).extrude(20).translate((0, 0, -10))
    return _assembly(a=cyl, b=box.cut(cyl) if cut else box)


def _two_material_fluxes(path, position, seed, particles=20_000):
    mats = {
        tag: _materials()[tag] for tag in ("a", "b")
    }
    geometry = yamc.MeshGeometry(path, mats, graveyard_offset=1.0)
    tallies = {t: yamc.Tally(scores=["flux"], materials=m, name=t) for t, m in mats.items()}
    model = yamc.Model(
        geometry=geometry,
        tallies=list(tallies.values()),
        source=yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=position
        ),
    )
    results = model.simulate_transport(total_particles=particles, seed=seed)
    return geometry, model, {t: float(results[x].mean[0]) for t, x in tallies.items()}


def test_smaller_wins_matches_the_cut_cad(tmp_path):
    """An uncut box around a cylinder, resolved with overlaps="smaller_wins",
    transports exactly like the same CAD with the box cut and imprinted: the
    cylinder is entered from the box, and the box's volume loses the
    cylinder's."""
    c2y = CadToYamc()
    c2y.add_cadquery_object(_embedded(cut=False), material_tags=["a", "b"])
    with pytest.warns(UserWarning, match="smaller solid wins"):
        c2y.mesh(imprint=False, overlaps="smaller_wins")
    resolved = str(tmp_path / "resolved.arrow")
    c2y.to_arrow(resolved)

    c2y = CadToYamc()
    c2y.add_cadquery_object(_embedded(cut=True), material_tags=["a", "b"])
    c2y.mesh()
    cut = str(tmp_path / "cut.arrow")
    c2y.to_arrow(cut)

    # Source in the box, outside the cylinder: only entry from outside can
    # put any flux in the cylinder.
    geometry, model, flux = _two_material_fluxes(resolved, (10.0, 0.0, 0.0), seed=4)
    reference_geometry, _, reference = _two_material_fluxes(cut, (10.0, 0.0, 0.0), seed=4)
    assert geometry.discontinuous
    assert len(model.lost_particles) == 0
    for tag in ("a", "b"):
        assert flux[tag] == pytest.approx(reference[tag], rel=1e-9), tag
    assert geometry.volume_measures == pytest.approx(
        reference_geometry.volume_measures, rel=1e-6
    )


def test_smaller_wins_takes_a_partial_overlap_off_the_larger_volume(tmp_path):
    """Box b (8 cm) pokes 3 cm into box a (10 cm): a gives up 3 x 8 x 8."""
    c2y = CadToYamc()
    c2y.add_cadquery_object(
        _assembly(
            a=cq.Workplane().box(10, 10, 10),
            b=cq.Workplane().box(8, 8, 8).translate((6, 0, 0)),
        ),
        material_tags=["a", "b"],
    )
    with pytest.warns(UserWarning, match="sampled"):
        c2y.mesh(imprint=False, overlaps="smaller_wins")
    path = str(tmp_path / "boxes.arrow")
    c2y.to_arrow(path)
    geometry = yamc.MeshGeometry(path, _materials(), graveyard_offset=1.0)
    volume_a, volume_b = geometry.volume_measures
    assert volume_b == pytest.approx(512.0, rel=1e-9)
    # Sampled with 200k points: a few tenths of a cm^3.
    assert volume_a == pytest.approx(1000.0 - 192.0, abs=2.0)


def test_smaller_wins_takes_a_nested_chain_off_once(tmp_path):
    """Three uncut nested cubes (10, 6 and 2 cm): each gives up the next one
    in, and the outermost loses the middle one only, not the innermost again."""
    import warnings

    c2y = CadToYamc()
    c2y.add_cadquery_object(
        _assembly(
            a=cq.Workplane().box(10, 10, 10),
            b=cq.Workplane().box(6, 6, 6),
            c=cq.Workplane().box(2, 2, 2),
        ),
        material_tags=["a", "b", "c"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = c2y.mesh(imprint=False, overlaps="smaller_wins")
    assert not [w for w in caught if "twice" in str(w.message)]
    path = str(tmp_path / "nested.arrow")
    c2y.to_arrow(path)
    names = {"a", "b", "c"}
    assert set(mesh.material_tags) == names
    geometry = yamc.MeshGeometry(path, {}, graveyard_offset=1.0)
    assert geometry.volume_measures == pytest.approx([1000.0 - 216.0, 216.0 - 8.0, 8.0], rel=1e-9)


def test_overlaps_option_is_checked():
    c2y = CadToYamc()
    c2y.add_cadquery_object(_embedded(cut=False), material_tags=["a", "b"])
    with pytest.raises(ValueError, match="'error', 'smaller_wins' or 'ignore'"):
        c2y.mesh(imprint=False, overlaps="largest")
    with pytest.raises(ValueError, match="needs imprint=False"):
        c2y.mesh(overlaps="smaller_wins")


def test_a_mesh_from_another_writer_whose_solids_touch_is_flagged(tmp_path):
    """Touching solids with separate surfaces and no provenance in the file
    are tracked conformally, which skips the solid on the far side without
    losing a particle: warn, and say how to track it properly."""
    path = str(tmp_path / "boxes.arrow")
    _write_touching_boxes(path)
    with pytest.warns(UserWarning, match="contact_tolerance"):
        _two_box_fluxes(path)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _two_box_fluxes(path, contact_tolerance=1e-3)


def test_an_imprint_that_left_touching_solids_unshared_is_flagged(tmp_path):
    """A file that says it was imprinted, but whose touching solids still have
    separate surfaces, is an imprint that went wrong: warn, naming the imprint
    and the way out."""
    path = str(tmp_path / "boxes.arrow")
    _write_touching_boxes(path, imprinted=True)
    with pytest.warns(UserWarning, match="imprint did not join them"):
        _two_box_fluxes(path)


def test_a_gap_in_the_cad_is_recorded_and_kept(tmp_path):
    """Two cubes 0.05 apart, meshed at a 0.1 deflection: the gap is within
    the 0.2 contact tolerance, but the CAD says the cubes do not touch, so the
    pair is recorded as separated and the gap is not filled. Touching cubes
    and the nested cylinders (touching, faceted differently) are not."""
    def cubes(gap):
        return _assembly(
            a=cq.Workplane().box(10, 10, 10),
            b=cq.Workplane().box(10, 10, 10).translate((10 + gap, 0, 0)),
        )

    c2y = CadToYamc()
    c2y.add_cadquery_object(cubes(0.05), material_tags=["a", "b"])
    c2y.mesh(tolerance=TOLERANCE, imprint=False)
    assert c2y._separated == [(1, 2)]
    path = str(tmp_path / "gap.arrow")
    c2y.to_arrow(path)
    geometry, model, _ = _fluxes_two(path)
    assert geometry.discontinuous
    assert len(model.lost_particles) == 0

    for assembly, tags in ((cubes(0.0), ["a", "b"]), (_nested_cylinders(), ["a", "b", "c"])):
        c2y = CadToYamc()
        c2y.add_cadquery_object(assembly, material_tags=tags)
        c2y.mesh(tolerance=TOLERANCE, imprint=False)
        assert c2y._separated == []


def _fluxes_two(path):
    mats = {t: _materials()[t] for t in ("a", "b")}
    geometry = yamc.MeshGeometry(path, mats, graveyard_offset=1.0)
    model = yamc.Model(
        geometry=geometry,
        tallies=[],
        source=yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(0.0, 0.0, 0.0)
        ),
    )
    model.simulate_transport(total_particles=2_000, seed=1)
    return geometry, model, None
