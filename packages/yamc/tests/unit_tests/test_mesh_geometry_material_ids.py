"""Material ids in yamc.MeshGeometry.

Every Material has an id from construction, so a MeshGeometry neither assigns
nor changes any: it keeps the ids the materials were built with, and refuses
two different materials that share one. Absolute automatic ids depend on what
ran earlier in the process, so these tests compare ids rather than assert them.

Volumes given the same Material object share it: it is held once, with the
total volume of the mesh volumes it fills.

Lives in ``unit_tests/`` (not ``cad/``) so it runs in CI, which installs no
cadquery.
"""

import pytest
import yamc

TWO_REGION_ARROW = "crates/yamt/tests/data/two_region.arrow"
FE56 = "tests/Fe56.arrow"


def _dummy_material(name, **kwargs):
    """Minimal Material (no nuclear data) used purely to test ID handling."""
    return yamc.Material(
        composition={"H": 1.0}, density=1.0, density_units="g/cc", name=name, **kwargs,
    )


def test_materials_keep_their_ids():
    """The materials have distinct ids before the mesh geometry is built, and
    building it does not change them."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    ids = (fuel.id, moderator.id)
    assert fuel.id != moderator.id

    yamc.MeshGeometry.from_arrow(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    assert (fuel.id, moderator.id) == ids


def test_explicit_id_is_preserved():
    """An explicit id survives construction, and a material built after it
    does not get it."""
    probe = _dummy_material("probe")
    fuel = _dummy_material("fuel", id=probe.id + 1)
    moderator = _dummy_material("moderator")

    yamc.MeshGeometry.from_arrow(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    assert fuel.id == probe.id + 1
    assert moderator.id not in (probe.id, fuel.id)


def test_duplicate_ids_rejected():
    """Two materials with the same explicit ID should raise."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    fuel.id = 5
    moderator.id = 5

    with pytest.raises(ValueError, match="Duplicate Material id"):
        yamc.MeshGeometry.from_arrow(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})


def test_explicit_id_repeating_an_automatic_one_is_rejected():
    """An explicit id set later can repeat an automatic one; the mesh
    geometry refuses the two different materials that share it."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator", id=fuel.id)

    with pytest.raises(ValueError, match=f"Duplicate Material id {fuel.id}"):
        yamc.MeshGeometry.from_arrow(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})


def test_bounding_box_for_material_works_after_construction():
    """Looking up a Material by reference finds its volumes by its id."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    mesh = yamc.MeshGeometry.from_arrow(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    bbox = mesh.bounding_box_for_material(fuel)
    assert bbox is not None
    # Two-region mesh: fuel occupies the left half (x in [0, 0.5]).
    assert bbox.lower_left[0] >= -1e-9
    assert bbox.upper_right[0] <= 0.5 + 1e-9


def test_calculate_volume_uses_analytic_values():
    """MeshGeometry.calculate_volume should return the exact analytic volumes
    (already known via the divergence theorem when the mesh was loaded), with
    std_dev == 0 -- no stochastic sampling is performed."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    mesh = yamc.MeshGeometry.from_arrow(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    # samples/bounding_box/seed are accepted for signature compatibility but ignored
    volumes = mesh.calculate_volume(samples=12345, seed=99)

    assert isinstance(volumes, dict)
    assert len(volumes) == mesh.num_volumes

    expected_measures = mesh.volume_measures
    for vol_id, result in volumes.items():
        assert result.volume == expected_measures[vol_id]
        assert result.std_dev == 0.0
        assert result.num_hits == 0


def test_calculate_volume_signature_matches_csg():
    """Calling calculate_volume with no args (the typical CSG idiom) works."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    mesh = yamc.MeshGeometry.from_arrow(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    volumes = mesh.calculate_volume()
    assert len(volumes) == mesh.num_volumes
    assert all(r.std_dev == 0.0 for r in volumes.values())


def test_implicit_complement_material_keeps_its_id():
    """The optional implicit-complement material follows the same rule as the
    named-dict materials."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    air = _dummy_material("air")
    air_id = air.id

    yamc.MeshGeometry.from_arrow(
        TWO_REGION_ARROW,
        {"fuel": fuel, "moderator": moderator},
        implicit_complement_material=air,
    )

    assert air.id == air_id
    assert air.id not in (fuel.id, moderator.id)


def test_same_material_under_multiple_keys_is_one_material():
    """If the user maps two names to the same ``Material`` object, it is one
    material with one id, not a duplicate."""
    shared = _dummy_material("shared")
    yamc.MeshGeometry.from_arrow(TWO_REGION_ARROW, {"fuel": shared, "moderator": shared})


def _embedded_cells(materials):
    """The two mesh volumes as cells of a Geometry, keyed by volume name.

    A MeshGeometry shows no materials to Python; the cells of a Geometry it
    fills do, so the mesh built with *materials* fills a sphere cell here.
    """
    sphere = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    cell = yamc.Cell(
        region=sphere.below, fill=yamc.MeshGeometry.from_arrow(TWO_REGION_ARROW, materials),
    )
    cells = yamc.Geometry([cell]).cells
    assert len(cells) == 3, "host + two embedded mesh volumes"
    return {
        name: next(c for c in cells[1:] if name in c.name)
        for name in ("fuel", "moderator")
    }


def test_volumes_given_one_material_share_it():
    """One Material object under both names is held once: both volumes report
    its id, and its volume is the total of the two, while each cell keeps its
    own. Held once per volume, it would carry one volume's 0.5 cm^3."""
    shared = _dummy_material("shared")
    cells = _embedded_cells({"fuel": shared, "moderator": shared})

    for cell in cells.values():
        assert cell.material.id == shared.id
        assert cell.volume == pytest.approx(0.5)
        assert cell.material.volume == pytest.approx(1.0)
    # The caller's material is not changed.
    assert shared.volume is None


def test_separate_materials_keep_their_own_volume():
    """Two Material objects are two materials, each with the volume of the
    one mesh volume it fills."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    cells = _embedded_cells({"fuel": fuel, "moderator": moderator})

    assert cells["fuel"].material.id == fuel.id
    assert cells["moderator"].material.id == moderator.id
    for cell in cells.values():
        assert cell.volume == pytest.approx(0.5)
        assert cell.material.volume == pytest.approx(0.5)


def test_calculate_volume_gives_a_shared_material_the_total():
    """Geometry.calculate_volume sets each cell's sampled volume, and gives a
    material in several cells their total, not the last cell's."""
    shared = _dummy_material("shared")
    sphere = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    cell = yamc.Cell(
        region=sphere.below,
        fill=yamc.MeshGeometry.from_arrow(TWO_REGION_ARROW, {"fuel": shared, "moderator": shared}),
    )
    geometry = yamc.Geometry([cell])
    geometry.calculate_volume(samples=200_000, seed=3)

    fuel, moderator = geometry.cells[1:]
    assert fuel.volume == pytest.approx(0.5, rel=0.2)
    assert moderator.volume == pytest.approx(0.5, rel=0.2)
    for embedded in (fuel, moderator):
        assert embedded.material.volume == pytest.approx(fuel.volume + moderator.volume)


def _iron():
    """Iron with nuclear data, for the transport comparison."""
    material = yamc.Material(
        composition={"Fe56": 1.0}, density=7.87, name="iron", temperature=294
    )
    material.read_nuclear_data({"Fe56": FE56})
    return material


def _flux_per_half(materials, **kwargs):
    """Flux in each half of the two-region cube from a 14 MeV point source in
    the fuel half: a regular mesh tally with one voxel per mesh volume, which
    does not depend on material ids."""
    geometry = yamc.MeshGeometry.from_arrow(TWO_REGION_ARROW, materials, **kwargs)
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(0.25, 0.5, 0.5)
    )
    mesh = yamc.RegularRectangularMesh(
        lower_left=[0.0, 0.0, 0.0], upper_right=[1.0, 1.0, 1.0], shape=[2, 1, 1]
    )
    tally = yamc.Tally(scores=["flux"], mesh=mesh, name="flux")
    model = yamc.Model(geometry=geometry, source=source, tallies=[tally], verbose=[])
    results = model.simulate_transport(total_particles=2000, seed=3)
    assert len(model.lost_particles) == 0
    return list(results[tally].mean)


@pytest.mark.parametrize(
    "complement", [False, True], ids=["void_complement", "material_complement"]
)
def test_a_shared_material_transports_like_separate_equal_ones(complement):
    """Sharing changes how a material is held, not the physics: with the same
    seed, the flux with one Material everywhere is the flux with an equal
    Material of its own (and its own id) in each place, history for history.
    With *complement* the material also fills the implicit complement, out to
    a vacuum box 1 cm from the cube."""
    shared = _iron()
    one = {"fuel": shared, "moderator": shared}
    several = {"fuel": _iron(), "moderator": _iron()}
    assert several["fuel"].id != several["moderator"].id
    if complement:
        flux_one = _flux_per_half(
            one, implicit_complement_material=shared, graveyard_offset=1.0
        )
        flux_several = _flux_per_half(
            several, implicit_complement_material=_iron(), graveyard_offset=1.0
        )
    else:
        flux_one = _flux_per_half(one)
        flux_several = _flux_per_half(several)

    assert all(flux > 0.0 for flux in flux_one)
    # Not exact: two runs of one model already differ in the last bit.
    assert flux_one == pytest.approx(flux_several, rel=1e-12)
