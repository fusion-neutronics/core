"""Test transport between two separated mesh volumes (implicit complement gap).

Two cubes with a 10 cm air gap. Source in the left cube, tally on the right.
Neutrons free-stream through the void implicit complement to reach the
second cube.
"""

import os
import tempfile

import pytest
import yamc

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc  # noqa: E402

SIDE = 10.0  # cm per cube
GAP = 10.0   # cm between cubes


@pytest.fixture(scope="module")
def two_cube_model():
    """Build mesh geometry with two separated cubes and run transport."""
    # -- CadQuery geometry --
    left_cube = cq.Workplane("XY").box(SIDE, SIDE, SIDE).translate(
        (-(GAP / 2 + SIDE / 2), 0, 0)
    )
    right_cube = cq.Workplane("XY").box(SIDE, SIDE, SIDE).translate(
        ((GAP / 2 + SIDE / 2), 0, 0)
    )

    assy = cq.Assembly()
    assy.add(left_cube, name="lithium")
    assy.add(right_cube, name="beryllium")

    # -- Mesh --
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags="assembly_names")
    c2y.mesh()

    output_dir = os.path.join(tempfile.gettempdir(), "yamc_test_two_cubes_gap")
    arrow_path = os.path.join(output_dir, "model.arrow")
    c2y.to_arrow(arrow_path)

    # -- Materials --
    li_mat = yamc.Material(
        composition={"Li6": 1.0},
        density=0.534,
        name="lithium",
        temperature=294,
    )
    li_mat.read_nuclear_data({"Li6": "tests/Li6.arrow"})

    be_mat = yamc.Material(
        composition={"Be9": 1.0},
        density=1.85,
        name="beryllium",
        temperature=294,
    )
    be_mat.read_nuclear_data({"Be9": "tests/Be9.arrow"})

    materials = {"lithium": li_mat, "beryllium": be_mat}
    mesh_geom = yamc.MeshGeometry(arrow_path, materials)

    # -- Source at centre of lithium cube --
    li_bbox = mesh_geom.bounding_box_for_material(li_mat)
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1.0]),
        position=list(li_bbox.center),
    )

    # -- Tallies --
    be_tally = yamc.Tally(
        scores=["flux"],
        materials=be_mat,
        name="beryllium_flux",
    )

    li_tally = yamc.Tally(
        scores=["flux"],
        materials=li_mat,
        name="lithium_flux",
    )

    model = yamc.Model(
        geometry=mesh_geom,
        tallies=[be_tally, li_tally],
        source=source,
    )
    results = model.simulate_transport(total_particles=25000, seed=42)

    return {
        "mesh_geom": mesh_geom,
        "li_mat": li_mat,
        "be_mat": be_mat,
        "be_tally": be_tally,
        "li_tally": li_tally,
        "results": results,
    }


def test_bounding_box_for_material(two_cube_model):
    """bounding_box_for_material returns correct AABB for each cube."""
    mesh_geom = two_cube_model["mesh_geom"]
    li_mat = two_cube_model["li_mat"]
    be_mat = two_cube_model["be_mat"]

    li_bbox = mesh_geom.bounding_box_for_material(li_mat)
    be_bbox = mesh_geom.bounding_box_for_material(be_mat)

    # Left cube: x = -15 .. -5
    assert li_bbox.lower_left[0] == pytest.approx(-15.0, abs=0.1)
    assert li_bbox.upper_right[0] == pytest.approx(-5.0, abs=0.1)

    # Right cube: x = 5 .. 15
    assert be_bbox.lower_left[0] == pytest.approx(5.0, abs=0.1)
    assert be_bbox.upper_right[0] == pytest.approx(15.0, abs=0.1)

    # Cubes should not overlap in x
    assert li_bbox.upper_right[0] < be_bbox.lower_left[0]


def test_lithium_flux_nonzero(two_cube_model):
    """Source cube should have non-zero flux."""
    li_tally = two_cube_model["li_tally"]
    results = two_cube_model["results"]
    assert results[li_tally].mean[0] > 0.0


def test_beryllium_flux_nonzero(two_cube_model):
    """Neutrons should reach the second cube across the gap.

    The implicit complement (void) allows particles to free-stream through
    the unmeshed space between the two cubes.
    """
    be_tally = two_cube_model["be_tally"]
    results = two_cube_model["results"]
    assert results[be_tally].mean[0] > 0.0
