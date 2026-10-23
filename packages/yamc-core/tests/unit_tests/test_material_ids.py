"""Every Material gets a unique id when it is built (issue #422).

``Tally(materials=...)`` used to print a Rust panic and raise for a material
made without ``id=``, because ids were only handed out when a Geometry was
built. The ids come from one allocator per process, so these tests assert
uniqueness, stability and how explicit ids are treated, never absolute values:
those depend on what else ran earlier in the pytest session.
"""

import json
import os
import subprocess
import sys

import pytest
import yamc

FE56 = "tests/Fe56.arrow"


def _material(**kwargs):
    return yamc.Material(composition={"Fe56": 1.0}, density=7.9, **kwargs)


def _every_constructor():
    """One material from each way a user can make one."""
    a = _material()
    b = yamc.Material(composition={"H1": 2.0, "O16": 1.0}, density=1.0)
    return [
        a,
        b,
        yamc.Material.from_atom_densities({"Fe56": 0.08}),
        yamc.Material.mix_materials([a, b], [0.5, 0.5], "volume"),
        yamc.materials.pnnl["Steel, Stainless 304"],
        yamc.materials.pnnl.material("Steel, Stainless 304"),
    ]


def test_the_documented_usage_works_without_a_panic():
    """The issue's reproduction, in a fresh interpreter so the panic hook's
    output (which goes straight to the process's stderr) can be seen."""
    script = (
        "import yamc\n"
        "steel = yamc.Material(composition={'Fe56': 1.0}, density=7.9, name='steel')\n"
        "water = yamc.Material(composition={'H1': 2.0, 'O16': 1.0}, density=1.0)\n"
        "one = yamc.Tally(scores=['flux'], materials=steel)\n"
        "two = yamc.Tally(scores=['flux'], materials=[steel, water])\n"
        "print(one.materials == [steel.id], two.materials == [steel.id, water.id])\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=os.environ.copy(), timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert "panicked" not in run.stderr, run.stderr
    assert run.stdout.split() == ["True", "True"], run.stdout


def test_ids_are_unique_across_constructors():
    materials = [m for _ in range(50) for m in _every_constructor()]
    ids = [m.id for m in materials]
    assert all(isinstance(i, int) and i > 0 for i in ids)
    assert len(set(ids)) == len(ids)


def test_an_id_is_stable():
    material = _material()
    first = material.id
    assert material.id == first
    tally = yamc.Tally(materials=material)
    cell = yamc.Cell(region=yamc.Sphere(radius=1.0, boundary="vacuum").below,
                     material=material)
    geometry = yamc.Geometry([cell])
    assert material.id == first
    assert tally.materials == [first]
    # The geometry's copies carry the same id.
    assert geometry.cells[0].material.id == first
    assert geometry.find_cell(0.0, 0.0, 0.0).material.id == first


def test_explicit_ids_are_kept():
    a = _material(id=1001)
    b = yamc.Material(composition={"H1": 1.0}, density=1.0, id=1002)
    assert a.id == 1001
    assert b.id == 1002
    assert yamc.Material.from_atom_densities({"Fe56": 0.08}, id=1003).id == 1003
    assert yamc.Material.mix_materials([a, b], [0.5, 0.5], "volume", id=1004).id == 1004
    assert yamc.materials.pnnl.material("Steel, Stainless 304", id=1005).id == 1005
    a.id = 1006
    assert a.id == 1006


@pytest.mark.parametrize("how", ["constructor", "setter", "from_atom_densities", "mix", "pnnl"])
def test_an_automatic_id_skips_an_explicit_one_set_earlier(how):
    """The explicit id is the one the allocator would hand out next."""
    probe = _material()
    taken = probe.id + 1
    if how == "constructor":
        explicit = _material(id=taken)
    elif how == "setter":
        explicit = _material()
        taken = explicit.id + 1
        explicit.id = taken
    elif how == "from_atom_densities":
        explicit = yamc.Material.from_atom_densities({"Fe56": 0.08}, id=taken)
    elif how == "mix":
        explicit = yamc.Material.mix_materials([probe], [1.0], "atom", id=taken)
    else:
        explicit = yamc.materials.pnnl.material("Steel, Stainless 304", id=taken)
    assert explicit.id == taken
    later = [m.id for m in _every_constructor()]
    assert taken not in later


def test_an_automatic_id_skips_one_read_from_a_model_file():
    """Ids in a saved model are explicit too: loading one keeps later
    automatic ids off it."""
    material = _material()
    cell = yamc.Cell(region=yamc.Sphere(radius=1.0, boundary="vacuum").below,
                     material=material)
    model = yamc.Model(
        geometry=yamc.Geometry([cell]),
        source=yamc.NeutronSource(energy=yamc.sources.Discrete([14.06e6], [1.0])),
    )
    spec = json.loads(model.to_json())
    taken = _material().id + 1
    replaced = []

    def give_taken_id(node):
        if isinstance(node, dict):
            if node.get("material_id") == material.id:
                node["material_id"] = taken
                replaced.append(node)
            for value in node.values():
                give_taken_id(value)
        elif isinstance(node, list):
            for value in node:
                give_taken_id(value)

    give_taken_id(spec)
    assert len(replaced) == 1
    loaded = yamc.Model.load(json.dumps(spec))
    assert loaded.geometry.cells[0].material.id == taken
    assert taken not in [m.id for m in _every_constructor()]


def test_a_shared_id_is_refused_by_the_tally_and_geometry(capfd):
    steel = _material()
    clash = _material(id=steel.id)
    with pytest.raises(ValueError, match=f"both have id {steel.id}"):
        yamc.Tally(materials=[steel, clash])
    with pytest.raises(ValueError, match=f"both have id {steel.id}"):
        yamc.Tally(materials=[steel, steel])
    with pytest.raises(ValueError, match="at least one material"):
        yamc.Tally(materials=[])
    tally = yamc.Tally(materials=steel)
    with pytest.raises(ValueError, match=f"both have id {steel.id}"):
        tally.materials = [clash, steel]
    cells = [
        yamc.Cell(region=yamc.Sphere(radius=1.0).below, material=steel),
        yamc.Cell(region=yamc.Sphere(x0=3.0, radius=1.0).below, material=clash),
    ]
    with pytest.raises(ValueError, match=f"Duplicate Material id {steel.id}"):
        yamc.Geometry(cells)
    assert "panicked" not in capfd.readouterr().err


def test_a_material_tallied_before_its_geometry_is_scored():
    """The tally is built from materials that have only their automatic ids,
    before the cells and geometry exist. Four nested shells alternate between
    the two materials, so each material bin must equal the sum of two
    non-adjacent cell bins: both tallies score the same histories."""
    thin = yamc.Material(composition={"Fe56": 1.0}, density=2.0, temperature=294)
    dense = yamc.Material(composition={"Fe56": 1.0}, density=7.0, temperature=294)
    by_material = yamc.Tally(scores=["flux"], materials=[thin, dense])
    for m in (thin, dense):
        m.read_nuclear_data({"Fe56": FE56})

    s1, s2, s3 = (yamc.Sphere(radius=r) for r in (3.0, 6.0, 9.0))
    s4 = yamc.Sphere(radius=12.0, boundary="vacuum")
    cells = [
        yamc.Cell(region=s1.below, material=thin),
        yamc.Cell(region=s1.above & s2.below, material=dense),
        yamc.Cell(region=s2.above & s3.below, material=thin),
        yamc.Cell(region=s3.above & s4.below, material=dense),
    ]
    geometry = yamc.Geometry(cells)
    by_cell = yamc.Tally(scores=["flux"], cells=cells)
    model = yamc.Model(
        geometry=geometry,
        source=yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(0, 0, 0),
        ),
        tallies=[by_material, by_cell],
    )
    results = model.simulate_transport(total_particles=2000, seed=4242)
    mat = list(results[by_material].mean)
    cell = list(results[by_cell].mean)

    assert all(c > 0.0 for c in cell), cell
    assert mat[0] == pytest.approx(cell[0] + cell[2], rel=1e-9)
    assert mat[1] == pytest.approx(cell[1] + cell[3], rel=1e-9)
    split = mat[0] / (mat[0] + mat[1])
    assert 0.05 < split < 0.95, split


def _two_shell_run(seed, ids, *, mix_inner=False):
    """Two Fe56 shells, thin then dense, with a flux tally per cell and one
    over the materials in id order. ``ids`` are the materials' explicit ids,
    or None each for automatic ones. With ``mix_inner`` the inner shell is an
    unnamed mix of the two, whose default name is part of the fingerprint."""
    thin = yamc.Material(composition={"Fe56": 1.0}, density=2.0, temperature=294, id=ids[0])
    dense = yamc.Material(composition={"Fe56": 1.0}, density=7.0, temperature=294, id=ids[1])
    if mix_inner:
        thin = yamc.Material.mix_materials([thin, dense], [0.5, 0.5], "volume", id=ids[0])
    for m in (thin, dense):
        m.read_nuclear_data({"Fe56": FE56})
    s1, s2 = yamc.Sphere(radius=3.0), yamc.Sphere(radius=6.0, boundary="vacuum")
    cells = [
        yamc.Cell(region=s1.below, material=thin),
        yamc.Cell(region=s1.above & s2.below, material=dense),
    ]
    model = yamc.Model(
        geometry=yamc.Geometry(cells),
        source=yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(0, 0, 0),
        ),
        tallies=[
            yamc.Tally(scores=["flux"], name="by_cell", cells=cells),
            yamc.Tally(scores=["flux"], name="by_material",
                       materials=sorted([thin, dense], key=lambda m: m.id)),
        ],
    )
    return (thin.id, dense.id), model.simulate_transport(total_particles=200, seed=seed)


def test_runs_of_a_model_built_twice_still_combine(tmp_path):
    """Building the same model twice in one process gives its materials new
    ids and the same physics, so ``combine_results`` must still pool the runs:
    the fingerprint labels materials by slot, a mix's default name leaves the
    ids out, and a material filter is matched by the material it names. One
    run goes through a file, as runs from separate jobs do."""
    first_ids, first = _two_shell_run(1, (None, None), mix_inner=True)
    second_ids, second = _two_shell_run(2, (None, None), mix_inner=True)
    assert set(first_ids).isdisjoint(second_ids)
    path = str(tmp_path / "second.arrow")
    second.to_arrow(path)
    combined = yamc.combine_results(first, yamc.SimulationResults.from_arrow(path))
    assert combined["by_cell"].n_histories == 400
    assert combined["by_material"].n_histories == 400
    assert combined["by_material"].tally.materials == sorted(first_ids)


def test_runs_whose_material_filters_name_different_materials_do_not_combine():
    """Two runs give the same two ids to the two materials the other way
    round, as two jobs building their materials in different orders do, and
    filter on the ids in order. The bins name different materials, so pooling
    them would average unrelated fluxes."""
    low = _material().id + 1
    _, first = _two_shell_run(1, (low, low + 1))
    _, second = _two_shell_run(2, (low + 1, low))
    with pytest.raises(ValueError, match="filter on different materials"):
        yamc.combine_results(first, second)
    _, third = _two_shell_run(3, (low, low + 1))
    assert yamc.combine_results(first, third)["by_material"].n_histories == 400


def test_a_failed_construction_takes_no_id():
    """The id is assigned once the material is valid, so a constructor that
    raises neither hands out nor reserves one."""
    before = _material().id
    with pytest.raises(ValueError, match="finite"):
        _material(temperature=float("nan"))
    with pytest.raises(ValueError, match="positive"):
        _material(id=before + 1, volume=-1.0)
    assert _material().id == before + 1
