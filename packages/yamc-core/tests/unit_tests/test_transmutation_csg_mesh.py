"""Transport-transmutation on CSG and mesh geometry, parametrized over method.

Parametrized over ``method`` in {"coupled", "independent"}:
- "coupled" re-runs transport at each step;
- "independent" runs transport once, then scales reaction rates by each
  step's source_rate.

Verifies that:
- Fe56 density decreases under 14 MeV neutron irradiation
- Daughter nuclides (Fe55, Fe57, Cr52, ...) appear after transmutation
- Stable isotopes remain constant during cooling steps
- CSG and mesh geometries produce matching results
"""

import os
import tempfile

import pytest
import yamc

try:
    import cadquery as cq
    from yamc.cad import CadToYamc

    HAS_CADQUERY = True
except ImportError:
    HAS_CADQUERY = False

SIDE = 10.0
HALF = SIDE / 2.0
NUC_DATA = "tests"
CHAIN_FILE = "tests/transmutation-endf-b8.1-sfr.arrow"
DAY = 86400.0


@pytest.fixture(autouse=True)
def _set_cross_sections():
    yamc.cross_section_data = NUC_DATA


@pytest.fixture(params=["coupled", "independent"])
def method(request):
    """Run every test once per transmutation method."""
    return request.param


def _make_iron():
    iron = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        transmutable=True,
        temperature=294)
    return iron


def _run_transmutation(geometry, method):
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1.0]),
        position=(0.0, 0.0, 0.0))
    model = yamc.Model(geometry=geometry, source=source)

    schedule = yamc.PulseSchedule([
        yamc.Pulse(rate=1e12, duration=DAY, source=source),
        yamc.Pulse(rate=1e12, duration=DAY, source=source),
        yamc.Cooldown(duration=DAY),
        yamc.Cooldown(duration=DAY),
    ])

    return model.simulate_transmutation(
        method=method,
        schedule=schedule,
        total_particles=2500,
    )


def _build_csg():
    iron = _make_iron()

    xn = yamc.Plane(axis="x", offset=-HALF, boundary="vacuum")
    xp = yamc.Plane(axis="x", offset=+HALF, boundary="vacuum")
    yn = yamc.Plane(axis="y", offset=-HALF, boundary="vacuum")
    yp = yamc.Plane(axis="y", offset=+HALF, boundary="vacuum")
    zn = yamc.Plane(axis="z", offset=-HALF, boundary="vacuum")
    zp = yamc.Plane(axis="z", offset=+HALF, boundary="vacuum")

    region = xn.above & xp.below & yn.above & yp.below & zn.above & zp.below
    cell = yamc.Cell(name="iron_cube", region=region, material=iron)
    geom = yamc.Geometry([cell])
    geom.calculate_volume(samples=1_000_000)
    return geom


@pytest.fixture
def csg_results(method):
    return _run_transmutation(_build_csg(), method)


def test_csg_fe56_decreases(csg_results):
    """Fe56 density should decrease under 14 MeV irradiation."""
    fe56 = csg_results.get_nuclide_evolution(material_id=1, nuclide="Fe56")
    assert len(fe56) == 5  # t=0, 1d, 2d, 3d, 4d
    initial = fe56[0]
    after_irradiation = fe56[2]  # after 2 days of irradiation
    assert initial > 0
    assert after_irradiation < initial, (
        f"Fe56 should decrease: initial={initial}, after={after_irradiation}"
    )


def test_csg_daughter_nuclides_appear(csg_results):
    """Daughter nuclides should appear after transmutation."""
    final_step = csg_results.num_steps
    final = csg_results.get_material_nuclides(material_id=1, step=final_step)
    assert final, "Final composition should not be empty"
    assert len(final) > 1, (
        f"Expected daughter nuclides, got only {list(final.keys())}"
    )
    # Fe56 (n,2n) -> Fe55, Fe56 (n,gamma) -> Fe57 are the main channels at 14 MeV
    assert "Fe56" in final


def test_csg_stable_isotopes_constant_during_cooling(csg_results):
    """Stable isotopes should not change between consecutive cooling steps."""
    fe56 = csg_results.get_nuclide_evolution(material_id=1, nuclide="Fe56")
    # Steps 3 and 4 are both cooling (source_rate=0).
    # Fe56 is stable so it cannot decay. Short-lived feeders like Mn56
    # (t1/2=2.58h) will have fully decayed within the first cooling day,
    # so the second cooling step should show no further change.
    cooling_step_1 = fe56[3]  # after 1 day cooling
    cooling_step_2 = fe56[4]  # after 2 days cooling
    assert cooling_step_1 == pytest.approx(cooling_step_2, rel=1e-6), (
        f"Fe56 should be constant during cooling: "
        f"step3={cooling_step_1}, step4={cooling_step_2}"
    )


def _irradiation_only_model():
    """CSG iron cube + a single 14 MeV pulse, no stop condition applied yet."""
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(0.0, 0.0, 0.0)
    )
    model = yamc.Model(geometry=_build_csg(), source=source)
    schedule = yamc.PulseSchedule([
        yamc.Pulse(rate=1e12, duration=DAY, source=source),
        yamc.Cooldown(duration=DAY),
    ])
    return model, schedule


def test_transmutation_max_runtime_only_runs_and_depletes():
    """A per-step wall-time budget alone (no total_particles) drives each
    transport solve to time and still depletes Fe56 under irradiation."""
    model, schedule = _irradiation_only_model()
    results = model.simulate_transmutation(
        method="coupled", schedule=schedule, max_runtime=(2, "s"),
    )
    fe56 = results.get_nuclide_evolution(material_id=1, nuclide="Fe56")
    assert len(fe56) == 3  # t=0, after pulse, after cooldown
    assert fe56[0] > 0
    assert fe56[1] < fe56[0]  # depleted during the irradiation step


def test_transmutation_requires_a_stop_condition():
    """Neither total_particles nor max_runtime is rejected before any
    transport runs (the two are the per-step stop conditions)."""
    model, schedule = _irradiation_only_model()
    with pytest.raises(ValueError, match="per-step stop condition"):
        model.simulate_transmutation(method="coupled", schedule=schedule)


@pytest.mark.skipif(not HAS_CADQUERY, reason="cadquery not installed")
class TestMeshTransmutation:

    @pytest.fixture
    def mesh_results(self, method):
        box = cq.Workplane("XY").box(SIDE, SIDE, SIDE)
        assy = cq.Assembly()
        assy.add(box, name="iron")

        tmpdir = tempfile.mkdtemp(prefix="yamc_test_dep_mesh_")
        arrow_path = os.path.join(tmpdir, "cube.arrow")

        c2y = CadToYamc()
        c2y.add_cadquery_object(assy, material_tags=["iron"])
        c2y.mesh(tolerance=0.1, angular_tolerance=0.1)
        c2y.to_arrow(arrow_path)

        iron = _make_iron()
        mesh_geom = yamc.MeshGeometry(arrow_path, {"iron": iron})
        return _run_transmutation(mesh_geom, method)

    def test_mesh_fe56_decreases(self, mesh_results):
        """Fe56 density should decrease under 14 MeV irradiation (mesh)."""
        fe56 = mesh_results.get_nuclide_evolution(material_id=1, nuclide="Fe56")
        assert len(fe56) == 5
        assert fe56[2] < fe56[0], (
            f"Fe56 should decrease: initial={fe56[0]}, after={fe56[2]}"
        )

    def test_mesh_daughter_nuclides_appear(self, mesh_results):
        """Daughter nuclides should appear after transmutation (mesh)."""
        final = mesh_results.get_material_nuclides(
            material_id=1, step=mesh_results.num_steps
        )
        assert len(final) > 1

    def test_mesh_matches_csg(self, mesh_results, method):
        """Mesh and CSG transmutation should produce identical Fe56 evolution."""
        csg_geom = _build_csg()
        csg_res = _run_transmutation(csg_geom, method)

        fe56_csg = csg_res.get_nuclide_evolution(material_id=1, nuclide="Fe56")
        fe56_mesh = mesh_results.get_nuclide_evolution(material_id=1, nuclide="Fe56")

        assert len(fe56_csg) == len(fe56_mesh)
        for i in range(len(fe56_csg)):
            assert fe56_csg[i] == pytest.approx(fe56_mesh[i], rel=1e-6), (
                f"Step {i}: CSG={fe56_csg[i]}, Mesh={fe56_mesh[i]}"
            )
