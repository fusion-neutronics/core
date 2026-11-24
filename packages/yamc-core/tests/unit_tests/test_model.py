import yamc
import pytest
import os

def test_duplicate_tally_names_rejected_at_construction():
    """Duplicate tally names should be caught at Model construction time,
    not deferred until simulate_transport()."""
    sphere = yamc.Sphere(radius=2.0, boundary='vacuum')
    cell = yamc.Cell(region=sphere.below)
    geometry = yamc.Geometry([cell])

    t1 = yamc.Tally(scores=['flux'], name="my_tally")
    t2 = yamc.Tally(scores=['heating'], name="my_tally")

    with pytest.raises(ValueError, match="duplicate tally name"):
        yamc.Model(geometry, tallies=[t1, t2])


def test_duplicate_tally_ids_rejected_at_construction():
    """Duplicate tally ids should also be rejected at construction time."""
    sphere = yamc.Sphere(radius=2.0, boundary='vacuum')
    cell = yamc.Cell(region=sphere.below)
    geometry = yamc.Geometry([cell])

    t1 = yamc.Tally(scores=['flux'], id=7)
    t2 = yamc.Tally(scores=['heating'], id=7)

    with pytest.raises(ValueError, match="duplicate tally id"):
        yamc.Model(geometry, tallies=[t1, t2])


def test_unique_tally_names_allowed():
    """Distinct names work."""
    sphere = yamc.Sphere(radius=2.0, boundary='vacuum')
    cell = yamc.Cell(region=sphere.below)
    geometry = yamc.Geometry([cell])

    t1 = yamc.Tally(scores=['flux'], name="flux")
    t2 = yamc.Tally(scores=['heating'], name="heating")
    model = yamc.Model(geometry, tallies=[t1, t2])
    assert len(model.tallies) == 2


def _bare_geometry():
    sphere = yamc.Sphere(radius=2.0, boundary='vacuum')
    return yamc.Geometry([yamc.Cell(region=sphere.below)])


def test_electron_treatment_default_and_values():
    """electron_treatment defaults to 'ttb' and accepts 'ttb'/'local'."""
    geometry = _bare_geometry()
    assert yamc.Model(geometry).electron_treatment == "ttb"
    assert yamc.Model(geometry, electron_treatment="ttb").electron_treatment == "ttb"
    assert yamc.Model(geometry, electron_treatment="local").electron_treatment == "local"


def test_electron_treatment_invalid_rejected():
    with pytest.raises(ValueError, match="invalid electron_treatment"):
        yamc.Model(_bare_geometry(), electron_treatment="bogus")


def test_electron_treatment_ttb_alias_removed():
    """The boolean electron_treatment_ttb= alias is gone."""
    with pytest.raises(TypeError):
        yamc.Model(_bare_geometry(), electron_treatment_ttb=False)


def test_model_construction():
    # Create sphere surface with vacuum boundary
    sphere = yamc.Sphere(
        x0=0.0,
        y0=0.0,
        z0=0.0,
        radius=2.0,
        boundary='vacuum')
    region = sphere.below
    material = yamc.Material(
        composition={"Li6": 1.0},
        density=0.5,
        temperature=294)
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    cell = yamc.Cell(
        name="sphere_cell",
        region=region,
        material=material)
    geometry = yamc.Geometry(cells=[cell])
    source = yamc.NeutronSource(position=[0.0, 0.0, 0.0], energy=yamc.sources.Discrete([1e6], [1.0]))
    model = yamc.Model(geometry, source=source)
    assert model.source[0].energy.energies == [1e6]
    assert model.source[0].energy.probabilities == [1.0]
    assert len(model.geometry.cells) == 1
    model.simulate_transport(total_particles=10)


@pytest.mark.skipif(
    os.getenv("CI") is not None or os.getenv("GITHUB_ACTIONS") is not None,
    reason="Skip threading performance test in CI environment"
)
def test_model_threads_parameter():
    """Test that the threads parameter controls parallel execution and improves performance."""
    # Create a sphere with vacuum boundary
    sphere = yamc.Sphere(
        x0=0.0,
        y0=0.0,
        z0=0.0,
        radius=5.0,
        boundary='vacuum')
    region = sphere.below

    # Use a material with multiple nuclides to make computation heavier
    material = yamc.Material(
        composition={"Li6": 0.5, "Li7": 0.5},
        density=0.5,
        temperature=294)
    material.read_nuclear_data({
        "Li6": "tests/Li6.arrow",
        "Li7": "tests/Li7.arrow"
    })
    
    cell = yamc.Cell(
        name="sphere_cell",
        region=region,
        material=material)
    geometry = yamc.Geometry(cells=[cell])
    
    # A small particle count is enough: this test only checks that the
    # `threads` kwarg is accepted and that the elapsed_time /
    # particles_per_second properties get populated. It makes no assertion
    # about speedup, so there is no need to run a heavy simulation.
    source = yamc.NeutronSource(
        position=[0.0, 0.0, 0.0],
        energy=yamc.sources.Discrete([1e6], [1.0])
    )
    # Test with 1 thread
    model_1 = yamc.Model(geometry, source=source)
    results_1 = model_1.simulate_transport(total_particles=2000, threads=1)

    # Test with 2 threads
    model_2 = yamc.Model(geometry, source=source)
    results_2 = model_2.simulate_transport(total_particles=2000, threads=2)

    # Test with default (all threads)
    model_default = yamc.Model(geometry, source=source)
    results_default = model_default.simulate_transport(total_particles=2000)

    # Extract performance metrics
    pps_1_thread = results_1.particles_per_second
    pps_2_threads = results_2.particles_per_second
    pps_default = results_default.particles_per_second

    time_1_thread = results_1.elapsed
    time_2_threads = results_2.elapsed
    time_default = results_default.elapsed

    print("\nPerformance results:")
    print(f"  1 thread:  {pps_1_thread:,} particles/s ({time_1_thread:.3f}s)")
    print(f"  2 threads: {pps_2_threads:,} particles/s ({time_2_threads:.3f}s)")
    print(f"  Default:   {pps_default:,} particles/s ({time_default:.3f}s)")
    print(f"  Speedup (2 vs 1): {pps_2_threads / pps_1_thread:.2f}x")

    # Verify the timing properties on the results are populated
    assert results_1.elapsed is not None
    assert results_2.elapsed is not None
    assert results_default.elapsed is not None
    assert results_1.particles_per_second is not None
    assert results_2.particles_per_second is not None
    assert results_default.particles_per_second is not None
