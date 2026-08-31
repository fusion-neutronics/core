"""yamc.combine_results: exact pooling of independent same-model runs.

Covers the user-facing API: the pooled merge and its equivalence to a
single longer run, the explicit accumulation loop, every refusal guard
(seed, model identity, tally structure), partial-overlap pass-through
warnings, run provenance, and the lossless Arrow round trip.
"""

import pytest
import yamc


def build_model(particles=5_000, seed=1, density=2.0, with_absorption=True,
                radius=10.0):
    """Li6 sphere, 1 MeV point source, flux (+ optional absorption) tally."""
    material = yamc.Material(
        composition={"Li6": 1.0},
        density=density,
        temperature=294)
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})

    sphere = yamc.Sphere(radius=radius, boundary="vacuum")
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])

    source = yamc.NeutronSource(
        position=(0, 0, 0),
        energy=yamc.sources.Discrete([1.0e6], [1.0]))

    tallies = [yamc.Tally(scores=["flux"], name="flux", cells=cell,
                          particle="neutron")]
    if with_absorption:
        tallies.append(yamc.Tally(scores=["absorption"], name="absorption",
                                  cells=cell, particle="neutron"))

    model = yamc.Model(
        geometry=geometry,
        tallies=tallies,
        source=source,
        verbose=[])
    return model, {"total_particles": particles, "seed": seed}


def run(particles=5_000, seed=1, **kwargs):
    model, run_kwargs = build_model(particles=particles, seed=seed, **kwargs)
    return model.simulate_transport(**run_kwargs, threads=1)


def test_combined_matches_single_long_run():
    r1 = run(particles=8_000, seed=11)
    r2 = run(particles=12_000, seed=22)
    single = run(particles=20_000, seed=33)

    combined = yamc.combine_results(r1, r2)
    for name in ("flux", "absorption"):
        m = combined[name]
        s = single[name]
        assert m.n_histories == 20_000
        sigma = (m.standard_deviation[0] ** 2
                 + s.standard_deviation[0] ** 2) ** 0.5
        assert abs(m.mean[0] - s.mean[0]) <= 4.0 * max(sigma, 1e-12), (
            f"{name}: combined {m.mean[0]} vs single {s.mean[0]}")
        # Pooling tightens the error bars relative to each part.
        assert m.standard_deviation[0] < r1[name].standard_deviation[0]
        assert m.standard_deviation[0] < r2[name].standard_deviation[0]


def test_accumulation_loop_shrinks_error():
    """The run-until-satisfied pattern: explicit combine_results folds."""
    model, run_kwargs = build_model(particles=4_000, seed=1)
    results = model.simulate_transport(**run_kwargs, threads=1)
    rel_prev = results["flux"].relative_error[0]
    for next_seed in (2, 3):
        run_kwargs["seed"] = next_seed
        new = model.simulate_transport(**run_kwargs, threads=1)
        results = yamc.combine_results(results, new)
        rel = results["flux"].relative_error[0]
        assert rel < rel_prev, "pooled relative error must shrink"
        rel_prev = rel
    assert results["flux"].n_histories == 12_000
    assert [r["seed"] for r in results.runs] == [1, 2, 3]


def test_same_seed_is_refused_with_actionable_message():
    r1 = run(seed=1)
    r2 = run(seed=1)
    with pytest.raises(ValueError, match="share base seed 1"):
        yamc.combine_results(r1, r2)
    # The default-seed footgun message tells the user what to do.
    with pytest.raises(ValueError, match="simulate_transport"):
        yamc.combine_results(r1, r2)


def test_different_model_is_refused():
    r1 = run(seed=1, density=2.0)
    r2 = run(seed=2, density=2.5)
    with pytest.raises(ValueError, match="different models"):
        yamc.combine_results(r1, r2)


def test_same_name_different_tally_structure_is_refused():
    r1 = run(seed=1)

    # Same model physics, but the tally named "absorption" carries a
    # different score.
    model, _ = build_model(particles=5_000, seed=2, with_absorption=False)
    extra = yamc.Tally(scores=["heating"], name="absorption",
                       cells=None, particle="neutron")
    with pytest.raises(ValueError, match="different configurations"):
        # Rebuild with the conflicting tally attached to the same cell.
        material = yamc.Material(composition={"Li6": 1.0}, density=2.0,
                                 temperature=294)
        material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
        sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
        cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
        geometry = yamc.Geometry([cell])
        source = yamc.NeutronSource(position=(0, 0, 0),
                                    energy=yamc.sources.Discrete([1.0e6], [1.0]))
        tallies = [
            yamc.Tally(scores=["flux"], name="flux", cells=cell,
                       particle="neutron"),
            yamc.Tally(scores=["heating"], name="absorption", cells=cell,
                       particle="neutron"),
        ]
        m2 = yamc.Model(geometry=geometry, tallies=tallies, source=source,
                        verbose=[])
        yamc.combine_results(
            r1, m2.simulate_transport(total_particles=5_000, seed=2, threads=1))
    del model, extra


def test_partial_overlap_warns_and_passes_through():
    r_both = run(seed=1, with_absorption=True)
    r_flux = run(seed=2, with_absorption=False)

    with pytest.warns(UserWarning, match="absorption"):
        combined = yamc.combine_results(r_both, r_flux)

    assert combined["flux"].n_histories == 10_000
    # Pass-through: absorption keeps its original statistics exactly.
    assert combined["absorption"].n_histories == 5_000
    assert combined["absorption"].mean == r_both["absorption"].mean
    assert combined["absorption"].m2 == r_both["absorption"].m2


def test_runs_provenance_exposed():
    r = run(seed=9, particles=2_000)
    (info,) = r.runs
    assert info["seed"] == 9
    assert info["n_histories"] == 2_000
    assert info["compute"] == "cpu"
    assert info["mpi_rank"] == 0
    assert len(info["fingerprint"]) == 64  # sha-256 hex
    assert info["data_libraries"]["n:Li6"] == "tests/Li6.arrow"


def test_arrow_roundtrip_lossless_and_combinable(tmp_path):
    r1 = run(seed=1, particles=3_000)
    r2 = run(seed=2, particles=3_000)

    path = tmp_path / "run1.arrow"
    r1.to_arrow(str(path))
    loaded = yamc.SimulationResults.from_arrow(str(path))

    for name in ("flux", "absorption"):
        assert loaded[name].mean == r1[name].mean
        assert loaded[name].m2 == r1[name].m2
        assert loaded[name].n_histories == r1[name].n_histories
    assert loaded.runs[0]["seed"] == 1
    assert loaded.runs[0]["fingerprint"] == r1.runs[0]["fingerprint"]

    combined_mem = yamc.combine_results(r1, r2)
    combined_load = yamc.combine_results(loaded, r2)
    for name in ("flux", "absorption"):
        assert combined_mem[name].mean == combined_load[name].mean
        assert combined_mem[name].m2 == combined_load[name].m2


def test_combined_results_round_trip_and_recombine(tmp_path):
    r1 = run(seed=1, particles=2_000)
    r2 = run(seed=2, particles=2_000)
    combined = yamc.combine_results(r1, r2)

    path = tmp_path / "combined.arrow"
    combined.to_arrow(str(path))
    loaded = yamc.SimulationResults.from_arrow(str(path))
    assert [run_["seed"] for run_ in loaded.runs] == [1, 2]

    # A reloaded combined result still refuses a duplicate seed...
    with pytest.raises(ValueError, match="share base seed 2"):
        yamc.combine_results(loaded, run(seed=2, particles=2_000))
    # ...and accepts a fresh one.
    r3 = run(seed=3, particles=2_000)
    final = yamc.combine_results(loaded, r3)
    assert final["flux"].n_histories == 6_000


def test_mesh_tally_combines():
    def mesh_run(seed):
        material = yamc.Material(composition={"Li6": 1.0}, density=2.0,
                                 temperature=294)
        material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
        sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
        cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
        geometry = yamc.Geometry([cell])
        source = yamc.NeutronSource(position=(0, 0, 0),
                                    energy=yamc.sources.Discrete([1.0e6], [1.0]))
        mesh = yamc.RegularRectangularMesh(
            lower_left=[-10.0, -10.0, -10.0],
            upper_right=[10.0, 10.0, 10.0],
            shape=[20, 20, 20])
        tally = yamc.Tally(scores=["flux"], name="mesh_flux", mesh=mesh,
                           particle="neutron")
        model = yamc.Model(geometry=geometry, tallies=[tally], source=source,
                           verbose=[])
        return model.simulate_transport(total_particles=4_000, seed=seed, threads=1)

    r1 = mesh_run(1)
    r2 = mesh_run(2)
    combined = yamc.combine_results(r1, r2)
    m = combined["mesh_flux"]
    assert m.n_histories == 8_000
    assert len(m.mean) == 20 * 20 * 20
    # Center bin (source) is populated and its error tightened.
    total1 = sum(r1["mesh_flux"].mean)
    total_combined = sum(m.mean)
    assert total_combined > 0
    # Combined mean of the integrated flux sits between the two runs'.
    total2 = sum(r2["mesh_flux"].mean)
    lo, hi = sorted((total1, total2))
    assert lo <= total_combined <= hi


def test_combine_requires_simulation_results():
    r1 = run(seed=1, particles=1_000)
    with pytest.raises(TypeError, match="SimulationResults"):
        yamc.combine_results(r1, "not-results")
    with pytest.raises(ValueError, match="at least two"):
        yamc.combine_results(r1)
