import yamc


def test_reproducibility_with_same_seed():
    """Test that simulations with the same seed produce identical results"""
    # Create simple geometry
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = sphere.below

    # Create material
    material = yamc.Material(
        composition={"Li6": 1.0},
        density=10.0,
        temperature=294)
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})

    # Create cell and geometry
    cell = yamc.Cell(name="test_cell", region=region, material=material)
    geometry = yamc.Geometry(cells=[cell])

    # Source
    source = yamc.NeutronSource(position=[0.0, 0.0, 0.0], energy=yamc.sources.Discrete([1e6], [1.0]))

    # Create separate tallies for each run
    tally1 = yamc.Tally(scores=[101], name="test_absorption_1")
    tally2 = yamc.Tally(scores=[101], name="test_absorption_2")
    tally3 = yamc.Tally(scores=[101], name="test_absorption_3")

    # Run simulation 1
    model1 = yamc.Model(geometry=geometry, tallies=[tally1], source=source)
    results1 = model1.simulate_transport(total_particles=1000, seed=42)

    # Run simulation 2 with same seed
    model2 = yamc.Model(geometry=geometry, tallies=[tally2], source=source)
    results2 = model2.simulate_transport(total_particles=1000, seed=42)

    # Run simulation 3 with same seed
    model3 = yamc.Model(geometry=geometry, tallies=[tally3], source=source)
    results3 = model3.simulate_transport(total_particles=1000, seed=42)

    # Verify all three runs produced identical results

    # Check absorption mean is identical across same-seed runs
    mean1 = results1[tally1].mean[0]
    mean2 = results2[tally2].mean[0]
    mean3 = results3[tally3].mean[0]

    assert abs(mean1 - mean2) < 1e-12, f"Absorption should be nearly identical with same seed: {mean1} vs {mean2}"
    assert abs(mean1 - mean3) < 1e-12, f"Absorption should be nearly identical with same seed: {mean1} vs {mean3}"

    print("✓ Python reproducibility test passed!")
    print(f"  Run 1 - Absorption: {mean1:.6f}")
    print(f"  Run 2 - Absorption: {mean2:.6f}")
    print(f"  Run 3 - Absorption: {mean3:.6f}")


def test_different_seeds_produce_different_results():
    """Test that simulations with different seeds produce different results"""
    # Create simple geometry
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = sphere.below

    # Create material
    material = yamc.Material(
        composition={"Li6": 1.0},
        density=10.0,
        temperature=294)
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})

    # Create cell and geometry
    cell = yamc.Cell(name="test_cell", region=region, material=material)
    geometry = yamc.Geometry(cells=[cell])

    # Source
    source = yamc.NeutronSource(position=[0.0, 0.0, 0.0], energy=yamc.sources.Discrete([1e6], [1.0]))

    # Create separate tallies for each run
    tally1 = yamc.Tally(scores=[101], name="test_absorption_1")
    tally2 = yamc.Tally(scores=[101], name="test_absorption_2")

    # Run simulation with seed 42
    model1 = yamc.Model(geometry=geometry, tallies=[tally1], source=source)
    results1 = model1.simulate_transport(total_particles=1000, seed=42)

    # Run simulation with seed 123
    model2 = yamc.Model(geometry=geometry, tallies=[tally2], source=source)
    results2 = model2.simulate_transport(total_particles=1000, seed=123)

    # Verify different seeds produce different results (with high probability)
    mean1 = results1[tally1].mean[0]
    mean2 = results2[tally2].mean[0]

    assert mean1 != mean2, \
        f"Different seeds should produce different results (absorption: {mean1} vs {mean2})"

    print("✓ Different seeds test passed!")
    print(f"  Seed 42  - Absorption: {mean1:.6f}")
    print(f"  Seed 123 - Absorption: {mean2:.6f}")


if __name__ == "__main__":
    test_reproducibility_with_same_seed()
    test_different_seeds_produce_different_results()
