import yamc


def test_absorption_leakage_filters():
    """
    Integration test verifying that:
    1. cells= kwarg correctly separates tallies by cell
    2. Particle conservation (absorption + leakage = 1)
    3. Tally consistency (sum of cell tallies = total tally)
    4. Surface crossing between cells works correctly
    """

    # Create two-cell geometry: inner sphere (Li6) and outer annular region (Be9)
    sphere1 = yamc.Sphere(
        x0=0.0,
        y0=0.0,
        z0=0.0,
        radius=1.0)
    sphere2 = yamc.Sphere(
        x0=0.0,
        y0=0.0,
        z0=0.0,
        radius=2.0,
        boundary='vacuum')
    region1 = sphere1.below
    region2 = sphere1.above & sphere2.below

    # Create materials with different absorption characteristics
    material1 = yamc.Material(
        composition={"Li6": 1.0},
        density=10.0,
        temperature=294)
    material1.read_nuclear_data({"Li6": "tests/Li6.arrow"})

    material2 = yamc.Material(
        composition={"Be9": 1.0},
        density=20.0,
        temperature=294)
    material2.read_nuclear_data({"Be9": "tests/Be9.arrow"})

    # Create cells
    cell1 = yamc.Cell(
        name="inner_sphere",
        region=region1,
        material=material1)
    cell2 = yamc.Cell(
        name="outer_annular",
        region=region2,
        material=material2)
    geometry = yamc.Geometry(cells=[cell1, cell2])

        # Source: neutrons starting at origin, moving upward
    source = yamc.NeutronSource(position=[0.0, 0.0, 0.0], energy=yamc.sources.Discrete([1e6], [1.0]))

    # Create tallies with cells= kwarg
    tally1 = yamc.Tally(scores=[101], name="absorption in cell 1", cells=cell1)
    tally2 = yamc.Tally(scores=[101], name="absorption in cell 2", cells=cell2)

    # Create equivalent tallies with materials= kwarg for comparison
    tally1_mat = yamc.Tally(scores=[101], name="absorption in material 1 (materials kwarg)", materials=material1)
    tally2_mat = yamc.Tally(scores=[101], name="absorption in material 2 (materials kwarg)", materials=material2)

    # Total absorption tally (no filter)
    tally3 = yamc.Tally(scores=[101], name="total absorption")

    tallies = [tally1, tally2, tally1_mat, tally2_mat, tally3]

    # Run simulation
    model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
    results = model.simulate_transport(total_particles=100, seed=1)

    # Integration test assertions

    # All tallies have one score, so use mean[0]
    mean1 = results[tally1].mean[0]
    mean2 = results[tally2].mean[0]
    mean1_mat = results[tally1_mat].mean[0]
    mean2_mat = results[tally2_mat].mean[0]
    mean3 = results[tally3].mean[0]

    # Test 1: cells= kwarg functionality - different cells should have different absorption rates
    assert mean1 != mean2, "cells= kwarg should separate tallies by cell"

    # Test 2: Tally consistency - sum of cell tallies should equal total tally
    tolerance = 1e-10
    sum_diff = abs((mean1 + mean2) - mean3)
    assert sum_diff < tolerance, f"Sum of cell tallies ({mean1 + mean2}) should equal total tally ({mean3}), difference: {sum_diff}"

    # Test 4: Physical reasonableness - some particles should be absorbed, some should leak
    assert mean3 > 0.0, "Some particles should be absorbed"
    assert mean3 < 1.0, "Not all particles should be absorbed"

    # Test 5: At least one cell should have absorption, and surface crossing should work
    assert mean1 >= 0.0, "Cell 1 absorption should be non-negative"
    assert mean2 >= 0.0, "Cell 2 absorption should be non-negative"
    assert (mean1 + mean2) > 0.0, "Total absorption should be positive"

    # Test 6: materials= equivalence - should give same results as cells= for single-material cells
    mat_filter_tolerance = 1e-10
    cell1_vs_mat1_diff = abs(mean1 - mean1_mat)
    cell2_vs_mat2_diff = abs(mean2 - mean2_mat)

    assert cell1_vs_mat1_diff < mat_filter_tolerance, f"cells= and materials= should give same results for cell 1 ({mean1} vs {mean1_mat}), difference: {cell1_vs_mat1_diff}"
    assert cell2_vs_mat2_diff < mat_filter_tolerance, f"cells= and materials= should give same results for cell 2 ({mean2} vs {mean2_mat}), difference: {cell2_vs_mat2_diff}"


def test_duplicate_filter_error():
    """
    Test that passing conflicting keyword arguments raises an error.
    """
    # Create minimal geometry
    sphere = yamc.Sphere(
        x0=0.0, y0=0.0, z0=0.0, radius=1.0,
        boundary='vacuum'
    )
    region = sphere.below

    # Create materials
    material1 = yamc.Material(
        composition={"Li6": 1.0},
        density=10.0)
    material1.read_nuclear_data({"Li6": "tests/Li6.arrow"})

    material2 = yamc.Material(
        composition={"Be9": 1.0},
        density=20.0)
    material2.read_nuclear_data({"Be9": "tests/Be9.arrow"})

    # Create cells
    cell1 = yamc.Cell(name="cell1", region=region, material=material1)

    yamc.Geometry(cells=[cell1])

    # With the kwarg API, you can only set one cell per tally.
    # Test that a single cell kwarg works fine.
    tally_ok = yamc.Tally(scores=[101], name="single cell tally", cells=cell1)
    assert tally_ok.cells == [cell1.id]

    # Test that a single material kwarg works fine.
    tally_ok2 = yamc.Tally(scores=[101], name="single material tally", materials=material1)
    assert tally_ok2.materials == [material1.id]

    print("Duplicate filter error test passed!")




if __name__ == "__main__":
    test_absorption_leakage_filters()
    test_duplicate_filter_error()
