import pytest
import yamc


def test_duplicate_filter_validation():
    """
    Test that passing conflicting keyword arguments raises a ValueError.

    Each tally can only bin over cells OR materials, not both.
    """

    sphere = yamc.Sphere(
        surface_id=1,
        x0=0.0, y0=0.0, z0=0.0, radius=1.0,
        boundary='vacuum'
    )
    region = sphere.below

    material1 = yamc.Material(
        composition={"Li6": 1.0},
        density=10.0,
        id=1)
    material1.read_nuclear_data({"Li6": "tests/Li6.arrow"})

    material2 = yamc.Material(
        composition={"Be9": 1.0},
        density=20.0,
        id=2)
    material2.read_nuclear_data({"Be9": "tests/Be9.arrow"})

    cell1 = yamc.Cell(id=1, name="cell1", region=region, material=material1)

    # Test 1: Passing both cells and materials should raise ValueError
    with pytest.raises(ValueError):
        yamc.Tally(
            scores=[101],
            name="tally with conflicting cells and materials",
            cells=cell1,
            materials=material1)

    # Test 2: Single cell via cells= kwarg should work fine
    tally4 = yamc.Tally(
        scores=[101],
        name="tally with single cell",
        cells=cell1)
    assert tally4.cells == [cell1.id]

    # Test 3: Single material via materials= kwarg should work fine
    tally5 = yamc.Tally(
        scores=[101],
        name="tally with single material",
        materials=material1)
    assert tally5.materials == [material1.id]


if __name__ == "__main__":
    test_duplicate_filter_validation()
    print("All duplicate filter validation tests passed!")
