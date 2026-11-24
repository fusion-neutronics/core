import yamc as csg

def test_cell_fill():
    surf1 = csg.Sphere(x0=0, y0=0, z0=0, radius=1, surface_id=1)
    mat = csg.Material(
        composition={"Li6": 1.0},
        density=1.0,
        name="fuel",
        temperature=294,
    )
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    cell = csg.Cell(id=1, region=surf1.below, material=mat)
    cell.material.macroscopic_cross_section(reaction=1)
    assert cell.material is not None
    assert cell.material.name == "fuel"

def test_cell_fill_optional():
    surf1 = csg.Sphere(x0=0, y0=0, z0=0, radius=1, surface_id=1)
    cell = csg.Cell(id=2, region=surf1.below)
    assert cell.material is None
