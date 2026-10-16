"""The overlap gate in ``CadToYamc.mesh``.

The imprint shares coinciding surfaces but does not split solids whose
surfaces cross, so overlapping solids come out of it whole and still overlap
in the mesh. The gate measures the mesh and refuses such overlaps by default.
"""

import pytest

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc, OverlappingSolidsError  # noqa: E402


def _assembly(**parts):
    assembly = cq.Assembly()
    for name, shape in parts.items():
        assembly.add(shape, name=name)
    return assembly


def _poking():
    """A rod poking out of a box, neither cut from the other."""
    return _assembly(
        box=cq.Workplane().box(20, 20, 20),
        rod=cq.Workplane().circle(4).extrude(30).translate((0, 0, -5)),
    )


def test_an_imprinted_mesh_whose_solids_cross_is_refused():
    """A rod poking out of a box comes back from the imprint as the whole box
    and the whole rod, overlapping; overlaps="ignore" lets it through."""
    c2y = CadToYamc()
    c2y.add_cadquery_object(_poking(), material_tags=["box", "rod"])
    with pytest.raises(OverlappingSolidsError, match="(?s)'box'.*'rod'.*does not split"):
        c2y.mesh()

    c2y = CadToYamc()
    c2y.add_cadquery_object(_poking(), material_tags=["box", "rod"])
    mesh = c2y.mesh(overlaps="ignore")
    assert mesh.num_solids == 2


def test_cut_solids_pass_the_gate():
    """The same rod with the box cut around it: the imprint shares the
    surfaces where they meet, and nothing overlaps."""
    box = cq.Workplane().box(20, 20, 20)
    rod = cq.Workplane().circle(4).extrude(30).translate((0, 0, -5))
    c2y = CadToYamc()
    c2y.add_cadquery_object(_assembly(box=box.cut(rod), rod=rod), material_tags=["box", "rod"])
    mesh = c2y.mesh()
    assert mesh.num_solids == 2


def test_overlaps_option_is_checked():
    c2y = CadToYamc()
    c2y.add_cadquery_object(_poking(), material_tags=["box", "rod"])
    with pytest.raises(ValueError, match="overlaps must be"):
        c2y.mesh(overlaps="warn")
    with pytest.raises(ValueError, match="overlaps must be"):
        c2y.mesh_surfaces(overlaps="warn")
