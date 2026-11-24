"""PyO3-binding tests for the `cells=` / `materials=` list kwargs on Tally.

Behavioral correctness of multi-cell / multi-material scoring lives in
``crates/yamc/tests/test_tally_multi_cell_material.rs`` and the filter unit
tests. This file only covers things that exist at the Python binding layer:

- mutual exclusivity of ``cells=`` vs ``materials=``
- bare-single fallback (``cells=cell1`` without wrapping in a list)
- empty-list rejection
- the getter ``tally.cells`` returns a list (always, when set)
"""

import pytest
import yamc


def _make_cell(cell_id: int) -> yamc.Cell:
    s = yamc.Sphere(radius=float(cell_id), boundary='vacuum')
    return yamc.Cell(region=s.below, id=cell_id)


def _make_material(material_id: int) -> yamc.Material:
    m = yamc.Material(composition={"Li6": 1.0}, density=2.0, temperature=294)
    m.id = material_id
    return m


class TestKwargParsing:
    def test_cells_bare_single_fallback(self):
        c = _make_cell(1)
        t = yamc.Tally(scores=['flux'], cells=c)
        assert t.cells == [1]

    def test_cells_list(self):
        c1, c2 = _make_cell(1), _make_cell(2)
        t = yamc.Tally(scores=['flux'], cells=[c1, c2])
        assert t.cells == [1, 2]

    def test_materials_bare_single_fallback(self):
        m = _make_material(7)
        t = yamc.Tally(scores=['flux'], materials=m)
        assert t.materials == [7]

    def test_materials_list(self):
        m1, m2 = _make_material(1), _make_material(2)
        t = yamc.Tally(scores=['flux'], materials=[m1, m2])
        assert t.materials == [1, 2]


class TestMutualExclusivity:
    def test_cells_and_materials_rejected(self):
        c = _make_cell(1)
        m = _make_material(1)
        with pytest.raises(ValueError, match="mutually exclusive"):
            yamc.Tally(scores=['flux'], cells=[c], materials=[m])

    def test_empty_cells_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            yamc.Tally(scores=['flux'], cells=[])

    def test_empty_materials_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            yamc.Tally(scores=['flux'], materials=[])
