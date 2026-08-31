"""Tests for automatic vacuum boundary (graveyard) on Arrow mesh geometries."""

import pytest
import yamc


# Path to the two-region Arrow test mesh (relative to project root)
TWO_REGION_ARROW = "crates/yamt/tests/data/two_region.arrow"
BOX_ARROW = "crates/yamt/tests/data/box.arrow"


def _dummy_materials(*names):
    """Create minimal materials (no nuclear data) for mesh loading."""
    mats = {}
    for name in names:
        m = yamc.Material(
            composition={"H": 1.0},
            density=1.0, units="g/cc",
            name=name,
        )
        mats[name] = m
    return mats


# ---------------------------------------------------------------------------
# Triangle count before and after adding vacuum boundary
# ---------------------------------------------------------------------------

class TestVacuumBoundaryTriangleCounts:
    """Verify that adding a vacuum boundary appends exactly 12 triangles."""

    def test_box_without_vacuum_boundary(self):
        mats = _dummy_materials("water")
        geom = yamc.MeshGeometry(BOX_ARROW, mats)
        # A box has 6 faces * 2 triangles = 12 triangles
        assert geom.num_triangles == 12

    def test_box_with_vacuum_boundary(self):
        mats = _dummy_materials("water")
        geom = yamc.MeshGeometry(BOX_ARROW, mats, graveyard_offset=5.0)
        # 12 original + 12 graveyard = 24 triangles
        assert geom.num_triangles == 24

    def test_two_region_without_vacuum_boundary(self):
        mats = _dummy_materials("fuel", "moderator")
        geom = yamc.MeshGeometry(TWO_REGION_ARROW, mats)
        # Should have triangles (no vacuum boundary added)
        assert geom.num_triangles > 0

    def test_two_region_with_vacuum_boundary(self):
        mats = _dummy_materials("fuel", "moderator")
        geom_before = yamc.MeshGeometry(TWO_REGION_ARROW, mats)
        geom_after = yamc.MeshGeometry(TWO_REGION_ARROW, mats, graveyard_offset=10.0)
        # Exactly 12 triangles added
        assert geom_after.num_triangles == geom_before.num_triangles + 12


# ---------------------------------------------------------------------------
# Bounding box expansion
# ---------------------------------------------------------------------------

class TestVacuumBoundaryBBox:
    """Verify bounding box expands by the specified offset."""

    def test_bounding_box_expanded(self):
        mats = _dummy_materials("water")
        geom_no_vb = yamc.MeshGeometry(BOX_ARROW, mats)
        geom_with_vb = yamc.MeshGeometry(BOX_ARROW, mats, graveyard_offset=5.0)

        bb_before = geom_no_vb.bounding_box()
        bb_after = geom_with_vb.bounding_box()

        # Each axis should expand by 5.0 on each side
        for i in range(3):
            assert bb_after.lower_left[i] == pytest.approx(
                bb_before.lower_left[i] - 5.0, abs=0.01
            )
            assert bb_after.upper_right[i] == pytest.approx(
                bb_before.upper_right[i] + 5.0, abs=0.01
            )


# ---------------------------------------------------------------------------
# Property setter rebuild
# ---------------------------------------------------------------------------

class TestVacuumBoundaryPropertySetter:
    """Test setting graveyard_offset as a property triggers rebuild."""

    def test_set_graveyard_offset_property(self):
        mats = _dummy_materials("water")
        geom = yamc.MeshGeometry(BOX_ARROW, mats)
        assert geom.num_triangles == 12
        assert geom.graveyard_offset is None

        # Set the property → triggers rebuild
        geom.graveyard_offset = 5.0
        assert geom.num_triangles == 24
        assert geom.graveyard_offset == 5.0

    def test_remove_graveyard_offset(self):
        mats = _dummy_materials("water")
        geom = yamc.MeshGeometry(BOX_ARROW, mats, graveyard_offset=5.0)
        assert geom.num_triangles == 24

        # Set to None → rebuild without vacuum boundary
        geom.graveyard_offset = None
        assert geom.num_triangles == 12


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestVacuumBoundaryErrors:
    """Test error cases for vacuum boundary."""

    def test_non_arrow_file_is_rejected(self):
        mats = _dummy_materials("water")
        with pytest.raises(ValueError, match=r"Arrow IPC meshes \(\.arrow\)"):
            yamc.MeshGeometry(
                "geometry.msh",
                mats,
                graveyard_offset=5.0,
            )
