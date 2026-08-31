//! Geometry backend abstraction.
//!
//! `GeometryKind` provides enum-based dispatch between the CSG geometry
//! and the mesh-based geometry (powered by `yamt`). The transport loop
//! calls methods on `GeometryKind` without knowing which backend is
//! active.

use crate::geo::BoundaryType;
use crate::geometry::cell::Cell;
use crate::geometry::Geometry;
use std::sync::Arc;
use yamc_materials::material::Material;

#[cfg(feature = "mesh")]
use crate::geometry::mesh::MeshGeometry;

/// Result of a closest-boundary query.
#[derive(Debug, Clone)]
pub struct BoundaryHit {
    /// Distance along the ray to the boundary.
    pub distance: f64,
    /// Boundary condition at the crossing surface.
    pub boundary: BoundaryType,
    /// For mesh geometry: the cell index on the other side of the surface.
    /// `None` for CSG geometry or vacuum boundaries (where it's not needed).
    pub next_cell_index: Option<usize>,
    /// Surface ID of the crossed surface (for diagnostics).
    pub surface_id: Option<usize>,
}

/// Backend-agnostic geometry. The transport loop only sees this type.
#[derive(serde::Serialize, serde::Deserialize)]
pub enum GeometryKind {
    /// Constructive Solid Geometry (planes, spheres, cylinders).
    Csg(Geometry),
    /// Mesh-based geometry (triangulated surfaces via `yamt`).
    #[cfg(feature = "mesh")]
    Mesh(Box<MeshGeometry>),
}

impl GeometryKind {
    // ---- Cell access (both backends store Vec<Cell>) -------------------------

    /// Number of cells (volumes) in the geometry.
    pub fn num_cells(&self) -> usize {
        match self {
            GeometryKind::Csg(g) => g.cells.len(),
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => g.cells.len(),
        }
    }

    /// Immutable access to the cells slice.
    pub fn cells(&self) -> &[Cell] {
        match self {
            GeometryKind::Csg(g) => &g.cells,
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => &g.cells,
        }
    }

    /// The geometry's vacuum-tagged surfaces, for the Woodcock flight
    /// exit check (issue #360). The mesh backend returns an empty slice:
    /// its `closest_boundary` / ray-fire path already classifies vacuum
    /// boundaries, and exact mesh flight truncation is future work.
    pub(crate) fn vacuum_surfaces(&self) -> &[std::sync::Arc<crate::geo::Surface>] {
        match self {
            GeometryKind::Csg(g) => g.vacuum_surfaces(),
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(_) => &[],
        }
    }

    /// Mutable access to the cells slice (for transmutation material updates).
    pub fn cells_mut(&mut self) -> &mut [Cell] {
        match self {
            GeometryKind::Csg(g) => &mut g.cells,
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => &mut g.cells,
        }
    }

    /// Flat material store. `Cell.material_idx` is a slot index into this slice.
    pub fn materials(&self) -> &[Arc<Material>] {
        match self {
            GeometryKind::Csg(g) => &g.materials,
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => &g.materials,
        }
    }

    /// Mutable access to the material store (for transmutation composition updates).
    pub fn materials_mut(&mut self) -> &mut Vec<Arc<Material>> {
        match self {
            GeometryKind::Csg(g) => &mut g.materials,
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => &mut g.materials,
        }
    }

    /// Resolve a cell's material through its `material_idx`.
    #[inline]
    pub fn material_for(&self, cell: &Cell) -> Option<&Arc<Material>> {
        cell.material_idx
            .and_then(|i| self.materials().get(i as usize))
    }

    // ---- Geometry queries ----------------------------------------------------

    /// Segment-occlusion check for spatial tracking verification
    /// (issue #254): is a foreign surface strictly inside the current
    /// cell along the accepted crossing segment? Verification targets
    /// adjacency-tracked crossings, so it covers the mesh backend and
    /// the mesh fills of a CSG geometry; ordinary CSG cells (which
    /// re-locate spatially) always return false.
    pub(crate) fn crossing_blocked(
        &self,
        cell_index: usize,
        origin: [f64; 3],
        direction: [f64; 3],
        t_max: f64,
    ) -> bool {
        match self {
            GeometryKind::Csg(g) => g.crossing_blocked(cell_index, origin, direction, t_max),
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => g.crossing_blocked(cell_index, origin, direction, t_max),
        }
    }

    /// True when any cell is filled by a mesh body (issue #232). The GPU
    /// kernel and Woodcock/hybrid delta tracking cannot see mesh fills
    /// and reject such models up front.
    pub fn has_mesh_fills(&self) -> bool {
        match self {
            GeometryKind::Csg(g) => g.has_mesh_fills(),
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(_) => false,
        }
    }

    /// Refine a spatially-located cell through any mesh fill: a point in
    /// a fill host's region resolves to the embedded mesh-volume cell it
    /// falls in (or stays with the host, the complement). No-op for the
    /// mesh backend and for ordinary CSG cells.
    #[cfg_attr(not(feature = "mesh"), allow(unused_variables))]
    #[inline]
    pub(crate) fn resolve_fill_at(&self, cell_index: usize, point: (f64, f64, f64)) -> usize {
        match self {
            #[cfg(feature = "mesh")]
            GeometryKind::Csg(g) => g.resolve_fill_at(cell_index, point),
            #[cfg(not(feature = "mesh"))]
            GeometryKind::Csg(_) => cell_index,
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(_) => cell_index,
        }
    }

    pub fn find_cell_index(&self, point: (f64, f64, f64)) -> Option<usize> {
        match self {
            GeometryKind::Csg(g) => g.find_cell_index(point),
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => g.find_cell_index(point),
        }
    }

    /// Find the first cell containing the given point.
    pub fn find_cell(&self, point: (f64, f64, f64)) -> Option<&Cell> {
        self.find_cell_index(point).map(|idx| &self.cells()[idx])
    }

    /// Find the closest boundary crossing from a position along a direction.
    ///
    /// For CSG: tests all surfaces of the cell and returns the nearest
    /// exit; cells involved in a mesh fill additionally test the next
    /// mesh-surface crossing and return the nearer of the two.
    /// For Mesh: fires a ray in the volume using the surface BVH.
    pub fn closest_boundary(
        &self,
        cell_index: usize,
        position: [f64; 3],
        direction: [f64; 3],
    ) -> Option<BoundaryHit> {
        match self {
            GeometryKind::Csg(g) => g.closest_boundary(cell_index, position, direction),
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => g.closest_boundary(cell_index, position, direction),
        }
    }

    /// Compute the bounding box of the geometry.
    pub fn bounding_box(&self) -> crate::geo::BoundingBox {
        match self {
            GeometryKind::Csg(g) => g.bounding_box(),
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => g.bounding_box(),
        }
    }
}

impl std::fmt::Debug for GeometryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryKind::Csg(g) => f.debug_tuple("Csg").field(g).finish(),
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(_) => f.debug_struct("Mesh").finish_non_exhaustive(),
        }
    }
}

impl Clone for GeometryKind {
    fn clone(&self) -> Self {
        match self {
            GeometryKind::Csg(g) => GeometryKind::Csg(g.clone()),
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => GeometryKind::Mesh(g.clone()),
        }
    }
}
