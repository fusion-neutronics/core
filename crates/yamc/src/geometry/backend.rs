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
    /// Set when the cell beyond is only worked out if the particle takes the
    /// crossing: a discontinuous mesh relocates a particle leaving a solid,
    /// which costs a ray query that a collision before the boundary would
    /// waste. [`GeometryKind::resolve_crossing`] fills `next_cell_index`.
    pub relocate_triangle: Option<u32>,
}

/// The surface a particle has just crossed, for a boundary query made
/// straight after the crossing. Transport moves a crossing particle a
/// small fixed distance past the surface; `offset` is that distance, so
/// the surface solvers can find the crossing point again and treat the
/// crossing as an exact root (see `Surface::distance_to_surface_on`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnSurface {
    /// ID of the crossed surface.
    pub surface_id: usize,
    /// Distance moved past the crossing along the direction of flight.
    pub offset: f64,
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

/// How far along its flight a particle with no cell yet is located (see
/// [`GeometryKind::find_cell_index_along`]).
///
/// Equal to the distance below which mesh ray-fire ignores a hit, so that
/// the cell found and the first crossing agree: a particle within it of a
/// facet ahead is put past the facet, which ray-fire then ignores, and one
/// farther away is put before it, where ray-fire sees the crossing. CSG
/// re-locates spatially after every crossing, so any nudge this small does.
pub const BIRTH_NUDGE: f64 = 1e-10;

#[cfg(feature = "mesh")]
const _: () = assert!(BIRTH_NUDGE == yamt::query::ray_fire::RAY_FIRE_MIN_T);

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
    /// exit check. The mesh backend returns an empty slice:
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

    /// Segment-occlusion check for spatial tracking verification: is a
    /// foreign surface strictly inside the current cell along the accepted
    /// crossing segment? Verification targets adjacency-tracked crossings,
    /// so it covers the mesh backend and the mesh fills of a CSG geometry;
    /// ordinary CSG cells (which re-locate spatially) always return false.
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

    /// True when any cell is filled by a mesh body. The GPU
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

    /// The cell containing `point`, searched from `hint` (the cell the
    /// particle was last in), or `None` outside the geometry.
    ///
    /// CSG asks the cells `hint` has learned as neighbours first, then every
    /// cell. A mesh asks `hint`'s volume first, then every volume, by point
    /// containment: its cells carry placeholder CSG regions that contain
    /// every point, so a neighbour-list search over them would put every
    /// point in the first cell.
    pub(crate) fn locate_cell(
        &self,
        neighbor_lists: &mut crate::geometry::neighbor_lists::NeighborLists,
        point: (f64, f64, f64),
        hint: usize,
    ) -> Option<usize> {
        match self {
            GeometryKind::Csg(g) => neighbor_lists.find_cell(&g.cells, point, hint),
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => g.locate_cell(point, hint),
        }
    }

    /// The cell a particle with no cell yet starts in, at `point` heading
    /// along `direction`.
    ///
    /// A particle born exactly on a surface belongs to the cell it is
    /// heading into, as in OpenMC, and the point alone does not say which:
    /// a CSG point on a surface is in neither half-space, and a mesh point
    /// on a facet is put on one side of it whatever the direction, while
    /// the crossing it starts on is closer than mesh ray-fire looks. So the
    /// cell is read [`BIRTH_NUDGE`] along the flight, falling back to the
    /// point itself when nothing is found there. Away from surfaces the two
    /// are the same cell.
    pub fn find_cell_index_along(
        &self,
        point: (f64, f64, f64),
        direction: [f64; 3],
    ) -> Option<usize> {
        let ahead = (
            point.0 + BIRTH_NUDGE * direction[0],
            point.1 + BIRTH_NUDGE * direction[1],
            point.2 + BIRTH_NUDGE * direction[2],
        );
        self.find_cell_index(ahead)
            .or_else(|| self.find_cell_index(point))
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
    ///
    /// `on_surface` names the surface the particle has just crossed, if
    /// any (CSG only; mesh crossings are resolved by triangle).
    pub fn closest_boundary(
        &self,
        cell_index: usize,
        position: [f64; 3],
        direction: [f64; 3],
        on_surface: Option<OnSurface>,
    ) -> Option<BoundaryHit> {
        self.closest_boundary_within(cell_index, position, direction, on_surface, f64::INFINITY)
    }

    /// [`GeometryKind::closest_boundary`] for a caller that only needs
    /// crossings nearer than `cap`: a hit at or beyond `cap` means none
    /// before it (only free-form CSG surfaces use the cap).
    #[inline]
    pub fn closest_boundary_within(
        &self,
        cell_index: usize,
        position: [f64; 3],
        direction: [f64; 3],
        on_surface: Option<OnSurface>,
        cap: f64,
    ) -> Option<BoundaryHit> {
        match self {
            GeometryKind::Csg(g) => {
                g.closest_boundary_within(cell_index, position, direction, on_surface, cap)
            }
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => g.closest_boundary(cell_index, position, direction),
        }
    }

    /// Work out the cell beyond a crossing the particle is about to take, when
    /// [`GeometryKind::closest_boundary`] deferred it (see
    /// [`BoundaryHit::relocate_triangle`]). No-op otherwise.
    #[cfg_attr(not(feature = "mesh"), allow(unused_variables))]
    #[inline]
    pub fn resolve_crossing(
        &self,
        cell_index: usize,
        hit: &mut BoundaryHit,
        position: [f64; 3],
        direction: [f64; 3],
    ) {
        if hit.relocate_triangle.is_none() {
            return;
        }
        match self {
            #[cfg(feature = "mesh")]
            GeometryKind::Csg(g) => g.resolve_fill_crossing(cell_index, hit, position, direction),
            #[cfg(not(feature = "mesh"))]
            GeometryKind::Csg(_) => {}
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(g) => g.resolve_crossing(cell_index, hit, position, direction),
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
