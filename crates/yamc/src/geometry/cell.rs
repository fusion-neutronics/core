use crate::geo::{BoundingBox, FlatRegion, Region};
use std::sync::Arc;

/// A Cell represents a geometric region
/// Cells are defined by:
/// - A region (combination of surfaces using boolean operations)
/// - A name for identification
///
/// Serialization goes through [`CellSerde`] -- only the user-supplied
/// fields (id, name, region, material_idx, volume) are persisted; the
/// cached surfaces + flat-RPN region are rebuilt from `region` on
/// deserialize via `Cell::new`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(into = "CellSerde", from = "CellSerde")]
pub struct Cell {
    pub cell_id: Option<u32>,
    pub name: Option<String>,
    pub region: Region,
    /// Index into `Geometry.materials` (or the matching mesh-geometry store).
    /// `None` for void cells. This is the only handle the hot path uses.
    pub material_idx: Option<u32>,
    pub volume: Option<f64>,
    /// Role in a hybrid CSG+mesh fill: `Host` for a cell
    /// filled by a mesh body, `Embedded` for a cell synthesized from one
    /// mesh volume of a fill. `None` for ordinary cells. Not serialized:
    /// a geometry with fills only fingerprints (it cannot round-trip).
    #[cfg(feature = "mesh")]
    pub(crate) fill_role: Option<crate::geometry::fill::FillRole>,
    /// Cached list of surfaces with sense for fast iteration (avoids repeated collection)
    pub(crate) cached_surfaces: Vec<(Arc<crate::geo::Surface>, bool)>,
    /// Whether any of `cached_surfaces` is free-form (searched for
    /// iteratively), so `closest_surface` caps those searches.
    pub(crate) has_free_form: bool,
    /// For a cell with free-form surfaces, the region's bounding box grown
    /// for round-off, if it is finite: `closest_surface` stops their
    /// searches where a ray leaves it.
    pub(crate) search_box: Option<BoundingBox>,
    /// Flattened RPN representation of the region for iterative evaluation.
    /// Built once at construction time; used by `contains()` and `distance_to_surface()`.
    pub(crate) flat_region: FlatRegion,
}

/// On-disk shape of `Cell` -- only the user-supplied fields. Caches are
/// rebuilt on `From<CellSerde> for Cell` via `Cell::new`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct CellSerde {
    pub cell_id: Option<u32>,
    pub name: Option<String>,
    pub region: Region,
    pub material_idx: Option<u32>,
    pub volume: Option<f64>,
}

impl From<Cell> for CellSerde {
    fn from(cell: Cell) -> Self {
        Self {
            cell_id: cell.cell_id,
            name: cell.name,
            region: cell.region,
            material_idx: cell.material_idx,
            volume: cell.volume,
        }
    }
}

impl From<CellSerde> for Cell {
    fn from(spec: CellSerde) -> Self {
        let mut cell = Cell::new(spec.cell_id, spec.region, spec.name, spec.material_idx);
        cell.volume = spec.volume;
        cell
    }
}

impl Cell {
    /// Find the closest surface of this cell to a point along a direction (first intersection with any region surface).
    /// Returns `(BoundaryType, distance, surface_id)` to avoid an Arc clone per call.
    /// `on_surface` is the surface the particle has just crossed, if any:
    /// that surface's distance treats the crossing as an exact root.
    #[inline]
    pub fn closest_surface(
        &self,
        point: [f64; 3],
        direction: [f64; 3],
        on_surface: Option<crate::geometry::backend::OnSurface>,
    ) -> Option<(crate::geo::BoundaryType, f64, Option<usize>)> {
        self.closest_surface_within(point, direction, on_surface, f64::INFINITY)
    }

    /// [`Cell::closest_surface`] for a caller that only needs crossings
    /// nearer than `cap` (transport passes its collision distance): the
    /// free-form surfaces stop searching there, and if nothing is found
    /// before it the answer is a crossing at `cap` itself, which such a
    /// caller treats as "none before the cap".
    ///
    /// A particle in the cell leaves it before it leaves the cell's
    /// bounding box, so where that box is finite the free-form searches
    /// first stop where the ray leaves the box. A crossing found before
    /// that point is the closest (no search stopped short of it); if there
    /// is none (the particle is not quite in the cell, say), the searches
    /// run again up to `cap`.
    pub fn closest_surface_within(
        &self,
        point: [f64; 3],
        direction: [f64; 3],
        on_surface: Option<crate::geometry::backend::OnSurface>,
        cap: f64,
    ) -> Option<(crate::geo::BoundaryType, f64, Option<usize>)> {
        let exit = self.box_exit(point, direction);
        if exit < cap
            && let (Some(hit), _) = self.nearest(point, direction, on_surface, exit)
            && hit.1 <= exit
        {
            return Some(hit);
        }
        match self.nearest(point, direction, on_surface, cap) {
            (Some(hit), _) => Some(hit),
            (None, true) => Some((crate::geo::BoundaryType::Transmission, cap, None)),
            (None, false) => None,
        }
    }

    /// The nearest crossing beyond 1e-10 among the cell's surfaces, the
    /// free-form ones searched no further than `limit` (closed-form ones
    /// are solved outright), and whether such a search stopped at `limit`
    /// without finding one.
    fn nearest(
        &self,
        point: [f64; 3],
        direction: [f64; 3],
        on_surface: Option<crate::geometry::backend::OnSurface>,
        limit: f64,
    ) -> (Option<(crate::geo::BoundaryType, f64, Option<usize>)>, bool) {
        let mut min_dist = f64::INFINITY;
        let mut capped = false;
        let mut closest_boundary = None;
        let mut closest_surface_id = None;

        // Closed-form surfaces first, then free-form ones, each of which
        // stops searching at the nearest crossing found so far: a crossing
        // beyond it could not be the closest anyway. The free-form surface
        // just crossed goes last, as its next crossing is usually the
        // farthest.
        for pass in 0..3 {
            if pass > 0 && !self.has_free_form {
                break;
            }
            for (surface_arc, _sense) in &self.cached_surfaces {
                let surface: &crate::geo::Surface = surface_arc.as_ref();
                let offset = on_surface
                    .filter(|on| surface.surface_id == Some(on.surface_id))
                    .map(|on| on.offset);
                let wanted = match pass {
                    0 => !surface.is_free_form(),
                    1 => surface.is_free_form() && offset.is_none(),
                    _ => surface.is_free_form() && offset.is_some(),
                };
                if !wanted {
                    continue;
                }
                let dist = if pass == 0 {
                    surface.distance_to_surface_on(point, direction, offset)
                } else {
                    let limit = min_dist.min(limit);
                    let dist = surface.distance_to_surface_capped(point, direction, offset, limit);
                    capped |= dist.is_none() && limit < f64::INFINITY;
                    dist
                };
                if let Some(dist) = dist
                    && dist > 1e-10
                    && dist < min_dist
                {
                    min_dist = dist;
                    closest_boundary = Some(surface.boundary.clone());
                    closest_surface_id = surface.surface_id;
                }
            }
        }

        let closest = closest_boundary.map(|bt| (bt, min_dist, closest_surface_id));
        (closest, capped)
    }

    /// Ray parameter at which `point + t * direction` leaves `search_box`;
    /// infinite with no box, or from a point outside it.
    fn box_exit(&self, point: [f64; 3], direction: [f64; 3]) -> f64 {
        let Some(bx) = &self.search_box else {
            return f64::INFINITY;
        };
        let mut exit = f64::INFINITY;
        for k in 0..3 {
            let (lo, hi) = (bx.lower_left[k], bx.upper_right[k]);
            if !(point[k] >= lo && point[k] <= hi) {
                return f64::INFINITY;
            }
            if direction[k] > 0.0 {
                exit = exit.min((hi - point[k]) / direction[k]);
            } else if direction[k] < 0.0 {
                exit = exit.min((lo - point[k]) / direction[k]);
            }
        }
        exit
    }

    /// Compute the distance to the closest surface from a point along a direction.
    /// Uses flat region evaluation to check exit (equivalent to `!contains(p_next)`).
    pub fn distance_to_surface(&self, point: [f64; 3], direction: [f64; 3]) -> Option<f64> {
        let mut min_dist = f64::INFINITY;
        let eps = 1e-8;

        // Use cached surfaces for fast iteration
        for (surface_arc, _sense) in &self.cached_surfaces {
            let surface: &crate::geo::Surface = surface_arc.as_ref();
            if let Some(dist) = surface.distance_to_surface(point, direction)
                && dist > 1e-10
                && dist < min_dist
            {
                // Check if crossing this surface actually exits the region
                let p_next = (
                    point[0] + direction[0] * (dist + eps),
                    point[1] + direction[1] * (dist + eps),
                    point[2] + direction[2] * (dist + eps),
                );
                if !self.flat_region.contains(p_next) {
                    min_dist = dist;
                }
            }
        }

        if min_dist < f64::INFINITY {
            Some(min_dist)
        } else {
            None
        }
    }
    /// Create a new cell with a region and optional material index (fill).
    /// The index refers to a slot in the parent `Geometry.materials` (or the
    /// mesh-geometry material store). `None` marks a void cell.
    pub fn new(
        cell_id: Option<u32>,
        region: Region,
        name: Option<String>,
        material_idx: Option<u32>,
    ) -> Self {
        // Pre-cache surfaces for fast iteration during transport
        let cached_surfaces = region.surfaces_with_sense();
        let has_free_form = cached_surfaces.iter().any(|(s, _)| s.is_free_form());
        let search_box = if has_free_form {
            search_box(region.bounding_box())
        } else {
            None
        };
        // Compile region to flat RPN for iterative containment checks
        let flat_region = region.flatten();
        Cell {
            cell_id,
            name,
            region,
            material_idx,
            volume: None,
            #[cfg(feature = "mesh")]
            fill_role: None,
            cached_surfaces,
            has_free_form,
            search_box,
            flat_region,
        }
    }

    /// Set the cell ID
    pub fn set_cell_id(&mut self, cell_id: u32) {
        self.cell_id = Some(cell_id);
    }

    /// Get the cell ID
    pub fn get_cell_id(&self) -> Option<u32> {
        self.cell_id
    }

    /// Check if a point is inside this cell's region (iterative flat evaluation).
    #[inline]
    pub fn contains(&self, point: (f64, f64, f64)) -> bool {
        self.flat_region.contains(point)
    }

    /// True when this cell takes part in a mesh fill: the
    /// host of a fill, or a cell synthesized from one of its volumes.
    /// Such cells only make sense inside the geometry that owns the
    /// fill; they cannot seed a new geometry.
    #[cfg(feature = "mesh")]
    pub fn in_mesh_fill(&self) -> bool {
        self.fill_role.is_some()
    }
}

/// `bx` grown by far more than the round-off in its coordinates (a box
/// computed from the surfaces' parameters, or a crossing on its face, can
/// be out by that much), if it is finite.
fn search_box(bx: BoundingBox) -> Option<BoundingBox> {
    if !bx.is_finite() {
        return None;
    }
    let scale = bx
        .lower_left
        .iter()
        .chain(&bx.upper_right)
        .fold(1.0_f64, |m, v| m.max(v.abs()));
    let pad = 1e-9 * scale;
    Some(BoundingBox::new(
        bx.lower_left.map(|v| v - pad),
        bx.upper_right.map(|v| v + pad),
    ))
}
