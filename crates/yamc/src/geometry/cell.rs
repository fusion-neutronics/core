use crate::geo::{FlatRegion, Region};
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
    /// Role in a hybrid CSG+mesh fill (issue #232): `Host` for a cell
    /// filled by a mesh body, `Embedded` for a cell synthesized from one
    /// mesh volume of a fill. `None` for ordinary cells. Not serialized:
    /// a geometry with fills only fingerprints (it cannot round-trip).
    #[cfg(feature = "mesh")]
    pub(crate) fill_role: Option<crate::geometry::fill::FillRole>,
    /// Cached list of surfaces with sense for fast iteration (avoids repeated collection)
    pub(crate) cached_surfaces: Vec<(Arc<crate::geo::Surface>, bool)>,
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
    #[inline]
    pub fn closest_surface(
        &self,
        point: [f64; 3],
        direction: [f64; 3],
    ) -> Option<(crate::geo::BoundaryType, f64, Option<usize>)> {
        let mut min_dist = f64::INFINITY;
        let mut closest_boundary = None;
        let mut closest_surface_id = None;

        // Use cached surfaces for fast iteration (avoids repeated collection)
        for (surface_arc, _sense) in &self.cached_surfaces {
            let surface: &crate::geo::Surface = surface_arc.as_ref();
            if let Some(dist) = surface.distance_to_surface(point, direction) {
                if dist > 1e-10 && dist < min_dist {
                    min_dist = dist;
                    closest_boundary = Some(surface.boundary.clone());
                    closest_surface_id = surface.surface_id;
                }
            }
        }

        closest_boundary.map(|bt| (bt, min_dist, closest_surface_id))
    }

    /// Compute the distance to the closest surface from a point along a direction.
    /// Uses flat region evaluation to check exit (equivalent to `!contains(p_next)`).
    pub fn distance_to_surface(&self, point: [f64; 3], direction: [f64; 3]) -> Option<f64> {
        let mut min_dist = f64::INFINITY;
        let eps = 1e-8;

        // Use cached surfaces for fast iteration
        for (surface_arc, _sense) in &self.cached_surfaces {
            let surface: &crate::geo::Surface = surface_arc.as_ref();
            if let Some(dist) = surface.distance_to_surface(point, direction) {
                if dist > 1e-10 && dist < min_dist {
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

    /// True when this cell takes part in a mesh fill (issue #232): the
    /// host of a fill, or a cell synthesized from one of its volumes.
    /// Such cells only make sense inside the geometry that owns the
    /// fill; they cannot seed a new geometry.
    #[cfg(feature = "mesh")]
    pub fn in_mesh_fill(&self) -> bool {
        self.fill_role.is_some()
    }
}
