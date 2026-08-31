use crate::geometry::cell::Cell;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::Arc; // Used for surface deduplication
use yamc_geo::Bvh;
use yamc_materials::material::Material;

/// Geometry is a collection of cells plus the materials they reference.
/// Each cell stores a `material_idx` into `materials`; the Vec layout is the
/// GPU-friendly representation needed by device-side transport kernels.
///
/// **Construction**: must go through [`Geometry::new`] -- that's the only
/// path that runs the validation pass (duplicate cell / material /
/// surface IDs, `material_idx` bounds, and the BVH build). The
/// `#[non_exhaustive]` attribute forbids struct-literal construction
/// from outside this crate, so external callers cannot accidentally
/// skip validation. (A latent duplicate-`surface_id` bug in
/// `examples/flux_example.rs` survived for some time precisely because
/// struct-literal construction bypassed the validator; this lock
/// prevents that bug class.)
/// Serialization goes via [`GeometrySerde`] -- only cells + materials.
/// The BVH is rebuilt from cell bounding boxes on `Geometry::new`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(into = "GeometrySerde", try_from = "GeometrySerde")]
#[non_exhaustive]
pub struct Geometry {
    pub cells: Vec<Cell>,
    /// Flat material store. `Cell.material_idx` is a slot index into this Vec.
    /// Callers are responsible for deduplication (one Arc per unique Material).
    pub materials: Vec<Arc<Material>>,
    /// Mesh bodies embedded in CSG cells (issue #232). Each fill's mesh
    /// volumes are flattened into `cells` as ordinary cells; the entries
    /// here carry the mesh, its placement transform and the host/embedded
    /// index mapping for the fill-aware geometry queries.
    #[cfg(feature = "mesh")]
    pub fills: Vec<crate::geometry::fill::MeshFill>,
    /// BVH over cell bounding boxes for O(log N) point-in-cell queries.
    /// Built once at `Geometry::new`; cells with non-finite bounding boxes
    /// (infinite half-spaces, complements without finite bounds, or empty
    /// regions) live in `bvh.unbounded` and are tested last. Cells
    /// embedded by a mesh fill are excluded: point location finds their
    /// host cell spatially and resolves the fill afterwards.
    pub bvh: Bvh,
    /// The geometry's vacuum-tagged surfaces (deduplicated), precomputed
    /// at `Geometry::new` for the Woodcock flight exit check (issue
    /// #360). Rebuilt on deserialization via `GeometrySerde`, never
    /// serialized.
    pub(crate) vacuum_surfaces: Vec<Arc<crate::geo::Surface>>,
}

/// Lossy identity summary of a mesh fill for model fingerprinting
/// (combine_results). Mirrors the fingerprint-only serialization of
/// `MeshGeometry`: it distinguishes geometrically different fills but
/// cannot reconstruct one, so deserializing a geometry with fills fails
/// loudly. Lives outside the `mesh` feature gate so non-mesh builds
/// still reject such JSON instead of silently dropping the bodies.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeshFillFingerprint {
    pub host_cell_index: usize,
    pub first_volume_cell: usize,
    pub translation: [f64; 3],
    pub rotation_degrees: [f64; 3],
    pub num_volumes: u32,
    pub global_aabb: [f64; 6],
    pub volume_measures: Vec<f64>,
    pub volume_materials: Vec<Option<String>>,
}

/// On-disk shape of [`Geometry`]. Materials are stored by value (each
/// `Arc` is serialized once via serde's `rc` feature); the BVH is
/// rebuilt by `Geometry::new` on load. Mesh fills serialize only a
/// fingerprint summary, so a geometry with fills cannot be deserialized.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct GeometrySerde {
    pub cells: Vec<Cell>,
    pub materials: Vec<Arc<Material>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fills: Vec<MeshFillFingerprint>,
}

impl From<Geometry> for GeometrySerde {
    fn from(g: Geometry) -> Self {
        Self {
            cells: g.cells,
            materials: g.materials,
            #[cfg(feature = "mesh")]
            fills: g.fills.iter().map(MeshFillFingerprint::from).collect(),
            #[cfg(not(feature = "mesh"))]
            fills: Vec::new(),
        }
    }
}

impl TryFrom<GeometrySerde> for Geometry {
    type Error = String;
    fn try_from(s: GeometrySerde) -> Result<Self, Self::Error> {
        if !s.fills.is_empty() {
            return Err(
                "geometry with mesh-filled cells cannot be deserialized from JSON: \
                 mesh fills serialize only a lossy fingerprint summary, not a \
                 reconstructable form"
                    .to_string(),
            );
        }
        Geometry::new(s.cells, s.materials)
    }
}

/// True for a cell synthesized from a mesh-fill volume (issue #232);
/// such cells are excluded from the point-location BVH because point
/// location finds their host cell spatially and resolves the fill.
fn is_embedded_fill_cell(cell: &Cell) -> bool {
    #[cfg(feature = "mesh")]
    {
        matches!(
            cell.fill_role,
            Some(crate::geometry::fill::FillRole::Embedded { .. })
        )
    }
    #[cfg(not(feature = "mesh"))]
    {
        let _ = cell;
        false
    }
}

impl Geometry {
    /// Create a new geometry with validation and auto-assignment of IDs.
    ///
    /// Cells and materials without an ID are automatically assigned one.
    /// Explicitly set IDs are preserved; duplicates are still an error.
    /// Every `cell.material_idx` must be either `None` or a valid index into
    /// `materials`.
    pub fn new(cells: Vec<Cell>, materials: Vec<Arc<Material>>) -> Result<Self, String> {
        // Fail fast on cells recycled from a mesh-filled geometry: their
        // fill roles reference fills this geometry does not have, so the
        // mesh bodies would silently vanish (embedded cells share the
        // host's region and are shadowed by it) or the queries would
        // index a missing fill.
        #[cfg(feature = "mesh")]
        if cells.iter().any(|c| c.in_mesh_fill()) {
            return Err(
                "cells taken from a mesh-filled geometry cannot seed a new geometry: \
                 the mesh fill is not carried by the cells. Rebuild the filled cell \
                 with Geometry::new_with_fills (Python: Cell(fill=...))"
                    .to_string(),
            );
        }
        let n_user_materials = materials.len();
        let (cells, materials, bvh, vacuum_surfaces) =
            Self::validate_and_index(cells, materials, n_user_materials)?;
        Ok(Geometry {
            cells,
            materials,
            #[cfg(feature = "mesh")]
            fills: Vec::new(),
            bvh,
            vacuum_surfaces,
        })
    }

    /// Create a geometry in which some cells are filled by mesh bodies
    /// (issue #232). Each fill's mesh volumes are appended to the cell
    /// list as ordinary cells (IDs auto-assigned after the user's cells,
    /// in fill order then volume order) and its materials join the flat
    /// material store; the host cell's own material becomes the
    /// complement inside its region. Runs the same validation as
    /// [`Geometry::new`] plus the fill checks (watertightness, protrusion
    /// unless `allow_clipping`).
    #[cfg(feature = "mesh")]
    pub fn new_with_fills(
        mut cells: Vec<Cell>,
        mut materials: Vec<Arc<Material>>,
        fills: Vec<crate::geometry::fill::CellFillSpec>,
    ) -> Result<Self, String> {
        let n_user_materials = materials.len();
        let fills = crate::geometry::fill::expand_fills(&mut cells, &mut materials, fills)?;
        let (cells, materials, bvh, vacuum_surfaces) =
            Self::validate_and_index(cells, materials, n_user_materials)?;
        Ok(Geometry {
            cells,
            materials,
            fills,
            bvh,
            vacuum_surfaces,
        })
    }

    /// Shared validation + index construction behind [`Geometry::new`]
    /// and [`Geometry::new_with_fills`]. Materials at index
    /// `n_user_materials` and beyond were appended by a mesh fill:
    /// per-volume clones of one material legitimately repeat an ID there
    /// (fill expansion has already vetted genuine collisions), so the
    /// duplicate-ID error applies only to the user's own materials.
    #[allow(clippy::type_complexity)]
    fn validate_and_index(
        mut cells: Vec<Cell>,
        mut materials: Vec<Arc<Material>>,
        n_user_materials: usize,
    ) -> Result<
        (
            Vec<Cell>,
            Vec<Arc<Material>>,
            Bvh,
            Vec<Arc<crate::geo::Surface>>,
        ),
        String,
    > {
        // --- Validate material_idx bounds ---------------------------------------
        for cell in &cells {
            if let Some(idx) = cell.material_idx {
                if (idx as usize) >= materials.len() {
                    return Err(format!(
                        "Cell material_idx {idx} out of bounds (materials.len() = {})",
                        materials.len()
                    ));
                }
            }
        }

        // --- Auto-assign cell IDs ------------------------------------------------
        let mut used_cell_ids: HashSet<u32> = HashSet::new();
        for cell in &cells {
            if let Some(id) = cell.cell_id {
                if !used_cell_ids.insert(id) {
                    return Err(format!(
                        "Duplicate cell_id {id} found. All cell IDs must be unique."
                    ));
                }
            }
        }
        let mut next_cell_id: u32 = used_cell_ids.iter().copied().max().map_or(1, |m| m + 1);
        for cell in &mut cells {
            if cell.cell_id.is_none() {
                while used_cell_ids.contains(&next_cell_id) {
                    next_cell_id += 1;
                }
                cell.cell_id = Some(next_cell_id);
                used_cell_ids.insert(next_cell_id);
                next_cell_id += 1;
            }
        }

        // --- Auto-assign material IDs -------------------------------------------
        let mut used_material_ids: HashSet<u32> = HashSet::new();
        for (idx, material_arc) in materials.iter().enumerate() {
            if let Some(id) = material_arc.get_material_id() {
                if !used_material_ids.insert(id) && idx < n_user_materials {
                    return Err(format!(
                        "Duplicate material_id {id} found. All material IDs must be unique."
                    ));
                }
            }
        }
        let mut next_material_id: u32 =
            used_material_ids.iter().copied().max().map_or(1, |m| m + 1);
        for material_arc in &mut materials {
            if material_arc.get_material_id().is_none() {
                while used_material_ids.contains(&next_material_id) {
                    next_material_id += 1;
                }
                Arc::make_mut(material_arc).set_material_id(next_material_id);
                used_material_ids.insert(next_material_id);
                next_material_id += 1;
            }
        }

        // --- Validate surface IDs ------------------------------------------------
        let mut unique_surface_ptrs = HashSet::new();
        let mut unique_surfaces = Vec::new();

        for cell in &cells {
            let surfaces = cell.region.surfaces_with_sense();
            for (surface, _sense) in surfaces {
                let ptr = Arc::as_ptr(&surface);
                if unique_surface_ptrs.insert(ptr) {
                    unique_surfaces.push(surface);
                }
            }
        }

        let mut used_surface_ids = HashSet::new();
        for surface in &unique_surfaces {
            if let Some(id) = surface.surface_id {
                if !used_surface_ids.insert(id) {
                    return Err(format!(
                        "Duplicate surface_id {id} found. All surface IDs must be unique."
                    ));
                }
            }
        }

        // --- Build BVH over cell bounding boxes ---------------------------------
        let items: Vec<(u32, yamc_geo::BoundingBox)> = cells
            .iter()
            .enumerate()
            .filter(|(_, c)| !is_embedded_fill_cell(c))
            .map(|(i, c)| (i as u32, c.region.bounding_box()))
            .collect();
        let bvh = Bvh::build(&items);

        // Vacuum surfaces, from the deduplicated set above: the Woodcock
        // flight exit check (issue #360) tests each flight only against
        // these few surfaces.
        let vacuum_surfaces: Vec<Arc<crate::geo::Surface>> = unique_surfaces
            .iter()
            .filter(|s| s.boundary == crate::geo::BoundaryType::Vacuum)
            .cloned()
            .collect();

        Ok((cells, materials, bvh, vacuum_surfaces))
    }

    /// True when any cell in this geometry is filled by a mesh body.
    /// Used by the backends that cannot see mesh fills (GPU kernel,
    /// Woodcock/hybrid delta tracking) to reject the model up front.
    pub fn has_mesh_fills(&self) -> bool {
        #[cfg(feature = "mesh")]
        {
            !self.fills.is_empty()
        }
        #[cfg(not(feature = "mesh"))]
        {
            false
        }
    }

    /// Refine a spatially-located cell through its mesh fill: when
    /// `cell_index` is a fill host, a point inside one of the fill's mesh
    /// volumes maps to that volume's embedded cell; a point in the gap
    /// stays with the host (the complement). No-op for ordinary cells.
    #[cfg(feature = "mesh")]
    #[inline]
    pub(crate) fn resolve_fill_at(&self, cell_index: usize, point: (f64, f64, f64)) -> usize {
        if self.fills.is_empty() {
            return cell_index;
        }
        if let Some(crate::geometry::fill::FillRole::Host { fill }) =
            self.cells[cell_index].fill_role
        {
            let f = &self.fills[fill as usize];
            let vol = f.find_volume_world([point.0, point.1, point.2]);
            if vol != f.mesh.topology.implicit_complement {
                return f.first_volume_cell + vol as usize;
            }
        }
        cell_index
    }

    /// Find the closest boundary crossing from a position along a
    /// direction. For an ordinary cell this is the nearest intersection
    /// with the cell's own surfaces. For a cell involved in a mesh fill
    /// it is the nearer of that CSG exit and the next mesh-surface
    /// crossing (min-of-two-boundary-systems): crossing a mesh surface
    /// switches to the neighbouring mesh volume (or back to the host's
    /// complement) within the same CSG region, while the CSG surface
    /// always exits to the neighbour cell, clipping any protruding mesh.
    /// Mesh boundary-condition tags (e.g. vacuum) apply only when the
    /// mesh IS the geometry; inside a fill every mesh face transmits.
    pub fn closest_boundary(
        &self,
        cell_index: usize,
        position: [f64; 3],
        direction: [f64; 3],
    ) -> Option<crate::geometry::backend::BoundaryHit> {
        use crate::geometry::backend::BoundaryHit;
        let cell = &self.cells[cell_index];
        let csg_hit = cell
            .closest_surface(position, direction)
            .map(|(boundary, dist, surf_id)| BoundaryHit {
                distance: dist,
                boundary,
                next_cell_index: None,
                surface_id: surf_id,
            });
        #[cfg(feature = "mesh")]
        if let Some(role) = cell.fill_role {
            use crate::geometry::fill::FillRole;
            let (fill_idx, volume) = match role {
                FillRole::Host { fill } => (fill, None),
                FillRole::Embedded { fill, volume } => (fill, Some(volume)),
            };
            let f = &self.fills[fill_idx as usize];
            let vol_id = volume.unwrap_or(f.mesh.topology.implicit_complement);
            if let Some((dist, surface_id)) = f.ray_fire_world(vol_id, position, direction) {
                let csg_dist = csg_hit.as_ref().map_or(f64::INFINITY, |h| h.distance);
                // The mesh crossing must beat the CSG exit by more than
                // the transport crossing nudge (SURFACE_TOLERANCE, 1e-8).
                // For a mesh face coincident with the region boundary
                // (a body touching its container) the two distances
                // differ only by float noise; letting the mesh win there
                // would step the particle past BOTH surfaces while the
                // topological next_cell_index still says host/embedded,
                // stranding it outside the region (and turning a vacuum
                // boundary into a silent kill). Within the margin the
                // CSG surface wins and clips, which also handles
                // protruding meshes.
                const CROSSING_MARGIN: f64 = 1e-8;
                if dist + CROSSING_MARGIN < csg_dist {
                    // next_volume == None leaves next_cell_index unset,
                    // falling back to spatial re-location (never a loss).
                    let next_cell_index = f
                        .mesh
                        .next_volume(surface_id, vol_id)
                        .map(|nv| f.cell_index_for_volume(nv));
                    return Some(BoundaryHit {
                        distance: dist,
                        boundary: crate::geo::BoundaryType::Transmission,
                        next_cell_index,
                        surface_id: None,
                    });
                }
            }
        }
        csg_hit
    }

    /// Spatial tracking verification (issue #254) for mesh fills: before
    /// accepting a crossing from a cell involved in a fill, check that no
    /// mesh surface foreign to the current volume lies strictly inside
    /// the accepted segment. Always false for ordinary CSG cells, whose
    /// tracking re-locates spatially anyway.
    #[cfg_attr(not(feature = "mesh"), allow(unused_variables))]
    pub(crate) fn crossing_blocked(
        &self,
        cell_index: usize,
        origin: [f64; 3],
        direction: [f64; 3],
        t_max: f64,
    ) -> bool {
        #[cfg(feature = "mesh")]
        {
            if self.fills.is_empty() {
                return false;
            }
            use crate::geometry::fill::FillRole;
            let (fill_idx, volume) = match self.cells[cell_index].fill_role {
                Some(FillRole::Host { fill }) => (fill, None),
                Some(FillRole::Embedded { fill, volume }) => (fill, Some(volume)),
                None => return false,
            };
            let f = &self.fills[fill_idx as usize];
            let vol_id = volume.unwrap_or(f.mesh.topology.implicit_complement);
            f.segment_blocked_world(vol_id, origin, direction, t_max)
        }
        #[cfg(not(feature = "mesh"))]
        {
            false
        }
    }

    /// The geometry's vacuum-tagged surfaces (deduplicated), for the
    /// Woodcock flight exit check.
    pub(crate) fn vacuum_surfaces(&self) -> &[Arc<crate::geo::Surface>] {
        &self.vacuum_surfaces
    }

    /// Return the material for a cell, or `None` for void cells.
    #[inline]
    pub fn material_for(&self, cell: &Cell) -> Option<&Arc<Material>> {
        cell.material_idx
            .and_then(|i| self.materials.get(i as usize))
    }

    /// Find the first cell containing the given point, or None if not found.
    /// Uses the BVH for O(log N) descent; falls back to unbounded cells.
    pub fn find_cell(&self, point: (f64, f64, f64)) -> Option<&Cell> {
        self.find_cell_index(point).map(|i| &self.cells[i])
    }

    /// Find the index of the first cell containing the given point, or None
    /// if not found. Uses the BVH built at `Geometry::new` for O(log N)
    /// descent; cells with non-finite bounding boxes are tested afterwards.
    pub fn find_cell_index(&self, point: (f64, f64, f64)) -> Option<usize> {
        let p = [point.0, point.1, point.2];
        let found = self
            .bvh
            .find(p, |idx| self.cells[idx as usize].contains(point))
            .map(|i| i as usize);
        #[cfg(feature = "mesh")]
        {
            found.map(|i| self.resolve_fill_at(i, point))
        }
        #[cfg(not(feature = "mesh"))]
        {
            found
        }
    }
    /// Compute the bounding box of the entire geometry (enclosing all cells' regions)
    pub fn bounding_box(&self) -> crate::geo::BoundingBox {
        let mut bbox_opt: Option<crate::geo::BoundingBox> = None;
        for cell in &self.cells {
            let cell_bbox = cell.region.bounding_box();
            match &mut bbox_opt {
                None => bbox_opt = Some(cell_bbox),
                Some(b) => b.expand_to_include(&cell_bbox),
            }
        }
        bbox_opt.unwrap_or_else(|| {
            crate::geo::BoundingBox::new([f64::INFINITY; 3], [f64::NEG_INFINITY; 3])
        })
    }

    /// Generate 2D maps of cell IDs and material IDs for a slice through the geometry.
    ///
    /// # Arguments
    /// * `origin` - Center point of the plot (x, y, z)
    /// * `width` - Width of the plot in each basis direction (width_h, width_v)
    /// * `pixels` - Number of pixels in each direction (pixels_h, pixels_v)
    /// * `basis` - The plane to slice: "xy", "xz", or "yz"
    ///
    /// # Returns
    /// A tuple of (cell_ids, material_ids) where each is a Vec<Vec<i32>>.
    /// - The outer Vec corresponds to the vertical axis (y for "xy", z for "xz"/"yz")
    /// - The inner Vec corresponds to the horizontal axis (x for "xy"/"xz", y for "yz")
    /// - A value of -1 indicates no cell was found at that location
    /// - Material ID is -1 for void cells (cells without materials)
    pub fn sample_slice(
        &self,
        origin: (f64, f64, f64),
        width: (f64, f64),
        pixels: (usize, usize),
        basis: &str,
    ) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
        let (pixels_h, pixels_v) = pixels;
        let (width_h, width_v) = width;

        // Calculate lower-left corner from origin and width
        let half_width_h = width_h / 2.0;
        let half_width_v = width_v / 2.0;

        // Create coordinate arrays centered on origin
        let h_vals: Vec<f64> = (0..pixels_h)
            .map(|i| {
                if pixels_h > 1 {
                    -half_width_h + width_h * (i as f64) / ((pixels_h - 1) as f64)
                } else {
                    0.0
                }
            })
            .collect();

        let v_vals: Vec<f64> = (0..pixels_v)
            .map(|i| {
                if pixels_v > 1 {
                    -half_width_v + width_v * (i as f64) / ((pixels_v - 1) as f64)
                } else {
                    0.0
                }
            })
            .collect();

        // Basis code: 0=xy, 1=xz, 2=yz; anything else yields all-(-1) grids,
        // matching the previous behavior of `_ => continue`.
        let basis_code: i32 = match basis {
            "xy" => 0,
            "xz" => 1,
            "yz" => 2,
            _ => {
                return (
                    vec![vec![-1i32; pixels_h]; pixels_v],
                    vec![vec![-1i32; pixels_h]; pixels_v],
                );
            }
        };

        // Fill the grid row-by-row in parallel. Each row is independent
        // (pure reads on &self, disjoint output Vecs).
        let (cell_ids, material_ids): (Vec<Vec<i32>>, Vec<Vec<i32>>) = v_vals
            .par_iter()
            .map(|&v| {
                let mut row_cells = vec![-1i32; pixels_h];
                let mut row_materials = vec![-1i32; pixels_h];
                for (j, &h) in h_vals.iter().enumerate() {
                    let point = match basis_code {
                        0 => (origin.0 + h, origin.1 + v, origin.2),
                        1 => (origin.0 + h, origin.1, origin.2 + v),
                        _ => (origin.0, origin.1 + h, origin.2 + v),
                    };
                    if let Some(cell) = self.find_cell(point) {
                        row_cells[j] = cell.cell_id.map(|id| id as i32).unwrap_or(-1);
                        row_materials[j] = cell
                            .material_idx
                            .and_then(|mi| self.materials.get(mi as usize))
                            .and_then(|m| m.get_material_id())
                            .map(|id| id as i32)
                            .unwrap_or(-1);
                    }
                }
                (row_cells, row_materials)
            })
            .unzip();

        (cell_ids, material_ids)
    }
}
