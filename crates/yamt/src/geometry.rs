//! Public API: `MeshGeometry` -- the main entry point for mesh-based
//! geometric queries. Analogous to DAGMC's `DagMC` class.

use crate::accel::bvh::Bvh;
use crate::mesh::topology::MeshTopology;
use crate::query::{
    closest, element_walk, intersect, point_in_volume, ray_fire, ray_history::RayHistory,
};
use crate::types::*;

/// Error type for mesh geometry operations.
#[derive(Debug)]
pub enum MeshError {
    Io(std::io::Error),
    Parse(String),
    /// A tetrahedron in the input mesh is negatively oriented, which the
    /// element walk cannot work with. Raised by
    /// [`crate::mesh::topology::validate_tet_orientation`]; see there for why
    /// the mesh is rejected instead of repaired.
    NegativeTetOrientation {
        /// Index into the mesh's tetrahedron array.
        tet_index: usize,
        /// Its signed volume (negative, in cm³).
        signed_volume: f64,
    },
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::Io(e) => write!(f, "I/O error: {e}"),
            MeshError::Parse(msg) => write!(f, "Parse error: {msg}"),
            MeshError::NegativeTetOrientation {
                tet_index,
                signed_volume,
            } => write!(
                f,
                "tetrahedron {tet_index} is negatively oriented (signed volume \
                 {signed_volume:.6e} cm^3). Tet face normals are read off a fixed vertex \
                 ordering that points outward only for positively oriented tets, so a \
                 negative tet makes the element walk leave through an entry face and \
                 unstructured track-length tallies read about 33 percent low (issue #316). \
                 Fix the mesh at its source: regenerate it with a writer that emits \
                 positively oriented tets (yamc's own mesher, yamm, guarantees this at its \
                 output boundary), or have that writer swap the last two vertices of every \
                 tet whose signed volume is negative."
            ),
        }
    }
}

impl std::error::Error for MeshError {}

/// Precomputed triangle data for cache-friendly traversal.
///
/// Stores v0, edge1 (v1-v0), edge2 (v2-v0) for each primitive,
/// eliminating the multi-level vertex indirection chain and
/// redundant edge computation from the inner loop.
#[derive(Clone)]
pub(crate) struct PrecomputedTriData {
    pub(crate) v0: Vec<[f64; 3]>,
    pub(crate) edge1: Vec<[f64; 3]>,
    pub(crate) edge2: Vec<[f64; 3]>,
}

/// Per-volume surface BVH with associated data.
#[derive(Clone)]
pub(crate) struct SurfaceBvhData {
    pub(crate) bvh: Bvh,
    pub(crate) tri_ids: Vec<TriangleId>,
    pub(crate) surf_ids: Vec<SurfaceId>,
    pub(crate) precomputed: PrecomputedTriData,
}

/// Complete mesh geometry with acceleration structures, ready for queries.
///
/// This is the primary public API of the crate. It mirrors the DAGMC
/// `DagMC` class for mesh-based particle transport.
#[derive(Clone)]
pub struct MeshGeometry {
    /// The underlying mesh topology.
    pub topology: MeshTopology,

    /// Per-volume surface BVH + mapping arrays.
    surface_bvhs: Vec<SurfaceBvhData>,

    /// Per-volume element BVH + mapping arrays.
    /// `element_bvhs[vol_id] = (bvh, tet_ids)`
    element_bvhs: Vec<(Bvh, Vec<TetrahedronId>)>,
}

impl MeshGeometry {
    /// Load from an Arrow IPC mesh file and build all acceleration structures.
    #[cfg(feature = "arrow")]
    pub fn from_arrow(path: &std::path::Path) -> Result<Self, MeshError> {
        let data = crate::io::arrow::read_arrow_mesh(path)?;
        let topo = crate::io::arrow::build_topology(data)?;
        Ok(Self::from_topology(topo))
    }

    /// Build from an existing topology (builds BVH trees and computes volumes).
    pub fn from_topology(mut topo: MeshTopology) -> Self {
        #[cfg(feature = "simd")]
        {
            static PRINT_ONCE: std::sync::Once = std::sync::Once::new();
            PRINT_ONCE.call_once(crate::accel::simd::print_simd_info);
        }

        let surface_bvhs = build_surface_bvhs(&topo);
        let element_bvhs = build_element_bvhs(&topo);

        // Precompute analytical volumes if not already set (e.g. from Arrow metadata)
        if topo.volume_measures.is_empty() && topo.num_volumes > 0 {
            // We need the MeshGeometry built first to call measure_volume,
            // but measure_volume only uses topology data, so we can compute
            // directly here using the same divergence-theorem logic.
            let mut measures = Vec::with_capacity(topo.num_volumes as usize);
            for vol_id in 0..topo.num_volumes {
                let mut total = 0.0;
                for &(surf_id, sense) in &topo.volume_surfaces[vol_id as usize] {
                    let range = &topo.surface_tri_ranges[surf_id as usize];
                    for i in range.start..range.end {
                        let tri_id = topo.surface_tri_indices[i as usize];
                        let tri = &topo.triangles[tri_id as usize];
                        let v0 = topo.vertices[tri[0] as usize];
                        let v1 = topo.vertices[tri[1] as usize];
                        let v2 = topo.vertices[tri[2] as usize];
                        let contrib =
                            crate::query::intersect::triangle_volume_contribution(v0, v1, v2);
                        match sense {
                            Sense::Forward => total += contrib,
                            Sense::Reverse => total -= contrib,
                        }
                    }
                }
                measures.push((total / 6.0).abs());
            }
            topo.volume_measures = measures;
        }

        Self {
            topology: topo,
            surface_bvhs,
            element_bvhs,
        }
    }

    /// Fire a ray within a volume. Returns `(distance, surface_id)` of
    /// the next surface crossing, or `None`.
    pub fn ray_fire(
        &self,
        volume: VolumeId,
        origin: [f64; 3],
        direction: [f64; 3],
        history: Option<&RayHistory>,
    ) -> Option<(f64, SurfaceId)> {
        if (volume as usize) >= self.surface_bvhs.len() {
            return None;
        }
        let sd = &self.surface_bvhs[volume as usize];
        ray_fire::ray_fire(
            &sd.bvh,
            &sd.tri_ids,
            &sd.surf_ids,
            &sd.precomputed.v0,
            &sd.precomputed.edge1,
            &sd.precomputed.edge2,
            origin,
            direction,
            history,
        )
        .map(|r| (r.distance, r.surface_id))
    }

    /// Fire a ray and also return the hit triangle ID (for ray history).
    pub fn ray_fire_detailed(
        &self,
        volume: VolumeId,
        origin: [f64; 3],
        direction: [f64; 3],
        history: Option<&RayHistory>,
    ) -> Option<ray_fire::RayFireResult> {
        if (volume as usize) >= self.surface_bvhs.len() {
            return None;
        }
        let sd = &self.surface_bvhs[volume as usize];
        ray_fire::ray_fire(
            &sd.bvh,
            &sd.tri_ids,
            &sd.surf_ids,
            &sd.precomputed.v0,
            &sd.precomputed.edge1,
            &sd.precomputed.edge2,
            origin,
            direction,
            history,
        )
    }

    /// Nearest surface NOT bounding `volume` that intersects the open
    /// ray segment `(eps, t_max - eps)`, or `None`.
    ///
    /// Spatial cross-check for adjacency tracking: a chord of a volume
    /// (from a point inside it to its nearest own boundary) cannot pass
    /// through any foreign surface when the geometry is valid, so a hit
    /// is a geometry error (overlapping or self-intersecting volumes).
    /// Uses the same early-exit nearest-hit queries as tracking; shared
    /// surfaces (which also bound `volume`) are not flagged.
    pub fn segment_blocked(
        &self,
        volume: VolumeId,
        origin: [f64; 3],
        direction: [f64; 3],
        t_max: f64,
    ) -> Option<(f64, SurfaceId)> {
        let eps = 1e-6_f64.max(t_max * 1e-9);
        let t_hi = t_max - eps;
        for w in 0..self.topology.num_volumes {
            if w == volume {
                continue;
            }
            let bb = &self.topology.volume_aabbs[w as usize];
            if !intersect::segment_hits_aabb(origin, direction, t_max, bb) {
                continue;
            }
            let sd = &self.surface_bvhs[w as usize];
            let mut blocked: Option<(f64, SurfaceId)> = None;
            sd.bvh.ray_traverse_upto(origin, direction, t_hi, |prim| {
                let idx = prim as usize;
                let v0 = sd.precomputed.v0[idx];
                let v1 = intersect::add(v0, sd.precomputed.edge1[idx]);
                let v2 = intersect::add(v0, sd.precomputed.edge2[idx]);
                let t = intersect::ray_triangle_intersect(origin, direction, v0, v1, v2)?;
                if t <= eps || t >= t_hi {
                    return None;
                }
                let sid = sd.surf_ids[idx];
                // A surface bounding the current volume is not foreign.
                // surface_volumes stores the implicit-complement side as
                // None, so when `volume` IS the implicit complement a
                // None side counts as bounding it (grazing re-hits of a
                // just-crossed curved surface land here).
                let ic = self.topology.implicit_complement;
                let (fwd, rev) = self.topology.surface_volumes[sid as usize];
                let bounds_current = |side: Option<VolumeId>| match side {
                    Some(v) => v == volume,
                    None => volume == ic,
                };
                if bounds_current(fwd) || bounds_current(rev) {
                    return None;
                }
                blocked = Some((t, sid));
                // Any-hit: collapse the remaining search immediately.
                Some(f64::MIN_POSITIVE)
            });
            if blocked.is_some() {
                return blocked;
            }
        }
        None
    }

    /// Test if a point is inside a volume.
    pub fn point_in_volume(&self, volume: VolumeId, point: [f64; 3]) -> bool {
        if (volume as usize) >= self.surface_bvhs.len() {
            return false;
        }
        let sd = &self.surface_bvhs[volume as usize];
        point_in_volume::point_in_volume(
            &sd.bvh,
            &sd.precomputed.v0,
            &sd.precomputed.edge1,
            &sd.precomputed.edge2,
            point,
        )
    }

    /// Find which volume contains a point.
    pub fn find_volume(&self, point: [f64; 3]) -> VolumeId {
        point_in_volume::find_volume(&self.topology, &self.surface_bvhs, point)
    }

    /// Get the volume on the other side of a surface.
    pub fn next_volume(&self, surface: SurfaceId, current: VolumeId) -> Option<VolumeId> {
        if (surface as usize) >= self.topology.surface_volumes.len() {
            return None;
        }
        let (fwd, rev) = self.topology.surface_volumes[surface as usize];
        if fwd == Some(current) {
            rev
        } else if rev == Some(current) {
            fwd
        } else {
            None
        }
    }

    /// Closest distance from a point to any surface of a volume.
    pub fn closest_to_surface(&self, volume: VolumeId, point: [f64; 3]) -> f64 {
        if (volume as usize) >= self.surface_bvhs.len() {
            return f64::MAX;
        }
        let sd = &self.surface_bvhs[volume as usize];
        let result = sd.bvh.closest_point_traverse(point, |bvh_prim_idx| {
            let idx = bvh_prim_idx as usize;
            let v0 = sd.precomputed.v0[idx];
            let v1 = intersect::add(v0, sd.precomputed.edge1[idx]);
            let v2 = intersect::add(v0, sd.precomputed.edge2[idx]);
            let (_, dist) = closest::closest_point_on_triangle(point, v0, v1, v2);
            dist * dist
        });
        match result {
            Some((_, dist_sq)) => dist_sq.sqrt(),
            None => f64::MAX,
        }
    }

    /// Surface normal at a point (from the closest triangle on the surface).
    ///
    /// Note: This uses a per-surface BVH if the surface is part of a volume.
    /// For surfaces not associated with volumes, it falls back to linear search.
    pub fn surface_normal(&self, surface: SurfaceId, point: [f64; 3]) -> [f64; 3] {
        if (surface as usize) >= self.topology.surface_tri_ranges.len() {
            return [0.0, 0.0, 1.0];
        }

        // Try to find a volume that has this surface to use its BVH
        for sd in &self.surface_bvhs {
            let result = sd.bvh.closest_point_traverse(point, |bvh_prim_idx| {
                let idx = bvh_prim_idx as usize;
                if sd.surf_ids[idx] != surface {
                    return f64::MAX;
                }
                let v0 = sd.precomputed.v0[idx];
                let v1 = intersect::add(v0, sd.precomputed.edge1[idx]);
                let v2 = intersect::add(v0, sd.precomputed.edge2[idx]);
                let (_, dist) = closest::closest_point_on_triangle(point, v0, v1, v2);
                dist * dist
            });
            if let Some((prim_idx, _)) = result {
                let idx = prim_idx as usize;
                if sd.surf_ids[idx] == surface {
                    let e1 = sd.precomputed.edge1[idx];
                    let e2 = sd.precomputed.edge2[idx];
                    return intersect::normalize(intersect::cross(e1, e2));
                }
            }
        }

        // Fallback: linear scan
        let range = &self.topology.surface_tri_ranges[surface as usize];
        let mut best_dist = f64::MAX;
        let mut best_normal = [0.0, 0.0, 1.0];

        for i in range.start..range.end {
            let tri_id = self.topology.surface_tri_indices[i as usize];
            let tri = &self.topology.triangles[tri_id as usize];
            let v0 = self.topology.vertices[tri[0] as usize];
            let v1 = self.topology.vertices[tri[1] as usize];
            let v2 = self.topology.vertices[tri[2] as usize];
            let (_, dist) = closest::closest_point_on_triangle(point, v0, v1, v2);
            if dist < best_dist {
                best_dist = dist;
                best_normal = intersect::triangle_normal(v0, v1, v2);
            }
        }

        best_normal
    }

    /// Boundary condition for a surface.
    pub fn boundary_condition(&self, surface: SurfaceId) -> BoundaryCondition {
        self.topology
            .physical_data
            .surface_bcs
            .get(&surface)
            .copied()
            .unwrap_or(BoundaryCondition::Transmission)
    }

    /// Material name for a volume.
    pub fn material_name(&self, volume: VolumeId) -> Option<&str> {
        self.topology
            .physical_data
            .volume_materials
            .get(&volume)
            .and_then(|m| m.material_name.as_deref())
    }

    // -- Element walking (tet mesh overlay) ---------------------------------

    /// Find which tetrahedron contains a point, within a volume.
    pub fn find_element(&self, volume: VolumeId, point: [f64; 3]) -> Option<TetrahedronId> {
        if (volume as usize) >= self.element_bvhs.len() {
            return None;
        }
        let (ref bvh, ref tet_ids) = self.element_bvhs[volume as usize];
        element_walk::find_element(&self.topology, bvh, tet_ids, point)
    }

    /// Walk a ray through the tet mesh, returning (tet_id, path_length) segments.
    pub fn walk_elements(
        &self,
        start_element: TetrahedronId,
        origin: [f64; 3],
        direction: [f64; 3],
        max_distance: f64,
    ) -> Vec<(TetrahedronId, f64)> {
        element_walk::walk_elements(
            &self.topology,
            start_element,
            origin,
            direction,
            max_distance,
        )
    }

    /// Compute segments from start to end through the tet mesh of a volume.
    pub fn segments(
        &self,
        volume: VolumeId,
        start: [f64; 3],
        end: [f64; 3],
    ) -> Vec<(TetrahedronId, f64)> {
        if (volume as usize) >= self.element_bvhs.len() {
            return Vec::new();
        }
        let (ref bvh, ref tet_ids) = self.element_bvhs[volume as usize];
        element_walk::segments(&self.topology, bvh, tet_ids, volume, start, end)
    }

    // -- Measurements -------------------------------------------------------

    /// Compute the volume of a mesh volume using the divergence theorem.
    pub fn measure_volume(&self, volume: VolumeId) -> f64 {
        if (volume as usize) >= self.topology.volume_surfaces.len() {
            return 0.0;
        }
        let mut total = 0.0;
        for &(surf_id, sense) in &self.topology.volume_surfaces[volume as usize] {
            let range = &self.topology.surface_tri_ranges[surf_id as usize];
            for i in range.start..range.end {
                let tri_id = self.topology.surface_tri_indices[i as usize];
                let tri = &self.topology.triangles[tri_id as usize];
                let v0 = self.topology.vertices[tri[0] as usize];
                let v1 = self.topology.vertices[tri[1] as usize];
                let v2 = self.topology.vertices[tri[2] as usize];
                let contrib = intersect::triangle_volume_contribution(v0, v1, v2);
                match sense {
                    Sense::Forward => total += contrib,
                    Sense::Reverse => total -= contrib,
                }
            }
        }
        (total / 6.0).abs()
    }

    /// Compute the area of a mesh surface.
    pub fn measure_surface_area(&self, surface: SurfaceId) -> f64 {
        if (surface as usize) >= self.topology.surface_tri_ranges.len() {
            return 0.0;
        }
        let range = &self.topology.surface_tri_ranges[surface as usize];
        let mut total = 0.0;
        for i in range.start..range.end {
            let tri_id = self.topology.surface_tri_indices[i as usize];
            let tri = &self.topology.triangles[tri_id as usize];
            let v0 = self.topology.vertices[tri[0] as usize];
            let v1 = self.topology.vertices[tri[1] as usize];
            let v2 = self.topology.vertices[tri[2] as usize];
            total += intersect::triangle_area(v0, v1, v2);
        }
        total
    }

    /// Compute the volume of a tetrahedron.
    pub fn tet_volume(&self, tet: TetrahedronId) -> f64 {
        if (tet as usize) >= self.topology.tetrahedra.len() {
            return 0.0;
        }
        let verts = &self.topology.tetrahedra[tet as usize];
        let v0 = self.topology.vertices[verts[0] as usize];
        let v1 = self.topology.vertices[verts[1] as usize];
        let v2 = self.topology.vertices[verts[2] as usize];
        let v3 = self.topology.vertices[verts[3] as usize];
        intersect::tet_volume(v0, v1, v2, v3)
    }

    /// Number of (non-implicit-complement) volumes.
    pub fn num_volumes(&self) -> u32 {
        self.topology.num_volumes
    }

    /// Number of surfaces.
    pub fn num_surfaces(&self) -> u32 {
        self.topology.num_surfaces
    }

    /// The implicit complement volume ID.
    pub fn implicit_complement(&self) -> VolumeId {
        self.topology.implicit_complement
    }
}

// ---------------------------------------------------------------------------
// BVH building helpers
// ---------------------------------------------------------------------------

/// Build per-volume surface BVHs with precomputed triangle data.
///
/// For each volume, collect all triangles on all its surfaces, build
/// one BVH for ray firing, and precompute edge vectors for each triangle.
fn build_surface_bvhs(topo: &MeshTopology) -> Vec<SurfaceBvhData> {
    // Build BVH for each real volume plus the implicit complement.
    // The complement entry (at index num_volumes) was appended to
    // volume_surfaces by the topology builder, so indexing is correct.
    let iter = 0..topo.num_volumes + 1;
    let mapper = |vol_id: u32| {
        let mut tri_ids: Vec<TriangleId> = Vec::new();
        let mut surf_ids: Vec<SurfaceId> = Vec::new();
        let mut aabbs: Vec<[f64; 6]> = Vec::new();

        for &(surf_id, _sense) in &topo.volume_surfaces[vol_id as usize] {
            let range = &topo.surface_tri_ranges[surf_id as usize];
            for i in range.start..range.end {
                let tri_id = topo.surface_tri_indices[i as usize];
                tri_ids.push(tri_id);
                surf_ids.push(surf_id);
                aabbs.push(topo.triangle_aabbs[tri_id as usize]);
            }
        }

        // Precompute v0, edge1, edge2 for each triangle
        let n = tri_ids.len();
        let mut v0s = Vec::with_capacity(n);
        let mut edge1s = Vec::with_capacity(n);
        let mut edge2s = Vec::with_capacity(n);
        for &tri_id in &tri_ids {
            let tri = &topo.triangles[tri_id as usize];
            let va = topo.vertices[tri[0] as usize];
            let vb = topo.vertices[tri[1] as usize];
            let vc = topo.vertices[tri[2] as usize];
            v0s.push(va);
            edge1s.push(intersect::sub(vb, va));
            edge2s.push(intersect::sub(vc, va));
        }

        let bvh = Bvh::build(&aabbs);
        SurfaceBvhData {
            bvh,
            tri_ids,
            surf_ids,
            precomputed: PrecomputedTriData {
                v0: v0s,
                edge1: edge1s,
                edge2: edge2s,
            },
        }
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        iter.into_par_iter().map(mapper).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        iter.map(mapper).collect()
    }
}

/// Build per-volume element BVHs (for point-in-tet queries).
fn build_element_bvhs(topo: &MeshTopology) -> Vec<(Bvh, Vec<TetrahedronId>)> {
    let iter = 0..topo.num_volumes;
    let mapper = |vol_id: u32| {
        let range = &topo.volume_tet_ranges[vol_id as usize];
        let mut tet_ids: Vec<TetrahedronId> = Vec::new();
        let mut aabbs: Vec<[f64; 6]> = Vec::new();

        for i in range.start..range.end {
            let tet_id = topo.volume_tet_indices[i as usize];
            tet_ids.push(tet_id);
            aabbs.push(topo.tet_aabbs[tet_id as usize]);
        }

        let bvh = Bvh::build(&aabbs);
        (bvh, tet_ids)
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        iter.into_par_iter().map(mapper).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        iter.map(mapper).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ----------------------------- Tests -----------------------------

#[cfg(all(test, feature = "arrow"))]
mod tests {
    use super::*;

    fn cube_geometry() -> MeshGeometry {
        MeshGeometry::from_arrow(std::path::Path::new("tests/data/cube.arrow")).unwrap()
    }

    fn two_region_geometry() -> MeshGeometry {
        MeshGeometry::from_arrow(std::path::Path::new("tests/data/two_region_tets.arrow")).unwrap()
    }

    #[test]
    fn test_point_outside_with_two_crossings_ahead() {
        // The parity ray from a point BELOW the cube crosses two faces
        // (bottom and top): even count, outside. Routing the parity
        // enumeration through the nearest-hit-pruned ray_traverse dropped
        // the second hit and misclassified this as inside (issue #256),
        // which put every mesh-transport history whose source was not in
        // the first tested volume into the wrong birth cell.
        let geom = cube_geometry();
        assert!(!geom.point_in_volume(0, [0.5, 0.5, -1.0]));
        assert!(!geom.point_in_volume(0, [0.5, -5.0, 0.5]));
        assert_eq!(
            geom.find_volume([0.5, 0.5, -1.0]),
            geom.topology.implicit_complement
        );
    }

    #[test]
    fn test_cube_point_in_volume() {
        let geom = cube_geometry();
        // Center of cube should be in volume 0
        assert!(geom.point_in_volume(0, [0.5, 0.5, 0.5]));
        // Outside the cube
        assert!(!geom.point_in_volume(0, [2.0, 0.5, 0.5]));
    }

    #[test]
    fn test_cube_find_volume() {
        let geom = cube_geometry();
        assert_eq!(geom.find_volume([0.5, 0.5, 0.5]), 0);
        assert_eq!(
            geom.find_volume([2.0, 2.0, 2.0]),
            geom.implicit_complement()
        );
    }

    #[test]
    fn test_cube_ray_fire() {
        let geom = cube_geometry();
        // Ray from center in +x direction should hit a surface at distance ~0.5
        let result = geom.ray_fire(0, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], None);
        assert!(result.is_some());
        let (dist, _surf) = result.unwrap();
        assert!((dist - 0.5).abs() < 0.1, "Expected ~0.5, got {dist}");
    }

    #[test]
    fn test_cube_next_volume() {
        let geom = cube_geometry();
        // Fire ray and cross surface
        if let Some((_, surf)) = geom.ray_fire(0, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], None) {
            let next = geom.next_volume(surf, 0);
            assert!(next.is_some());
            // For the cube, the other side is the implicit complement
            assert_eq!(next.unwrap(), geom.implicit_complement());
        }
    }

    #[test]
    fn test_cube_boundary_condition() {
        let geom = cube_geometry();
        for surf_id in 0..geom.num_surfaces() {
            assert_eq!(geom.boundary_condition(surf_id), BoundaryCondition::Vacuum);
        }
    }

    #[test]
    fn test_cube_material() {
        let geom = cube_geometry();
        assert_eq!(geom.material_name(0), Some("water"));
    }

    #[test]
    fn test_cube_measure_volume() {
        let geom = cube_geometry();
        let vol = geom.measure_volume(0);
        assert!(
            (vol - 1.0).abs() < 0.1,
            "Unit cube volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn test_cube_volume_measures_precomputed() {
        let geom = cube_geometry();
        // volume_measures should be populated by from_topology()
        assert_eq!(geom.topology.volume_measures.len(), 1);
        let vol = geom.topology.volume_measures[0];
        assert!(
            (vol - 1.0).abs() < 0.1,
            "Precomputed volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn test_two_region_volume_measures() {
        let geom = two_region_geometry();
        assert_eq!(geom.topology.volume_measures.len(), 2);
        // Each half-cube is 0.5 x 1 x 1 = 0.5 cm³
        for &vol in &geom.topology.volume_measures {
            assert!(
                (vol - 0.5).abs() < 0.01,
                "Half-cube volume should be ~0.5, got {vol}"
            );
        }
    }

    #[test]
    fn test_cube_surface_normal() {
        let geom = cube_geometry();
        // Get normal at a point on the +x face (x=1.0)
        // This is approximate -- depends on which surface is the +x face
        let _normal = geom.surface_normal(0, [1.0, 0.5, 0.5]);
        // Just verify it returns a unit vector
        let n = geom.surface_normal(0, [0.5, 0.5, 0.0]);
        let len = intersect::length(n);
        assert!((len - 1.0).abs() < 1e-6 || len < 1e-10);
    }

    #[test]
    fn test_cube_closest_to_surface() {
        let geom = cube_geometry();
        let dist = geom.closest_to_surface(0, [0.5, 0.5, 0.5]);
        // Center of unit cube → closest surface is 0.5 away
        assert!(
            (dist - 0.5).abs() < 0.1,
            "Distance from center to nearest face should be ~0.5, got {dist}"
        );
    }

    #[test]
    fn test_two_region_materials() {
        let geom = two_region_geometry();
        assert_eq!(geom.num_volumes(), 2);
        assert_eq!(geom.material_name(0), Some("fuel"));
        assert_eq!(geom.material_name(1), Some("moderator"));
    }

    #[test]
    fn test_cube_find_element() {
        let geom = cube_geometry();
        let tet = geom.find_element(0, [0.5, 0.5, 0.5]);
        assert!(tet.is_some(), "Should find a tet at cube center");
    }

    #[test]
    fn test_cube_tet_volume() {
        let geom = cube_geometry();
        // Sum of all tet volumes should equal the cube volume (~1.0)
        let mut total = 0.0;
        for tet_id in 0..geom.topology.tetrahedra.len() as u32 {
            total += geom.tet_volume(tet_id);
        }
        assert!(
            (total - 1.0).abs() < 0.1,
            "Sum of tet volumes should be ~1.0, got {total}"
        );
    }
}
