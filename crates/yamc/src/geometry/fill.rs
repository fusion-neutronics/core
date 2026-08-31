//! Mesh fills for hybrid CSG + CAD-mesh geometry (issue #232).
//!
//! A CSG [`Cell`] can be *filled* by a surface-mesh body: inside the cell's
//! region a particle is either inside one of the mesh volumes (scoring that
//! volume's material) or in the gap between them (scoring the cell's own
//! material, which acts as the complement). The mesh volumes are flattened
//! into the parent [`Geometry`](crate::geometry::Geometry) as ordinary
//! cells, so cell filters, `material_for` and results address them exactly
//! like CSG cells.
//!
//! The tracking contract is min-of-two-boundary-systems: from any point in
//! a filled cell the next boundary is the nearer of the host cell's CSG
//! surface exit and the next mesh-surface crossing. A mesh that protrudes
//! past the host region is clipped by the CSG surface at runtime (the
//! particle exits to the neighbour cell, it is never lost), but because
//! silent clipping corrupts the geometry the constructor rejects
//! protruding fills unless `allow_clipping` is set.

use std::collections::HashMap;
use std::sync::Arc;

use crate::geometry::cell::Cell;
use crate::geometry::csg::MeshFillFingerprint;
use yamc_materials::material::Material;

/// Which part a cell plays in a mesh fill. Stored on [`Cell`] so the
/// hot-path geometry queries can branch without any lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillRole {
    /// The CSG cell that carries the fill. Its own material is the
    /// complement (the space inside the region but outside every mesh
    /// volume).
    Host { fill: u32 },
    /// A cell synthesized from one mesh volume of a fill.
    Embedded { fill: u32, volume: u32 },
}

/// User-facing specification of one mesh fill, consumed by
/// [`Geometry::new_with_fills`](crate::geometry::Geometry::new_with_fills).
pub struct CellFillSpec {
    /// Index into the `cells` argument of the host cell.
    pub host_cell_index: usize,
    /// The mesh body (volumes + materials) that fills the host cell.
    pub mesh_geometry: crate::geometry::mesh::MeshGeometry,
    /// Placement of the mesh body in the CSG frame, in cm.
    pub translation: [f64; 3],
    /// Rotation of the mesh body about the CSG-frame x, then y, then z
    /// axes, in degrees, applied before the translation.
    pub rotation_degrees: [f64; 3],
    /// Permit the mesh to protrude past the host region. The protruding
    /// part is clipped by the CSG surface (the effective body is
    /// `mesh intersected with region`); particles are never lost.
    pub allow_clipping: bool,
}

/// A mesh body embedded in a CSG cell, ready for transport queries.
#[derive(Clone)]
pub struct MeshFill {
    /// The underlying mesh (BVH-accelerated). Shared so cloning the
    /// geometry stays cheap.
    pub mesh: Arc<yamt::MeshGeometry>,
    /// Placement of the mesh body in the CSG frame, in cm.
    pub translation: [f64; 3],
    /// Rotation angles in degrees about the x, y, z axes (see
    /// [`CellFillSpec::rotation_degrees`]).
    pub rotation_degrees: [f64; 3],
    /// Precomputed world-from-mesh rotation matrix; `None` when the
    /// rotation is identity.
    rotation: Option<[[f64; 3]; 3]>,
    /// Index of the host cell in `Geometry.cells`.
    pub host_cell_index: usize,
    /// Index in `Geometry.cells` of the cell for mesh volume 0; the
    /// remaining volumes follow contiguously.
    pub first_volume_cell: usize,
}

// yamt::MeshGeometry has no Debug impl (BVH-backed); summarize instead.
impl std::fmt::Debug for MeshFill {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshFill")
            .field("num_volumes", &self.mesh.topology.num_volumes)
            .field("translation", &self.translation)
            .field("rotation_degrees", &self.rotation_degrees)
            .field("host_cell_index", &self.host_cell_index)
            .field("first_volume_cell", &self.first_volume_cell)
            .finish_non_exhaustive()
    }
}

impl MeshFill {
    /// A standalone fill view for point/ray queries against a placed
    /// mesh (used by the Python cell-level helpers before the geometry
    /// exists). The cell-index fields are meaningless here.
    pub fn for_queries(
        mesh: Arc<yamt::MeshGeometry>,
        translation: [f64; 3],
        rotation_degrees: [f64; 3],
    ) -> Self {
        MeshFill {
            mesh,
            translation,
            rotation_degrees,
            rotation: rotation_matrix(rotation_degrees),
            host_cell_index: 0,
            first_volume_cell: 0,
        }
    }

    /// The cell index for a mesh volume, mapping the implicit complement
    /// back to the host cell.
    #[inline]
    pub fn cell_index_for_volume(&self, volume: yamt::VolumeId) -> usize {
        if volume == self.mesh.topology.implicit_complement {
            self.host_cell_index
        } else {
            self.first_volume_cell + volume as usize
        }
    }

    /// Transform a CSG-frame point into the mesh frame.
    #[inline]
    pub fn world_to_mesh_point(&self, p: [f64; 3]) -> [f64; 3] {
        let q = [
            p[0] - self.translation[0],
            p[1] - self.translation[1],
            p[2] - self.translation[2],
        ];
        match &self.rotation {
            // Inverse of a rotation matrix is its transpose.
            Some(r) => [
                r[0][0] * q[0] + r[1][0] * q[1] + r[2][0] * q[2],
                r[0][1] * q[0] + r[1][1] * q[1] + r[2][1] * q[2],
                r[0][2] * q[0] + r[1][2] * q[1] + r[2][2] * q[2],
            ],
            None => q,
        }
    }

    /// Transform a CSG-frame direction into the mesh frame.
    #[inline]
    pub fn world_to_mesh_dir(&self, d: [f64; 3]) -> [f64; 3] {
        match &self.rotation {
            Some(r) => [
                r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2],
                r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2],
                r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2],
            ],
            None => d,
        }
    }

    /// Transform a mesh-frame point into the CSG frame.
    #[inline]
    pub fn mesh_to_world_point(&self, p: [f64; 3]) -> [f64; 3] {
        let q = match &self.rotation {
            Some(r) => [
                r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
                r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
                r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2],
            ],
            None => p,
        };
        [
            q[0] + self.translation[0],
            q[1] + self.translation[1],
            q[2] + self.translation[2],
        ]
    }

    /// The mesh volume containing a CSG-frame point (implicit complement
    /// when the point is in the gap).
    #[inline]
    pub fn find_volume_world(&self, p: [f64; 3]) -> yamt::VolumeId {
        self.mesh.find_volume(self.world_to_mesh_point(p))
    }

    /// Distance to the next mesh-surface crossing from a CSG-frame ray
    /// inside `volume`. Distances are frame-independent (the transform is
    /// rigid).
    #[inline]
    pub fn ray_fire_world(
        &self,
        volume: yamt::VolumeId,
        position: [f64; 3],
        direction: [f64; 3],
    ) -> Option<(f64, yamt::SurfaceId)> {
        self.mesh.ray_fire(
            volume,
            self.world_to_mesh_point(position),
            self.world_to_mesh_dir(direction),
            None,
        )
    }

    /// Spatial cross-check for adjacency tracking (issue #254): does a
    /// surface foreign to `volume` intersect the open CSG-frame segment?
    #[inline]
    pub fn segment_blocked_world(
        &self,
        volume: yamt::VolumeId,
        origin: [f64; 3],
        direction: [f64; 3],
        t_max: f64,
    ) -> bool {
        self.mesh
            .segment_blocked(
                volume,
                self.world_to_mesh_point(origin),
                self.world_to_mesh_dir(direction),
                t_max,
            )
            .is_some()
    }
}

impl From<&MeshFill> for MeshFillFingerprint {
    fn from(fill: &MeshFill) -> Self {
        let topo = &fill.mesh.topology;
        MeshFillFingerprint {
            host_cell_index: fill.host_cell_index,
            first_volume_cell: fill.first_volume_cell,
            translation: fill.translation,
            rotation_degrees: fill.rotation_degrees,
            num_volumes: topo.num_volumes,
            global_aabb: topo.global_aabb,
            volume_measures: topo.volume_measures.clone(),
            volume_materials: (0..topo.num_volumes)
                .map(|v| fill.mesh.material_name(v).map(|n| n.to_string()))
                .collect(),
        }
    }
}

/// Build the world-from-mesh rotation matrix R = Rz * Ry * Rx from angles
/// in degrees, or `None` for the identity.
fn rotation_matrix(degrees: [f64; 3]) -> Option<[[f64; 3]; 3]> {
    if degrees == [0.0, 0.0, 0.0] {
        return None;
    }
    let [rx, ry, rz] = degrees.map(f64::to_radians);
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    // R = Rz(rz) * Ry(ry) * Rx(rx), rotating the body about the fixed
    // CSG-frame x, then y, then z axes.
    Some([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])
}

/// True when every edge of the volume's bounding surfaces is shared by
/// exactly two triangles (a closed 2-manifold), i.e. the volume is
/// watertight. Vertices are welded by exact coordinate bits first so
/// meshes with duplicated-but-coincident vertices still pass.
fn volume_is_closed(topo: &yamt::MeshTopology, volume: u32) -> bool {
    // Weld vertices with bit-identical coordinates to one canonical id.
    let mut canonical: HashMap<[u64; 3], u32> = HashMap::new();
    let weld = |canonical: &mut HashMap<[u64; 3], u32>, v: u32| -> u32 {
        let p = topo.vertices[v as usize];
        let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
        *canonical.entry(key).or_insert(v)
    };

    let mut edge_counts: HashMap<(u32, u32), u32> = HashMap::new();
    for &(surf, _sense) in &topo.volume_surfaces[volume as usize] {
        for i in topo.surface_tri_ranges[surf as usize].clone() {
            let tri = topo.triangles[topo.surface_tri_indices[i as usize] as usize];
            let w = [
                weld(&mut canonical, tri[0]),
                weld(&mut canonical, tri[1]),
                weld(&mut canonical, tri[2]),
            ];
            for k in 0..3 {
                let (a, b) = (w[k], w[(k + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    !edge_counts.is_empty() && edge_counts.values().all(|&c| c == 2)
}

/// Expand fill specifications into embedded cells, appended materials and
/// transport-ready [`MeshFill`]s. Called by `Geometry::new_with_fills`
/// before the shared validation pass; `cells` and `materials` arrive with
/// only the user's entries and leave with the embedded volumes appended.
pub(crate) fn expand_fills(
    cells: &mut Vec<Cell>,
    materials: &mut Vec<Arc<Material>>,
    specs: Vec<CellFillSpec>,
) -> Result<Vec<MeshFill>, String> {
    let n_user_cells = cells.len();
    let mut fills: Vec<MeshFill> = Vec::with_capacity(specs.len());
    let mut filled_hosts: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for spec in specs {
        let CellFillSpec {
            host_cell_index,
            mesh_geometry,
            translation,
            rotation_degrees,
            allow_clipping,
        } = spec;
        // Partial moves: MeshGeometry is not Drop, so the pub fields can
        // be taken without cloning the BVH-backed mesh.
        let mesh = Arc::new(mesh_geometry.mesh);
        let mesh_cells = mesh_geometry.cells;
        let mesh_materials = mesh_geometry.materials;

        let host_label = |cells: &[Cell]| -> String {
            cells[host_cell_index]
                .name
                .clone()
                .unwrap_or_else(|| format!("cell index {host_cell_index}"))
        };

        // Bound the host to the caller's own cells: cells past
        // n_user_cells were appended by an earlier fill in this loop.
        if host_cell_index >= n_user_cells {
            return Err(format!(
                "fill host_cell_index {host_cell_index} out of bounds ({n_user_cells} cells)"
            ));
        }
        if !filled_hosts.insert(host_cell_index) {
            return Err(format!(
                "cell {} has more than one mesh fill",
                host_label(cells)
            ));
        }
        if !translation.iter().all(|t| t.is_finite())
            || !rotation_degrees.iter().all(|r| r.is_finite())
        {
            return Err(format!(
                "fill translation/rotation for {} must be finite",
                host_label(cells)
            ));
        }

        let topo = &mesh.topology;
        let num_volumes = topo.num_volumes;
        if num_volumes == 0 {
            return Err(format!(
                "mesh fill for {} has no volumes",
                host_label(cells)
            ));
        }
        // The host cell's material IS the complement; an implicit
        // complement material on the fill mesh would be silently ignored.
        if mesh_cells
            .last()
            .is_some_and(|ic| ic.material_idx.is_some())
        {
            return Err(format!(
                "the MeshGeometry filling {} has an implicit_complement_material; \
                 for a fill the host cell's own material is the complement, so set \
                 material= on the cell instead",
                host_label(cells)
            ));
        }

        // Watertightness: a hole in a volume makes point-in-volume and
        // ray crossings ambiguous, which genuinely loses particles.
        let leaky: Vec<u32> = (0..num_volumes)
            .filter(|&v| !volume_is_closed(topo, v))
            .collect();
        if !leaky.is_empty() {
            return Err(format!(
                "mesh fill for {}: volume(s) {leaky:?} are not watertight \
                 (every edge must be shared by exactly two triangles)",
                host_label(cells)
            ));
        }

        let rotation = rotation_matrix(rotation_degrees);
        let fill_idx = fills.len() as u32;
        let fill = MeshFill {
            mesh: Arc::clone(&mesh),
            translation,
            rotation_degrees,
            rotation,
            host_cell_index,
            first_volume_cell: cells.len(),
        };

        // Protrusion guard: every mesh vertex, placed in the CSG frame,
        // must lie inside the host region. This is a conservative skin
        // sample (a triangle can still bridge a concave notch between
        // vertices). Vertices exactly on the region boundary (a body
        // touching its container) are tolerated by re-testing a point
        // pulled slightly towards the mesh centre.
        if !allow_clipping {
            let host = &cells[host_cell_index];
            let aabb = &topo.global_aabb;
            let centre = fill.mesh_to_world_point([
                0.5 * (aabb[0] + aabb[3]),
                0.5 * (aabb[1] + aabb[4]),
                0.5 * (aabb[2] + aabb[5]),
            ]);
            const PULL: f64 = 1e-6;
            for v in &topo.vertices {
                let p = fill.mesh_to_world_point(*v);
                if host.contains((p[0], p[1], p[2])) {
                    continue;
                }
                let to_centre = [centre[0] - p[0], centre[1] - p[1], centre[2] - p[2]];
                let norm =
                    (to_centre[0].powi(2) + to_centre[1].powi(2) + to_centre[2].powi(2)).sqrt();
                let inward = if norm > 0.0 {
                    (
                        p[0] + to_centre[0] / norm * PULL,
                        p[1] + to_centre[1] / norm * PULL,
                        p[2] + to_centre[2] / norm * PULL,
                    )
                } else {
                    (p[0], p[1], p[2])
                };
                if !host.contains(inward) {
                    return Err(format!(
                        "mesh fill for {} protrudes past the host cell's region \
                         (vertex at ({:.6}, {:.6}, {:.6}) is outside). The CSG \
                         surface would silently clip the body; pass \
                         allow_clipping=True if that is intended",
                        host_label(cells),
                        p[0],
                        p[1],
                        p[2]
                    ));
                }
            }
        }

        // Fill materials join the flat store after the user's entries.
        // Per-volume clones of one material legitimately share an id, and
        // a fill material may share an id with a user material or another
        // fill's material only when it is the same material (matching
        // names); genuine collisions between different materials are
        // rejected. This matters across fills: each MeshGeometry
        // auto-assigns ids independently, so two separately built fills
        // hand out overlapping ids by default.
        let material_base = materials.len() as u32;
        for mat in &mesh_materials {
            let (id, name) = (mat.get_material_id(), mat.get_name());
            if let Some(id) = id {
                for existing in materials.iter() {
                    if existing.get_material_id() == Some(id) && existing.get_name() != name {
                        return Err(format!(
                            "mesh fill material {:?} (id {id}) collides with a \
                             different material of the same id in the geometry. \
                             Assign explicit distinct ids to the materials of \
                             each MeshGeometry",
                            name.unwrap_or("<unnamed>")
                        ));
                    }
                }
            }
            // With clipping allowed the effective body is mesh
            // intersected with the region, so the analytic mesh volume
            // propagated onto the per-volume material clones is an
            // overestimate; drop it rather than silently corrupting
            // volume-normalized results (transmutation, activity).
            // Geometry.calculate_volume can estimate the clipped value.
            if allow_clipping && mat.volume.is_some() {
                let mut stripped = (**mat).clone();
                stripped.volume = None;
                materials.push(Arc::new(stripped));
            } else {
                materials.push(Arc::clone(mat));
            }
        }

        // One embedded cell per mesh volume. The region is the host's
        // (an embedded volume lives inside the host region, and linear
        // containment scans must resolve to the host, which always
        // precedes these appended cells); spatial queries identify
        // embedded cells through their FillRole, never their region, and
        // they are excluded from the point-location BVH.
        let host_prefix = cells[host_cell_index].name.clone();
        let first_volume_cell = cells.len();
        for vol in 0..num_volumes {
            let mesh_cell = &mesh_cells[vol as usize];
            let base_name = mesh_cell
                .name
                .clone()
                .unwrap_or_else(|| format!("mesh_vol_{vol}"));
            let name = match &host_prefix {
                Some(h) => format!("{h}/{base_name}"),
                None => base_name,
            };
            let mut cell = Cell::new(
                None,
                cells[host_cell_index].region.clone(),
                Some(name),
                mesh_cell.material_idx.map(|i| i + material_base),
            );
            // Analytic mesh volume, unless clipping may have cut it.
            cell.volume = if allow_clipping {
                None
            } else {
                mesh_cell.volume
            };
            cell.fill_role = Some(FillRole::Embedded {
                fill: fill_idx,
                volume: vol,
            });
            cells.push(cell);
        }

        debug_assert_eq!(first_volume_cell, fill.first_volume_cell);
        cells[host_cell_index].fill_role = Some(FillRole::Host { fill: fill_idx });
        fills.push(fill);
    }

    Ok(fills)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const CUBE_ARROW: &str = "../yamt/tests/data/cube.arrow";

    fn cube_topology() -> yamt::MeshTopology {
        yamt::build_topology(yamt::read_arrow_mesh(std::path::Path::new(CUBE_ARROW)).unwrap())
            .unwrap()
    }

    #[test]
    fn rotation_matrix_identity_and_z90() {
        assert!(rotation_matrix([0.0, 0.0, 0.0]).is_none());
        // 90 degrees about z maps mesh-frame +x to world +y.
        let r = rotation_matrix([0.0, 0.0, 90.0]).unwrap();
        let fill = MeshFill {
            mesh: Arc::new(yamt::MeshGeometry::from_topology(cube_topology())),
            translation: [0.0, 0.0, 0.0],
            rotation_degrees: [0.0, 0.0, 90.0],
            rotation: Some(r),
            host_cell_index: 0,
            first_volume_cell: 1,
        };
        let p = fill.mesh_to_world_point([1.0, 0.0, 0.0]);
        assert!((p[0]).abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12);
        // Round trip.
        let q = fill.world_to_mesh_point(p);
        assert!((q[0] - 1.0).abs() < 1e-12 && q[1].abs() < 1e-12);
    }

    #[test]
    fn cube_volume_is_closed() {
        let topo = cube_topology();
        assert!(volume_is_closed(&topo, 0));
    }

    #[test]
    fn holed_cube_is_not_closed() {
        let mut topo = cube_topology();
        // Drop the last triangle of the first surface: opens a hole.
        let range = topo.surface_tri_ranges[0].clone();
        assert!(range.end > range.start);
        topo.surface_tri_ranges[0] = range.start..(range.end - 1);
        assert!(!volume_is_closed(&topo, 0));
    }
}
