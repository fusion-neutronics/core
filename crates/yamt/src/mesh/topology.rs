//! Mesh topology: vertices, triangles, tetrahedra, adjacency, and
//! volume/surface groupings. The runtime representation the geometric
//! queries operate on, built from the Arrow IPC mesh reader.

use std::collections::HashMap;

use crate::geometry::MeshError;
use crate::mesh::physical_groups::PhysicalGroupData;
use crate::types::*;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The implicit complement volume ID is always the last volume.
pub const IMPLICIT_COMPLEMENT_MATERIAL: &str = "void";

/// Canonical tet face vertex ordering.
/// Face i is opposite to vertex i.
/// Vertices are ordered so that the outward normal points away from the
/// opposite vertex.
///
/// That last property only holds for a *positively oriented* tet
/// (`signed_tet_volume > 0`); for a negatively oriented one every normal
/// points inward instead. A mesh file therefore has to arrive with every tet
/// positively oriented: [`validate_tet_orientation`] rejects one that does not
/// at construction, so the property is an invariant of
/// `MeshTopology::tetrahedra`.
pub const TET_FACE_VERTICES: [[usize; 3]; 4] = [
    [1, 2, 3], // Face 0: opposite vertex 0
    [0, 3, 2], // Face 1: opposite vertex 1
    [0, 1, 3], // Face 2: opposite vertex 2
    [0, 2, 1], // Face 3: opposite vertex 3
];

// ---------------------------------------------------------------------------
// Main data structure
// ---------------------------------------------------------------------------

/// Complete mesh topology built from an Arrow IPC mesh file.
#[derive(Debug, Clone)]
pub struct MeshTopology {
    // -- Vertex storage --
    /// Dense vertex coordinates: `[x, y, z]` indexed by `VertexId`.
    pub vertices: Vec<[f64; 3]>,

    // -- Triangle storage --
    /// Dense triangle connectivity: `[v0, v1, v2]` indexed by `TriangleId`.
    pub triangles: Vec<[VertexId; 3]>,

    // -- Tetrahedron storage --
    /// Dense tet connectivity: `[v0, v1, v2, v3]` indexed by `TetrahedronId`.
    ///
    /// Every tet is positively oriented (see [`validate_tet_orientation`],
    /// which refuses a mesh where one is not), so [`TET_FACE_VERTICES`]
    /// yields outward-pointing face normals.
    pub tetrahedra: Vec<[VertexId; 4]>,

    // -- Volume groupings --
    /// Number of volumes (not counting implicit complement).
    pub num_volumes: u32,
    /// Tetrahedra belonging to each volume: `volume_tets[vol_id]` → range
    /// into `volume_tet_indices`.
    pub volume_tet_ranges: Vec<std::ops::Range<u32>>,
    /// Flat buffer of tet indices, sliced by `volume_tet_ranges`.
    pub volume_tet_indices: Vec<TetrahedronId>,
    /// Inverse of the two above: the volume each tet belongs to, indexed by
    /// `TetrahedronId`, or `None` for a tet in no volume. Element walking
    /// needs the O(1) direction to stop at a volume boundary (tet adjacency
    /// is global and crosses volumes on a conformal mesh).
    pub tet_volume_ids: Vec<Option<VolumeId>>,

    // -- Surface groupings --
    /// Number of surfaces.
    pub num_surfaces: u32,
    /// Triangles belonging to each surface: `surface_tri_ranges[surf_id]`
    /// → range into `surface_tri_indices`.
    pub surface_tri_ranges: Vec<std::ops::Range<u32>>,
    /// Flat buffer of triangle indices.
    pub surface_tri_indices: Vec<TriangleId>,

    // -- Volume ↔ Surface relationships --
    /// Surfaces bounding each volume, with sense.
    pub volume_surfaces: Vec<Vec<(SurfaceId, Sense)>>,
    /// For each surface: `(forward_volume, reverse_volume)`.
    /// Forward = surface normal points outward from this volume.
    pub surface_volumes: Vec<(Option<VolumeId>, Option<VolumeId>)>,

    // -- Tet adjacency --
    /// For each tet, 4 neighbor tets (one per face). `None` = boundary face.
    pub tet_adjacency: Vec<[Option<TetrahedronId>; 4]>,

    // -- Bounding boxes --
    /// AABB per triangle: `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    pub triangle_aabbs: Vec<[f64; 6]>,
    /// AABB per tetrahedron.
    pub tet_aabbs: Vec<[f64; 6]>,
    /// AABB per surface.
    pub surface_aabbs: Vec<[f64; 6]>,
    /// AABB per volume.
    pub volume_aabbs: Vec<[f64; 6]>,
    /// Global AABB.
    pub global_aabb: [f64; 6],

    // -- Volume measures (divergence theorem) --
    /// Analytical volume of each mesh volume (cm³), computed via the
    /// divergence theorem over the bounding surface triangles.
    /// Length = `num_volumes`. Empty if not yet computed.
    pub volume_measures: Vec<f64>,

    // -- Physical group metadata --
    pub physical_data: PhysicalGroupData,

    // -- Implicit complement --
    /// Volume ID of the implicit complement (virtual "outside" volume).
    pub implicit_complement: VolumeId,

    // -- Provenance --
    /// `Some(false)` when the writer meshed every solid on its own (no
    /// imprinting), so touching solids each have their own surface where they
    /// meet. Selects discontinuous tracking in
    /// [`crate::geometry::MeshGeometry::from_topology`].
    pub imprinted: Option<bool>,
    /// Contact tolerance (cm) the writer recorded for a non-imprinted mesh.
    pub contact_tolerance: Option<f64>,
    /// Overlapping solids of a non-imprinted mesh, as `(low, high)` pairs:
    /// where the two overlap, the cell is `high`. Free of cycles (checked
    /// when the topology is built).
    pub overlaps: Vec<(VolumeId, VolumeId)>,
    /// Solid pairs `(a, b)`, `a < b`, that come within the contact tolerance
    /// in the mesh although the CAD solids do not touch: a real gap, which
    /// relocation must not fill.
    pub separated: Vec<(VolumeId, VolumeId)>,
}

/// Rank every volume so that, for each `(low, high)` overlap pair, `high`
/// outranks `low`: `ranks[v]` is the length of the longest chain of pairs
/// ending at `v`. `None` when the pairs contain a cycle.
pub fn overlap_ranks(num_volumes: u32, pairs: &[(VolumeId, VolumeId)]) -> Option<Vec<u32>> {
    let n = num_volumes as usize;
    let mut indegree = vec![0u32; n];
    let mut above: Vec<Vec<VolumeId>> = vec![Vec::new(); n];
    for &(low, high) in pairs {
        above[low as usize].push(high);
        indegree[high as usize] += 1;
    }
    let mut ranks = vec![0u32; n];
    let mut ready: Vec<usize> = (0..n).filter(|&v| indegree[v] == 0).collect();
    let mut seen = 0;
    while let Some(v) = ready.pop() {
        seen += 1;
        for &h in &above[v] {
            let h = h as usize;
            ranks[h] = ranks[h].max(ranks[v] + 1);
            indegree[h] -= 1;
            if indegree[h] == 0 {
                ready.push(h);
            }
        }
    }
    (seen == n).then_some(ranks)
}

// ---------------------------------------------------------------------------
// Tet adjacency
// ---------------------------------------------------------------------------

/// Invert `volume_tet_ranges` / `volume_tet_indices` into a per-tet volume id.
pub fn build_tet_volume_ids(
    num_tets: usize,
    volume_tet_ranges: &[std::ops::Range<u32>],
    volume_tet_indices: &[TetrahedronId],
) -> Vec<Option<VolumeId>> {
    let mut out = vec![None; num_tets];
    for (vol_id, range) in volume_tet_ranges.iter().enumerate() {
        for i in range.start..range.end {
            out[volume_tet_indices[i as usize] as usize] = Some(vol_id as VolumeId);
        }
    }
    out
}

/// Reject a mesh that contains a negatively oriented tet.
///
/// [`TET_FACE_VERTICES`] only produces outward-pointing face normals for a tet
/// whose signed volume is positive; for a negatively oriented tet the same
/// orderings produce inward normals. The element walk picks its exit face with
/// `dot(direction, normal) > 0`, so on a negatively oriented tet it selects an
/// *entry* face instead: the walk hops backwards or stops at the mesh boundary
/// early, and unstructured track-length tallies read about 33 percent low.
///
/// yamt used to re-wind such a tet silently. It no longer does, for two
/// reasons. Silently repairing a mesh hides the producer's bug, so the same
/// broken file keeps circulating and only the symptom is patched. And the
/// producers have since been narrowed: yamm, yamc's own mesher, guarantees
/// positive orientation at its output boundary, and the gmsh input route is
/// gone, so Arrow IPC is the only way a mesh enters. A negatively oriented tet
/// therefore means a broken or third-party writer, which is worth saying out
/// loud.
///
/// Only a strictly negative signed volume is an orientation error; an exactly
/// degenerate (zero-volume) tet has no orientation to correct and is out of
/// scope here.
///
/// O(n) in the number of tets, allocation-free, and returns at the first
/// offender.
pub fn validate_tet_orientation(
    tetrahedra: &[[VertexId; 4]],
    vertices: &[[f64; 3]],
) -> Result<(), MeshError> {
    for (tet_index, verts) in tetrahedra.iter().enumerate() {
        let signed = crate::query::intersect::signed_tet_volume(
            vertices[verts[0] as usize],
            vertices[verts[1] as usize],
            vertices[verts[2] as usize],
            vertices[verts[3] as usize],
        );
        if signed < 0.0 {
            return Err(MeshError::NegativeTetOrientation {
                tet_index,
                signed_volume: signed,
            });
        }
    }
    Ok(())
}

/// Refuse a volume whose boundary lists the same facet twice.
///
/// Two distinct triangles bounding one volume with vertices at the same three
/// positions (compared by bits, as a CSG mesh fill welds them) are either a
/// copy, facing the same way relative to the volume, or a contradiction,
/// facing opposite ways. A copy is counted twice in `volume_measures` and in
/// the ray parity point location uses, so particles born inside it are lost;
/// a contradiction says the volume lies on both sides of one face.
///
/// The one legitimate pair is two opposite-facing triangles on surfaces with
/// the volume on both sides, an internal face listed from each side. The same
/// triangle reached twice through such a surface (once per sense) is one
/// facet, not a copy.
///
/// The implicit complement is not checked: touching solids of a mesh made
/// without imprinting each put their own coincident facets on its boundary,
/// and a copy on a solid's boundary is found when that solid is checked.
pub fn validate_no_duplicate_facets(topo: &MeshTopology) -> Result<(), MeshError> {
    let mut welded: HashMap<[u64; 3], VertexId> = HashMap::with_capacity(topo.vertices.len());
    let canonical: Vec<VertexId> = topo
        .vertices
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
            *welded.entry(key).or_insert(i as VertexId)
        })
        .collect();
    drop(welded);

    struct Seen {
        triangle: TriangleId,
        surface: SurfaceId,
        winding: bool,
        facing: bool,
        both_sides: bool,
    }
    let mut seen: HashMap<[VertexId; 3], Seen> = HashMap::new();
    for volume in 0..topo.num_volumes {
        seen.clear();
        for &(surface, sense) in &topo.volume_surfaces[volume as usize] {
            let both_sides = topo.surface_volumes[surface as usize] == (Some(volume), Some(volume));
            for &triangle in &topo.surface_tri_indices[topo.surface_tri_ranges[surface as usize]
                .start as usize
                ..topo.surface_tri_ranges[surface as usize].end as usize]
            {
                let t = topo.triangles[triangle as usize];
                let mut key = [
                    canonical[t[0] as usize],
                    canonical[t[1] as usize],
                    canonical[t[2] as usize],
                ];
                // The winding is the parity of the sort that orders the key.
                let mut winding = false;
                for (i, j) in [(0, 1), (1, 2), (0, 1)] {
                    if key[i] > key[j] {
                        key.swap(i, j);
                        winding = !winding;
                    }
                }
                if key[0] == key[1] || key[1] == key[2] {
                    continue; // a degenerate triangle has no facing to compare
                }
                let facing = winding ^ (sense == Sense::Reverse);
                match seen.entry(key) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(Seen {
                            triangle,
                            surface,
                            winding,
                            facing,
                            both_sides,
                        });
                    }
                    std::collections::hash_map::Entry::Occupied(e) => {
                        let first = e.get();
                        if first.triangle == triangle {
                            continue;
                        }
                        let same_facing = if first.both_sides || both_sides {
                            if first.both_sides && both_sides && first.winding != winding {
                                continue;
                            }
                            first.winding == winding
                        } else {
                            first.facing == facing
                        };
                        return Err(MeshError::DuplicateFacet {
                            volume,
                            triangles: [first.triangle, triangle],
                            surfaces: [first.surface, surface],
                            same_facing,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

/// Build tet-to-tet adjacency by hashing shared faces.
///
/// For each tet face (4 per tet), hash the sorted vertex triple. When the same
/// face key appears in two tets, they are neighbors across that face.
///
/// Complexity: O(n) where n = number of tets.
pub fn build_tet_adjacency(tetrahedra: &[[VertexId; 4]]) -> Vec<[Option<TetrahedronId>; 4]> {
    let mut adjacency = vec![[None; 4]; tetrahedra.len()];

    // Map from sorted face vertex triple → (tet_id, local_face_index)
    let mut face_map: HashMap<[VertexId; 3], (TetrahedronId, u8)> =
        HashMap::with_capacity(tetrahedra.len() * 2);

    for (tet_idx, verts) in tetrahedra.iter().enumerate() {
        let tet_id = tet_idx as TetrahedronId;

        for (face_idx, face_verts) in TET_FACE_VERTICES.iter().enumerate() {
            let mut fv = [
                verts[face_verts[0]],
                verts[face_verts[1]],
                verts[face_verts[2]],
            ];
            fv.sort();

            match face_map.entry(fv) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert((tet_id, face_idx as u8));
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    let (other_tet, other_face) = e.remove();
                    adjacency[tet_idx][face_idx] = Some(other_tet);
                    adjacency[other_tet as usize][other_face as usize] = Some(tet_id);
                }
            }
        }
    }

    adjacency
}

// ---------------------------------------------------------------------------
// Bounding boxes
// ---------------------------------------------------------------------------

pub fn build_triangle_aabbs(triangles: &[[VertexId; 3]], vertices: &[[f64; 3]]) -> Vec<[f64; 6]> {
    triangles
        .iter()
        .map(|tri| {
            let v0 = vertices[tri[0] as usize];
            let v1 = vertices[tri[1] as usize];
            let v2 = vertices[tri[2] as usize];
            [
                v0[0].min(v1[0]).min(v2[0]),
                v0[1].min(v1[1]).min(v2[1]),
                v0[2].min(v1[2]).min(v2[2]),
                v0[0].max(v1[0]).max(v2[0]),
                v0[1].max(v1[1]).max(v2[1]),
                v0[2].max(v1[2]).max(v2[2]),
            ]
        })
        .collect()
}

pub fn build_tet_aabbs(tetrahedra: &[[VertexId; 4]], vertices: &[[f64; 3]]) -> Vec<[f64; 6]> {
    tetrahedra
        .iter()
        .map(|tet| {
            let v0 = vertices[tet[0] as usize];
            let v1 = vertices[tet[1] as usize];
            let v2 = vertices[tet[2] as usize];
            let v3 = vertices[tet[3] as usize];
            [
                v0[0].min(v1[0]).min(v2[0]).min(v3[0]),
                v0[1].min(v1[1]).min(v2[1]).min(v3[1]),
                v0[2].min(v1[2]).min(v2[2]).min(v3[2]),
                v0[0].max(v1[0]).max(v2[0]).max(v3[0]),
                v0[1].max(v1[1]).max(v2[1]).max(v3[1]),
                v0[2].max(v1[2]).max(v2[2]).max(v3[2]),
            ]
        })
        .collect()
}

// ----------------------------- Tests -----------------------------

#[cfg(all(test, feature = "arrow"))]
mod tests {
    use super::*;

    fn cube_topology() -> MeshTopology {
        crate::io::arrow::build_topology(
            crate::io::arrow::read_arrow_mesh(std::path::Path::new("tests/data/cube.arrow"))
                .unwrap(),
        )
        .unwrap()
    }

    fn two_region_topology() -> MeshTopology {
        crate::io::arrow::build_topology(
            crate::io::arrow::read_arrow_mesh(std::path::Path::new(
                "tests/data/two_region_tets.arrow",
            ))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn test_cube_topology() {
        let topo = cube_topology();

        // 8 welded surface corners, then the tet mesher's own vertex block for
        // the solid appended after them (the boundary it was handed, with no
        // interior points needed at this coarseness): 8 + 8 = 16.
        assert_eq!(topo.vertices.len(), 16);
        assert_eq!(topo.num_volumes, 1);
        assert_eq!(topo.num_surfaces, 6);
        assert!(!topo.triangles.is_empty());
        assert!(!topo.tetrahedra.is_empty());

        // Check global bounding box: unit cube
        assert!((topo.global_aabb[0] - 0.0).abs() < 1e-10);
        assert!((topo.global_aabb[1] - 0.0).abs() < 1e-10);
        assert!((topo.global_aabb[2] - 0.0).abs() < 1e-10);
        assert!((topo.global_aabb[3] - 1.0).abs() < 1e-10);
        assert!((topo.global_aabb[4] - 1.0).abs() < 1e-10);
        assert!((topo.global_aabb[5] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cube_physical_groups() {
        let topo = cube_topology();

        // Volume 0 should be "water"
        let mat = &topo.physical_data.volume_materials[&0];
        assert_eq!(mat.material_name.as_deref(), Some("water"));

        // All surfaces should be vacuum
        for surf_id in 0..topo.num_surfaces {
            let bc = topo.physical_data.surface_bcs.get(&surf_id);
            assert_eq!(bc, Some(&BoundaryCondition::Vacuum));
        }
    }

    #[test]
    fn test_tet_adjacency() {
        let topo = cube_topology();

        // Every shared face should have been found exactly twice
        // (once from each side). Verify: for each tet, if adj[face] = Some(other),
        // then other has this tet as a neighbor too.
        for (tet_idx, adj) in topo.tet_adjacency.iter().enumerate() {
            for (face, &neighbor) in adj.iter().enumerate() {
                if let Some(other) = neighbor {
                    // The other tet should have us as a neighbor
                    let other_adj = &topo.tet_adjacency[other as usize];
                    let has_back_link = other_adj.contains(&Some(tet_idx as TetrahedronId));
                    assert!(
                        has_back_link,
                        "Tet {tet_idx} face {face} → tet {other}, but no back-link"
                    );
                }
            }
        }
    }

    #[test]
    fn test_surface_senses() {
        let topo = cube_topology();

        // Each surface should have exactly 2 parent volumes
        // (one real + implicit complement for outer surfaces)
        for surf_id in 0..topo.num_surfaces {
            let (fwd, rev) = topo.surface_volumes[surf_id as usize];
            assert!(
                fwd.is_some() && rev.is_some(),
                "Surface {surf_id} missing parent volume: fwd={fwd:?}, rev={rev:?}"
            );
        }
    }

    #[test]
    fn test_two_region_topology() {
        let topo = two_region_topology();

        assert_eq!(topo.num_volumes, 2);
        // 12 welded surface vertices (two conformal half-boxes sharing the 4
        // corners of the x = 0.5 interface), then one 8-vertex tet block per
        // solid appended after them: 12 + 8 + 8 = 28.
        assert_eq!(topo.vertices.len(), 28);

        // Check materials
        let mat0 = &topo.physical_data.volume_materials[&0];
        let mat1 = &topo.physical_data.volume_materials[&1];
        assert_eq!(mat0.material_name.as_deref(), Some("fuel"));
        assert_eq!(mat1.material_name.as_deref(), Some("moderator"));
    }

    #[test]
    fn test_implicit_complement() {
        let topo = cube_topology();

        // Implicit complement should be volume num_volumes
        assert_eq!(topo.implicit_complement, topo.num_volumes);

        // At least some surfaces should have the implicit complement as a parent
        let has_ic = topo.surface_volumes.iter().any(|(fwd, rev)| {
            *fwd == Some(topo.implicit_complement) || *rev == Some(topo.implicit_complement)
        });
        assert!(has_ic, "No surfaces reference the implicit complement");
    }
}
