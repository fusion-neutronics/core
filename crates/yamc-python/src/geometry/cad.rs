use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use yamm::mesher::{self, EdgeInput, FaceInput, MeshMode};
use yamt::io::arrow::{write_arrow_mesh, MeshData};

/// Mesh a single face in UV space.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (face_id, boundary_edges, holes=None, is_planar=true, mesh_mode="coarse".to_string()))]
pub fn mesh_face(
    face_id: u64,
    boundary_edges: Vec<Bound<'_, PyAny>>,
    holes: Option<Vec<Vec<Bound<'_, PyAny>>>>,
    is_planar: bool,
    mesh_mode: String,
) -> PyResult<PyFaceOutput> {
    let edges = parse_edges(&boundary_edges)?;

    let hole_edges: Vec<Vec<EdgeInput>> = match holes {
        Some(h) => h
            .iter()
            .map(|wire| parse_edges(wire))
            .collect::<PyResult<_>>()?,
        None => vec![],
    };

    let mode = parse_mode(&mesh_mode)?;

    let input = FaceInput {
        face_id,
        boundary_edges: edges,
        holes: hole_edges,
        is_planar,
        mode,
    };

    let output = mesher::mesh_face(&input)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyFaceOutput {
        face_id: output.face_id,
        interior_uv_vertices: output.interior_uv_vertices,
        triangles: output.triangles,
        num_boundary_vertices: output.num_boundary_vertices,
    })
}

fn parse_edges(edges: &[Bound<'_, PyAny>]) -> PyResult<Vec<EdgeInput>> {
    edges
        .iter()
        .map(|e| {
            let edge_id: u64 = e.get_item("edge_id")?.extract()?;
            let uv_points: Vec<[f64; 2]> = e.get_item("uv_points")?.extract()?;
            let reversed: bool = e.get_item("reversed")?.extract()?;
            Ok(EdgeInput {
                edge_id,
                uv_points,
                reversed,
            })
        })
        .collect()
}

fn parse_mode(mode: &str) -> PyResult<MeshMode> {
    if mode == "coarse" {
        Ok(MeshMode::Coarse)
    } else if let Some(len_str) = mode.strip_prefix("fine:") {
        let target_edge_length: f64 = len_str.parse().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid edge length: {e}"))
        })?;
        Ok(MeshMode::Fine { target_edge_length })
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid mesh_mode: '{mode}'. Use 'coarse' or 'fine:<length>'"
        )))
    }
}

/// Mesh multiple faces in batch.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (faces, scheduler="dag".to_string()))]
pub fn mesh_faces(faces: Vec<Bound<'_, PyAny>>, scheduler: String) -> PyResult<Vec<PyFaceOutput>> {
    let inputs = parse_face_inputs(&faces)?;

    let outputs = match scheduler.as_str() {
        "dag" => mesher::mesh_faces_dag(inputs),
        "parallel" => mesher::mesh_faces_parallel(&inputs),
        "sequential" => mesher::mesh_faces(&inputs),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid scheduler: '{}'. Use 'dag', 'parallel', or 'sequential'",
                scheduler
            )));
        }
    }
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(outputs
        .into_iter()
        .map(|o| PyFaceOutput {
            face_id: o.face_id,
            interior_uv_vertices: o.interior_uv_vertices,
            triangles: o.triangles,
            num_boundary_vertices: o.num_boundary_vertices,
        })
        .collect())
}

fn parse_face_inputs(faces: &[Bound<'_, PyAny>]) -> PyResult<Vec<FaceInput>> {
    faces
        .iter()
        .map(|f| {
            let face_id: u64 = f.get_item("face_id")?.extract()?;
            let boundary_edges: Vec<Bound<'_, PyAny>> = f.get_item("boundary_edges")?.extract()?;
            let edges = parse_edges(&boundary_edges)?;

            let holes: Option<Vec<Vec<Bound<'_, PyAny>>>> =
                f.get_item("holes").ok().and_then(|h| h.extract().ok());
            let hole_edges: Vec<Vec<EdgeInput>> = match holes {
                Some(h) => h
                    .iter()
                    .map(|wire| parse_edges(wire))
                    .collect::<PyResult<_>>()?,
                None => vec![],
            };

            let is_planar: bool = f
                .get_item("is_planar")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(true);
            let mesh_mode: String = f
                .get_item("mesh_mode")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or_else(|| "coarse".to_string());
            let mode = parse_mode(&mesh_mode)?;

            Ok(FaceInput {
                face_id,
                boundary_edges: edges,
                holes: hole_edges,
                is_planar,
                mode,
            })
        })
        .collect()
}

/// Python-visible face output.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "FaceOutput")]
pub struct PyFaceOutput {
    #[pyo3(get)]
    pub face_id: u64,
    #[pyo3(get)]
    pub interior_uv_vertices: Vec<[f64; 2]>,
    #[pyo3(get)]
    pub triangles: Vec<[usize; 3]>,
    #[pyo3(get)]
    pub num_boundary_vertices: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFaceOutput {
    fn __repr__(&self) -> String {
        format!(
            "PyFaceOutput(face_id={}, interior_uv_vertices={}, triangles={}, num_boundary_vertices={})",
            self.face_id,
            self.interior_uv_vertices.len(),
            self.triangles.len(),
            self.num_boundary_vertices
        )
    }
}

/// Compute adjacency/AABBs from the flat arrays and write the Arrow file.
#[allow(clippy::too_many_arguments)]
fn write_mesh_file(
    path: &str,
    vertices: Vec<f64>,
    triangles: Vec<u32>,
    triangle_surface_ids: Vec<u32>,
    triangle_physical_groups: Vec<i32>,
    tetrahedra: Vec<u32>,
    tet_volume_ids: Vec<u32>,
    tet_physical_groups: Vec<i32>,
    physical_groups_json: String,
    surface_volumes_json: String,
    volume_measures_json: String,
    imprinted: Option<bool>,
    contact_tolerance: Option<f64>,
    overlaps: Vec<[u32; 2]>,
    separated: Vec<[u32; 2]>,
) -> PyResult<()> {
    use yamt::mesh::cad_build::{tet_aabbs_flat, tet_adjacency_flat, tri_aabbs_flat};

    if !overlaps.is_empty() && imprinted != Some(false) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "overlaps only apply to a mesh written with imprinted=False: an imprinted \
             mesh has already split overlapping solids apart",
        ));
    }
    if !separated.is_empty() && imprinted != Some(false) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "separated only applies to a mesh written with imprinted=False",
        ));
    }
    if let Some(tol) = contact_tolerance {
        if !(tol.is_finite() && tol >= 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "contact_tolerance must be a non-negative number of cm, got {tol}"
            )));
        }
        if imprinted != Some(false) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "contact_tolerance only applies to a mesh written with imprinted=False: an \
                 imprinted mesh shares the surface between touching solids, so there is no \
                 contact to resolve",
            ));
        }
    }

    let tet_adjacency = tet_adjacency_flat(&tetrahedra);
    let tri_aabbs = tri_aabbs_flat(&vertices, &triangles);
    let tet_aabbs = tet_aabbs_flat(&vertices, &tetrahedra);

    let data = MeshData {
        vertices,
        triangles,
        triangle_surface_ids,
        triangle_physical_groups,
        tetrahedra,
        tet_volume_ids,
        tet_physical_groups,
        tet_adjacency,
        tri_aabbs,
        tet_aabbs,
        physical_groups_json,
        surface_volumes_json,
        volume_measures_json,
        imprinted,
        contact_tolerance,
        overlaps,
        separated,
    };

    write_arrow_mesh(&data, &PathBuf::from(path)).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to write Arrow file: {e}"))
    })
}

/// Write mesh data to an Arrow IPC file.
///
/// Tet adjacency and triangle/tet AABBs are computed here from the flat
/// connectivity (previously per-element Python loops), so the caller no longer
/// supplies them.
///
/// ``imprinted=False`` records that the solids were meshed one by one, so
/// touching solids each have their own surface where they meet;
/// :class:`MeshGeometry` then tracks the mesh discontinuously, with
/// ``contact_tolerance`` (cm) as the distance within which two solids'
/// surfaces count as touching. Both are left out of the file when ``None``.
/// ``overlaps`` lists ``(low, high)`` pairs of 0-based volume ids whose
/// solids overlap; where they do, the cell is ``high``.
/// ``separated`` lists ``(a, b)`` pairs of 0-based volume ids whose surfaces
/// come within the contact tolerance although the solids do not touch, so the
/// gap between them is kept rather than filled.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    path,
    vertices,
    triangles,
    triangle_surface_ids,
    triangle_physical_groups,
    tetrahedra=vec![],
    tet_volume_ids=vec![],
    tet_physical_groups=vec![],
    physical_groups_json="{}".to_string(),
    surface_volumes_json="[]".to_string(),
    volume_measures_json="".to_string(),
    imprinted=None,
    contact_tolerance=None,
    overlaps=vec![],
    separated=vec![],
))]
#[allow(clippy::too_many_arguments)]
pub fn mesh_to_arrow(
    path: String,
    vertices: Vec<f64>,
    triangles: Vec<u32>,
    triangle_surface_ids: Vec<u32>,
    triangle_physical_groups: Vec<i32>,
    tetrahedra: Vec<u32>,
    tet_volume_ids: Vec<u32>,
    tet_physical_groups: Vec<i32>,
    physical_groups_json: String,
    surface_volumes_json: String,
    volume_measures_json: String,
    imprinted: Option<bool>,
    contact_tolerance: Option<f64>,
    overlaps: Vec<(u32, u32)>,
    separated: Vec<(u32, u32)>,
) -> PyResult<()> {
    write_mesh_file(
        &path,
        vertices,
        triangles,
        triangle_surface_ids,
        triangle_physical_groups,
        tetrahedra,
        tet_volume_ids,
        tet_physical_groups,
        physical_groups_json,
        surface_volumes_json,
        volume_measures_json,
        imprinted,
        contact_tolerance,
        overlaps.into_iter().map(|(l, h)| [l, h]).collect(),
        separated.into_iter().map(|(a, b)| [a, b]).collect(),
    )
}

/// Physical group id for a solid: group ids equal solid ids, and a solid
/// without a material tag maps to the -1 sentinel.
fn pg_of(solid_id: u32, n_tags: usize) -> i32 {
    if solid_id >= 1 && solid_id as usize <= n_tags {
        solid_id as i32
    } else {
        -1
    }
}

/// Build the `yamc.physical_groups` JSON for a CAD export.
///
/// One `dim=3` group per material, numbered `1..` in `material_tags` order (so
/// group id `i + 1` is solid `i + 1`, which the reader maps to volume `i`),
/// followed by one `dim=2` group per boundary tag carrying the surface ids it
/// applies to. Boundary group names are prefixed `boundary:` so the reader's
/// group-name parser turns them into boundary conditions.
fn physical_groups_json(material_tags: &[String], boundary_tags: &[(String, Vec<u32>)]) -> String {
    let mut groups = serde_json::Map::new();
    for (i, tag) in material_tags.iter().enumerate() {
        groups.insert(
            (i + 1).to_string(),
            serde_json::json!({"name": format!("mat:{tag}"), "dim": 3}),
        );
    }
    for (id, (name, surface_ids)) in (material_tags.len() + 1..).zip(boundary_tags) {
        groups.insert(
            id.to_string(),
            serde_json::json!({
                "name": format!("boundary:{name}"),
                "dim": 2,
                "surface_ids": surface_ids,
            }),
        );
    }
    serde_json::Value::Object(groups).to_string()
}

/// Finalize the CAD pipeline's mesh and write it to an Arrow IPC file.
///
/// Runs the whole export in the compiled core: assigns each
/// triangle and tet the physical group of its first owning solid, offsets the
/// per-solid tet blocks into the global vertex array, builds the physical-group
/// and surface-to-volume topology metadata, and writes the file. The Python
/// exporter does no per-element work.
///
/// Args:
///     path: Output file path.
///     vertices: Global ``[x, y, z]`` vertex array (tet vertex blocks appended).
///     triangles: Per-triangle global vertex indices.
///     triangle_surface_ids: 1-based surface id per triangle.
///     triangle_face_ids: BRep face id per triangle.
///     solid_faces: ``(solid_id, face_ids)`` pairs in ascending solid order.
///     material_tags: Material tag per solid; solid ``i + 1`` carries tag ``i``.
///     face_to_surface_id: ``(face_id, surface_id)`` pairs (surface ids 1-based).
///     face_solid_reversed: ``(solid_id, face_id) -> reversed`` orientation map.
///     tet_blocks: ``(solid_id, vertex_offset, tets)`` per tetrahedralized solid.
///     boundary_tags: ``(boundary_name, surface_ids)`` pairs written as
///         ``dim=2`` ``boundary:<name>`` physical groups (surface ids 1-based).
///     imprinted: Whether the assembly was imprinted before meshing. ``False``
///         makes :class:`MeshGeometry` track the mesh discontinuously.
///     contact_tolerance: For a mesh that was not imprinted, the distance
///         (cm) within which two solids' surfaces count as touching.
///     overlaps: For a mesh that was not imprinted, ``(low, high)`` solid
///         pairs (1-based solid ids) that overlap; where they do, the cell is
///         ``high``.
///     volume_measures: Optional volume (cm³) per solid, in solid order,
///         written instead of the volumes the reader would compute from the
///         surfaces (for solids that give up an overlap).
///     separated: For a mesh that was not imprinted, ``(a, b)`` solid pairs
///         (1-based solid ids) within the contact tolerance in the mesh that
///         do not touch in the CAD, so the gap between them is kept.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    path,
    vertices,
    triangles,
    triangle_surface_ids,
    triangle_face_ids,
    solid_faces,
    material_tags,
    face_to_surface_id,
    face_solid_reversed,
    tet_blocks=vec![],
    boundary_tags=vec![],
    imprinted=true,
    contact_tolerance=None,
    overlaps=vec![],
    volume_measures=None,
    separated=vec![],
))]
#[allow(clippy::too_many_arguments)]
pub fn cad_mesh_to_arrow(
    path: String,
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    triangle_surface_ids: Vec<u32>,
    triangle_face_ids: Vec<i64>,
    solid_faces: Vec<(u32, Vec<i64>)>,
    material_tags: Vec<String>,
    face_to_surface_id: Vec<(i64, u32)>,
    face_solid_reversed: HashMap<(u32, i64), bool>,
    tet_blocks: Vec<(u32, usize, Vec<[u32; 4]>)>,
    boundary_tags: Vec<(String, Vec<u32>)>,
    imprinted: bool,
    contact_tolerance: Option<f64>,
    overlaps: Vec<(u32, u32)>,
    volume_measures: Option<Vec<f64>>,
    separated: Vec<(u32, u32)>,
) -> PyResult<()> {
    use yamt::mesh::cad_build::{
        face_owners, flatten_tet_blocks, surface_volume_pairs, triangle_owner_ids,
    };

    let owners = face_owners(&solid_faces);
    let n_tags = material_tags.len();

    let triangle_physical_groups: Vec<i32> = triangle_owner_ids(&triangle_face_ids, &owners)
        .into_iter()
        .map(|s| if s > 0 { pg_of(s as u32, n_tags) } else { -1 })
        .collect();

    let (tets, tet_volume_ids) = flatten_tet_blocks(&tet_blocks);
    let tet_physical_groups: Vec<i32> = tet_volume_ids.iter().map(|&s| pg_of(s, n_tags)).collect();

    let surface_volumes = surface_volume_pairs(&face_to_surface_id, &owners, &face_solid_reversed);

    write_mesh_file(
        &path,
        vertices.into_iter().flatten().collect(),
        triangles.into_iter().flatten().collect(),
        triangle_surface_ids,
        triangle_physical_groups,
        tets.into_iter().flatten().collect(),
        tet_volume_ids,
        tet_physical_groups,
        physical_groups_json(&material_tags, &boundary_tags),
        serde_json::to_string(&surface_volumes).expect("serializing surface_volumes"),
        volume_measures
            .map(|v| serde_json::to_string(&v).expect("serializing volume_measures"))
            .unwrap_or_default(),
        Some(imprinted),
        contact_tolerance,
        overlaps
            .into_iter()
            .map(|(low, high)| {
                if low == 0 || high == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "overlaps take 1-based solid ids",
                    ));
                }
                Ok([low - 1, high - 1])
            })
            .collect::<PyResult<_>>()?,
        separated
            .into_iter()
            .map(|(a, b)| {
                if a == 0 || b == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "separated takes 1-based solid ids",
                    ));
                }
                Ok([a - 1, b - 1])
            })
            .collect::<PyResult<_>>()?,
    )
}

/// Per-cell volume and material labels for the CAD VTKHDF export.
///
/// Returns ``(tri_volume_ids, tri_material_ids, tets, tet_volume_ids)``: the
/// owning solid id (1-based, -1 when unowned) and material tag index (0-based,
/// -1 when unowned) per triangle, plus the per-solid tet blocks flattened into
/// the global vertex array with the owning solid id per tet.
///
/// Args:
///     triangle_face_ids: BRep face id per triangle.
///     solid_faces: ``(solid_id, face_ids)`` pairs in ascending solid order.
///     tet_blocks: ``(solid_id, vertex_offset, tets)`` per tetrahedralized solid.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (triangle_face_ids, solid_faces, tet_blocks=vec![]))]
#[allow(clippy::type_complexity)]
pub fn cad_mesh_labels(
    triangle_face_ids: Vec<i64>,
    solid_faces: Vec<(u32, Vec<i64>)>,
    tet_blocks: Vec<(u32, usize, Vec<[u32; 4]>)>,
) -> (Vec<i32>, Vec<i32>, Vec<[u32; 4]>, Vec<u32>) {
    use yamt::mesh::cad_build::{face_owners, flatten_tet_blocks, triangle_owner_ids};

    let owners = face_owners(&solid_faces);
    let tri_volume_ids = triangle_owner_ids(&triangle_face_ids, &owners);
    let tri_material_ids: Vec<i32> = tri_volume_ids
        .iter()
        .map(|&s| if s > 0 { s - 1 } else { -1 })
        .collect();
    let (tets, tet_volume_ids) = flatten_tet_blocks(&tet_blocks);
    (tri_volume_ids, tri_material_ids, tets, tet_volume_ids)
}

/// Pairs of solids whose surface meshes overlap deeper than ``depth_tolerance``.
///
/// Every vertex, edge and face interior of each solid's closed boundary is
/// tested against every other solid whose bounding box it reaches (the
/// compiled mesh-level overlap check). Contact is not overlap: a point within
/// ``depth_tolerance`` of the other surface counts as touching it.
///
/// Args:
///     vertices: Global ``[x, y, z]`` vertex array.
///     solids: ``(solid_id, triangles)`` per solid, each its full closed
///         boundary as ``[i, j, k]`` indices into ``vertices``.
///     depth_tolerance: Penetration (3D units) at or below which a feature
///         counts as touching rather than inside.
///
/// Returns:
///     ``[(solid_a, solid_b, max_depth, exposure), ...]`` for every
///     overlapping pair, ``max_depth`` the deepest penetration found and
///     ``exposure`` an order-of-magnitude estimate of the fraction of
///     isotropic tracks through the model that cross the overlap.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (vertices, solids, depth_tolerance))]
pub fn solid_overlaps(
    vertices: Vec<[f64; 3]>,
    solids: Vec<(usize, Vec<[usize; 3]>)>,
    depth_tolerance: f64,
) -> Vec<(usize, usize, f64, f64)> {
    yamm::overlap::find_solid_overlaps(&vertices, &solids, &[], depth_tolerance)
        .into_iter()
        .map(|o| (o.solid_a, o.solid_b, o.max_depth, o.exposure))
        .collect()
}

/// Volume shared by two solids' closed surface meshes.
///
/// Args:
///     vertices: Global ``[x, y, z]`` vertex array.
///     triangles_a: Closed boundary of the first solid, as indices into
///         ``vertices``.
///     triangles_b: Closed boundary of the second solid.
///     tolerance: Distance below which the two surfaces count as touching.
///     samples: Points sampled when the volume has to be estimated.
///     seed: Seed of the deterministic sampler.
///
/// Returns:
///     ``(volume, std_err, embedded)``. When the surfaces never come within
///     ``tolerance`` of each other and one solid lies inside the other,
///     ``embedded`` is ``0`` or ``1`` (that solid is wholly shared; ``volume``
///     is then 0 and the caller supplies the solid's own volume). Otherwise
///     ``embedded`` is ``None`` and ``volume`` is a Monte Carlo estimate with
///     standard error ``std_err``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (vertices, triangles_a, triangles_b, tolerance, samples=200_000, seed=1))]
pub fn solid_shared_volume(
    vertices: Vec<[f64; 3]>,
    triangles_a: Vec<[usize; 3]>,
    triangles_b: Vec<[usize; 3]>,
    tolerance: f64,
    samples: usize,
    seed: u64,
) -> (f64, f64, Option<usize>) {
    match yamm::overlap::shared_volume(
        &vertices,
        &triangles_a,
        &triangles_b,
        tolerance,
        samples,
        seed,
    ) {
        yamm::overlap::SharedVolume::Embedded(i) => (0.0, 0.0, Some(i)),
        yamm::overlap::SharedVolume::Estimate { volume, std_err } => (volume, std_err, None),
    }
}

/// Points where two solids' surface meshes come within ``tolerance``.
///
/// For every pair of solids whose meshes come that close, up to ``samples``
/// mesh vertices of one of them lying within ``tolerance`` of the other's
/// surface. A mesh vertex lies on its solid's CAD surface, so its exact
/// distance to the other CAD solid says whether the two touch there or are
/// separated by a real gap that faceting has narrowed.
///
/// Args:
///     vertices: Global ``[x, y, z]`` vertex array.
///     solids: ``(solid_id, triangles)`` per solid.
///     tolerance: The contact tolerance (3D units).
///     samples: Points to return per pair at most.
///
/// Returns:
///     ``[(solid_a, solid_b, [[x, y, z], ...]), ...]``, the points on
///     ``solid_a``'s surface. A pair whose meshes come close only between
///     vertices returns no points, and so does a pair whose meshes share a
///     vertex (which lies on both CAD surfaces, so the two touch).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (vertices, solids, tolerance, samples=8))]
#[allow(clippy::type_complexity)]
pub fn solid_contact_samples(
    vertices: Vec<[f64; 3]>,
    solids: Vec<(usize, Vec<[usize; 3]>)>,
    tolerance: f64,
    samples: usize,
) -> Vec<(usize, usize, Vec<[f64; 3]>)> {
    yamm::overlap::contact_samples(&vertices, &solids, tolerance, samples)
}

/// Weld coincident vertices of a triangle mesh into a watertight, index-shared
/// boundary (the compiled core of the CAD pipeline's vertex welding).
///
/// Args:
///     vertices: List of ``[x, y, z]`` coordinates.
///     triangles: List of ``[i, j, k]`` vertex indices.
///     rel_tol: Merge tolerance as a fraction of the mesh extent.
///
/// Returns:
///     ``(welded_vertices, welded_triangles, kept)``: the merged vertices, the
///     triangles reindexed into them (degenerate ones dropped), and the
///     original indices of the surviving triangles (to filter parallel arrays).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (vertices, triangles, rel_tol=1e-6))]
pub fn weld_mesh(
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    rel_tol: f64,
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>, Vec<usize>) {
    yamt::mesh::cad_build::weld_vertices(&vertices, &triangles, rel_tol)
}

#[cfg(test)]
mod tests {
    use super::physical_groups_json;

    #[test]
    fn physical_groups_json_materials_only() {
        let json = physical_groups_json(&["fuel".into(), "moderator".into()], &[]);
        assert_eq!(
            json,
            r#"{"1":{"dim":3,"name":"mat:fuel"},"2":{"dim":3,"name":"mat:moderator"}}"#
        );
    }

    #[test]
    fn physical_groups_json_appends_boundary_groups_after_materials() {
        let json = physical_groups_json(
            &["fuel".into(), "moderator".into()],
            &[("vacuum".into(), vec![2, 3, 4, 5, 6, 7])],
        );
        assert_eq!(
            json,
            r#"{"1":{"dim":3,"name":"mat:fuel"},"2":{"dim":3,"name":"mat:moderator"},"3":{"dim":2,"name":"boundary:vacuum","surface_ids":[2,3,4,5,6,7]}}"#
        );
    }
}
