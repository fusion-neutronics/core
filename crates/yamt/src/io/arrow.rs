//! Read YAMC Arrow IPC mesh files produced by cad_to_yamc.
//!
//! The Arrow IPC file contains up to 3 record batches sharing a single schema:
//! - Batch 0 "vertices": coordinates as FixedSizeList<3, Float64>
//! - Batch 1 "triangles": indices as FixedSizeList<3, UInt32> + surface_id + physical_group
//! - Batch 2 "tetrahedra" (optional): indices as FixedSizeList<4, UInt32> + volume_id + physical_group
//!
//! Schema metadata:
//! - "yamc.version": format version
//! - "yamc.physical_groups": JSON physical group definitions
//! - "yamc.surface_volumes": JSON surface→volume topology

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float64Array, Int32Array, UInt32Array};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};

use crate::geometry::MeshError;
use crate::mesh::physical_groups::{self, PhysicalGroupData};
use crate::mesh::topology::MeshTopology;
use crate::types::*;

/// Raw data read from an Arrow IPC mesh file.
#[derive(Clone)]
pub struct ArrowMeshData {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[VertexId; 3]>,
    pub triangle_surface_ids: Vec<u32>,
    pub triangle_physical_groups: Vec<i32>,
    pub tetrahedra: Vec<[VertexId; 4]>,
    pub tet_volume_ids: Vec<u32>,
    pub tet_physical_groups: Vec<i32>,
    pub tet_adjacency: Vec<[Option<TetrahedronId>; 4]>,
    pub triangle_aabbs: Vec<[f64; 6]>,
    pub tet_aabbs: Vec<[f64; 6]>,
    pub physical_groups_json: String,
    pub surface_volumes_json: String,
    /// Precomputed volume measures (cm³), one per volume. Empty if not in file.
    pub volume_measures: Vec<f64>,
}

/// Flat mesh data for writing to Arrow IPC files.
///
/// Used by the CAD meshing pipeline to export mesh data.
/// All arrays are flattened (e.g. vertices = [x0,y0,z0, x1,y1,z1, ...]).
#[derive(Clone, Debug)]
pub struct MeshData {
    pub vertices: Vec<f64>,
    pub triangles: Vec<u32>,
    pub triangle_surface_ids: Vec<u32>,
    pub triangle_physical_groups: Vec<i32>,
    pub tetrahedra: Vec<u32>,
    pub tet_volume_ids: Vec<u32>,
    pub tet_physical_groups: Vec<i32>,
    pub tet_adjacency: Vec<i32>,
    pub tri_aabbs: Vec<f64>,
    pub tet_aabbs: Vec<f64>,
    pub physical_groups_json: String,
    pub surface_volumes_json: String,
    /// Optional JSON array of precomputed volume measures (cm³).
    /// Format: `"[1.0, 2.5, ...]"`. Empty string means not provided.
    pub volume_measures_json: String,
}

/// Write mesh data to an Arrow IPC file.
///
/// The file contains up to 3 record batches sharing one schema:
/// - Batch 0 "vertices": coordinates as FixedSizeList<3, Float64>
/// - Batch 1 "triangles": indices + surface_id + physical_group
/// - Batch 2 "tetrahedra" (optional): indices + volume_id + physical_group + adjacency
pub fn write_arrow_mesh(data: &MeshData, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;

    let mut metadata = HashMap::new();
    metadata.insert("yamc.version".to_string(), "1".to_string());
    metadata.insert(
        "yamc.physical_groups".to_string(),
        data.physical_groups_json.clone(),
    );
    metadata.insert(
        "yamc.surface_volumes".to_string(),
        data.surface_volumes_json.clone(),
    );
    if !data.volume_measures_json.is_empty() {
        metadata.insert(
            "yamc.volume_measures".to_string(),
            data.volume_measures_json.clone(),
        );
    }

    let schema = Schema::new_with_metadata(
        vec![
            Field::new(
                "coordinates",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, false)), 3),
                true,
            ),
            Field::new(
                "tri_indices",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt32, false)), 3),
                true,
            ),
            Field::new("tri_surface_id", DataType::UInt32, true),
            Field::new("tri_physical_group", DataType::Int32, true),
            Field::new(
                "tet_indices",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt32, false)), 4),
                true,
            ),
            Field::new("tet_volume_id", DataType::UInt32, true),
            Field::new("tet_physical_group", DataType::Int32, true),
            Field::new(
                "tet_adjacency",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, false)), 4),
                true,
            ),
            Field::new(
                "tri_aabbs",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, false)), 6),
                true,
            ),
            Field::new(
                "tet_aabbs",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, false)), 6),
                true,
            ),
        ],
        metadata,
    );

    let schema_ref = Arc::new(schema);
    let mut writer = FileWriter::try_new(file, &schema_ref)?;

    // Batch 0: Vertices
    let n_verts = data.vertices.len() / 3;
    let coords_values = Float64Array::from(data.vertices.clone());
    let coords_list = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float64, false)),
        3,
        Arc::new(coords_values),
        None,
    )?;
    let vertex_batch = arrow_array::RecordBatch::try_new(
        schema_ref.clone(),
        vec![
            Arc::new(coords_list),
            Arc::new(null_fsl_u32(3, n_verts)),
            Arc::new(UInt32Array::from(vec![None::<u32>; n_verts])),
            Arc::new(Int32Array::from(vec![None::<i32>; n_verts])),
            Arc::new(null_fsl_u32(4, n_verts)),
            Arc::new(UInt32Array::from(vec![None::<u32>; n_verts])),
            Arc::new(Int32Array::from(vec![None::<i32>; n_verts])),
            Arc::new(null_fsl_i32(4, n_verts)),
            Arc::new(null_fsl_f64(6, n_verts)),
            Arc::new(null_fsl_f64(6, n_verts)),
        ],
    )?;
    writer.write(&vertex_batch)?;

    // Batch 1: Triangles
    let n_tris = data.triangles.len() / 3;
    let tri_values = UInt32Array::from(data.triangles.clone());
    let tri_list = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::UInt32, false)),
        3,
        Arc::new(tri_values),
        None,
    )?;
    let tri_aabbs_list = if data.tri_aabbs.is_empty() {
        null_fsl_f64(6, n_tris)
    } else {
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float64, false)),
            6,
            Arc::new(Float64Array::from(data.tri_aabbs.clone())),
            None,
        )?
    };
    let tri_batch = arrow_array::RecordBatch::try_new(
        schema_ref.clone(),
        vec![
            Arc::new(null_fsl_f64(3, n_tris)),
            Arc::new(tri_list),
            Arc::new(UInt32Array::from(data.triangle_surface_ids.clone())),
            Arc::new(Int32Array::from(data.triangle_physical_groups.clone())),
            Arc::new(null_fsl_u32(4, n_tris)),
            Arc::new(UInt32Array::from(vec![None::<u32>; n_tris])),
            Arc::new(Int32Array::from(vec![None::<i32>; n_tris])),
            Arc::new(null_fsl_i32(4, n_tris)),
            Arc::new(tri_aabbs_list),
            Arc::new(null_fsl_f64(6, n_tris)),
        ],
    )?;
    writer.write(&tri_batch)?;

    // Batch 2: Tetrahedra (optional)
    if !data.tetrahedra.is_empty() {
        let n_tets = data.tetrahedra.len() / 4;
        let tet_values = UInt32Array::from(data.tetrahedra.clone());
        let tet_list = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::UInt32, false)),
            4,
            Arc::new(tet_values),
            None,
        )?;
        let tet_adj_list = if data.tet_adjacency.is_empty() {
            null_fsl_i32(4, n_tets)
        } else {
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Int32, false)),
                4,
                Arc::new(Int32Array::from(data.tet_adjacency.clone())),
                None,
            )?
        };
        let tet_aabbs_list = if data.tet_aabbs.is_empty() {
            null_fsl_f64(6, n_tets)
        } else {
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float64, false)),
                6,
                Arc::new(Float64Array::from(data.tet_aabbs.clone())),
                None,
            )?
        };
        let tet_batch = arrow_array::RecordBatch::try_new(
            schema_ref.clone(),
            vec![
                Arc::new(null_fsl_f64(3, n_tets)),
                Arc::new(null_fsl_u32(3, n_tets)),
                Arc::new(UInt32Array::from(vec![None::<u32>; n_tets])),
                Arc::new(Int32Array::from(vec![None::<i32>; n_tets])),
                Arc::new(tet_list),
                Arc::new(UInt32Array::from(data.tet_volume_ids.clone())),
                Arc::new(Int32Array::from(data.tet_physical_groups.clone())),
                Arc::new(tet_adj_list),
                Arc::new(null_fsl_f64(6, n_tets)),
                Arc::new(tet_aabbs_list),
            ],
        )?;
        writer.write(&tet_batch)?;
    }

    writer.finish()?;
    Ok(())
}

fn make_null_buffer(len: usize) -> arrow_buffer::NullBuffer {
    arrow_buffer::NullBuffer::new_null(len)
}

fn null_fsl_i32(size: i32, len: usize) -> FixedSizeListArray {
    let values = Int32Array::from(vec![None::<i32>; len * size as usize]);
    FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Int32, false)),
        size,
        Arc::new(values),
        Some(make_null_buffer(len)),
    )
    .unwrap()
}

fn null_fsl_u32(size: i32, len: usize) -> FixedSizeListArray {
    let values = UInt32Array::from(vec![None::<u32>; len * size as usize]);
    FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::UInt32, false)),
        size,
        Arc::new(values),
        Some(make_null_buffer(len)),
    )
    .unwrap()
}

fn null_fsl_f64(size: i32, len: usize) -> FixedSizeListArray {
    let values = Float64Array::from(vec![None::<f64>; len * size as usize]);
    FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float64, false)),
        size,
        Arc::new(values),
        Some(make_null_buffer(len)),
    )
    .unwrap()
}

/// Read an Arrow IPC mesh file.
pub fn read_arrow_mesh(path: &Path) -> Result<ArrowMeshData, MeshError> {
    let file = File::open(path).map_err(MeshError::Io)?;
    let reader = FileReader::try_new(file, None)
        .map_err(|e| MeshError::Parse(format!("Arrow IPC read error: {e}")))?;

    let schema = reader.schema();
    let metadata = schema.metadata();

    let physical_groups_json = metadata
        .get("yamc.physical_groups")
        .cloned()
        .unwrap_or_else(|| "{}".to_string());
    let surface_volumes_json = metadata
        .get("yamc.surface_volumes")
        .cloned()
        .unwrap_or_else(|| "[]".to_string());
    let volume_measures_json = metadata
        .get("yamc.volume_measures")
        .cloned()
        .unwrap_or_default();

    let mut flat_vertices = Vec::new();
    let mut flat_triangles = Vec::new();
    let mut triangle_surface_ids = Vec::new();
    let mut triangle_physical_groups = Vec::new();
    let mut flat_tetrahedra = Vec::new();
    let mut tet_volume_ids = Vec::new();
    let mut tet_physical_groups = Vec::new();
    let mut flat_tet_adjacency = Vec::new();
    let mut flat_tri_aabbs = Vec::new();
    let mut flat_tet_aabbs = Vec::new();

    for batch_result in reader {
        let batch =
            batch_result.map_err(|e| MeshError::Parse(format!("Arrow batch read error: {e}")))?;

        let coords_col = batch.column(0);
        let tri_col = batch.column(1);
        let tet_col = batch.column(4);

        // Vertex batch: coordinates column has non-null data
        if coords_col.null_count() < coords_col.len() {
            let coords_list = coords_col
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    MeshError::Parse("Expected FixedSizeListArray for coordinates".into())
                })?;
            let values = coords_list
                .values()
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    MeshError::Parse("Expected Float64Array for coordinate values".into())
                })?;
            flat_vertices = values.values().to_vec();
        }

        // Triangle batch: tri_indices column has non-null data
        if tri_col.null_count() < tri_col.len() {
            let tri_list = tri_col
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    MeshError::Parse("Expected FixedSizeListArray for tri_indices".into())
                })?;
            let values = tri_list
                .values()
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| {
                    MeshError::Parse("Expected UInt32Array for triangle values".into())
                })?;
            flat_triangles = values.values().to_vec();

            let surface_ids = batch
                .column(2)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| {
                    MeshError::Parse("Expected UInt32Array for tri_surface_id".into())
                })?;
            triangle_surface_ids = surface_ids.values().to_vec();

            let phys_groups = batch
                .column(3)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    MeshError::Parse("Expected Int32Array for tri_physical_group".into())
                })?;
            triangle_physical_groups = phys_groups.values().to_vec();

            // tri_aabbs (column 8)
            let tri_aabbs_col = batch.column(8);
            if tri_aabbs_col.null_count() < tri_aabbs_col.len() {
                let list = tri_aabbs_col
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| {
                        MeshError::Parse("Expected FixedSizeListArray for tri_aabbs".into())
                    })?;
                let values = list
                    .values()
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        MeshError::Parse("Expected Float64Array for tri_aabbs values".into())
                    })?;
                flat_tri_aabbs = values.values().to_vec();
            }
        }

        // Tet batch: tet_indices column has non-null data
        if tet_col.null_count() < tet_col.len() {
            let tet_list = tet_col
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    MeshError::Parse("Expected FixedSizeListArray for tet_indices".into())
                })?;
            let values = tet_list
                .values()
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| MeshError::Parse("Expected UInt32Array for tet values".into()))?;
            flat_tetrahedra = values.values().to_vec();

            let vol_ids = batch
                .column(5)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| MeshError::Parse("Expected UInt32Array for tet_volume_id".into()))?;
            tet_volume_ids = vol_ids.values().to_vec();

            let phys_groups = batch
                .column(6)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    MeshError::Parse("Expected Int32Array for tet_physical_group".into())
                })?;
            tet_physical_groups = phys_groups.values().to_vec();

            // tet_adjacency (column 7)
            let tet_adj_col = batch.column(7);
            if tet_adj_col.null_count() < tet_adj_col.len() {
                let list = tet_adj_col
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| {
                        MeshError::Parse("Expected FixedSizeListArray for tet_adjacency".into())
                    })?;
                let values = list
                    .values()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| {
                        MeshError::Parse("Expected Int32Array for tet_adjacency values".into())
                    })?;
                flat_tet_adjacency = values.values().to_vec();
            }

            // tet_aabbs (column 9)
            let tet_aabbs_col = batch.column(9);
            if tet_aabbs_col.null_count() < tet_aabbs_col.len() {
                let list = tet_aabbs_col
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| {
                        MeshError::Parse("Expected FixedSizeListArray for tet_aabbs".into())
                    })?;
                let values = list
                    .values()
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        MeshError::Parse("Expected Float64Array for tet_aabbs values".into())
                    })?;
                flat_tet_aabbs = values.values().to_vec();
            }
        }
    }

    // Convert flat arrays to structured arrays
    let vertices: Vec<[f64; 3]> = flat_vertices.as_chunks::<3>().0.to_vec();

    let triangles: Vec<[VertexId; 3]> = flat_triangles.as_chunks::<3>().0.to_vec();

    let tetrahedra: Vec<[VertexId; 4]> = flat_tetrahedra.as_chunks::<4>().0.to_vec();

    let tet_adjacency: Vec<[Option<TetrahedronId>; 4]> = flat_tet_adjacency
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| {
            [
                if c[0] < 0 {
                    None
                } else {
                    Some(c[0] as TetrahedronId)
                },
                if c[1] < 0 {
                    None
                } else {
                    Some(c[1] as TetrahedronId)
                },
                if c[2] < 0 {
                    None
                } else {
                    Some(c[2] as TetrahedronId)
                },
                if c[3] < 0 {
                    None
                } else {
                    Some(c[3] as TetrahedronId)
                },
            ]
        })
        .collect();

    let triangle_aabbs: Vec<[f64; 6]> = flat_tri_aabbs.as_chunks::<6>().0.to_vec();

    let tet_aabbs: Vec<[f64; 6]> = flat_tet_aabbs.as_chunks::<6>().0.to_vec();

    // Parse volume measures from metadata
    let volume_measures = parse_volume_measures_json(&volume_measures_json);

    Ok(ArrowMeshData {
        vertices,
        triangles,
        triangle_surface_ids,
        triangle_physical_groups,
        tetrahedra,
        tet_volume_ids,
        tet_physical_groups,
        tet_adjacency,
        triangle_aabbs,
        tet_aabbs,
        physical_groups_json,
        surface_volumes_json,
        volume_measures,
    })
}

/// Append a vacuum boundary box to Arrow mesh data.
///
/// Creates a bounding box expanded by `offset` around the existing geometry,
/// with inward-pointing normals and vacuum boundary conditions on all 6 faces.
/// Adds 8 corner vertices, 12 triangles (2 per face), and 6 new surfaces.
///
/// # Arguments
/// * `data` -- mutable Arrow mesh data to extend
/// * `offset` -- distance (cm) to expand the bounding box on each side
pub fn add_vacuum_boundary(data: &mut ArrowMeshData, offset: f64) {
    // 1. Compute bounding box from existing vertices
    let mut bb_min = [f64::MAX; 3];
    let mut bb_max = [f64::MIN; 3];
    for v in &data.vertices {
        for i in 0..3 {
            bb_min[i] = bb_min[i].min(v[i]);
            bb_max[i] = bb_max[i].max(v[i]);
        }
    }

    // 2. Expand by offset on each side
    for i in 0..3 {
        bb_min[i] -= offset;
        bb_max[i] += offset;
    }

    // 3. Append 8 corner vertices
    let base_vertex = data.vertices.len() as u32;
    let corners = [
        [bb_min[0], bb_min[1], bb_min[2]], // v0
        [bb_max[0], bb_min[1], bb_min[2]], // v1
        [bb_max[0], bb_max[1], bb_min[2]], // v2
        [bb_min[0], bb_max[1], bb_min[2]], // v3
        [bb_min[0], bb_min[1], bb_max[2]], // v4
        [bb_max[0], bb_min[1], bb_max[2]], // v5
        [bb_max[0], bb_max[1], bb_max[2]], // v6
        [bb_min[0], bb_max[1], bb_max[2]], // v7
    ];
    data.vertices.extend_from_slice(&corners);

    // 4. Append 12 triangles (2 per face, inward-pointing normals)
    let v = |i: u32| base_vertex + i;
    let tris: [[u32; 3]; 12] = [
        // -Z face (normal +z inward)
        [v(0), v(1), v(2)],
        [v(0), v(2), v(3)],
        // +Z face (normal -z inward)
        [v(4), v(7), v(6)],
        [v(4), v(6), v(5)],
        // -X face (normal +x inward)
        [v(0), v(3), v(7)],
        [v(0), v(7), v(4)],
        // +X face (normal -x inward)
        [v(1), v(5), v(6)],
        [v(1), v(6), v(2)],
        // -Y face (normal +y inward)
        [v(0), v(4), v(5)],
        [v(0), v(5), v(1)],
        // +Y face (normal -y inward)
        [v(3), v(2), v(6)],
        [v(3), v(6), v(7)],
    ];
    data.triangles.extend_from_slice(&tris);

    // 5. Assign 6 new 1-indexed surface_ids (2 triangles per face/surface)
    let max_existing_surface = data.triangle_surface_ids.iter().copied().max().unwrap_or(0);
    let first_new_surface = max_existing_surface + 1;
    for face in 0..6u32 {
        let sid = first_new_surface + face;
        data.triangle_surface_ids.push(sid);
        data.triangle_surface_ids.push(sid);
    }

    // 6. Set physical_group to -1 for graveyard triangles
    for _ in 0..12 {
        data.triangle_physical_groups.push(-1);
    }

    // 7. Compute and append AABBs for the 12 new triangles
    for tri in &tris {
        let v0 = &data.vertices[tri[0] as usize];
        let v1 = &data.vertices[tri[1] as usize];
        let v2 = &data.vertices[tri[2] as usize];
        let aabb = [
            v0[0].min(v1[0]).min(v2[0]),
            v0[1].min(v1[1]).min(v2[1]),
            v0[2].min(v1[2]).min(v2[2]),
            v0[0].max(v1[0]).max(v2[0]),
            v0[1].max(v1[1]).max(v2[1]),
            v0[2].max(v1[2]).max(v2[2]),
        ];
        data.triangle_aabbs.push(aabb);
    }

    // 8. Append [null, null] entries to surface_volumes_json for each graveyard surface
    let new_sv_entries = (0..6)
        .map(|_| "[null, null]")
        .collect::<Vec<_>>()
        .join(", ");
    let sv_trimmed = data.surface_volumes_json.trim();
    if sv_trimmed == "[]" || sv_trimmed.is_empty() {
        data.surface_volumes_json = format!("[{new_sv_entries}]");
    } else {
        let without_bracket = sv_trimmed.strip_suffix(']').unwrap_or(sv_trimmed);
        data.surface_volumes_json = format!("{without_bracket}, {new_sv_entries}]");
    }

    // 9. Append a dim=2 physical group with surface_ids to physical_groups_json
    let surface_ids: Vec<u32> = (first_new_surface..first_new_surface + 6).collect();
    let surface_ids_str = surface_ids
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let existing_groups = parse_physical_groups_json(&data.physical_groups_json);
    let max_pg_id = existing_groups
        .iter()
        .map(|(id, _, _, _)| *id)
        .max()
        .unwrap_or(0);
    let new_pg_id = max_pg_id + 1;

    let new_entry = format!(
        "\"{new_pg_id}\": {{\"name\": \"boundary:vacuum\", \"dim\": 2, \"surface_ids\": [{surface_ids_str}]}}"
    );

    let pg_trimmed = data.physical_groups_json.trim();
    if pg_trimmed == "{}" || pg_trimmed.is_empty() {
        data.physical_groups_json = format!("{{{new_entry}}}");
    } else {
        let without_brace = pg_trimmed.strip_suffix('}').unwrap_or(pg_trimmed);
        data.physical_groups_json = format!("{without_brace}, {new_entry}}}");
    }
}

/// Build a MeshTopology from Arrow mesh data.
///
/// Fails when the file breaks the positive-orientation invariant the element
/// walk depends on; see [`crate::mesh::topology::validate_tet_orientation`].
pub fn build_topology(data: ArrowMeshData) -> Result<MeshTopology, MeshError> {
    // Refuse a mesh whose tets are not all positively oriented, before any
    // work is done on it (issue #316).
    crate::mesh::topology::validate_tet_orientation(&data.tetrahedra, &data.vertices)?;

    // Parse surface_volumes from JSON: [[fwd_vol, rev_vol], ...]
    let surface_volumes_raw: Vec<[Option<u32>; 2]> =
        parse_surface_volumes_json(&data.surface_volumes_json);

    // Determine number of surfaces from triangle_surface_ids (1-indexed)
    let num_surfaces = if surface_volumes_raw.is_empty() {
        data.triangle_surface_ids.iter().copied().max().unwrap_or(0)
    } else {
        surface_volumes_raw.len() as u32
    };

    // Determine number of volumes from physical_groups_json (dim=3 groups)
    let num_volumes = count_volumes_from_physical_groups(&data.physical_groups_json);

    // Build surface → triangle ranges
    // surface_ids in Arrow are 1-indexed, convert to 0-based
    let mut surface_tris: Vec<Vec<TriangleId>> = vec![Vec::new(); num_surfaces as usize];
    for (tri_idx, &surf_id) in data.triangle_surface_ids.iter().enumerate() {
        if surf_id > 0 && (surf_id - 1) < num_surfaces {
            surface_tris[(surf_id - 1) as usize].push(tri_idx as TriangleId);
        }
    }

    let mut surface_tri_ranges = Vec::with_capacity(num_surfaces as usize);
    let mut surface_tri_indices = Vec::new();
    for tris in &surface_tris {
        let start = surface_tri_indices.len() as u32;
        surface_tri_indices.extend_from_slice(tris);
        let end = surface_tri_indices.len() as u32;
        surface_tri_ranges.push(start..end);
    }

    // Build volume → tet ranges
    // tet volume_ids from cad_to_yamc use solid_id (1-indexed)
    let mut volume_tets: Vec<Vec<TetrahedronId>> = vec![Vec::new(); num_volumes as usize];
    for (tet_idx, &vol_id) in data.tet_volume_ids.iter().enumerate() {
        let vol_0based = if vol_id > 0 { vol_id - 1 } else { vol_id };
        if (vol_0based as usize) < num_volumes as usize {
            volume_tets[vol_0based as usize].push(tet_idx as TetrahedronId);
        }
    }

    let mut volume_tet_ranges = Vec::with_capacity(num_volumes as usize);
    let mut volume_tet_indices = Vec::new();
    for tets in &volume_tets {
        let start = volume_tet_indices.len() as u32;
        volume_tet_indices.extend_from_slice(tets);
        let end = volume_tet_indices.len() as u32;
        volume_tet_ranges.push(start..end);
    }

    // Build volume ↔ surface relationships
    let implicit_complement = num_volumes;
    let mut volume_surfaces: Vec<Vec<(SurfaceId, Sense)>> = vec![Vec::new(); num_volumes as usize];
    let mut surface_volumes: Vec<(Option<VolumeId>, Option<VolumeId>)> =
        vec![(None, None); num_surfaces as usize];

    for (surf_idx, sv_pair) in surface_volumes_raw.iter().enumerate() {
        if surf_idx >= num_surfaces as usize {
            break;
        }
        let surf_id = surf_idx as SurfaceId;

        // Forward volume (first entry)
        let fwd_vol = sv_pair[0];
        let rev_vol = sv_pair[1];

        if let Some(vol) = fwd_vol {
            if (vol as usize) < num_volumes as usize {
                volume_surfaces[vol as usize].push((surf_id, Sense::Forward));
                surface_volumes[surf_idx].0 = Some(vol);
            }
        }
        if let Some(vol) = rev_vol {
            if (vol as usize) < num_volumes as usize {
                volume_surfaces[vol as usize].push((surf_id, Sense::Reverse));
                surface_volumes[surf_idx].1 = Some(vol);
            }
        }
    }

    // Assign implicit complement to surfaces with missing parent(s).
    // Surfaces with one parent get IC on the other side.
    // Surfaces with no parents (e.g. graveyard) get IC on the forward side.
    for sv in &mut surface_volumes {
        if sv.0.is_none() && sv.1.is_some() {
            sv.0 = Some(implicit_complement);
        } else if sv.0.is_some() && sv.1.is_none() {
            sv.1 = Some(implicit_complement);
        } else if sv.0.is_none() && sv.1.is_none() {
            sv.0 = Some(implicit_complement);
        }
    }

    // Build volume_surfaces entry for the implicit complement so it
    // gets a BVH and can participate in ray_fire queries.
    let mut ic_surfaces = Vec::new();
    for (surf_id, sv) in surface_volumes.iter().enumerate() {
        if sv.0 == Some(implicit_complement) {
            ic_surfaces.push((surf_id as SurfaceId, Sense::Forward));
        }
        if sv.1 == Some(implicit_complement) {
            ic_surfaces.push((surf_id as SurfaceId, Sense::Reverse));
        }
    }
    volume_surfaces.push(ic_surfaces);

    let tet_volume_ids = crate::mesh::topology::build_tet_volume_ids(
        data.tetrahedra.len(),
        &volume_tet_ranges,
        &volume_tet_indices,
    );

    // Use precomputed data from the Arrow file
    let triangle_aabbs = data.triangle_aabbs;
    let tet_aabbs = data.tet_aabbs;
    let surface_aabbs =
        build_group_aabbs(&surface_tri_ranges, &surface_tri_indices, &triangle_aabbs);
    let mut volume_aabbs = build_group_aabbs(&volume_tet_ranges, &volume_tet_indices, &tet_aabbs);

    // For surface-only meshes (no tets), compute volume AABBs from bounding surfaces
    for vol_id in 0..num_volumes as usize {
        if volume_tet_ranges[vol_id].is_empty() {
            let mut aabb = [f64::MAX, f64::MAX, f64::MAX, f64::MIN, f64::MIN, f64::MIN];
            for &(surf_id, _) in &volume_surfaces[vol_id] {
                let range = &surface_tri_ranges[surf_id as usize];
                for i in range.start..range.end {
                    let tri_aabb = &triangle_aabbs[surface_tri_indices[i as usize] as usize];
                    aabb[0] = aabb[0].min(tri_aabb[0]);
                    aabb[1] = aabb[1].min(tri_aabb[1]);
                    aabb[2] = aabb[2].min(tri_aabb[2]);
                    aabb[3] = aabb[3].max(tri_aabb[3]);
                    aabb[4] = aabb[4].max(tri_aabb[4]);
                    aabb[5] = aabb[5].max(tri_aabb[5]);
                }
            }
            volume_aabbs[vol_id] = aabb;
        }
    }
    let global_aabb = compute_global_aabb(&data.vertices);

    // Build physical group data from JSON
    let physical_data =
        build_physical_group_data_from_json(&data.physical_groups_json, num_volumes, num_surfaces);

    Ok(MeshTopology {
        vertices: data.vertices,
        triangles: data.triangles,
        tetrahedra: data.tetrahedra,
        num_volumes,
        volume_tet_ranges,
        volume_tet_indices,
        tet_volume_ids,
        num_surfaces,
        surface_tri_ranges,
        surface_tri_indices,
        volume_surfaces,
        surface_volumes,
        tet_adjacency: data.tet_adjacency,
        triangle_aabbs,
        tet_aabbs,
        surface_aabbs,
        volume_aabbs,
        global_aabb,
        volume_measures: data.volume_measures,
        physical_data,
        implicit_complement,
    })
}

// ---------------------------------------------------------------------------
// JSON parsing helpers
// ---------------------------------------------------------------------------

/// Parse surface_volumes JSON: [[fwd_vol_or_null, rev_vol_or_null], ...]
fn parse_surface_volumes_json(json_str: &str) -> Vec<[Option<u32>; 2]> {
    // Minimal JSON parser for the specific format: [[0, null], [1, 0], ...]
    let trimmed = json_str.trim();
    if trimmed.is_empty() || trimmed == "[]" {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut depth = 0;
    let mut current_pair = String::new();

    for ch in trimmed.chars() {
        match ch {
            '[' => {
                depth += 1;
                if depth >= 2 {
                    current_pair.clear();
                }
            }
            ']' => {
                depth -= 1;
                if depth == 1 {
                    // Parse the pair
                    let pair = parse_pair(&current_pair);
                    result.push(pair);
                    current_pair.clear();
                }
            }
            _ if depth >= 2 => {
                current_pair.push(ch);
            }
            _ => {}
        }
    }

    result
}

fn parse_pair(s: &str) -> [Option<u32>; 2] {
    let parts: Vec<&str> = s.split(',').collect();
    let mut result = [None, None];
    for (i, part) in parts.iter().enumerate().take(2) {
        let trimmed = part.trim();
        if trimmed != "null" && !trimmed.is_empty() {
            if let Ok(v) = trimmed.parse::<u32>() {
                result[i] = Some(v);
            }
        }
    }
    result
}

/// Parse volume measures JSON: `[1.0, 2.5, ...]` → `Vec<f64>`.
fn parse_volume_measures_json(json_str: &str) -> Vec<f64> {
    let trimmed = json_str.trim();
    if trimmed.is_empty() || trimmed == "[]" {
        return Vec::new();
    }
    // Strip outer brackets
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(trimmed);
    inner
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect()
}

/// Count volumes (dim=3) from physical_groups JSON.
/// Format: {"1": {"name": "mat:fuel", "dim": 3}, ...}
fn count_volumes_from_physical_groups(json_str: &str) -> u32 {
    // Simple parse: count entries with "dim": 3 or "dim":3
    let mut count = 0u32;
    // Split by "dim" occurrences
    for part in json_str.split("\"dim\"") {
        if let Some(rest) = part.strip_prefix(':') {
            let rest = rest.trim();
            if rest.starts_with('3') {
                count += 1;
            }
        } else if let Some(rest) = part.strip_prefix(": ") {
            let rest = rest.trim();
            if rest.starts_with('3') {
                count += 1;
            }
        }
    }
    count
}

/// Build PhysicalGroupData from the physical_groups JSON.
fn build_physical_group_data_from_json(
    json_str: &str,
    num_volumes: u32,
    _num_surfaces: u32,
) -> PhysicalGroupData {
    let mut data = PhysicalGroupData::default();

    // Parse physical groups JSON
    // Format: {"1": {"name": "mat:fuel", "dim": 3}, "2": {"name": "mat:moderator", "dim": 3}}
    // We need to find entries with dim=3, extract their names, and map to volume IDs.
    //
    // The physical group IDs correspond to material_pg_ids in cad_to_yamc:
    // pg_id 1 → solid_id 1 → VolumeId 0
    // pg_id 2 → solid_id 2 → VolumeId 1
    // etc.
    //
    // So for dim=3 groups, the n-th one (in order of pg_id) maps to VolumeId n-1.

    let groups = parse_physical_groups_json(json_str);
    let mut volume_idx = 0u32;
    for (_pg_id, name, dim, surface_ids) in &groups {
        if *dim == 3 && volume_idx < num_volumes {
            let info = physical_groups::parse_group_name(name);
            let props = MaterialProps {
                material_name: info.material,
                temperature: info.temperature,
            };
            data.volume_materials.insert(volume_idx, props);
            volume_idx += 1;
        } else if *dim == 2 {
            // Surface-level physical groups -- apply boundary conditions
            let info = physical_groups::parse_group_name(name);
            if let Some(bc) = info.boundary {
                for &sid in surface_ids {
                    if sid > 0 {
                        // surface_ids in Arrow are 1-indexed; surface_bcs uses 0-indexed
                        data.surface_bcs.insert(sid - 1, bc);
                    }
                }
            }
        }
    }

    data
}

/// Parse physical groups JSON into a list of (pg_id, name, dim, surface_ids).
///
/// The `surface_ids` field is optional; entries without it return an empty Vec.
fn parse_physical_groups_json(json_str: &str) -> Vec<(i32, String, i32, Vec<u32>)> {
    let mut result = Vec::new();
    let trimmed = json_str.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return result;
    }

    // Simple stateful parser for the specific JSON format
    // {"1": {"name": "mat:fuel", "dim": 3}, "2": {"name": "boundary:vacuum", "dim": 2, "surface_ids": [7,8,9]}}
    let mut in_entry = false;
    let mut current_key = String::new();
    let mut current_name = String::new();
    let mut current_dim: i32 = -1;
    let mut current_surface_ids: Vec<u32> = Vec::new();
    let mut collecting_key = false;
    let mut collecting_name = false;
    let mut i = 0;
    let chars: Vec<char> = trimmed.chars().collect();

    while i < chars.len() {
        let ch = chars[i];

        if !in_entry {
            // Looking for top-level keys
            if ch == '"' {
                // Start of key
                current_key.clear();
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    current_key.push(chars[i]);
                    i += 1;
                }
                // After the closing quote, expect : {
                collecting_key = true;
            } else if ch == '{' && collecting_key {
                in_entry = true;
                collecting_key = false;
                current_name.clear();
                current_dim = -1;
                current_surface_ids.clear();
            }
        } else {
            // Inside an entry object
            if ch == '}' {
                // End of entry
                if let Ok(pg_id) = current_key.parse::<i32>() {
                    if !current_name.is_empty() && current_dim >= 0 {
                        result.push((
                            pg_id,
                            current_name.clone(),
                            current_dim,
                            current_surface_ids.clone(),
                        ));
                    }
                }
                in_entry = false;
            } else if ch == '"' {
                // Read a field name or value
                let mut field = String::new();
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    field.push(chars[i]);
                    i += 1;
                }
                if field == "name" {
                    collecting_name = true;
                } else if field == "dim" {
                    // Next number is the dimension
                    i += 1;
                    while i < chars.len() && !chars[i].is_ascii_digit() {
                        i += 1;
                    }
                    let mut num_str = String::new();
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        num_str.push(chars[i]);
                        i += 1;
                    }
                    current_dim = num_str.parse().unwrap_or(-1);
                    continue;
                } else if field == "surface_ids" {
                    // Skip to '['
                    i += 1;
                    while i < chars.len() && chars[i] != '[' {
                        i += 1;
                    }
                    i += 1; // skip '['
                            // Read numbers until ']'
                    let mut num_str = String::new();
                    while i < chars.len() && chars[i] != ']' {
                        if chars[i].is_ascii_digit() {
                            num_str.push(chars[i]);
                        } else if (chars[i] == ',' || chars[i] == ' ') && !num_str.is_empty() {
                            if let Ok(n) = num_str.parse::<u32>() {
                                current_surface_ids.push(n);
                            }
                            num_str.clear();
                        }
                        i += 1;
                    }
                    if !num_str.is_empty() {
                        if let Ok(n) = num_str.parse::<u32>() {
                            current_surface_ids.push(n);
                        }
                    }
                    // i now points at ']', will be incremented at end of loop
                    continue;
                } else if collecting_name {
                    current_name = field;
                    collecting_name = false;
                }
            }
        }
        i += 1;
    }

    result.sort_by_key(|(pg_id, _, _, _)| *pg_id);
    result
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn build_group_aabbs(
    ranges: &[std::ops::Range<u32>],
    indices: &[u32],
    element_aabbs: &[[f64; 6]],
) -> Vec<[f64; 6]> {
    ranges
        .iter()
        .map(|range| {
            let mut aabb = [f64::MAX, f64::MAX, f64::MAX, f64::MIN, f64::MIN, f64::MIN];
            for i in range.start..range.end {
                let elem_aabb = &element_aabbs[indices[i as usize] as usize];
                aabb[0] = aabb[0].min(elem_aabb[0]);
                aabb[1] = aabb[1].min(elem_aabb[1]);
                aabb[2] = aabb[2].min(elem_aabb[2]);
                aabb[3] = aabb[3].max(elem_aabb[3]);
                aabb[4] = aabb[4].max(elem_aabb[4]);
                aabb[5] = aabb[5].max(elem_aabb[5]);
            }
            aabb
        })
        .collect()
}

fn compute_global_aabb(vertices: &[[f64; 3]]) -> [f64; 6] {
    let mut aabb = [f64::MAX, f64::MAX, f64::MAX, f64::MIN, f64::MIN, f64::MIN];
    for v in vertices {
        aabb[0] = aabb[0].min(v[0]);
        aabb[1] = aabb[1].min(v[1]);
        aabb[2] = aabb[2].min(v[2]);
        aabb[3] = aabb[3].max(v[0]);
        aabb[4] = aabb[4].max(v[1]);
        aabb[5] = aabb[5].max(v[2]);
    }
    aabb
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::MeshGeometry;

    #[test]
    fn test_parse_surface_volumes_json() {
        let json = "[[0, null], [0, 1], [1, null]]";
        let result = parse_surface_volumes_json(json);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], [Some(0), None]);
        assert_eq!(result[1], [Some(0), Some(1)]);
        assert_eq!(result[2], [Some(1), None]);
    }

    #[test]
    fn test_parse_surface_volumes_empty() {
        let result = parse_surface_volumes_json("[]");
        assert!(result.is_empty());
    }

    #[test]
    fn test_count_volumes() {
        let json =
            r#"{"1": {"name": "mat:fuel", "dim": 3}, "2": {"name": "mat:moderator", "dim": 3}}"#;
        assert_eq!(count_volumes_from_physical_groups(json), 2);
    }

    #[test]
    fn test_count_volumes_mixed() {
        let json =
            r#"{"1": {"name": "mat:fuel", "dim": 3}, "2": {"name": "boundary:vacuum", "dim": 2}}"#;
        assert_eq!(count_volumes_from_physical_groups(json), 1);
    }

    #[test]
    fn test_parse_physical_groups() {
        let json =
            r#"{"1": {"name": "mat:fuel", "dim": 3}, "2": {"name": "mat:moderator", "dim": 3}}"#;
        let groups = parse_physical_groups_json(json);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], (1, "mat:fuel".to_string(), 3, vec![]));
        assert_eq!(groups[1], (2, "mat:moderator".to_string(), 3, vec![]));
    }

    #[test]
    fn test_parse_physical_groups_with_surface_ids() {
        let json = r#"{"1": {"name": "mat:fuel", "dim": 3}, "2": {"name": "boundary:vacuum", "dim": 2, "surface_ids": [7, 8, 9, 10, 11, 12]}}"#;
        let groups = parse_physical_groups_json(json);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], (1, "mat:fuel".to_string(), 3, vec![]));
        assert_eq!(
            groups[1],
            (
                2,
                "boundary:vacuum".to_string(),
                2,
                vec![7, 8, 9, 10, 11, 12]
            )
        );
    }

    #[test]
    fn test_add_vacuum_boundary() {
        // Start with a minimal box: 2 vertices at (0,0,0) and (1,1,1)
        let mut data = ArrowMeshData {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            triangles: vec![],
            triangle_surface_ids: vec![],
            triangle_physical_groups: vec![],
            tetrahedra: vec![],
            tet_volume_ids: vec![],
            tet_physical_groups: vec![],
            tet_adjacency: vec![],
            triangle_aabbs: vec![],
            tet_aabbs: vec![],
            physical_groups_json: r#"{"1": {"name": "mat:water", "dim": 3}}"#.to_string(),
            surface_volumes_json: "[[0, null]]".to_string(),
            volume_measures: vec![],
        };

        add_vacuum_boundary(&mut data, 5.0);

        // Should have 2 + 8 = 10 vertices
        assert_eq!(data.vertices.len(), 10);
        // Should have 12 triangles
        assert_eq!(data.triangles.len(), 12);
        // Should have 12 surface IDs (6 surfaces, 2 tris each)
        assert_eq!(data.triangle_surface_ids.len(), 12);
        // Should have 12 physical groups (all -1)
        assert_eq!(data.triangle_physical_groups.len(), 12);
        assert!(data.triangle_physical_groups.iter().all(|&pg| pg == -1));
        // Should have 12 AABBs
        assert_eq!(data.triangle_aabbs.len(), 12);

        // Bounding box should be [-5, -5, -5] to [6, 6, 6]
        let last_8 = &data.vertices[2..];
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];
        for v in last_8 {
            for i in 0..3 {
                min[i] = min[i].min(v[i]);
                max[i] = max[i].max(v[i]);
            }
        }
        assert!((min[0] - (-5.0)).abs() < 1e-10);
        assert!((max[0] - 6.0).abs() < 1e-10);

        // Physical groups should include the vacuum boundary
        let groups = parse_physical_groups_json(&data.physical_groups_json);
        let vacuum_group = groups
            .iter()
            .find(|(_, name, _, _)| name == "boundary:vacuum");
        assert!(vacuum_group.is_some());
        let (_, _, dim, sids) = vacuum_group.unwrap();
        assert_eq!(*dim, 2);
        assert_eq!(sids.len(), 6);

        // Surface volumes should have 7 entries (1 original + 6 new)
        let sv = parse_surface_volumes_json(&data.surface_volumes_json);
        assert_eq!(sv.len(), 7);
        // Original entry preserved
        assert_eq!(sv[0], [Some(0), None]);
        // New entries are [null, null]
        for entry in &sv[1..] {
            assert_eq!(*entry, [None, None]);
        }
    }

    /// A negatively oriented tet is refused, not silently re-wound: the
    /// element walk reads outward face normals off `TET_FACE_VERTICES` and
    /// would otherwise leave through an entry face (issue #316). Swapping the
    /// last two vertices of one tet of a good fixture flips exactly that sign.
    #[test]
    fn test_negative_tet_is_rejected() {
        let path = std::path::Path::new("tests/data/cube.arrow");
        let mut data = read_arrow_mesh(path).unwrap();
        assert!(data.tetrahedra.len() > 3, "fixture needs at least 4 tets");
        data.tetrahedra[3].swap(2, 3);

        let err = build_topology(data).unwrap_err();
        assert!(
            matches!(
                err,
                MeshError::NegativeTetOrientation { tet_index: 3, signed_volume } if signed_volume < 0.0
            ),
            "expected a negative-orientation error naming tet 3, got {err:?}"
        );

        // The message has to say which tet and point at the fix.
        let msg = err.to_string();
        assert!(msg.contains("tetrahedron 3"), "{msg}");
        assert!(msg.contains("negatively oriented"), "{msg}");
        assert!(msg.contains("#316"), "{msg}");
        assert!(msg.contains("yamm"), "{msg}");
    }

    /// Issue #324: every committed fixture's triangle winding must agree with
    /// the sense recorded in `yamc.surface_volumes`.
    ///
    /// `box.arrow` shipped with all six faces marked forward while three were
    /// wound inward, so the divergence-theorem sum cancelled and the unit cube
    /// measured 0.0 cm3. `calculate_volume` takes the absolute value, which
    /// also hid a whole-mesh sign flip, so neither defect showed up in a test.
    /// The invariant is: taking each volume's bounding triangles and flipping
    /// the ones its sense marks reverse must give a closed, consistently
    /// oriented surface (every directed edge used once, its opposite present)
    /// whose *signed* volume is positive.
    #[test]
    fn fixture_winding_agrees_with_recorded_senses() {
        for name in [
            "box.arrow",
            "cube.arrow",
            "two_region.arrow",
            "two_region_tets.arrow",
            "sphere_in_cube.arrow",
            "nested_cylinders_tets.arrow",
        ] {
            let path = format!("tests/data/{name}");
            let built = build_topology(read_arrow_mesh(std::path::Path::new(&path)).unwrap())
                .unwrap_or_else(|e| panic!("{name} should load: {e}"));
            // Going through MeshGeometry populates volume_measures, so the
            // number the test compares against is the one callers see.
            let geom = crate::geometry::MeshGeometry::from_topology(built);
            let topo = &geom.topology;

            for vol in 0..topo.num_volumes {
                // The volume's boundary, oriented outward by its stored sense.
                let mut oriented: Vec<[u32; 3]> = Vec::new();
                for &(surf_id, sense) in &topo.volume_surfaces[vol as usize] {
                    let range = &topo.surface_tri_ranges[surf_id as usize];
                    for i in range.start..range.end {
                        let t = topo.triangles[topo.surface_tri_indices[i as usize] as usize];
                        oriented.push(match sense {
                            Sense::Forward => t,
                            Sense::Reverse => [t[0], t[2], t[1]],
                        });
                    }
                }
                assert!(
                    !oriented.is_empty(),
                    "{name} volume {vol} has no bounding triangles"
                );

                // Edge matching has to be positional: a surface mesh may carry
                // per-face duplicated vertices, so shared edges of adjacent
                // faces do not share vertex indices.
                let key = |v: [f64; 3]| {
                    let q = |x: f64| (x * 1e9).round() as i64;
                    (q(v[0]), q(v[1]), q(v[2]))
                };
                let mut directed = std::collections::HashMap::new();
                for t in &oriented {
                    let p = t.map(|i| key(topo.vertices[i as usize]));
                    for (a, b) in [(p[0], p[1]), (p[1], p[2]), (p[2], p[0])] {
                        *directed.entry((a, b)).or_insert(0u32) += 1;
                    }
                }
                for (&(a, b), &count) in &directed {
                    assert_eq!(
                        count, 1,
                        "{name} volume {vol}: directed edge {a:?} -> {b:?} used \
                         {count} times, so two triangles are wound against each other"
                    );
                    assert!(
                        directed.contains_key(&(b, a)),
                        "{name} volume {vol}: directed edge {a:?} -> {b:?} has no \
                         opposite, so the bounding surface is not closed"
                    );
                }

                let signed: f64 = oriented
                    .iter()
                    .map(|t| {
                        crate::query::intersect::triangle_volume_contribution(
                            topo.vertices[t[0] as usize],
                            topo.vertices[t[1] as usize],
                            topo.vertices[t[2] as usize],
                        )
                    })
                    .sum::<f64>()
                    / 6.0;
                assert!(
                    signed > 0.0,
                    "{name} volume {vol}: signed volume {signed} is not positive, so \
                     the winding runs inward for the sense recorded in surface_volumes"
                );
                assert!(
                    (signed - topo.volume_measures[vol as usize]).abs() < 1e-9 * signed.max(1.0),
                    "{name} volume {vol}: reported measure {} disagrees with the signed \
                     volume {signed}",
                    topo.volume_measures[vol as usize]
                );
            }
        }
    }

    /// The committed tet fixtures are stored positively oriented, so the check
    /// above is a guard on bad input rather than a trap for our own data.
    #[test]
    fn test_fixture_tets_are_stored_positively_oriented() {
        for name in [
            "cube.arrow",
            "two_region_tets.arrow",
            "nested_cylinders_tets.arrow",
        ] {
            let path = format!("tests/data/{name}");
            let data = read_arrow_mesh(std::path::Path::new(&path)).unwrap();
            assert!(!data.tetrahedra.is_empty(), "{name} has no tets");

            // As stored in the file, before any topology build.
            for (i, t) in data.tetrahedra.iter().enumerate() {
                let signed = crate::query::intersect::signed_tet_volume(
                    data.vertices[t[0] as usize],
                    data.vertices[t[1] as usize],
                    data.vertices[t[2] as usize],
                    data.vertices[t[3] as usize],
                );
                assert!(signed > 0.0, "{name} tet {i} has signed volume {signed}");
            }

            assert!(build_topology(data).is_ok(), "{name} should load");
        }
    }

    #[test]
    fn test_read_box_arrow() {
        let path = std::path::Path::new("tests/data/box.arrow");
        let data = read_arrow_mesh(path).unwrap();

        // A CadQuery box(1,1,1): the surface assembler welds the per-face
        // boundary vertices, so the 6 faces share 8 corners, and 6 faces * 2
        // tris = 12 triangles.
        assert_eq!(data.vertices.len(), 8);
        assert_eq!(data.triangles.len(), 12);
        assert_eq!(data.triangle_surface_ids.len(), 12);
        assert!(data.tetrahedra.is_empty());
    }

    #[test]
    fn test_box_arrow_topology() {
        let path = std::path::Path::new("tests/data/box.arrow");
        let data = read_arrow_mesh(path).unwrap();
        let topo = build_topology(data).unwrap();

        assert_eq!(topo.num_volumes, 1);
        assert_eq!(topo.num_surfaces, 6);
        assert_eq!(topo.triangles.len(), 12);

        // Volume 0 should be "water"
        let mat = &topo.physical_data.volume_materials[&0];
        assert_eq!(mat.material_name.as_deref(), Some("water"));

        // Global AABB should be approximately a unit cube centered at origin
        // CadQuery box(1,1,1) centered at origin: [-0.5, 0.5] in each axis
        assert!((topo.global_aabb[0] - (-0.5)).abs() < 0.01);
        assert!((topo.global_aabb[3] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_box_arrow_geometry() {
        let path = std::path::Path::new("tests/data/box.arrow");
        let geom = MeshGeometry::from_arrow(path).unwrap();

        // Point at center should be in volume 0
        assert!(geom.point_in_volume(0, [0.0, 0.0, 0.0]));
        // Point outside should not be in volume 0
        assert!(!geom.point_in_volume(0, [2.0, 0.0, 0.0]));

        // Ray fire from center
        let result = geom.ray_fire(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], None);
        assert!(result.is_some());
        let (dist, _surf) = result.unwrap();
        assert!((dist - 0.5).abs() < 0.1, "Expected ~0.5, got {dist}");
    }

    #[test]
    fn test_two_region_arrow_topology() {
        let path = std::path::Path::new("tests/data/two_region.arrow");
        let data = read_arrow_mesh(path).unwrap();
        let topo = build_topology(data).unwrap();

        assert_eq!(topo.num_volumes, 2);

        // Check materials
        let mat0 = &topo.physical_data.volume_materials[&0];
        let mat1 = &topo.physical_data.volume_materials[&1];
        assert_eq!(mat0.material_name.as_deref(), Some("fuel"));
        assert_eq!(mat1.material_name.as_deref(), Some("moderator"));
    }

    #[test]
    fn test_two_region_arrow_geometry() {
        let path = std::path::Path::new("tests/data/two_region.arrow");
        let geom = MeshGeometry::from_arrow(path).unwrap();

        assert_eq!(geom.num_volumes(), 2);

        // Left box center should be in volume 0 (fuel)
        assert!(geom.point_in_volume(0, [0.0, 0.0, 0.0]));
        // Right box center should be in volume 1 (moderator)
        assert!(geom.point_in_volume(1, [1.0, 0.0, 0.0]));

        // Material names
        assert_eq!(geom.material_name(0), Some("fuel"));
        assert_eq!(geom.material_name(1), Some("moderator"));
    }

    #[test]
    fn test_box_arrow_with_vacuum_boundary() {
        let path = std::path::Path::new("tests/data/box.arrow");
        let mut data = read_arrow_mesh(path).unwrap();

        // Before: 12 triangles, 6 surfaces
        assert_eq!(data.triangles.len(), 12);
        let orig_tri_count = data.triangles.len();
        let orig_vert_count = data.vertices.len();

        add_vacuum_boundary(&mut data, 5.0);

        // After: 12 + 12 = 24 triangles, 24 + 8 = 32 vertices
        assert_eq!(data.triangles.len(), orig_tri_count + 12);
        assert_eq!(data.vertices.len(), orig_vert_count + 8);

        // Build topology and check vacuum BCs are applied
        let topo = build_topology(data).unwrap();

        // Original 6 surfaces + 6 graveyard surfaces = 12
        assert_eq!(topo.num_surfaces, 12);
        // Original triangles + graveyard triangles
        assert_eq!(topo.triangles.len(), 24);

        // The 6 new surfaces should have vacuum boundary conditions
        for sid in 6..12u32 {
            assert!(
                topo.physical_data.surface_bcs.contains_key(&sid),
                "Surface {sid} should have a BC entry"
            );
            assert_eq!(
                topo.physical_data.surface_bcs[&sid],
                crate::types::BoundaryCondition::Vacuum,
                "Surface {sid} should be vacuum"
            );
        }

        // The geometry should allow ray_fire from inside to hit the vacuum boundary
        let geom = MeshGeometry::from_topology(topo);
        let ic = geom.topology.implicit_complement;

        // Fire ray from origin in +x -- should hit the vacuum boundary
        let hit = geom.ray_fire(ic, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], None);
        assert!(hit.is_some(), "Ray should hit the vacuum boundary");
        let (dist, surf) = hit.unwrap();
        // Box is [-0.5, 0.5], expanded by 5 → vacuum face at x=5.5
        // From inside the original box (x=0.5 face), the IC starts.
        // From origin, the first hit could be the box face (x=0.5) or vacuum (x=5.5)
        assert!(dist > 0.0, "Distance should be positive, got {dist}");

        // Verify boundary condition of the hit surface
        let bc = geom.boundary_condition(surf);
        // The hit is either the box face (transmission) or vacuum boundary
        assert!(
            bc == crate::types::BoundaryCondition::Transmission
                || bc == crate::types::BoundaryCondition::Vacuum
        );
    }
}
