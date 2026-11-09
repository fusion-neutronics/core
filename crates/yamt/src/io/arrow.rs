//! Read YAMC Arrow IPC mesh files produced by cad_to_yamc.
//!
//! The Arrow IPC file holds one record batch of one row. Each column is a whole
//! table, stored as a list of its rows, so the tables keep their own lengths and
//! no column is padded:
//! - "vertices": List<FixedSizeList<3, Float64>>
//! - "triangles": List<FixedSizeList<3, UInt32>>, with "triangle_surface_ids"
//!   (List<UInt32>, 1-based) and "triangle_physical_groups" (List<Int32>)
//! - "tetrahedra": List<FixedSizeList<4, UInt32>>, with "tet_volume_ids"
//!   (List<UInt32>), "tet_physical_groups" (List<Int32>) and "tet_adjacency"
//!   (List<FixedSizeList<4, Int32>>, -1 for a face on no other tet); empty for
//!   a surface-only mesh
//!
//! Bounding boxes are not stored: the reader computes them from the vertices.
//!
//! Schema metadata:
//! - "yamc.version": format version, "2" (version 1 padded every table to a
//!   shared schema with null columns, and is refused)
//! - "yamc.physical_groups": JSON physical group definitions
//! - "yamc.surface_volumes": JSON surface→volume topology
//! - "yamc.imprinted" (optional): "false" when every solid was meshed on its
//!   own, without imprinting, so touching solids do not share surfaces
//! - "yamc.contact_tolerance" (optional): for such a mesh, the distance (cm)
//!   within which two solids' surfaces count as touching
//! - "yamc.overlaps" (optional): for such a mesh, `[[low, high], ...]` pairs of
//!   0-based volume ids whose solids overlap, where `high` takes the shared
//!   region
//! - "yamc.separated" (optional): for such a mesh, `[[a, b], ...]` pairs of
//!   0-based volume ids whose meshes come within the contact tolerance although
//!   the CAD solids do not touch, so the gap between them is never filled

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::types::{Float64Type, Int32Type, UInt32Type};
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, FixedSizeListArray, ListArray, PrimitiveArray, RecordBatch,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{ArrowError, Field, Schema};

use crate::geometry::MeshError;
use crate::mesh::physical_groups::{self, PhysicalGroupData};
use crate::mesh::topology::{build_tet_aabbs, build_triangle_aabbs, MeshTopology};
use crate::types::*;

/// Raw data read from an Arrow IPC mesh file.
#[derive(Clone, Default)]
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
    /// `Some(false)` for a mesh whose solids were meshed independently
    /// (not imprinted). `None` when the file does not say.
    pub imprinted: Option<bool>,
    /// Contact tolerance (cm) recorded for a non-imprinted mesh.
    pub contact_tolerance: Option<f64>,
    /// Overlapping solid pairs `[low, high]` (0-based volume ids) of a
    /// non-imprinted mesh: `high` takes the region the two share.
    pub overlaps: Vec<[u32; 2]>,
    /// Solid pairs (0-based volume ids) of a non-imprinted mesh that are
    /// within the contact tolerance in the mesh but do not touch in the CAD.
    pub separated: Vec<[u32; 2]>,
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
    pub physical_groups_json: String,
    pub surface_volumes_json: String,
    /// Optional JSON array of precomputed volume measures (cm³).
    /// Format: `"[1.0, 2.5, ...]"`. Empty string means not provided.
    pub volume_measures_json: String,
    /// Written as `yamc.imprinted` when set.
    pub imprinted: Option<bool>,
    /// Written as `yamc.contact_tolerance` when set.
    pub contact_tolerance: Option<f64>,
    /// Written as `yamc.overlaps` when not empty.
    pub overlaps: Vec<[u32; 2]>,
    /// Written as `yamc.separated` when not empty.
    pub separated: Vec<[u32; 2]>,
}

/// Version of the mesh file layout, written as `yamc.version`. Version 1 kept
/// every table in batches of one shared schema, padded with null columns.
const FORMAT_VERSION: &str = "2";

/// The mesh tables, as `(column, element width)`. Each column holds one row:
/// the whole table as a list of fixed-size lists (or of scalars, width 1).
const TRIANGLE_COLUMNS: [(&str, i32); 3] = [
    ("triangles", 3),
    ("triangle_surface_ids", 1),
    ("triangle_physical_groups", 1),
];
const TET_COLUMNS: [(&str, i32); 4] = [
    ("tetrahedra", 4),
    ("tet_volume_ids", 1),
    ("tet_physical_groups", 1),
    ("tet_adjacency", 4),
];

/// One table column: `values` as a single list row of `width`-sized items.
fn table_column<T: ArrowPrimitiveType>(
    values: Vec<T::Native>,
    width: i32,
) -> Result<ListArray, ArrowError> {
    let n = values.len() / width as usize;
    let mut item: ArrayRef = Arc::new(PrimitiveArray::<T>::new(ScalarBuffer::from(values), None));
    if width > 1 {
        let field = Arc::new(Field::new("item", T::DATA_TYPE, false));
        item = Arc::new(FixedSizeListArray::try_new(field, width, item, None)?);
    }
    let field = Arc::new(Field::new("item", item.data_type().clone(), false));
    ListArray::try_new(field, OffsetBuffer::from_lengths([n]), item, None)
}

/// Write mesh data to an Arrow IPC file.
///
/// The file holds one record batch of one row. Each column is a whole table
/// (vertices, triangles and their ids, tetrahedra and their ids), so no column
/// carries another table's rows. Bounding boxes are not stored: they follow
/// from the vertices, and [`read_arrow_mesh`] computes them.
pub fn write_arrow_mesh(data: MeshData, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut metadata = HashMap::from([
        ("yamc.version".to_string(), FORMAT_VERSION.to_string()),
        (
            "yamc.physical_groups".to_string(),
            data.physical_groups_json,
        ),
        (
            "yamc.surface_volumes".to_string(),
            data.surface_volumes_json,
        ),
    ]);
    if !data.volume_measures_json.is_empty() {
        metadata.insert("yamc.volume_measures".into(), data.volume_measures_json);
    }
    if let Some(imprinted) = data.imprinted {
        metadata.insert("yamc.imprinted".into(), imprinted.to_string());
    }
    if let Some(tol) = data.contact_tolerance {
        metadata.insert("yamc.contact_tolerance".into(), format!("{tol:e}"));
    }
    for (key, pairs) in [
        ("yamc.overlaps", &data.overlaps),
        ("yamc.separated", &data.separated),
    ] {
        if !pairs.is_empty() {
            let pairs: Vec<String> = pairs.iter().map(|[a, b]| format!("[{a}, {b}]")).collect();
            metadata.insert(key.into(), format!("[{}]", pairs.join(", ")));
        }
    }

    let columns: Vec<(&str, ListArray)> = vec![
        ("vertices", table_column::<Float64Type>(data.vertices, 3)?),
        (
            TRIANGLE_COLUMNS[0].0,
            table_column::<UInt32Type>(data.triangles, 3)?,
        ),
        (
            TRIANGLE_COLUMNS[1].0,
            table_column::<UInt32Type>(data.triangle_surface_ids, 1)?,
        ),
        (
            TRIANGLE_COLUMNS[2].0,
            table_column::<Int32Type>(data.triangle_physical_groups, 1)?,
        ),
        (
            TET_COLUMNS[0].0,
            table_column::<UInt32Type>(data.tetrahedra, 4)?,
        ),
        (
            TET_COLUMNS[1].0,
            table_column::<UInt32Type>(data.tet_volume_ids, 1)?,
        ),
        (
            TET_COLUMNS[2].0,
            table_column::<Int32Type>(data.tet_physical_groups, 1)?,
        ),
        (
            TET_COLUMNS[3].0,
            table_column::<Int32Type>(data.tet_adjacency, 4)?,
        ),
    ];
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, col)| Field::new(*name, col.data_type().clone(), false))
        .collect();
    let schema = Arc::new(Schema::new_with_metadata(fields, metadata));
    let arrays: Vec<ArrayRef> = columns
        .into_iter()
        .map(|(_, c)| Arc::new(c) as ArrayRef)
        .collect();
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;

    let mut writer = FileWriter::try_new(File::create(path)?, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(())
}

/// Read back one table column written by [`table_column`], flattened.
fn read_column<T: ArrowPrimitiveType>(
    batch: &RecordBatch,
    name: &str,
    width: i32,
) -> Result<Vec<T::Native>, MeshError> {
    let bad = |what: &str| MeshError::Parse(format!("mesh file column {name:?}: {what}"));
    let list = batch
        .column_by_name(name)
        .ok_or_else(|| bad("missing"))?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| bad("expected a list"))?;
    let item = list.value(0);
    let values = if width == 1 {
        item
    } else {
        let fsl = item
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .filter(|f| f.value_length() == width)
            .ok_or_else(|| bad(&format!("expected lists of {width}")))?;
        fsl.values().clone()
    };
    let values = values
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| bad(&format!("expected {} values", T::DATA_TYPE)))?;
    if values.null_count() > 0 {
        return Err(bad("holds nulls"));
    }
    Ok(values.values().to_vec())
}

/// Turn `-1`-for-none adjacency into `Option`s.
fn adjacency(flat: &[i32]) -> Vec<[Option<TetrahedronId>; 4]> {
    let id = |i: i32| (i >= 0).then_some(i as TetrahedronId);
    flat.as_chunks::<4>().0.iter().map(|c| c.map(id)).collect()
}

/// Read an Arrow IPC mesh file written by [`write_arrow_mesh`].
pub fn read_arrow_mesh(path: &Path) -> Result<ArrowMeshData, MeshError> {
    let file = File::open(path).map_err(MeshError::Io)?;
    let reader = FileReader::try_new(file, None)
        .map_err(|e| MeshError::Parse(format!("Arrow IPC read error: {e}")))?;
    let schema = reader.schema();
    let metadata = schema.metadata();

    match metadata.get("yamc.version").map(|v| v.trim()) {
        Some(FORMAT_VERSION) => {}
        found => {
            return Err(MeshError::Parse(format!(
                "{} is a version {} yamc mesh file and this yamc reads version {FORMAT_VERSION}. \
                 Write it again with CadToYamc.to_arrow.",
                path.display(),
                found.unwrap_or("unknown")
            )));
        }
    }
    let (imprinted, contact_tolerance, overlaps, separated) = read_contact_metadata(metadata)?;
    let get =
        |key: &str, default: &str| metadata.get(key).cloned().unwrap_or_else(|| default.into());
    let physical_groups_json = get("yamc.physical_groups", "{}");
    let surface_volumes_json = get("yamc.surface_volumes", "[]");
    let volume_measures = parse_volume_measures_json(&get("yamc.volume_measures", ""));

    let mut batches = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| MeshError::Parse(format!("Arrow batch read error: {e}")))?;
    let batch = match (batches.pop(), batches.is_empty()) {
        (Some(batch), true) if batch.num_rows() == 1 => batch,
        _ => {
            return Err(MeshError::Parse(
                "a yamc mesh file holds one record batch of one row".into(),
            ));
        }
    };

    let vertices = read_column::<Float64Type>(&batch, "vertices", 3)?;
    let vertices: Vec<[f64; 3]> = vertices.as_chunks::<3>().0.to_vec();
    let [tris, tri_sids, tri_pgs] = TRIANGLE_COLUMNS;
    let triangles: Vec<[VertexId; 3]> = read_column::<UInt32Type>(&batch, tris.0, tris.1)?
        .as_chunks::<3>()
        .0
        .to_vec();
    let triangle_surface_ids = read_column::<UInt32Type>(&batch, tri_sids.0, tri_sids.1)?;
    let triangle_physical_groups = read_column::<Int32Type>(&batch, tri_pgs.0, tri_pgs.1)?;
    let [tets, tet_vids, tet_pgs, tet_adj] = TET_COLUMNS;
    let tetrahedra: Vec<[VertexId; 4]> = read_column::<UInt32Type>(&batch, tets.0, tets.1)?
        .as_chunks::<4>()
        .0
        .to_vec();
    let tet_volume_ids = read_column::<UInt32Type>(&batch, tet_vids.0, tet_vids.1)?;
    let tet_physical_groups = read_column::<Int32Type>(&batch, tet_pgs.0, tet_pgs.1)?;
    let tet_adjacency = adjacency(&read_column::<Int32Type>(&batch, tet_adj.0, tet_adj.1)?);

    let n_tris = triangles.len();
    let n_tets = tetrahedra.len();
    let lengths = [
        ("triangle_surface_ids", triangle_surface_ids.len(), n_tris),
        (
            "triangle_physical_groups",
            triangle_physical_groups.len(),
            n_tris,
        ),
        ("tet_volume_ids", tet_volume_ids.len(), n_tets),
        ("tet_physical_groups", tet_physical_groups.len(), n_tets),
        ("tet_adjacency", tet_adjacency.len(), n_tets),
    ];
    for (name, len, want) in lengths {
        if len != want {
            return Err(MeshError::Parse(format!(
                "{name} has {len} rows, expected {want}"
            )));
        }
    }
    let n_verts = vertices.len();
    let max_vertex = triangles
        .iter()
        .flatten()
        .chain(tetrahedra.iter().flatten())
        .max();
    if let Some(&v) = max_vertex.filter(|&&v| v as usize >= n_verts) {
        return Err(MeshError::Parse(format!(
            "an element uses vertex {v}, but the mesh has {n_verts} vertices"
        )));
    }

    Ok(ArrowMeshData {
        triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
        tet_aabbs: build_tet_aabbs(&tetrahedra, &vertices),
        vertices,
        triangles,
        triangle_surface_ids,
        triangle_physical_groups,
        tetrahedra,
        tet_volume_ids,
        tet_physical_groups,
        tet_adjacency,
        physical_groups_json,
        surface_volumes_json,
        volume_measures,
        imprinted,
        contact_tolerance,
        overlaps,
        separated,
    })
}

type ContactMetadata = (Option<bool>, Option<f64>, Vec<[u32; 2]>, Vec<[u32; 2]>);

/// The non-imprinted mesh entries: `yamc.imprinted`, `yamc.contact_tolerance`,
/// `yamc.overlaps` and `yamc.separated`.
fn read_contact_metadata(metadata: &arrow_schema::Metadata) -> Result<ContactMetadata, MeshError> {
    let imprinted = match metadata.get("yamc.imprinted").map(|v| v.trim()) {
        None => None,
        Some("true") => Some(true),
        Some("false") => Some(false),
        Some(other) => {
            return Err(MeshError::Parse(format!(
                "yamc.imprinted must be \"true\" or \"false\", got {other:?}"
            )));
        }
    };
    let contact_tolerance = match metadata.get("yamc.contact_tolerance") {
        None => None,
        Some(v) => match v.trim().parse::<f64>() {
            Ok(t) if t.is_finite() && t >= 0.0 => Some(t),
            _ => {
                return Err(MeshError::Parse(format!(
                    "yamc.contact_tolerance must be a non-negative number of cm, got {v:?}"
                )));
            }
        },
    };
    let pairs = |key: &str, shape: &str| match metadata.get(key) {
        None => Ok(Vec::new()),
        Some(json) => parse_surface_volumes_json(json)
            .into_iter()
            .map(|pair| match pair {
                [Some(a), Some(b)] if a != b => Ok([a, b]),
                _ => Err(MeshError::Parse(format!(
                    "{key} must be {shape} pairs of distinct volume ids, got {json:?}"
                ))),
            })
            .collect::<Result<Vec<_>, _>>(),
    };
    let overlaps = pairs("yamc.overlaps", "[[low, high], ...]")?;
    let separated = pairs("yamc.separated", "[[a, b], ...]")?;
    Ok((imprinted, contact_tolerance, overlaps, separated))
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
/// walk depends on (see [`crate::mesh::topology::validate_tet_orientation`]),
/// or when a volume's boundary lists the same facet twice (see
/// [`crate::mesh::topology::validate_no_duplicate_facets`]).
pub fn build_topology(data: ArrowMeshData) -> Result<MeshTopology, MeshError> {
    // Refuse a mesh whose tets are not all positively oriented, before any
    // work is done on it.
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

        if let Some(vol) = fwd_vol
            && (vol as usize) < num_volumes as usize
        {
            volume_surfaces[vol as usize].push((surf_id, Sense::Forward));
            surface_volumes[surf_idx].0 = Some(vol);
        }
        if let Some(vol) = rev_vol
            && (vol as usize) < num_volumes as usize
        {
            volume_surfaces[vol as usize].push((surf_id, Sense::Reverse));
            surface_volumes[surf_idx].1 = Some(vol);
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
    let overlaps = overlap_pairs(&data.overlaps, num_volumes)?;

    // Build physical group data from JSON
    let physical_data =
        build_physical_group_data_from_json(&data.physical_groups_json, num_volumes, num_surfaces);

    let topology = MeshTopology {
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
        imprinted: data.imprinted,
        contact_tolerance: data.contact_tolerance,
        overlaps,
        separated: data
            .separated
            .iter()
            .filter(|&&[a, b]| a < num_volumes && b < num_volumes)
            .map(|&[a, b]| (a.min(b), a.max(b)))
            .collect(),
    };
    crate::mesh::topology::validate_no_duplicate_facets(&topology)?;
    Ok(topology)
}

/// Check the recorded overlap pairs: ids in range, and the "takes the shared
/// region from" relation free of cycles, so the solids can be ranked.
fn overlap_pairs(
    pairs: &[[u32; 2]],
    num_volumes: u32,
) -> Result<Vec<(VolumeId, VolumeId)>, MeshError> {
    for &[low, high] in pairs {
        if low >= num_volumes || high >= num_volumes {
            return Err(MeshError::Parse(format!(
                "yamc.overlaps names volume {} but the mesh has {num_volumes} volumes",
                low.max(high)
            )));
        }
    }
    let pairs: Vec<(VolumeId, VolumeId)> = pairs.iter().map(|&[l, h]| (l, h)).collect();
    if crate::mesh::topology::overlap_ranks(num_volumes, &pairs).is_none() {
        return Err(MeshError::Parse(
            "yamc.overlaps is cyclic: some solid both takes and gives up the same \
             shared region, so there is no order to resolve overlaps in"
                .to_string(),
        ));
    }
    Ok(pairs)
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
        if trimmed != "null"
            && !trimmed.is_empty()
            && let Ok(v) = trimmed.parse::<u32>()
        {
            result[i] = Some(v);
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
                if let Ok(pg_id) = current_key.parse::<i32>()
                    && !current_name.is_empty()
                    && current_dim >= 0
                {
                    result.push((
                        pg_id,
                        current_name.clone(),
                        current_dim,
                        current_surface_ids.clone(),
                    ));
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
                    if !num_str.is_empty()
                        && let Ok(n) = num_str.parse::<u32>()
                    {
                        current_surface_ids.push(n);
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
            imprinted: None,
            contact_tolerance: None,
            overlaps: vec![],
            separated: vec![],
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
    /// would otherwise leave through an entry face. Swapping the
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
        assert!(msg.contains("regenerate it"), "{msg}");
        assert!(msg.contains("yamm"), "{msg}");
    }

    /// Append a copy of triangle `tri` to `data` on surface id `surface_id`
    /// (1-based, as in the file), reversed when `reverse` is set.
    fn push_triangle_copy(data: &mut ArrowMeshData, tri: usize, surface_id: u32, reverse: bool) {
        let mut t = data.triangles[tri];
        if reverse {
            t.swap(1, 2);
        }
        data.triangles.push(t);
        data.triangle_surface_ids.push(surface_id);
        data.triangle_physical_groups
            .push(data.triangle_physical_groups[tri]);
        data.triangle_aabbs.push(data.triangle_aabbs[tri]);
    }

    /// Issue #515: a volume whose boundary lists a facet twice is refused,
    /// whether the copy faces the same way (a copy) or the opposite way (a
    /// contradiction), and the message names the triangles.
    #[test]
    fn test_duplicate_facet_is_rejected() {
        let path = std::path::Path::new("tests/data/cube.arrow");
        for reverse in [false, true] {
            let mut data = read_arrow_mesh(path).unwrap();
            let n = data.triangles.len();
            let surface_id = data.triangle_surface_ids[0];
            push_triangle_copy(&mut data, 0, surface_id, reverse);

            let err = build_topology(data).unwrap_err();
            match err {
                MeshError::DuplicateFacet {
                    volume: 0,
                    triangles,
                    same_facing,
                    ..
                } => {
                    assert_eq!(triangles, [0, n as u32]);
                    assert_eq!(same_facing, !reverse);
                }
                ref other => panic!("expected a duplicate-facet error, got {other:?}"),
            }
            let msg = err.to_string();
            assert!(msg.contains(&format!("triangles 0 and {n}")), "{msg}");
        }
    }

    /// A copy whose vertices are separate rows at the same positions (as an
    /// STL soup writes them) is the same facet.
    #[test]
    fn test_duplicate_facet_with_unwelded_vertices_is_rejected() {
        let path = std::path::Path::new("tests/data/cube.arrow");
        let mut data = read_arrow_mesh(path).unwrap();
        let t = data.triangles[0];
        let first_new = data.vertices.len() as VertexId;
        for v in t {
            data.vertices.push(data.vertices[v as usize]);
        }
        let surface_id = data.triangle_surface_ids[0];
        push_triangle_copy(&mut data, 0, surface_id, false);
        *data.triangles.last_mut().unwrap() = [first_new, first_new + 1, first_new + 2];
        assert!(matches!(
            build_topology(data),
            Err(MeshError::DuplicateFacet {
                same_facing: true,
                ..
            })
        ));
    }

    /// On a surface with the volume on both sides (an internal face), one
    /// triangle is reached twice (once per sense) and is not a copy, and an
    /// opposite-wound pair is the face listed from each side. A same-wound
    /// pair there is still a copy.
    #[test]
    fn test_internal_face_is_not_a_duplicate() {
        let path = std::path::Path::new("tests/data/cube.arrow");
        let base = read_arrow_mesh(path).unwrap();
        let mut sv = parse_surface_volumes_json(&base.surface_volumes_json);
        assert!(sv.iter().all(|e| e[0] == Some(0) || e[1] == Some(0)));
        // A small triangle well inside the cube, on a new [V, V] surface.
        let (lo, hi) = base
            .vertices
            .iter()
            .fold(([f64::MAX; 3], [f64::MIN; 3]), |(lo, hi), p| {
                (
                    [lo[0].min(p[0]), lo[1].min(p[1]), lo[2].min(p[2])],
                    [hi[0].max(p[0]), hi[1].max(p[1]), hi[2].max(p[2])],
                )
            });
        let at = |f: [f64; 3]| -> [f64; 3] {
            [
                lo[0] + f[0] * (hi[0] - lo[0]),
                lo[1] + f[1] * (hi[1] - lo[1]),
                lo[2] + f[2] * (hi[2] - lo[2]),
            ]
        };
        sv.push([Some(0), Some(0)]);
        let surface_id = sv.len() as u32;
        let json = |sv: &[[Option<u32>; 2]]| -> String {
            let cell = |v: Option<u32>| v.map_or("null".to_string(), |v| v.to_string());
            let rows: Vec<String> = sv
                .iter()
                .map(|e| format!("[{}, {}]", cell(e[0]), cell(e[1])))
                .collect();
            format!("[{}]", rows.join(", "))
        };
        let with_internal_face = |copies: &[bool]| -> ArrowMeshData {
            let mut data = base.clone();
            data.surface_volumes_json = json(&sv);
            let v0 = data.vertices.len() as VertexId;
            data.vertices.push(at([0.4, 0.4, 0.5]));
            data.vertices.push(at([0.6, 0.4, 0.5]));
            data.vertices.push(at([0.5, 0.6, 0.5]));
            let face = data.triangles.len();
            push_triangle_copy(&mut data, 0, surface_id, false);
            data.triangles[face] = [v0, v0 + 1, v0 + 2];
            let p = &data.vertices[v0 as usize..];
            data.triangle_aabbs[face] = [
                p[0][0].min(p[1][0]).min(p[2][0]),
                p[0][1].min(p[1][1]).min(p[2][1]),
                p[0][2].min(p[1][2]).min(p[2][2]),
                p[0][0].max(p[1][0]).max(p[2][0]),
                p[0][1].max(p[1][1]).max(p[2][1]),
                p[0][2].max(p[1][2]).max(p[2][2]),
            ];
            for &reverse in copies {
                push_triangle_copy(&mut data, face, surface_id, reverse);
            }
            data
        };

        build_topology(with_internal_face(&[])).unwrap();
        build_topology(with_internal_face(&[true])).unwrap();
        assert!(matches!(
            build_topology(with_internal_face(&[false])),
            Err(MeshError::DuplicateFacet {
                same_facing: true,
                ..
            })
        ));
    }

    /// Every committed fixture's triangle winding must agree with
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
        assert!(geom.point_in_volume(0, [0.25, 0.5, 0.5]));
        // Right box center should be in volume 1 (moderator)
        assert!(geom.point_in_volume(1, [0.75, 0.5, 0.5]));

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

    /// A strip of `n` triangles over `n + 2` vertices, and one tet.
    fn strip(n: u32) -> MeshData {
        let vertices = (0..n + 2)
            .flat_map(|i| [i as f64, (i % 2) as f64, 0.5])
            .collect();
        let triangles = (0..n).flat_map(|i| [i, i + 1, i + 2]).collect();
        MeshData {
            vertices,
            triangles,
            triangle_surface_ids: vec![1; n as usize],
            triangle_physical_groups: vec![-1; n as usize],
            tetrahedra: vec![0, 1, 2, 3],
            tet_volume_ids: vec![1],
            tet_physical_groups: vec![1],
            tet_adjacency: vec![-1; 4],
            physical_groups_json: r#"{"1": [3, "mat:steel", [1]]}"#.into(),
            surface_volumes_json: "[[1, null]]".into(),
            volume_measures_json: "[1.5]".into(),
            imprinted: Some(false),
            contact_tolerance: Some(1e-3),
            overlaps: vec![[0, 1]],
            separated: vec![[1, 2]],
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("yamt-{}-{name}.arrow", std::process::id()))
    }

    #[test]
    fn test_write_read_round_trip() {
        let path = temp_path("round-trip");
        let data = strip(5);
        write_arrow_mesh(data.clone(), &path).unwrap();
        let read = read_arrow_mesh(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let flat = |v: &[[f64; 3]]| v.iter().flatten().copied().collect::<Vec<_>>();
        assert_eq!(flat(&read.vertices), data.vertices);
        assert_eq!(read.triangles.concat(), data.triangles);
        assert_eq!(read.triangle_surface_ids, data.triangle_surface_ids);
        assert_eq!(read.triangle_physical_groups, data.triangle_physical_groups);
        assert_eq!(read.tetrahedra.concat(), data.tetrahedra);
        assert_eq!(read.tet_volume_ids, data.tet_volume_ids);
        assert_eq!(read.tet_physical_groups, data.tet_physical_groups);
        assert_eq!(read.tet_adjacency, vec![[None; 4]]);
        assert_eq!(read.physical_groups_json, data.physical_groups_json);
        assert_eq!(read.surface_volumes_json, data.surface_volumes_json);
        assert_eq!(read.volume_measures, vec![1.5]);
        assert_eq!(
            (
                read.imprinted,
                read.contact_tolerance,
                read.overlaps,
                read.separated
            ),
            (Some(false), Some(1e-3), vec![[0, 1]], vec![[1, 2]])
        );
        assert_eq!(read.triangle_aabbs[0], [0.0, 0.0, 0.5, 2.0, 1.0, 0.5]);
        assert_eq!(read.tet_aabbs, vec![[0.0, 0.0, 0.5, 3.0, 1.0, 0.5]]);
    }

    #[test]
    fn test_surface_only_mesh_round_trips() {
        let path = temp_path("surface-only");
        let data = MeshData {
            tetrahedra: vec![],
            tet_volume_ids: vec![],
            tet_physical_groups: vec![],
            tet_adjacency: vec![],
            ..strip(3)
        };
        write_arrow_mesh(data, &path).unwrap();
        let read = read_arrow_mesh(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(read.triangles.len(), 3);
        assert!(read.tetrahedra.is_empty() && read.tet_aabbs.is_empty());
    }

    /// No column carries another table's rows: the file is the data plus a
    /// fixed overhead (#504 measured 184 B per row in the old layout).
    #[test]
    fn test_file_holds_only_the_data() {
        let path = temp_path("size");
        let n = 100_000u32;
        write_arrow_mesh(strip(n), &path).unwrap();
        let size = std::fs::metadata(&path).unwrap().len();
        std::fs::remove_file(&path).unwrap();
        // 24 B per vertex; 12 B indices, 4 B surface id, 4 B group per triangle.
        let data = (n as u64 + 2) * 24 + n as u64 * 20;
        // arrow-rs writes a validity bitmap even for columns with no nulls, one
        // bit per value: at most 1/32 of 4-byte values.
        assert!(
            size < data + data / 32 + 4096,
            "{size} B for {data} B of mesh data"
        );
    }

    #[test]
    fn test_version_1_file_is_refused() {
        let path = temp_path("version-1");
        let metadata = HashMap::from([("yamc.version".to_string(), "1".to_string())]);
        let schema = Arc::new(Schema::new_with_metadata(Vec::<Field>::new(), metadata));
        let mut writer = FileWriter::try_new(File::create(&path).unwrap(), &schema).unwrap();
        writer.finish().unwrap();
        let err = read_arrow_mesh(&path).err().unwrap().to_string();
        std::fs::remove_file(&path).unwrap();
        assert!(
            err.contains("version 1") && err.contains("to_arrow"),
            "{err}"
        );
    }

    #[test]
    fn test_out_of_range_vertex_is_refused() {
        let path = temp_path("bad-vertex");
        let mut data = strip(2);
        data.triangles[5] = 99;
        write_arrow_mesh(data, &path).unwrap();
        let err = read_arrow_mesh(&path).err().unwrap().to_string();
        std::fs::remove_file(&path).unwrap();
        assert!(
            err.contains("vertex 99") && err.contains("4 vertices"),
            "{err}"
        );
    }
}
