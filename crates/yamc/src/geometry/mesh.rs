//! Mesh-based geometry backend using `yamt`.
//!
//! Wraps a `yamt::MeshGeometry` to provide the same interface that the
//! transport loop expects. Each mesh volume maps to a `Cell` (with
//! material from physical group names). Ray-surface queries use the
//! mesh BVH instead of CSG evaluation.

use std::collections::HashMap;
use std::sync::Arc;

use crate::geo::BoundingBox;
use crate::geo::{BoundaryType, Surface, SurfaceKind};
use crate::geo::{HalfspaceType, Region, RegionExpr};
use crate::geometry::backend::BoundaryHit;
use crate::geometry::cell::Cell;
use yamc_materials::material::Material;

/// Mesh-based geometry backend.
///
/// Each mesh volume becomes a [`Cell`] with the material looked up from
/// physical group names (`mat:<name>`). The implicit complement (unmeshed
/// space between volumes) is also represented as a cell -- void by default,
/// or with a user-supplied material for air/coolant interactions.
///
/// Ray-fire and point-in-volume queries are delegated to the underlying
/// [`yamt::MeshGeometry`].
// `MeshGeometry` wraps a `yamt::MeshGeometry` (BVH + triangulated surfaces)
// that has no full JSON form. For the model *fingerprint* (combine_results)
// it only needs a STABLE, identity-capturing summary -- not a round-trippable
// encoding. `serialize` emits that summary (counts, AABB, per-volume measures
// and material names): the same mesh fingerprints identically, geometrically
// different meshes differ. `deserialize` still fails loudly -- a fingerprint
// summary cannot reconstruct a mesh, and no path round-trips a Mesh-variant
// model from JSON.
impl serde::Serialize for MeshGeometry {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let topo = &self.mesh.topology;
        let volume_materials: Vec<Option<&str>> = (0..topo.num_volumes)
            .map(|v| self.mesh.material_name(v))
            .collect();
        let mut st = serializer.serialize_struct("MeshGeometry", 5)?;
        st.serialize_field("kind", "mesh")?;
        st.serialize_field("num_volumes", &topo.num_volumes)?;
        st.serialize_field("global_aabb", &topo.global_aabb)?;
        st.serialize_field("volume_measures", &topo.volume_measures)?;
        st.serialize_field("volume_materials", &volume_materials)?;
        st.end()
    }
}

impl<'de> serde::Deserialize<'de> for MeshGeometry {
    fn deserialize<D: serde::Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(serde::de::Error::custom(
            "MeshGeometry cannot be deserialized from JSON -- it serializes only a \
             lossy fingerprint summary, not a reconstructable form",
        ))
    }
}

#[derive(Clone)]
pub struct MeshGeometry {
    /// The underlying mesh geometry (BVH-accelerated).
    pub mesh: yamt::MeshGeometry,
    /// Synthetic cells: one per mesh volume + one for the implicit complement.
    pub cells: Vec<Cell>,
    /// Flat material store. `Cell.material_idx` is a slot index into this Vec.
    pub materials: Vec<Arc<Material>>,
    /// Volume ID for each cell index: `volume_ids[cell_index] = VolumeId`.
    volume_ids: Vec<yamt::VolumeId>,
}

impl MeshGeometry {
    /// Build a `MeshGeometry` from a loaded mesh and a material map.
    ///
    /// # Arguments
    /// * `mesh` -- already-loaded `yamt::MeshGeometry`
    /// * `materials` -- mapping from material name (e.g. `"fuel"`) to YAMC
    ///   `Material`. Names are matched against the `mat:` prefix in the
    ///   mesh's physical group names (without the prefix).
    /// * `implicit_complement_material` -- optional material for the implicit
    ///   complement (unmeshed space). `None` means void (free-streaming,
    ///   `dist_collision = INFINITY`). `Some(mat)` assigns a real material
    ///   for air/coolant interactions in the gaps between mesh volumes.
    ///
    /// # Errors
    /// Returns an error if any mesh volume references a material name that
    /// is not present in `materials`.
    pub fn new(
        mesh: yamt::MeshGeometry,
        materials: &HashMap<String, Arc<Material>>,
        implicit_complement_material: Option<Arc<Material>>,
    ) -> Result<Self, String> {
        let num_volumes = mesh.topology.num_volumes;

        let mut cells = Vec::with_capacity(num_volumes as usize + 1);
        let mut volume_ids = Vec::with_capacity(num_volumes as usize + 1);

        // Auto-assign material IDs to materials that don't have one.
        // Build a mutable copy of the map so we can set IDs via Arc::make_mut.
        let mut materials: HashMap<String, Arc<Material>> = materials.clone();
        let mut used_mat_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for mat in materials.values() {
            if let Some(id) = mat.get_material_id() {
                used_mat_ids.insert(id);
            }
        }
        if let Some(ref ic) = implicit_complement_material {
            if let Some(id) = ic.get_material_id() {
                used_mat_ids.insert(id);
            }
        }
        let mut next_mat_id: u32 = used_mat_ids.iter().copied().max().map_or(1, |m| m + 1);
        for mat in materials.values_mut() {
            if mat.get_material_id().is_none() {
                while used_mat_ids.contains(&next_mat_id) {
                    next_mat_id += 1;
                }
                Arc::make_mut(mat).set_material_id(next_mat_id);
                used_mat_ids.insert(next_mat_id);
                next_mat_id += 1;
            }
        }

        // Dummy region for mesh cells -- CSG operations are never evaluated.
        let dummy_region = make_dummy_region();

        let mut flat_materials: Vec<Arc<Material>> = Vec::new();

        for vol_id in 0..num_volumes {
            let mat_name = mesh.material_name(vol_id);

            // Use precomputed volume from topology (divergence theorem)
            let vol_measure = mesh.topology.volume_measures.get(vol_id as usize).copied();

            let material_idx = match mat_name {
                Some(name) => match materials.get(name) {
                    Some(mat) => {
                        // Propagate volume to material for transmutation
                        let per_volume = if let Some(vol) = vol_measure {
                            let mut mat_with_vol = (**mat).clone();
                            mat_with_vol.volume = Some(vol);
                            Arc::new(mat_with_vol)
                        } else {
                            Arc::clone(mat)
                        };
                        let idx = flat_materials.len() as u32;
                        flat_materials.push(per_volume);
                        Some(idx)
                    }
                    None => {
                        eprintln!(
                            "Warning: volume {vol_id} material \"{name}\" \
                             not in materials map; treating as void"
                        );
                        None
                    }
                },
                None => None, // Void volume (no material)
            };

            let cell = Cell {
                cell_id: Some(vol_id),
                name: mat_name.map(|n| format!("mesh_vol_{vol_id}_{n}")),
                flat_region: dummy_region.flatten(),
                region: dummy_region.clone(),
                material_idx,
                volume: vol_measure,
                fill_role: None,
                cached_surfaces: Vec::new(), // Not used -- mesh handles boundaries
            };

            cells.push(cell);
            volume_ids.push(vol_id);
        }

        // Implicit complement cell -- the last cell. Void (material_idx: None)
        // unless the user supplied a material for air/coolant interactions.
        let ic_vol = mesh.topology.implicit_complement;
        let ic_material_idx = implicit_complement_material.map(|mat| {
            let idx = flat_materials.len() as u32;
            flat_materials.push(mat);
            idx
        });
        let ic_cell = Cell {
            cell_id: Some(ic_vol),
            name: Some("implicit_complement".to_string()),
            flat_region: dummy_region.flatten(),
            region: dummy_region,
            material_idx: ic_material_idx,
            volume: None,
            fill_role: None,
            cached_surfaces: Vec::new(),
        };
        cells.push(ic_cell);
        volume_ids.push(ic_vol);

        Ok(MeshGeometry {
            mesh,
            cells,
            materials: flat_materials,
            volume_ids,
        })
    }

    /// Build from an Arrow IPC file produced by cad_to_yamc.
    pub fn from_arrow(
        path: &std::path::Path,
        materials: &HashMap<String, Arc<Material>>,
    ) -> Result<Self, String> {
        let mesh = yamt::MeshGeometry::from_arrow(path)
            .map_err(|e| format!("Failed to load Arrow mesh: {e}"))?;
        Self::new(mesh, materials, None)
    }

    /// True when a surface that does not bound `cell_index`'s volume
    /// intersects the open segment from `origin` towards the accepted
    /// crossing at distance `t_max`. Used by the spatial tracking
    /// verification path (issue #254): for valid geometry no foreign
    /// surface can lie strictly inside the current volume, so a hit
    /// means the mesh volumes overlap or self-intersect and adjacency
    /// tracking is about to go wrong. Works for the implicit complement
    /// cell as well (foreign then means any surface not bounding it).
    pub(crate) fn crossing_blocked(
        &self,
        cell_index: usize,
        origin: [f64; 3],
        direction: [f64; 3],
        t_max: f64,
    ) -> bool {
        let vol = self.volume_ids[cell_index];
        self.mesh
            .segment_blocked(vol, origin, direction, t_max)
            .is_some()
    }

    /// Find the cell containing a point.
    ///
    /// Delegates to [`yamt::MeshGeometry::find_volume`] and maps the result
    /// to a cell index. Points in the implicit complement return the
    /// complement cell (last cell).
    pub fn find_cell_index(&self, point: (f64, f64, f64)) -> Option<usize> {
        let vol = self.mesh.find_volume([point.0, point.1, point.2]);
        if vol == self.mesh.topology.implicit_complement {
            // Return the complement cell (last cell)
            Some(self.cells.len() - 1)
        } else {
            // volume_ids[i] == i for 0..num_volumes, so vol == cell_index
            Some(vol as usize)
        }
    }

    /// Find the closest boundary crossing from a position along a direction.
    pub fn closest_boundary(
        &self,
        cell_index: usize,
        position: [f64; 3],
        direction: [f64; 3],
    ) -> Option<BoundaryHit> {
        let vol_id = self.volume_ids[cell_index];
        let ic_cell_index = self.cells.len() - 1;

        self.mesh
            .ray_fire(vol_id, position, direction, None)
            .map(|(distance, surface_id)| {
                let bc = self.mesh.boundary_condition(surface_id);
                let boundary = match bc {
                    yamt::BoundaryCondition::Vacuum => BoundaryType::Vacuum,
                    _ => BoundaryType::Transmission,
                };

                // Use topology to determine the next cell (O(1) lookup).
                let next_cell_index = if boundary == BoundaryType::Vacuum {
                    None // Particle dies at vacuum boundary
                } else {
                    self.mesh.next_volume(surface_id, vol_id).map(|next_vol| {
                        if next_vol == self.mesh.topology.implicit_complement {
                            // Entering the implicit complement -- particle
                            // free-streams (void) or scatters (if material set).
                            ic_cell_index
                        } else {
                            next_vol as usize
                        }
                    })
                };

                BoundaryHit {
                    distance,
                    boundary,
                    next_cell_index,
                    surface_id: None, // Mesh surfaces don't have user-facing IDs
                }
            })
    }

    /// Compute the bounding box of the mesh geometry.
    pub fn bounding_box(&self) -> BoundingBox {
        let bb = &self.mesh.topology.global_aabb;
        BoundingBox::new([bb[0], bb[1], bb[2]], [bb[3], bb[4], bb[5]])
    }

    /// Compute the bounding box of volumes matching a specific material ID.
    ///
    /// Returns `None` if no volumes match the given material.
    /// Only considers real volumes (excludes implicit complement).
    pub fn bounding_box_for_material(&self, material_id: u32) -> Option<BoundingBox> {
        let num_real = self.mesh.topology.num_volumes as usize;
        let mut result: Option<BoundingBox> = None;
        for (i, cell) in self.cells[..num_real].iter().enumerate() {
            let matches = cell
                .material_idx
                .and_then(|mi| self.materials.get(mi as usize))
                .and_then(|m| m.get_material_id())
                .map(|id| id == material_id)
                .unwrap_or(false);
            if matches {
                let aabb = &self.mesh.topology.volume_aabbs[i];
                let cell_bbox =
                    BoundingBox::new([aabb[0], aabb[1], aabb[2]], [aabb[3], aabb[4], aabb[5]]);
                match &mut result {
                    None => result = Some(cell_bbox),
                    Some(b) => b.expand_to_include(&cell_bbox),
                }
            }
        }
        result
    }
}

/// Create a dummy CSG region for mesh cells.
///
/// The region is never evaluated during mesh transport (containment and
/// boundary queries go through the mesh backend). We create a simple
/// "inside an infinite sphere" region to satisfy the `Cell` constructor.
fn make_dummy_region() -> Region {
    let big_sphere = Arc::new(Surface {
        surface_id: None,
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 1e30,
        },
        boundary: BoundaryType::Transmission,
        name: None,
    });
    Region {
        expr: RegionExpr::Halfspace(HalfspaceType::Below(big_sphere)),
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const CUBE_ARROW: &str = "../yamt/tests/data/cube.arrow";

    fn cube_mesh() -> yamt::MeshGeometry {
        yamt::MeshGeometry::from_arrow(std::path::Path::new(CUBE_ARROW)).unwrap()
    }

    #[test]
    fn test_dagmc_geometry_cube() {
        // Load the cube mesh from yamt test data
        let mesh = cube_mesh();

        // Create a material map -- the cube has mat:water
        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.material_id = Some(1);

        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();

        // Should have cells (at least 1 for the water volume)
        assert!(!geom.cells.is_empty());

        // Point at center should be inside the water volume
        let idx = geom.find_cell_index((0.5, 0.5, 0.5));
        assert!(idx.is_some(), "Point at center should be found");
        let cell = &geom.cells[idx.unwrap()];
        assert!(cell.material_idx.is_some());

        // Point outside the mesh volumes should return the complement cell
        let outside = geom.find_cell_index((10.0, 10.0, 10.0));
        assert!(
            outside.is_some(),
            "Point outside mesh should be complement cell"
        );
        let ic_cell = &geom.cells[outside.unwrap()];
        assert!(
            ic_cell.material_idx.is_none(),
            "Complement cell should be void"
        );
    }

    #[test]
    fn test_dagmc_geometry_closest_boundary() {
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.material_id = Some(1);

        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();

        let idx = geom.find_cell_index((0.5, 0.5, 0.5)).unwrap();

        // Fire ray in +z direction -- should hit the z=1 face
        let hit = geom.closest_boundary(idx, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0]);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!(
            h.distance > 0.0 && h.distance < 1.0,
            "Distance should be ~0.5, got {}",
            h.distance
        );
        // Outer face should be vacuum (cube.arrow has boundary:vacuum)
        assert_eq!(h.boundary, BoundaryType::Vacuum);
    }

    #[test]
    fn test_dagmc_geometry_missing_material_treated_as_void() {
        let mesh = cube_mesh();

        // Empty material map -- missing "water" is treated as void with a warning
        let materials = HashMap::new();
        let result = MeshGeometry::new(mesh, &materials, None);
        assert!(result.is_ok());
        let geom = result.unwrap();
        // Volume cell should have no material (void)
        assert!(geom.cells[0].material_idx.is_none());
    }

    #[test]
    fn test_mesh_cell_volume_populated() {
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.material_id = Some(1);
        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();
        let cell = &geom.cells[0];

        // Cell volume should be populated (not None)
        assert!(cell.volume.is_some(), "Cell volume should be set from mesh");
        let vol = cell.volume.unwrap();
        assert!(
            (vol - 1.0).abs() < 0.1,
            "Unit cube cell volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn test_mesh_material_volume_propagated() {
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.material_id = Some(1);
        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();
        let cell = &geom.cells[0];
        let mat = geom
            .materials
            .get(cell.material_idx.expect("cell has material") as usize)
            .expect("material slot populated");

        // Material volume should also be set (for transmutation)
        assert!(
            mat.volume.is_some(),
            "Material volume should be propagated from mesh"
        );
        let vol = mat.volume.unwrap();
        assert!(
            (vol - 1.0).abs() < 0.1,
            "Unit cube material volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn test_bounding_box_for_material() {
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.material_id = Some(1);
        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();

        // Should find bounding box for material_id=1 (water)
        let bb = geom.bounding_box_for_material(1);
        assert!(bb.is_some(), "Should find bbox for water material");
        let bb = bb.unwrap();
        // Cube is [0,1]^3
        for i in 0..3 {
            assert!(bb.lower_left[i] <= 0.01);
            assert!(bb.upper_right[i] >= 0.99);
        }

        // Should return None for non-existent material
        let bb_none = geom.bounding_box_for_material(999);
        assert!(bb_none.is_none(), "Should return None for unknown material");
    }

    #[test]
    fn test_dagmc_geometry_bounding_box() {
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.material_id = Some(1);
        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();
        let bb = geom.bounding_box();

        // Cube is [0,1]^3
        for i in 0..3 {
            assert!(bb.lower_left[i] <= 0.01);
            assert!(bb.upper_right[i] >= 0.99);
        }
    }

    #[test]
    fn test_mesh_geometry_serializes_for_fingerprint() {
        // Regression: a Mesh-variant model must fingerprint (combine_results /
        // simulate_transport), which serializes the geometry to JSON. This
        // previously errored ("MeshGeometry is not yet JSON-serializable"),
        // breaking ALL mesh-geometry transport.
        let mesh = cube_mesh();

        let mut water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        water.set_name("water");
        water.material_id = Some(1);
        let mut materials = HashMap::new();
        materials.insert("water".to_string(), Arc::new(water));

        let geom = MeshGeometry::new(mesh, &materials, None).unwrap();

        let v1 = serde_json::to_value(&geom).expect("mesh geometry should serialize");
        let v2 = serde_json::to_value(&geom).expect("serialize again");
        assert_eq!(v1, v2, "fingerprint summary must be deterministic");
        assert_eq!(v1["kind"], "mesh");
        assert_eq!(v1["num_volumes"], 1);
        assert!(v1["volume_materials"].is_array());
    }
}
