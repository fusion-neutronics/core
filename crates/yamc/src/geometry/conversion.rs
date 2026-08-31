//! Conversion functions from yamc transport types to yamc-geo serializable types.

use crate::geometry::Geometry;
use yamc_geo::cell::GeoCell;
use yamc_geo::geometry::CsgGeometry;

/// Convert a transport-ready Geometry to a serializable CsgGeometry.
///
/// Extracts cell_id, material_id, names, and region from each cell.
/// The Region and Surface types are identical (re-exported from yamc-geo).
pub fn geometry_to_csg(geometry: &Geometry) -> CsgGeometry {
    let cells = geometry
        .cells
        .iter()
        .map(|c| cell_to_geo_cell(c, &geometry.materials))
        .collect();
    CsgGeometry { cells }
}

/// Convert a single transport Cell to a serializable GeoCell.
pub fn cell_to_geo_cell(
    cell: &crate::geometry::cell::Cell,
    materials: &[std::sync::Arc<yamc_materials::material::Material>],
) -> GeoCell {
    let mat = cell.material_idx.and_then(|i| materials.get(i as usize));
    GeoCell {
        cell_id: cell.cell_id.map(|id| id as i32).unwrap_or(-1),
        material_id: mat
            .and_then(|m| m.get_material_id())
            .map(|id| id as i32)
            .unwrap_or(-1),
        name: cell.name.clone(),
        material_name: mat.and_then(|m| m.get_name().map(|n| n.to_string())),
        region: cell.region.clone(),
    }
}

/// Convert a mesh-based MeshGeometry to a serializable GeoMesh.
///
/// Pre-expands triangle vertex coordinates so WASM doesn't need
/// vertex indirection.
#[cfg(feature = "mesh")]
pub fn mesh_geometry_to_geo_mesh(mesh: &crate::geometry::mesh::MeshGeometry) -> yamc_geo::GeoMesh {
    let topo = &mesh.mesh.topology;
    let num_volumes = topo.num_volumes as usize;

    let mut volumes = Vec::with_capacity(num_volumes);
    for vol_id in 0..num_volumes {
        let cell = &mesh.cells[vol_id];
        let cell_id = cell.cell_id.map(|id| id as i32).unwrap_or(-1);
        let mat = cell
            .material_idx
            .and_then(|i| mesh.materials.get(i as usize));
        let material_id = mat
            .and_then(|m| m.get_material_id())
            .map(|id| id as i32)
            .unwrap_or(-1);
        let material_name = mat.and_then(|m| m.get_name().map(|n| n.to_string()));

        // Collect all triangle indices for this volume
        let mut tri_indices = Vec::new();
        for &(surf_id, _sense) in &topo.volume_surfaces[vol_id] {
            let range = &topo.surface_tri_ranges[surf_id as usize];
            for idx in range.clone() {
                tri_indices.push(topo.surface_tri_indices[idx as usize]);
            }
        }

        // Pre-expand triangle vertex coordinates
        let triangles: Vec<[[f64; 3]; 3]> = tri_indices
            .iter()
            .map(|&tri_idx| {
                let [v0, v1, v2] = topo.triangles[tri_idx as usize];
                [
                    topo.vertices[v0 as usize],
                    topo.vertices[v1 as usize],
                    topo.vertices[v2 as usize],
                ]
            })
            .collect();

        volumes.push(yamc_geo::GeoMeshVolume {
            cell_id,
            material_id,
            material_name,
            triangles,
        });
    }

    let bb = &topo.global_aabb;
    yamc_geo::GeoMesh {
        volumes,
        global_min: [bb[0], bb[1], bb[2]],
        global_max: [bb[3], bb[4], bb[5]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::{HalfspaceType, Region, Surface};
    use crate::geometry::cell::Cell;
    use std::collections::HashMap;
    use std::sync::Arc;
    use yamc_materials::material::Material;

    /// Region = inside a sphere of radius 3 centred at the origin.
    fn inside_sphere() -> Region {
        let s = Surface::new_sphere(0.0, 0.0, 0.0, 3.0, Some(1), None);
        Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s)))
    }

    fn water() -> Arc<Material> {
        let mut m = Material::new(
            HashMap::from([("Li6".to_string(), 1.0)]),
            "atom",
            "g/cm3",
            Some(0.5),
        )
        .unwrap();
        m.set_material_id(42);
        m.set_name("water");
        Arc::new(m)
    }

    #[test]
    fn cell_to_geo_cell_maps_id_material_and_names() {
        let materials = vec![water()];
        let cell = Cell::new(Some(7), inside_sphere(), Some("ball".to_string()), Some(0));
        let geo = cell_to_geo_cell(&cell, &materials);
        assert_eq!(geo.cell_id, 7);
        assert_eq!(geo.material_id, 42);
        assert_eq!(geo.name.as_deref(), Some("ball"));
        assert_eq!(geo.material_name.as_deref(), Some("water"));
        // The region is carried over intact.
        assert!(geo.contains((0.0, 0.0, 0.0)));
        assert!(!geo.contains((5.0, 0.0, 0.0)));
    }

    #[test]
    fn cell_to_geo_cell_void_cell_uses_minus_one_sentinels() {
        // No cell_id and no material -> -1 sentinels and no names.
        let materials = vec![water()];
        let cell = Cell::new(None, inside_sphere(), None, None);
        let geo = cell_to_geo_cell(&cell, &materials);
        assert_eq!(geo.cell_id, -1);
        assert_eq!(geo.material_id, -1);
        assert!(geo.name.is_none());
        assert!(geo.material_name.is_none());
    }

    #[test]
    fn geometry_to_csg_converts_every_cell() {
        let cell = Cell::new(Some(7), inside_sphere(), Some("ball".to_string()), Some(0));
        let geom = crate::geometry::Geometry::new(vec![cell], vec![water()]).unwrap();
        let csg = geometry_to_csg(&geom);
        assert_eq!(csg.cells.len(), 1);
        assert_eq!(csg.cells[0].cell_id, 7);
        assert_eq!(csg.cells[0].material_id, 42);
    }
}
