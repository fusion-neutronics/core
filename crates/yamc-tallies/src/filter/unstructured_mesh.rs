//! Unstructured mesh filter for tally scoring on a tetrahedral mesh overlay.
//!
//! Uses `yamt`'s element walking to decompose particle tracks into
//! per-tetrahedron segments for unstructured mesh tallies.

use std::sync::Arc;

use crate::mesh::MeshCrossing;

/// Filter that scores tallies on an unstructured tetrahedral mesh.
///
/// Each tetrahedron in the mesh is a scoring bin. When a particle
/// travels from point A to point B, the filter uses element walking
/// to determine which tetrahedra the track passes through and how
/// much path length falls in each.
#[derive(Clone)]
pub struct UnstructuredMeshFilter {
    /// The mesh geometry with element BVH for fast lookups.
    mesh: Arc<yamt::MeshGeometry>,
    /// Which volume of the mesh to score on.
    volume_id: yamt::VolumeId,
    /// Total number of tets in the mesh (bins = global tet IDs).
    num_tets: usize,
}

// The mesh geometry is shared via `Arc<yamt::MeshGeometry>` and has no
// round-trippable JSON form, but serializing is not about round-tripping: the
// model *fingerprint* (`combine_results` identity, computed on every Python
// `simulate_transport`) only needs a STABLE, identity-capturing summary. This
// emits that summary -- the scored volume plus the mesh's shape counts, extent
// and per-volume measures -- so the same mesh and volume fingerprint
// identically while a different mesh or volume differs. Same treatment
// `yamc::geometry::MeshGeometry` already gives mesh geometry.
//
// Erroring here instead used to fail EVERY Python run carrying a tet-mesh
// tally, after the transport had finished (issue #290).
//
// `deserialize` still fails loudly: a summary cannot reconstruct a mesh, and no
// path round-trips a tet-tally model from JSON.
impl serde::Serialize for UnstructuredMeshFilter {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let topo = &self.mesh.topology;
        let mut st = serializer.serialize_struct("UnstructuredMeshFilter", 7)?;
        st.serialize_field("kind", "unstructured_mesh")?;
        st.serialize_field("volume_id", &self.volume_id)?;
        st.serialize_field("num_tets", &self.num_tets)?;
        st.serialize_field("num_volumes", &topo.num_volumes)?;
        st.serialize_field("num_vertices", &topo.vertices.len())?;
        st.serialize_field("global_aabb", &topo.global_aabb)?;
        st.serialize_field("volume_measures", &topo.volume_measures)?;
        st.end()
    }
}

impl<'de> serde::Deserialize<'de> for UnstructuredMeshFilter {
    fn deserialize<D: serde::Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(serde::de::Error::custom(
            "UnstructuredMeshFilter is not yet deserializable from JSON",
        ))
    }
}

impl UnstructuredMeshFilter {
    /// Create a new unstructured mesh filter.
    ///
    /// # Arguments
    /// * `mesh` -- the mesh geometry (shared via Arc for thread safety)
    /// * `volume_id` -- which volume of the mesh to walk through
    pub fn new(mesh: Arc<yamt::MeshGeometry>, volume_id: yamt::VolumeId) -> Self {
        let num_tets = mesh.topology.tetrahedra.len();
        UnstructuredMeshFilter {
            mesh,
            volume_id,
            num_tets,
        }
    }

    /// Get all bins (tetrahedra) crossed by a particle track with their length fractions.
    ///
    /// Delegates to `yamt::MeshGeometry::segments()` which uses element
    /// walking through the tetrahedral mesh.
    pub fn get_bins_crossed(
        &self,
        r0: [f64; 3],
        r1: [f64; 3],
        _direction: [f64; 3],
    ) -> Vec<MeshCrossing> {
        let segments = self.mesh.segments(self.volume_id, r0, r1);

        if segments.is_empty() {
            return Vec::new();
        }

        // Compute total track length for normalization
        let dx = r1[0] - r0[0];
        let dy = r1[1] - r0[1];
        let dz = r1[2] - r0[2];
        let total_length = (dx * dx + dy * dy + dz * dz).sqrt();

        if total_length < 1e-15 {
            return Vec::new();
        }

        segments
            .into_iter()
            .map(|(tet_id, segment_length)| MeshCrossing {
                bin: tet_id as usize,
                length_fraction: segment_length / total_length,
            })
            .collect()
    }

    /// Bin (tetrahedron) containing the given position, or `None` when
    /// the point lies outside the mesh volume. The bin index is the
    /// global tetrahedron id -- the same mapping `get_bins_crossed`
    /// uses -- resolved with yamt's element-BVH point query (issue
    /// #354).
    pub fn get_bin(&self, position: [f64; 3]) -> Option<usize> {
        self.mesh
            .find_element(self.volume_id, position)
            .map(|tet_id| tet_id as usize)
    }

    /// Total number of bins (tetrahedra).
    pub fn num_bins(&self) -> usize {
        self.num_tets
    }

    /// Get the tet volume for a given bin (for normalizing tally results).
    pub fn get_element_volume(&self, bin: usize) -> f64 {
        self.mesh.tet_volume(bin as yamt::TetrahedronId)
    }

    /// Access the underlying mesh geometry.
    pub fn mesh(&self) -> &yamt::MeshGeometry {
        &self.mesh
    }

    /// The volume ID being scored.
    pub fn volume_id(&self) -> yamt::VolumeId {
        self.volume_id
    }
}

impl std::fmt::Debug for UnstructuredMeshFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnstructuredMeshFilter")
            .field("volume_id", &self.volume_id)
            .field("num_tets", &self.num_tets)
            .finish()
    }
}

impl PartialEq for UnstructuredMeshFilter {
    fn eq(&self, other: &Self) -> bool {
        // Compare by volume_id and num_tets (Arc pointer equality for mesh)
        self.volume_id == other.volume_id
            && self.num_tets == other.num_tets
            && Arc::ptr_eq(&self.mesh, &other.mesh)
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_mesh() -> Arc<yamt::MeshGeometry> {
        Arc::new(
            yamt::MeshGeometry::from_arrow(std::path::Path::new("../yamt/tests/data/cube.arrow"))
                .unwrap(),
        )
    }

    #[test]
    fn test_unstructured_mesh_filter_bins() {
        let mesh = cube_mesh();

        // Volume 0 is the water volume
        let filter = UnstructuredMeshFilter::new(mesh.clone(), 0);

        assert!(filter.num_bins() > 0);
        assert_eq!(filter.volume_id(), 0);
    }

    #[test]
    fn test_unstructured_mesh_filter_crossing() {
        let mesh = cube_mesh();

        let filter = UnstructuredMeshFilter::new(mesh, 0);

        // A ray through the cube [0,1]^3
        let r0 = [0.3, 0.3, 0.05];
        let r1 = [0.3, 0.3, 0.95];
        let direction = [0.0, 0.0, 1.0];

        let crossings = filter.get_bins_crossed(r0, r1, direction);

        // Should have at least one crossing
        assert!(
            !crossings.is_empty(),
            "Should find crossings through the cube"
        );

        // Total length fractions should sum to approximately 1.0
        // (or less if the mesh doesn't cover the full track)
        let total_fraction: f64 = crossings.iter().map(|c| c.length_fraction).sum();
        assert!(
            total_fraction > 0.5,
            "Total length fraction should be substantial, got {total_fraction}"
        );
    }

    #[test]
    fn test_unstructured_mesh_filter_outside() {
        let mesh = cube_mesh();

        let filter = UnstructuredMeshFilter::new(mesh, 0);

        // Completely outside the cube
        let crossings = filter.get_bins_crossed([5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [1.0, 0.0, 0.0]);
        assert!(
            crossings.is_empty(),
            "Outside mesh should have no crossings"
        );
    }

    /// Regression for issue #290: a tally carrying this filter must serialize,
    /// because the model fingerprint (computed on every Python
    /// `simulate_transport`) serializes the whole model. This used to error and
    /// took every tet-mesh tally run down with it.
    #[test]
    fn serializes_a_stable_identity_summary() {
        let mesh = cube_mesh();
        let filter = UnstructuredMeshFilter::new(Arc::clone(&mesh), 0);

        let v1 = serde_json::to_value(&filter).expect("filter must serialize");
        let v2 = serde_json::to_value(&filter).expect("serialize again");
        assert_eq!(v1, v2, "the identity summary must be deterministic");
        assert_eq!(v1["kind"], "unstructured_mesh");
        assert_eq!(v1["volume_id"], 0);
        assert_eq!(v1["num_tets"], filter.num_bins());
        assert!(v1["num_vertices"].as_u64().unwrap() > 0);
        assert!(v1["global_aabb"].is_array() || v1["global_aabb"].is_object());

        // The scored volume is part of the identity, so a different volume must
        // fingerprint differently (a same-mesh, same-volume filter must not).
        let same = UnstructuredMeshFilter::new(Arc::clone(&mesh), 0);
        assert_eq!(v1, serde_json::to_value(&same).unwrap());
        if mesh.topology.num_volumes > 1 {
            let other = UnstructuredMeshFilter::new(Arc::clone(&mesh), 1);
            assert_ne!(v1, serde_json::to_value(&other).unwrap());
        }
    }

    /// A summary cannot reconstruct a mesh, so loading must still fail loudly
    /// rather than silently produce a filter with no geometry.
    #[test]
    fn deserialize_still_fails_loudly() {
        let json = r#"{"kind":"unstructured_mesh","volume_id":0,"num_tets":1}"#;
        let err = serde_json::from_str::<UnstructuredMeshFilter>(json)
            .expect_err("deserialize must fail");
        assert!(err.to_string().contains("not yet deserializable"));
    }

    #[test]
    fn test_element_volume() {
        let mesh = cube_mesh();

        let filter = UnstructuredMeshFilter::new(mesh, 0);

        // All tets should have positive volume
        for bin in 0..filter.num_bins() {
            let vol = filter.get_element_volume(bin);
            assert!(
                vol > 0.0,
                "Tet {bin} should have positive volume, got {vol}"
            );
        }
    }
}
