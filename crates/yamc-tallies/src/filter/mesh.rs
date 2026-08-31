use crate::mesh::{BinsCrossedIter, CylindricalMesh, MeshCrossing, RegularRectangularMesh};

/// The structured mesh a [`MeshFilter`] scores on.
///
/// Both variants share the same flat-bin tally interface (`get_bin`,
/// `get_bins_crossed`, `num_bins`, per-bin volume); the filter dispatches over
/// this enum so the tally machinery is agnostic to the coordinate system.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MeshKind {
    /// Axis-aligned Cartesian mesh (uniform voxel volume).
    Rectangular(RegularRectangularMesh),
    /// Cylindrical `(r, φ, z)` mesh (per-ring voxel volume).
    Cylindrical(CylindricalMesh),
}

/// Filter for scoring on a structured spatial mesh.
///
/// A MeshFilter allows tallies to score based on spatial position, enabling
/// voxel-based tallying for flux, heating, and other scores on either a
/// rectangular or a cylindrical mesh.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MeshFilter {
    mesh: MeshKind,
}

/// Lazy iterator over the mesh cells crossed by a track, abstracting over the
/// mesh kind. The rectangular arm streams without allocating; the cylindrical
/// arm walks a materialised `Vec` (one allocation per track), matching the
/// unstructured-mesh path.
pub enum MeshCrossingsIter<'a> {
    Rectangular(BinsCrossedIter<'a>),
    Cylindrical(std::vec::IntoIter<MeshCrossing>),
}

impl Iterator for MeshCrossingsIter<'_> {
    type Item = MeshCrossing;

    #[inline]
    fn next(&mut self) -> Option<MeshCrossing> {
        match self {
            MeshCrossingsIter::Rectangular(it) => it.next(),
            MeshCrossingsIter::Cylindrical(it) => it.next(),
        }
    }
}

impl MeshFilter {
    /// Create a MeshFilter from a rectangular mesh.
    pub fn new(mesh: RegularRectangularMesh) -> Self {
        MeshFilter {
            mesh: MeshKind::Rectangular(mesh),
        }
    }

    /// Create a MeshFilter from a cylindrical mesh.
    pub fn new_cylindrical(mesh: CylindricalMesh) -> Self {
        MeshFilter {
            mesh: MeshKind::Cylindrical(mesh),
        }
    }

    /// The mesh kind this filter scores on.
    pub fn kind(&self) -> &MeshKind {
        &self.mesh
    }

    /// Get the bin index for a position. `None` if outside the mesh.
    pub fn get_bin(&self, position: [f64; 3]) -> Option<usize> {
        match &self.mesh {
            MeshKind::Rectangular(m) => m.get_bin(position),
            MeshKind::Cylindrical(m) => m.get_bin(position),
        }
    }

    /// All bins crossed by a track with their length fractions (track-length
    /// scoring across multiple cells in one step).
    pub fn get_bins_crossed(
        &self,
        r0: [f64; 3],
        r1: [f64; 3],
        direction: [f64; 3],
    ) -> Vec<MeshCrossing> {
        match &self.mesh {
            MeshKind::Rectangular(m) => m.bins_crossed(r0, r1, direction),
            MeshKind::Cylindrical(m) => m.bins_crossed(r0, r1, direction),
        }
    }

    /// Iterator form of [`Self::get_bins_crossed`]. The rectangular arm avoids
    /// a `Vec` allocation; the cylindrical arm iterates a materialised list.
    pub fn get_bins_crossed_iter(
        &self,
        r0: [f64; 3],
        r1: [f64; 3],
        direction: [f64; 3],
    ) -> MeshCrossingsIter<'_> {
        match &self.mesh {
            MeshKind::Rectangular(m) => {
                MeshCrossingsIter::Rectangular(m.bins_crossed_iter(r0, r1, direction))
            }
            MeshKind::Cylindrical(m) => {
                MeshCrossingsIter::Cylindrical(m.bins_crossed(r0, r1, direction).into_iter())
            }
        }
    }

    /// Total number of bins in the mesh, i.e. the length of the flat scoring
    /// buffer. Every index the mesh yields is `< num_bins()`, and every index
    /// below it is reachable.
    pub fn num_bins(&self) -> usize {
        match &self.mesh {
            MeshKind::Rectangular(m) => m.num_voxels(),
            MeshKind::Cylindrical(m) => m.num_bins(),
        }
    }

    /// Volume of mesh cell `bin` (for normalising tally results). Constant for
    /// a rectangular mesh; per-ring for a cylindrical mesh.
    pub fn get_element_volume(&self, bin: usize) -> f64 {
        match &self.mesh {
            MeshKind::Rectangular(m) => m.get_voxel_volume(bin),
            MeshKind::Cylindrical(m) => m.get_voxel_volume(bin),
        }
    }

    /// True if a position lies within the mesh.
    pub fn matches(&self, position: [f64; 3]) -> bool {
        self.get_bin(position).is_some()
    }

    /// The underlying rectangular mesh, or `None` for a cylindrical filter.
    pub fn rectangular_mesh(&self) -> Option<&RegularRectangularMesh> {
        match &self.mesh {
            MeshKind::Rectangular(m) => Some(m),
            MeshKind::Cylindrical(_) => None,
        }
    }

    /// The underlying cylindrical mesh, or `None` for a rectangular filter.
    pub fn cylindrical_mesh(&self) -> Option<&CylindricalMesh> {
        match &self.mesh {
            MeshKind::Cylindrical(m) => Some(m),
            MeshKind::Rectangular(_) => None,
        }
    }
}
