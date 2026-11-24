//! `UnstructuredMesh`: a mesh that tallies score on, apart from the geometry.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::geometry::mesh_geometry::tet_mesh_bytes;

/// An unstructured mesh for tallies, apart from the transport geometry.
///
/// Like an OpenMC/XDG unstructured mesh: any number of them, at any
/// resolution, conformal with the geometry or not, from any mesher. A tally
/// on it (``Tally(mesh=mesh)``) scores in each cell the track
/// length through it, walking the mesh alongside the transport, so a fine
/// tally mesh does not change how particles are tracked. It carries no
/// materials: where the material state varies cell by cell, the cells belong
/// in the geometry (``MeshGeometry.from_tets``, ``MeshGeometry.add_tets``),
/// whose own tets a tally takes directly
/// (``Tally(mesh=geometry)``). The cells are tetrahedra for now.
///
/// Args:
///     vertices (Sequence[Sequence[float]]): ``[x, y, z]`` per vertex (cm).
///     cells (Sequence[Sequence[int]]): Four vertex indices per cell (a
///         tetrahedron); the tally's bins are in this order. Negatively
///         wound cells are re-wound.
///     cell_regions (Sequence[str] | None): A region name per cell, to tally
///         one region with ``mesh=(mesh, "name")``; ``None``
///         puts every cell in one region, ``"mesh"``.
///
/// Raises:
///     ValueError: For a cell with no volume, a vertex index out of range, or
///         a face shared by more than two cells.
///
/// Examples:
///     >>> thermal = yamc.UnstructuredMesh(vertices, cells)  # e.g. a FEniCSx mesh
///     >>> heating = yamc.Tally(scores=["heating"], mesh=thermal)
#[gen_stub_pyclass]
#[pyclass(
    module = "yamc._core",
    name = "UnstructuredMesh",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyUnstructuredMesh {
    pub(crate) mesh: Arc<yamt::MeshGeometry>,
    regions: Arc<Vec<String>>,
}

impl PyUnstructuredMesh {
    pub(crate) fn build(
        py: Python<'_>,
        vertices: Vec<[f64; 3]>,
        tets: Vec<[u32; 4]>,
        tet_regions: Vec<String>,
    ) -> PyResult<Self> {
        let bytes = tet_mesh_bytes(py, vertices, tets, &tet_regions)?;
        let mesh = py
            .detach(|| {
                let data = yamt::read_arrow_mesh_from(std::io::Cursor::new(&bytes))
                    .map_err(|e| e.to_string())?;
                let topo = yamt::build_topology(data).map_err(|e| e.to_string())?;
                Ok::<_, String>(yamt::MeshGeometry::from_topology(topo))
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let regions = (0..mesh.topology.num_volumes)
            .map(|v| mesh.material_name(v).unwrap_or_default().to_owned())
            .collect();
        Ok(Self {
            mesh: Arc::new(mesh),
            regions: Arc::new(regions),
        })
    }

    /// The 0-based volume of the region named `name`.
    pub(crate) fn region_id(&self, name: &str) -> PyResult<u32> {
        self.regions
            .iter()
            .position(|r| r == name)
            .map(|v| v as u32)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "no region named {name:?}; the regions are {}",
                    self.regions
                        .iter()
                        .map(|r| format!("{r:?}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }

    /// The volume ids of every region that has tets.
    pub(crate) fn all_regions(&self) -> Vec<u32> {
        let topo = &self.mesh.topology;
        (0..self.regions.len() as u32)
            .filter(|&v| {
                topo.volume_tet_ranges
                    .get(v as usize)
                    .is_some_and(|r| !r.is_empty())
            })
            .collect()
    }

    /// The tets of a mesh geometry, as a tally mesh that shares their memory.
    pub(crate) fn of_geometry(mesh: &Arc<yamt::MeshGeometry>) -> Self {
        let regions = (0..mesh.topology.num_volumes)
            .map(|v| mesh.material_name(v).unwrap_or_default().to_owned())
            .collect();
        Self {
            mesh: Arc::clone(mesh),
            regions: Arc::new(regions),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyUnstructuredMesh {
    #[new]
    #[pyo3(signature = (vertices, cells, cell_regions=None))]
    fn new(
        py: Python<'_>,
        vertices: Vec<[f64; 3]>,
        cells: Vec<[u32; 4]>,
        cell_regions: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let regions = cell_regions.unwrap_or_else(|| vec!["mesh".to_string(); cells.len()]);
        Self::build(py, vertices, cells, regions)
    }

    /// The region names, in order of first appearance.
    #[getter]
    fn regions(&self) -> Vec<String> {
        self.regions.to_vec()
    }

    /// All vertex coordinates as ``[x, y, z]`` triples (cm).
    #[getter]
    fn vertices(&self) -> Vec<[f64; 3]> {
        self.mesh.topology.vertices.clone()
    }

    /// The cells, as four vertex indices each, in the order given.
    #[getter]
    fn cells(&self) -> Vec<[u32; 4]> {
        self.mesh.topology.tetrahedra.clone()
    }

    /// The region name of each cell, parallel to ``cells``.
    #[getter]
    fn cell_regions(&self) -> Vec<String> {
        self.mesh
            .topology
            .tet_volume_ids
            .iter()
            .map(|v| {
                v.map(|v| self.regions[v as usize].clone())
                    .unwrap_or_default()
            })
            .collect()
    }

    /// The volume of each cell (cm³), parallel to ``cells``.
    #[getter]
    fn cell_volumes(&self) -> Vec<f64> {
        (0..self.mesh.topology.tetrahedra.len() as u32)
            .map(|t| self.mesh.tet_volume(t))
            .collect()
    }

    /// Number of cells.
    #[getter]
    fn num_cells(&self) -> usize {
        self.mesh.topology.tetrahedra.len()
    }

    fn __len__(&self) -> usize {
        self.num_cells()
    }

    fn __repr__(&self) -> String {
        format!(
            "UnstructuredMesh(cells={}, regions={})",
            self.num_cells(),
            self.regions.len()
        )
    }
}
