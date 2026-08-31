use crate::geometry::PyBoundingBox;
use crate::material::PyMaterial;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use yamc::geometry::mesh::MeshGeometry;

/// Mesh-based geometry loaded from a mesh file.
///
/// The mesh format is the YAMC Arrow IPC mesh (``.arrow``), as produced by
/// ``cad_to_yamc``. Other meshers are supported by writing that format.
///
/// Each volume in the mesh is identified by a physical group whose name
/// starts with ``mat:<material_name>``. The *materials* dictionary maps
/// those names to :class:`Material` objects.
///
/// Outer boundary surfaces should be tagged with a physical group named
/// ``boundary:vacuum`` (reflecting boundaries are not yet supported for
/// mesh geometry).
///
/// The implicit complement (unmeshed space between volumes) is void by
/// default -- particles free-stream through it with no collisions.  To
/// model air or coolant interactions in the gaps, pass a :class:`Material`
/// as *implicit_complement_material*.
///
/// A vacuum boundary box can be added automatically by setting
/// *graveyard_offset*.  This adds an axis-aligned bounding box expanded by
/// the given offset (cm) around the geometry, with vacuum boundary
/// conditions on all 6 faces.  Particles reaching this boundary are killed.
///
/// Args:
///     filename: Path to an Arrow IPC mesh file (``.arrow``).
///     materials: Mapping of material name to :class:`Material`.
///     implicit_complement_material: Optional material for the unmeshed
///         space between volumes.  Default ``None`` means void (free-streaming).
///     graveyard_offset: Optional offset (cm) for automatic vacuum boundary
///         box generation.
///
/// Raises:
///     ValueError: If the file is not an Arrow IPC mesh, or if it contains a
///         negatively oriented tetrahedron. Transport reads outward tet face
///         normals off a fixed vertex ordering, so an inverted tet is rejected
///         rather than silently re-wound (issue #316); regenerate the mesh with
///         a writer that emits positively oriented tets.
///
/// Examples:
///     >>> fuel = yamc.Material(composition={"Li6": 1.0}, density=0.534, name="fuel")
///     >>> fuel.read_nuclear_data({"Li6": "tests/Li6.arrow"})
///     >>>
///     >>> mesh_geom = yamc.MeshGeometry("model.arrow", {"fuel": fuel},
///     ...                               graveyard_offset=5.0)
///     >>> model = yamc.Model(geometry=mesh_geom, source=source)
#[gen_stub_pyclass]
#[pyclass(
    module = "yamc._core",
    name = "MeshGeometry",
    unsendable,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMeshGeometry {
    pub inner: MeshGeometry,
    /// Stored Arrow mesh data for rebuild when graveyard_offset changes.
    arrow_data: yamt::ArrowMeshData,
    /// Material map stored for rebuild.
    mat_map: HashMap<String, Arc<yamc_materials::material::Material>>,
    /// Implicit complement material stored for rebuild.
    ic_mat: Option<Arc<yamc_materials::material::Material>>,
    /// Current graveyard offset.
    graveyard_offset: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMeshGeometry {
    #[new]
    #[pyo3(signature = (filename, materials, implicit_complement_material=None, graveyard_offset=None))]
    pub fn new<'py>(
        filename: &str,
        materials: &Bound<'py, PyDict>,
        implicit_complement_material: Option<&Bound<'py, PyMaterial>>,
        graveyard_offset: Option<f64>,
    ) -> PyResult<Self> {
        // Mirror PyGeometry::new: auto-assign material IDs to any Material
        // that doesn't have one yet, mutating the user's Python objects in
        // place. This means subsequent calls like
        // `mesh_geom.bounding_box_for_material(li_mat)` see the assigned ID
        // without the user having to call `set_material_id` manually.
        //
        // Two passes over the dict: first collect already-assigned IDs (and
        // dedup the Materials that need IDs by Python object identity, in
        // case the user maps two names to the same Material), then assign
        // the next free integer ID to each one.
        let mut used_material_ids: HashSet<u32> = HashSet::new();
        let mut seen_ptrs: HashSet<usize> = HashSet::new();
        let mut materials_needing_ids: Vec<Bound<'py, PyMaterial>> = Vec::new();

        // Helper closure: process one PyMaterial reference (from the dict
        // or the implicit-complement option). Records its ID if set, or
        // schedules it for assignment if not -- deduped by pointer.
        let mut visit = |py_mat: &Bound<'py, PyMaterial>| -> PyResult<()> {
            let ptr = py_mat.as_ptr() as usize;
            if !seen_ptrs.insert(ptr) {
                return Ok(());
            }
            let mat_borrow = py_mat.try_borrow()?;
            match mat_borrow.internal.get_material_id() {
                Some(id) => {
                    if !used_material_ids.insert(id) {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Duplicate Material id {id} found. All material IDs must be unique."
                        )));
                    }
                }
                None => {
                    drop(mat_borrow);
                    materials_needing_ids.push(py_mat.clone());
                }
            }
            Ok(())
        };

        for (_key, value) in materials.iter() {
            let py_mat: Bound<'py, PyMaterial> = value.cast_into()?;
            visit(&py_mat)?;
        }
        if let Some(ic) = implicit_complement_material {
            visit(ic)?;
        }

        let mut next_material_id = 1u32;
        for py_mat in &materials_needing_ids {
            while used_material_ids.contains(&next_material_id) {
                next_material_id += 1;
            }
            py_mat
                .try_borrow_mut()?
                .internal
                .set_material_id(next_material_id);
            used_material_ids.insert(next_material_id);
            next_material_id += 1;
        }

        // Build the Arc<Material> map now that every Material has an ID.
        let mut mat_map: HashMap<String, Arc<yamc_materials::material::Material>> = HashMap::new();
        for (key, value) in materials.iter() {
            let name: String = key.extract()?;
            let py_mat: Bound<'py, PyMaterial> = value.cast_into()?;
            mat_map.insert(name, Arc::new(py_mat.try_borrow()?.internal.clone()));
        }

        // Convert optional complement material
        let ic_mat = implicit_complement_material
            .map(|py_mat| -> PyResult<_> { Ok(Arc::new(py_mat.try_borrow()?.internal.clone())) })
            .transpose()?;

        // Load the mesh. The Arrow IPC mesh is the only supported format.
        let path = std::path::Path::new(filename);
        if path.extension().and_then(|e| e.to_str()) != Some("arrow") {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported mesh file '{filename}': MeshGeometry reads YAMC Arrow IPC \
                 meshes (.arrow), as written by cad_to_yamc"
            )));
        }

        let mut data = yamt::read_arrow_mesh(path).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load Arrow mesh: {e}"))
        })?;
        let arrow_data = data.clone(); // Store original for rebuild
        if let Some(offset) = graveyard_offset {
            yamt::add_vacuum_boundary(&mut data, offset);
        }
        let topo = yamt::build_topology(data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid mesh '{filename}': {e}"))
        })?;
        let mesh = yamt::MeshGeometry::from_topology(topo);

        let inner = MeshGeometry::new(mesh, &mat_map, ic_mat.clone())
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok(PyMeshGeometry {
            inner,
            arrow_data,
            mat_map,
            ic_mat,
            graveyard_offset,
        })
    }

    /// Number of volumes (cells) in the mesh (excluding implicit complement).
    #[getter]
    pub fn num_volumes(&self) -> u32 {
        self.inner.mesh.topology.num_volumes
    }

    /// Material names for each volume, in volume-ID order.
    ///
    /// These are the names from ``mat:<name>`` physical groups.
    /// Pass any of these strings as the second element of a tally's
    /// ``unstructured_mesh=(mesh, name)`` argument to bin on that volume by name
    /// (an integer volume ID is also accepted).
    #[getter]
    pub fn volume_names(&self) -> Vec<Option<String>> {
        (0..self.inner.mesh.topology.num_volumes)
            .map(|v| self.inner.mesh.material_name(v).map(|s| s.to_owned()))
            .collect()
    }

    /// All vertex coordinates as a list of ``[x, y, z]`` triples.
    #[getter]
    pub fn vertices(&self) -> Vec<[f64; 3]> {
        self.inner.mesh.topology.vertices.clone()
    }

    /// All triangle connectivity as a list of ``[v0, v1, v2]`` triples
    /// (vertex indices).
    #[getter]
    pub fn triangles(&self) -> Vec<[u32; 3]> {
        self.inner.mesh.topology.triangles.clone()
    }

    /// Number of triangles in the entire mesh.
    #[getter]
    pub fn num_triangles(&self) -> usize {
        self.inner.mesh.topology.triangles.len()
    }

    /// Analytical volume of each mesh volume (cm³), computed via the
    /// divergence theorem over the bounding surface triangles.
    #[getter]
    pub fn volume_measures(&self) -> Vec<f64> {
        self.inner.mesh.topology.volume_measures.clone()
    }

    /// Compute the volume of every mesh volume.
    ///
    /// Mesh geometries already know their volumes analytically (via the
    /// divergence theorem applied to their bounding triangles when the
    /// mesh is loaded), so this method is exact: ``std_dev`` is always 0
    /// and no stochastic sampling is performed. The signature matches
    /// :meth:`Geometry.calculate_volume` so the same call works on both
    /// CSG and mesh geometries.
    ///
    /// Args:
    ///     samples: Ignored (kept for signature compatibility).
    ///     bounding_box: Ignored (kept for signature compatibility).
    ///     seed: Ignored (kept for signature compatibility).
    ///
    /// Returns:
    ///     dict[int, VolumeResult]: Mapping volume_id -> VolumeResult.
    #[pyo3(signature = (samples=0, bounding_box=None, seed=None))]
    pub fn calculate_volume(
        &self,
        samples: u64,
        bounding_box: Option<&PyBoundingBox>,
        seed: Option<u64>,
    ) -> HashMap<u32, crate::geometry::PyVolumeResult> {
        let _ = (samples, bounding_box, seed);
        let measures = &self.inner.mesh.topology.volume_measures;
        let mut out = HashMap::with_capacity(measures.len());
        for (i, &v) in measures.iter().enumerate() {
            out.insert(
                i as u32,
                crate::geometry::PyVolumeResult {
                    volume: v,
                    std_dev: 0.0,
                    num_hits: 0,
                },
            );
        }
        out
    }

    /// Triangle indices belonging to a specific volume.
    ///
    /// Returns the global triangle indices for every surface that bounds
    /// the given volume.  These can be used to index into :attr:`triangles`.
    ///
    /// Args:
    ///     volume_id (int): Volume index (0-based).
    ///
    /// Returns:
    ///     list[int]: Triangle indices for that volume.
    fn volume_triangle_indices(&self, volume_id: u32) -> PyResult<Vec<u32>> {
        let topo = &self.inner.mesh.topology;
        if volume_id >= topo.num_volumes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "volume_id {volume_id} out of range (num_volumes={})",
                topo.num_volumes
            )));
        }
        let mut tri_ids = Vec::new();
        for &(surf_id, _sense) in &topo.volume_surfaces[volume_id as usize] {
            let range = &topo.surface_tri_ranges[surf_id as usize];
            for idx in range.clone() {
                tri_ids.push(topo.surface_tri_indices[idx as usize]);
            }
        }
        tri_ids.sort_unstable();
        tri_ids.dedup();
        Ok(tri_ids)
    }

    /// Bounding box of the entire mesh geometry.
    ///
    /// Returns:
    ///     BoundingBox: Axis-aligned bounding box enclosing all mesh volumes.
    pub fn bounding_box(&self) -> PyBoundingBox {
        let bbox = self.inner.bounding_box();
        PyBoundingBox::new(bbox.lower_left, bbox.upper_right)
    }

    /// Bounding box of volumes matching a specific material.
    ///
    /// Args:
    ///     material: A :class:`Material` with an ``id`` set.
    ///
    /// Returns:
    ///     BoundingBox: Axis-aligned bounding box enclosing only the volumes
    ///         that use the given material.
    ///
    /// Raises:
    ///     ValueError: If the material has no ID or no volumes match.
    pub fn bounding_box_for_material(&self, material: &PyMaterial) -> PyResult<PyBoundingBox> {
        let mat_id = material.internal.get_material_id().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Material has no id -- assign one first")
        })?;
        let bbox = self
            .inner
            .bounding_box_for_material(mat_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "No mesh volumes match material id {mat_id}"
                ))
            })?;
        Ok(PyBoundingBox::new(bbox.lower_left, bbox.upper_right))
    }

    /// The vacuum boundary (graveyard) offset in cm.
    ///
    /// Setting this property triggers a full rebuild (re-reads stored mesh
    /// data, adds the vacuum boundary box, and rebuilds the BVH).  Set to
    /// ``None`` to remove the boundary.
    #[getter]
    pub fn get_graveyard_offset(&self) -> Option<f64> {
        self.graveyard_offset
    }

    #[setter]
    pub fn set_graveyard_offset(&mut self, offset: Option<f64>) -> PyResult<()> {
        let mut data = self.arrow_data.clone();
        if let Some(off) = offset {
            yamt::add_vacuum_boundary(&mut data, off);
        }
        let topo = yamt::build_topology(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid mesh: {e}")))?;
        let mesh = yamt::MeshGeometry::from_topology(topo);

        self.inner = MeshGeometry::new(mesh, &self.mat_map, self.ic_mat.clone())
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        self.graveyard_offset = offset;
        Ok(())
    }

    /// Generate an interactive 2D geometry viewer as a self-contained HTML page.
    ///
    /// Produces an interactive viewer where you can switch slice planes (xy/xz/yz),
    /// pan with mouse drag, zoom with scroll wheel, and adjust all parameters
    /// in real time.
    ///
    /// Args:
    ///     origin: Origin of the plot (tuple/list of 3 floats). Defaults to bbox center or (0,0,0).
    ///     width: Width of the plot (tuple/list of 2 floats). Defaults to bbox width or (10,10).
    ///     resolution: Raster resolution as a total pixel budget (int) or an explicit (h, v) tuple. Defaults to 40000.
    ///     basis: The plane to slice - "xy", "xz", or "yz". Defaults to "xy".
    ///     color_by: What to color by - "cell" or "material". Defaults to "cell".
    ///     outline: Add outline around regions - "material", "cell", or None. Defaults to "cell".
    ///     axis_units: Units for axis labels - "mm", "cm", "m", or "km". Defaults to "cm".
    ///     colors: Dictionary mapping integer IDs to color strings. Defaults to None.
    ///     font_size: Font scale for PNG axis labels (1=tiny, 2=normal, 3=large).
    ///         Defaults to 2.
    ///     contour_kwargs: Outline appearance options. Supported keys:
    ///         ``"colors"`` (hex string, default ``"#000000"``),
    ///         ``"linewidths"`` (int, default ``1``).
    ///
    /// Returns:
    ///     InteractivePlot: Result object that renders in Jupyter and supports
    ///         ``.save("file.html")`` and ``.save("file.png")``.
    #[gen_stub(skip)] // FIXME(stub): pyo3-stub-gen mishandles Option<&str> default; hand-add in stub patch-merge
    #[pyo3(signature = (origin=None, width=None, resolution=None, basis="xy", color_by="cell", outline="cell", axis_units="cm", colors=None, font_size=2, contour_kwargs=None))]
    pub fn plot(
        &self,
        origin: Option<pyo3::Py<pyo3::PyAny>>,
        width: Option<pyo3::Py<pyo3::PyAny>>,
        resolution: Option<pyo3::Py<pyo3::PyAny>>,
        basis: &str,
        color_by: &str,
        outline: Option<&str>,
        axis_units: &str,
        #[allow(unused_variables)] colors: Option<pyo3::prelude::Bound<'_, pyo3::types::PyDict>>,
        font_size: usize,
        contour_kwargs: Option<pyo3::prelude::Bound<'_, pyo3::types::PyDict>>,
        py: pyo3::Python<'_>,
    ) -> PyResult<crate::ui::PyInteractivePlot> {
        use crate::ui::{
            build_interactive_html, parse_contour_kwargs, InteractiveViewParams, PyInteractivePlot,
            SourceParams,
        };
        use yamc::geometry::conversion::mesh_geometry_to_geo_mesh;

        let contour = parse_contour_kwargs(contour_kwargs.as_ref())?;
        let source = SourceParams::default();
        let bbox = self.inner.bounding_box();
        let params = crate::id_map_helper::parse_sample_slice_params(
            origin, width, resolution, basis, &bbox, py,
        )?;

        let geo_mesh = mesh_geometry_to_geo_mesh(&self.inner);
        let geometry_json = serde_json::to_string(&geo_mesh)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut cell_names = std::collections::HashMap::new();
        let mut material_names = std::collections::HashMap::new();
        for vol in &geo_mesh.volumes {
            cell_names.insert(vol.cell_id, format!("Volume {}", vol.cell_id));
            if vol.material_id != -1 {
                if let Some(ref name) = vol.material_name {
                    material_names.insert(vol.material_id, name.clone());
                }
            }
        }

        let view_params = InteractiveViewParams {
            origin: params.origin,
            width: params.width,
            total_pixels: params.pixels.0 * params.pixels.1,
            basis: params.basis.clone(),
            color_by: color_by.to_string(),
            outline: outline.map(|s| s.to_string()),
            axis_units: axis_units.to_string(),
            bbox_widths: self.inner.bounding_box().width(),
            outline_color: contour.colors.clone(),
            outline_thickness: contour.linewidths,
            n_samples: None,
            plane_tolerance: 1.0,
            source_color: source.color.clone(),
            source_size: source.size,
        };

        // Pre-sample the initial view with native Rust (BVH)
        let presampled = {
            use crate::id_map_helper::{cell_to_ids, generate_presampled_grid};
            let mesh = &self.inner;
            generate_presampled_grid(&params, |point| {
                mesh.find_cell_index(point)
                    .map(|idx| cell_to_ids(&mesh.cells[idx], &mesh.materials))
                    .unwrap_or((-1, -1))
            })
        };

        let html = build_interactive_html(
            &geometry_json,
            "mesh",
            "",
            &view_params,
            &cell_names,
            &material_names,
            None,
            None,
            Some(&presampled),
            // A pure mesh geometry has no CSG fills; the WASM mesh sampler
            // resolves its volumes directly (issue #291).
            false,
            // Mesh geometries don't carry CSG surfaces to hover over.
            None,
        );

        Ok(PyInteractivePlot::new(
            html,
            &presampled,
            color_by,
            outline,
            None,
            params.origin,
            params.width,
            basis,
            axis_units,
            font_size,
            &contour.colors,
            contour.linewidths,
            None,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "MeshGeometry(volumes={}, triangles={})",
            self.inner.mesh.topology.num_volumes,
            self.inner.mesh.topology.triangles.len(),
        )
    }
}
