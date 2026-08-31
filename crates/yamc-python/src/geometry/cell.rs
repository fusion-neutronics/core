use crate::geometry::PyRegion;
use crate::material::PyMaterial;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc::geometry::cell::Cell;

/// A region of space filled with an optional material.
///
/// A Cell pairs a CSG :class:`Region` (built from surfaces via the
/// boolean operators ``&``, ``|``, ``~`` and the halfspace operators
/// ``+``, ``-``) with an optional :class:`Material`. Cells with no
/// material are void -- particles free-stream through them with no
/// interactions. Each cell may carry an optional integer id and string
/// name; ids are auto-assigned when the cell is added to a
/// :class:`Geometry` if the user did not set one.
///
/// A cell may also be *filled* by a :class:`MeshGeometry` (a
/// triangulated CAD body): inside the cell's region a particle is
/// either inside one of the mesh volumes (that volume's material) or in
/// the gap (the cell's own ``material``, which acts as the complement).
/// ``translation`` and ``rotation`` place the body in the CSG frame.
///
/// Examples:
///     >>> import yamc
///     >>> sphere = yamc.Sphere(radius=10.0)
///     >>> iron = yamc.Material(composition={"Fe": 1.0}, density=7.874)
///     >>> cell = yamc.Cell(region=sphere.below, material=iron, name="iron_ball")
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "Cell", from_py_object)]
pub struct PyCell {
    pub inner: Cell,
    /// Reference to the original Python Material object (for ID propagation)
    pub py_material: Option<Py<PyMaterial>>,
    /// Mesh body filling this cell (issue #232), consumed by
    /// PyGeometry::new like py_material.
    #[cfg(feature = "mesh")]
    pub py_fill: Option<Py<crate::geometry::PyMeshGeometry>>,
    /// Placement of the fill body in the CSG frame, in cm.
    #[cfg(feature = "mesh")]
    pub fill_translation: Option<[f64; 3]>,
    /// Rotation of the fill body about the CSG-frame x, y, z axes in
    /// degrees, applied in that order before the translation.
    #[cfg(feature = "mesh")]
    pub fill_rotation: Option<[f64; 3]>,
    /// Permit the fill body to protrude past the region (it is clipped
    /// by the CSG surface).
    #[cfg(feature = "mesh")]
    pub fill_allow_clipping: bool,
}

impl PyCell {
    /// Wrap an existing Rust cell (no fill: reconstructed views of a
    /// geometry's cells carry the fill on the geometry itself).
    pub(crate) fn from_parts(inner: Cell, py_material: Option<Py<PyMaterial>>) -> Self {
        PyCell {
            inner,
            py_material,
            #[cfg(feature = "mesh")]
            py_fill: None,
            #[cfg(feature = "mesh")]
            fill_translation: None,
            #[cfg(feature = "mesh")]
            fill_rotation: None,
            #[cfg(feature = "mesh")]
            fill_allow_clipping: false,
        }
    }
}

impl Clone for PyCell {
    fn clone(&self) -> Self {
        Python::attach(|py| PyCell {
            inner: self.inner.clone(),
            py_material: self.py_material.as_ref().map(|p| p.clone_ref(py)),
            #[cfg(feature = "mesh")]
            py_fill: self.py_fill.as_ref().map(|p| p.clone_ref(py)),
            #[cfg(feature = "mesh")]
            fill_translation: self.fill_translation,
            #[cfg(feature = "mesh")]
            fill_rotation: self.fill_rotation,
            #[cfg(feature = "mesh")]
            fill_allow_clipping: self.fill_allow_clipping,
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCell {
    pub fn __repr__(&self) -> String {
        let id = self
            .inner
            .cell_id
            .map_or("None".to_string(), |id| id.to_string());
        let name = self.inner.name.as_deref().unwrap_or("None");
        let has_material = self.py_material.is_some();
        format!("Cell(id={id}, name={name}, material={has_material})")
    }

    /// Compute the distance to the closest surface from a point in a direction (if any)
    pub fn distance_to_surface(
        &self,
        point: (f64, f64, f64),
        direction: (f64, f64, f64),
    ) -> Option<f64> {
        let point_arr = [point.0, point.1, point.2];
        let dir_arr = [direction.0, direction.1, direction.2];
        self.inner.distance_to_surface(point_arr, dir_arr)
    }

    /// Return the boundary type and distance to the closest surface, or None if no intersection
    pub fn closest_surface(
        &self,
        point: (f64, f64, f64),
        direction: (f64, f64, f64),
    ) -> Option<(String, f64)> {
        let point_arr = [point.0, point.1, point.2];
        let dir_arr = [direction.0, direction.1, direction.2];
        if let Some((boundary, dist, _surface_id)) = self.inner.closest_surface(point_arr, dir_arr)
        {
            return Some((format!("{:?}", boundary), dist));
        }
        None
    }
    /// Create a cell from a region, with an optional material and an
    /// optional mesh fill.
    ///
    /// Args:
    ///     region: CSG region bounding the cell.
    ///     id: Optional unique integer id (auto-assigned by Geometry).
    ///     name: Optional name label.
    ///     material: Material for the cell. When ``fill`` is given this
    ///         is the complement, filling the region minus the mesh body.
    ///     volume: Optional known volume in cm^3.
    ///     fill: Optional :class:`MeshGeometry` body embedded in the
    ///         cell. Its volumes keep their own materials; the cell's
    ///         CSG surfaces still bound everything.
    ///     translation: Optional (x, y, z) placement of the fill body in
    ///         the CSG frame, in cm.
    ///     rotation: Optional (rx, ry, rz) rotation of the fill body in
    ///         degrees about the CSG-frame x, then y, then z axes,
    ///         applied before the translation.
    ///     allow_clipping: Permit the fill body to protrude past the
    ///         region; the protruding part is clipped by the CSG surface.
    ///         Without it, a protruding fill is a construction error.
    #[new]
    #[pyo3(signature = (region, id=None, name=None, material=None, volume=None, fill=None, translation=None, rotation=None, allow_clipping=false))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _py: Python<'_>,
        region: PyRegion,
        id: Option<u32>,
        name: Option<String>,
        material: Option<Py<PyMaterial>>,
        volume: Option<f64>,
        #[gen_stub(override_type(type_repr = "MeshGeometry | None"))] fill: Option<
            &Bound<'_, PyAny>,
        >,
        translation: Option<(f64, f64, f64)>,
        rotation: Option<(f64, f64, f64)>,
        allow_clipping: bool,
    ) -> PyResult<Self> {
        #[cfg(feature = "mesh")]
        let py_fill: Option<Py<crate::geometry::PyMeshGeometry>> = match fill {
            Some(obj) => Some(obj.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("fill must be a MeshGeometry")
            })?),
            None => None,
        };
        #[cfg(not(feature = "mesh"))]
        if fill.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fill requires yamc built with the 'mesh' feature",
            ));
        }
        if fill.is_none() && (translation.is_some() || rotation.is_some() || allow_clipping) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "translation, rotation and allow_clipping only apply to a cell \
                 with a fill",
            ));
        }
        // The Cell itself only carries a `material_idx` assigned when it's
        // placed into a Geometry; keep the original PyMaterial reference so
        // PyGeometry::new can dedupe and flatten materials at construction.
        let mut inner = Cell::new(id, region.region, name, None);
        inner.volume = volume;
        Ok(PyCell {
            inner,
            py_material: material,
            #[cfg(feature = "mesh")]
            py_fill,
            #[cfg(feature = "mesh")]
            fill_translation: translation.map(|t| [t.0, t.1, t.2]),
            #[cfg(feature = "mesh")]
            fill_rotation: rotation.map(|r| [r.0, r.1, r.2]),
            #[cfg(feature = "mesh")]
            fill_allow_clipping: allow_clipping,
        })
    }

    /// Unique numeric identifier for the cell.
    ///
    /// Returns:
    ///     Integer ID or None if not set
    #[getter]
    pub fn id(&self) -> Option<u32> {
        self.inner.cell_id
    }

    #[setter(id)]
    pub fn set_id(&mut self, id: u32) {
        self.inner.set_cell_id(id);
    }

    /// Volume of the cell in cm^3.
    ///
    /// Returns:
    ///     Float volume or None if not set
    #[getter]
    pub fn volume(&self) -> Option<f64> {
        self.inner.volume
    }

    #[setter]
    pub fn set_volume(&mut self, volume: Option<f64>) {
        self.inner.volume = volume;
    }

    /// Optional name for the cell.
    ///
    /// Returns:
    ///     String name or None
    #[getter]
    pub fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// Set the cell's name label.
    #[setter(name)]
    pub fn set_name(&mut self, name: Option<String>) {
        self.inner.name = name;
    }

    /// Material assigned to this cell.
    ///
    /// Returns:
    ///     Material object or None for void cells
    #[getter]
    pub fn material(&self, py: Python<'_>) -> Option<PyMaterial> {
        self.py_material
            .as_ref()
            .map(|py_mat| py_mat.bind(py).borrow().clone())
    }

    /// Mesh body filling this cell.
    ///
    /// Returns:
    ///     MeshGeometry or None for an ordinary cell
    // Type-erased signature: the stub macro resolves signature types
    // outside the method's cfg, so naming PyMeshGeometry here breaks
    // non-mesh builds. The returned object IS the MeshGeometry.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "MeshGeometry | None"))]
    pub fn fill(&self, py: Python<'_>) -> Option<Py<pyo3::PyAny>> {
        #[cfg(feature = "mesh")]
        {
            self.py_fill.as_ref().map(|p| p.clone_ref(py).into_any())
        }
        #[cfg(not(feature = "mesh"))]
        {
            let _ = py;
            None
        }
    }

    /// Placement of the fill body in the CSG frame, in cm.
    ///
    /// Returns:
    ///     (x, y, z) tuple or None
    #[cfg(feature = "mesh")]
    #[getter]
    pub fn translation(&self) -> Option<(f64, f64, f64)> {
        self.fill_translation.map(|t| (t[0], t[1], t[2]))
    }

    /// Rotation of the fill body in degrees about the CSG-frame x, y, z
    /// axes, applied in that order before the translation.
    ///
    /// Returns:
    ///     (rx, ry, rz) tuple or None
    #[cfg(feature = "mesh")]
    #[getter]
    pub fn rotation(&self) -> Option<(f64, f64, f64)> {
        self.fill_rotation.map(|r| (r[0], r[1], r[2]))
    }

    /// Whether the fill body may protrude past the region (clipped by
    /// the CSG surface).
    #[cfg(feature = "mesh")]
    #[getter]
    pub fn allow_clipping(&self) -> bool {
        self.fill_allow_clipping
    }

    /// Get the axis-aligned bounding box for this cell's region.
    ///
    /// Returns:
    ///     BoundingBox object with lower_left and upper_right coordinates
    pub fn bounding_box(&self) -> crate::geometry::PyBoundingBox {
        let bbox = self.inner.region.bounding_box();
        crate::geometry::PyBoundingBox::new(bbox.lower_left, bbox.upper_right)
    }

    /// Check if a point is inside this cell.
    ///
    /// Args:
    ///     x: X coordinate
    ///     y: Y coordinate
    ///     z: Z coordinate
    ///
    /// Returns:
    ///     True if the point is inside the cell, False otherwise
    pub fn contains(&self, x: f64, y: f64, z: f64) -> bool {
        self.inner.contains((x, y, z))
    }

    /// Estimate the volume of this cell by stochastic sampling.
    ///
    /// Sets `cell.volume` to the estimated value and returns a VolumeResult.
    /// For a cell with a mesh ``fill`` the estimate covers the complement
    /// only (the region minus the placed body), which is the space the
    /// cell's own material occupies.
    ///
    /// Args:
    ///     samples: Number of random points to sample (default 100,000)
    ///     bounding_box: Optional BoundingBox to sample within. Defaults to the cell's region bounding box.
    ///     seed: Optional RNG seed for reproducibility (default 1)
    ///
    /// Returns:
    ///     VolumeResult with volume, std_dev, and num_hits
    #[pyo3(signature = (samples=100_000, bounding_box=None, seed=None))]
    pub fn calculate_volume(
        &mut self,
        py: Python<'_>,
        samples: u64,
        bounding_box: Option<&crate::geometry::PyBoundingBox>,
        seed: Option<u64>,
    ) -> pyo3::PyResult<crate::geometry::PyVolumeResult> {
        let cell = &self.inner;
        // Points inside the placed fill body do not belong to this
        // cell's material. The mesh is cloned once here because the
        // rayon classifier needs a Sync view, not a Python borrow.
        #[cfg(feature = "mesh")]
        let fill_view = match &self.py_fill {
            Some(py_fill) => Some(yamc::geometry::fill::MeshFill::for_queries(
                std::sync::Arc::new(py_fill.bind(py).try_borrow()?.inner.mesh.clone()),
                self.fill_translation.unwrap_or([0.0; 3]),
                self.fill_rotation.unwrap_or([0.0; 3]),
            )),
            None => None,
        };
        #[cfg(not(feature = "mesh"))]
        let _ = py;
        let result = crate::geometry::compute_single_volume(
            samples,
            bounding_box,
            seed,
            cell.region.bounding_box(),
            "Cell",
            |point| {
                if !cell.contains(point) {
                    return false;
                }
                #[cfg(feature = "mesh")]
                if let Some(fv) = &fill_view {
                    return fv.find_volume_world([point.0, point.1, point.2])
                        == fv.mesh.topology.implicit_complement;
                }
                true
            },
        )?;
        self.inner.volume = Some(result.volume);
        Ok(result)
    }

    /// Generate 2D maps of cell ID and material ID for this cell.
    ///
    /// A mesh ``fill`` is not resolved here (the whole region maps to
    /// the cell's own ids); use ``Geometry.sample_slice`` to see the
    /// fill bodies.
    ///
    /// Args:
    ///     origin: Origin of the plot (tuple/list of 3 floats). Defaults to bbox center or (0,0,0).
    ///     width: Width of the plot (tuple/list of 2 floats). Defaults to bbox width or (10,10).
    ///     resolution: Raster resolution as a total pixel budget (int) or an explicit (h, v) tuple. Defaults to 40000.
    ///     basis: The plane to slice - "xy", "xz", or "yz". Defaults to "xy".
    ///
    /// Returns:
    ///     A :class:`GeometrySliceData` -- also supports tuple unpacking.
    #[pyo3(signature = (origin=None, width=None, resolution=None, basis="xy"))]
    pub fn sample_slice(
        &self,
        origin: Option<pyo3::Py<pyo3::PyAny>>,
        width: Option<pyo3::Py<pyo3::PyAny>>,
        resolution: Option<pyo3::Py<pyo3::PyAny>>,
        basis: &str,
        py: pyo3::Python<'_>,
    ) -> pyo3::PyResult<crate::id_map_helper::PyGeometrySliceData> {
        use crate::id_map_helper::{cell_to_ids, compute_sample_slice};

        let cell = &self.inner;
        // PyCell isn't placed in a Geometry here; construct a transient
        // single-material slice view on the fly.
        let materials: Vec<std::sync::Arc<yamc_materials::material::Material>> =
            match &self.py_material {
                Some(py_mat) => vec![std::sync::Arc::new(
                    py_mat.bind(py).borrow().internal.clone(),
                )],
                None => Vec::new(),
            };
        let cell_view = yamc::geometry::cell::Cell::new(
            cell.cell_id,
            cell.region.clone(),
            cell.name.clone(),
            if materials.is_empty() { None } else { Some(0) },
        );
        compute_sample_slice(
            origin,
            width,
            resolution,
            basis,
            &cell.region.bounding_box(),
            py,
            |point| {
                if cell_view.contains(point) {
                    cell_to_ids(&cell_view, &materials)
                } else {
                    (-1, -1)
                }
            },
        )
    }

    /// Rasterize this cell onto a 3D voxel grid (the compiled core of
    /// ``Cell.to_vtkhdf``).
    ///
    /// The caller resolves the cell / material id; this returns only the
    /// inside/outside mask so the hot ``contains`` loop runs in Rust.
    ///
    /// Args:
    ///     lower_left: Grid minimum corner ``(x, y, z)``.
    ///     spacing: Voxel size ``(dx, dy, dz)``.
    ///     shape: Voxel counts ``(nx, ny, nz)``.
    ///
    /// Returns:
    ///     Flat mask in (nz, ny, nx) C-order: 1 inside, 0 outside.
    #[pyo3(signature = (lower_left, spacing, shape))]
    pub fn voxelize(
        &self,
        lower_left: [f64; 3],
        spacing: [f64; 3],
        shape: (usize, usize, usize),
    ) -> Vec<i32> {
        use crate::id_map_helper::{generate_voxel_mask, VoxelParams};

        let cell = &self.inner;
        let params = VoxelParams {
            lower_left,
            spacing,
            shape: [shape.0, shape.1, shape.2],
        };
        generate_voxel_mask(&params, |point| cell.contains(point))
    }

    /// Generate an interactive 2D plot of this cell.
    ///
    /// Creates a single-cell Geometry and delegates to Geometry.plot().
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
        colors: Option<Bound<'_, PyDict>>,
        font_size: usize,
        contour_kwargs: Option<Bound<'_, PyDict>>,
        py: pyo3::Python<'_>,
    ) -> pyo3::PyResult<crate::ui::PyInteractivePlot> {
        use crate::geometry::PyGeometry;
        // Build a single-cell Geometry and delegate. Materialise the cell's
        // PyMaterial into the flat Vec that Geometry::new now expects.
        let cell_id = self.inner.cell_id.unwrap_or(1);
        let (material_idx, materials) = if let Some(py_mat) = &self.py_material {
            let arc = std::sync::Arc::new(py_mat.bind(py).borrow().internal.clone());
            (Some(0u32), vec![arc])
        } else {
            (None, Vec::new())
        };
        let rust_cell = yamc::geometry::cell::Cell::new(
            Some(cell_id),
            self.inner.region.clone(),
            self.inner.name.clone(),
            material_idx,
        );
        let geometry = yamc::geometry::Geometry::new(vec![rust_cell], materials)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let py_geom = PyGeometry { inner: geometry };
        py_geom.plot(
            origin,
            width,
            resolution,
            basis,
            color_by,
            outline,
            axis_units,
            colors,
            font_size,
            contour_kwargs,
            py,
        )
    }
}
