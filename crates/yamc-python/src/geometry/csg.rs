use crate::geometry::PyCell;
use crate::material::PyMaterial;
use std::collections::{HashMap, HashSet};
use yamc::geometry;

use crate::geometry::PyBoundingBox;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Result of a stochastic volume calculation: the estimated `volume`, its
/// `std_dev`, and the number of `num_hits` sampled inside the cell.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "VolumeResult", from_py_object)]
#[derive(Clone)]
pub struct PyVolumeResult {
    #[pyo3(get)]
    pub volume: f64,
    #[pyo3(get)]
    pub std_dev: f64,
    #[pyo3(get)]
    pub num_hits: u64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVolumeResult {
    fn __repr__(&self) -> String {
        format!(
            "VolumeResult(volume={}, std_dev={}, num_hits={})",
            self.volume, self.std_dev, self.num_hits
        )
    }
}

/// Compute a single-bin stochastic volume estimate.
///
/// Resolves the bounding box (user-provided or `default_bbox`), runs
/// [`calculate_stochastic_volumes`] with the given classifier, and
/// returns a [`PyVolumeResult`].
pub fn compute_single_volume(
    samples: u64,
    bounding_box: Option<&PyBoundingBox>,
    seed: Option<u64>,
    default_bbox: yamc::geo::BoundingBox,
    label: &str,
    contains: impl Fn((f64, f64, f64)) -> bool + Sync,
) -> pyo3::PyResult<PyVolumeResult> {
    let bbox = match bounding_box {
        Some(py_bbox) => yamc::geo::BoundingBox::new(py_bbox.lower_left(), py_bbox.upper_right()),
        None => {
            if !default_bbox.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{} bounding box is infinite. Provide an explicit bounding_box.",
                    label
                )));
            }
            default_bbox
        }
    };
    let seed = seed.unwrap_or(1);
    let results =
        yamc::stochastic_volume::calculate_stochastic_volumes(samples, &bbox, seed, 1, |point| {
            if contains(point) {
                Some(0)
            } else {
                None
            }
        });
    let r = &results[0];
    Ok(PyVolumeResult {
        volume: r.volume,
        std_dev: r.std_dev,
        num_hits: r.num_hits,
    })
}

/// A CSG geometry: a list of cells defining the simulation domain.
///
/// A Geometry collects :class:`Cell` objects into a complete spatial
/// description for transport. Cell and material IDs are auto-assigned
/// where the user did not set them. The geometry validates that surface
/// IDs (and material IDs) are unique on construction.
///
/// For mesh-based geometry (an Arrow IPC mesh, e.g. from CAD), use
/// :class:`MeshGeometry` instead.
///
/// Examples:
///     >>> import yamc
///     >>> sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
///     >>> iron = yamc.Material(composition={"Fe": 1.0}, density=7.874)
///     >>> cell = yamc.Cell(region=sphere.below, material=iron)
///     >>> geometry = yamc.Geometry([cell])
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "Geometry", from_py_object)]
#[derive(Clone)]
pub struct PyGeometry {
    pub inner: geometry::Geometry,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGeometry {
    #[new]
    pub fn new(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Sequence[Cell]"))] cells: &Bound<
            '_,
            pyo3::types::PyAny,
        >,
    ) -> pyo3::PyResult<Self> {
        let seq = cells.cast::<PySequence>()?;
        let len = seq.len()?;

        // Collect references to original PyCell objects
        let mut py_cell_refs: Vec<Bound<'_, PyCell>> = Vec::with_capacity(len);

        for i in 0..len {
            let item = seq.get_item(i)?;
            let pycell_ref: Bound<'_, PyCell> = item.cast_into()?;
            py_cell_refs.push(pycell_ref);
        }

        // Step 1: Auto-assign cell IDs to cells that don't have them
        let mut used_cell_ids: HashSet<u32> = HashSet::new();
        let mut cells_needing_ids: Vec<usize> = Vec::new();

        for (index, py_cell_ref) in py_cell_refs.iter().enumerate() {
            let cell = py_cell_ref.try_borrow()?;
            // Cell views returned by geometry.cells / find_cell on a
            // mesh-filled geometry do not carry the fill; rebuilding a
            // geometry from them would silently drop the mesh bodies.
            #[cfg(feature = "mesh")]
            if cell.inner.in_mesh_fill() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cells taken from a mesh-filled geometry cannot seed a new \
                     Geometry: the mesh fill is not carried by the cell views. \
                     Construct a fresh Cell(region=..., fill=...) instead",
                ));
            }
            match cell.inner.cell_id {
                None => cells_needing_ids.push(index),
                Some(id) => {
                    if used_cell_ids.contains(&id) {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Duplicate Cell id {} found. All cell IDs must be unique.",
                            id
                        )));
                    }
                    used_cell_ids.insert(id);
                }
            }
        }

        // Generate IDs for cells that need them
        let mut next_cell_id = 1u32;
        for &index in &cells_needing_ids {
            while used_cell_ids.contains(&next_cell_id) {
                next_cell_id += 1;
            }
            py_cell_refs[index]
                .try_borrow_mut()?
                .inner
                .set_cell_id(next_cell_id);
            used_cell_ids.insert(next_cell_id);
            next_cell_id += 1;
        }

        // Step 2: Auto-assign material IDs to materials that don't have them
        // Track unique materials by their Python object identity (pointer address)
        let mut used_material_ids: HashSet<u32> = HashSet::new();
        // Set of already-seen material pointers (to avoid processing same material twice)
        let mut seen_material_ptrs: HashSet<usize> = HashSet::new();
        // Materials that need ID assignment
        let mut materials_needing_ids: Vec<Py<PyMaterial>> = Vec::new();

        for py_cell_ref in py_cell_refs.iter() {
            let cell = py_cell_ref.try_borrow()?;
            if let Some(py_mat_ref) = &cell.py_material {
                let ptr = py_mat_ref.as_ptr() as usize;

                // Skip if we've already processed this Python Material object
                if !seen_material_ptrs.insert(ptr) {
                    continue;
                }

                let mat = py_mat_ref.bind(py).try_borrow()?;
                match mat.internal.get_material_id() {
                    Some(id) => {
                        if used_material_ids.contains(&id) {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                format!("Duplicate Material id {} found. All material IDs must be unique across all cells.", id)
                            ));
                        }
                        used_material_ids.insert(id);
                    }
                    None => {
                        materials_needing_ids.push(py_mat_ref.clone_ref(py));
                    }
                }
            }
        }

        // Mesh-fill materials already carry ids (assigned at MeshGeometry
        // construction); reserve them so the auto-assignment below cannot
        // collide. A shared Python Material naturally keeps one id.
        #[cfg(feature = "mesh")]
        for py_cell_ref in py_cell_refs.iter() {
            let cell = py_cell_ref.try_borrow()?;
            if let Some(py_fill) = &cell.py_fill {
                let fill = py_fill.bind(py).try_borrow()?;
                for mat in &fill.inner.materials {
                    if let Some(id) = mat.get_material_id() {
                        used_material_ids.insert(id);
                    }
                }
            }
        }

        // Generate IDs for materials that need them
        let mut next_material_id = 1u32;
        for py_mat_ref in materials_needing_ids {
            while used_material_ids.contains(&next_material_id) {
                next_material_id += 1;
            }
            // Update the Python Material object
            py_mat_ref
                .bind(py)
                .try_borrow_mut()?
                .internal
                .set_material_id(next_material_id);
            used_material_ids.insert(next_material_id);
            next_material_id += 1;
        }

        // Step 3: Collect unique materials into a flat Vec (dedup by Python object
        // pointer so shared Python Materials share a single slot) and assign each
        // cell the corresponding `material_idx`.
        let mut material_slot: std::collections::HashMap<usize, u32> =
            std::collections::HashMap::new();
        let mut materials: Vec<std::sync::Arc<yamc_materials::material::Material>> = Vec::new();

        let mut rust_cells = Vec::with_capacity(len);
        #[cfg(feature = "mesh")]
        let mut fill_specs: Vec<yamc::geometry::fill::CellFillSpec> = Vec::new();
        for (host_cell_index, py_cell_ref) in py_cell_refs.iter().enumerate() {
            let cell = py_cell_ref.try_borrow()?;
            #[cfg(not(feature = "mesh"))]
            let _ = host_cell_index;
            #[cfg(feature = "mesh")]
            if let Some(py_fill) = &cell.py_fill {
                let fill = py_fill.bind(py).try_borrow()?;
                if fill.get_graveyard_offset().is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "a MeshGeometry with graveyard_offset cannot fill a cell: \
                         the graveyard box is a standalone-transport boundary, and \
                         the cell's CSG region already bounds the fill",
                    ));
                }
                fill_specs.push(yamc::geometry::fill::CellFillSpec {
                    host_cell_index,
                    mesh_geometry: fill.inner.clone(),
                    translation: cell.fill_translation.unwrap_or([0.0; 3]),
                    rotation_degrees: cell.fill_rotation.unwrap_or([0.0; 3]),
                    allow_clipping: cell.fill_allow_clipping,
                });
            }
            let material_idx = match cell.py_material.as_ref() {
                Some(py_mat_ref) => {
                    let ptr = py_mat_ref.as_ptr() as usize;
                    let slot = match material_slot.get(&ptr) {
                        Some(&slot) => slot,
                        None => {
                            // Borrow with `?` so a re-entrant BorrowError surfaces
                            // as a clean PyResult instead of panicking.
                            let arc = std::sync::Arc::new(
                                py_mat_ref.bind(py).try_borrow()?.internal.clone(),
                            );
                            let slot = materials.len() as u32;
                            materials.push(arc);
                            material_slot.insert(ptr, slot);
                            slot
                        }
                    };
                    Some(slot)
                }
                None => None,
            };
            let rust_cell = yamc::geometry::cell::Cell::new(
                cell.inner.cell_id,
                cell.inner.region.clone(),
                cell.inner.name.clone(),
                material_idx,
            );
            rust_cells.push(rust_cell);
        }

        // Create geometry (validation only, no auto-assignment needed).
        // Cells with a mesh fill route through new_with_fills, which
        // expands the embedded mesh volumes into cells and validates the
        // fill (watertightness, protrusion unless allow_clipping).
        #[cfg(feature = "mesh")]
        let geometry = if fill_specs.is_empty() {
            geometry::Geometry::new(rust_cells, materials)
        } else {
            geometry::Geometry::new_with_fills(rust_cells, materials, fill_specs)
        }
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        #[cfg(not(feature = "mesh"))]
        let geometry = geometry::Geometry::new(rust_cells, materials)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        Ok(PyGeometry { inner: geometry })
    }

    pub fn __repr__(&self) -> String {
        format!("Geometry(cells={})", self.inner.cells.len())
    }

    /// Find the cell containing a given point in space.
    ///
    /// Args:
    ///     x: X coordinate
    ///     y: Y coordinate
    ///     z: Z coordinate
    ///
    /// Returns:
    ///     Cell object if found, None if the point is outside all cells
    pub fn find_cell(
        &self,
        x: f64,
        y: f64,
        z: f64,
        py: Python<'_>,
    ) -> pyo3::PyResult<Option<PyCell>> {
        let Some(cell) = self.inner.find_cell((x, y, z)).cloned() else {
            return Ok(None);
        };
        let py_material = if let Some(idx) = cell.material_idx {
            self.inner
                .materials
                .get(idx as usize)
                .map(|arc| {
                    Py::new(
                        py,
                        PyMaterial {
                            internal: arc.as_ref().clone(),
                        },
                    )
                })
                .transpose()?
        } else {
            None
        };
        Ok(Some(PyCell::from_parts(cell, py_material)))
    }

    /// Get the axis-aligned bounding box containing all cells in the geometry.
    ///
    /// Returns:
    ///     BoundingBox object with lower_left and upper_right coordinates
    pub fn bounding_box(&self) -> PyBoundingBox {
        use yamc::geo::BoundingBox;
        let bbox: BoundingBox = self.inner.bounding_box();
        PyBoundingBox::new(bbox.lower_left, bbox.upper_right)
    }

    /// List of all cells in the geometry.
    ///
    /// Returns:
    ///     List of Cell objects
    #[getter]
    pub fn cells(&self, py: Python<'_>) -> pyo3::PyResult<Vec<PyCell>> {
        // Reconstruct each cell's Python-visible material from the flat
        // `materials` store. Each call creates a fresh `PyMaterial` cloned
        // from the Arc'd Material; object identity with the caller's original
        // Python Material is not preserved, but `.id` / `.volume` / etc. are.
        let mut out = Vec::with_capacity(self.inner.cells.len());
        for cell in &self.inner.cells {
            let py_material = if let Some(idx) = cell.material_idx {
                self.inner
                    .materials
                    .get(idx as usize)
                    .map(|arc| {
                        Py::new(
                            py,
                            PyMaterial {
                                internal: arc.as_ref().clone(),
                            },
                        )
                    })
                    .transpose()?
            } else {
                None
            };
            out.push(PyCell::from_parts(cell.clone(), py_material));
        }
        Ok(out)
    }

    /// Estimate volumes of all cells by stochastic sampling.
    ///
    /// Populates each cell's `volume` attribute with the estimated value.
    ///
    /// Args:
    ///     samples: Number of random points to sample (default 100,000)
    ///     bounding_box: Optional BoundingBox to sample within. Defaults to the geometry's bounding box.
    ///     seed: Optional RNG seed for reproducibility (default 1)
    ///
    /// Returns:
    ///     Dictionary mapping cell id -> VolumeResult
    #[pyo3(signature = (samples=100_000, bounding_box=None, seed=None))]
    pub fn calculate_volume(
        &mut self,
        samples: u64,
        bounding_box: Option<&PyBoundingBox>,
        seed: Option<u64>,
    ) -> pyo3::PyResult<HashMap<u32, PyVolumeResult>> {
        let bbox = match bounding_box {
            Some(py_bbox) => {
                yamc::geo::BoundingBox::new(py_bbox.lower_left(), py_bbox.upper_right())
            }
            None => {
                let b = self.inner.bounding_box();
                if !b.is_finite() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Geometry bounding box is infinite. Provide an explicit bounding_box.",
                    ));
                }
                b
            }
        };

        let seed = seed.unwrap_or(1);
        let n_bins = self.inner.cells.len();
        let geometry = &self.inner;

        let results = yamc::stochastic_volume::calculate_stochastic_volumes(
            samples,
            &bbox,
            seed,
            n_bins,
            |point| geometry.find_cell_index(point),
        );

        // Check for transmutable materials shared across multiple cells.
        // Each transmutable material needs a unique volume for tally
        // normalization, so sharing one material_id across cells is ambiguous.
        let mut transmutable_mat_cells: HashMap<u32, Vec<u32>> = HashMap::new();
        for cell in self.inner.cells.iter() {
            if let Some(mat) = cell
                .material_idx
                .and_then(|i| self.inner.materials.get(i as usize))
            {
                if mat.transmutable {
                    if let (Some(mat_id), Some(cell_id)) = (mat.material_id, cell.cell_id) {
                        transmutable_mat_cells
                            .entry(mat_id)
                            .or_default()
                            .push(cell_id);
                    }
                }
            }
        }
        for (mat_id, cell_ids) in &transmutable_mat_cells {
            if cell_ids.len() > 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Transmutable material {} is used in {} cells ({:?}). \
                     Cannot automatically assign volume. \
                     Use a unique material per cell for transmutable materials.",
                    mat_id,
                    cell_ids.len(),
                    cell_ids,
                )));
            }
        }

        let mut output = HashMap::new();
        for (i, result) in results.iter().enumerate() {
            self.inner.cells[i].volume = Some(result.volume);
            // Propagate volume to the material inside the cell so that
            // transmutation can read material.volume without the user having
            // to set it manually.
            if let Some(slot) = self.inner.cells[i].material_idx {
                let slot = slot as usize;
                let mut mat = (*self.inner.materials[slot]).clone();
                mat.volume = Some(result.volume);
                self.inner.materials[slot] = std::sync::Arc::new(mat);
            }
            if let Some(cell_id) = self.inner.cells[i].cell_id {
                output.insert(
                    cell_id,
                    PyVolumeResult {
                        volume: result.volume,
                        std_dev: result.std_dev,
                        num_hits: result.num_hits,
                    },
                );
            }
        }

        Ok(output)
    }

    /// Generate 2D maps of cell IDs and material IDs for a slice through the geometry.
    ///
    /// Args:
    ///     origin: Origin of the plot (tuple of 3 floats). If unspecified, defaults to
    ///         the center of the bounding box, or (0.0, 0.0, 0.0) if bounding box contains inf.
    ///     width: Width of the plot in each basis direction (tuple of 2 floats). If unspecified,
    ///         defaults to the width of the bounding box, or (10.0, 10.0) if bounding box contains inf.
    ///     resolution: Raster resolution as a total pixel budget (int, default 40000) or an explicit (h, v) tuple.
    ///         If a single int, resolution in each direction are calculated from the aspect ratio.
    ///     basis: The plane to slice - "xy", "xz", or "yz". Defaults to "xy".
    ///
    /// Returns:
    ///     A :class:`GeometrySliceData` with ``cell_ids``, ``material_ids``,
    ///     ``h_edges``, ``v_edges``, ``extent``, ``h_label``, ``v_label``, and
    ///     an ``edges()`` method.  Also supports tuple unpacking:
    ///     ``cell_ids, material_ids = geometry.sample_slice(...)``.
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

        let geometry = &self.inner;
        compute_sample_slice(
            origin,
            width,
            resolution,
            basis,
            &geometry.bounding_box(),
            py,
            |point| {
                geometry
                    .find_cell(point)
                    .map(|c| cell_to_ids(c, &geometry.materials))
                    .unwrap_or((-1, -1))
            },
        )
    }

    /// Rasterize this geometry onto a 3D voxel grid (the compiled core of
    /// ``Geometry.to_vtkhdf``).
    ///
    /// Args:
    ///     lower_left: Grid minimum corner ``(x, y, z)``.
    ///     spacing: Voxel size ``(dx, dy, dz)``.
    ///     shape: Voxel counts ``(nx, ny, nz)``.
    ///
    /// Returns:
    ///     VoxelData with flat ``cell_ids`` / ``material_ids`` in (nz, ny, nx)
    ///     C-order (``-1`` where no cell is present).
    #[pyo3(signature = (lower_left, spacing, shape))]
    pub fn voxelize(
        &self,
        lower_left: [f64; 3],
        spacing: [f64; 3],
        shape: (usize, usize, usize),
    ) -> crate::id_map_helper::PyVoxelData {
        use crate::id_map_helper::{cell_to_ids, generate_voxel_grid, PyVoxelData, VoxelParams};

        let geometry = &self.inner;
        let params = VoxelParams {
            lower_left,
            spacing,
            shape: [shape.0, shape.1, shape.2],
        };
        let (cell_ids, material_ids) = generate_voxel_grid(&params, |point| {
            geometry
                .find_cell(point)
                .map(|c| cell_to_ids(c, &geometry.materials))
                .unwrap_or((-1, -1))
        });
        PyVoxelData {
            cell_ids,
            material_ids,
            shape,
        }
    }

    /// Generate an interactive 2D geometry viewer as a self-contained HTML page.
    ///
    /// Produces an interactive viewer where you can switch slice planes (xy/xz/yz),
    /// pan with mouse drag, zoom with scroll wheel, and adjust all parameters
    /// in real time. The geometry is re-sampled in the browser using a JavaScript
    /// CSG engine.
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
        use crate::id_map_helper::parse_sample_slice_params;
        use crate::ui::parse_colors_ids_cells_materials;
        use crate::ui::{
            build_interactive_html, parse_contour_kwargs, InteractiveViewParams, PyInteractivePlot,
            SourceParams,
        };
        use yamc::geometry::conversion::geometry_to_csg;

        let color_map = parse_colors_ids_cells_materials(colors)?;
        let contour = parse_contour_kwargs(contour_kwargs.as_ref())?;
        let source = SourceParams::default();
        let bbox = self.inner.bounding_box();
        let params = parse_sample_slice_params(origin, width, resolution, basis, &bbox, py)?;

        // Convert geometry to serializable form
        let csg = geometry_to_csg(&self.inner);
        let geometry_json = serde_json::to_string(&csg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Build name maps
        let mut cell_names = std::collections::HashMap::new();
        let mut material_names = std::collections::HashMap::new();
        for cell in &csg.cells {
            if let Some(ref name) = cell.name {
                cell_names.insert(cell.cell_id, name.clone());
            }
            if cell.material_id != -1 {
                if let Some(ref name) = cell.material_name {
                    material_names.insert(cell.material_id, name.clone());
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
            bbox_widths: bbox.width(),
            outline_color: contour.colors.clone(),
            outline_thickness: contour.linewidths,
            n_samples: None,
            plane_tolerance: 1.0,
            source_color: source.color.clone(),
            source_size: source.size,
        };

        // Pre-sample the initial view with native Rust
        let presampled = {
            use crate::id_map_helper::{cell_to_ids, generate_presampled_grid};
            let geometry = &self.inner;
            generate_presampled_grid(&params, |point| {
                geometry
                    .find_cell(point)
                    .map(|c| cell_to_ids(c, &geometry.materials))
                    .unwrap_or((-1, -1))
            })
        };

        // Skip embedding the pre-sample for CSG -- the browser sampler (Patch 2
        // worker pool) is fast enough to render the initial view directly,
        // and the ~3-4 MB base64-encoded grid was the dominant file-size cost.
        // We still generate `presampled` above because PyInteractivePlot::new
        // uses it for PNG rendering.
        //
        // A mesh-filled cell is the exception (issue #291): the browser sampler
        // cannot resolve a fill body from `geometry_json` (fills serialize as an
        // identity fingerprint), so it would draw the bare CSG frame. Ship the
        // fill-aware raster and let `viewer.js` refuse to re-sample.
        let has_mesh_fills = !self.inner.fills.is_empty();
        let presampled_for_html = if has_mesh_fills {
            Some(&presampled)
        } else {
            None
        };
        let csg = yamc::geometry::conversion::geometry_to_csg(&self.inner);
        let surface_table = yamc_plot::build_surface_table(&csg);
        let html = build_interactive_html(
            &geometry_json,
            "csg",
            "",
            &view_params,
            &cell_names,
            &material_names,
            color_map.as_ref(),
            None,
            presampled_for_html,
            has_mesh_fills,
            Some(&surface_table),
        );

        Ok(PyInteractivePlot::new(
            html,
            &presampled,
            color_by,
            outline,
            color_map.as_ref(),
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
}
