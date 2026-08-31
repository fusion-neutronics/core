use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc::geo::BoundingBox;

/// Map a basis string to axis indices and labels.
///
/// Returns `(h_idx, v_idx, fixed_idx, h_label, v_label, fixed_label)`.
pub fn basis_axes(
    basis: &str,
) -> Result<
    (
        usize,
        usize,
        usize,
        &'static str,
        &'static str,
        &'static str,
    ),
    String,
> {
    match basis {
        "xy" => Ok((0, 1, 2, "x", "y", "z")),
        "xz" => Ok((0, 2, 1, "x", "z", "y")),
        "yz" => Ok((1, 2, 0, "y", "z", "x")),
        _ => Err(format!("basis must be 'xy', 'xz', or 'yz', got '{basis}'")),
    }
}

/// Compute (n+1)-length edge arrays from `IdMapParams` for matplotlib pcolormesh.
pub fn compute_edge_arrays(params: &IdMapParams) -> (Vec<f64>, Vec<f64>) {
    let (h_idx, v_idx, _, _, _, _) = basis_axes(&params.basis).unwrap();
    let origin_arr = [params.origin.0, params.origin.1, params.origin.2];
    let h_min = origin_arr[h_idx] - params.width.0 / 2.0;
    let h_max = origin_arr[h_idx] + params.width.0 / 2.0;
    let v_min = origin_arr[v_idx] - params.width.1 / 2.0;
    let v_max = origin_arr[v_idx] + params.width.1 / 2.0;
    let ph = params.pixels.0;
    let pv = params.pixels.1;
    let h_edges: Vec<f64> = (0..=ph)
        .map(|i| h_min + (h_max - h_min) * (i as f64) / (ph as f64))
        .collect();
    let v_edges: Vec<f64> = (0..=pv)
        .map(|i| v_min + (v_max - v_min) * (i as f64) / (pv as f64))
        .collect();
    (h_edges, v_edges)
}

/// Detect boundaries in a 2D ID grid.
/// Returns a boolean grid where `true` means the pixel borders a different ID.
pub fn detect_edges(id_grid: &[Vec<i32>]) -> Vec<Vec<bool>> {
    let pv = id_grid.len();
    if pv == 0 {
        return vec![];
    }
    let ph = id_grid[0].len();
    let mut edges = vec![vec![false; ph]; pv];
    for i in 0..pv {
        for j in 0..ph {
            let id = id_grid[i][j];
            if (i > 0 && id_grid[i - 1][j] != id)
                || (i + 1 < pv && id_grid[i + 1][j] != id)
                || (j > 0 && id_grid[i][j - 1] != id)
                || (j + 1 < ph && id_grid[i][j + 1] != id)
            {
                edges[i][j] = true;
            }
        }
    }
    edges
}

/// Rich return type for `sample_slice()` on Geometry, Cell, and Model.
///
/// Behaves like a ``(cell_ids, material_ids)`` tuple for backward compatibility
/// (supports tuple unpacking), while also exposing coordinate metadata and an
/// ``edges()`` method for matplotlib outline overlays.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "GeometrySliceData")]
pub struct PyGeometrySliceData {
    #[pyo3(get)]
    pub cell_ids: Vec<Vec<i32>>,
    #[pyo3(get)]
    pub material_ids: Vec<Vec<i32>>,
    #[pyo3(get)]
    pub h_edges: Vec<f64>,
    #[pyo3(get)]
    pub v_edges: Vec<f64>,
    #[pyo3(get)]
    pub extent: (f64, f64, f64, f64),
    #[pyo3(get)]
    pub h_label: String,
    #[pyo3(get)]
    pub v_label: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGeometrySliceData {
    /// Compute edge-detection mask for cell or material boundaries.
    ///
    /// Args:
    ///     color_by: ``"cell"`` or ``"material"``. Defaults to ``"material"``.
    ///
    /// Returns:
    ///     2D list of bools -- ``True`` at boundary pixels.
    #[pyo3(signature = (color_by="material"))]
    pub fn edges(&self, color_by: &str) -> Vec<Vec<bool>> {
        let grid = match color_by {
            "cell" => &self.cell_ids,
            _ => &self.material_ids,
        };
        detect_edges(grid)
    }

    /// Backward-compatible tuple unpacking: ``cell_ids, mat_ids = geom.sample_slice(...)``.
    fn __len__(&self) -> usize {
        2
    }

    fn __getitem__(&self, idx: isize) -> PyResult<Vec<Vec<i32>>> {
        match idx {
            0 | -2 => Ok(self.cell_ids.clone()),
            1 | -1 => Ok(self.material_ids.clone()),
            _ => Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            )),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>, py: pyo3::Python<'_>) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        use pyo3::types::PyList;
        let items = PyList::new(py, [slf.cell_ids.clone(), slf.material_ids.clone()])?;
        items.call_method0("__iter__").map(|it| it.into())
    }

    fn __repr__(&self) -> String {
        let pv = self.cell_ids.len();
        let ph = if pv > 0 { self.cell_ids[0].len() } else { 0 };
        format!(
            "GeometrySliceData(pixels=({ph}, {pv}), h_label='{}', v_label='{}')",
            self.h_label, self.v_label
        )
    }
}

/// Shared parameters for sample_slice functions
pub struct IdMapParams {
    pub origin: (f64, f64, f64),
    pub width: (f64, f64),
    pub pixels: (usize, usize),
    pub basis: String,
}

// The PresampledGrid struct moved into yamc-plot so the browser-side
// editor (which can't depend on yamc-python) can use it too. Re-exported
// here so existing imports `use crate::id_map_helper::PresampledGrid;`
// continue to resolve unchanged.
pub use yamc_plot::PresampledGrid;

/// Parse the sample_slice parameters, applying defaults from bounding box
pub fn parse_sample_slice_params(
    origin: Option<pyo3::Py<pyo3::PyAny>>,
    width: Option<pyo3::Py<pyo3::PyAny>>,
    pixels: Option<pyo3::Py<pyo3::PyAny>>,
    basis: &str,
    bbox: &BoundingBox,
    py: pyo3::Python<'_>,
) -> PyResult<IdMapParams> {
    let (h_idx, v_idx, _, _, _, _) =
        basis_axes(basis).map_err(pyo3::exceptions::PyValueError::new_err)?;

    // Determine origin - default to bbox center or (0,0,0)
    let center = bbox.center();
    let origin: (f64, f64, f64) = if let Some(origin_val) = origin {
        let seq: Vec<f64> = origin_val.extract(py).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "origin must be a sequence of 3 floats (e.g., tuple or list)",
            )
        })?;
        if seq.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "origin must have exactly 3 elements",
            ));
        }
        (seq[0], seq[1], seq[2])
    } else if center[0].is_finite() && center[1].is_finite() && center[2].is_finite() {
        (center[0], center[1], center[2])
    } else {
        (0.0, 0.0, 0.0)
    };

    // Determine width - default to bbox width or (10,10)
    let width: (f64, f64) = if let Some(width_val) = width {
        let seq: Vec<f64> = width_val.extract(py).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "width must be a sequence of 2 floats (e.g., tuple or list)",
            )
        })?;
        if seq.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "width must have exactly 2 elements",
            ));
        }
        (seq[0], seq[1])
    } else {
        let bbox_width = bbox.width();
        let w_h = bbox_width[h_idx];
        let w_v = bbox_width[v_idx];
        if w_h.is_finite() && w_v.is_finite() {
            (w_h, w_v)
        } else {
            (10.0, 10.0)
        }
    };

    // Handle pixels - default to 40000
    let pixels_tuple: (usize, usize) = if let Some(pixels_val) = pixels {
        if let Ok(total) = pixels_val.extract::<usize>(py) {
            let aspect_ratio = width.0 / width.1;
            let pixels_v = (total as f64 / aspect_ratio).sqrt();
            let pixels_h = (total as f64 / pixels_v) as usize;
            (pixels_h, pixels_v as usize)
        } else if let Ok(tuple) = pixels_val.extract::<(usize, usize)>(py) {
            tuple
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "pixels must be an int or a tuple of two ints",
            ));
        }
    } else {
        let total = 40000usize;
        let aspect_ratio = width.0 / width.1;
        let pixels_v = (total as f64 / aspect_ratio).sqrt();
        let pixels_h = (total as f64 / pixels_v) as usize;
        (pixels_h, pixels_v as usize)
    };

    Ok(IdMapParams {
        origin,
        width,
        pixels: pixels_tuple,
        basis: basis.to_string(),
    })
}

/// Extract (cell_id, material_id) from a Cell reference and the flat material
/// store that owns it.
pub fn cell_to_ids(
    cell: &yamc::geometry::cell::Cell,
    materials: &[std::sync::Arc<yamc_materials::material::Material>],
) -> (i32, i32) {
    let cell_id = cell.cell_id.map(|id| id as i32).unwrap_or(-1);
    let material_id = cell
        .material_idx
        .and_then(|i| materials.get(i as usize))
        .and_then(|m| m.get_material_id())
        .map(|id| id as i32)
        .unwrap_or(-1);
    (cell_id, material_id)
}

/// Compute a sample_slice: parse params, sample the grid, return rich `GeometrySliceData`.
pub fn compute_sample_slice(
    origin: Option<pyo3::Py<pyo3::PyAny>>,
    width: Option<pyo3::Py<pyo3::PyAny>>,
    pixels: Option<pyo3::Py<pyo3::PyAny>>,
    basis: &str,
    bbox: &BoundingBox,
    py: pyo3::Python<'_>,
    point_fn: impl FnMut((f64, f64, f64)) -> (i32, i32),
) -> PyResult<PyGeometrySliceData> {
    let params = parse_sample_slice_params(origin, width, pixels, basis, bbox, py)?;
    let (cell_ids, material_ids) = generate_sample_slice(&params, point_fn);
    let (h_edges, v_edges) = compute_edge_arrays(&params);
    let (_, _, _, h_label, v_label, _) = basis_axes(basis).unwrap();
    let extent = (
        h_edges[0],
        *h_edges.last().unwrap(),
        v_edges[0],
        *v_edges.last().unwrap(),
    );
    Ok(PyGeometrySliceData {
        cell_ids,
        material_ids,
        h_edges,
        v_edges,
        extent,
        h_label: h_label.to_string(),
        v_label: v_label.to_string(),
    })
}

/// Generate a sample_slice using the provided callback for each point.
/// The callback takes a 3D point and returns (cell_id, material_id).
pub fn generate_sample_slice<F>(
    params: &IdMapParams,
    mut point_fn: F,
) -> (Vec<Vec<i32>>, Vec<Vec<i32>>)
where
    F: FnMut((f64, f64, f64)) -> (i32, i32),
{
    let (pixels_h, pixels_v) = params.pixels;
    let (width_h, width_v) = params.width;
    let origin = params.origin;

    let half_width_h = width_h / 2.0;
    let half_width_v = width_v / 2.0;

    // Create coordinate arrays centered on origin
    let h_vals: Vec<f64> = (0..pixels_h)
        .map(|i| {
            if pixels_h > 1 {
                -half_width_h + width_h * (i as f64) / ((pixels_h - 1) as f64)
            } else {
                0.0
            }
        })
        .collect();

    let v_vals: Vec<f64> = (0..pixels_v)
        .map(|i| {
            if pixels_v > 1 {
                -half_width_v + width_v * (i as f64) / ((pixels_v - 1) as f64)
            } else {
                0.0
            }
        })
        .collect();

    // Initialize result arrays
    let mut cell_ids = vec![vec![-1i32; pixels_h]; pixels_v];
    let mut material_ids = vec![vec![-1i32; pixels_h]; pixels_v];

    // Axis indices for converting 2D→3D
    let (h_idx, v_idx, _, _, _, _) = basis_axes(&params.basis).unwrap();
    let origin_arr = [origin.0, origin.1, origin.2];

    // Fill the grid
    for (i, &v) in v_vals.iter().enumerate() {
        for (j, &h) in h_vals.iter().enumerate() {
            let mut point = origin_arr;
            point[h_idx] = origin_arr[h_idx] + h;
            point[v_idx] = origin_arr[v_idx] + v;

            let (cell_id, material_id) = point_fn((point[0], point[1], point[2]));
            cell_ids[i][j] = cell_id;
            material_ids[i][j] = material_id;
        }
    }

    (cell_ids, material_ids)
}

/// Convert `(cell_ids, material_ids)` from `generate_sample_slice` to a
/// `PresampledGrid` with flat interleaved data and reversed V-axis (top-to-bottom).
pub fn to_presampled_grid(cell_ids: &[Vec<i32>], material_ids: &[Vec<i32>]) -> PresampledGrid {
    let pv = cell_ids.len();
    let ph = if pv > 0 { cell_ids[0].len() } else { 0 };
    let mut data = vec![-1i32; ph * pv * 2];
    for i in 0..pv {
        // Reverse V: row 0 in PresampledGrid = last row of id_map
        let src_row = pv - 1 - i;
        for j in 0..ph {
            let idx = (i * ph + j) * 2;
            data[idx] = cell_ids[src_row][j];
            data[idx + 1] = material_ids[src_row][j];
        }
    }
    PresampledGrid { data, ph, pv }
}

/// Convenience: generate_sample_slice + to_presampled_grid in one call.
pub fn generate_presampled_grid<F>(params: &IdMapParams, point_fn: F) -> PresampledGrid
where
    F: FnMut((f64, f64, f64)) -> (i32, i32),
{
    let (cell_ids, material_ids) = generate_sample_slice(params, point_fn);
    to_presampled_grid(&cell_ids, &material_ids)
}

// ---------------------------------------------------------------------------
// 3D voxelization (the compiled core of `*.to_vtkhdf` geometry export).
//
// The 3D sibling of `generate_sample_slice`: rasterize a geometry-like object
// onto a regular Cartesian voxel grid. Moving the per-voxel loop into Rust
// removes one Python<->Rust crossing per voxel (nx*ny*nz of them). The h5py
// write stays in Python.
// ---------------------------------------------------------------------------

/// Parameters for a regular 3D voxel grid.
///
/// The center of voxel `(ix, iy, iz)` is `lower_left[k] + (i + 0.5) * spacing[k]`.
pub struct VoxelParams {
    /// Lower-left (minimum) corner of the grid.
    pub lower_left: [f64; 3],
    /// Voxel size along each axis.
    pub spacing: [f64; 3],
    /// Voxel counts along x, y, z.
    pub shape: [usize; 3],
}

/// Flat index for voxel `(ix, iy, iz)` in (nz, ny, nx) C-order, i.e. matching a
/// numpy `reshape(nz, ny, nx)` and the legacy `arr[iz, iy, ix]` layout.
#[inline]
fn voxel_index(ix: usize, iy: usize, iz: usize, nx: usize, ny: usize) -> usize {
    (iz * ny + iy) * nx + ix
}

/// Rasterize a geometry onto a voxel grid, returning flat `(cell_ids,
/// material_ids)` in (nz, ny, nx) C-order (-1 where `point_fn` reports no cell).
pub fn generate_voxel_grid<F>(p: &VoxelParams, mut point_fn: F) -> (Vec<i32>, Vec<i32>)
where
    F: FnMut((f64, f64, f64)) -> (i32, i32),
{
    let [nx, ny, nz] = p.shape;
    let n = nx * ny * nz;
    let mut cell_ids = vec![-1i32; n];
    let mut material_ids = vec![-1i32; n];
    for iz in 0..nz {
        let z = p.lower_left[2] + (iz as f64 + 0.5) * p.spacing[2];
        for iy in 0..ny {
            let y = p.lower_left[1] + (iy as f64 + 0.5) * p.spacing[1];
            for ix in 0..nx {
                let x = p.lower_left[0] + (ix as f64 + 0.5) * p.spacing[0];
                let (c, m) = point_fn((x, y, z));
                let idx = voxel_index(ix, iy, iz, nx, ny);
                cell_ids[idx] = c;
                material_ids[idx] = m;
            }
        }
    }
    (cell_ids, material_ids)
}

/// Rasterize an inside/outside predicate onto a voxel grid, returning a flat
/// mask in (nz, ny, nx) C-order: 1 inside, 0 outside.
pub fn generate_voxel_mask<F>(p: &VoxelParams, mut inside_fn: F) -> Vec<i32>
where
    F: FnMut((f64, f64, f64)) -> bool,
{
    let [nx, ny, nz] = p.shape;
    let mut mask = vec![0i32; nx * ny * nz];
    for iz in 0..nz {
        let z = p.lower_left[2] + (iz as f64 + 0.5) * p.spacing[2];
        for iy in 0..ny {
            let y = p.lower_left[1] + (iy as f64 + 0.5) * p.spacing[1];
            for ix in 0..nx {
                let x = p.lower_left[0] + (ix as f64 + 0.5) * p.spacing[0];
                if inside_fn((x, y, z)) {
                    mask[voxel_index(ix, iy, iz, nx, ny)] = 1;
                }
            }
        }
    }
    mask
}

/// Result of `Geometry.voxelize`: flat cell/material id volumes.
///
/// `cell_ids` and `material_ids` are in (nz, ny, nx) C-order, so
/// `numpy.asarray(v.cell_ids).reshape(nz, ny, nx)` gives the voxel grid with
/// `-1` where no cell is present.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "VoxelData")]
pub struct PyVoxelData {
    #[pyo3(get)]
    pub cell_ids: Vec<i32>,
    #[pyo3(get)]
    pub material_ids: Vec<i32>,
    /// Grid shape as (nx, ny, nz).
    #[pyo3(get)]
    pub shape: (usize, usize, usize),
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVoxelData {
    fn __repr__(&self) -> String {
        format!(
            "VoxelData(shape=({}, {}, {}))",
            self.shape.0, self.shape.1, self.shape.2
        )
    }
}
