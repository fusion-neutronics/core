impl PyRegionExpr {
    pub fn to_region_expr(&self) -> yamc::geo::RegionExpr {
        match self {
            PyRegionExpr::Halfspace(hs) => {
                // Convert PyHalfspace to HalfspaceType using the existing Arc
                Python::attach(|py| {
                    let surface = hs.surface.bind(py);
                    let arc_surface = surface.borrow().inner.clone(); // This now clones the Arc, not the Surface
                    if hs.is_above {
                        yamc::geo::RegionExpr::Halfspace(yamc::geo::HalfspaceType::Above(
                            arc_surface,
                        ))
                    } else {
                        yamc::geo::RegionExpr::Halfspace(yamc::geo::HalfspaceType::Below(
                            arc_surface,
                        ))
                    }
                })
            }
            PyRegionExpr::Union(a, b) => yamc::geo::RegionExpr::Union(
                Box::new(a.to_region_expr()),
                Box::new(b.to_region_expr()),
            ),
            PyRegionExpr::Intersection(a, b) => yamc::geo::RegionExpr::Intersection(
                Box::new(a.to_region_expr()),
                Box::new(b.to_region_expr()),
            ),
            PyRegionExpr::Complement(inner) => {
                yamc::geo::RegionExpr::Complement(Box::new(inner.to_region_expr()))
            }
        }
    }
}

use crate::geometry::PyBoundingBox;
use crate::geometry::PySurface;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// A CSG region -- half-spaces combined with boolean operators
/// (`&` intersection, `|` union, `~` complement) -- used to define a `Cell`.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "Region", from_py_object)]
#[derive(Clone)]
pub struct PyRegion {
    pub expr: PyRegionExpr,
    pub region: yamc::geo::Region,
}

#[derive(Clone)]
pub enum PyRegionExpr {
    Halfspace(PyHalfspace),
    Union(Box<PyRegionExpr>, Box<PyRegionExpr>),
    Intersection(Box<PyRegionExpr>, Box<PyRegionExpr>),
    Complement(Box<PyRegionExpr>),
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRegion {
    pub fn __repr__(&self) -> String {
        fn fmt_expr(e: &PyRegionExpr) -> String {
            match e {
                PyRegionExpr::Halfspace(hs) => {
                    let sign = if hs.is_above { "+" } else { "-" };
                    Python::attach(|py| {
                        let surface = hs.surface.bind(py);
                        let s = surface.borrow();
                        let id = s
                            .inner
                            .surface_id
                            .map_or("?".to_string(), |id| id.to_string());
                        format!("{sign}S{id}")
                    })
                }
                PyRegionExpr::Union(a, b) => {
                    format!("({} | {})", fmt_expr(a), fmt_expr(b))
                }
                PyRegionExpr::Intersection(a, b) => {
                    format!("({} & {})", fmt_expr(a), fmt_expr(b))
                }
                PyRegionExpr::Complement(inner) => {
                    format!("~{}", fmt_expr(inner))
                }
            }
        }
        format!("Region({})", fmt_expr(&self.expr))
    }

    fn __invert__(self_: PyRef<'_, Self>) -> PyResult<Self> {
        let expr = PyRegionExpr::Complement(Box::new(self_.expr.clone()));
        let region = yamc::geo::Region {
            expr: expr.to_region_expr(),
        };
        Ok(PyRegion { expr, region })
    }

    /// Check if a point is inside this region.
    ///
    /// Args:
    ///     point: Tuple of 3 floats (x, y, z)
    ///
    /// Returns:
    ///     True if the point satisfies the region's Boolean expression
    pub fn contains(&self, point: (f64, f64, f64)) -> bool {
        self.region.contains(point)
    }

    /// Get the axis-aligned bounding box for this region.
    ///
    /// Returns:
    ///     BoundingBox object containing the region
    pub fn bounding_box(&self) -> PyBoundingBox {
        use yamc::geo::BoundingBox;
        let bbox: BoundingBox = self.region.bounding_box();
        PyBoundingBox::new(bbox.lower_left, bbox.upper_right)
    }

    /// Estimate the volume of this region by stochastic sampling.
    ///
    /// Args:
    ///     samples: Number of random points to sample (default 100,000)
    ///     bounding_box: Optional BoundingBox to sample within. Defaults to the region's bounding box.
    ///     seed: Optional RNG seed for reproducibility (default 1)
    ///
    /// Returns:
    ///     VolumeResult with volume, std_dev, and num_hits
    #[pyo3(signature = (samples=100_000, bounding_box=None, seed=None))]
    pub fn calculate_volume(
        &self,
        samples: u64,
        bounding_box: Option<&PyBoundingBox>,
        seed: Option<u64>,
    ) -> pyo3::PyResult<crate::geometry::PyVolumeResult> {
        let region = &self.region;
        crate::geometry::compute_single_volume(
            samples,
            bounding_box,
            seed,
            region.bounding_box(),
            "Region",
            |point| region.contains(point),
        )
    }

    /// Generate a 2D map showing which points are inside the region.
    ///
    /// Args:
    ///     origin: Origin of the plot (tuple/list of 3 floats). Defaults to bbox center or (0,0,0).
    ///     width: Width of the plot (tuple/list of 2 floats). Defaults to bbox width or (10,10).
    ///     resolution: Raster resolution as a total pixel budget (int) or an explicit (h, v) tuple. Defaults to 40000.
    ///     basis: The plane to slice - "xy", "xz", or "yz". Defaults to "xy".
    ///
    /// Returns:
    ///     A 2D list of ints where 1 means inside, -1 means outside.
    #[pyo3(signature = (origin=None, width=None, resolution=None, basis="xy"))]
    pub fn sample_slice(
        &self,
        origin: Option<pyo3::Py<pyo3::PyAny>>,
        width: Option<pyo3::Py<pyo3::PyAny>>,
        resolution: Option<pyo3::Py<pyo3::PyAny>>,
        basis: &str,
        py: pyo3::Python<'_>,
    ) -> pyo3::PyResult<Vec<Vec<i32>>> {
        use crate::id_map_helper::compute_sample_slice;

        let region = &self.region;
        let result = compute_sample_slice(
            origin,
            width,
            resolution,
            basis,
            &region.bounding_box(),
            py,
            |point| {
                if region.contains(point) {
                    (1, -1)
                } else {
                    (-1, -1)
                }
            },
        )?;
        Ok(result.cell_ids)
    }

    /// Rasterize this region onto a 3D voxel grid (the compiled core of
    /// ``Region.to_vtkhdf``).
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

        let region = &self.region;
        let params = VoxelParams {
            lower_left,
            spacing,
            shape: [shape.0, shape.1, shape.2],
        };
        generate_voxel_mask(&params, |point| region.contains(point))
    }

    /// Generate an interactive 2D plot of this region.
    ///
    /// Creates a temporary Cell + Geometry and delegates to Geometry.plot().
    ///
    /// Args:
    ///     origin: Origin of the plot (tuple/list of 3 floats). Defaults to bbox center or (0,0,0).
    ///     width: Width of the plot (tuple/list of 2 floats). Defaults to bbox width or (10,10).
    ///     resolution: Raster resolution as a total pixel budget (int) or an explicit (h, v) tuple. Defaults to 40000.
    ///     basis: The plane to slice - "xy", "xz", or "yz". Defaults to "xy".
    ///     outline: Add outline around regions - "cell" or None. Defaults to "cell".
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
    #[pyo3(signature = (origin=None, width=None, resolution=None, basis="xy", outline="cell", axis_units="cm", colors=None, font_size=2, contour_kwargs=None))]
    pub fn plot(
        &self,
        origin: Option<pyo3::Py<pyo3::PyAny>>,
        width: Option<pyo3::Py<pyo3::PyAny>>,
        resolution: Option<pyo3::Py<pyo3::PyAny>>,
        basis: &str,
        outline: Option<&str>,
        axis_units: &str,
        colors: Option<Bound<'_, PyDict>>,
        font_size: usize,
        contour_kwargs: Option<Bound<'_, PyDict>>,
        py: pyo3::Python<'_>,
    ) -> pyo3::PyResult<crate::ui::PyInteractivePlot> {
        use crate::geometry::PyGeometry;
        // Build a temporary Cell + Geometry and delegate
        let rust_cell = yamc::geometry::cell::Cell::new(Some(1), self.region.clone(), None, None);
        let geometry = yamc::geometry::Geometry::new(vec![rust_cell], Vec::new())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let py_geom = PyGeometry { inner: geometry };
        py_geom.plot(
            origin,
            width,
            resolution,
            basis,
            "cell",
            outline,
            axis_units,
            colors,
            font_size,
            contour_kwargs,
            py,
        )
    }

    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRegion> {
        let expr = if let Ok(other_region) = other.extract::<PyRef<PyRegion>>() {
            PyRegionExpr::Intersection(
                Box::new(self.expr.clone()),
                Box::new(other_region.expr.clone()),
            )
        } else if let Ok(other_halfspace) = other.extract::<PyRef<PyHalfspace>>() {
            PyRegionExpr::Intersection(
                Box::new(self.expr.clone()),
                Box::new(PyRegionExpr::Halfspace(other_halfspace.clone())),
            )
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Operand must be PyRegion or PyHalfspace",
            ));
        };
        let region = yamc::geo::Region {
            expr: expr.to_region_expr(),
        };
        Ok(PyRegion { expr, region })
    }

    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRegion> {
        let expr = if let Ok(other_region) = other.extract::<PyRef<PyRegion>>() {
            PyRegionExpr::Union(
                Box::new(self.expr.clone()),
                Box::new(other_region.expr.clone()),
            )
        } else if let Ok(other_halfspace) = other.extract::<PyRef<PyHalfspace>>() {
            PyRegionExpr::Union(
                Box::new(self.expr.clone()),
                Box::new(PyRegionExpr::Halfspace(other_halfspace.clone())),
            )
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Operand must be PyRegion or PyHalfspace",
            ));
        };
        let region = yamc::geo::Region {
            expr: expr.to_region_expr(),
        };
        Ok(PyRegion { expr, region })
    }
}

/// One side of a `Surface` (positive or negative half-space); the atom from
/// which `Region`s are composed via boolean operators.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "Halfspace", from_py_object)]
pub struct PyHalfspace {
    pub surface: Py<PySurface>,
    pub is_above: bool,
}

impl Clone for PyHalfspace {
    fn clone(&self) -> Self {
        Python::attach(|py| PyHalfspace {
            surface: self.surface.clone_ref(py),
            is_above: self.is_above,
        })
    }
}

impl PyHalfspace {
    pub fn new_above(surface: Py<PySurface>) -> Self {
        PyHalfspace {
            surface,
            is_above: true,
        }
    }
    pub fn new_below(surface: Py<PySurface>) -> Self {
        PyHalfspace {
            surface,
            is_above: false,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHalfspace {
    pub fn __repr__(&self) -> String {
        let sign = if self.is_above { "+" } else { "-" };
        Python::attach(|py| {
            let surface = self.surface.bind(py);
            let s = surface.borrow();
            let id = s
                .inner
                .surface_id
                .map_or("?".to_string(), |id| id.to_string());
            format!("Halfspace({sign}S{id})")
        })
    }

    fn __invert__(slf: PyRef<'_, Self>) -> PyResult<PyRegion> {
        let expr = PyRegionExpr::Complement(Box::new(PyRegionExpr::Halfspace(slf.clone())));
        let region = yamc::geo::Region {
            expr: expr.to_region_expr(),
        };
        Ok(PyRegion { expr, region })
    }
    pub fn contains(&self, point: (f64, f64, f64)) -> bool {
        Python::attach(|py| {
            let surface = self.surface.bind(py);
            if self.is_above {
                surface.borrow().inner.evaluate(point) > 0.0
            } else {
                surface.borrow().inner.evaluate(point) < 0.0
            }
        })
    }

    pub fn bounding_box(&self) -> PyBoundingBox {
        Python::attach(|py| {
            let surface = self.surface.bind(py);
            let bbox = surface.borrow().inner.halfspace_bounding_box(self.is_above);
            PyBoundingBox::new(bbox.lower_left, bbox.upper_right)
        })
    }

    /// Rasterize this halfspace onto a 3D voxel grid (the compiled core of
    /// ``Halfspace.to_vtkhdf``).
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

        Python::attach(|py| {
            let surface = self.surface.bind(py);
            let s = surface.borrow();
            let is_above = self.is_above;
            let params = VoxelParams {
                lower_left,
                spacing,
                shape: [shape.0, shape.1, shape.2],
            };
            generate_voxel_mask(&params, |point| {
                let v = s.inner.evaluate(point);
                if is_above {
                    v > 0.0
                } else {
                    v < 0.0
                }
            })
        })
    }

    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRegion> {
        let expr = if let Ok(other_halfspace) = other.extract::<PyRef<PyHalfspace>>() {
            PyRegionExpr::Intersection(
                Box::new(PyRegionExpr::Halfspace(self.clone())),
                Box::new(PyRegionExpr::Halfspace(other_halfspace.clone())),
            )
        } else if let Ok(other_region) = other.extract::<PyRef<PyRegion>>() {
            PyRegionExpr::Intersection(
                Box::new(PyRegionExpr::Halfspace(self.clone())),
                Box::new(other_region.expr.clone()),
            )
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Operand must be PyRegion or PyHalfspace",
            ));
        };
        let region = yamc::geo::Region {
            expr: expr.to_region_expr(),
        };
        Ok(PyRegion { expr, region })
    }
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRegion> {
        let expr = if let Ok(other_halfspace) = other.extract::<PyRef<PyHalfspace>>() {
            PyRegionExpr::Union(
                Box::new(PyRegionExpr::Halfspace(self.clone())),
                Box::new(PyRegionExpr::Halfspace(other_halfspace.clone())),
            )
        } else if let Ok(other_region) = other.extract::<PyRef<PyRegion>>() {
            PyRegionExpr::Union(
                Box::new(PyRegionExpr::Halfspace(self.clone())),
                Box::new(other_region.expr.clone()),
            )
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Operand must be PyRegion or PyHalfspace",
            ));
        };
        let region = yamc::geo::Region {
            expr: expr.to_region_expr(),
        };
        Ok(PyRegion { expr, region })
    }
}
