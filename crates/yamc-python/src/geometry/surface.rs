#![allow(non_snake_case)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
// ...existing code...

use crate::geometry::PyBoundingBox;
use crate::geometry::PyHalfspace;
use yamc::geo::{BoundaryType, Surface};

fn parse_boundary(boundary: Option<&str>) -> PyResult<Option<BoundaryType>> {
    match boundary {
        None => Ok(None),
        Some(s) => BoundaryType::from_str_option(s)
            .map(Some)
            .ok_or_else(|| PyValueError::new_err("boundary must be None or 'vacuum'")),
    }
}

// Internal only: this wrapper is never registered on the module (no
// `add_class`) and no binding returns it -- surfaces accept a boundary `str`
// (e.g. "vacuum"). It is intentionally NOT given the `gen_stub_*` macros, so it
// stays out of the generated stub and `__all__` (otherwise the stub would
// advertise a `BoundaryType` symbol that cannot be imported).
#[pyclass(module = "yamc._core", name = "BoundaryType", from_py_object)]
#[derive(Clone)]
pub struct PyBoundaryType {
    pub inner: BoundaryType,
}

#[pymethods]
impl PyBoundaryType {
    #[new]
    fn new(boundary: &str) -> PyResult<Self> {
        let boundary = BoundaryType::from_str_option(boundary)
            .ok_or_else(|| PyValueError::new_err("boundary must be 'transmission' or 'vacuum'"))?;
        Ok(PyBoundaryType { inner: boundary })
    }

    fn __str__(&self) -> &str {
        match self.inner {
            BoundaryType::Transmission => "transmission",
            BoundaryType::Vacuum => "vacuum",
        }
    }

    fn __repr__(&self) -> String {
        format!("BoundaryType('{}')", self.__str__())
    }
}

/// A geometric surface (plane, cylinder, sphere, …); combine with `+`/`-` into
/// the half-spaces that build CSG `Region`s.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "Surface", from_py_object)]
#[derive(Clone)]
pub struct PySurface {
    pub inner: std::sync::Arc<Surface>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySurface {
    pub fn __repr__(&self) -> String {
        let id = self
            .inner
            .surface_id
            .map_or("None".to_string(), |id| id.to_string());
        // Label an axis vector as "x"/"y"/"z" when it is exactly ±a basis
        // vector, else as its components. Mirrors the unified `axis=` argument.
        fn axis_label(axis: &[f64; 3]) -> String {
            match parallel_basis_index(axis) {
                Some(k) => ["x", "y", "z"][k].to_string(),
                None => format!("({}, {}, {})", axis[0], axis[1], axis[2]),
            }
        }
        let kind = match &self.inner.kind {
            yamc::geo::SurfaceKind::Plane { a, b, c, .. } => {
                format!("Plane(axis={})", axis_label(&[*a, *b, *c]))
            }
            yamc::geo::SurfaceKind::Sphere { .. } => "Sphere".to_string(),
            yamc::geo::SurfaceKind::Cylinder { axis, .. } => {
                format!("Cylinder(axis={})", axis_label(axis))
            }
            yamc::geo::SurfaceKind::ZTorus { .. } => "Torus(axis=z)".to_string(),
            yamc::geo::SurfaceKind::XTorus { .. } => "Torus(axis=x)".to_string(),
            yamc::geo::SurfaceKind::YTorus { .. } => "Torus(axis=y)".to_string(),
            yamc::geo::SurfaceKind::Quadric { .. } => "Quadric".to_string(),
            yamc::geo::SurfaceKind::Cone { axis, .. } => {
                format!("Cone(axis={})", axis_label(axis))
            }
        };
        let boundary = match self.inner.boundary {
            BoundaryType::Transmission => "None",
            BoundaryType::Vacuum => "'vacuum'",
        };
        format!("Surface(id={id}, type={kind}, boundary={boundary})")
    }

    /// Evaluate the surface equation at a point.
    ///
    /// Args:
    ///     point: Tuple of 3 floats (x, y, z)
    ///
    /// Returns:
    ///     Float value. Negative means inside/below, positive means outside/above, zero means on surface.
    pub fn evaluate(&self, point: (f64, f64, f64)) -> f64 {
        // Call the core Rust implementation
        self.inner.evaluate(point)
    }

    /// Offset of a plane perpendicular to the x-axis (axis='x'), else None
    #[getter]
    pub fn x0(&self) -> Option<f64> {
        match &self.inner.kind {
            yamc::geo::SurfaceKind::Plane { a, b, c, d } if *a == 1.0 && *b == 0.0 && *c == 0.0 => {
                Some(*d)
            }
            _ => None,
        }
    }

    /// Offset of a plane perpendicular to the y-axis (axis='y'), else None
    #[getter]
    pub fn y0(&self) -> Option<f64> {
        match &self.inner.kind {
            yamc::geo::SurfaceKind::Plane { a, b, c, d } if *a == 0.0 && *b == 1.0 && *c == 0.0 => {
                Some(*d)
            }
            _ => None,
        }
    }

    /// Offset of a plane perpendicular to the z-axis (axis='z'), else None
    #[getter]
    pub fn z0(&self) -> Option<f64> {
        match &self.inner.kind {
            yamc::geo::SurfaceKind::Plane { a, b, c, d } if *a == 0.0 && *b == 0.0 && *c == 1.0 => {
                Some(*d)
            }
            _ => None,
        }
    }

    /// Radius for Sphere or Cylinder surfaces, else None
    #[getter]
    pub fn radius(&self) -> Option<f64> {
        match &self.inner.kind {
            yamc::geo::SurfaceKind::Sphere { radius, .. } => Some(*radius),
            yamc::geo::SurfaceKind::Cylinder { radius, .. } => Some(*radius),
            _ => None,
        }
    }

    /// Get the axis_x value for Cylinder surfaces, or None otherwise
    #[getter]
    pub fn axis_x(&self) -> Option<f64> {
        match &self.inner.kind {
            yamc::geo::SurfaceKind::Cylinder { axis, .. } => Some(axis[0]),
            _ => None,
        }
    }

    /// Get the axis_y value for Cylinder surfaces, or None otherwise
    #[getter]
    pub fn axis_y(&self) -> Option<f64> {
        match &self.inner.kind {
            yamc::geo::SurfaceKind::Cylinder { axis, .. } => Some(axis[1]),
            _ => None,
        }
    }

    /// Get the axis_z value for Cylinder surfaces, or None otherwise
    #[getter]
    pub fn axis_z(&self) -> Option<f64> {
        match &self.inner.kind {
            yamc::geo::SurfaceKind::Cylinder { axis, .. } => Some(axis[2]),
            _ => None,
        }
    }

    /// Compute the distance to the surface from a point in a direction.
    /// Returns the distance as a float, or None if no intersection.
    pub fn distance_to_surface(
        &self,
        point: (f64, f64, f64),
        direction: (f64, f64, f64),
    ) -> Option<f64> {
        let point_arr = [point.0, point.1, point.2];
        let dir_arr = [direction.0, direction.1, direction.2];
        self.inner.distance_to_surface(point_arr, dir_arr)
    }

    /// Get the bounding box for the inside (negative halfspace) of this surface
    pub fn bounding_box_inside(&self) -> Option<PyBoundingBox> {
        self.inner
            .bounding_box(true)
            .map(|(lower, upper)| PyBoundingBox::new(lower, upper))
    }

    pub fn bounding_box_outside(&self) -> Option<PyBoundingBox> {
        self.inner
            .bounding_box(false)
            .map(|(lower, upper)| PyBoundingBox::new(lower, upper))
    }

    /// Get axis constraint for this surface when used as a halfspace
    /// Returns (axis_index, is_upper_bound, value) or None
    /// axis_index: 0=X, 1=Y, 2=Z
    /// is_upper_bound: True if this constrains the upper bound, False for lower bound
    pub fn axis_constraint(&self, halfspace_below: bool) -> Option<(usize, bool, f64)> {
        self.inner.axis_constraint(halfspace_below)
    }

    #[getter]
    pub fn surface_id(&self) -> Option<usize> {
        self.inner.surface_id
    }

    #[setter(surface_id)]
    pub fn set_surface_id(&mut self, surface_id: Option<usize>) {
        if let Some(surface) = std::sync::Arc::get_mut(&mut self.inner) {
            surface.surface_id = surface_id;
        } else {
            // If Arc is shared, clone it to get a mutable version
            let mut surface = (*self.inner).clone();
            surface.surface_id = surface_id;
            self.inner = std::sync::Arc::new(surface);
        }
    }

    /// Boundary condition: None for transmission (default) or 'vacuum'.
    #[getter]
    pub fn boundary(&self) -> Option<&'static str> {
        match self.inner.boundary {
            BoundaryType::Transmission => None,
            BoundaryType::Vacuum => Some("vacuum"),
        }
    }

    #[setter(boundary)]
    pub fn set_boundary(&mut self, boundary: Option<&str>) -> PyResult<()> {
        let boundary = match boundary {
            None => BoundaryType::Transmission,
            Some(s) => BoundaryType::from_str_option(s)
                .ok_or_else(|| PyValueError::new_err("boundary must be None or 'vacuum'"))?,
        };

        if let Some(surface) = std::sync::Arc::get_mut(&mut self.inner) {
            surface.set_boundary(boundary);
        } else {
            // If Arc is shared, clone it to get a mutable version
            let mut surface = (*self.inner).clone();
            surface.set_boundary(boundary);
            self.inner = std::sync::Arc::new(surface);
        }
        Ok(())
    }

    /// The below (negative-sense) half-space of this surface, as a `Region`:
    /// points where `surface.evaluate(p) < 0` (inside a sphere/cylinder, below a
    /// plane). Combine with `&`/`|`/`~`. (Replaces the old unary `-surface`.)
    #[getter]
    fn below(slf: PyRef<'_, Self>) -> PyResult<crate::geometry::PyRegion> {
        use crate::geometry::{PyRegion, PyRegionExpr};
        let py = slf.py();
        let py_surface: Py<crate::geometry::PySurface> = Py::new(py, slf.clone())?;
        let expr = PyRegionExpr::Halfspace(PyHalfspace::new_below(py_surface));
        let region = yamc::geo::Region {
            expr: expr.to_region_expr(),
        };
        Ok(PyRegion { expr, region })
    }

    /// The above (positive-sense) half-space of this surface, as a `Region`:
    /// points where `surface.evaluate(p) > 0` (outside a sphere/cylinder, above a
    /// plane). Combine with `&`/`|`/`~`. (Replaces the old unary `+surface`.)
    #[getter]
    fn above(slf: PyRef<'_, Self>) -> PyResult<crate::geometry::PyRegion> {
        use crate::geometry::{PyRegion, PyRegionExpr};
        let py = slf.py();
        let py_surface: Py<crate::geometry::PySurface> = Py::new(py, slf.clone())?;
        let expr = PyRegionExpr::Halfspace(PyHalfspace::new_above(py_surface));
        let region = yamc::geo::Region {
            expr: expr.to_region_expr(),
        };
        Ok(PyRegion { expr, region })
    }
}

/// Parse an `axis=` argument that is either an axis name ('x'/'y'/'z') or a
/// 3-element direction vector, returning a NORMALIZED `[f64; 3]`. Rejects
/// zero-length vectors and unknown axis names.
fn parse_axis(axis: &Bound<'_, PyAny>) -> PyResult<[f64; 3]> {
    if let Ok(name) = axis.extract::<String>() {
        return match name.to_lowercase().as_str() {
            "x" => Ok([1.0, 0.0, 0.0]),
            "y" => Ok([0.0, 1.0, 0.0]),
            "z" => Ok([0.0, 0.0, 1.0]),
            other => Err(PyValueError::new_err(format!(
                "axis string must be 'x', 'y', or 'z' (got '{other}')"
            ))),
        };
    }
    let v: [f64; 3] = axis.extract().map_err(|_| {
        PyValueError::new_err("axis must be 'x'/'y'/'z' or a 3-element direction vector")
    })?;
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm == 0.0 {
        return Err(PyValueError::new_err("axis vector must be non-zero"));
    }
    Ok([v[0] / norm, v[1] / norm, v[2] / norm])
}

/// If the unit `axis` is exactly ±e_k (parallel to a coordinate axis), return
/// Some(k); else None. Used to reject the gauge-redundant axial coordinate of a
/// named-axis cylinder. Exact equality is intentional: a slightly-off axis is a
/// genuinely oblique cylinder for which every coordinate is meaningful.
fn parallel_basis_index(axis: &[f64; 3]) -> Option<usize> {
    (0..3).find(|&k| axis[k].abs() == 1.0 && (0..3).all(|i| i == k || axis[i] == 0.0))
}

/// Create a plane surface perpendicular to a direction.
///
/// The plane is the set of points where `n·(x, y, z) = offset`, with `n` the
/// (normalized) `axis`.
///
/// Args:
///     axis: Plane normal - 'x'/'y'/'z' or a direction vector (a, b, c).
///     offset: Signed position of the plane along `axis` (default: 0.0).
///         With axis='x', offset=5 is the plane x = 5; with axis=(1, 1, 0),
///         offset=3 (after normalization) is the plane x + y = 3·√2.
///     surface_id: Optional surface ID
///     boundary: Boundary condition - None (default, transmission) or 'vacuum'
///     name: Optional surface name
///
/// Returns:
///     A Surface object representing the plane
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(non_snake_case)]
#[pyo3(signature = (axis, offset = 0.0, surface_id = None, boundary = None, name = None))]
pub fn Plane(
    axis: &Bound<'_, PyAny>,
    offset: f64,
    surface_id: Option<usize>,
    boundary: Option<&str>,
    name: Option<String>,
) -> PyResult<PySurface> {
    let n = parse_axis(axis)?;
    let boundary_validated = parse_boundary(boundary)?;
    let mut surface = Surface::new_plane(n[0], n[1], n[2], offset, surface_id, boundary_validated);
    if let Some(nm) = name {
        surface = surface.with_name(nm);
    }
    Ok(PySurface {
        inner: std::sync::Arc::new(surface),
    })
}

/// Create a sphere surface.
///
/// Args:
///     x0: X-coordinate of the sphere center (default: 0.0)
///     y0: Y-coordinate of the sphere center (default: 0.0)
///     z0: Z-coordinate of the sphere center (default: 0.0)
///     radius: Radius of the sphere (default: 1.0)
///     surface_id: Optional surface ID
///     boundary: Boundary condition - None (default, transmission) or 'vacuum'
///
/// Returns:
///     A Surface object representing the sphere
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(non_snake_case)]
#[pyo3(signature = (x0 = 0.0, y0 = 0.0, z0 = 0.0, radius = 1.0, surface_id = None, boundary = None, name = None))]
pub fn Sphere(
    x0: f64,
    y0: f64,
    z0: f64,
    radius: f64,
    surface_id: Option<usize>,
    boundary: Option<&str>,
    name: Option<String>,
) -> PyResult<PySurface> {
    let boundary_validated = parse_boundary(boundary)?;
    let mut surface = Surface::sphere(x0, y0, z0, radius, surface_id, boundary_validated);
    if let Some(n) = name {
        surface = surface.with_name(n);
    }
    Ok(PySurface {
        inner: std::sync::Arc::new(surface),
    })
}

/// Create a cylinder surface.
///
/// The cylinder passes through the point ``(x0, y0, z0)`` (default: origin).
/// When the axis is parallel to a coordinate axis, the coordinate along that
/// axis is meaningless (the cylinder is infinite along it) and supplying it
/// raises ValueError; for an oblique axis all three are meaningful.
///
/// Args:
///     axis: Cylinder axis - 'x'/'y'/'z' or a direction vector. The axis is an
///         undirected line: its sign and magnitude don't matter.
///     radius: Cylinder radius (default: 1.0)
///     surface_id: Optional surface ID
///     boundary: Boundary condition - None (default, transmission) or 'vacuum'
///     name: Optional surface name
///
/// Returns:
///     A Surface object representing the cylinder
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(non_snake_case, clippy::too_many_arguments)]
#[pyo3(signature = (axis, x0 = None, y0 = None, z0 = None, radius = 1.0, surface_id = None, boundary = None, name = None))]
pub fn Cylinder(
    axis: &Bound<'_, PyAny>,
    x0: Option<f64>,
    y0: Option<f64>,
    z0: Option<f64>,
    radius: f64,
    surface_id: Option<usize>,
    boundary: Option<&str>,
    name: Option<String>,
) -> PyResult<PySurface> {
    let a = parse_axis(axis)?;
    if let Some(k) = parallel_basis_index(&a) {
        let (coord, label) = match k {
            0 => (x0, "x0"),
            1 => (y0, "y0"),
            _ => (z0, "z0"),
        };
        if coord.is_some() {
            return Err(PyValueError::new_err(format!(
                "{label} has no meaning for a cylinder whose axis is parallel to '{}' (the cylinder is infinite along it)",
                ["x", "y", "z"][k]
            )));
        }
    }
    let origin = [x0.unwrap_or(0.0), y0.unwrap_or(0.0), z0.unwrap_or(0.0)];
    let boundary_validated = parse_boundary(boundary)?;
    let mut surface = Surface::new_cylinder(a, origin, radius, surface_id, boundary_validated);
    if let Some(nm) = name {
        surface = surface.with_name(nm);
    }
    Ok(PySurface {
        inner: std::sync::Arc::new(surface),
    })
}

/// Create a general quadric surface (``Quadric``):
/// ``a x^2 + b y^2 + c z^2 + d xy + e yz + f xz + g x + h y + j z + k = 0``.
/// The coefficients ``a`` through ``k`` (``i`` is skipped) are the terms of
/// that equation.
///
/// Args:
///     surface_id: Optional surface ID
///     boundary: Boundary condition - None (default, transmission) or 'vacuum'
///     name: Optional surface name
///
/// Returns:
///     A Surface object representing the quadric
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(non_snake_case, clippy::too_many_arguments)]
#[pyo3(signature = (a = 0.0, b = 0.0, c = 0.0, d = 0.0, e = 0.0, f = 0.0, g = 0.0, h = 0.0, j = 0.0, k = 0.0, surface_id = None, boundary = None, name = None))]
pub fn Quadric(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    j: f64,
    k: f64,
    surface_id: Option<usize>,
    boundary: Option<&str>,
    name: Option<String>,
) -> PyResult<PySurface> {
    let boundary_validated = parse_boundary(boundary)?;
    let mut surface =
        Surface::new_quadric(a, b, c, d, e, f, g, h, j, k, surface_id, boundary_validated);
    if let Some(nm) = name {
        surface = surface.with_name(nm);
    }
    Ok(PySurface {
        inner: std::sync::Arc::new(surface),
    })
}

/// Create a cone surface (double-sheeted / both nappes). The apex is at
/// ``(x0, y0, z0)`` (default: origin); unlike a cylinder, all three apex
/// coordinates are meaningful for every axis.
///
/// Args:
///     axis: Cone axis - 'x'/'y'/'z' or a direction vector.
///     opening_angle: Half-opening angle in DEGREES (default: 45.0).
///     surface_id: Optional surface ID
///     boundary: Boundary condition - None (default, transmission) or 'vacuum'
///     name: Optional surface name
///
/// Returns:
///     A Surface object representing the (double) cone.
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(non_snake_case, clippy::too_many_arguments)]
#[pyo3(signature = (axis, x0 = 0.0, y0 = 0.0, z0 = 0.0, opening_angle = 45.0, surface_id = None, boundary = None, name = None))]
pub fn Cone(
    axis: &Bound<'_, PyAny>,
    x0: f64,
    y0: f64,
    z0: f64,
    opening_angle: f64,
    surface_id: Option<usize>,
    boundary: Option<&str>,
    name: Option<String>,
) -> PyResult<PySurface> {
    let a = parse_axis(axis)?;
    let t = (opening_angle * std::f64::consts::PI / 180.0).tan();
    let tan2_theta = t * t;
    let boundary_validated = parse_boundary(boundary)?;
    let mut surface =
        Surface::new_cone([x0, y0, z0], a, tan2_theta, surface_id, boundary_validated);
    if let Some(nm) = name {
        surface = surface.with_name(nm);
    }
    Ok(PySurface {
        inner: std::sync::Arc::new(surface),
    })
}

/// Create a torus surface.
///
/// The torus is centered at ``(x0, y0, z0)`` (default: origin); all three
/// coordinates are meaningful for every axis.
///
/// Args:
///     axis: Symmetry axis - 'x', 'y', or 'z'. Arbitrary direction vectors are
///         NOT yet supported (a torus is degree-4 and the engine only has
///         axis-locked variants; see issue #396); an oblique vector raises
///         ValueError.
///     r_major: Major radius (default: 1.0)
///     r_minor: Minor radius along the symmetry axis (default: 0.5)
///     r_minor_2: Minor radius in the transverse plane. Defaults to `r_minor`
///         (a circular cross-section); set it different from `r_minor` for an
///         elliptical cross-section.
///     surface_id: Optional surface ID
///     boundary: Boundary condition - None (default, transmission) or 'vacuum'
///     name: Optional surface name
///
/// Returns:
///     A Surface object representing the torus
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(non_snake_case, clippy::too_many_arguments)]
#[pyo3(signature = (axis, x0 = 0.0, y0 = 0.0, z0 = 0.0, r_major = 1.0, r_minor = 0.5, r_minor_2 = None, surface_id = None, boundary = None, name = None))]
pub fn Torus(
    axis: &Bound<'_, PyAny>,
    x0: f64,
    y0: f64,
    z0: f64,
    r_major: f64,
    r_minor: f64,
    r_minor_2: Option<f64>,
    surface_id: Option<usize>,
    boundary: Option<&str>,
    name: Option<String>,
) -> PyResult<PySurface> {
    let a = parse_axis(axis)?;
    let boundary_validated = parse_boundary(boundary)?;
    // `r_minor` is the minor radius along the symmetry axis (engine `b`),
    // `r_minor_2` the transverse minor radius (engine `c`); equal => circular.
    let c = r_minor_2.unwrap_or(r_minor);
    // The engine only has axis-locked X/Y/ZTorus variants. An arbitrary-axis
    // torus (degree-4, not a quadric) needs a new engine variant -- see #396.
    let mut surface = match parallel_basis_index(&a) {
        Some(0) => Surface::new_xtorus(
            x0,
            y0,
            z0,
            r_major,
            r_minor,
            c,
            surface_id,
            boundary_validated,
        ),
        Some(1) => Surface::new_ytorus(
            x0,
            y0,
            z0,
            r_major,
            r_minor,
            c,
            surface_id,
            boundary_validated,
        ),
        Some(2) => Surface::new_ztorus(
            x0,
            y0,
            z0,
            r_major,
            r_minor,
            c,
            surface_id,
            boundary_validated,
        ),
        _ => {
            return Err(PyValueError::new_err(
                "arbitrary-axis torus is not yet supported; use axis='x', 'y', or 'z' (see issue #396)",
            ));
        }
    };
    if let Some(nm) = name {
        surface = surface.with_name(nm);
    }
    Ok(PySurface {
        inner: std::sync::Arc::new(surface),
    })
}
