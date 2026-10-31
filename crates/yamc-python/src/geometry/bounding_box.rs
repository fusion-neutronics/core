use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Axis-aligned bounding box in 3D space.
///
/// This class is typically returned by geometry methods rather than constructed directly.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "BoundingBox", from_py_object)]
#[derive(Clone)]
pub struct PyBoundingBox {
    /// Lower-left corner coordinates [x, y, z] in cm.
    #[pyo3(get)]
    lower_left: [f64; 3],
    /// Upper-right corner coordinates [x, y, z] in cm.
    #[pyo3(get)]
    upper_right: [f64; 3],
    /// Center coordinates [x, y, z] in cm.
    #[pyo3(get)]
    center: [f64; 3],
    /// Width along each axis [dx, dy, dz] in cm.
    #[pyo3(get)]
    width: [f64; 3],
}

impl PyBoundingBox {
    pub fn lower_left(&self) -> [f64; 3] {
        self.lower_left
    }
    pub fn upper_right(&self) -> [f64; 3] {
        self.upper_right
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBoundingBox {
    /// True when every corner coordinate is finite.
    ///
    /// An unbounded region (a half-space, or a cell with no outer surface)
    /// produces infinite extents, which cannot be voxelized. Mirrors
    /// [`yamc_geo::BoundingBox::is_finite`].
    ///
    /// Returns:
    ///     bool
    pub fn is_finite(&self) -> bool {
        self.lower_left.iter().all(|v| v.is_finite())
            && self.upper_right.iter().all(|v| v.is_finite())
    }

    /// Create a bounding box from corner coordinates.
    ///
    /// Args:
    ///     lower_left: Lower-left corner [x, y, z] in cm.
    ///     upper_right: Upper-right corner [x, y, z] in cm.
    ///
    /// Returns:
    ///     BoundingBox: An axis-aligned bounding box.
    #[new]
    pub fn new(lower_left: [f64; 3], upper_right: [f64; 3]) -> Self {
        let center = [
            0.5 * (lower_left[0] + upper_right[0]),
            0.5 * (lower_left[1] + upper_right[1]),
            0.5 * (lower_left[2] + upper_right[2]),
        ];
        let width = [
            upper_right[0] - lower_left[0],
            upper_right[1] - lower_left[1],
            upper_right[2] - lower_left[2],
        ];
        PyBoundingBox {
            lower_left,
            upper_right,
            center,
            width,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "BoundingBox(lower_left={:?}, upper_right={:?}, center={:?}, width={:?})",
            self.lower_left, self.upper_right, self.center, self.width
        )
    }
}
