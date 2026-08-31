//! Lump shapes, for turning a material's volume into a self-shielding chord.
//!
//! Each class carries exactly the dimension a volume cannot supply and no more,
//! so nothing is ever given twice and there is no disagreement to resolve. A
//! sphere and a cube are fixed by volume alone and take no arguments; a foil
//! needs its thickness, a cylinder and a wire their radius.
//!
//! The names carry a `Lump` suffix because `yamc` -- which links these bindings
//! and exposes transmutation as well as transport -- already has CSG surface
//! constructors called `Sphere` and `Cylinder`. Those are geometry for a model;
//! these are the shape of the sample being activated, and the two must not
//! answer to the same name in one namespace.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yani_transmute::Shape;

/// A sphere, sized by the material's volume.
///
/// The extreme case: a sphere has the least surface for its volume, so it has
/// the longest mean chord and shields more than any other shape of the same
/// volume. Convenient, but an upper bound rather than a neutral choice.
#[gen_stub_pyclass]
#[pyclass(name = "SphereLump", from_py_object)]
#[derive(Clone)]
pub struct PySphere;

#[gen_stub_pymethods]
#[pymethods]
impl PySphere {
    /// A sphere. Its radius follows from ``Material(volume=...)``.
    #[new]
    fn new() -> Self {
        Self
    }
    fn __repr__(&self) -> String {
        "SphereLump()".to_string()
    }
}

/// A cube, sized by the material's volume.
#[gen_stub_pyclass]
#[pyclass(name = "CubeLump", from_py_object)]
#[derive(Clone)]
pub struct PyCube;

#[gen_stub_pymethods]
#[pymethods]
impl PyCube {
    /// A cube. Its side follows from ``Material(volume=...)``.
    #[new]
    fn new() -> Self {
        Self
    }
    fn __repr__(&self) -> String {
        "CubeLump()".to_string()
    }
}

/// A flat slab of a stated thickness.
///
/// The chord is twice the thickness whatever the area, so a foil's shielding
/// does not depend on the volume at all -- the volume is still what makes
/// ``activity`` and ``decay_heat`` available, but it plays no part here.
#[gen_stub_pyclass]
#[pyclass(name = "FoilLump", from_py_object)]
#[derive(Clone)]
pub struct PyFoil {
    #[pyo3(get)]
    thickness: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFoil {
    /// Args:
    ///     thickness (float): Foil thickness in cm.
    #[new]
    #[pyo3(signature = (thickness))]
    fn new(thickness: f64) -> PyResult<Self> {
        positive(thickness, "thickness")?;
        Ok(Self { thickness })
    }
    fn __repr__(&self) -> String {
        format!("FoilLump(thickness={})", self.thickness)
    }
}

/// A cylinder of a stated radius; its height follows from the volume.
#[gen_stub_pyclass]
#[pyclass(name = "CylinderLump", from_py_object)]
#[derive(Clone)]
pub struct PyCylinder {
    #[pyo3(get)]
    radius: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCylinder {
    /// Args:
    ///     radius (float): Cylinder radius in cm.
    #[new]
    #[pyo3(signature = (radius))]
    fn new(radius: f64) -> PyResult<Self> {
        positive(radius, "radius")?;
        Ok(Self { radius })
    }
    fn __repr__(&self) -> String {
        format!("CylinderLump(radius={})", self.radius)
    }
}

/// A cylinder long enough that its ends do not matter. Chord ``2r``.
#[gen_stub_pyclass]
#[pyclass(name = "WireLump", from_py_object)]
#[derive(Clone)]
pub struct PyWire {
    #[pyo3(get)]
    radius: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWire {
    /// Args:
    ///     radius (float): Wire radius in cm.
    #[new]
    #[pyo3(signature = (radius))]
    fn new(radius: f64) -> PyResult<Self> {
        positive(radius, "radius")?;
        Ok(Self { radius })
    }
    fn __repr__(&self) -> String {
        format!("WireLump(radius={})", self.radius)
    }
}

fn positive(value: f64, what: &str) -> PyResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{what} must be a positive length in cm, got {value}"
        )));
    }
    Ok(())
}

/// The `Shape` a Python shape object stands for.
pub fn shape_of(ob: &Bound<'_, PyAny>) -> PyResult<Shape> {
    if ob.extract::<PySphere>().is_ok() {
        return Ok(Shape::Sphere);
    }
    if ob.extract::<PyCube>().is_ok() {
        return Ok(Shape::Cube);
    }
    if let Ok(f) = ob.extract::<PyFoil>() {
        return Ok(Shape::Foil {
            thickness_cm: f.thickness,
        });
    }
    if let Ok(c) = ob.extract::<PyCylinder>() {
        return Ok(Shape::Cylinder {
            radius_cm: c.radius,
        });
    }
    if let Ok(w) = ob.extract::<PyWire>() {
        return Ok(Shape::Wire {
            radius_cm: w.radius,
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "self_shielding_shape must be one of shapes.SphereLump(), shapes.CubeLump(), \
         shapes.FoilLump(thickness=...), shapes.CylinderLump(radius=...) or \
         shapes.WireLump(radius=...)",
    ))
}

pub fn register_shape_classes(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySphere>()?;
    m.add_class::<PyCube>()?;
    m.add_class::<PyFoil>()?;
    m.add_class::<PyCylinder>()?;
    m.add_class::<PyWire>()?;
    Ok(())
}
