use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc_source::distribution::angular::AngularDistribution;

/// Isotropic angular distribution (uniform over 4π steradians).
#[gen_stub_pyclass]
#[pyclass(name = "Isotropic", from_py_object)]
#[derive(Clone)]
pub struct PyIsotropic {
    pub inner: AngularDistribution,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyIsotropic {
    /// Create an isotropic angular distribution.
    ///
    /// Returns:
    ///     Isotropic: An isotropic angular distribution.
    ///
    /// Examples:
    ///     >>> import yamc
    ///     >>> angle_dist = yamc.Isotropic()
    #[new]
    pub fn new() -> Self {
        PyIsotropic {
            inner: AngularDistribution::Isotropic,
        }
    }

    /// Sample a direction vector from the distribution.
    ///
    /// Returns:
    ///     List[float]: Unit direction vector [u, v, w].
    pub fn sample(&self) -> [f64; 3] {
        // Python bindings use thread_rng for backwards compatibility
        let mut rng = rand::rng();
        self.inner.sample(&mut rng)
    }

    /// Rich Jupyter display.
    pub fn _repr_html_(&self) -> String {
        crate::html_repr::card(
            "Isotropic",
            "uniform over 4π sr",
            "<em style=\"color:#656d76;\">no parameters</em>",
        )
    }

    pub fn __repr__(&self) -> String {
        "Isotropic()".to_string()
    }
}

/// Monodirectional angular distribution (single fixed direction).
#[gen_stub_pyclass]
#[pyclass(name = "Monodirectional", from_py_object)]
#[derive(Clone)]
pub struct PyMonodirectional {
    pub inner: AngularDistribution,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMonodirectional {
    /// Create a monodirectional angular distribution.
    ///
    /// Args:
    ///     direction: Direction vector [u, v, w]. Will be normalized automatically.
    ///
    /// Returns:
    ///     Monodirectional: A monodirectional angular distribution.
    ///
    /// Examples:
    ///     >>> import yamc
    ///     >>> # Particles emitted in +z direction
    ///     >>> angle_dist = yamc.Monodirectional([0.0, 0.0, 1.0])
    #[new]
    pub fn new(direction: [f64; 3]) -> Self {
        PyMonodirectional {
            inner: AngularDistribution::new_monodirectional(
                direction[0],
                direction[1],
                direction[2],
            ),
        }
    }

    /// Sample a direction vector from the distribution.
    ///
    /// Returns:
    ///     List[float]: Unit direction vector [u, v, w] (always the reference direction).
    pub fn sample(&self) -> [f64; 3] {
        // Python bindings use thread_rng for backwards compatibility
        let mut rng = rand::rng();
        self.inner.sample(&mut rng)
    }

    /// Get the direction vector.
    ///
    /// Returns:
    ///     List[float]: Unit direction vector [u, v, w].
    #[getter]
    pub fn direction(&self) -> [f64; 3] {
        match &self.inner {
            AngularDistribution::Monodirectional { direction } => *direction,
            AngularDistribution::Isotropic => panic!("Cannot get direction from Isotropic"),
        }
    }

    /// Set the direction vector.
    ///
    /// Args:
    ///     direction: Direction vector [u, v, w]. Will be normalized automatically.
    #[setter]
    pub fn set_direction(&mut self, direction: [f64; 3]) {
        self.inner =
            AngularDistribution::new_monodirectional(direction[0], direction[1], direction[2]);
    }

    /// Rich Jupyter display: the direction vector.
    pub fn _repr_html_(&self) -> String {
        use crate::html_repr::{card, kv, num};
        let d = self.direction();
        let body = kv(&[(
            "direction",
            format!("[{}, {}, {}]", num(d[0]), num(d[1]), num(d[2])),
        )]);
        card("Monodirectional", "", &body)
    }

    pub fn __repr__(&self) -> String {
        match &self.inner {
            AngularDistribution::Monodirectional { direction } => {
                format!("Monodirectional(direction={:?})", direction)
            }
            AngularDistribution::Isotropic => {
                panic!("Invalid state: Monodirectional wrapper contains Isotropic")
            }
        }
    }
}

pub fn register_distribution_classes(
    _py: Python,
    parent_module: &Bound<'_, PyModule>,
) -> PyResult<()> {
    use crate::distribution::PyCylindricalRing;
    use crate::distribution::{
        py_fusion_neutron_spectrum, PyDiscrete, PyHistogram, PyNormal, PyUniform,
    };

    // Register distribution classes directly on the parent module
    parent_module.add_class::<PyIsotropic>()?;
    parent_module.add_class::<PyMonodirectional>()?;
    parent_module.add_class::<PyDiscrete>()?;
    parent_module.add_class::<PyHistogram>()?;
    parent_module.add_class::<PyUniform>()?;
    parent_module.add_class::<PyNormal>()?;
    parent_module.add_class::<PyCylindricalRing>()?;
    parent_module.add_function(wrap_pyfunction!(py_fusion_neutron_spectrum, parent_module)?)?;
    // The parametric tokamak plasma source and the plasma profile helpers it
    // is built from.
    parent_module.add_function(wrap_pyfunction!(
        crate::distribution::py_tokamak_source,
        parent_module
    )?)?;
    parent_module.add_function(wrap_pyfunction!(
        crate::distribution::py_tokamak_ion_density,
        parent_module
    )?)?;
    parent_module.add_function(wrap_pyfunction!(
        crate::distribution::py_tokamak_ion_temperature,
        parent_module
    )?)?;
    parent_module.add_function(wrap_pyfunction!(
        crate::distribution::py_tokamak_convert_a_alpha_to_r_z,
        parent_module
    )?)?;
    parent_module.add_function(wrap_pyfunction!(
        crate::distribution::py_tokamak_neutron_source_density,
        parent_module
    )?)?;
    Ok(())
}
