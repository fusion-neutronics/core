use crate::distribution::{PyDiscrete, PyNormal, PyUniform};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc_source::distribution::mesh::{MeshDistribution, MeshElements};
use yamc_source::distribution::spatial::{CylindricalRing, Univariate};
use yamc_source::source::SourceEnergyDistribution;

/// Cylindrical spatial distribution (ring, annulus, or full cylinder).
///
/// Samples radius, phi, and z independently from univariate distributions
/// in a cylindrical coordinate system centered at ``origin``.
///
/// Positions are computed as:
///   ``x = origin[0] + radius * cos(phi)``
///   ``y = origin[1] + radius * sin(phi)``
///   ``z = origin[2] + z``
///
/// Args:
///     radius: Distribution of radial coordinates (e.g., ``yamc.Discrete([300], [1])``
///         for a thin ring, or ``yamc.Uniform(200, 400)`` for an annulus).
///     phi: Distribution of azimuthal angle in radians (e.g.,
///         ``yamc.Uniform(0, 2*math.pi)`` for full ring).
///     z: Distribution of axial coordinates (e.g., ``yamc.Discrete([0], [1])``
///         for the midplane).
///     origin: Center of the cylindrical reference frame [x, y, z] in cm.
///         Defaults to [0, 0, 0].
///
/// Raises:
///     ValueError: If ``origin`` has a NaN or infinite coordinate, or a
///         ``radius``, ``phi`` or ``z`` given as a number is NaN or infinite.
///
/// Examples:
///     >>> import yamc, math
///     >>> # Ring source at radius=300 cm in the midplane
///     >>> space = yamc.CylindricalRing(
///     ...     radius=yamc.Discrete([300], [1]),
///     ...     phi=yamc.Uniform(0, 2*math.pi),
///     ...     z=yamc.Discrete([0], [1]),
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "CylindricalRing", from_py_object)]
#[derive(Clone)]
pub struct PyCylindricalRing {
    pub inner: CylindricalRing,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCylindricalRing {
    #[new]
    #[pyo3(signature = (*, radius, phi, z, origin=None))]
    pub fn new(
        #[gen_stub(override_type(type_repr = "Discrete | Uniform | Normal"))] radius: &Bound<
            '_,
            PyAny,
        >,
        #[gen_stub(override_type(type_repr = "Discrete | Uniform | Normal"))] phi: &Bound<
            '_,
            PyAny,
        >,
        #[gen_stub(override_type(type_repr = "Discrete | Uniform | Normal"))] z: &Bound<'_, PyAny>,
        origin: Option<[f64; 3]>,
    ) -> PyResult<Self> {
        let r_dist = extract_univariate(radius, "radius")?;
        let phi_dist = extract_univariate(phi, "phi")?;
        let z_dist = extract_univariate(z, "z")?;
        let origin = origin.unwrap_or([0.0, 0.0, 0.0]);
        Ok(Self {
            inner: CylindricalRing::new(r_dist, phi_dist, z_dist, origin)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        })
    }

    /// Sample a position from the distribution.
    ///
    /// Returns:
    ///     List[float]: Position vector [x, y, z] in cm.
    pub fn sample(&self) -> [f64; 3] {
        let mut rng = rand::rng();
        self.inner.sample(&mut rng)
    }

    /// Center of the cylindrical reference frame, ``[x, y, z]`` in cm.
    #[getter]
    pub fn origin(&self) -> [f64; 3] {
        self.inner.origin
    }

    /// Radial distribution.
    #[getter]
    pub fn radius(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::distribution::source_energy_distribution_into_py(
            py,
            univariate_to_energy_dist(&self.inner.r),
        )
    }

    /// Azimuthal-angle distribution (radians).
    #[getter]
    pub fn phi(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::distribution::source_energy_distribution_into_py(
            py,
            univariate_to_energy_dist(&self.inner.phi),
        )
    }

    /// Axial distribution.
    #[getter]
    pub fn z(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::distribution::source_energy_distribution_into_py(
            py,
            univariate_to_energy_dist(&self.inner.z),
        )
    }

    /// Rich Jupyter display: the radial / azimuthal / axial distributions
    /// and the cylinder origin.
    pub fn _repr_html_(&self) -> String {
        use crate::html_repr::{card, esc, kv, num};
        let o = self.inner.origin;
        let body = kv(&[
            ("radius", esc(&univariate_repr(&self.inner.r))),
            ("phi", esc(&univariate_repr(&self.inner.phi))),
            ("z", esc(&univariate_repr(&self.inner.z))),
            (
                "origin",
                format!("[{}, {}, {}]", num(o[0]), num(o[1]), num(o[2])),
            ),
        ]);
        card("CylindricalRing", "", &body)
    }

    pub fn __repr__(&self) -> String {
        let o = self.inner.origin;
        format!(
            "CylindricalRing(radius={}, phi={}, z={}, origin=[{}, {}, {}])",
            univariate_repr(&self.inner.r),
            univariate_repr(&self.inner.phi),
            univariate_repr(&self.inner.z),
            o[0],
            o[1],
            o[2],
        )
    }
}

/// Extract a Univariate distribution from a Python object.
fn extract_univariate(ob: &Bound<'_, PyAny>, name: &str) -> PyResult<Univariate> {
    if let Ok(d) = ob.extract::<PyDiscrete>() {
        Ok(Univariate::Discrete(d.inner.clone()))
    } else if let Ok(u) = ob.extract::<PyUniform>() {
        Ok(Univariate::Uniform(u.inner.clone()))
    } else if let Ok(n) = ob.extract::<PyNormal>() {
        Ok(Univariate::Normal(n.inner.clone()))
    } else if let Ok(val) = ob.extract::<f64>() {
        yamc_source::distribution::check_finite(name, val)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let d = yamc_source::distribution::energy::Discrete::new(vec![val], vec![1.0]).map_err(
            |e| pyo3::exceptions::PyValueError::new_err(format!("Invalid {name} value: {e}")),
        )?;
        Ok(Univariate::Discrete(d))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a Discrete, Uniform, or Normal distribution, or a float"
        )))
    }
}

fn univariate_to_energy_dist(u: &Univariate) -> SourceEnergyDistribution {
    match u {
        Univariate::Discrete(d) => SourceEnergyDistribution::Discrete(d.clone()),
        Univariate::Uniform(u) => SourceEnergyDistribution::Uniform(u.clone()),
        Univariate::Normal(n) => SourceEnergyDistribution::Normal(n.clone()),
    }
}

fn univariate_repr(u: &Univariate) -> String {
    match u {
        Univariate::Discrete(d) => {
            format!("Discrete({:?}, {:?})", d.energies(), d.probabilities())
        }
        Univariate::Uniform(u) => format!("Uniform({}, {})", u.a(), u.b()),
        Univariate::Normal(n) => format!("Normal({}, {})", n.mean_val(), n.std_dev()),
    }
}

/// Reads a mesh object's elements. Set by the yamc wheel, which owns the mesh
/// classes; unset in the yani wheel, which has no meshes.
pub type MeshElementsReader = for<'py> fn(&Bound<'py, PyAny>) -> PyResult<MeshElements>;

/// The yamc wheel's [`MeshElementsReader`].
pub static MESH_ELEMENTS_READER: std::sync::OnceLock<MeshElementsReader> =
    std::sync::OnceLock::new();

/// A source spread over the elements of a mesh.
///
/// Each element emits in proportion to its strength, from a point uniform
/// within it. Pass it as ``position=`` to ``NeutronSource`` or ``PhotonSource``.
///
/// Args:
///     mesh: A ``RegularRectangularMesh``, ``RegularCylindricalMesh`` or a
///         tet ``MeshGeometry``.
///     strengths: One non-negative strength per element (e.g. decay photons
///         per second in each voxel), in the order a tally on the mesh
///         numbers its bins, so a mesh tally's ``mean`` can be passed
///         directly. For a ``MeshGeometry`` this is one per tetrahedron.
///         Defaults to each element's volume: uniform over the mesh.
///
/// Raises:
///     ValueError: If ``strengths`` has the wrong length, a value that is
///         NaN, infinite or negative, or sums to zero.
///
/// Examples:
///     >>> import yamc
///     >>> mesh = yamc.RegularRectangularMesh([0, 0, 0], [10, 10, 10], [2, 1, 1])
///     >>> space = yamc.sources.MeshDistribution(mesh, strengths=[3.0, 1.0])
///     >>> source = yamc.PhotonSource(position=space, energy=1e6)
#[gen_stub_pyclass]
#[pyclass(name = "MeshDistribution", from_py_object)]
#[derive(Clone)]
pub struct PyMeshDistribution {
    pub inner: MeshDistribution,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMeshDistribution {
    #[new]
    #[pyo3(signature = (mesh, strengths=None))]
    pub fn new(mesh: &Bound<'_, PyAny>, strengths: Option<Vec<f64>>) -> PyResult<Self> {
        let read = MESH_ELEMENTS_READER.get().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("MeshDistribution needs the yamc package")
        })?;
        Ok(Self {
            inner: MeshDistribution::new(read(mesh)?, strengths.as_deref())
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        })
    }

    /// Sample a position from the distribution.
    ///
    /// Returns:
    ///     List[float]: Position vector [x, y, z] in cm.
    pub fn sample(&self) -> [f64; 3] {
        self.inner.sample(&mut rand::rng())
    }

    /// Sum of the element strengths: the source rate when they are rates, to
    /// scale per-source-particle tallies by.
    #[getter]
    pub fn total_strength(&self) -> f64 {
        self.inner.total()
    }

    /// Number of elements in the mesh.
    #[getter]
    pub fn num_elements(&self) -> usize {
        self.inner.num_elements()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MeshDistribution({} of {} elements emitting, total_strength={})",
            self.inner.num_emitting(),
            self.inner.num_elements(),
            self.inner.total()
        )
    }
}
