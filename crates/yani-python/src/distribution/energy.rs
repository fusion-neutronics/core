use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use yamc_source::distribution::energy::{Discrete, Histogram, Normal, Uniform};

/// Resolve a `boundaries` argument into explicit energy edges (eV).
///
/// Accepts either a named built-in group structure (e.g. `"CCFE-709"`,
/// resolved via [`yamc_nuclide::group_structures::get_group_structure`]) or an
/// explicit ascending list of edges.
fn resolve_group_boundaries(ob: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    // A named group structure (checked first: a str does not extract as Vec<f64>).
    if let Ok(name) = ob.extract::<String>() {
        return yamc_nuclide::group_structures::get_group_structure(&name)
            .map(|edges| edges.to_vec())
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>);
    }
    if let Ok(edges) = ob.extract::<Vec<f64>>() {
        return Ok(edges);
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "boundaries must be a list of energy edges (eV, ascending) or the name \
         of a built-in group structure (e.g. 'CCFE-709', 'VITAMIN-J-175')",
    ))
}

/// Discrete energy distribution with specific energies and their probabilities.
#[gen_stub_pyclass]
#[pyclass(name = "Discrete", from_py_object)]
#[derive(Clone)]
pub struct PyDiscrete {
    pub inner: Discrete,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDiscrete {
    /// Create a discrete energy distribution.
    ///
    /// Args:
    ///     energies: List of energy values in eV.
    ///     probabilities: Relative probability for each energy. Normalized
    ///         internally, so only the relative values matter.
    ///
    /// Returns:
    ///     Discrete: A discrete energy distribution.
    ///
    /// Raises:
    ///     ValueError: If energies and probabilities have different lengths, if
    ///                 any probability is negative, or if all are zero.
    ///
    /// Examples:
    ///     >>> import yamc
    ///     >>> # 14.06 MeV neutrons (D-T fusion)
    ///     >>> energy_dist = yamc.Discrete([14.06e6], [1.0])
    #[new]
    pub fn new(energies: Vec<f64>, probabilities: Vec<f64>) -> PyResult<Self> {
        match Discrete::new(energies, probabilities) {
            Ok(dist) => Ok(PyDiscrete { inner: dist }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        }
    }

    /// Sample an energy value from the distribution.
    ///
    /// Returns:
    ///     float: Sampled energy in eV.
    pub fn sample(&self) -> f64 {
        let mut rng = rand::rng();
        self.inner.sample(&mut rng)
    }

    /// Get the energy values.
    ///
    /// Returns:
    ///     List[float]: Energy values in eV.
    #[getter]
    pub fn energies(&self) -> Vec<f64> {
        self.inner.energies().to_vec()
    }

    /// Get the probability values.
    ///
    /// Returns:
    ///     List[float]: Probability for each energy.
    #[getter]
    pub fn probabilities(&self) -> Vec<f64> {
        self.inner.probabilities().to_vec()
    }

    /// Rich Jupyter display: the (energy, probability) points as a table.
    pub fn _repr_html_(&self) -> String {
        use crate::html_repr::{card, num, table};
        let e = self.energies();
        let p = self.probabilities();
        let limit = 6usize;
        let mut rows: Vec<Vec<String>> = Vec::new();
        for i in 0..e.len().min(limit) {
            rows.push(vec![num(e[i]), num(p[i])]);
        }
        let mut body = table(&["energy (eV)", "probability"], &rows, 0);
        if e.len() > limit {
            body.push_str(&format!(
                "<div style=\"color:#656d76;font-size:11px;margin-top:4px;\">… {} more</div>",
                e.len() - limit
            ));
        }
        card("Discrete", &format!("{} point(s)", e.len()), &body)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Discrete(energies={:?}, probabilities={:?})",
            self.inner.energies(),
            self.inner.probabilities()
        )
    }
}

/// Histogram (piecewise-constant) energy distribution over energy bins.
#[gen_stub_pyclass]
#[pyclass(name = "Histogram", from_py_object)]
#[derive(Clone)]
pub struct PyHistogram {
    pub inner: Histogram,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHistogram {
    /// Create a histogram (piecewise-constant) energy distribution.
    ///
    /// Each bin ``i`` spans ``boundaries[i]`` to ``boundaries[i + 1]`` and holds
    /// probability mass ``probabilities[i]``; an energy is sampled by choosing a
    /// bin in proportion to its mass and then drawing uniformly within it.
    ///
    /// Args:
    ///     boundaries: Either explicit energy bin edges in eV (length ``n + 1``,
    ///         strictly ascending) or the name of a built-in group structure
    ///         (e.g. ``"CCFE-709"``, ``"VITAMIN-J-175"``), whose edges are used.
    ///     probabilities: Relative probability mass for each of the ``n`` bins.
    ///         Normalized internally, so only the relative values matter (a raw
    ///         multigroup flux per group can be passed directly).
    ///
    /// Returns:
    ///     Histogram: A histogram energy distribution.
    ///
    /// Raises:
    ///     ValueError: If a named group structure is unknown, if ``boundaries``
    ///         is not exactly one longer than ``probabilities``, if the
    ///         boundaries are not strictly ascending, if any probability is
    ///         negative, or if all are zero.
    ///
    /// Examples:
    ///     >>> import yamc
    ///     >>> # 30% of source neutrons in [0, 1] MeV, 70% in [1, 20] MeV
    ///     >>> energy_dist = yamc.sources.Histogram([0.0, 1e6, 20e6], [0.3, 0.7])
    ///     >>> # or from a named group structure (709 per-group weights)
    ///     >>> energy_dist = yamc.sources.Histogram("CCFE-709", flux_709)
    #[new]
    pub fn new(boundaries: &Bound<'_, PyAny>, probabilities: Vec<f64>) -> PyResult<Self> {
        let edges = resolve_group_boundaries(boundaries)?;
        match Histogram::new(edges, probabilities) {
            Ok(dist) => Ok(PyHistogram { inner: dist }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        }
    }

    /// Sample an energy value from the distribution.
    ///
    /// Returns:
    ///     float: Sampled energy in eV.
    pub fn sample(&self) -> f64 {
        let mut rng = rand::rng();
        self.inner.sample(&mut rng)
    }

    /// Get the bin boundaries.
    ///
    /// Returns:
    ///     List[float]: Energy bin edges in eV, length ``n + 1``.
    #[getter]
    pub fn boundaries(&self) -> Vec<f64> {
        self.inner.boundaries().to_vec()
    }

    /// Get the per-bin probabilities, as supplied.
    ///
    /// Returns:
    ///     List[float]: Relative probability mass for each bin.
    #[getter]
    pub fn probabilities(&self) -> Vec<f64> {
        self.inner.probabilities().to_vec()
    }

    /// Rich Jupyter display: the per-bin (range, probability) rows as a table.
    pub fn _repr_html_(&self) -> String {
        use crate::html_repr::{card, num, table};
        let b = self.boundaries();
        let p = self.probabilities();
        let limit = 6usize;
        let mut rows: Vec<Vec<String>> = Vec::new();
        for i in 0..p.len().min(limit) {
            rows.push(vec![
                format!("{} - {}", num(b[i]), num(b[i + 1])),
                num(p[i]),
            ]);
        }
        let mut body = table(&["energy bin (eV)", "probability"], &rows, 0);
        if p.len() > limit {
            body.push_str(&format!(
                "<div style=\"color:#656d76;font-size:11px;margin-top:4px;\">… {} more</div>",
                p.len() - limit
            ));
        }
        card("Histogram", &format!("{} bin(s)", p.len()), &body)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Histogram(boundaries={:?}, probabilities={:?})",
            self.inner.boundaries(),
            self.inner.probabilities()
        )
    }
}

/// Uniform energy distribution between two bounds.
#[gen_stub_pyclass]
#[pyclass(name = "Uniform", from_py_object)]
#[derive(Clone)]
pub struct PyUniform {
    pub inner: Uniform,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyUniform {
    /// Create a uniform energy distribution.
    ///
    /// Args:
    ///     low: Lower energy bound in eV.
    ///     high: Upper energy bound in eV.
    ///
    /// Returns:
    ///     Uniform: A uniform energy distribution.
    ///
    /// Raises:
    ///     ValueError: If low >= high (lower bound must be less than upper bound).
    ///
    /// Examples:
    ///     >>> import yamc
    ///     >>> # Uniform distribution from 1 MeV to 20 MeV
    ///     >>> energy_dist = yamc.Uniform(1e6, 20e6)
    #[new]
    pub fn new(low: f64, high: f64) -> PyResult<Self> {
        match Uniform::new(low, high) {
            Ok(dist) => Ok(PyUniform { inner: dist }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        }
    }

    /// Sample an energy value from the distribution.
    ///
    /// Returns:
    ///     float: Sampled energy in eV, uniformly distributed between low and high.
    pub fn sample(&self) -> f64 {
        let mut rng = rand::rng();
        self.inner.sample(&mut rng)
    }

    /// Get the lower energy bound.
    ///
    /// Returns:
    ///     float: Lower bound in eV.
    #[getter]
    pub fn low(&self) -> f64 {
        self.inner.a()
    }

    /// Get the upper energy bound.
    ///
    /// Returns:
    ///     float: Upper bound in eV.
    #[getter]
    pub fn high(&self) -> f64 {
        self.inner.b()
    }

    /// Rich Jupyter display: the low / high bounds.
    pub fn _repr_html_(&self) -> String {
        use crate::html_repr::{card, kv, num};
        let body = kv(&[("low", num(self.inner.a())), ("high", num(self.inner.b()))]);
        card("Uniform", "", &body)
    }

    pub fn __repr__(&self) -> String {
        format!("Uniform(low={}, high={})", self.inner.a(), self.inner.b())
    }
}

/// Normal (Gaussian) energy distribution.
#[gen_stub_pyclass]
#[pyclass(name = "Normal", from_py_object)]
#[derive(Clone)]
pub struct PyNormal {
    pub inner: Normal,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNormal {
    /// Create a normal (Gaussian) energy distribution.
    ///
    /// Args:
    ///     mean: Mean energy in eV.
    ///     std_dev: Standard deviation in eV (must be positive).
    ///
    /// Returns:
    ///     Normal: A normal energy distribution.
    ///
    /// Raises:
    ///     ValueError: If std_dev is not positive.
    ///
    /// Examples:
    ///     >>> import yamc
    ///     >>> # Normal distribution centered at 14.06 MeV
    ///     >>> energy_dist = yamc.Normal(14.06e6, 0.1e6)
    #[new]
    pub fn new(mean: f64, std_dev: f64) -> PyResult<Self> {
        match Normal::new(mean, std_dev) {
            Ok(dist) => Ok(PyNormal { inner: dist }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        }
    }

    /// Sample an energy value from the distribution.
    ///
    /// Returns:
    ///     float: Sampled energy in eV.
    pub fn sample(&self) -> f64 {
        let mut rng = rand::rng();
        self.inner.sample(&mut rng)
    }

    /// Get the mean energy.
    ///
    /// Returns:
    ///     float: Mean energy in eV.
    #[getter]
    pub fn mean(&self) -> f64 {
        self.inner.mean_val()
    }

    /// Get the standard deviation.
    ///
    /// Returns:
    ///     float: Standard deviation in eV.
    #[getter]
    pub fn std_dev(&self) -> f64 {
        self.inner.std_dev()
    }

    /// Rich Jupyter display: the mean and standard deviation.
    pub fn _repr_html_(&self) -> String {
        use crate::html_repr::{card, kv, num};
        let body = kv(&[
            ("mean", num(self.inner.mean_val())),
            ("std_dev", num(self.inner.std_dev())),
        ]);
        card("Normal", "", &body)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Normal(mean={}, std_dev={})",
            self.inner.mean_val(),
            self.inner.std_dev()
        )
    }
}

/// Return a Gaussian energy distribution for fusion neutron emission.
///
/// Computes the mean energy and spectral width using relativistic
/// interpolation formulas from Ballabio et al. (Nucl. Fusion 38, 1998).
///
/// Args:
///     ion_temp: Ion temperature of the plasma in eV.
///     reactants: Fusion reactants, either 'DD' or 'DT' (default: 'DT').
///
/// Returns:
///     Normal: A Normal distribution with mean and standard deviation
///             corresponding to the fusion neutron energy spectrum (in eV).
///
/// Raises:
///     ValueError: If ion_temp is outside [0, 100 keV] or reactants is invalid.
///
/// Examples:
///     >>> import yamc
///     >>> # D-T fusion neutron spectrum at 20 keV ion temperature
///     >>> energy_dist = yamc.fusion_neutron_spectrum(20000.0)
///     >>> # D-D fusion neutron spectrum
///     >>> energy_dist = yamc.fusion_neutron_spectrum(20000.0, reactants='DD')
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "fusion_neutron_spectrum", signature = (ion_temp, reactants="DT"))]
pub fn py_fusion_neutron_spectrum(ion_temp: f64, reactants: &str) -> PyResult<PyNormal> {
    let r = match reactants {
        "DD" => yamc_source::distribution::energy::FusionReactants::DD,
        "DT" => yamc_source::distribution::energy::FusionReactants::DT,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid reactants. Must be 'DD' or 'DT'.",
            ))
        }
    };
    match yamc_source::distribution::energy::fusion_neutron_spectrum(ion_temp, r) {
        Ok(normal) => Ok(PyNormal { inner: normal }),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
    }
}
