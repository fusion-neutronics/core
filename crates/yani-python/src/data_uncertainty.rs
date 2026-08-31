//! `DataUncertainty`: ask a transmutation for nuclear-data uncertainty.
//!
//! Optional everywhere it appears. Without it nothing is read, folded,
//! factorized or sampled, and the inventories are bit-identical to a build that
//! never had this feature.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use yani_transmute::uncertainty::{DataUncertainty, Info, Source};

/// Request nuclear-data uncertainty on a transmutation.
///
/// Pass one to :meth:`Material.transmute` and the result carries a standard
/// deviation on every nuclide density alongside the mean.
///
/// What it covers is the **activation cross sections**, sampled from the ENDF
/// MF=33 covariance folded against this material's own spectrum. Half-lives,
/// decay branching ratios, fission yields and the isomeric-branching overlay
/// are held at their evaluated values; they carry uncertainties of their own
/// that this does not propagate. ``TransmutationResults.data_uncertainty_info``
/// says so per run, along with any nuclide whose evaluation carries no
/// covariance at all.
///
/// Args:
///     seed (int): Base seed. A given nuclide's perturbation in a given replica
///         is a pure function of ``(seed, replica, nuclide)``, so the same seed
///         reproduces the same answer regardless of replica count, iteration
///         order, or what else is in the material.
///     samples (int, optional): Fixed replica count. Leave as ``None`` (the
///         default) to let the solver add replicas until the reported standard
///         deviations stop moving. A number here bounds cost or reproduces a
///         specific run; it is not an accuracy dial.
///     sources (list[str], optional): Which inputs to perturb. ``None`` (the
///         default) means every source this build implements. Restricting it is
///         how a run isolates one contribution, so that adding a source and
///         watching the inventory sigma grow is a measurement rather than a
///         guess.
///
///         Naming a source this build cannot perturb **raises**, rather than
///         being ignored. A source that has not landed yet must not look like
///         one that contributed nothing. ``DataUncertainty.available_sources()``
///         lists what there is.
///
/// Examples:
///     >>> results = iron.transmute(
///     ...     schedule=sched,
///     ...     data_uncertainty=yamc.DataUncertainty(seed=42),
///     ... )
///     >>> results.get_nuclide_density(iron.id or 0, "Mn56", 1)
///     1.234e-08
///     >>> results.get_nuclide_uncertainty(iron.id or 0, "Mn56", 1)
///     8.7e-10
#[gen_stub_pyclass]
// `from_py_object` because this is taken as an ARGUMENT to
// `Material.transmute`, so it has to convert back out of Python.
#[pyclass(name = "DataUncertainty", from_py_object)]
#[derive(Clone)]
pub struct PyDataUncertainty {
    pub inner: DataUncertainty,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDataUncertainty {
    #[new]
    #[pyo3(signature = (seed = 1, samples = None, sources = None))]
    fn new(seed: u64, samples: Option<usize>, sources: Option<Vec<String>>) -> PyResult<Self> {
        if samples == Some(0) {
            return Err(PyValueError::new_err(
                "samples must be at least 1; pass samples=None to let the solver \
                 choose when the standard deviations have settled",
            ));
        }
        let sources = match sources {
            None => Source::IMPLEMENTED.to_vec(),
            Some(names) => {
                if names.is_empty() {
                    return Err(PyValueError::new_err(
                        "sources must name at least one source; pass sources=None \
                         for every source this build implements",
                    ));
                }
                names
                    .iter()
                    .map(|n| Source::parse(n).map_err(PyValueError::new_err))
                    .collect::<PyResult<Vec<_>>>()?
            }
        };
        Ok(Self {
            inner: DataUncertainty {
                seed,
                samples,
                sources,
            },
        })
    }

    /// The uncertainty sources this build can perturb.
    ///
    /// Returns:
    ///     list[str]: Names accepted by the ``sources`` argument.
    #[staticmethod]
    fn available_sources() -> Vec<String> {
        Source::IMPLEMENTED
            .iter()
            .map(|s| s.name().to_string())
            .collect()
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn samples(&self) -> Option<usize> {
        self.inner.samples
    }

    #[getter]
    fn sources(&self) -> Vec<String> {
        self.inner
            .sources
            .iter()
            .map(|s| s.name().to_string())
            .collect()
    }

    fn __repr__(&self) -> String {
        let samples = match self.inner.samples {
            Some(n) => n.to_string(),
            None => "None".to_string(),
        };
        format!(
            "DataUncertainty(seed={}, samples={samples}, sources={:?})",
            self.inner.seed,
            self.sources(),
        )
    }
}

/// Render an [`Info`] as a plain dict.
///
/// A dict rather than a class because it is a report, not an interface: a
/// caller prints it, logs it, or checks one key. Every entry exists so that a
/// gap is visible, since a nuclide with no published covariance and a nuclide
/// whose covariance is genuinely zero would otherwise both read as a confident
/// zero.
pub fn info_to_dict<'py>(py: Python<'py>, info: &Info) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("samples", info.samples)?;
    d.set_item("converged", info.converged)?;
    d.set_item(
        "perturbed",
        info.perturbed.iter().cloned().collect::<Vec<_>>(),
    )?;
    d.set_item(
        "no_covariance_data",
        info.no_covariance_data.iter().cloned().collect::<Vec<_>>(),
    )?;
    d.set_item("skipped_cross_material", info.skipped_cross_material)?;
    d.set_item("skipped_nc", info.skipped_nc)?;

    let layouts = PyDict::new(py);
    for (lb, n) in &info.unsupported_layouts {
        layouts.set_item(lb, n)?;
    }
    d.set_item("unsupported_layouts", layouts)?;
    d.set_item("malformed_blocks", info.malformed_blocks)?;

    // Keyed "Nuclide (n,gamma)" rather than by a tuple, so the dict survives
    // JSON and a glance.
    let covered = PyDict::new(py);
    for ((nuclide, kind), fraction) in &info.rate_fraction_covered {
        covered.set_item(format!("{nuclide} {kind}"), fraction)?;
    }
    d.set_item("rate_fraction_covered", covered)?;

    // The one number that says whether the sigmas above are a spread over the
    // answer or over a corner of it. `None` for a decay-only schedule, which
    // drove no production and so has no share to report.
    d.set_item(
        "rate_fraction_covered_total",
        info.rate_fraction_covered_total,
    )?;

    d.set_item("matrices_clipped", info.matrices_clipped)?;
    d.set_item("worst_relative_clip", info.worst_relative_clip)?;
    d.set_item("rates_floored", info.rates_floored)?;
    d.set_item("rates_sampled", info.rates_sampled)?;
    d.set_item("spectra_with_flux_sigma", info.spectra_with_flux_sigma)?;
    d.set_item(
        "spectra_without_flux_sigma",
        info.spectra_without_flux_sigma,
    )?;
    d.set_item("flux_bins_floored", info.flux_bins_floored)?;
    d.set_item("flux_bins_sampled", info.flux_bins_sampled)?;
    d.set_item("not_perturbed", info.not_perturbed.clone())?;
    d.set_item("sources", info.sources.clone())?;
    d.set_item("has_gaps", info.has_gaps())?;
    Ok(d)
}
