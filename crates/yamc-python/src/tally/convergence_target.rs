//! Python binding for precision-based stopping criteria (`ConvergenceTarget`).

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use yamc_tallies::{ConvergenceMetric, ConvergenceTarget, TallySelector};

/// A precision-based stopping criterion. Attach to ``model.convergence_targets``
/// to stop a (single-process) run early once a tally is precise enough.
///
/// ```python
/// model.convergence_targets = [ConvergenceTarget("relative_error", 0.02, tally="flux")]
/// ```
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "ConvergenceTarget", from_py_object)]
#[derive(Clone)]
pub struct PyConvergenceTarget {
    pub inner: ConvergenceTarget,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyConvergenceTarget {
    /// ``metric`` is one of ``"relative_error"``, ``"std_dev"``, or
    /// ``"variance_of_variance"``. Restrict to one tally with ``tally``
    /// (name) or ``tally_id``; otherwise the threshold must hold for every
    /// tally.
    #[new]
    #[pyo3(signature = (metric, threshold, tally=None, tally_id=None))]
    fn new(
        metric: &str,
        threshold: f64,
        tally: Option<String>,
        tally_id: Option<u32>,
    ) -> PyResult<Self> {
        let metric = match metric.to_ascii_lowercase().as_str() {
            "relative_error" | "rel_err" | "rel" => ConvergenceMetric::RelativeError,
            "std_dev" | "standard_deviation" | "std" => ConvergenceMetric::StandardDeviation,
            "variance_of_variance" | "vov" => ConvergenceMetric::VarianceOfVariance,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown convergence metric {other:?}; expected \"relative_error\", \
                     \"std_dev\", or \"variance_of_variance\""
                )))
            }
        };
        let tally = match (tally, tally_id) {
            (Some(name), _) => Some(TallySelector::Name(name)),
            (None, Some(id)) => Some(TallySelector::Id(id)),
            (None, None) => None,
        };
        Ok(Self {
            inner: ConvergenceTarget {
                metric,
                threshold,
                tally,
            },
        })
    }

    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    #[getter]
    fn metric(&self) -> &'static str {
        match self.inner.metric {
            ConvergenceMetric::RelativeError => "relative_error",
            ConvergenceMetric::StandardDeviation => "std_dev",
            ConvergenceMetric::VarianceOfVariance => "variance_of_variance",
        }
    }

    fn __repr__(&self) -> String {
        let tally = match &self.inner.tally {
            None => "all".to_string(),
            Some(TallySelector::Name(n)) => format!("{n:?}"),
            Some(TallySelector::Id(i)) => format!("id {i}"),
        };
        format!(
            "ConvergenceTarget(metric={:?}, threshold={:.3e}, tally={tally})",
            self.metric(),
            self.inner.threshold
        )
    }
}
