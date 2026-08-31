//! Python bindings for `SimulationResults` and `TallyResult`.
//!
//! `Model.simulate_transport()` returns a `SimulationResults` that exposes
//! the four access flavors from §3 of the refactor plan:
//!
//! ```python
//! results[tally]       # by Tally object (always works)
//! results["flux"]      # by name (only if Tally.name set)
//! results[2]           # by int id (only if Tally.id set)
//! for tally, r in results.items():
//!     ...
//! ```
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyKeyError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use yamc_tallies::result::TallyResult;
use yamc_tallies::simulation_results::{RunProvenance, SimulationResults};
use yamc_tallies::{ConvergencePoint, ScorePdf, StatisticalChecks};

use crate::simulation::PyTracks;
use crate::tally::PyTally;

/// Python-visible wrapper over `Arc<TallyResult>`.
#[gen_stub_pyclass]
#[pyclass(
    module = "yamc._core",
    name = "TallyResult",
    unsendable,
    from_py_object
)]
#[derive(Clone)]
pub struct PyTallyResult {
    pub inner: Arc<TallyResult>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTallyResult {
    /// Mean value per bin, flat row-major over `shape`.
    #[getter]
    pub fn mean(&self) -> Vec<f64> {
        self.inner.mean.clone()
    }

    /// Standard error of the mean per bin.
    #[getter]
    pub fn standard_deviation(&self) -> Vec<f64> {
        self.inner.standard_deviation.clone()
    }

    /// Relative error (std_dev / mean) per bin.
    #[getter]
    pub fn relative_error(&self) -> Vec<f64> {
        self.inner.relative_error.clone()
    }

    /// Variance of the mean per bin (``standard_deviation**2``), computed
    /// on the fly. This is the variance of the mean estimate -- the square
    /// of the reported uncertainty.
    #[getter]
    pub fn variance(&self) -> Vec<f64> {
        self.inner.variance()
    }

    /// Total count per bin: `mean * particles_per_chunk * n_batches`.
    #[getter]
    pub fn total_count(&self) -> Vec<u64> {
        self.inner.total_count.clone()
    }

    /// Per-dimension sizes (`score → nuclide → energy → mesh_z/y/x` as applicable).
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    /// Dimension names matching `shape` axes.
    #[getter]
    pub fn dim_labels(&self) -> Vec<String> {
        self.inner.dim_labels.clone()
    }

    /// Number of batches accumulated.
    #[getter]
    pub fn n_batches(&self) -> u32 {
        self.inner.n_batches
    }

    /// Source particles per batch.
    #[getter]
    pub fn particles_per_chunk(&self) -> u32 {
        self.inner.particles_per_chunk
    }

    /// Raw per-bin Welford sum of squared deviations: the exact merge
    /// state behind ``standard_deviation``. Empty when the producing
    /// path recorded no Welford state (e.g. GPU runs).
    #[getter]
    pub fn m2(&self) -> Vec<f64> {
        self.inner.m2.clone()
    }

    /// Exact total source-history count behind this tally's statistics
    /// (u64; unlike ``n_batches`` it does not saturate).
    #[getter]
    pub fn n_histories(&self) -> u64 {
        self.inner.n_histories
    }

    /// Wall-clock seconds attributed to this tally's data (its producing
    /// run, or for a combined result the sum over contributing runs).
    #[getter]
    pub fn elapsed_secs(&self) -> f64 {
        self.inner.elapsed_secs
    }

    /// Per-bin figure of merit ``1 / (relative_error^2 x elapsed_secs)``.
    #[getter]
    pub fn figure_of_merit(&self) -> Vec<f64> {
        self.inner.figure_of_merit.clone()
    }

    /// Tally-level scalar figure of merit of the integrated response.
    #[getter]
    pub fn aggregate_figure_of_merit(&self) -> f64 {
        self.inner.aggregate_figure_of_merit
    }

    /// Mean of the per-history total score (the tally total).
    #[getter]
    pub fn aggregate_mean(&self) -> f64 {
        self.inner.aggregate_mean()
    }

    /// Standard error of the per-history total mean (captures within-history
    /// correlation between bins).
    #[getter]
    pub fn aggregate_std_dev(&self) -> f64 {
        self.inner.aggregate_std_dev()
    }

    /// Relative error of the per-history total.
    #[getter]
    pub fn aggregate_relative_error(&self) -> f64 {
        self.inner.aggregate_relative_error()
    }

    /// Variance of the variance of the per-history total -- a sensitive
    /// reliability metric (large values flag an unreliable error estimate).
    #[getter]
    pub fn aggregate_variance_of_variance(&self) -> f64 {
        self.inner.aggregate_variance_of_variance()
    }

    /// Skewness of the per-history total score distribution.
    #[getter]
    pub fn aggregate_skewness(&self) -> f64 {
        self.inner.aggregate_skewness()
    }

    /// Excess kurtosis of the per-history total score distribution.
    #[getter]
    pub fn aggregate_kurtosis(&self) -> f64 {
        self.inner.aggregate_kurtosis()
    }

    /// Slope of the large-score PDF tail (the tally's variance is well
    /// defined only when this exceeds 3). 0.0 if not estimable.
    #[getter]
    pub fn aggregate_tail_slope(&self) -> f64 {
        self.inner.aggregate_tail_slope()
    }

    /// The statistical-reliability checks (pass/fail verdict).
    #[getter]
    pub fn statistical_checks(&self) -> PyStatisticalChecks {
        PyStatisticalChecks {
            inner: self.inner.statistical_checks(),
        }
    }

    /// Aggregate statistics versus number of histories (a list of points),
    /// for inspecting convergence. Empty unless the run recorded it.
    #[getter]
    pub fn convergence_history(&self) -> Vec<PyConvergencePoint> {
        self.inner
            .convergence_history
            .iter()
            .map(|&inner| PyConvergencePoint { inner })
            .collect()
    }

    /// Raw empirical probability density of the per-history total score.
    #[getter]
    pub fn score_pdf(&self) -> PyScorePdf {
        PyScorePdf {
            inner: self.inner.score_pdf.clone(),
        }
    }

    /// A human-readable multi-line summary: mean +/- std, figure of merit,
    /// the higher-moment reliability metrics, and the pass/fail checks.
    pub fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Reference back to the input `Tally` config.
    #[getter]
    pub fn tally(&self) -> PyTally {
        PyTally {
            inner: self.inner.tally.clone(),
        }
    }

    /// A tally field reshaped to ``self.shape`` as a numpy ``ndarray``.
    ///
    /// Args:
    ///     field: which array to return -- ``"mean"`` (default),
    ///         ``"standard_deviation"``, ``"relative_error"``,
    ///         ``"variance"``, or ``"total_count"``. The flat getters of the
    ///         same names return the unshaped 1-D lists.
    ///
    /// Requires numpy -- ``pip install numpy``. Raises ``ImportError`` if
    /// numpy is not available, or ``ValueError`` for an unknown ``field``.
    #[pyo3(signature = (field="mean"))]
    pub fn to_numpy<'py>(&self, py: Python<'py>, field: &str) -> PyResult<Bound<'py, PyAny>> {
        let numpy = py.import("numpy").map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(
                "to_numpy() requires numpy. Install with `pip install numpy`.",
            )
        })?;
        let arr = match field {
            "mean" => numpy.call_method1("array", (self.inner.mean.clone(),))?,
            "standard_deviation" => {
                numpy.call_method1("array", (self.inner.standard_deviation.clone(),))?
            }
            "relative_error" => {
                numpy.call_method1("array", (self.inner.relative_error.clone(),))?
            }
            "variance" => numpy.call_method1("array", (self.inner.variance(),))?,
            "total_count" => numpy.call_method1("array", (self.inner.total_count.clone(),))?,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown field {other:?}; expected \"mean\", \"standard_deviation\", \
                     \"relative_error\", \"variance\", or \"total_count\""
                )))
            }
        };
        if !self.inner.shape.is_empty() {
            let shape: Vec<usize> = self.inner.shape.clone();
            arr.call_method1("reshape", (shape,))
        } else {
            Ok(arr)
        }
    }

    /// Rich Jupyter display: a stats panel (mean ± σ, relative error, figure
    /// of merit, shape, history / batch counts) beside the
    /// statistical-reliability checks.
    fn _repr_html_(&self) -> String {
        use crate::html_repr::{card, esc, kv, num};
        let t = PyTally {
            inner: self.inner.tally.clone(),
        };
        let name = t
            .name()
            .or_else(|| t.id().map(|id| format!("#{id}")))
            .unwrap_or_else(|| "result".to_string());
        let shape = {
            let s = self.shape();
            if s.is_empty() {
                "scalar".to_string()
            } else {
                s.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(" × ")
            }
        };
        let stats = kv(&[
            (
                "mean ± σ",
                format!(
                    "{} ± {}",
                    num(self.aggregate_mean()),
                    num(self.aggregate_std_dev())
                ),
            ),
            (
                "relative error",
                format!("{}%", num(self.aggregate_relative_error() * 100.0)),
            ),
            ("figure of merit", num(self.aggregate_figure_of_merit())),
            ("shape", esc(&shape)),
            ("histories", format!("{}", self.inner.n_histories)),
            ("batches", format!("{}", self.inner.n_batches)),
        ]);
        let checks = self.statistical_checks().checks_table();
        let body = format!(
            "<div style=\"display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start;\">\
             <div>{stats}</div><div>{checks}</div></div>"
        );
        card(&format!("TallyResult: {name}"), t.estimator(), &body)
    }

    fn __repr__(&self) -> String {
        self.inner.summary()
    }

    fn __str__(&self) -> String {
        self.inner.summary()
    }
}

/// Python-visible outcome of a tally's statistical-reliability checks.
#[gen_stub_pyclass]
#[pyclass(
    module = "yamc._core",
    name = "StatisticalChecks",
    unsendable,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyStatisticalChecks {
    pub inner: StatisticalChecks,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStatisticalChecks {
    /// True when every evaluated check passed.
    #[getter]
    fn passed(&self) -> bool {
        self.inner.passed()
    }
    /// Number of evaluated checks that passed.
    #[getter]
    fn n_passed(&self) -> usize {
        self.inner.n_passed()
    }
    /// Number of checks that could be evaluated.
    #[getter]
    fn n_evaluated(&self) -> usize {
        self.inner.n_evaluated()
    }
    #[getter]
    fn relative_error_ok(&self) -> Option<bool> {
        self.inner.relative_error_ok
    }
    #[getter]
    fn variance_of_variance_ok(&self) -> Option<bool> {
        self.inner.variance_of_variance_ok
    }
    #[getter]
    fn tail_slope_ok(&self) -> Option<bool> {
        self.inner.tail_slope_ok
    }
    #[getter]
    fn mean_stable(&self) -> Option<bool> {
        self.inner.mean_stable
    }
    #[getter]
    fn relative_error_decreasing(&self) -> Option<bool> {
        self.inner.relative_error_decreasing
    }
    #[getter]
    fn variance_of_variance_decreasing(&self) -> Option<bool> {
        self.inner.variance_of_variance_decreasing
    }
    #[getter]
    fn figure_of_merit_stable(&self) -> Option<bool> {
        self.inner.figure_of_merit_stable
    }
    /// A plain-text pass / FAIL / n/a table.
    fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Rich Jupyter display: a PASS / FAIL / n/a grid of the individual
    /// reliability checks.
    fn _repr_html_(&self) -> String {
        let subtitle = format!(
            "{}/{} passed",
            self.inner.n_passed(),
            self.inner.n_evaluated()
        );
        crate::html_repr::card("Statistical checks", &subtitle, &self.checks_table())
    }

    fn __repr__(&self) -> String {
        self.inner.summary()
    }
}

impl PyStatisticalChecks {
    /// Inner HTML table of the individual checks (no surrounding card),
    /// shared by `TallyResult` and `StatisticalChecks` rich display.
    fn checks_table(&self) -> String {
        use crate::html_repr::{badge, table};
        let c = &self.inner;
        let checks: [(&str, Option<bool>); 7] = [
            ("relative error", c.relative_error_ok),
            ("rel. error decreasing", c.relative_error_decreasing),
            ("variance of variance", c.variance_of_variance_ok),
            ("VoV decreasing", c.variance_of_variance_decreasing),
            ("PDF tail slope", c.tail_slope_ok),
            ("mean stable", c.mean_stable),
            ("figure of merit stable", c.figure_of_merit_stable),
        ];
        let rows: Vec<Vec<String>> = checks
            .iter()
            .map(|(label, ok)| vec![label.to_string(), badge(*ok)])
            .collect();
        table(&["check", "status"], &rows, 1)
    }
}

/// One snapshot of a tally's aggregate statistics versus history count.
#[gen_stub_pyclass]
#[pyclass(
    module = "yamc._core",
    name = "ConvergencePoint",
    unsendable,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyConvergencePoint {
    pub inner: ConvergencePoint,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyConvergencePoint {
    #[getter]
    fn n_histories(&self) -> u64 {
        self.inner.n_histories
    }
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean
    }
    #[getter]
    fn relative_error(&self) -> f64 {
        self.inner.relative_error
    }
    #[getter]
    fn variance_of_variance(&self) -> f64 {
        self.inner.variance_of_variance
    }
    #[getter]
    fn figure_of_merit(&self) -> f64 {
        self.inner.figure_of_merit
    }
    #[getter]
    fn tail_slope(&self) -> f64 {
        self.inner.tail_slope
    }
    fn __repr__(&self) -> String {
        format!(
            "ConvergencePoint(n_histories={}, mean={:.4e}, relative_error={:.4e}, fom={:.3e})",
            self.inner.n_histories,
            self.inner.mean,
            self.inner.relative_error,
            self.inner.figure_of_merit
        )
    }
}

/// Raw empirical probability density of the per-history total score: a
/// log-spaced histogram of `|total|` (`counts` per magnitude bin, with
/// `bin_edges` / `bin_centers`), plus the count of exactly-zero histories.
#[gen_stub_pyclass]
#[pyclass(
    module = "yamc._core",
    name = "ScorePdf",
    unsendable,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyScorePdf {
    pub inner: ScorePdf,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyScorePdf {
    /// Per-bin counts (length = number of magnitude bins). Empty when no
    /// per-history sampling occurred (e.g. GPU runs).
    #[getter]
    fn counts(&self) -> Vec<u64> {
        self.inner.counts.clone()
    }
    /// Lower magnitude edges, one more than `counts` (so bin `i` spans
    /// `bin_edges[i] .. bin_edges[i+1]`).
    #[getter]
    fn bin_edges(&self) -> Vec<f64> {
        if self.inner.counts.is_empty() {
            return Vec::new();
        }
        (0..=self.inner.counts.len())
            .map(ScorePdf::bin_lower_edge)
            .collect()
    }
    /// Geometric center magnitude of each bin.
    #[getter]
    fn bin_centers(&self) -> Vec<f64> {
        (0..self.inner.counts.len())
            .map(|i| (ScorePdf::bin_lower_edge(i) * ScorePdf::bin_lower_edge(i + 1)).sqrt())
            .collect()
    }
    /// Histories whose total score was exactly zero.
    #[getter]
    fn zero(&self) -> u64 {
        self.inner.zero
    }
    /// Slope of the large-score tail (variance exists when > 3); 0 if not
    /// estimable.
    #[getter]
    fn tail_slope(&self) -> f64 {
        self.inner.tail_slope()
    }
    fn __repr__(&self) -> String {
        let populated = self.inner.counts.iter().filter(|&&c| c > 0).count();
        format!(
            "ScorePdf({populated} populated bins, zero={}, tail_slope={:.2})",
            self.inner.zero,
            self.inner.tail_slope()
        )
    }
}

/// Finalized tally results of a simulation run.
///
/// Index by tally object, name, or id (``results[tally]``,
/// ``results["flux"]``, ``results[2]``); iterate with ``items()``.
///
/// See also:
///     - ``yamc.combine_results(r1, r2, ...)`` pools independent runs of
///       the same model (distinct seeds) into one result, statistically
///       identical to a single longer run.
///     - ``to_arrow`` / ``from_arrow`` save and reload results
///       losslessly; reloaded results remain combinable.
///     - ``runs`` lists the provenance (seed, history count, model
///       fingerprint, ...) that combining validates.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "SimulationResults", unsendable)]
pub struct PySimulationResults {
    pub inner: SimulationResults,
    pub tracks: Option<PyTracks>,
    /// Per-run timing captured when these results were built. `None` for
    /// results that were combined (`combine_results`) or reloaded
    /// (`from_arrow`), which do not carry a single run's split timing.
    pub data_load_secs: Option<f64>,
    pub transport_secs: Option<f64>,
    pub particles_per_second: Option<u64>,
    pub total_particles: usize,
}

impl PySimulationResults {
    fn index_for_key(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        // Note: integer (`u32`, treated as tally id) keys are fully handled by
        // `__getitem__` before it ever calls this, so they never reach here.
        // PyTally first (object identity).
        if let Ok(pyt) = key.extract::<PyTally>() {
            let ptr = Arc::as_ptr(&pyt.inner) as usize;
            for (i, r) in self.inner.iter().enumerate() {
                if Arc::as_ptr(&r.tally) as usize == ptr {
                    return Ok(i);
                }
            }
            return Err(PyKeyError::new_err(
                "Tally not found in results (was it passed to simulate()?)",
            ));
        }

        if let Ok(name) = key.extract::<String>() {
            return self.find_index_by_name(&name).ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "no tally named {name:?}; available names: {:?}",
                    self.inner.available_names()
                ))
            });
        }

        Err(PyKeyError::new_err(
            "results[...] key must be a Tally, int (id), or str (name)",
        ))
    }

    fn find_index_by_name(&self, name: &str) -> Option<usize> {
        self.inner
            .iter()
            .position(|r| r.tally.name.as_deref() == Some(name))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PySimulationResults {
    /// Number of tallies in the results.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// `results[tally]` / `results["name"]` / `results[id]`.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyTallyResult> {
        // Non-negative integer could be an id OR a positional index. The
        // plan's API treats integers as `Tally.id` lookup only -- positional
        // access is via iteration. So we match id first.
        if let Ok(id) = key.extract::<u32>() {
            if let Some(r) = self.inner.get_by_id(id) {
                return Ok(PyTallyResult { inner: r.clone() });
            }
            return Err(PyKeyError::new_err(format!(
                "no tally with id {id}; available ids: {:?}",
                self.inner.available_ids()
            )));
        }
        let i = self.index_for_key(key)?;
        let r = self
            .inner
            .get(i)
            .ok_or_else(|| PyIndexError::new_err(format!("index {i} out of range")))?;
        Ok(PyTallyResult { inner: r.clone() })
    }

    /// `tally in results` -- true for either a `Tally` object that was passed
    /// to simulate, a name that was set, or an id that was set.
    fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool {
        if let Ok(pyt) = key.extract::<PyTally>() {
            let ptr = Arc::as_ptr(&pyt.inner) as usize;
            return self
                .inner
                .iter()
                .any(|r| Arc::as_ptr(&r.tally) as usize == ptr);
        }
        if let Ok(id) = key.extract::<u32>() {
            return self.inner.get_by_id(id).is_some();
        }
        if let Ok(name) = key.extract::<String>() {
            return self.inner.get_by_name(&name).is_some();
        }
        false
    }

    /// Iterate over tally configs in input order.
    fn __iter__(slf: PyRef<'_, Self>) -> Py<PyTallyIter> {
        let tallies: Vec<PyTally> = slf
            .inner
            .iter()
            .map(|r| PyTally {
                inner: r.tally.clone(),
            })
            .collect();
        Py::new(
            slf.py(),
            PyTallyIter {
                items: tallies,
                idx: 0,
            },
        )
        .expect("PyTallyIter allocation should not fail")
    }

    /// `for tally, result in results.items(): ...`
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Bound<'py, PyTuple>> = self
            .inner
            .iter()
            .map(|r| {
                let t = PyTally {
                    inner: r.tally.clone(),
                };
                let tr = PyTallyResult { inner: r.clone() };
                PyTuple::new(
                    py,
                    [
                        t.into_pyobject(py)?.into_any(),
                        tr.into_pyobject(py)?.into_any(),
                    ],
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    /// List of `TallyResult` objects in input order.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let values: Vec<PyTallyResult> = self
            .inner
            .iter()
            .map(|r| PyTallyResult { inner: r.clone() })
            .collect();
        PyList::new(py, values)
    }

    /// List of `Tally` configs in input order.
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<PyTally> = self
            .inner
            .iter()
            .map(|r| PyTally {
                inner: r.tally.clone(),
            })
            .collect();
        PyList::new(py, keys)
    }

    /// Number of batches accumulated.
    #[getter]
    fn n_batches(&self) -> u32 {
        self.inner.n_batches
    }

    /// Source particles per batch.
    #[getter]
    fn particles_per_chunk(&self) -> u32 {
        self.inner.particles_per_chunk
    }

    /// Wall-clock elapsed seconds for the simulate call.
    #[getter]
    fn elapsed(&self) -> f64 {
        self.inner.elapsed_secs
    }

    /// Particle tracks captured during the run, or ``None`` if tracking
    /// was not requested via ``simulate_transport(capture_tracks=...)``.
    #[getter]
    fn tracks(&self) -> Option<PyTracks> {
        self.tracks.clone()
    }

    /// Time spent loading nuclear data and preparing materials (seconds), or
    /// ``None`` for combined / reloaded results.
    #[getter]
    fn data_load_time(&self) -> Option<f64> {
        self.data_load_secs
    }

    /// Time spent in particle transport only (seconds), or ``None`` for
    /// combined / reloaded results.
    #[getter]
    fn transport_time(&self) -> Option<f64> {
        self.transport_secs
    }

    /// Particles per second including data loading, or ``None`` for combined /
    /// reloaded results.
    #[getter]
    fn particles_per_second(&self) -> Option<u64> {
        self.particles_per_second
    }

    /// Particles per second for transport only (excludes data loading), or
    /// ``None`` for combined / reloaded results.
    #[getter]
    fn transport_particles_per_second(&self) -> Option<u64> {
        let secs = self.transport_secs?;
        if secs <= 0.0 {
            return None;
        }
        Some((self.total_particles as f64 / secs) as u64)
    }

    /// Provenance of the run(s) behind these results: a list of dicts
    /// with ``seed``, ``n_histories``, ``elapsed_secs``, ``fingerprint``,
    /// ``data_libraries``, ``compute``, ``yamc_version``, ``mpi_size``
    /// and ``mpi_rank``. One entry per ``simulate_transport`` run;
    /// concatenated by ``combine_results``.
    #[getter]
    fn runs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Bound<'py, PyDict>> = self
            .inner
            .runs
            .iter()
            .map(|run| -> PyResult<Bound<'py, PyDict>> {
                let d = PyDict::new(py);
                d.set_item("seed", run.seed)?;
                d.set_item("n_histories", run.n_histories)?;
                d.set_item("elapsed_secs", run.elapsed_secs)?;
                d.set_item("fingerprint", run.fingerprint.clone())?;
                d.set_item("data_libraries", run.data_libraries.clone())?;
                d.set_item("compute", run.compute.clone())?;
                d.set_item("yamc_version", run.yamc_version.clone())?;
                d.set_item("mpi_size", run.mpi_size)?;
                d.set_item("mpi_rank", run.mpi_rank)?;
                Ok(d)
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    /// Pool this result with one or more other independent runs of the same
    /// model into one combined result, statistically identical to a single
    /// longer run. Equivalent to ``yamc.combine_results(self, *others)`` and
    /// to ``self + other``; see ``combine_results`` for the validation rules.
    ///
    /// Args:
    ///     *others: further ``SimulationResults`` from distinct-seed runs of
    ///         the same model.
    #[pyo3(signature = (*others))]
    fn combine<'py>(
        &self,
        py: Python<'py>,
        others: &Bound<'py, PyTuple>,
    ) -> PyResult<PySimulationResults> {
        let borrowed: Vec<PyRef<'py, PySimulationResults>> = others
            .iter()
            .map(|item| {
                item.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "combine() arguments must be SimulationResults",
                    )
                })
            })
            .collect::<PyResult<_>>()?;
        let mut inputs: Vec<&SimulationResults> = Vec::with_capacity(borrowed.len() + 1);
        inputs.push(&self.inner);
        inputs.extend(borrowed.iter().map(|b| &b.inner));
        combine_inner(py, &inputs)
    }

    /// ``results + other`` pools two independent runs (see ``combine``).
    fn __add__<'py>(
        &self,
        py: Python<'py>,
        other: PyRef<'py, PySimulationResults>,
    ) -> PyResult<PySimulationResults> {
        combine_inner(py, &[&self.inner, &other.inner])
    }

    /// Write to an Arrow IPC file at ``path``.
    ///
    /// The round trip is lossless: numeric data, the raw Welford merge
    /// state (per-bin ``m2`` and history counts), shapes, the full
    /// ``Tally`` configurations and the run provenance all survive, so a
    /// reloaded result remains fully combinable via ``combine_results``.
    /// The file is plain Arrow IPC, directly readable with
    /// pyarrow/polars. Captured particle tracks are NOT serialized.
    fn to_arrow(&self, path: &str) -> PyResult<()> {
        self.inner
            .to_arrow(std::path::Path::new(path))
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Read a ``SimulationResults`` from an Arrow IPC file previously
    /// written by ``to_arrow``.
    #[classmethod]
    fn from_arrow(_cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Self> {
        let inner = yamc_tallies::simulation_results::SimulationResults::from_arrow(
            std::path::Path::new(path),
        )
        .map_err(pyo3::exceptions::PyIOError::new_err)?;
        Ok(Self {
            inner,
            tracks: None,
            data_load_secs: None,
            transport_secs: None,
            particles_per_second: None,
            total_particles: 0,
        })
    }

    /// Rich Jupyter display: one row per tally (name, shape, integrated mean,
    /// relative error, and the statistical-reliability verdict), with an
    /// overall PASS / FAIL badge. Surfaces the reliability metrics yamc
    /// computes for every tally directly in the notebook.
    fn _repr_html_(&self) -> String {
        use crate::html_repr::{badge, card, esc, num, table};
        let n = self.inner.len();
        if n == 0 {
            return card(
                "SimulationResults",
                "empty",
                "<em style=\"color:#656d76;\">no tallies</em>",
            );
        }
        let mut rows: Vec<Vec<String>> = Vec::with_capacity(n);
        let mut any_eval = false;
        let mut all_pass = true;
        for (i, r) in self.inner.iter().enumerate() {
            let tr = PyTallyResult { inner: r.clone() };
            let t = PyTally {
                inner: r.tally.clone(),
            };
            let name = t
                .name()
                .or_else(|| t.id().map(|id| format!("#{id}")))
                .unwrap_or_else(|| format!("[{i}]"));
            let shape = {
                let s = tr.shape();
                if s.is_empty() {
                    "scalar".to_string()
                } else {
                    s.iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(" × ")
                }
            };
            let checks = tr.statistical_checks();
            let check_cell = if checks.n_evaluated() == 0 {
                badge(None)
            } else {
                any_eval = true;
                let passed = checks.passed();
                if !passed {
                    all_pass = false;
                }
                format!(
                    "{} <span style=\"color:#656d76;font-size:11px;\">{}/{}</span>",
                    badge(Some(passed)),
                    checks.n_passed(),
                    checks.n_evaluated()
                )
            };
            rows.push(vec![
                esc(&name),
                esc(&shape),
                num(tr.aggregate_mean()),
                format!("{}%", num(tr.aggregate_relative_error() * 100.0)),
                check_cell,
            ]);
        }
        let tword = if n == 1 { "tally" } else { "tallies" };
        let subtitle = format!(
            "{n} {tword} · {} batches · {:.3}s",
            self.inner.n_batches, self.inner.elapsed_secs
        );
        let tbl = table(&["tally", "shape", "mean", "rel. err", "checks"], &rows, 2);
        let body = if any_eval {
            format!(
                "<div style=\"margin-bottom:6px;\">overall reliability: {}</div>{tbl}",
                badge(Some(all_pass))
            )
        } else {
            tbl
        };
        card("SimulationResults", &subtitle, &body)
    }

    fn __repr__(&self) -> String {
        format!(
            "SimulationResults(n_tallies={}, n_batches={}, elapsed_secs={:.3})",
            self.inner.len(),
            self.inner.n_batches,
            self.inner.elapsed_secs
        )
    }
}

/// Iterator over the `Tally` configs in a `SimulationResults`.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "TallyIter", unsendable)]
pub struct PyTallyIter {
    items: Vec<PyTally>,
    idx: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTallyIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyTally> {
        if slf.idx < slf.items.len() {
            let i = slf.idx;
            slf.idx += 1;
            Some(slf.items[i].clone())
        } else {
            None
        }
    }
}

/// The version to stamp into run provenance: the installed `yamc-core`
/// distribution, not this crate's compile-time version.
///
/// A results file is provenance for whoever reads it later, so the number in
/// it has to be one they can act on -- cite in a paper, or reinstall to
/// reproduce the run -- and that is the number `pip` knows about. Asking the
/// installed metadata also keeps it right when the two diverge, which is what
/// a wheel built with an overridden version does.
///
/// Falls back to the compile-time version when there is no installed
/// distribution to ask: running against a source tree, or an embedded
/// interpreter that never saw a wheel.
fn provenance_version() -> String {
    Python::attach(|py| {
        py.import("importlib.metadata")
            .and_then(|m| m.call_method1("version", ("yamc-core",)))
            .and_then(|v| v.extract::<String>())
            .unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string())
    })
}

/// Entry point used by `PyModel.simulate_transport()`. Builds the results
/// from the model's current tallies with run provenance attached (seed,
/// model fingerprint, data libraries, compute path, MPI placement), so the
/// returned results are combinable via `combine_results`. Surfaces
/// duplicate-name / duplicate-id validation as `ValueError`.
pub(crate) fn build_results(
    model: &yamc::model::Model,
    settings: &yamc::model::TransportSettings,
    compute: &str,
    elapsed_secs: f64,
    tracks: Option<PyTracks>,
) -> PyResult<PySimulationResults> {
    let fingerprint = model
        .fingerprint()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    // Histories actually run (equals `total_particles` on a normal capped
    // run; fewer on an early stop; the whole count on an uncapped run). Used
    // for both the run provenance and the reported particle total, so the
    // latter is meaningful when `settings.total_particles` is None.
    let actual_histories = model
        .tallies
        .first()
        .map(|t| t.get_n_histories())
        .unwrap_or(0);
    let run = RunProvenance {
        seed: settings.seed,
        n_histories: actual_histories,
        elapsed_secs,
        fingerprint,
        data_libraries: model.data_libraries(),
        compute: compute.to_string(),
        yamc_version: provenance_version(),
        mpi_size: yamc::mpi_context::mpi_size(),
        mpi_rank: yamc::mpi_context::mpi_rank(),
    };
    match SimulationResults::from_tallies_with_run(&model.tallies, elapsed_secs, run) {
        Ok(r) => Ok(PySimulationResults {
            inner: r,
            tracks,
            data_load_secs: model.last_data_load_secs,
            transport_secs: model.last_transport_secs,
            particles_per_second: model.last_particles_per_second,
            total_particles: actual_histories as usize,
        }),
        Err(msg) => Err(pyo3::exceptions::PyValueError::new_err(msg)),
    }
}

/// Run the Rust combine and surface its pass-through warnings as Python
/// `UserWarning`s.
pub(crate) fn combine_inner(
    py: Python<'_>,
    inputs: &[&SimulationResults],
) -> PyResult<PySimulationResults> {
    let (combined, warnings) = yamc_tallies::combine::combine_results(inputs)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    if !warnings.is_empty() {
        let warnings_mod = py.import("warnings")?;
        for w in &warnings {
            warnings_mod.call_method1("warn", (w.as_str(),))?;
        }
    }
    Ok(PySimulationResults {
        inner: combined,
        tracks: None,
        data_load_secs: None,
        transport_secs: None,
        particles_per_second: None,
        total_particles: 0,
    })
}

/// Combine results of independent runs of the same model into one pooled
/// result, statistically identical to a single longer run.
///
/// The merge pools the raw per-bin Welford state ``(mean, m2, n)`` of
/// every tally, so means, standard deviations and figures of merit come
/// out exactly as if all histories had been run in one simulation (FOM
/// uses the summed wall-clock time).
///
/// Validation (raises ``ValueError``, never silently combines):
///
/// - every input must carry run provenance (results from
///   ``simulate_transport`` or ``from_arrow``);
/// - all runs must have **distinct seeds** -- same-seed runs share their
///   per-particle RNG streams and are not independent;
/// - all runs must come from the **same model** (geometry, materials,
///   source, physics settings and nuclear-data libraries -- library
///   mismatches are reported per nuclide);
/// - tallies with the same name must have identical configurations;
/// - GPU-produced and non-root MPI results are refused (no complete
///   Welford merge state).
///
/// Tallies present in only some inputs are carried through unchanged
/// with a ``UserWarning`` (their statistics come from the runs that
/// scored them; their FOM uses those runs' time only).
///
/// Captured particle tracks are not merged: the combined result has
/// ``tracks=None``. Keep the individual results if you need their
/// tracks.
///
/// Examples:
///     >>> r1 = model.simulate_transport(seed=1)
///     >>> r2 = model.simulate_transport(seed=2)
///     >>> combined = yamc.combine_results(r1, r2)
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (*results))]
pub fn combine_results<'py>(
    py: Python<'py>,
    results: &Bound<'py, PyTuple>,
) -> PyResult<PySimulationResults> {
    let mut borrowed: Vec<PyRef<'py, PySimulationResults>> = Vec::with_capacity(results.len());
    for item in results.iter() {
        borrowed.push(item.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "combine_results arguments must be SimulationResults objects",
            )
        })?);
    }
    let inputs: Vec<&SimulationResults> = borrowed.iter().map(|p| &p.inner).collect();
    combine_inner(py, &inputs)
}
