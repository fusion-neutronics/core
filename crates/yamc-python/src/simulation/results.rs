//! Python bindings for `TransportResults` and `TallyResult`.
//!
//! `Model.simulate_transport()` returns a `TransportResults` that exposes
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
use yamc_tallies::transport_results::TransportResults;
use yamc_tallies::{ConvergencePoint, ScorePdf, SourceStrength, StatisticalChecks, StrengthPer};

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
    /// Mean value per bin, flat row-major over `shape`, in ``units``.
    #[getter]
    pub fn mean(&self) -> Vec<f64> {
        self.inner.scaled(&self.inner.mean)
    }

    /// Standard error of the mean per bin, in ``units``.
    #[getter]
    pub fn standard_deviation(&self) -> Vec<f64> {
        self.inner.scaled(&self.inner.standard_deviation)
    }

    /// Per score, the unit of ``mean`` and ``standard_deviation``: the
    /// score's own per source particle (``"eV/source-particle"``), per second
    /// or per pulse once a source strength is set (``"eV/s"``), or the unit
    /// given to ``to``.
    #[getter]
    pub fn units(&self) -> Vec<String> {
        self.inner.units()
    }

    /// The source strength as ``(value, per)``, ``per`` being ``"second"`` or
    /// ``"pulse"``, or ``None`` for results per source particle.
    #[getter]
    pub fn source_strength(&self) -> Option<(f64, &'static str)> {
        self.inner
            .normalisation
            .source_strength
            .map(|s| (s.value, s.per.name()))
    }

    /// This result scaled to a source strength: source particles per second
    /// (``per="second"``, a steady source) or per pulse (``per="pulse"``).
    /// Transport is linear in the source, so no new run is needed.
    ///
    /// Returns a new result. Calling it again replaces the strength rather
    /// than multiplying by it, and ``None`` returns to values per source
    /// particle. ``mean`` and ``standard_deviation`` scale; relative errors
    /// and the statistical checks do not change.
    ///
    /// Args:
    ///     strength: Source particles per second or per pulse, or None.
    ///     per: ``"second"`` (default) or ``"pulse"``.
    #[pyo3(signature = (strength, per="second"))]
    pub fn with_source_strength(&self, strength: Option<f64>, per: &str) -> PyResult<Self> {
        let strength = parse_strength(strength, per)?;
        self.inner
            .with_source_strength(strength)
            .map(|r| Self { inner: Arc::new(r) })
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// This result in ``units``: one unit for every score, or a list with
    /// one per score.
    ///
    /// Energies convert between eV, keV, MeV and J (W per second once a
    /// strength per second is set, J per pulse for a strength per pulse).
    /// Times are ``s`` (with SI prefixes), ``min``, ``h``, ``d`` and ``a`` for
    /// a year (annum, 365.25 days), so a dose rate is ``"uSv/h"`` or
    /// ``"mSv/a"``.
    ///
    /// A per-volume unit (``"W/cm3"``, ``"1/(cm2 s)"``) divides each bin by
    /// its volume: the mesh element, or the cell when the cell has a
    /// ``volume``. A per-mass unit (``"W/g"``, ``"Gy/s"``: a gray is a J/kg)
    /// divides each bin by its mass, which a cell with a ``volume`` and a
    /// material has; a mesh element can straddle materials, so has none. A
    /// unit that needs a source strength the result does not have raises
    /// ``ValueError`` saying so.
    ///
    /// Args:
    ///     units: The unit, e.g. ``"W/cm3"``, ``"MeV"``, ``"Sv/h"``, ``"Gy/s"``.
    ///
    /// Examples:
    ///     >>> heating = results.with_source_strength(5e15)[tally]
    ///     >>> heating.to("W/cm3").mean
    pub fn to(&self, units: &Bound<'_, PyAny>) -> PyResult<Self> {
        let units: Vec<String> = match units.extract::<String>() {
            Ok(u) => vec![u],
            Err(_) => units.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("units must be a str or a list of str")
            })?,
        };
        self.inner
            .to_units(&units)
            .map(|r| Self { inner: Arc::new(r) })
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Volume in cm3 of each bin, flat in the same order as ``mean``: the
    /// mesh element of a mesh tally, or the cell of a cell-filtered tally
    /// whose cells have a ``volume``. Raises ``ValueError`` when a bin's
    /// volume is not known.
    #[getter]
    pub fn bin_volumes(&self) -> PyResult<Vec<f64>> {
        self.inner
            .bin_volumes()
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Relative error (std_dev / mean) per bin.
    #[getter]
    pub fn relative_error(&self) -> Vec<f64> {
        self.inner.relative_error.clone()
    }

    /// Nuclear-data standard deviation per bin, when the run was given
    /// ``data_uncertainty``; ``None`` otherwise.
    ///
    /// The score's first-order variance along the cross sections' principal
    /// modes, which carries no sampling error from the replicas, plus the
    /// nonlinear remainder the replicas estimate, each with the Monte Carlo
    /// noise of the shared histories deconvolved
    /// (``nuclear_data_replica_spread`` is before, ``nuclear_data_noise``
    /// what was removed). A bin where that noise swamps the estimate reads
    /// zero; it is flagged in ``nuclear_data_noise_dominated``, and more
    /// particles would resolve it.
    #[getter]
    pub fn nuclear_data_standard_deviation(&self) -> Option<Vec<f64>> {
        self.inner
            .nuclear_data_standard_deviation()
            .map(|v| self.inner.scaled(&v))
    }

    /// Standard deviation of the replicas per bin as scored, before the
    /// Monte Carlo noise is deconvolved, ``sqrt(sigma**2 + noise**2)``, when
    /// the run was given ``data_uncertainty``; ``None`` otherwise.
    #[getter]
    pub fn nuclear_data_replica_spread(&self) -> Option<Vec<f64>> {
        self.inner
            .nuclear_data_replica_spread()
            .map(|v| self.inner.scaled(&v))
    }

    /// Standard deviation of the Monte Carlo noise deconvolved from the
    /// replicas' spread per bin, when the run was given ``data_uncertainty``;
    /// ``None`` otherwise. Each replica's shift from the nominal is scored on
    /// the shared histories, which cancel most but not all of its noise; it
    /// falls as ``1/sqrt(particles)``.
    #[getter]
    pub fn nuclear_data_noise(&self) -> Option<Vec<f64>> {
        self.inner
            .nuclear_data_noise()
            .map(|v| self.inner.scaled(&v))
    }

    /// Per bin, whether the noise variance deconvolved exceeded a quarter of
    /// the nuclear-data variance left (or that variance came out negative),
    /// when the run was given ``data_uncertainty``; ``None`` otherwise. The
    /// sigma is still the deconvolved one; more particles would sharpen it.
    #[getter]
    pub fn nuclear_data_noise_dominated(&self) -> Option<Vec<bool>> {
        self.inner.nuclear_data_noise_dominated()
    }

    /// Average of the cross-section replicas' means per bin, when the run was
    /// given ``data_uncertainty``; ``None`` otherwise.
    ///
    /// ``mean`` stays the nominal result. The two differ where the score
    /// responds nonlinearly to the cross sections, such as attenuation
    /// through a thick shield. The replicas' linear part, whose average is
    /// the nominal exactly, is subtracted from them first, so only the
    /// nonlinear part is left to their sampling.
    #[getter]
    pub fn replica_mean(&self) -> Option<Vec<f64>> {
        self.inner.replica_mean().map(|v| self.inner.scaled(&v))
    }

    /// First-order nuclear-data variance per bin, when the run was given
    /// ``data_uncertainty``; ``None`` otherwise.
    ///
    /// The variance of the score's linear response to the cross sections
    /// along their principal modes. ``nuclear_data_standard_deviation`` is the
    /// square root of this plus the nonlinear remainder.
    #[getter]
    pub fn nuclear_data_linear_variance(&self) -> Option<Vec<f64>> {
        self.inner
            .nuclear_data_linear_variance()
            .map(|v| self.inner.scaled_squared(&v))
    }

    /// Each nuclide's share of ``nuclear_data_linear_variance`` per bin, as a
    /// dict from nuclide name to a list over bins, when the run was given
    /// ``data_uncertainty``; ``None`` otherwise.
    ///
    /// The shares add up to the first-order variance: they say which
    /// nuclides' cross sections the score's uncertainty comes from.
    #[getter]
    pub fn nuclear_data_variance_by_nuclide(
        &self,
    ) -> Option<std::collections::BTreeMap<String, Vec<f64>>> {
        self.inner.nuclear_data_variance_by_nuclide().map(|m| {
            m.into_iter()
                .map(|(k, v)| (k, self.inner.scaled_squared(&v)))
                .collect()
        })
    }

    /// Monte Carlo standard error of one replica's mean per bin, averaged over
    /// replicas, when the run was given ``data_uncertainty``; ``None``
    /// otherwise. Much larger than ``nuclear_data_standard_deviation`` means
    /// the nuclear-data spread is not resolved.
    #[getter]
    pub fn replica_standard_error(&self) -> Option<Vec<f64>> {
        self.inner
            .replica_standard_error()
            .map(|v| self.inner.scaled(&v))
    }

    /// Variance of the mean per bin (``standard_deviation**2``), computed
    /// on the fly. This is the variance of the mean estimate -- the square
    /// of the reported uncertainty.
    #[getter]
    pub fn variance(&self) -> Vec<f64> {
        self.inner.scaled_squared(&self.inner.variance())
    }

    /// Total score per bin over all histories, per source particle times
    /// the history count; not scaled by a source strength or units.
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

    /// Covariance of the bin means, a ``num_bins x num_bins`` nested list in
    /// the same bin order as ``mean``, or ``None`` unless the tally was built
    /// with ``covariance=True``. Its diagonal is ``standard_deviation``
    /// squared; for an energy-binned flux tally it is what
    /// ``Pulse(flux_covariance=...)`` takes.
    #[getter]
    pub fn covariance(&self) -> Option<Vec<Vec<f64>>> {
        let n = self.inner.mean.len();
        let scale = self.inner.scaled(&vec![1.0; n]);
        self.inner.covariance().map(|c| {
            c.chunks(n.max(1))
                .zip(&scale)
                .map(|(row, si)| row.iter().zip(&scale).map(|(v, sj)| v * si * sj).collect())
                .collect()
        })
    }

    /// Raw per-bin Welford sum of squared deviations, per source particle:
    /// the exact merge state behind ``standard_deviation``. Empty when the
    /// producing path recorded no Welford state (e.g. GPU runs).
    #[getter]
    pub fn m2(&self) -> Vec<f64> {
        self.inner.m2.clone()
    }

    /// Exact total source-history count behind this tally's statistics.
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

    /// Mean of the per-history total score (the tally total), in ``units``.
    /// Raises ``ValueError`` after a per-volume ``to``, when the bins were
    /// divided by different volumes and their total has no unit.
    #[getter]
    pub fn aggregate_mean(&self) -> PyResult<f64> {
        Ok(self.inner.aggregate_mean() * self.uniform_scale()?)
    }

    /// Standard error of the per-history total mean (captures within-history
    /// correlation between bins), in ``units``. Raises ``ValueError`` as
    /// ``aggregate_mean`` does.
    #[getter]
    pub fn aggregate_std_dev(&self) -> PyResult<f64> {
        Ok(self.inner.aggregate_std_dev() * self.uniform_scale()?)
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
    ///         ``"std_dev"``, ``"rel_err"``, ``"variance"``, or ``"count"``,
    ///         the names every export uses. The flat getters ``mean``,
    ///         ``standard_deviation``, ``relative_error``, ``variance`` and
    ///         ``total_count`` return the unshaped 1-D lists.
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
            "mean" => numpy.call_method1("array", (self.mean(),))?,
            "std_dev" => numpy.call_method1("array", (self.standard_deviation(),))?,
            "rel_err" => numpy.call_method1("array", (self.inner.relative_error.clone(),))?,
            "variance" => numpy.call_method1("array", (self.variance(),))?,
            "count" => numpy.call_method1("array", (self.inner.total_count.clone(),))?,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown field {other:?}; expected \"mean\", \"std_dev\", \"rel_err\", \
                     \"variance\", or \"count\""
                )));
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
    /// of merit, shape, total particle count) beside the
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
        let mean = match (self.aggregate_mean(), self.aggregate_std_dev()) {
            (Ok(m), Ok(s)) => format!("{} ± {}", num(m), num(s)),
            _ => "per-bin volumes differ".to_string(),
        };
        let units = esc(&self.units().join(", "));
        let stats = kv(&[
            ("mean ± σ", mean),
            ("units", units),
            (
                "relative error",
                format!("{}%", num(self.aggregate_relative_error() * 100.0)),
            ),
            ("figure of merit", num(self.aggregate_figure_of_merit())),
            ("shape", esc(&shape)),
            ("particles", format!("{}", self.inner.n_histories)),
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

impl PyTallyResult {
    fn uniform_scale(&self) -> PyResult<f64> {
        self.inner.uniform_scale().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "the bins were divided by different volumes, so their total has no unit; \
                 read the total from the result before the per-volume to(...)",
            )
        })
    }
}

/// A source strength from Python's ``(strength, per)``.
fn parse_strength(strength: Option<f64>, per: &str) -> PyResult<Option<SourceStrength>> {
    let per = StrengthPer::parse(per).map_err(pyo3::exceptions::PyValueError::new_err)?;
    strength
        .map(|v| SourceStrength::new(v, per))
        .transpose()
        .map_err(pyo3::exceptions::PyValueError::new_err)
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
    /// True when the tail is estimable and every evaluated check passed.
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
    /// Every populated bin has relative error below 0.10; no per-bin tail
    /// or higher-moment state is allocated by this check.
    #[getter]
    fn per_bin_relative_error_ok(&self) -> Option<bool> {
        self.inner.per_bin_relative_error_ok
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
    /// Relative error falling as ``1/sqrt(N)`` over the last half of the run.
    #[getter]
    fn relative_error_rate_ok(&self) -> Option<bool> {
        self.inner.relative_error_rate_ok
    }
    /// Variance of the variance falling as ``1/N`` over the last half of the run.
    #[getter]
    fn variance_of_variance_rate_ok(&self) -> Option<bool> {
        self.inner.variance_of_variance_rate_ok
    }
    /// Figure of merit shows no systematic trend with ``N`` over the last half
    /// of the run. ``None`` when that half spans under 0.1 s of wall time.
    #[getter]
    fn figure_of_merit_no_trend(&self) -> Option<bool> {
        self.inner.figure_of_merit_no_trend
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
        let checks: [(&str, Option<bool>); 10] = [
            ("relative error", c.relative_error_ok),
            ("rel. error decreasing", c.relative_error_decreasing),
            ("rel. error falls as 1/√N", c.relative_error_rate_ok),
            ("variance of variance", c.variance_of_variance_ok),
            ("VoV decreasing", c.variance_of_variance_decreasing),
            ("VoV falls as 1/N", c.variance_of_variance_rate_ok),
            ("PDF tail slope", c.tail_slope_ok),
            ("mean stable", c.mean_stable),
            ("figure of merit stable", c.figure_of_merit_stable),
            ("figure of merit no trend", c.figure_of_merit_no_trend),
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
    /// Transport wall-clock time elapsed at this snapshot, in seconds.
    #[getter]
    fn elapsed_secs(&self) -> f64 {
        self.inner.elapsed_secs
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
#[pyclass(module = "yamc._core", name = "TransportResults", unsendable)]
pub struct PyTransportResults {
    pub inner: TransportResults,
    pub tracks: Option<PyTracks>,
    /// Per-run timing captured when these results were built. `None` for
    /// results that were combined (`combine_results`) or reloaded
    /// (`from_arrow`), which do not carry a single run's split timing.
    pub data_load_secs: Option<f64>,
    pub transport_secs: Option<f64>,
    pub particles_per_second: Option<u64>,
    pub total_particles: usize,
    /// What the weight windows did in the run, or `None` when it had none and
    /// for combined, reloaded or transmutation-step results.
    pub weight_window_counts: Option<yamc_physics::util::bank::WeightWindowCounts>,
}

impl PyTransportResults {
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
impl PyTransportResults {
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

    /// These results scaled to a source strength: source particles per
    /// second (``per="second"``, a steady source) or per pulse
    /// (``per="pulse"``, such as one ICF shot). Transport is linear in the
    /// source, so the strength is set after the run and changing it needs
    /// no new run.
    ///
    /// Returns new results and leaves these unchanged. Calling it again
    /// replaces the strength rather than multiplying by it, and ``None``
    /// returns to values per source particle. Every tally's ``mean`` and
    /// ``standard_deviation`` scale; relative errors and the statistical
    /// checks do not change. The strength is saved by ``to_arrow``, and
    /// ``combine_results`` refuses results with different strengths.
    ///
    /// Args:
    ///     strength: Source particles per second or per pulse, or None.
    ///     per: ``"second"`` (default) or ``"pulse"``.
    ///
    /// Examples:
    ///     >>> per_second = results.with_source_strength(5e15)
    ///     >>> per_pulse = results.with_source_strength(1e19, per="pulse")
    ///     >>> per_second[heating].to("W/cm3").mean
    #[pyo3(signature = (strength, per="second"))]
    fn with_source_strength(&self, strength: Option<f64>, per: &str) -> PyResult<Self> {
        let strength = parse_strength(strength, per)?;
        let inner = self
            .inner
            .with_source_strength(strength)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner,
            tracks: self.tracks.clone(),
            data_load_secs: self.data_load_secs,
            transport_secs: self.transport_secs,
            particles_per_second: self.particles_per_second,
            total_particles: self.total_particles,
            weight_window_counts: self.weight_window_counts,
        })
    }

    /// The source strength as ``(value, per)``, ``per`` being ``"second"`` or
    /// ``"pulse"``, or ``None`` for results per source particle.
    #[getter]
    fn source_strength(&self) -> Option<(f64, &'static str)> {
        self.inner
            .source_strength()
            .map(|s| (s.value, s.per.name()))
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

    /// What the weight windows did in this run, as a dict of counts, or
    /// ``None`` when the run had no windows and for combined, reloaded or
    /// transmutation-step results. Window counts cover collisions of the
    /// particle types that have a window:
    ///
    /// - ``histories``, ``particles`` (sources, secondaries and split copies)
    ///   and ``max_particles_per_history``.
    /// - ``checks``: collisions at which a window was checked. ``no_window``:
    ///   collisions outside the window mesh or energy bins, or in a voxel
    ///   without a window, where the windows do nothing.
    /// - ``split_events``, ``split_copies`` and
    ///   ``max_split_copies_per_history``. ``splits_capped_at_max_split``:
    ///   splits that needed more than ``max_split`` copies, which leave the
    ///   copies above their window.
    /// - ``split_budget``, ``histories_at_split_budget`` and
    ///   ``splits_cut_short``: the per-history budget of split copies, the
    ///   histories that reached it and the copies it refused. Tallies stay
    ///   unbiased, but what those histories reach last gets fewer, heavier
    ///   particles.
    /// - ``roulette_kills`` and ``roulette_survivals``.
    /// - ``weight_floor_kills`` and ``weight_floor_weight``: particles killed
    ///   below ``weight_floor`` and the weight they carried, summed over the
    ///   run. A kill at the floor is not a roulette, so any biases tallies low.
    #[getter]
    fn weight_window_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(c) = self.weight_window_counts else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("histories", c.histories)?;
        d.set_item("particles", c.particles)?;
        d.set_item("max_particles_per_history", c.max_particles_per_history)?;
        d.set_item("checks", c.checks)?;
        d.set_item("no_window", c.no_window)?;
        d.set_item("split_events", c.split_events)?;
        d.set_item("split_copies", c.split_copies)?;
        d.set_item(
            "max_split_copies_per_history",
            c.max_split_copies_per_history,
        )?;
        d.set_item("splits_capped_at_max_split", c.splits_capped_at_max_split)?;
        d.set_item("split_budget", c.split_budget)?;
        d.set_item("histories_at_split_budget", c.histories_at_split_budget)?;
        d.set_item("splits_cut_short", c.splits_cut_short)?;
        d.set_item("roulette_kills", c.roulette_kills)?;
        d.set_item("roulette_survivals", c.roulette_survivals)?;
        d.set_item("weight_floor_kills", c.weight_floor_kills)?;
        d.set_item("weight_floor_weight", c.weight_floor_weight)?;
        Ok(Some(d))
    }

    /// Provenance of the run(s) behind these results: a list of dicts
    /// with ``seed``, ``n_histories``, ``elapsed_secs``, ``fingerprint``,
    /// ``data_libraries``, ``data_releases``, ``material_slots``,
    /// ``compute``, ``yamc_version``, ``mpi_size`` and ``mpi_rank``. One
    /// entry per ``simulate_transport`` run; concatenated by
    /// ``combine_results``.
    ///
    /// ``material_slots`` maps each geometry material's id to the first slot
    /// of the geometry's material store holding it, which is how
    /// ``combine_results`` tells which material a material filter's id names.
    ///
    /// ``data_releases`` maps each library keyword the run downloaded data
    /// from, and the absolute path of each local release folder it read data
    /// from, to a dict with ``release`` (the release identifier),
    /// ``manifest_sha256`` (the hash of that release's manifest, which pins
    /// every file in it), ``format_version``, ``offline`` (True when the
    /// data origin was unreachable and the newest complete cached release was
    /// used; always False for a local folder) and ``converters``
    /// (every build that wrote the release's files, as its manifest lists
    /// them: the package, its version and the core commit it was compiled
    /// from, e.g. ``["yani-python 0.22.1 (core 250c439e...)"]``; empty for a
    /// release published before releases recorded them), ``citations`` (what
    /// to cite for the library's data: a list of dicts with ``reference`` and
    /// ``doi``, the references its makers ask for) and ``terms`` (what its
    /// data is redistributed under). Cite every library a run read: see
    /// "Cite the nuclear data you use" on the nuclear data libraries page.
    /// Empty when all data came from local directories that are not release
    /// folders.
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
                d.set_item("data_releases", data_releases_dict(py, &run.data_releases)?)?;
                d.set_item("material_slots", run.material_slots.clone())?;
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
    ///     *others: further ``TransportResults`` from distinct-seed runs of
    ///         the same model.
    #[pyo3(signature = (*others))]
    fn combine<'py>(
        &self,
        py: Python<'py>,
        others: &Bound<'py, PyTuple>,
    ) -> PyResult<PyTransportResults> {
        let borrowed: Vec<PyRef<'py, PyTransportResults>> = others
            .iter()
            .map(|item| {
                item.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "combine() arguments must be TransportResults",
                    )
                })
            })
            .collect::<PyResult<_>>()?;
        let mut inputs: Vec<&TransportResults> = Vec::with_capacity(borrowed.len() + 1);
        inputs.push(&self.inner);
        inputs.extend(borrowed.iter().map(|b| &b.inner));
        combine_inner(py, &inputs)
    }

    /// ``results + other`` pools two independent runs (see ``combine``).
    fn __add__<'py>(
        &self,
        py: Python<'py>,
        other: PyRef<'py, PyTransportResults>,
    ) -> PyResult<PyTransportResults> {
        combine_inner(py, &[&self.inner, &other.inner])
    }

    /// Save to an Arrow IPC file at ``path``, or return the file's bytes
    /// when ``path`` is None (for a browser or Pyodide session).
    ///
    /// The round trip is lossless: numeric data, the raw Welford merge
    /// state (per-bin ``m2`` and history counts), shapes, the full
    /// ``Tally`` configurations and the run provenance all survive, so a
    /// reloaded result remains fully combinable via ``combine_results``.
    /// The file is plain Arrow IPC, directly readable with
    /// pyarrow/polars. Captured particle tracks are NOT serialized.
    ///
    /// Args:
    ///     path (str | os.PathLike | None): Where to write.
    ///
    /// Returns:
    ///     bytes | None: The file's bytes when ``path`` is None.
    #[pyo3(signature = (path=None))]
    fn to_arrow(
        &self,
        py: Python<'_>,
        path: Option<std::path::PathBuf>,
    ) -> PyResult<Option<Py<pyo3::types::PyBytes>>> {
        let bytes = self
            .inner
            .to_arrow_bytes()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        yani_python::file_io::write_or_return(py, path, &bytes)
    }

    /// Load a ``TransportResults`` that ``to_arrow`` saved.
    ///
    /// Args:
    ///     source (str | os.PathLike | bytes): A file path, or the bytes
    ///         ``to_arrow()`` returned.
    #[classmethod]
    fn from_arrow(
        _cls: &Bound<'_, pyo3::types::PyType>,
        source: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let bytes = yani_python::file_io::read_source(source)?;
        let inner = yamc_tallies::transport_results::TransportResults::from_arrow_bytes(&bytes)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner,
            tracks: None,
            data_load_secs: None,
            transport_secs: None,
            particles_per_second: None,
            total_particles: 0,
            weight_window_counts: None,
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
                "TransportResults",
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
            let mean = tr.aggregate_mean().map_or_else(|_| "n/a".to_string(), num);
            rows.push(vec![
                esc(&name),
                esc(&shape),
                mean,
                esc(&tr.units().join(", ")),
                format!("{}%", num(tr.aggregate_relative_error() * 100.0)),
                check_cell,
            ]);
        }
        let tword = if n == 1 { "tally" } else { "tallies" };
        let strength = self
            .inner
            .source_strength()
            .map_or_else(String::new, |s| format!(" · source {s}"));
        let subtitle = format!(
            "{n} {tword} · {} particles · {:.3}s{strength}",
            total_particles(&self.inner),
            self.inner.elapsed_secs
        );
        let tbl = table(
            &["tally", "shape", "mean", "units", "rel. err", "checks"],
            &rows,
            2,
        );
        let body = if any_eval {
            format!(
                "<div style=\"margin-bottom:6px;\">overall reliability: {}</div>{tbl}",
                badge(Some(all_pass))
            )
        } else {
            tbl
        };
        card("TransportResults", &subtitle, &body)
    }

    fn __repr__(&self) -> String {
        let strength = self
            .inner
            .source_strength()
            .map_or_else(String::new, |s| format!(", source_strength={s}"));
        format!(
            "TransportResults(n_tallies={}, particles={}, elapsed_secs={:.3}{strength})",
            self.inner.len(),
            total_particles(&self.inner),
            self.inner.elapsed_secs
        )
    }
}

/// Total source particles behind the results, for display. Every tally of a
/// run (or of a combined result) carries the same history count, so the
/// first tally's is used; 0 when there are no tallies.
fn total_particles(results: &TransportResults) -> u64 {
    results.iter().next().map_or(0, |r| r.n_histories)
}

/// Iterator over the `Tally` configs in a `TransportResults`.
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

/// The version to stamp into run provenance: the installed `yamc`
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
pub(crate) fn provenance_version() -> String {
    Python::attach(|py| {
        py.import("importlib.metadata")
            .and_then(|m| m.call_method1("version", ("yamc",)))
            .and_then(|v| v.extract::<String>())
            .unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string())
    })
}

/// Entry point used by `PyModel.simulate_transport()`. Builds the results
/// from the model's current tallies with run provenance attached (seed,
/// model fingerprint, data libraries, compute path, MPI placement), so the
/// returned results are combinable via `combine_results`. Surfaces
/// duplicate-name / duplicate-id validation as `ValueError`.
/// The `data_releases` provenance of a run as a dict of dicts, keyed by
/// library keyword, local release folder path, or the path of a locally
/// processed transmutation subsection.
pub(crate) fn data_releases_dict<'py>(
    py: Python<'py>,
    releases: &std::collections::BTreeMap<String, yamc_nuclide::storage::release::DataRelease>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (keyword, release) in releases {
        let d = PyDict::new(py);
        d.set_item("release", &release.release)?;
        d.set_item("manifest_sha256", &release.manifest_sha256)?;
        d.set_item("format_version", release.format_version)?;
        d.set_item("offline", release.offline)?;
        d.set_item("converters", &release.converters)?;
        d.set_item("locally_processed", release.locally_processed)?;
        d.set_item("library", &release.library)?;
        d.set_item("citations", citations_list(py, &release.citations)?)?;
        d.set_item("terms", &release.terms)?;
        out.set_item(keyword, d)?;
    }
    Ok(out)
}

/// A release's citations as a list of ``{"reference": ..., "doi": ...}``
/// dicts.
pub(crate) fn citations_list<'py>(
    py: Python<'py>,
    citations: &[yamc_nuclide::storage::release::Citation],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for c in citations {
        let d = PyDict::new(py);
        d.set_item("reference", &c.reference)?;
        d.set_item("doi", &c.doi)?;
        list.append(d)?;
    }
    Ok(list)
}

pub(crate) fn build_results(
    model: &yamc::model::Model,
    settings: &yamc::model::TransportSettings,
    compute: &str,
    elapsed_secs: f64,
    tracks: Option<PyTracks>,
) -> PyResult<PyTransportResults> {
    let inner = model
        .tally_results(settings.seed, compute, elapsed_secs, provenance_version())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyTransportResults {
        tracks,
        data_load_secs: model.last_data_load_secs,
        transport_secs: model.last_transport_secs,
        particles_per_second: model.last_particles_per_second,
        weight_window_counts: model.last_weight_window_counts,
        ..PyTransportResults::from(inner)
    })
}

impl From<TransportResults> for PyTransportResults {
    /// Results with no single run's split timing and no tracks. Histories
    /// actually run (fewer than asked on an early stop) are the reported
    /// particle total.
    fn from(inner: TransportResults) -> Self {
        let total_particles = total_particles(&inner) as usize;
        Self {
            inner,
            tracks: None,
            data_load_secs: None,
            transport_secs: None,
            particles_per_second: None,
            total_particles,
            weight_window_counts: None,
        }
    }
}

/// Run the Rust combine and surface its pass-through warnings as Python
/// `UserWarning`s.
pub(crate) fn combine_inner(
    py: Python<'_>,
    inputs: &[&TransportResults],
) -> PyResult<PyTransportResults> {
    let (combined, warnings) = yamc_tallies::combine::combine_results(inputs)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    if !warnings.is_empty() {
        let warnings_mod = py.import("warnings")?;
        for w in &warnings {
            warnings_mod.call_method1("warn", (w.as_str(),))?;
        }
    }
    Ok(PyTransportResults {
        inner: combined,
        tracks: None,
        data_load_secs: None,
        transport_secs: None,
        particles_per_second: None,
        total_particles: 0,
        weight_window_counts: None,
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
/// - tallies with the same name must have identical configurations. A
///   material filter's bins are matched by the geometry material each id
///   names, not by the id, so runs of a model built anew (whose materials
///   get new ids) still combine, and filters naming different materials
///   under the same ids do not;
/// - GPU-produced results are refused (no Welford merge state), as are
///   non-root MPI results (every rank holds the same statistics, so a
///   non-root result would count the run twice).
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
) -> PyResult<PyTransportResults> {
    let mut borrowed: Vec<PyRef<'py, PyTransportResults>> = Vec::with_capacity(results.len());
    for item in results.iter() {
        borrowed.push(item.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "combine_results arguments must be TransportResults objects",
            )
        })?);
    }
    let inputs: Vec<&TransportResults> = borrowed.iter().map(|p| &p.inner).collect();
    combine_inner(py, &inputs)
}
