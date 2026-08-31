//! `PulseSchedule`: an irradiation/cooling timeline plus the single
//! post-processing entry point for decay-photon shutdown-dose-rate (SDR).
//!
//! This replaces the three free `decay_photons` functions (the verbatim
//! `openmc.deplete.d1s` names) with one object that owns the irradiation
//! history. Each `Pulse` carries its own neutron source + rate + duration, so
//! a mixed DD/DT campaign with cooling in between is one schedule. The
//! per-nuclide time-correction maths stays in the `yamc-physics` core; this
//! module is the orchestration + typed result wrapper.
//!
//! Mixed-spectrum (e.g. DD + DT) schedules are handled **exactly**: each
//! distinct source's activation is reconstructed from its own transport run
//! (one tally result per source) and the corrected doses are summed
//! (independent runs, so variances add in quadrature).

use std::collections::BTreeMap;

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use super::decay_photons::resolve_chain;
use super::source::PyNeutronSource;

/// Extract a duration -- either a plain number of seconds or a `(value, unit)`
/// tuple -- from a Python object and convert it to seconds via the core unit
/// table (`yani_transmute::duration_to_seconds`). Shared with
/// `Model.simulate_transport`'s `max_runtime` kwarg.
pub fn parse_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    // (value, unit) tuple/list, e.g. (1, "a"). Checked before the plain-number
    // arm so a 2-sequence isn't mis-read.
    if let Ok((value, unit)) = obj.extract::<(f64, String)>() {
        return yani_transmute::duration_to_seconds(value, &unit).map_err(PyValueError::new_err);
    }
    // Plain number -> seconds.
    if let Ok(secs) = obj.extract::<f64>() {
        if !secs.is_finite() || secs < 0.0 {
            return Err(PyValueError::new_err(
                "duration must be a finite, non-negative number",
            ));
        }
        return Ok(secs);
    }
    Err(PyTypeError::new_err(
        "duration must be a number of seconds or a (value, unit) tuple, e.g. (1, 'a') or (30, 'd')",
    ))
}

/// One irradiation campaign: a neutron source held at a constant `rate` (n/s)
/// for `duration`.
#[gen_stub_pyclass]
#[pyclass(name = "Pulse")]
pub struct PyPulse {
    /// The neutron source for this campaign (kept by identity so a schedule can
    /// group steps that share a source). `None` is allowed for
    /// `Model.simulate_transmutation` (transport uses the model's source);
    /// `Material.transmute` requires a source whose energy is a `Histogram`.
    source: Option<Py<PyAny>>,
    rate: f64,
    duration_seconds: f64,
    /// Per-bin standard deviation of the flux, when the caller knows it.
    ///
    /// Lives here rather than on `Histogram` because `Histogram` is a shared
    /// transport source distribution and this is an activation-only concern.
    /// It also sits beside `rate`, which is the other half of the flux.
    flux_std_dev: Option<Vec<f64>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPulse {
    /// Create an irradiation pulse.
    ///
    /// Args:
    ///     rate: For ``Model.simulate_transmutation``, the source emission rate
    ///         in particles/second (n/s). For ``Material.transmute``, the total
    ///         flux magnitude [n/cm^2/s] (the source's ``Histogram`` energy gives
    ///         the spectrum shape).
    ///     duration: A number of seconds, or a ``(value, unit)`` tuple such as
    ///         ``(1, "a")`` or ``(30, "d")``. Units: s, min, h, d, a (year).
    ///     source: The NeutronSource for this campaign. Defaults to ``None``,
    ///         which ``Model.simulate_transmutation`` accepts (transport uses the
    ///         model's source); ``Material.transmute`` requires a source whose
    ///         energy is a ``Histogram`` and ignores its position/direction.
    ///     flux_std_dev: Per-bin standard deviation of the flux, in the same
    ///         units and the same order as the ``Histogram`` values. Optional,
    ///         and omitting it is the common case: a spectrum taken from a
    ///         published reference set carries no stated error. From a yamc
    ///         run it is ``results[tally].standard_deviation``.
    ///
    ///         Used only by ``Material.transmute(data_uncertainty=...)`` with
    ///         the ``flux_spectrum`` source enabled. Omitted, that source
    ///         contributes nothing and says so in
    ///         ``data_uncertainty_info["spectra_without_flux_sigma"]`` rather
    ///         than reading as a flux known exactly.
    #[new]
    #[pyo3(signature = (rate, duration, source=None, flux_std_dev=None))]
    fn new(
        rate: f64,
        duration: &Bound<'_, PyAny>,
        source: Option<&Bound<'_, PyAny>>,
        flux_std_dev: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        // Validate any source is a neutron source (activation is by neutrons),
        // but keep the original object so identity grouping works across steps.
        if let Some(s) = source {
            if !s.is_instance_of::<PyNeutronSource>() {
                return Err(PyTypeError::new_err("Pulse source must be a NeutronSource"));
            }
        }
        if rate < 0.0 {
            return Err(PyValueError::new_err("rate must be non-negative"));
        }
        if let Some(sigma) = &flux_std_dev {
            if source.is_none() {
                return Err(PyValueError::new_err(
                    "flux_std_dev needs a source: it is the error on that source's \
                     Histogram values, and a sourceless pulse has no flux",
                ));
            }
            if sigma.iter().any(|s| *s < 0.0) {
                return Err(PyValueError::new_err(
                    "flux_std_dev entries must be non-negative",
                ));
            }
        }
        Ok(PyPulse {
            source: source.map(|s| s.clone().unbind()),
            rate,
            duration_seconds: parse_duration(duration)?,
            flux_std_dev,
        })
    }

    /// The neutron source active during this pulse, or ``None`` if sourceless.
    #[getter]
    fn source(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.source.as_ref().map(|s| s.clone_ref(py))
    }

    /// The per-bin flux standard deviation, or ``None`` if none was given.
    #[getter]
    fn flux_std_dev(&self) -> Option<Vec<f64>> {
        self.flux_std_dev.clone()
    }

    /// Source emission rate in particles/second.
    #[getter]
    fn rate(&self) -> f64 {
        self.rate
    }

    /// Pulse duration in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        self.duration_seconds
    }

    fn __repr__(&self) -> String {
        format!(
            "Pulse(rate={:e}, duration={} s)",
            self.rate, self.duration_seconds
        )
    }
}

/// A cooling step: no source, the materials just decay for `duration`.
#[gen_stub_pyclass]
#[pyclass(name = "Cooldown")]
pub struct PyCooldown {
    duration_seconds: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCooldown {
    /// Create a cooling step.
    ///
    /// Args:
    ///     duration: A number of seconds, or a ``(value, unit)`` tuple such as
    ///         ``(7, "d")``. Units: s, min, h, d, a (year).
    #[new]
    #[pyo3(signature = (duration))]
    fn new(duration: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyCooldown {
            duration_seconds: parse_duration(duration)?,
        })
    }

    /// Cooldown duration in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        self.duration_seconds
    }

    fn __repr__(&self) -> String {
        format!("Cooldown(duration={} s)", self.duration_seconds)
    }
}

/// Build a run of cooling steps at log-spaced *cumulative* times.
///
/// `Cooldown` takes the duration OF THAT STEP, which is the right primitive and
/// the wrong thing to type. Anyone plotting a decay curve wants points at
/// cumulative times, and had to difference them by hand (issue #453):
///
/// ```text
/// HOUR, YEAR = 3600.0, 365.25 * 86400.0
/// cum  = [HOUR * (10 * YEAR / HOUR) ** (k / (n - 1)) for k in range(n)]
/// gaps = [cum[0]] + [cum[k] - cum[k - 1] for k in range(1, n)]
/// ```
///
/// Log spacing is the default because decay is exponential, so a linear sweep
/// is wrong twice over: it wastes steps in the flat tail and misses the early
/// fall entirely. Activity and decay heat are plotted on log-log axes, so
/// evenly spaced PLOTTED points means log-spaced times.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (start, stop, n, spacing="log"))]
pub fn cooldown_steps(
    start: &Bound<'_, PyAny>,
    stop: &Bound<'_, PyAny>,
    n: usize,
    spacing: &str,
) -> PyResult<Vec<PyCooldown>> {
    let start_s = parse_duration(start)?;
    let stop_s = parse_duration(stop)?;
    if n < 2 {
        return Err(PyValueError::new_err(
            "n must be at least 2: one point is a single Cooldown, which needs no help",
        ));
    }
    if stop_s.is_nan() || start_s.is_nan() || stop_s <= start_s {
        return Err(PyValueError::new_err(format!(
            "stop ({stop_s} s) must be after start ({start_s} s)"
        )));
    }

    // Cumulative times, then the gaps between them. The first step runs from
    // the end of the irradiation to `start`, so the durations sum to `stop`
    // exactly rather than to `stop - start`.
    let cumulative: Vec<f64> = match spacing {
        "log" => {
            if start_s <= 0.0 {
                return Err(PyValueError::new_err(
                    "start must be positive for log spacing: the first point is a ratio from \
                     it, and zero has no logarithm. Use a short first time such as (1, 'h'), \
                     or spacing='linear'.",
                ));
            }
            let ratio = stop_s / start_s;
            (0..n)
                .map(|k| start_s * ratio.powf(k as f64 / (n - 1) as f64))
                .collect()
        }
        "linear" => {
            let step = (stop_s - start_s) / (n - 1) as f64;
            (0..n).map(|k| start_s + step * k as f64).collect()
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown spacing={other:?}; expected 'log' or 'linear'"
            )))
        }
    };

    let mut steps = Vec::with_capacity(n);
    let mut previous = 0.0;
    for time in cumulative {
        steps.push(PyCooldown {
            duration_seconds: time - previous,
        });
        previous = time;
    }
    Ok(steps)
}

/// The result of time-correcting a tally: dose at the selected cooling step(s).
///
/// `mean` / `std_dev` are 1-D (one selected step, when ``steps`` was a single
/// int) or 2-D (one row per selected step). `by_nuclide` maps each parent
/// radionuclide to its mean contribution, in the same shape. `times` is the
/// elapsed time (seconds) at the end of each selected step.
#[gen_stub_pyclass]
#[pyclass(name = "DoseResult")]
pub struct PyDoseResult {
    mean: Py<PyAny>,
    std_dev: Py<PyAny>,
    by_nuclide: Py<PyAny>,
    times: Py<PyAny>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDoseResult {
    /// Mean dose rate, 1-D for a single selected step or 2-D (one row per step).
    #[getter]
    fn mean(&self, py: Python<'_>) -> Py<PyAny> {
        self.mean.clone_ref(py)
    }

    /// Standard error on `mean`, same shape.
    #[getter]
    fn std_dev(&self, py: Python<'_>) -> Py<PyAny> {
        self.std_dev.clone_ref(py)
    }

    /// Per-radionuclide mean contribution: ``{nuclide: values}`` in the same
    /// shape as `mean`.
    #[getter]
    fn by_nuclide(&self, py: Python<'_>) -> Py<PyAny> {
        self.by_nuclide.clone_ref(py)
    }

    /// Elapsed time (seconds) at the end of each selected step.
    #[getter]
    fn times(&self, py: Python<'_>) -> Py<PyAny> {
        self.times.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        "DoseResult(mean=..., std_dev=..., by_nuclide=..., times=...)".to_string()
    }
}

/// An irradiation/cooling timeline built from `Pulse` and `Cooldown` steps.
///
/// `time_correct_tally` post-processes a decay-photon tally into shutdown dose
/// rates over the schedule.
#[gen_stub_pyclass]
#[pyclass(name = "PulseSchedule")]
pub struct PyPulseSchedule {
    /// The core numeric `(rate, dt)` timeline that drives transmutation.
    schedule: yani_transmute::Schedule,
    /// Per-step neutron source (by Python identity), aligned 1:1 with
    /// `schedule.steps()`. `None` for a cooldown or a sourceless pulse. Held
    /// here (not in the core) because source identity is only needed for the
    /// decay-photon SDR grouping in `time_correct_tally`.
    sources: Vec<Option<Py<PyAny>>>,
    /// Per-step flux sigma, parallel to `sources`.
    flux_std_dev: Vec<Option<Vec<f64>>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPulseSchedule {
    /// Build a schedule from a list of `Pulse` / `Cooldown` steps.
    #[new]
    fn new(steps: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut core_steps = Vec::new();
        let mut sources: Vec<Option<Py<PyAny>>> = Vec::new();
        let mut flux_std_dev: Vec<Option<Vec<f64>>> = Vec::new();
        for item in steps.try_iter()? {
            let item = item?;
            if let Ok(pulse) = item.cast::<PyPulse>() {
                let p = pulse.borrow();
                sources.push(p.source.as_ref().map(|s| s.clone_ref(item.py())));
                flux_std_dev.push(p.flux_std_dev.clone());
                core_steps.push(yani_transmute::ScheduleStep {
                    rate: p.rate,
                    dt: p.duration_seconds,
                    is_pulse: true,
                });
            } else if let Ok(cool) = item.cast::<PyCooldown>() {
                sources.push(None);
                flux_std_dev.push(None);
                core_steps.push(yani_transmute::ScheduleStep {
                    rate: 0.0,
                    dt: cool.borrow().duration_seconds,
                    is_pulse: false,
                });
            } else {
                return Err(PyTypeError::new_err(
                    "PulseSchedule steps must be Pulse or Cooldown objects",
                ));
            }
        }
        let schedule = yani_transmute::Schedule::new(core_steps).map_err(PyValueError::new_err)?;
        Ok(PyPulseSchedule {
            schedule,
            sources,
            flux_std_dev,
        })
    }

    /// The distinct neutron sources used by the pulses, in first-appearance
    /// order. For a mixed-spectrum schedule, pass ``time_correct_tally`` a dict
    /// mapping each of these to its transport run's tally result.
    #[getter]
    fn sources(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.distinct_sources(py)
            .into_iter()
            .map(|(s, _)| s)
            .collect()
    }

    /// Number of steps in the schedule.
    fn __len__(&self) -> usize {
        self.schedule.len()
    }

    fn __repr__(&self) -> String {
        let pulses = self.schedule.pulse_count();
        format!(
            "PulseSchedule({} steps: {} pulses, {} cooldowns)",
            self.schedule.len(),
            pulses,
            self.schedule.len() - pulses
        )
    }

    /// Time-correct a decay-photon tally into shutdown dose rate(s).
    ///
    /// Args:
    ///     results: For a single-source schedule, the ``TallyResult`` from the
    ///         decay-photon run. For a mixed-spectrum (multi-source) schedule, a
    ///         dict mapping each source in ``self.sources`` to its run's
    ///         ``TallyResult``; the per-source corrections are summed exactly.
    ///     steps: Which schedule step(s) to evaluate the dose at the END of.
    ///         ``None`` (default) selects every step; an ``int`` selects one
    ///         (returns 1-D arrays; negative counts from the end); a sequence of
    ///         ints selects a subset in order. The pre-irradiation baseline is
    ///         never included, so there is no row-0 footgun.
    ///
    /// The transmutation network is assembled from the configured per-subsection
    /// sources (``yamc.transmutation_decay_data`` etc.).
    ///
    /// Returns:
    ///     DoseResult with ``.mean`` / ``.std_dev`` / ``.by_nuclide`` / ``.times``.
    #[pyo3(signature = (results, steps=None))]
    fn time_correct_tally(
        &self,
        py: Python<'_>,
        results: &Bound<'_, PyAny>,
        steps: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDoseResult> {
        let distinct = self.distinct_sources(py);
        if distinct.is_empty() {
            return Err(PyValueError::new_err(
                "schedule has no pulses (all cooldowns); nothing to time-correct",
            ));
        }

        // Match each distinct source to its transport-run tally result.
        let per_source: Vec<Bound<'_, PyAny>> = self.match_results(py, results, &distinct)?;

        let timesteps: Vec<f64> = self.schedule.timesteps();
        let n = self.schedule.len();
        let (tcf_indices, single) = resolve_steps(steps, n)?;

        let chain = resolve_chain()?.chain;

        // Elapsed time at the end of each schedule step (cumulative dt).
        // tcf index t corresponds to the end of schedule step t-1.
        let cumulative = self.schedule.cumulative_times();
        let times: Vec<f64> = tcf_indices.iter().map(|&t| cumulative[t - 1]).collect();

        let n_steps = tcf_indices.len();
        let mut total_mean: Vec<Vec<f64>> = Vec::new();
        let mut total_var: Vec<Vec<f64>> = Vec::new();
        let mut by_nuclide: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();

        for (src_idx, (src_obj, _)) in distinct.iter().enumerate() {
            let tr = &per_source[src_idx];
            let (nuclides, n_scores, mean, std_dev) = extract_tally(tr)?;

            // This source's rate history: its own steps carry their rate, every
            // other step (cooldowns and other campaigns) contributes 0.
            let src_bound = src_obj.bind(py);
            let core_steps = self.schedule.steps();
            let source_rates: Vec<f64> = self
                .sources
                .iter()
                .enumerate()
                .map(|(i, s)| match s {
                    Some(o) if o.bind(py).is(src_bound) => core_steps[i].rate,
                    _ => 0.0,
                })
                .collect();

            let tcf =
                yani_decay::time_correction_factors(&nuclides, &timesteps, &source_rates, &chain)
                    .map_err(PyValueError::new_err)?;

            let (summed_means, summed_stds) = yani_decay::apply_time_correction(
                &mean,
                &std_dev,
                &nuclides,
                n_scores,
                &tcf,
                &tcf_indices,
                true,
            )
            .map_err(PyValueError::new_err)?;

            let (per_nuc_means, _) = yani_decay::apply_time_correction(
                &mean,
                &std_dev,
                &nuclides,
                n_scores,
                &tcf,
                &tcf_indices,
                false,
            )
            .map_err(PyValueError::new_err)?;

            // Accumulate the summed-over-nuclide dose (means add, variances add).
            if total_mean.is_empty() {
                total_mean = summed_means;
                total_var = summed_stds
                    .iter()
                    .map(|row| row.iter().map(|&v| v * v).collect())
                    .collect();
            } else {
                for (r, row) in summed_means.iter().enumerate() {
                    if row.len() != total_mean[r].len() {
                        return Err(PyValueError::new_err(
                            "per-source tally results have mismatched bin counts; \
                             all runs must use the same tally shape and parent_nuclides",
                        ));
                    }
                    for (k, &v) in row.iter().enumerate() {
                        total_mean[r][k] += v;
                    }
                }
                for (r, row) in summed_stds.iter().enumerate() {
                    for (k, &v) in row.iter().enumerate() {
                        total_var[r][k] += v * v;
                    }
                }
            }

            // Per-nuclide breakdown (mean only): slice parent -> (score*em).
            // Skipped when there are no parent nuclides (empty-filter zero
            // result): there is nothing to break down, and it avoids dividing
            // the row length by zero parents.
            let n_parent = nuclides.len();
            for (r, row) in per_nuc_means.iter().enumerate().filter(|_| n_parent > 0) {
                let stride = row.len() / n_parent;
                for (p, name) in nuclides.iter().enumerate() {
                    let slice = &row[p * stride..(p + 1) * stride];
                    let entry = by_nuclide
                        .entry(name.clone())
                        .or_insert_with(|| vec![vec![0.0; stride]; n_steps]);
                    for (k, &v) in slice.iter().enumerate() {
                        entry[r][k] += v;
                    }
                }
            }
        }

        let total_std: Vec<Vec<f64>> = total_var
            .iter()
            .map(|row| row.iter().map(|&v| v.sqrt()).collect())
            .collect();

        // Shape the output: drop the step dimension for a single int selection.
        let mean_obj = shape_rows(py, total_mean, single)?;
        let std_obj = shape_rows(py, total_std, single)?;
        let times_obj: Py<PyAny> = if single {
            times[0].into_pyobject(py)?.into_any().unbind()
        } else {
            times.into_pyobject(py)?.into_any().unbind()
        };

        let by_nuc_dict = PyDict::new(py);
        for (name, rows) in by_nuclide {
            by_nuc_dict.set_item(name, shape_rows(py, rows, single)?)?;
        }

        Ok(PyDoseResult {
            mean: mean_obj,
            std_dev: std_obj,
            by_nuclide: by_nuc_dict.into_any().unbind(),
            times: times_obj,
        })
    }
}

impl PyPulseSchedule {
    /// The core numeric timeline backing this schedule. The transmutation
    /// bindings hand this straight to the core driver; the per-pulse Python
    /// sources live separately (see `sources` / `distinct_sources`) because
    /// source identity is only needed for decay-photon SDR grouping.
    pub fn core_schedule(&self) -> &yani_transmute::Schedule {
        &self.schedule
    }

    /// Number of distinct pulse sources (by Python identity); 0 if every pulse
    /// is sourceless. Used to reject multi-spectrum schedules in
    /// `simulate_transmutation` (which honours only the model's source).
    pub fn distinct_source_count(&self, py: Python<'_>) -> usize {
        self.distinct_sources(py).len()
    }

    /// Build the per-spectrum / per-step plan for `Material.transmute`.
    ///
    /// Each irradiation `Pulse` must carry a `NeutronSource` whose energy is a
    /// `Histogram` (the spectrum); its `rate` is the total flux magnitude
    /// [n/cm^2/s]. Distinct sources (by Python identity) become distinct
    /// spectra, collapsed once each. A `Cooldown` is a decay-only step. A
    /// sourceless `Pulse` is rejected (irradiation needs a spectrum source).
    /// The source's position/direction are unused here (no transport runs).
    pub fn transmute_plan(
        &self,
        py: Python<'_>,
    ) -> PyResult<(
        Vec<yani_transmute::MultigroupSpectrum>,
        Vec<yani_transmute::TransmuteStep>,
    )> {
        use yani_transmute::TransmuteStep;
        let core_steps = self.schedule.steps();
        let mut spectra: Vec<yani_transmute::MultigroupSpectrum> = Vec::new();
        let mut distinct: Vec<(Py<PyAny>, Option<Vec<f64>>)> = Vec::new();
        let mut steps: Vec<TransmuteStep> = Vec::with_capacity(self.sources.len());

        for (i, src_opt) in self.sources.iter().enumerate() {
            let dt = core_steps[i].dt;
            match src_opt {
                Some(src) => {
                    let src_b = src.bind(py);
                    let sigma = self.flux_std_dev[i].as_ref();
                    // Keyed on the source AND its stated error: two pulses may
                    // share a spectrum object and give it different errors, and
                    // collapsing those onto one spectrum would silently apply
                    // one pulse's uncertainty to the other.
                    let idx = match distinct
                        .iter()
                        .position(|(d, s)| d.bind(py).is(src_b) && s.as_ref() == sigma)
                    {
                        Some(j) => j,
                        None => {
                            spectra.push(histogram_spectrum(src_b, sigma)?);
                            distinct.push((src.clone_ref(py), sigma.cloned()));
                            spectra.len() - 1
                        }
                    };
                    steps.push(TransmuteStep {
                        dt,
                        irradiation: Some((idx, core_steps[i].rate)),
                    });
                }
                None => {
                    if core_steps[i].is_pulse {
                        return Err(PyValueError::new_err(
                            "Material.transmute irradiation pulses must carry a NeutronSource \
                             whose energy is a Histogram (the spectrum); use Cooldown for \
                             decay-only steps",
                        ));
                    }
                    steps.push(TransmuteStep {
                        dt,
                        irradiation: None,
                    });
                }
            }
        }
        Ok((spectra, steps))
    }

    /// Distinct pulse sources in first-appearance order, paired with the index
    /// of the first step that uses each.
    fn distinct_sources(&self, py: Python<'_>) -> Vec<(Py<PyAny>, usize)> {
        let mut out: Vec<(Py<PyAny>, usize)> = Vec::new();
        for (i, source) in self.sources.iter().enumerate() {
            if let Some(src) = source {
                let src_b = src.bind(py);
                if !out.iter().any(|(o, _)| o.bind(py).is(src_b)) {
                    out.push((src.clone_ref(py), i));
                }
            }
        }
        out
    }

    /// Resolve the `results` argument to one tally result per distinct source.
    fn match_results<'py>(
        &self,
        py: Python<'py>,
        results: &Bound<'py, PyAny>,
        distinct: &[(Py<PyAny>, usize)],
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        if let Ok(dict) = results.cast::<PyDict>() {
            // Mixed-spectrum: dict {source: tally_result}, matched by identity.
            let mut out = Vec::with_capacity(distinct.len());
            for (src, _) in distinct {
                let src_b = src.bind(py);
                let mut found = None;
                for (k, v) in dict.iter() {
                    if k.is(src_b) {
                        found = Some(v);
                        break;
                    }
                }
                match found {
                    Some(v) => out.push(v),
                    None => {
                        return Err(PyValueError::new_err(
                            "results dict is missing a tally result for one of the schedule's \
                             sources; pass {source: results[tally]} for every source in \
                             schedule.sources",
                        ))
                    }
                }
            }
            Ok(out)
        } else {
            // Single tally result: only valid when there is exactly one source.
            if distinct.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "this schedule uses {} distinct sources, so pass a dict \
                     {{source: results[tally]}} mapping each source in schedule.sources \
                     to its transport run's tally result",
                    distinct.len()
                )));
            }
            Ok(vec![results.clone()])
        }
    }
}

/// Extract a normalized multigroup spectrum from a pulse's `NeutronSource`,
/// whose energy must be a `Histogram`. The per-group probabilities are
/// normalized to masses (only the shape matters; `rate` carries the magnitude).
fn histogram_spectrum(
    src: &Bound<'_, PyAny>,
    flux_std_dev: Option<&Vec<f64>>,
) -> PyResult<yani_transmute::MultigroupSpectrum> {
    use yamc_source::source::{ParticleSource, SourceEnergyDistribution};
    let ns = src.cast::<PyNeutronSource>().map_err(|_| {
        PyValueError::new_err("Material.transmute pulse source must be a NeutronSource")
    })?;
    let ns = ns.borrow();
    let energy = match &ns.inner {
        ParticleSource::Neutron(s) => &s.energy,
        _ => {
            return Err(PyValueError::new_err(
                "Material.transmute pulse source must be a NeutronSource",
            ))
        }
    };
    match energy {
        SourceEnergyDistribution::Histogram(h) => {
            let probs = h.probabilities();
            let total: f64 = probs.iter().sum();
            if total <= 0.0 {
                return Err(PyValueError::new_err(
                    "Material.transmute pulse source Histogram has zero total probability",
                ));
            }
            let masses: Vec<f64> = probs.iter().map(|p| p / total).collect();
            // The sigma is given against the Histogram's own values, so it is
            // relativized against those rather than against `masses`: the two
            // differ by the normalization, which divides out of a relative
            // error but not an absolute one.
            let relative_std_dev = match flux_std_dev {
                None => None,
                Some(sigma) => Some(
                    yani_transmute::flux_uncertainty::relative_std_dev(probs, sigma).ok_or_else(
                        || {
                            PyValueError::new_err(format!(
                                "flux_std_dev has {} entries but the pulse source's Histogram \
                                 has {} bins; they must line up, since each entry is the error \
                                 on the bin beside it",
                                sigma.len(),
                                probs.len(),
                            ))
                        },
                    )?,
                ),
            };
            Ok(yani_transmute::MultigroupSpectrum {
                boundaries: h.boundaries().to_vec(),
                masses,
                relative_std_dev,
            })
        }
        _ => Err(PyValueError::new_err(
            "Material.transmute pulse source energy must be a Histogram, e.g. \
             NeutronSource(energy=yamc.sources.Histogram(boundaries, flux)); the pulse rate \
             gives the flux magnitude [n/cm^2/s]",
        )),
    }
}

/// Pull (parent_nuclides, n_scores, mean, std_dev) off a tally result via its
/// public attributes (so this stays decoupled from the core Tally internals).
fn extract_tally(tr: &Bound<'_, PyAny>) -> PyResult<(Vec<String>, usize, Vec<f64>, Vec<f64>)> {
    let tally = tr.getattr("tally")?;
    let nuclides: Vec<String> = tally
        .getattr("parent_nuclides")?
        .extract::<Option<Vec<String>>>()?
        .ok_or_else(|| {
            PyValueError::new_err(
                "tally has no parent_nuclides; build the Tally with \
                 parent_nuclides=model.radionuclides() for a decay-photon dose tally",
            )
        })?;
    let scores: Vec<String> = tally.getattr("scores")?.extract()?;
    let mean: Vec<f64> = tr.getattr("mean")?.extract()?;
    let std_dev: Vec<f64> = tr.getattr("standard_deviation")?.extract()?;
    Ok((nuclides, scores.len(), mean, std_dev))
}

/// Resolve the `steps` argument into TCF indices (end-of-step), returning
/// `(indices, single)` where `single` is true for a bare-int selection (so the
/// caller drops the step dimension). Step `i` maps to TCF index `i + 1`; the
/// baseline (index 0) is never selectable.
fn resolve_steps(steps: Option<&Bound<'_, PyAny>>, n: usize) -> PyResult<(Vec<usize>, bool)> {
    let resolve_one = |i: i64| -> PyResult<usize> {
        let j = if i < 0 { i + n as i64 } else { i };
        if j < 0 || j >= n as i64 {
            return Err(PyIndexError::new_err(format!(
                "schedule step {i} out of range for {n} steps"
            )));
        }
        Ok(j as usize + 1)
    };
    match steps {
        None => Ok(((1..=n).collect(), false)),
        Some(obj) if obj.is_none() => Ok(((1..=n).collect(), false)),
        Some(obj) => {
            if obj.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "steps must be None, an int, or a sequence of ints",
                ));
            }
            if let Ok(i) = obj.extract::<i64>() {
                return Ok((vec![resolve_one(i)?], true));
            }
            if let Ok(v) = obj.extract::<Vec<i64>>() {
                let mut out = Vec::with_capacity(v.len());
                for i in v {
                    out.push(resolve_one(i)?);
                }
                return Ok((out, false));
            }
            Err(PyTypeError::new_err(
                "steps must be None, an int, or a sequence of ints",
            ))
        }
    }
}

/// Turn per-step rows into a Python object: a flat 1-D list for a single
/// selected step, else a 2-D list (one row per step).
fn shape_rows(py: Python<'_>, mut rows: Vec<Vec<f64>>, single: bool) -> PyResult<Py<PyAny>> {
    if single {
        let row = rows.pop().unwrap_or_default();
        Ok(row.into_pyobject(py)?.into_any().unbind())
    } else {
        Ok(PyList::new(py, rows)?.into_any().unbind())
    }
}
