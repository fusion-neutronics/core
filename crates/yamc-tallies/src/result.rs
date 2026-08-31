//! Finalized tally result.
//!
//! `TallyResult` is a snapshot of a `Tally`'s accumulated data at the point
//! simulation ends. It contains only the computed outputs (mean, std-dev,
//! relative error, counts) plus metadata (shape, dim-labels,
//! batch counts) and a reference back to the config `Tally` that produced it.
//!
//! This type is additive in PR A: `Tally` is unchanged, and `Tally::finalize`
//! produces a `TallyResult` without disturbing any existing code paths. PR B
//! builds `SimulationResults` on top of this type.
//!
//! ## Shape & dim labels
//!
//! The shape/dim-labels logic appends dimensions in a canonical order:
//!
//!   score → nuclide → parent_nuclide → energy → mesh_z → mesh_y → mesh_x
//!
//! A dimension only appears if it has more than one bin (or, for mesh, if a
//! mesh filter is present). The flat arrays (mean, std-dev, etc.) are laid
//! out in row-major order over this dimension list.
//!
//! ## Lifetime
//!
//! Holds an `Arc<Tally>` back-reference to the config. The config is cheap
//! to share and the Arc ensures the filters / scores used to compute the
//! shape remain valid for as long as the result lives.
use std::sync::Arc;

#[cfg(feature = "mesh")]
use crate::filter::Filter;
use crate::tally::Tally;

/// Immutable snapshot of a tally's finalized results.
#[derive(Debug, Clone)]
pub struct TallyResult {
    /// Reference back to the input `Tally` config. Shared via `Arc` so multiple
    /// `TallyResult`s can reference the same config without copying scores/filters.
    pub tally: Arc<Tally>,

    // --- Finalized numeric data, flat (row-major over `dim_labels`) ---
    /// Mean value per bin: `sum / n_batches`.
    pub mean: Vec<f64>,
    /// Standard error of the mean per bin (Bessel-corrected, divided by sqrt(n)).
    pub standard_deviation: Vec<f64>,
    /// `standard_deviation / mean`, clamped to 0 when `mean == 0`.
    pub relative_error: Vec<f64>,
    /// Raw per-bin Welford sum of squared deviations: the exact merge
    /// state from which `standard_deviation` is derived. Carried so
    /// results stay combinable (`combine_results`) without precision
    /// loss; empty when the producing path installed no Welford state
    /// (e.g. GPU runs).
    pub m2: Vec<f64>,
    /// Exact total source-history count. `n_batches` is the legacy u32
    /// mirror of this and saturates at ~4.29e9; this field does not.
    pub n_histories: u64,
    /// Total count per bin: `mean * particles_per_chunk * n_batches`.
    pub total_count: Vec<u64>,
    /// Figure of merit per bin: `1 / (relative_error² × elapsed_secs)`. The
    /// standard Monte Carlo metric for "convergence rate": higher = faster
    /// convergence of THIS bin's mean estimate. Bins with zero
    /// `relative_error` (uncertain, untouched, or zero-mean) get 0.0 to
    /// avoid `inf` propagating. Populated by [`TallyResult::with_fom`];
    /// empty (`vec![]`) if the result was built via [`Tally::finalize`]
    /// without subsequently calling `with_fom`.
    pub figure_of_merit: Vec<f64>,
    /// Tally-level scalar figure of merit, computed from the bin-summed
    /// mean and std-error (matching `Tally::total_mean` / `total_std`).
    /// Useful as a headline convergence number for a tally even when
    /// individual bins are too noisy to compare. `0.0` if not populated
    /// or if `total_std` is zero.
    pub aggregate_figure_of_merit: f64,

    // --- Aggregate reliability stats (per-history total score) ---
    /// Moments of the per-history *total* score (sum over bins). The source
    /// for the tally-level variance-of-the-variance, skewness and kurtosis
    /// ([`aggregate_variance_of_variance`](Self::aggregate_variance_of_variance)
    /// etc.). `AggMoments::ZERO` when no per-history sampling occurred
    /// (e.g. GPU runs). Stored raw so `combine_results` can merge it.
    pub agg: crate::welford::AggMoments,
    /// Raw empirical probability density of the per-history total score
    /// (log-spaced histogram of `|total|`). Empty when no per-history
    /// sampling occurred. Stored raw so it can be merged and inspected.
    pub score_pdf: crate::welford::ScorePdf,
    /// Snapshots of the tally's aggregate statistics versus number of
    /// histories, for convergence/trend inspection. Empty unless the run
    /// recorded them; not merged across `combine_results`.
    pub convergence_history: Vec<ConvergencePoint>,

    // --- Shape metadata ---
    /// Per-dimension sizes in canonical order (see module docs).
    pub shape: Vec<usize>,
    /// Dimension name per axis of `shape`. Same length as `shape`.
    pub dim_labels: Vec<String>,

    // --- Batch metadata ---
    /// Number of batches accumulated (`n_realizations`).
    pub n_batches: u32,
    /// Number of source particles per batch.
    pub particles_per_chunk: u32,

    // --- Provenance ---
    /// Wall-clock seconds attributed to this tally's data: the producing
    /// run's elapsed time, or after a `combine_results` merge the sum
    /// over contributing runs. This is the time used by `with_fom`.
    pub elapsed_secs: f64,
    /// Indices into `SimulationResults::runs` of the runs that
    /// contributed to this tally's statistics.
    pub run_indices: Vec<usize>,
}

impl TallyResult {
    /// Number of scalar bins (product of `shape`).
    pub fn num_bins(&self) -> usize {
        self.shape.iter().product()
    }

    /// Variance of the mean per bin: `standard_deviation²`. Computed on
    /// the fly (not a stored field). This is the variance OF THE MEAN --
    /// the square of the reported statistical uncertainty -- not the
    /// per-history sample variance (which is recoverable from `m2`).
    pub fn variance(&self) -> Vec<f64> {
        self.standard_deviation.iter().map(|s| s * s).collect()
    }

    /// Mean of the per-history total score (sum over bins). Equals the
    /// bin-summed mean in exact arithmetic.
    pub fn aggregate_mean(&self) -> f64 {
        self.agg.mean
    }

    /// Standard error of the per-history total score's mean. Unlike a
    /// bin-independence assumption, this captures within-history correlation
    /// between bins. 0 when fewer than two histories.
    pub fn aggregate_std_dev(&self) -> f64 {
        if self.agg.n < 2 {
            return 0.0;
        }
        let n = self.agg.n as f64;
        (self.agg.m2 / ((n - 1.0) * n)).max(0.0).sqrt()
    }

    /// Relative error of the per-history total: `aggregate_std_dev / |mean|`.
    pub fn aggregate_relative_error(&self) -> f64 {
        let m = self.agg.mean.abs();
        if m > 0.0 {
            self.aggregate_std_dev() / m
        } else {
            0.0
        }
    }

    /// Relative variance of the variance of the per-history total score.
    /// A sensitive reliability metric: large values mean the reported error
    /// is itself unreliable (rare large contributions undersampled).
    pub fn aggregate_variance_of_variance(&self) -> f64 {
        self.agg.variance_of_variance()
    }

    /// Skewness of the per-history total score distribution.
    pub fn aggregate_skewness(&self) -> f64 {
        self.agg.skewness()
    }

    /// Excess kurtosis of the per-history total score distribution.
    pub fn aggregate_kurtosis(&self) -> f64 {
        self.agg.excess_kurtosis()
    }

    /// Slope of the large-score tail of the empirical PDF (`f(x) ∝ x^{-s}`).
    /// The tally's variance is well-defined only when `s > 3`. Returns 0.0
    /// when not estimable (too few populated tail bins).
    pub fn aggregate_tail_slope(&self) -> f64 {
        self.score_pdf.tail_slope()
    }

    /// Evaluate the statistical-reliability checks for this tally. Each
    /// check is `None` when it cannot be evaluated (too few histories, or
    /// no convergence history recorded for the trend checks).
    pub fn statistical_checks(&self) -> StatisticalChecks {
        let evaluable = self.agg.n >= 2 && self.agg.mean.abs() > 0.0 && self.agg.m2 > 0.0;
        let relative_error_ok = evaluable.then(|| self.aggregate_relative_error() < 0.10);
        let variance_of_variance_ok =
            evaluable.then(|| self.aggregate_variance_of_variance() < 0.10);
        let slope = self.aggregate_tail_slope();
        let tail_slope_ok = (slope > 0.0).then_some(slope > 3.0);
        let (mean_stable, relative_error_decreasing, variance_of_variance_decreasing, fom_stable) =
            trend_checks(&self.convergence_history);
        StatisticalChecks {
            relative_error_ok,
            variance_of_variance_ok,
            tail_slope_ok,
            mean_stable,
            relative_error_decreasing,
            variance_of_variance_decreasing,
            figure_of_merit_stable: fom_stable,
        }
    }

    /// A multi-line, human-readable summary: headline mean ± std, figure of
    /// merit, the higher-moment reliability metrics, and the pass/fail
    /// statistical checks. Used by the Python repr and the end-of-run log.
    pub fn summary(&self) -> String {
        let name = self.tally.name.as_deref().unwrap_or("(unnamed)");
        let has_agg = self.agg.n >= 2;
        let mean = if self.agg.n >= 1 {
            self.agg.mean
        } else {
            self.mean.iter().sum()
        };
        // Aggregate std: from the per-history total when available (captures
        // within-history bin correlation), else the bin-independence fallback.
        let std = if has_agg {
            self.aggregate_std_dev()
        } else {
            self.mean
                .iter()
                .zip(self.relative_error.iter())
                .map(|(&m, &re)| {
                    let s = re * m.abs();
                    s * s
                })
                .sum::<f64>()
                .sqrt()
        };
        let rel = if mean.abs() > 0.0 {
            std / mean.abs()
        } else {
            0.0
        };

        let mut s = String::new();
        s.push_str(&format!(
            "TallyResult '{name}'   shape={:?}   histories={}   elapsed={:.3} s\n",
            self.shape, self.n_histories, self.elapsed_secs
        ));
        s.push_str(&format!(
            "  mean . . . . . . . {mean:.4e} +/- {std:.2e}   ({:.2}% rel err)\n",
            rel * 100.0
        ));
        s.push_str(&format!(
            "  figure of merit .. {:.3e}\n",
            self.aggregate_figure_of_merit
        ));
        if has_agg {
            s.push_str(&format!(
                "  variance of var. . {:.3e}   skewness {:.3}   kurtosis {:.3}\n",
                self.aggregate_variance_of_variance(),
                self.aggregate_skewness(),
                self.aggregate_kurtosis(),
            ));
            let slope = self.aggregate_tail_slope();
            if slope > 0.0 {
                s.push_str(&format!("  large-score tail slope .. {slope:.2}\n"));
            } else {
                s.push_str("  large-score tail slope .. n/a\n");
            }
        }
        let checks = self.statistical_checks();
        s.push_str(&format!(
            "  statistical checks: {} ({}/{})\n",
            if checks.passed() { "PASSED" } else { "FAILED" },
            checks.n_passed(),
            checks.n_evaluated(),
        ));
        for (label, v) in checks.items() {
            let status = match v {
                Some(true) => "pass",
                Some(false) => "FAIL",
                None => "n/a",
            };
            s.push_str(&format!("    {label:.<34} {status}\n"));
        }
        s
    }

    /// Populate `figure_of_merit` (per-bin) and `aggregate_figure_of_merit`
    /// (scalar) using the standard Monte Carlo definition:
    ///
    /// ```text
    /// FOM = 1 / (relative_error² × elapsed_secs)
    /// ```
    ///
    /// `elapsed_secs` should be the wall-clock time of the simulation that
    /// produced this result (typically `SimulationResults.elapsed_secs`).
    /// Consumes `self` and returns a new `TallyResult` with the FOM fields
    /// filled; designed to chain after [`Tally::finalize`]:
    ///
    /// ```ignore
    /// let result = tally.finalize().with_fom(elapsed_secs);
    /// ```
    ///
    /// Bins where `relative_error == 0.0` (uncertain or zero-mean) get
    /// `figure_of_merit = 0.0`. If `elapsed_secs <= 0.0`, all FOMs are 0.0.
    pub fn with_fom(mut self, elapsed_secs: f64) -> Self {
        self.elapsed_secs = elapsed_secs;
        if elapsed_secs <= 0.0 {
            self.figure_of_merit = vec![0.0; self.relative_error.len()];
            self.aggregate_figure_of_merit = 0.0;
            return self;
        }

        self.figure_of_merit = self
            .relative_error
            .iter()
            .map(|&re| {
                if re > 0.0 && re.is_finite() {
                    1.0 / (re * re * elapsed_secs)
                } else {
                    0.0
                }
            })
            .collect();

        // Aggregate: use the bin-summed mean and std-error (mirrors
        // `Tally::total_mean` / `total_std`). For a scalar tally this
        // equals the only bin's FOM; for a mesh tally it's the FOM of
        // the integrated total response.
        let total_mean: f64 = self.mean.iter().sum();
        // Bin scores are independent across batches for the score
        // patterns used here, so var(total) = sum_over_bins(var_of_mean).
        // We derive var-of-mean from relative_error and mean:
        // var_of_mean = (rel_err × mean)².
        let var_total: f64 = self
            .mean
            .iter()
            .zip(self.relative_error.iter())
            .map(|(&m, &re)| {
                let std = re * m.abs();
                std * std
            })
            .sum();
        let total_std = var_total.sqrt();
        self.aggregate_figure_of_merit = if total_mean.abs() > 0.0 && total_std > 0.0 {
            let rel_total = total_std / total_mean.abs();
            1.0 / (rel_total * rel_total * elapsed_secs)
        } else {
            0.0
        };

        self
    }
}

/// A snapshot of a tally's aggregate statistics at a point during the run,
/// for convergence / trend inspection (statistics versus number of
/// histories).
#[derive(Debug, Clone, Copy)]
pub struct ConvergencePoint {
    /// Source histories accumulated at this snapshot.
    pub n_histories: u64,
    /// Aggregate (per-history total) mean.
    pub mean: f64,
    /// Aggregate relative error.
    pub relative_error: f64,
    /// Aggregate variance of the variance.
    pub variance_of_variance: f64,
    /// Aggregate figure of merit at this point.
    pub figure_of_merit: f64,
    /// Large-score tail slope at this point.
    pub tail_slope: f64,
}

/// Outcome of a tally's statistical-reliability checks. Each field is
/// `Some(true)` / `Some(false)` when evaluable, or `None` when it could not
/// be assessed (too few histories, or no convergence history for the trend
/// checks).
#[derive(Debug, Clone, Default)]
pub struct StatisticalChecks {
    /// Relative error below the 0.10 reliability threshold.
    pub relative_error_ok: Option<bool>,
    /// Variance of the variance below 0.10.
    pub variance_of_variance_ok: Option<bool>,
    /// Large-score tail slope above 3 (finite variance).
    pub tail_slope_ok: Option<bool>,
    /// Mean shows no systematic drift over the last half of the run.
    pub mean_stable: Option<bool>,
    /// Relative error decreasing over the last half of the run.
    pub relative_error_decreasing: Option<bool>,
    /// Variance of the variance decreasing over the last half of the run.
    pub variance_of_variance_decreasing: Option<bool>,
    /// Figure of merit statistically constant over the last half of the run.
    pub figure_of_merit_stable: Option<bool>,
}

impl StatisticalChecks {
    /// The checks in display order: `(label, outcome)`.
    pub fn items(&self) -> [(&'static str, Option<bool>); 7] {
        [
            ("mean stable", self.mean_stable),
            ("relative error < 0.10", self.relative_error_ok),
            ("relative error decreasing", self.relative_error_decreasing),
            ("variance of variance < 0.10", self.variance_of_variance_ok),
            (
                "variance of variance decreasing",
                self.variance_of_variance_decreasing,
            ),
            ("figure of merit stable", self.figure_of_merit_stable),
            ("large-score tail slope > 3", self.tail_slope_ok),
        ]
    }

    /// Number of evaluated checks that passed.
    pub fn n_passed(&self) -> usize {
        self.items()
            .iter()
            .filter(|(_, v)| *v == Some(true))
            .count()
    }

    /// Number of checks that were actually evaluated (not `None`).
    pub fn n_evaluated(&self) -> usize {
        self.items().iter().filter(|(_, v)| v.is_some()).count()
    }

    /// True when every evaluated check passed (unevaluated checks don't fail).
    pub fn passed(&self) -> bool {
        self.items().iter().all(|(_, v)| v.unwrap_or(true))
    }

    /// A plain-text pass / FAIL / n/a table of the checks.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        for (label, v) in self.items() {
            let status = match v {
                Some(true) => "pass",
                Some(false) => "FAIL",
                None => "n/a",
            };
            s.push_str(&format!("  {label:.<34} {status}\n"));
        }
        let verdict = if self.passed() { "PASSED" } else { "FAILED" };
        s.push_str(&format!(
            "  result: {verdict} ({}/{})\n",
            self.n_passed(),
            self.n_evaluated()
        ));
        s
    }
}

/// Heuristic trend checks over the last half of the convergence history.
/// Returns `(mean_stable, rel_err_decreasing, vov_decreasing, fom_stable)`,
/// each `None` when there are too few points to judge.
fn trend_checks(
    h: &[ConvergencePoint],
) -> (Option<bool>, Option<bool>, Option<bool>, Option<bool>) {
    if h.len() < 4 {
        return (None, None, None, None);
    }
    let half = &h[h.len() / 2..];
    if half.len() < 3 {
        return (None, None, None, None);
    }
    let means: Vec<f64> = half.iter().map(|p| p.mean).collect();
    let res: Vec<f64> = half.iter().map(|p| p.relative_error).collect();
    let vovs: Vec<f64> = half.iter().map(|p| p.variance_of_variance).collect();
    let foms: Vec<f64> = half.iter().map(|p| p.figure_of_merit).collect();
    (
        Some(!is_monotonic(&means)),
        Some(is_non_increasing(&res)),
        Some(is_non_increasing(&vovs)),
        Some(is_roughly_constant(&foms)),
    )
}

/// True only for a consistent one-directional drift (strictly increasing or
/// strictly decreasing). A constant or wandering series is not monotonic, so
/// `mean_stable = !is_monotonic` treats those as stable.
fn is_monotonic(v: &[f64]) -> bool {
    let inc = v.windows(2).all(|w| w[1] > w[0]);
    let dec = v.windows(2).all(|w| w[1] < w[0]);
    inc || dec
}

fn is_non_increasing(v: &[f64]) -> bool {
    v.windows(2).all(|w| w[1] <= w[0] * 1.05 + 1e-300)
}

fn is_roughly_constant(v: &[f64]) -> bool {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &x in v {
        if x.is_finite() && x > 0.0 {
            lo = lo.min(x);
            hi = hi.max(x);
        }
    }
    lo.is_finite() && hi.is_finite() && lo > 0.0 && hi / lo <= 2.0
}

/// Compute `(shape, dim_labels)` for a tally.
///
/// Dimension order (each included only if it has >1 bin, except score which
/// is always included and mesh which is included whenever a mesh filter is
/// present):
///
///   score → nuclide → parent_nuclide → energy → mesh_z → mesh_y → mesh_x
pub(crate) fn compute_shape_and_dims(tally: &Tally) -> (Vec<usize>, Vec<String>) {
    let mut shape = Vec::with_capacity(8);
    let mut dims: Vec<String> = Vec::with_capacity(8);

    // Score dimension is always present.
    shape.push(tally.scores.len());
    dims.push("score".to_string());

    // Nuclide dimension only when nuclides list is non-empty.
    if !tally.nuclides.is_empty() {
        shape.push(tally.nuclides.len());
        dims.push("nuclide".to_string());
    }

    // Parent-nuclide dimension only when the filter has >1 bins.
    let n_parent = tally.num_parent_nuclide_bins();
    if n_parent > 1 {
        shape.push(n_parent);
        dims.push("parent_nuclide".to_string());
    }

    // Energy dimension only when the filter has >1 bins.
    let n_energy = tally.num_energy_bins();
    if n_energy > 1 {
        shape.push(n_energy);
        dims.push("energy".to_string());
    }

    // Mesh: present whenever a mesh filter exists. Split into the mesh's three
    // axes to match the Python helper's canonical row-major layout. The slowest
    // axis comes first (z for rectangular, z for cylindrical).
    if let Some(mesh_filter) = tally.get_mesh_filter() {
        if let Some(rect) = mesh_filter.rectangular_mesh() {
            let [nx, ny, nz] = rect.shape();
            shape.extend_from_slice(&[nz, ny, nx]);
            dims.extend_from_slice(&[
                "mesh_z".to_string(),
                "mesh_y".to_string(),
                "mesh_x".to_string(),
            ]);
        } else if let Some(cyl) = mesh_filter.cylindrical_mesh() {
            let [nr, nphi, nz] = cyl.shape();
            shape.extend_from_slice(&[nz, nphi, nr]);
            dims.extend_from_slice(&[
                "mesh_z".to_string(),
                "mesh_phi".to_string(),
                "mesh_r".to_string(),
            ]);
        }
    }

    // Also consider the unstructured mesh filter when the `mesh` feature is on.
    #[cfg(feature = "mesh")]
    {
        // Avoid double-dimensioning: only add if no regular mesh was added above
        // AND we have an unstructured mesh filter.
        let has_regular_mesh = tally.get_mesh_filter().is_some();
        if !has_regular_mesh {
            if let Some(f) = tally.filters.iter().find_map(|f| match f {
                Filter::UnstructuredMesh(um) => Some(um),
                _ => None,
            }) {
                shape.push(f.num_bins());
                dims.push("mesh".to_string());
            }
        }
    }

    (shape, dims)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{FluxScore, Score};

    #[test]
    fn finalize_matches_getters() {
        // Build a minimal tally with a single flux score, no filters.
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.initialize_batches(1);
        let tally = Arc::new(tally);

        let result = tally.finalize();

        // Finalized fields should match the current getter values bit-exact.
        assert_eq!(result.mean, tally.get_mean());
        assert_eq!(result.standard_deviation, tally.get_std_dev());
        assert_eq!(result.relative_error, tally.get_rel_error());
        assert_eq!(result.total_count, tally.total_count());
        assert_eq!(result.n_batches, tally.get_n_realizations());

        // FOM fields default to empty / zero until `with_fom` is called.
        assert!(result.figure_of_merit.is_empty());
        assert_eq!(result.aggregate_figure_of_merit, 0.0);

        // Config back-reference should point at the same Arc.
        assert!(Arc::ptr_eq(&result.tally, &tally));
    }

    /// `with_fom(elapsed)` populates `figure_of_merit` per bin to
    /// `1 / (rel_err² × elapsed)` and `aggregate_figure_of_merit`
    /// using the integrated total mean / total std-error.
    #[test]
    fn with_fom_computes_standard_definition() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.initialize_batches(1);
        let tally = Arc::new(tally);
        let mut result = tally.finalize();

        // Inject synthetic stats so we can exercise the FOM math
        // independent of any actual simulation.
        result.mean = vec![1.0, 4.0];
        result.standard_deviation = vec![0.1, 0.2];
        result.relative_error = vec![0.1, 0.05];

        let elapsed = 4.0;
        let result = result.with_fom(elapsed);

        // FOM[i] = 1 / (rel_err[i]² × t)
        // bin 0: 1 / (0.01 × 4) = 25
        // bin 1: 1 / (0.0025 × 4) = 100
        assert!((result.figure_of_merit[0] - 25.0).abs() < 1e-12);
        assert!((result.figure_of_merit[1] - 100.0).abs() < 1e-12);

        // Aggregate: total_mean = 5.0; var_total = 0.1² + 0.2² = 0.05;
        // total_std = sqrt(0.05) ≈ 0.2236; rel_total = 0.04472;
        // FOM = 1 / (0.002 × 4) = 125.
        let total_mean = 5.0_f64;
        let var_total = 0.1_f64 * 0.1 + 0.2 * 0.2;
        let total_std = var_total.sqrt();
        let rel_total = total_std / total_mean;
        let expected_aggregate = 1.0 / (rel_total * rel_total * elapsed);
        assert!(
            (result.aggregate_figure_of_merit - expected_aggregate).abs() < 1e-12,
            "aggregate FOM {} vs expected {}",
            result.aggregate_figure_of_merit,
            expected_aggregate
        );
    }

    /// `with_fom(0.0)` (or negative) should return all-zero FOMs
    /// rather than producing `inf` / `NaN`.
    #[test]
    fn with_fom_zero_elapsed_returns_zero_foms() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.initialize_batches(1);
        let tally = Arc::new(tally);
        let mut result = tally.finalize();
        result.mean = vec![1.0, 2.0];
        result.standard_deviation = vec![0.1, 0.2];
        result.relative_error = vec![0.1, 0.1];

        let result = result.with_fom(0.0);
        assert_eq!(result.figure_of_merit, vec![0.0, 0.0]);
        assert_eq!(result.aggregate_figure_of_merit, 0.0);
    }

    /// Bins with zero relative_error (uncertain or zero-mean) should
    /// get FOM = 0.0 rather than `inf`.
    #[test]
    fn with_fom_zero_rel_err_bins_get_zero_fom() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.initialize_batches(1);
        let tally = Arc::new(tally);
        let mut result = tally.finalize();
        result.mean = vec![1.0, 0.0, 2.0];
        result.standard_deviation = vec![0.1, 0.0, 0.2];
        result.relative_error = vec![0.1, 0.0, 0.1];

        let result = result.with_fom(1.0);
        // bin 0: rel_err = 0.1 → FOM = 100
        // bin 1: rel_err = 0   → FOM = 0 (would be inf)
        // bin 2: rel_err = 0.1 → FOM = 100
        assert!((result.figure_of_merit[0] - 100.0).abs() < 1e-12);
        assert_eq!(result.figure_of_merit[1], 0.0);
        assert!((result.figure_of_merit[2] - 100.0).abs() < 1e-12);
        for fom in &result.figure_of_merit {
            assert!(fom.is_finite(), "FOM should be finite, got {fom}");
        }
    }

    /// Verify the standard MC scaling: doubling the simulation time
    /// should halve the FOM (since FOM ∝ 1/t at fixed rel_err).
    #[test]
    fn with_fom_inversely_proportional_to_elapsed() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.initialize_batches(1);
        let tally = Arc::new(tally);
        let mut result = tally.finalize();
        result.mean = vec![1.0];
        result.standard_deviation = vec![0.05];
        result.relative_error = vec![0.05];

        let r1 = result.clone().with_fom(1.0);
        let r2 = result.with_fom(2.0);
        assert!(
            (r1.figure_of_merit[0] / r2.figure_of_merit[0] - 2.0).abs() < 1e-12,
            "FOM at t=1 should be 2× FOM at t=2"
        );
        assert!((r1.aggregate_figure_of_merit / r2.aggregate_figure_of_merit - 2.0).abs() < 1e-12,);
    }

    /// `variance()` returns `standard_deviation²` element-wise.
    #[test]
    fn variance_is_std_dev_squared() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.initialize_batches(1);
        let tally = Arc::new(tally);
        let mut result = tally.finalize();
        result.standard_deviation = vec![0.1, 0.2, 0.5];
        let var = result.variance();
        assert_eq!(var.len(), 3);
        assert!((var[0] - 0.01).abs() < 1e-15);
        assert!((var[1] - 0.04).abs() < 1e-15);
        assert!((var[2] - 0.25).abs() < 1e-15);
    }

    #[test]
    fn shape_and_dims_minimal_tally() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];

        let (shape, dims) = compute_shape_and_dims(&tally);
        assert_eq!(shape, vec![1]);
        assert_eq!(dims, vec!["score".to_string()]);
    }

    #[test]
    fn shape_and_dims_multiple_scores() {
        use crate::score::HeatingScore;

        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore), Score::Heating(HeatingScore)];

        let (shape, dims) = compute_shape_and_dims(&tally);
        assert_eq!(shape, vec![2]);
        assert_eq!(dims, vec!["score".to_string()]);
    }

    fn flux_result() -> TallyResult {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.initialize_batches(1);
        Arc::new(tally).finalize()
    }

    /// A well-converged aggregate (large n, tiny spread, light tail) passes
    /// every evaluable reliability check.
    #[test]
    fn statistical_checks_pass_for_well_converged() {
        let mut r = flux_result();
        r.agg = crate::welford::AggMoments {
            n: 100_000,
            mean: 2.0,
            m2: 1.0e-3,
            m3: 0.0,
            m4: 1.0e-9,
        };
        let mut pdf = crate::welford::ScorePdf::new();
        for _ in 0..200 {
            pdf.record(2.0); // identical scores -> light tail
        }
        r.score_pdf = pdf;
        let c = r.statistical_checks();
        assert_eq!(c.relative_error_ok, Some(true));
        assert_eq!(c.variance_of_variance_ok, Some(true));
        assert_eq!(c.tail_slope_ok, Some(true));
        assert!(c.passed());
    }

    /// A large fourth moment (heavy contributions) drives the variance of the
    /// variance over threshold, failing the verdict.
    #[test]
    fn statistical_checks_flag_high_variance_of_variance() {
        let mut r = flux_result();
        r.agg = crate::welford::AggMoments {
            n: 1000,
            mean: 1.0,
            m2: 1.0,
            m3: 0.0,
            m4: 1.0e6,
        };
        let c = r.statistical_checks();
        assert_eq!(c.variance_of_variance_ok, Some(false));
        assert!(!c.passed());
    }

    /// With no per-history samples the magnitude/slope checks are "not
    /// evaluated" (None) and the verdict passes vacuously.
    #[test]
    fn statistical_checks_not_evaluated_without_samples() {
        let r = flux_result();
        let c = r.statistical_checks();
        assert_eq!(c.relative_error_ok, None);
        assert_eq!(c.variance_of_variance_ok, None);
        assert_eq!(c.tail_slope_ok, None);
        assert_eq!(c.n_evaluated(), 0);
        assert!(c.passed());
    }

    /// Trend checks read the last half of the convergence history: a constant
    /// mean is stable, and decreasing relative error / VOV pass.
    #[test]
    fn trend_checks_from_convergence_history() {
        let mut r = flux_result();
        r.convergence_history = (1..=8)
            .map(|i| {
                let f = i as f64;
                ConvergencePoint {
                    n_histories: (i as u64) * 1000,
                    mean: 2.0,
                    relative_error: 0.1 / f.sqrt(),
                    variance_of_variance: 0.01 / f,
                    figure_of_merit: 100.0,
                    tail_slope: 8.0,
                }
            })
            .collect();
        let c = r.statistical_checks();
        assert_eq!(c.mean_stable, Some(true));
        assert_eq!(c.relative_error_decreasing, Some(true));
        assert_eq!(c.variance_of_variance_decreasing, Some(true));
        assert_eq!(c.figure_of_merit_stable, Some(true));
    }
}
