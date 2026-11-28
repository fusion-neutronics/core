//! Finalized tally result.
//!
//! `TallyResult` is a snapshot of a `Tally`'s accumulated data at the point
//! simulation ends. It contains only the computed outputs (mean, std-dev,
//! relative error, counts) plus metadata (shape, dim-labels,
//! history count) and a reference back to the config `Tally` that produced it.
//!
//! `Tally::finalize` produces a `TallyResult` without disturbing the tally's
//! accumulation state, and `TransportResults` is built on top of this type.
//!
//! ## Shape & dim labels
//!
//! The shape/dim-labels logic appends dimensions in a canonical order:
//!
//!   score → cell → material → nuclide → parent_nuclide → energy → mesh_z → mesh_y → mesh_x
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
use crate::units::SourceStrength;

/// Immutable snapshot of a tally's finalized results.
#[derive(Debug, Clone)]
pub struct TallyResult {
    /// Reference back to the input `Tally` config. Shared via `Arc` so multiple
    /// `TallyResult`s can reference the same config without copying scores/filters.
    pub tally: Arc<Tally>,

    // --- Finalized numeric data, flat (row-major over `dim_labels`) ---
    /// Mean value per bin per source history: `sum / n_histories`.
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
    /// Exact total source-history count.
    pub n_histories: u64,
    /// Total score per bin over all histories: `mean * n_histories`.
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
    /// Raw per-history products of the bin scores, packed upper triangle, when
    /// the tally asked for its covariance. The merge state behind
    /// [`TallyResult::covariance`], carried like `m2` so combined results keep
    /// it. Not written by the Arrow result writer.
    pub comoment: Option<Vec<f64>>,
    /// Nuclear-data replica sums, when the run carried correlated replica
    /// weights: the merge state behind
    /// [`TallyResult::nuclear_data_standard_deviation`] and
    /// [`TallyResult::replica_mean`], carried like `comoment` so combined
    /// results of the same draws keep it. Not written by the Arrow result
    /// writer.
    pub replicas: Option<crate::welford::ReplicaSums>,
    /// Snapshots of the tally's aggregate statistics versus number of
    /// histories, for convergence/trend inspection. Empty unless the run
    /// recorded them; not merged across `combine_results`.
    pub convergence_history: Vec<ConvergencePoint>,

    // --- Shape metadata ---
    /// Per-dimension sizes in canonical order (see module docs).
    pub shape: Vec<usize>,
    /// Dimension name per axis of `shape`. Same length as `shape`.
    pub dim_labels: Vec<String>,

    // --- Provenance ---
    /// Wall-clock seconds attributed to this tally's data: the producing
    /// run's elapsed time, or after a `combine_results` merge the sum
    /// over contributing runs. This is the time used by `with_fom`.
    pub elapsed_secs: f64,
    /// Indices into `TransportResults::runs` of the runs that
    /// contributed to this tally's statistics.
    pub run_indices: Vec<usize>,

    /// Source strength and units the values are reported in. The fields
    /// above stay per source particle; [`TallyResult::scaled`] applies this.
    pub normalisation: Normalisation,
}

/// What turns a result per source particle into the values it reports: the
/// source strength, the units it was converted to, and the cell volumes a
/// per-volume unit divides by.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Normalisation {
    /// Source particles per second or per pulse, `None` for results per
    /// source particle.
    pub source_strength: Option<SourceStrength>,
    /// Per score, the unit [`TallyResult::to_units`] converted to, `None` for
    /// the result's own unit.
    pub units: Option<Vec<String>>,
    /// Volume in cm3 of each cell of the tally's cell filter, in filter
    /// order, `None` for a cell without one. Set from the geometry when the
    /// results are built.
    pub cell_volumes: Vec<Option<f64>>,
    /// Mass density in g/cm3 of each cell of the tally's cell filter, in
    /// filter order, `None` for a void cell. Set from the geometry with
    /// `cell_volumes`, so a per-mass unit can divide by each cell's mass.
    #[serde(default)]
    pub cell_densities: Vec<Option<f64>>,
    /// Per-bin factor from the per-source-particle fields to the reported
    /// values: empty for 1, one entry when every bin shares it.
    #[serde(skip)]
    pub scale: Vec<f64>,
}

/// The share of the nuclear-data variance the deconvolved Monte Carlo noise
/// may reach before [`TallyResult::nuclear_data_noise_dominated`] flags a
/// bin. Transmutation outputs are flagged at the same share
/// (`yani_transmute::uncertainty::NOISE_FLAG_FRACTION`).
pub const NOISE_FLAG_FRACTION: f64 = 0.25;

impl TallyResult {
    /// This result with `strength` applied, replacing any strength it had.
    /// `None` returns it to values per source particle. Units it was
    /// converted to are kept, and must still fit the new strength.
    pub fn with_source_strength(&self, strength: Option<SourceStrength>) -> Result<Self, String> {
        let mut r = self.clone();
        r.normalisation.source_strength = strength;
        r.rescale()?;
        Ok(r)
    }

    /// This result converted to `units`: one unit for every score, or one per
    /// score. A per-volume unit (`W/cm3`, `1/(cm2 s)`) divides each bin by
    /// its volume.
    pub fn to_units(&self, units: &[String]) -> Result<Self, String> {
        let n_scores = self.tally.scores.len();
        let units = match units.len() {
            1 => vec![units[0].trim().to_string(); n_scores],
            n if n == n_scores => units.iter().map(|u| u.trim().to_string()).collect(),
            n => {
                return Err(format!(
                    "got {n} units for a tally with {n_scores} scores: give one unit, or one per score"
                ));
            }
        };
        let mut r = self.clone();
        r.normalisation.units = Some(units);
        r.rescale()?;
        Ok(r)
    }

    /// Per score, the unit the reported values are in: `"eV/s"`,
    /// `"reactions/source-particle"`, or the unit given to
    /// [`TallyResult::to_units`].
    pub fn units(&self) -> Vec<String> {
        let n = &self.normalisation;
        match &n.units {
            Some(units) if n.source_strength.is_some() => units.clone(),
            Some(units) => units
                .iter()
                .map(|u| format!("{u}/source-particle"))
                .collect(),
            None => crate::units::result_units(&self.tally, n.source_strength),
        }
    }

    /// `raw` (per source particle, one value per bin) as reported.
    pub fn scaled(&self, raw: &[f64]) -> Vec<f64> {
        match self.normalisation.scale.as_slice() {
            [] => raw.to_vec(),
            [s] => raw.iter().map(|v| v * s).collect(),
            scale => raw.iter().zip(scale).map(|(v, s)| v * s).collect(),
        }
    }

    /// `raw` variances (per source particle squared) as reported.
    pub fn scaled_squared(&self, raw: &[f64]) -> Vec<f64> {
        match self.normalisation.scale.as_slice() {
            [] => raw.to_vec(),
            [s] => raw.iter().map(|v| v * s * s).collect(),
            scale => raw.iter().zip(scale).map(|(v, s)| v * s * s).collect(),
        }
    }

    /// The factor every bin shares, `None` when bins were divided by
    /// different volumes (so a total over bins has no unit).
    pub fn uniform_scale(&self) -> Option<f64> {
        match self.normalisation.scale.as_slice() {
            [] => Some(1.0),
            [s] => Some(*s),
            _ => None,
        }
    }

    /// Recompute `normalisation.scale` from the strength and units.
    pub fn rescale(&mut self) -> Result<(), String> {
        let strength = self.normalisation.source_strength;
        let s = strength.map_or(1.0, |s| s.value);
        let Some(units) = self.normalisation.units.clone() else {
            self.normalisation.scale = if s == 1.0 { Vec::new() } else { vec![s] };
            return Ok(());
        };
        let conversions = self
            .tally
            .scores
            .iter()
            .zip(&units)
            .map(|(score, unit)| crate::units::conversion(&self.tally, score, strength, unit))
            .collect::<Result<Vec<_>, _>>()?;
        use crate::units::Per;
        let per_volume = conversions.iter().any(|c| c.per == Per::Volume);
        let per_mass = conversions.iter().any(|c| c.per == Per::Mass);
        let uniform = conversions.windows(2).all(|w| w[0] == w[1]);
        self.normalisation.scale = if !per_volume && !per_mass && uniform {
            vec![s * conversions.first().map_or(1.0, |c| c.factor)]
        } else {
            let volumes = if per_volume {
                self.bin_volumes()?
            } else {
                Vec::new()
            };
            let masses = if per_mass {
                self.bin_masses()?
            } else {
                Vec::new()
            };
            let n = self.num_bins();
            let per_score = (n / conversions.len().max(1)).max(1);
            let scale: Vec<f64> = (0..n)
                .map(|b| {
                    let c = conversions[(b / per_score).min(conversions.len() - 1)];
                    let v = match c.per {
                        Per::Bin => 1.0,
                        Per::Volume => volumes[b],
                        Per::Mass => masses[b],
                    };
                    s * c.factor / v
                })
                .collect();
            // Equal volumes (a rectangular mesh) leave one shared factor.
            match scale.first() {
                Some(&k) if scale.iter().all(|&x| x == k) => vec![k],
                _ => scale,
            }
        };
        Ok(())
    }

    /// Volume in cm3 of each bin, flat in the same order as `mean`: the mesh
    /// element of a mesh tally, or the cell of a cell-filtered tally.
    pub fn bin_volumes(&self) -> Result<Vec<f64>, String> {
        let t = &self.tally;
        let n = self.num_bins();
        let name = t.display_name();
        if t.mesh_parts.is_some() {
            return Err(format!(
                "tally {name}: a mesh tally split by cell has no bin volumes, so it cannot be converted to a per-volume unit"
            ));
        }
        let mesh: Option<Vec<f64>> = t
            .get_mesh_filter()
            .map(|m| (0..m.num_bins()).map(|b| m.get_element_volume(b)).collect());
        #[cfg(feature = "mesh")]
        let mesh = mesh.or_else(|| {
            t.get_unstructured_mesh_filter()
                .map(|m| (0..m.num_bins()).map(|b| m.get_element_volume(b)).collect())
        });
        match (mesh, t.get_cell_filter()) {
            (Some(_), Some(_)) => Err(format!(
                "tally {name} has both a mesh and a cell filter, so the volume of each bin is not known; drop one to convert to a per-volume unit"
            )),
            (Some(v), None) => Ok((0..n).map(|b| v[b % v.len()]).collect()),
            (None, Some(cf)) => {
                let volumes = cf
                    .cell_ids
                    .iter()
                    .enumerate()
                    .map(|(i, id)| {
                        self.normalisation
                            .cell_volumes
                            .get(i)
                            .copied()
                            .flatten()
                            .ok_or_else(|| {
                                format!(
                                    "tally {name}: cell {id} has no volume; set Cell(volume=...) to convert to a per-volume unit"
                                )
                            })
                    })
                    .collect::<Result<Vec<f64>, String>>()?;
                let stride: usize = match self.dim_labels.iter().position(|d| d == "cell") {
                    Some(p) => self.shape[p + 1..].iter().product(),
                    None => n,
                };
                Ok((0..n)
                    .map(|b| volumes[(b / stride.max(1)) % volumes.len()])
                    .collect())
            }
            (None, None) => Err(format!(
                "tally {name} has no bin volumes: a per-volume unit needs a mesh filter, or a cell filter on cells with a volume"
            )),
        }
    }

    /// Mass in g of each bin, flat in the same order as `mean`: the cell's
    /// volume times its material's mass density, for a cell-filtered tally.
    /// A mesh element can straddle materials, so its mass is not known.
    pub fn bin_masses(&self) -> Result<Vec<f64>, String> {
        let t = &self.tally;
        let name = t.display_name();
        let Some(cf) = t.get_cell_filter() else {
            return Err(format!(
                "tally {name} has no bin masses: a per-mass unit needs a cell filter on cells \
                 with a volume and a material (a mesh element's mass is not known)"
            ));
        };
        let volumes = self.bin_volumes()?;
        let densities = cf
            .cell_ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                self.normalisation
                    .cell_densities
                    .get(i)
                    .copied()
                    .flatten()
                    .ok_or_else(|| {
                        format!(
                            "tally {name}: cell {id} has no material, so no mass; a per-mass unit needs one"
                        )
                    })
            })
            .collect::<Result<Vec<f64>, String>>()?;
        let n = self.num_bins();
        let stride: usize = match self.dim_labels.iter().position(|d| d == "cell") {
            Some(p) => self.shape[p + 1..].iter().product(),
            None => n,
        };
        Ok((0..n)
            .map(|b| volumes[b] * densities[(b / stride.max(1)) % densities.len()])
            .collect())
    }

    /// Per-bin nuclear-data standard deviation of the score, when the run
    /// carried replica weights: the first-order variance along the principal
    /// modes plus the nonlinear remainder the replicas estimate, with the
    /// Monte Carlo noise of the shared histories deconvolved (see
    /// [`ReplicaSums::nuclear_data_variance`](crate::welford::ReplicaSums::nuclear_data_variance)).
    /// A bin whose deconvolved variance came out negative, where the noise
    /// swamped the spread, reads zero here; it is among those
    /// [`TallyResult::nuclear_data_noise_dominated`] flags.
    pub fn nuclear_data_standard_deviation(&self) -> Option<Vec<f64>> {
        let r = self.replicas.as_ref()?;
        Some(
            r.nuclear_data_variance(self.n_histories)
                .iter()
                .map(|(v, _)| v.max(0.0).sqrt())
                .collect(),
        )
    }

    /// Per-bin standard deviation of the replicas as scored, before the
    /// Monte Carlo noise is deconvolved: `sqrt(sigma^2 + noise^2)`, when the
    /// run carried replica weights.
    pub fn nuclear_data_replica_spread(&self) -> Option<Vec<f64>> {
        let r = self.replicas.as_ref()?;
        Some(
            r.nuclear_data_variance(self.n_histories)
                .iter()
                .map(|(v, n)| (v + n).max(0.0).sqrt())
                .collect(),
        )
    }

    /// Per-bin standard deviation of the Monte Carlo noise deconvolved from
    /// the replicas' spread, when the run carried replica weights: the noise
    /// in each replica's shift from the nominal that the shared histories do
    /// not cancel. It falls as `1/sqrt(histories)`.
    pub fn nuclear_data_noise(&self) -> Option<Vec<f64>> {
        let r = self.replicas.as_ref()?;
        Some(
            r.nuclear_data_variance(self.n_histories)
                .iter()
                .map(|(_, n)| n.max(0.0).sqrt())
                .collect(),
        )
    }

    /// Per-bin flag: the noise variance deconvolved exceeded
    /// [`NOISE_FLAG_FRACTION`] of the nuclear-data variance left (or that
    /// variance came out negative, and the sigma reads zero). The sigma is
    /// still unbiased, but more histories would sharpen it.
    pub fn nuclear_data_noise_dominated(&self) -> Option<Vec<bool>> {
        let r = self.replicas.as_ref()?;
        Some(
            r.nuclear_data_variance(self.n_histories)
                .iter()
                .map(|(v, n)| *v <= 0.0 && *n > 0.0 || *n > NOISE_FLAG_FRACTION * v.max(0.0))
                .collect(),
        )
    }

    /// Per-bin average of the replicas' means: the expected score over the
    /// nuclear-data uncertainty, beside the nominal `mean`. The two differ
    /// where the score responds nonlinearly to the cross sections.
    pub fn replica_mean(&self) -> Option<Vec<f64>> {
        Some(self.replicas.as_ref()?.replica_mean(self.n_histories))
    }

    /// Per-bin first-order nuclear-data variance: the variance of the score's
    /// linear response to the cross sections along the run's principal modes,
    /// with no sampling error from the replicas. The nuclear-data variance is
    /// this plus the nonlinear remainder the replicas estimate. `None` when
    /// the run carried no modes.
    pub fn nuclear_data_linear_variance(&self) -> Option<Vec<f64>> {
        self.replicas.as_ref()?.linear_variance(self.n_histories)
    }

    /// Per nuclide, per bin, its share of
    /// [`TallyResult::nuclear_data_linear_variance`]: which nuclides' cross
    /// sections the score's uncertainty comes from. The shares add up to the
    /// first-order variance. `None` when the run carried no modes.
    pub fn nuclear_data_variance_by_nuclide(
        &self,
    ) -> Option<std::collections::BTreeMap<String, Vec<f64>>> {
        self.replicas
            .as_ref()?
            .linear_variance_by_nuclide(self.n_histories)
    }

    /// Per-bin Monte Carlo standard error of one replica's mean, averaged over
    /// replicas: compare with the nuclear-data standard deviation to see
    /// whether the replicas resolve it.
    pub fn replica_standard_error(&self) -> Option<Vec<f64>> {
        Some(
            self.replicas
                .as_ref()?
                .replica_standard_error(self.n_histories),
        )
    }

    /// Covariance of the bin means, `num_bins x num_bins` row-major, when the
    /// tally asked for it. Its diagonal is `standard_deviation` squared, and
    /// its off-diagonal terms are the correlations a per-bin standard
    /// deviation cannot carry: bins scored by the same histories move
    /// together.
    pub fn covariance(&self) -> Option<Vec<f64>> {
        let stats = crate::welford::WelfordTallyStats {
            mean: self.mean.clone(),
            m2: self.m2.clone(),
            n_histories: self.n_histories,
            agg: self.agg,
            score_pdf: crate::welford::ScorePdf::default(),
            comoment: self.comoment.clone(),
            replicas: None,
        };
        stats.covariance_of_mean()
    }

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
        StatisticalChecks {
            per_bin_relative_error_ok: self.per_bin_relative_error_check(),
            ..StatisticalChecks::aggregate(
                &self.agg,
                self.aggregate_tail_slope(),
                &self.convergence_history,
                1,
            )
        }
    }

    fn per_bin_relative_error_check(&self) -> Option<bool> {
        if self.n_histories < 2 {
            return None;
        }
        let mut populated = false;
        let mut passed = true;
        for (&mean, &error) in self.mean.iter().zip(&self.standard_deviation) {
            if mean != 0.0 || error != 0.0 {
                populated = true;
                passed &= mean.is_finite()
                    && error.is_finite()
                    && error >= 0.0
                    && mean != 0.0
                    && error / mean.abs() < 0.10;
            }
        }
        populated.then_some(passed)
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
            "TallyResult '{name}'   shape={:?}   particles={}   elapsed={:.3} s\n",
            self.shape, self.n_histories, self.elapsed_secs
        ));
        let units = self.units();
        let unit = if units.windows(2).all(|w| w[0] == w[1]) {
            units.first().cloned().unwrap_or_default()
        } else {
            units.join(", ")
        };
        match self.uniform_scale() {
            Some(k) => s.push_str(&format!(
                "  mean . . . . . . . {:.4e} +/- {:.2e} {unit}   ({:.2}% rel err)\n",
                mean * k,
                std * k,
                rel * 100.0
            )),
            None => s.push_str(&format!(
                "  mean . . . . . . . per bin in {unit} (bins divided by their volumes)   ({:.2}% rel err)\n",
                rel * 100.0
            )),
        }
        if let Some(strength) = self.normalisation.source_strength {
            s.push_str(&format!("  source strength .. {strength}\n"));
        }
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
    /// produced this result (typically `TransportResults.elapsed_secs`).
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

        // Aggregate: the FOM of the tally total, from the relative error of the
        // per-history total, which keeps the correlation between bins that one
        // history scores together. Summing the per-bin variances as if the bins
        // were independent overstated it three to four times on a mesh tally.
        // Results without the per-history total (MPI runs, results read back
        // from Arrow) still fall back to that sum.
        let rel_total = if self.agg.n >= 2 {
            self.aggregate_relative_error()
        } else {
            let total_mean: f64 = self.mean.iter().sum();
            let var_total: f64 = self
                .mean
                .iter()
                .zip(self.relative_error.iter())
                .map(|(&m, &re)| (re * m.abs()).powi(2))
                .sum();
            if total_mean.abs() > 0.0 {
                var_total.sqrt() / total_mean.abs()
            } else {
                0.0
            }
        };
        self.aggregate_figure_of_merit = if rel_total > 0.0 && rel_total.is_finite() {
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
    /// Transport wall-clock time elapsed at this point, in seconds.
    pub elapsed_secs: f64,
}

/// Outcome of a tally's statistical-reliability checks. Each field is
/// `Some(true)` / `Some(false)` when evaluable, or `None` when it could not
/// be assessed (too few histories, or no convergence history for the trend
/// checks).
#[derive(Debug, Clone, Default)]
pub struct StatisticalChecks {
    /// Relative error below the 0.10 reliability threshold.
    pub relative_error_ok: Option<bool>,
    /// Every populated bin has relative error below 0.10. Higher-moment and
    /// tail checks remain aggregate-only; this uses existing per-bin data.
    pub per_bin_relative_error_ok: Option<bool>,
    /// Variance of the variance below 0.10.
    pub variance_of_variance_ok: Option<bool>,
    /// Large-score tail slope above 3 (finite variance).
    pub tail_slope_ok: Option<bool>,
    /// Mean shows no systematic drift over the last half of the run.
    pub mean_stable: Option<bool>,
    /// Relative error decreasing over the last half of the run.
    pub relative_error_decreasing: Option<bool>,
    /// Relative error falling as `1/sqrt(N)` over the last half of the run.
    pub relative_error_rate_ok: Option<bool>,
    /// Variance of the variance decreasing over the last half of the run.
    pub variance_of_variance_decreasing: Option<bool>,
    /// Variance of the variance falling as `1/N` over the last half of the run.
    pub variance_of_variance_rate_ok: Option<bool>,
    /// Figure of merit statistically constant over the last half of the run.
    pub figure_of_merit_stable: Option<bool>,
    /// Figure of merit shows no systematic trend with `N` over the last half
    /// of the run.
    pub figure_of_merit_no_trend: Option<bool>,
}

impl StatisticalChecks {
    /// The checks that need only the per-history aggregate: its moments, the
    /// large-score tail slope and the convergence history. The per-bin check
    /// is left `None`. Cheap enough to run at every batch checkpoint, which is
    /// how a run with no stop condition decides it has converged.
    ///
    /// `family` is how many of these checks are judged together (one per
    /// tally when a run must pass them for every tally). The mean-drift test
    /// is Bonferroni-corrected for it, so the family as a whole keeps the 5%
    /// false-alarm rate of a single test; `1` is a lone tally's own verdict.
    pub fn aggregate(
        agg: &crate::welford::AggMoments,
        tail_slope: f64,
        history: &[ConvergencePoint],
        family: usize,
    ) -> Self {
        let evaluable = agg.n >= 2 && agg.mean.abs() > 0.0 && agg.m2 > 0.0;
        let rel = if evaluable {
            let n = agg.n as f64;
            (agg.m2 / ((n - 1.0) * n)).max(0.0).sqrt() / agg.mean.abs()
        } else {
            0.0
        };
        let (mean_stable, relative_error_decreasing, variance_of_variance_decreasing, fom_stable) =
            trend_checks(history, family);
        let (relative_error_rate_ok, variance_of_variance_rate_ok, figure_of_merit_no_trend) =
            rate_checks(history);
        Self {
            relative_error_ok: evaluable.then_some(rel < 0.10),
            per_bin_relative_error_ok: None,
            variance_of_variance_ok: evaluable.then(|| agg.variance_of_variance() < 0.10),
            tail_slope_ok: (tail_slope > 0.0).then_some(tail_slope > 3.0),
            mean_stable,
            relative_error_decreasing,
            relative_error_rate_ok,
            variance_of_variance_decreasing,
            variance_of_variance_rate_ok,
            figure_of_merit_stable: fom_stable,
            figure_of_merit_no_trend,
        }
    }

    /// The checks in display order: `(label, outcome)`.
    pub fn items(&self) -> [(&'static str, Option<bool>); 11] {
        [
            ("mean stable", self.mean_stable),
            ("relative error < 0.10", self.relative_error_ok),
            (
                "per-bin relative error < 0.10",
                self.per_bin_relative_error_ok,
            ),
            ("relative error decreasing", self.relative_error_decreasing),
            (
                "relative error falls as 1/sqrt(N)",
                self.relative_error_rate_ok,
            ),
            ("variance of variance < 0.10", self.variance_of_variance_ok),
            (
                "variance of variance decreasing",
                self.variance_of_variance_decreasing,
            ),
            (
                "variance of variance falls as 1/N",
                self.variance_of_variance_rate_ok,
            ),
            ("figure of merit stable", self.figure_of_merit_stable),
            (
                "figure of merit has no trend",
                self.figure_of_merit_no_trend,
            ),
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

    /// True when the tail is estimable and every evaluated check passed.
    /// Unavailable optional trend checks remain n/a, not evidence of a pass.
    pub fn passed(&self) -> bool {
        self.tail_slope_ok == Some(true) && self.items().iter().all(|(_, v)| v.unwrap_or(true))
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
        let verdict = if self.passed() {
            "PASSED"
        } else if self.items().iter().any(|(_, value)| *value == Some(false)) {
            "FAILED"
        } else {
            "INCOMPLETE"
        };
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
    family: usize,
) -> (Option<bool>, Option<bool>, Option<bool>, Option<bool>) {
    if h.len() < 4 {
        return (None, None, None, None);
    }
    let half = &h[h.len() / 2..];
    if half.len() < 3 {
        return (None, None, None, None);
    }
    let res: Vec<f64> = half.iter().map(|p| p.relative_error).collect();
    let vovs: Vec<f64> = half.iter().map(|p| p.variance_of_variance).collect();
    let foms: Vec<f64> = half.iter().map(|p| p.figure_of_merit).collect();
    let fom_span = half.last().unwrap().elapsed_secs - half.first().unwrap().elapsed_secs;
    (
        mean_drift_check(half, family),
        Some(is_non_increasing(&res)),
        Some(is_non_increasing(&vovs)),
        (fom_span >= FOM_MIN_SPAN_SECS).then(|| is_roughly_constant(&foms)),
    )
}

/// Fewest last-half snapshots a log-log rate fit is attempted on.
const RATE_MIN_POINTS: usize = 4;
/// Accepted log-log slope of relative error against `N`: `-0.5 ± 0.25`.
const RE_SLOPE_RANGE: (f64, f64) = (-0.75, -0.25);
/// Accepted log-log slope of the variance of the variance against `N`:
/// `-1 ± 0.5`.
const VOV_SLOPE_RANGE: (f64, f64) = (-1.5, -0.5);
/// Largest accepted magnitude of the log-log slope of the figure of merit
/// against `N`.
const FOM_SLOPE_MAX: f64 = 0.5;
/// Least wall-clock time the last half of the run must span before the
/// figure-of-merit trend is judged. Over shorter spans the time in the FOM is
/// dominated by warm-up and timer jitter, not by the tally's statistics.
const FOM_MIN_SPAN_SECS: f64 = 0.1;

/// Rate checks over the last half of the convergence history, fitted as
/// log-log slopes against the history count `N`. Healthy tallies have a
/// relative error falling as `N^-0.5`, a variance of the variance falling as
/// `N^-1`, and a figure of merit with no trend. Returns
/// `(rel_err_rate_ok, vov_rate_ok, fom_no_trend)`, each `None` when there are
/// too few usable points (or, for the figure of merit, too little run time).
///
/// Tolerances were set from runs of well-behaved tallies, whose fitted slopes
/// stay within a few hundredths of the ideal for relative error and within
/// about 0.25 for the variance of the variance, even with only five points.
fn rate_checks(h: &[ConvergencePoint]) -> (Option<bool>, Option<bool>, Option<bool>) {
    let half = &h[h.len() / 2..];
    let in_range = |slope: f64, (lo, hi): (f64, f64)| slope >= lo && slope <= hi;
    let re = log_log_slope(half, |p| p.relative_error).map(|s| in_range(s, RE_SLOPE_RANGE));
    let vov = log_log_slope(half, |p| p.variance_of_variance).map(|s| in_range(s, VOV_SLOPE_RANGE));
    let span = match (half.first(), half.last()) {
        (Some(a), Some(b)) => b.elapsed_secs - a.elapsed_secs,
        _ => 0.0,
    };
    let fom = (span >= FOM_MIN_SPAN_SECS)
        .then(|| log_log_slope(half, |p| p.figure_of_merit).map(|s| s.abs() <= FOM_SLOPE_MAX))
        .flatten();
    (re, vov, fom)
}

/// Least-squares slope of `ln(y)` against `ln(n_histories)`, using only points
/// where both are positive and finite. `None` with fewer than
/// [`RATE_MIN_POINTS`] such points or no spread in `N`.
fn log_log_slope(points: &[ConvergencePoint], y: impl Fn(&ConvergencePoint) -> f64) -> Option<f64> {
    let xy: Vec<(f64, f64)> = points
        .iter()
        .filter_map(|p| {
            let (x, y) = (p.n_histories as f64, y(p));
            (x > 0.0 && y > 0.0 && y.is_finite()).then(|| (x.ln(), y.ln()))
        })
        .collect();
    if xy.len() < RATE_MIN_POINTS {
        return None;
    }
    let n = xy.len() as f64;
    let mx = xy.iter().map(|p| p.0).sum::<f64>() / n;
    let my = xy.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = xy.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = xy.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    (sxx > 0.0).then(|| sxy / sxx)
}

/// Significance of the mean-drift test for one tally: two-sided 5%.
const DRIFT_ALPHA: f64 = 0.05;

/// Whether the block means of the run show no linear drift in `N`: a
/// two-sided t-test on the weighted slope at [`DRIFT_ALPHA`] / `family`
/// (Bonferroni), so `family` simultaneous tests together fail a stationary
/// run no more than [`DRIFT_ALPHA`] of the time.
fn mean_drift_check(points: &[ConvergencePoint], family: usize) -> Option<bool> {
    use statrs::distribution::{ContinuousCDF, StudentsT};

    let last_count = points.last()?.n_histories as f64;
    if last_count <= 0.0 {
        return None;
    }
    let mut blocks = Vec::with_capacity(points.len().saturating_sub(1));
    for pair in points.windows(2) {
        let count = pair[1].n_histories.checked_sub(pair[0].n_histories)?;
        if count == 0 {
            return None;
        }
        let start = pair[0].n_histories as f64;
        let end = pair[1].n_histories as f64;
        let mean = (end * pair[1].mean - start * pair[0].mean) / count as f64;
        if !mean.is_finite() {
            return None;
        }
        blocks.push(((start + end) / (2.0 * last_count), mean, count as f64));
    }
    if blocks.len() < 3 {
        return None;
    }
    let total_weight: f64 = blocks.iter().map(|point| point.2).sum();
    let mean_x = blocks.iter().map(|point| point.2 * point.0).sum::<f64>() / total_weight;
    let mean_y = blocks.iter().map(|point| point.2 * point.1).sum::<f64>() / total_weight;
    let spread: f64 = blocks
        .iter()
        .map(|point| point.2 * (point.0 - mean_x).powi(2))
        .sum();
    if spread <= 0.0 || !spread.is_finite() {
        return None;
    }
    let slope = blocks
        .iter()
        .map(|point| point.2 * (point.0 - mean_x) * (point.1 - mean_y))
        .sum::<f64>()
        / spread;
    let residual: f64 = blocks
        .iter()
        .map(|point| point.2 * (point.1 - mean_y - slope * (point.0 - mean_x)).powi(2))
        .sum();
    let degrees = (blocks.len() - 2) as f64;
    let standard_error = (residual / (degrees * spread)).sqrt();
    if !slope.is_finite() || !standard_error.is_finite() {
        return None;
    }
    if standard_error == 0.0 {
        return Some(slope == 0.0);
    }
    // The quantile is the same for every bin judged at one checkpoint, and a
    // per-bin run judges millions, so it is cached per (degrees, family).
    thread_local! {
        static CRITICAL: std::cell::RefCell<rustc_hash::FxHashMap<(usize, usize), f64>> =
            Default::default();
    }
    let key = (blocks.len() - 2, family.max(1));
    let critical = CRITICAL.with(|c| {
        *c.borrow_mut().entry(key).or_insert_with(|| {
            StudentsT::new(0.0, 1.0, degrees)
                .expect("positive degrees of freedom")
                .inverse_cdf(1.0 - DRIFT_ALPHA / (2.0 * key.1 as f64))
        })
    });
    Some(slope.abs() <= critical * standard_error)
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
///   score → surface → cosine → cell → material → nuclide → parent_nuclide → energy → mesh_z → mesh_y → mesh_x
///
/// This must match the flat bin layout of `Tally::get_bin_index_7d` and
/// `Tally::get_surface_bin_index`.
pub(crate) fn compute_shape_and_dims(tally: &Tally) -> (Vec<usize>, Vec<String>) {
    let mut shape = Vec::with_capacity(8);
    let mut dims: Vec<String> = Vec::with_capacity(8);

    // Score dimension is always present.
    shape.push(tally.scores.len());
    dims.push("score".to_string());

    // Surface and cosine dimensions (surface tallies only) when >1 bin.
    let n_surface = tally.num_surface_bins();
    if n_surface > 1 {
        shape.push(n_surface);
        dims.push("surface".to_string());
    }
    let n_cosine = tally.num_cosine_bins();
    if n_cosine > 1 {
        shape.push(n_cosine);
        dims.push("cosine".to_string());
    }

    // Cell dimension only when the filter has >1 bins.
    let n_cell = tally.num_cell_bins();
    if n_cell > 1 {
        shape.push(n_cell);
        dims.push("cell".to_string());
    }

    // Material dimension only when the filter has >1 bins.
    let n_material = tally.num_material_bins();
    if n_material > 1 {
        shape.push(n_material);
        dims.push("material".to_string());
    }

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
    if let Some(parts) = &tally.mesh_parts {
        // One bin per (element, cell) part, in the parts' own order.
        shape.push(parts.len().max(1));
        dims.push("mesh_part".to_string());
        return (shape, dims);
    }
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
        if !has_regular_mesh
            && let Some(f) = tally.filters.iter().find_map(|f| match f {
                Filter::UnstructuredMesh(um) => Some(um),
                _ => None,
            })
        {
            shape.push(f.num_bins());
            dims.push("mesh".to_string());
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
        tally.reset_accumulation();
        let tally = Arc::new(tally);

        let result = tally.finalize();

        // Finalized fields should match the current getter values bit-exact.
        assert_eq!(result.mean, tally.get_mean());
        assert_eq!(result.standard_deviation, tally.get_std_dev());
        assert_eq!(result.relative_error, tally.get_rel_error());
        assert_eq!(result.total_count, tally.total_count());
        assert_eq!(result.n_histories, tally.get_n_histories());

        // FOM fields default to empty / zero until `with_fom` is called.
        assert!(result.figure_of_merit.is_empty());
        assert_eq!(result.aggregate_figure_of_merit, 0.0);

        // Config back-reference should point at the same Arc.
        assert!(Arc::ptr_eq(&result.tally, &tally));
    }

    /// The plain-text summary and the tally's `Display` report the total
    /// particle count, not batches.
    #[test]
    fn summary_and_display_show_particles_not_batches() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        let tally = Arc::new(tally);
        tally.install_finalized(crate::welford::WelfordTallyStats {
            mean: vec![1.0],
            m2: vec![0.5],
            n_histories: 5000,
            agg: crate::welford::AggMoments::ZERO,
            score_pdf: crate::welford::ScorePdf::default(),
            comoment: None,
            replicas: None,
        });

        let summary = tally.finalize().summary();
        assert!(summary.contains("particles=5000"), "{summary}");
        let display = tally.to_string();
        assert!(display.contains("Particles: 5000"), "{display}");
        for text in [&summary, &display] {
            assert!(!text.to_lowercase().contains("batch"), "{text}");
            assert!(!text.contains("histories"), "{text}");
        }
    }

    /// Multi-bin cell and material filters each add a dimension, in the
    /// same order as the flat bin layout, so `shape` covers every bin.
    #[test]
    fn shape_includes_cell_and_material_dims() {
        use crate::filter::Filter;
        use crate::{CellFilter, EnergyFilter, MaterialFilter};

        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.filters = vec![
            Filter::Cell(CellFilter::from_ids(vec![1, 2, 3])),
            Filter::Material(MaterialFilter {
                material_ids: vec![10, 20],
            }),
            Filter::Energy(EnergyFilter::new(vec![0.0, 1e6, 2e7])),
        ];
        tally.reset_accumulation();
        let tally = Arc::new(tally);
        let result = tally.finalize();

        assert_eq!(result.shape, vec![1, 3, 2, 2]);
        assert_eq!(
            result.dim_labels,
            vec!["score", "cell", "material", "energy"]
        );
        assert_eq!(result.shape.iter().product::<usize>(), result.mean.len());
    }

    /// `with_fom(elapsed)` populates `figure_of_merit` per bin to
    /// `1 / (rel_err² × elapsed)`, and, for a result without the per-history
    /// total, `aggregate_figure_of_merit` from the summed per-bin variances.
    #[test]
    fn with_fom_computes_standard_definition() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.reset_accumulation();
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

    /// With the per-history total recorded, the aggregate FOM comes from its
    /// relative error, not from per-bin variances summed as if the bins were
    /// independent: two bins that one history scores together have a total
    /// whose variance is larger than the sum.
    #[test]
    fn with_fom_aggregate_uses_the_per_history_total() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.reset_accumulation();
        let mut result = Arc::new(tally).finalize();
        result.mean = vec![1.0, 4.0];
        result.relative_error = vec![0.1, 0.05];
        // 100 histories whose total has mean 5 and variance of the mean 0.09
        // (std error 0.3, above the 0.2236 of independent bins).
        result.agg = crate::welford::AggMoments {
            n: 100,
            mean: 5.0,
            m2: 0.09 * 99.0 * 100.0,
            m3: 0.0,
            m4: 0.0,
        };
        let elapsed = 4.0;
        let result = result.with_fom(elapsed);
        let rel = result.aggregate_relative_error();
        assert!((rel - 0.06).abs() < 1e-12);
        let expected = 1.0 / (rel * rel * elapsed);
        assert!(
            (result.aggregate_figure_of_merit - expected).abs() < 1e-9 * expected,
            "aggregate FOM {} vs {expected}",
            result.aggregate_figure_of_merit
        );
    }

    /// `with_fom(0.0)` (or negative) should return all-zero FOMs
    /// rather than producing `inf` / `NaN`.
    #[test]
    fn with_fom_zero_elapsed_returns_zero_foms() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.reset_accumulation();
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
        tally.reset_accumulation();
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
        tally.reset_accumulation();
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
        tally.reset_accumulation();
        let tally = Arc::new(tally);
        let mut result = tally.finalize();
        result.standard_deviation = vec![0.1, 0.2, 0.5];
        let var = result.variance();
        assert_eq!(var.len(), 3);
        assert!((var[0] - 0.01).abs() < 1e-15);
        assert!((var[1] - 0.04).abs() < 1e-15);
        assert!((var[2] - 0.25).abs() < 1e-15);
    }

    fn strength(value: f64) -> Option<SourceStrength> {
        Some(SourceStrength::new(value, crate::units::StrengthPer::Second).unwrap())
    }

    fn scored(score: Score) -> Tally {
        let mut t = Tally::new();
        t.scores = vec![score];
        t
    }

    fn with_mean(mut tally: Tally, mean: Vec<f64>) -> TallyResult {
        tally.reset_accumulation();
        let mut r = Arc::new(tally).finalize();
        r.standard_deviation = mean.iter().map(|m| m * 0.1).collect();
        r.relative_error = vec![0.1; mean.len()];
        r.mean = mean;
        r
    }

    /// A strength replaces the one before rather than multiplying it, and
    /// leaves the relative error alone.
    #[test]
    fn source_strength_replaces_and_scales_mean_and_std() {
        let r = with_mean(scored(Score::Flux(FluxScore)), vec![2.0]);
        let a = r.with_source_strength(strength(10.0)).unwrap();
        assert_eq!(a.scaled(&a.mean), vec![20.0]);
        assert_eq!(a.scaled(&a.standard_deviation), vec![2.0]);
        assert!((a.scaled_squared(&a.variance())[0] - 4.0).abs() < 1e-12);
        assert_eq!(a.relative_error, r.relative_error);
        let b = a.with_source_strength(strength(3.0)).unwrap();
        assert_eq!(b.scaled(&b.mean), vec![6.0]);
        let c = b.with_source_strength(None).unwrap();
        assert_eq!(c.scaled(&c.mean), vec![2.0]);
        assert_eq!(c.units(), vec!["cm/source-particle"]);
    }

    /// A per-volume unit divides each cylindrical-mesh bin by its own volume.
    #[test]
    fn per_volume_units_divide_by_each_mesh_element() {
        use crate::filter::Filter;
        use crate::{CylindricalMesh, MeshFilter};
        let mesh = CylindricalMesh::new(
            [0.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, std::f64::consts::TAU],
            vec![0.0, 1.0],
        );
        let mut t = scored(Score::Heating(crate::score::HeatingScore));
        t.filters = vec![Filter::Mesh(MeshFilter::new_cylindrical(mesh))];
        let r = with_mean(t, vec![1.0, 1.0]);
        let volumes = r.bin_volumes().unwrap();
        assert!((volumes[0] - std::f64::consts::PI).abs() < 1e-12);
        assert!((volumes[1] - 3.0 * std::f64::consts::PI).abs() < 1e-12);
        let w = r
            .with_source_strength(strength(1.0 / 1.602_176_634e-19))
            .unwrap()
            .to_units(&["W/cm3".to_string()])
            .unwrap();
        let got = w.scaled(&w.mean);
        assert!((got[0] - 1.0 / volumes[0]).abs() < 1e-12);
        assert!((got[1] - 1.0 / volumes[1]).abs() < 1e-12);
        assert_eq!(w.uniform_scale(), None);
        assert_eq!(w.units(), vec!["W/cm3"]);
    }

    /// Cell volumes come from the geometry; a cell without one is named in
    /// the error.
    #[test]
    fn per_volume_units_use_cell_volumes() {
        use crate::filter::Filter;
        let mut t = scored(Score::Flux(FluxScore));
        t.filters = vec![Filter::Cell(crate::CellFilter::from_ids(vec![4, 9]))];
        let mut r = with_mean(t, vec![10.0, 10.0]);
        r.normalisation.cell_volumes = vec![Some(2.0), None];
        let err = r.to_units(&["1/cm2".to_string()]).unwrap_err();
        assert!(err.contains("cell 9 has no volume"), "{err}");
        r.normalisation.cell_volumes = vec![Some(2.0), Some(5.0)];
        let flux = r.to_units(&["1/cm2".to_string()]).unwrap();
        assert_eq!(flux.scaled(&flux.mean), vec![5.0, 2.0]);
        assert_eq!(flux.units(), vec!["1/cm2/source-particle"]);
        let err = r.to_units(&["1/(cm2 s)".to_string()]).unwrap_err();
        assert!(err.contains("with_source_strength"), "{err}");
    }

    /// A per-mass unit divides by each cell's volume times its density; a void
    /// cell is refused by name.
    #[test]
    fn per_mass_units_use_cell_volumes_and_densities() {
        use crate::filter::Filter;
        let mut t = scored(Score::Heating(crate::score::HeatingScore));
        t.filters = vec![Filter::Cell(crate::CellFilter::from_ids(vec![4, 9]))];
        let mut r = with_mean(t, vec![1.0, 1.0]);
        r.normalisation.cell_volumes = vec![Some(2.0), Some(5.0)];
        r.normalisation.cell_densities = vec![Some(8.0), None];
        let err = r.to_units(&["eV/g".to_string()]).unwrap_err();
        assert!(err.contains("cell 9 has no material"), "{err}");
        r.normalisation.cell_densities = vec![Some(8.0), Some(1.0)];
        let per_g = r.to_units(&["eV/g".to_string()]).unwrap();
        assert_eq!(per_g.scaled(&per_g.mean), vec![1.0 / 16.0, 1.0 / 5.0]);
        let gy = r
            .with_source_strength(strength(1.0))
            .unwrap()
            .to_units(&["Gy/s".to_string()])
            .unwrap();
        let got = gy.scaled(&gy.mean);
        assert!((got[0] - 1.602_176_634e-16 / 16.0).abs() < 1e-28, "{got:?}");
        assert_eq!(gy.units(), vec!["Gy/s"]);
    }

    /// One unit per score, each converted on its own; a total over scores of
    /// different conversions has no unit.
    #[test]
    fn one_unit_per_score() {
        let mut t = Tally::new();
        t.scores = vec![
            Score::Heating(crate::score::HeatingScore),
            Score::Flux(FluxScore),
        ];
        let r = with_mean(t, vec![1e6, 3.0]);
        let c = r.to_units(&["MeV".to_string(), "m".to_string()]).unwrap();
        let got = c.scaled(&c.mean);
        assert!((got[0] - 1.0).abs() < 1e-12 && (got[1] - 0.03).abs() < 1e-12);
        assert!(r
            .to_units(&["MeV".to_string()])
            .unwrap_err()
            .contains("cannot convert"));
        assert!(r.to_units(&[]).unwrap_err().contains("2 scores"));
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
        tally.reset_accumulation();
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
    /// evaluated" (None) and the verdict is incomplete, not a vacuous pass.
    #[test]
    fn statistical_checks_not_evaluated_without_samples() {
        let r = flux_result();
        let c = r.statistical_checks();
        assert_eq!(c.relative_error_ok, None);
        assert_eq!(c.variance_of_variance_ok, None);
        assert_eq!(c.tail_slope_ok, None);
        assert_eq!(c.n_evaluated(), 0);
        assert!(!c.passed());
        assert!(c.summary().contains("INCOMPLETE"));
    }

    #[test]
    fn too_few_tail_samples_do_not_pass_otherwise_healthy_checks() {
        let mut result = flux_result();
        result.agg = crate::welford::AggMoments {
            n: 100_000,
            mean: 2.0,
            m2: 1.0e-3,
            m3: 0.0,
            m4: 1.0e-9,
        };
        result.score_pdf = crate::welford::ScorePdf::new();
        for _ in 0..4 {
            result.score_pdf.record(2.0);
        }
        let checks = result.statistical_checks();
        assert_eq!(checks.relative_error_ok, Some(true));
        assert_eq!(checks.variance_of_variance_ok, Some(true));
        assert_eq!(checks.tail_slope_ok, None);
        assert!(!checks.passed());
        assert!(checks.summary().contains("INCOMPLETE"));
    }

    #[test]
    fn a_noisy_bin_is_not_hidden_by_the_tally_total() {
        let mut result = flux_result();
        result.n_histories = 100_000;
        result.mean = vec![100.0, 1.0, 0.0];
        result.standard_deviation = vec![0.01, 0.5, 0.0];
        assert_eq!(
            result.statistical_checks().per_bin_relative_error_ok,
            Some(false)
        );
        result.standard_deviation[1] = 0.01;
        assert_eq!(
            result.statistical_checks().per_bin_relative_error_ok,
            Some(true)
        );
    }

    #[test]
    fn per_bin_relative_error_does_not_pass_zero_mean_variance() {
        let mut result = flux_result();
        result.n_histories = 1_000;
        result.mean = vec![0.0];
        result.standard_deviation = vec![1.0];
        assert_eq!(
            result.statistical_checks().per_bin_relative_error_ok,
            Some(false)
        );
        result.standard_deviation[0] = 0.0;
        assert_eq!(result.statistical_checks().per_bin_relative_error_ok, None);
    }

    #[test]
    fn mean_trend_detects_drift_hidden_by_noisy_snapshots() {
        let make_history = |drift: f64| {
            let mut total = 0.0;
            (1..=1_000)
                .map(|index| {
                    let noise = if index % 2 == 0 { 0.1 } else { -0.1 };
                    total += 1_000.0 * (1.0 + drift * index as f64 + noise);
                    ConvergencePoint {
                        n_histories: index * 1_000,
                        mean: total / (index * 1_000) as f64,
                        relative_error: 0.01,
                        variance_of_variance: 0.01,
                        figure_of_merit: 1.0,
                        tail_slope: 8.0,
                        elapsed_secs: index as f64,
                    }
                })
                .collect::<Vec<_>>()
        };
        let drifting = make_history(1.0e-4);
        let half = &drifting[drifting.len() / 2..];
        assert!(half.windows(2).any(|pair| pair[1].mean < pair[0].mean));
        assert!(half.windows(2).any(|pair| pair[1].mean > pair[0].mean));
        assert_eq!(mean_drift_check(half, 1), Some(false));
        let stationary = make_history(0.0);
        assert_eq!(
            mean_drift_check(&stationary[stationary.len() / 2..], 1),
            Some(true)
        );
    }

    /// Bonferroni: a drift that one test flags at 5% is let through when it
    /// is one of many simultaneous tests, and a strong drift is still caught.
    #[test]
    fn mean_drift_family_widens_the_threshold() {
        let make = |drift: f64| {
            let mut total = 0.0;
            (1..=40u64)
                .map(|i| {
                    let noise = [0.03, -0.02, 0.01, -0.03, 0.02][i as usize % 5];
                    total += 1_000.0 * (1.0 + drift * i as f64 + noise);
                    ConvergencePoint {
                        n_histories: i * 1_000,
                        mean: total / (i * 1_000) as f64,
                        relative_error: 0.01,
                        variance_of_variance: 0.01,
                        figure_of_merit: 1.0,
                        tail_slope: 8.0,
                        elapsed_secs: i as f64,
                    }
                })
                .collect::<Vec<_>>()
        };
        // Find a drift on the edge: flagged alone, passed among 1000 tests.
        let edge = (1..200).map(|k| k as f64 * 2.0e-5).find(|&d| {
            let h = make(d);
            let half = &h[h.len() / 2..];
            mean_drift_check(half, 1) == Some(false) && mean_drift_check(half, 1000) == Some(true)
        });
        assert!(edge.is_some(), "no drift sits between the two thresholds");
        let strong = make(1.0e-2);
        assert_eq!(
            mean_drift_check(&strong[strong.len() / 2..], 1000),
            Some(false)
        );
    }

    #[test]
    fn mean_trend_does_not_pass_nonfinite_block_statistics() {
        let mut points = history(|_| 0.01, |_| 0.01, |_| 1.0, 1.0);
        for (index, point) in points.iter_mut().enumerate() {
            point.mean = if index % 2 == 0 { 1.0e200 } else { -1.0e200 };
        }
        assert_eq!(mean_drift_check(&points, 1), None);
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
                    elapsed_secs: f,
                }
            })
            .collect();
        let c = r.statistical_checks();
        assert_eq!(c.mean_stable, Some(true));
        assert_eq!(c.relative_error_decreasing, Some(true));
        assert_eq!(c.variance_of_variance_decreasing, Some(true));
        assert_eq!(c.figure_of_merit_stable, Some(true));
    }

    /// A ten-snapshot history whose relative error, variance of the variance
    /// and figure of merit follow `re(N)`, `vov(N)`, `fom(N)`, with
    /// `secs_per_snapshot` of wall time between snapshots.
    fn history(
        re: impl Fn(f64) -> f64,
        vov: impl Fn(f64) -> f64,
        fom: impl Fn(f64) -> f64,
        secs_per_snapshot: f64,
    ) -> Vec<ConvergencePoint> {
        (1..=10)
            .map(|i| {
                let n = i as f64 * 1000.0;
                ConvergencePoint {
                    n_histories: n as u64,
                    mean: 2.0,
                    relative_error: re(n),
                    variance_of_variance: vov(n),
                    figure_of_merit: fom(n),
                    tail_slope: 8.0,
                    elapsed_secs: i as f64 * secs_per_snapshot,
                }
            })
            .collect()
    }

    /// Ideal `1/sqrt(N)` relative error, `1/N` variance of the variance and a
    /// constant figure of merit pass all three rate checks.
    #[test]
    fn rate_checks_pass_for_ideal_scaling() {
        let mut r = flux_result();
        r.convergence_history = history(|n| 3.0 / n.sqrt(), |n| 10.0 / n, |_| 100.0, 1.0);
        let c = r.statistical_checks();
        assert_eq!(c.relative_error_rate_ok, Some(true));
        assert_eq!(c.variance_of_variance_rate_ok, Some(true));
        assert_eq!(c.figure_of_merit_no_trend, Some(true));
    }

    #[test]
    fn both_fom_checks_ignore_short_timer_spans() {
        let mut result = flux_result();
        result.convergence_history = history(
            |count| 3.0 / count.sqrt(),
            |count| 10.0 / count,
            |count| count.powi(4),
            0.001,
        );
        let checks = result.statistical_checks();
        assert_eq!(checks.figure_of_merit_stable, None);
        assert_eq!(checks.figure_of_merit_no_trend, None);
        for point in &mut result.convergence_history {
            point.elapsed_secs *= 1_000.0;
        }
        let checks = result.statistical_checks();
        assert_eq!(checks.figure_of_merit_stable, Some(false));
        assert_eq!(checks.figure_of_merit_no_trend, Some(false));
    }

    /// A relative error and variance of the variance that stall (as when a
    /// rare large score lands late) fail their rate checks even though they
    /// never increase.
    #[test]
    fn rate_checks_fail_when_relative_error_and_vov_stall() {
        let mut r = flux_result();
        r.convergence_history =
            history(|n| 0.5 / n.powf(0.1), |n| 0.5 / n.powf(0.2), |_| 100.0, 1.0);
        let c = r.statistical_checks();
        assert_eq!(c.relative_error_decreasing, Some(true));
        assert_eq!(c.relative_error_rate_ok, Some(false));
        assert_eq!(c.variance_of_variance_decreasing, Some(true));
        assert_eq!(c.variance_of_variance_rate_ok, Some(false));
        assert!(!c.passed());
    }

    /// A figure of merit falling as `1/N` fails the trend check.
    #[test]
    fn rate_checks_fail_when_figure_of_merit_drifts() {
        let mut r = flux_result();
        r.convergence_history = history(|n| 3.0 / n.sqrt(), |n| 10.0 / n, |n| 1e6 / n, 1.0);
        let c = r.statistical_checks();
        assert_eq!(c.figure_of_merit_no_trend, Some(false));
    }

    /// The figure-of-merit trend is not judged when the last half of the run
    /// spans too little wall time, and no rate check is judged with too few
    /// snapshots.
    #[test]
    fn rate_checks_not_evaluated_on_short_runs() {
        let mut r = flux_result();
        r.convergence_history = history(|n| 3.0 / n.sqrt(), |n| 10.0 / n, |n| 1e6 / n, 1e-3);
        let c = r.statistical_checks();
        assert_eq!(c.figure_of_merit_no_trend, None);
        assert_eq!(c.relative_error_rate_ok, Some(true));

        r.convergence_history.truncate(6);
        let c = r.statistical_checks();
        assert_eq!(c.relative_error_rate_ok, None);
        assert_eq!(c.variance_of_variance_rate_ok, None);
        assert_eq!(c.figure_of_merit_no_trend, None);
    }
}
