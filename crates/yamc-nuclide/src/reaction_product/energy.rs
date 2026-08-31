//! Outgoing-energy distributions (level inelastic, tabulated, Maxwell, Watt, ...).

use super::*;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Energy distribution types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EnergyDistribution {
    LevelInelastic {
        /// Threshold energy for this level
        threshold: f64,
        /// Mass ratio = (A/(A+1))^2 for two-body kinematics
        mass_ratio: f64,
    },
    Tabulated {
        energy: Vec<f64>,
        energy_out: Vec<Vec<f64>>,
    },
    #[serde(rename = "ContinuousTabular")]
    ContinuousTabular {
        energy: Vec<f64>,
        energy_out: Vec<TabulatedProbability>,
        /// If true, use histogram interpolation for incident energy (no stochastic bin selection)
        /// This matches the histogram_interp flag in ContinuousTabular::sample()
        #[serde(default)]
        histogram_interp: bool,
    },
    /// Maxwell fission spectrum
    /// p(E) ~ sqrt(E) * exp(-E/theta)
    #[serde(rename = "Maxwell")]
    Maxwell {
        /// Nuclear temperature parameter theta(E) as function of incident energy
        theta: Tabulated1D,
        /// Restriction energy u - samples must satisfy E_out <= E_in - u
        u: f64,
    },
    /// Watt fission spectrum
    /// p(E) ~ exp(-E/a) * sinh(sqrt(b*E))
    #[serde(rename = "Watt")]
    Watt {
        /// Watt parameter a(E) as function of incident energy
        a: Tabulated1D,
        /// Watt parameter b(E) as function of incident energy
        b: Tabulated1D,
        /// Restriction energy u - samples must satisfy E_out <= E_in - u
        u: f64,
    },
    /// Evaporation spectrum
    /// p(E) ~ exp(-(E-u)/theta) with 0 < E_out < E_in - u
    #[serde(rename = "Evaporation")]
    Evaporation {
        /// Nuclear temperature parameter theta(E) as function of incident energy
        theta: Tabulated1D,
        /// Restriction energy u - samples must satisfy E_out <= E_in - u
        u: f64,
    },
    /// Discrete photon energy (gamma lines from nuclear de-excitation)
    /// Discrete photon energy for gamma lines from nuclear de-excitation
    #[serde(rename = "DiscretePhoton")]
    DiscretePhoton {
        /// Primary flag: if 2, energy depends on incident energy; otherwise fixed
        primary_flag: i32,
        /// Photon energy in eV (for primary_flag != 2, this is the outgoing energy)
        energy: f64,
        /// Atomic weight ratio of the target nuclide
        atomic_weight_ratio: f64,
    },
}

impl EnergyDistribution {
    pub fn sample<R: Rng>(&self, incoming_energy: f64, rng: &mut R) -> f64 {
        #[cfg(feature = "debug_sampling")]
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static E_DIST_SAMPLE_COUNT: AtomicU64 = AtomicU64::new(0);
            let count = E_DIST_SAMPLE_COUNT.fetch_add(1, Ordering::Relaxed);
            if count < 100 || count.is_multiple_of(10000) {
                let dist_type = match self {
                    EnergyDistribution::LevelInelastic { .. } => "LevelInelastic",
                    EnergyDistribution::Tabulated { .. } => "Tabulated",
                    EnergyDistribution::ContinuousTabular { .. } => "ContinuousTabular",
                    EnergyDistribution::Maxwell { .. } => "Maxwell",
                    EnergyDistribution::Watt { .. } => "Watt",
                    EnergyDistribution::Evaporation { .. } => "Evaporation",
                    EnergyDistribution::DiscretePhoton { .. } => "DiscretePhoton",
                };
                eprintln!(
                    "[DEBUG_SAMPLING] EnergyDist: type={}, E_in={:.4e} eV",
                    dist_type, incoming_energy
                );
            }
        }
        match self {
            EnergyDistribution::LevelInelastic {
                threshold,
                mass_ratio,
            } => {
                // E_out = mass_ratio * (E_in - threshold)
                // mass_ratio = (A/(A+1))^2 accounts for two-body kinematics
                if incoming_energy > *threshold {
                    mass_ratio * (incoming_energy - threshold)
                } else {
                    // Below threshold - this shouldn't happen in valid physics
                    0.0
                }
            }
            EnergyDistribution::Tabulated { energy, energy_out } => {
                if energy.is_empty() || energy_out.is_empty() {
                    return incoming_energy;
                }
                let i = Self::find_energy_index(incoming_energy, energy);
                if i >= energy_out.len() {
                    return incoming_energy;
                }
                let xs = &energy_out[i];
                if xs.is_empty() {
                    return incoming_energy;
                }
                let idx = rng.random_range(0..xs.len());
                xs[idx]
            }
            EnergyDistribution::ContinuousTabular {
                energy,
                energy_out,
                histogram_interp,
            } => {
                if energy.is_empty() || energy_out.is_empty() {
                    return incoming_energy;
                }

                let n_energy = energy.len();

                // Find energy bin and interpolation factor
                let (i, r) = if incoming_energy <= energy[0] {
                    (0, 0.0)
                } else if incoming_energy >= energy[n_energy - 1] {
                    (n_energy.saturating_sub(2), 1.0)
                } else {
                    let idx = Self::find_energy_index(incoming_energy, energy);
                    let interp = if idx + 1 < n_energy && energy[idx + 1] > energy[idx] {
                        (incoming_energy - energy[idx]) / (energy[idx + 1] - energy[idx])
                    } else {
                        0.0
                    };
                    (idx, interp)
                };

                // Check bounds
                if i >= energy_out.len() || i + 1 >= energy_out.len() {
                    if i < energy_out.len() {
                        return energy_out[i].sample(rng);
                    }
                    return incoming_energy;
                }

                // If histogram interpolation, always use bin i (no stochastic selection)
                // Otherwise, stochastically choose between i and i+1
                let l = if *histogram_interp {
                    i // histogram: always use lower bin
                } else if r > rng.random::<f64>() {
                    i + 1
                } else {
                    i
                };

                // Sample from chosen distribution, getting discrete line info
                // Sample from chosen distribution, getting discrete line info
                let (e_out_raw, is_discrete) = energy_out[l].sample_with_discrete_info(rng);

                // Skip E_out interpolation for histogram interpolation or discrete lines
                if *histogram_interp || is_discrete {
                    #[cfg(feature = "debug_sampling")]
                    {
                        use std::sync::atomic::{AtomicU64, Ordering};
                        static E_OUT_HIST_COUNT: AtomicU64 = AtomicU64::new(0);
                        let count = E_OUT_HIST_COUNT.fetch_add(1, Ordering::Relaxed);
                        if count < 100 || count.is_multiple_of(10000) {
                            let reason = if is_discrete { "discrete" } else { "hist" };
                            eprintln!(
                                "[DEBUG_SAMPLING] E_out({}): E_in={:.4e} eV, bin={}, E_out={:.4e} eV",
                                reason, incoming_energy, l, e_out_raw
                            );
                        }
                    }
                    return e_out_raw;
                }

                // Get E_out bounds for both bins using FIRST CONTINUOUS energy (not first discrete)
                // Use first continuous energy (not first discrete) as E_i_1
                let e_i_1 = energy_out[i].get_first_continuous_energy();
                let e_i_k = energy_out[i].get_last_energy();
                let e_i1_1 = energy_out[i + 1].get_first_continuous_energy();
                let e_i1_k = energy_out[i + 1].get_last_energy();

                // Check for valid bounds
                if e_i_k <= e_i_1 || e_i1_k <= e_i1_1 {
                    #[cfg(feature = "debug_sampling")]
                    {
                        use std::sync::atomic::{AtomicU64, Ordering};
                        static E_OUT_EMPTY_COUNT: AtomicU64 = AtomicU64::new(0);
                        let count = E_OUT_EMPTY_COUNT.fetch_add(1, Ordering::Relaxed);
                        if count < 100 || count.is_multiple_of(10000) {
                            eprintln!(
                                "[DEBUG_SAMPLING] E_out(invalid bounds): E_in={:.4e} eV, bin={}, E_out={:.4e} eV",
                                incoming_energy, l, e_out_raw
                            );
                        }
                    }
                    return e_out_raw;
                }

                // Interpolated E_out bounds
                let e_1 = e_i_1 + r * (e_i1_1 - e_i_1);
                let e_k = e_i_k + r * (e_i1_k - e_i_k);

                // Interpolate outgoing energy between incident energy bins
                let (e_l_1, e_l_k) = if l == i {
                    (e_i_1, e_i_k)
                } else {
                    (e_i1_1, e_i1_k)
                };

                let e_out_final = if e_l_k > e_l_1 && e_k > e_1 {
                    e_1 + (e_out_raw - e_l_1) * (e_k - e_1) / (e_l_k - e_l_1)
                } else {
                    e_out_raw
                };

                #[cfg(feature = "debug_sampling")]
                {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static E_OUT_SAMPLE_COUNT: AtomicU64 = AtomicU64::new(0);
                    let count = E_OUT_SAMPLE_COUNT.fetch_add(1, Ordering::Relaxed);
                    if count < 100 || count.is_multiple_of(10000) {
                        eprintln!(
                            "[DEBUG_SAMPLING] E_out: E_in={:.4e} eV, bin={}, E_out_raw={:.4e}, E_out_final={:.4e} eV",
                            incoming_energy, l, e_out_raw, e_out_final
                        );
                    }
                }

                e_out_final
            }
            EnergyDistribution::Maxwell { theta, u } => {
                // Get temperature at incident energy
                let theta_val = theta.evaluate(incoming_energy);

                // Rejection sampling with restriction energy check
                loop {
                    let e_out = crate::sampling::sample_maxwell_spectrum(theta_val, rng);

                    // Accept if E_out <= E_in - u (restriction energy check)
                    if e_out <= incoming_energy - *u {
                        return e_out;
                    }
                }
            }
            EnergyDistribution::Watt { a, b, u } => {
                // Get Watt parameters at incident energy
                let a_val = a.evaluate(incoming_energy);
                let b_val = b.evaluate(incoming_energy);

                // Rejection sampling with restriction energy check
                loop {
                    let e_out = crate::sampling::sample_watt_spectrum_params(a_val, b_val, rng);

                    // Accept if E_out <= E_in - u (restriction energy check)
                    if e_out <= incoming_energy - *u {
                        return e_out;
                    }
                }
            }
            EnergyDistribution::Evaporation { theta, u } => {
                // Get temperature corresponding to incoming energy
                let theta_val = theta.evaluate(incoming_energy);

                let y = (incoming_energy - *u) / theta_val;
                let v = 1.0 - (-y).exp();

                // Sample outgoing energy based on evaporation spectrum probability density function
                loop {
                    let x =
                        -((1.0 - v * rng.random::<f64>()) * (1.0 - v * rng.random::<f64>())).ln();
                    if x <= y {
                        return x * theta_val;
                    }
                }
            }
            EnergyDistribution::DiscretePhoton {
                primary_flag,
                energy,
                atomic_weight_ratio,
            } => {
                if *primary_flag == 2 {
                    energy + atomic_weight_ratio / (atomic_weight_ratio + 1.0) * incoming_energy
                } else {
                    *energy
                }
            }
        }
    }

    fn find_energy_index(target_energy: f64, energy_grid: &[f64]) -> usize {
        if target_energy <= energy_grid[0] {
            return 0;
        }
        if target_energy >= energy_grid[energy_grid.len() - 1] {
            return energy_grid.len() - 1;
        }
        energy_grid
            .binary_search_by(|&val| val.partial_cmp(&target_energy).unwrap())
            .unwrap_or_else(|i| i.saturating_sub(1))
    }

    /// Get the name of this energy distribution type
    pub fn distribution_name(&self) -> &'static str {
        match self {
            EnergyDistribution::LevelInelastic { .. } => "LevelInelastic",
            EnergyDistribution::Tabulated { .. } => "Tabulated",
            EnergyDistribution::ContinuousTabular { .. } => "ContinuousTabular",
            EnergyDistribution::Maxwell { .. } => "Maxwell",
            EnergyDistribution::Watt { .. } => "Watt",
            EnergyDistribution::Evaporation { .. } => "Evaporation",
            EnergyDistribution::DiscretePhoton { .. } => "DiscretePhoton",
        }
    }

    /// Flatten a fission / secondary outgoing-energy (chi) distribution into the
    /// sampler-ready [`FissionChiFlat`] layout consumed by the shared GPU/CPU
    /// flat samplers (issue #111). Built once per nuclide and cached
    /// ([`FissionChiFlatCache`]); the transport crate feeds these slices to
    /// `yamc_physics::gpu::flat::{watt, maxwell, evaporation, tabulated_continuous_eout}`
    /// driven by the per-particle PCG stream, so the CPU fission chi and the GPU
    /// kernel share one sampling implementation. Covers the fission-spectrum
    /// kinds; any other kind (LevelInelastic, equiprobable Tabulated,
    /// DiscretePhoton) returns [`FissionChiFlat::None`] and the caller keeps the
    /// production path.
    pub fn to_fission_chi_flat(&self) -> FissionChiFlat {
        match self {
            EnergyDistribution::Watt { a, b, u } => {
                let Tabulated1D::Tabulated1D { x: ax, y: ay, .. } = a;
                if ax.is_empty() {
                    return FissionChiFlat::None;
                }
                // `a` and `b` may carry different incident-energy grids; align
                // `b` onto `a`'s grid via `Tabulated1D::evaluate` so the flat
                // Watt sampler interpolates both on one grid.
                let b_grid: Vec<f64> = ax.iter().map(|&e| b.evaluate(e)).collect();
                FissionChiFlat::Watt {
                    energy_grid: ax.clone(),
                    a: ay.clone(),
                    b: b_grid,
                    u: *u,
                }
            }
            EnergyDistribution::Maxwell { theta, u } => {
                let Tabulated1D::Tabulated1D { x, y, .. } = theta;
                if x.is_empty() {
                    return FissionChiFlat::None;
                }
                FissionChiFlat::Maxwell {
                    energy_grid: x.clone(),
                    theta: y.clone(),
                    u: *u,
                }
            }
            EnergyDistribution::Evaporation { theta, u } => {
                let Tabulated1D::Tabulated1D { x, y, .. } = theta;
                if x.is_empty() {
                    return FissionChiFlat::None;
                }
                FissionChiFlat::Evaporation {
                    energy_grid: x.clone(),
                    theta: y.clone(),
                    u: *u,
                }
            }
            EnergyDistribution::ContinuousTabular {
                energy,
                energy_out,
                histogram_interp,
            } => continuous_to_flat(energy, energy_out, *histogram_interp),
            _ => FissionChiFlat::None,
        }
    }
}

/// Flat, sampler-ready form of a fission / secondary chi distribution
/// (issue #111). See [`EnergyDistribution::to_fission_chi_flat`].
#[derive(Debug, Clone, Default, PartialEq)]
pub enum FissionChiFlat {
    /// No usable chi data; the caller keeps the incident energy / falls back.
    #[default]
    None,
    /// Watt: `p(E) ~ exp(-E/a)·sinh(sqrt(bE))`. `a`/`b` share `energy_grid`.
    Watt {
        energy_grid: Vec<f64>,
        a: Vec<f64>,
        b: Vec<f64>,
        u: f64,
    },
    /// Maxwell: `p(E) ~ sqrt(E)·exp(-E/theta)`.
    Maxwell {
        energy_grid: Vec<f64>,
        theta: Vec<f64>,
        u: f64,
    },
    /// Evaporation: `p(E) ~ exp(-E/theta)`.
    Evaporation {
        energy_grid: Vec<f64>,
        theta: Vec<f64>,
        u: f64,
    },
    /// Tabulated continuous chi (`x`/`p`/`c` per incident energy), flat layout
    /// matching `sample_tabulated_continuous_eout` (stride `max_x`). CDF is
    /// renormalized to end at 1.0 with the PDF scaled by the same factor.
    Continuous {
        energy_grid: Vec<f64>,
        n_x: Vec<u32>,
        interp: Vec<u32>,
        n_discrete: Vec<u32>,
        x: Vec<f64>,
        p: Vec<f64>,
        c: Vec<f64>,
        max_x: usize,
        histogram_outer: bool,
    },
}

/// Per-bracket interpolation flag for [`FissionChiFlat::Continuous`], matching
/// `yamc_physics::gpu::flat::tabulated_continuous_eout::{INTERP_HISTOGRAM, INTERP_LINLIN}`.
const CHI_INTERP_HISTOGRAM: u32 = 0;
const CHI_INTERP_LINLIN: u32 = 1;

fn continuous_to_flat(
    energy: &[f64],
    energy_out: &[TabulatedProbability],
    histogram_interp: bool,
) -> FissionChiFlat {
    let n_e = energy.len();
    if n_e == 0 || energy_out.len() != n_e {
        return FissionChiFlat::None;
    }
    let row_len = |t: &TabulatedProbability| {
        let TabulatedProbability::Tabulated { x, .. } = t;
        x.len()
    };
    let max_x = energy_out.iter().map(row_len).max().unwrap_or(0);
    if max_x == 0 {
        return FissionChiFlat::None;
    }
    let mut n_x = vec![0u32; n_e];
    let mut interp = vec![CHI_INTERP_HISTOGRAM; n_e];
    let mut n_discrete = vec![0u32; n_e];
    let mut x_t = vec![0.0; n_e * max_x];
    let mut p_t = vec![0.0; n_e * max_x];
    let mut c_t = vec![0.0; n_e * max_x];
    for (i, tab) in energy_out.iter().enumerate() {
        let TabulatedProbability::Tabulated {
            x,
            p,
            c,
            interp: tab_interp,
            n_discrete: nd,
        } = tab;
        let m = x.len();
        if m == 0 {
            continue;
        }
        n_x[i] = m as u32;
        n_discrete[i] = *nd as u32;
        interp[i] = match tab_interp {
            TabulatedInterp::Histogram => CHI_INTERP_HISTOGRAM,
            TabulatedInterp::LinLin => CHI_INTERP_LINLIN,
        };
        // Renormalize the CDF to end at 1.0; scale the PDF by the same factor
        // (mirrors the GPU host extraction + `to_elastic_flat`).
        let cdf_max = c.last().copied().unwrap_or(0.0);
        let pdf_scale = if cdf_max > 0.0 { 1.0 / cdf_max } else { 1.0 };
        let off = i * max_x;
        for j in 0..m {
            x_t[off + j] = x[j];
            c_t[off + j] = if cdf_max > 0.0 {
                c[j] / cdf_max
            } else {
                j as f64 / (m - 1).max(1) as f64
            };
            p_t[off + j] = p.get(j).copied().unwrap_or(0.0) * pdf_scale;
        }
    }
    FissionChiFlat::Continuous {
        energy_grid: energy.to_vec(),
        n_x,
        interp,
        n_discrete,
        x: x_t,
        p: p_t,
        c: c_t,
        max_x,
        histogram_outer: histogram_interp,
    }
}

/// The fission MTs that can carry a prompt spectrum: total fission, first
/// through third chance, and fourth chance. One cache slot each.
///
/// This is the single definition of the set. [`crate::nuclide::is_fission_mt`]
/// tests membership of it, so a channel cannot be added to one and forgotten in
/// the other.
pub(crate) const FISSION_CHI_MTS: [i32; 5] = [18, 19, 20, 21, 38];

/// Per-nuclide cache of the flattened fission chi (issue #111), one slot per
/// fission MT, each built lazily on that channel's first fission and shared
/// read-only across transport threads. Resets on `clone()` and skipped by serde,
/// mirroring [`crate::reaction_product::ElasticFlatCache`].
///
/// Keyed by MT rather than held in one slot because an evaluation with partial
/// fission channels carries a DIFFERENT prompt spectrum on each of them. U240 is
/// the only such nuclide in ENDF/B-VIII.1 (its MT 18 is redundant and carries no
/// neutron product at all, while MT 19, 20, 21 and 38 each carry their own), so
/// it is also the only one this changes. With a single slot the run's entire
/// fission spectrum was whichever channel the FIRST fission event happened to
/// sample, frozen for every fission after it, which made the answer depend on
/// which thread got there first (issue #425).
#[derive(Debug, Default)]
pub struct FissionChiFlatCache([std::sync::OnceLock<FissionChiFlat>; FISSION_CHI_MTS.len()]);

impl Clone for FissionChiFlatCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl FissionChiFlatCache {
    /// Cache slot for a fission MT.
    ///
    /// The set is closed, so nothing outside it reaches here from the transport
    /// path. An unrecognised MT lands on MT 18's slot, which is what the caller's
    /// "18 when unresolved" fallback already means; the assertion is what stops a
    /// channel added to [`FISSION_CHI_MTS`]'s siblings but not to the array from
    /// silently sharing MT 18's spectrum, which is the bug this cache exists to
    /// prevent.
    fn slot(mt: i32) -> usize {
        debug_assert!(
            crate::nuclide::is_fission_mt(mt),
            "MT {mt} is not a fission MT and has no chi slot"
        );
        FISSION_CHI_MTS
            .iter()
            .position(|&m| m == mt)
            .unwrap_or_default()
    }

    /// Return `mt`'s cached flat chi, building it from `dist` (or
    /// [`FissionChiFlat::None`] when `dist` is `None`) on that slot's first access.
    pub fn get_or_build(&self, mt: i32, dist: Option<&EnergyDistribution>) -> &FissionChiFlat {
        self.0[Self::slot(mt)].get_or_init(|| match dist {
            Some(d) => d.to_fission_chi_flat(),
            None => FissionChiFlat::None,
        })
    }
}

/// Combine several outgoing-energy spectra into one, weighted by `weight`.
///
/// Used for the delayed-neutron groups (issue #364). ENDF gives each of the six
/// delayed groups its own spectrum plus its own `nu_d,g(E)`, and yamc emits
/// delayed neutrons with no time delay, so the group a neutron came from has no
/// observable consequence: only the mixture matters. Where the group weights are
/// energy-independent -- they are, exactly, for every evaluation checked (U235,
/// U238, Pu239 all hold `w_g` fixed to 4 decimals from 1e-5 eV to 20 MeV) -- the
/// mixture is a single fixed distribution, so folding the groups here is EXACT
/// rather than an approximation, and it leaves one spectrum for both backends to
/// sample instead of six.
///
/// Every part must be `ContinuousTabular` with no discrete lines and they must
/// share one interpolation law; the delayed groups are all `Histogram`. Each
/// part's FIRST incident-energy row is used, because the delayed spectra are
/// energy-independent (ENDF stores two identical rows spanning the range). The
/// result carries the same two-row shape so it flattens and samples exactly like
/// any other chi.
///
/// The fold is exact for both laws. Every part's breakpoints go into the union
/// grid, so no part's structure is lost, and a mixture of piecewise-constant
/// parts is piecewise-constant while a mixture of piecewise-linear parts is
/// piecewise-linear on that grid.
///
/// Returns `None` if `parts` is empty, if any part is not a single-law
/// `ContinuousTabular` without discrete lines, or if the weights do not sum to
/// something positive.
pub fn weighted_energy_mixture(parts: &[(f64, &EnergyDistribution)]) -> Option<EnergyDistribution> {
    let total_w: f64 = parts.iter().map(|(w, _)| *w).sum();
    if parts.is_empty() || total_w <= 0.0 || !total_w.is_finite() {
        return None;
    }

    // Each part's first row, normalised so its pdf integrates to 1 under its OWN
    // interpolation law. A group whose stored pdf is unnormalised would otherwise
    // enter the mixture with the wrong weight (the delayed groups' pdfs are
    // histograms that integrate to 1 as histograms but to ~0.99 as trapezoids).
    let mut rows: Vec<(f64, &[f64], Vec<f64>)> = Vec::with_capacity(parts.len());
    let mut law: Option<TabulatedInterp> = None;
    for (w, dist) in parts {
        let EnergyDistribution::ContinuousTabular { energy_out, .. } = dist else {
            return None;
        };
        let TabulatedProbability::Tabulated {
            x,
            p,
            interp,
            n_discrete,
            ..
        } = energy_out.first()?;
        if x.len() < 2 || p.len() != x.len() || *n_discrete != 0 {
            return None;
        }
        match law {
            None => law = Some(*interp),
            Some(seen) if seen == *interp => {}
            // A histogram part and a linear part have no common exact
            // representation; refuse rather than silently reinterpret one.
            Some(_) => return None,
        }
        let area = pdf_area(x, p, *interp);
        if area <= 0.0 || !area.is_finite() {
            return None;
        }
        let scaled: Vec<f64> = p.iter().map(|v| v / area).collect();
        rows.push((w / total_w, x.as_slice(), scaled));
    }
    let law = law?;

    // Union of the groups' outgoing-energy grids: no group's structure is lost,
    // and every group is evaluated on every other's points.
    let mut grid: Vec<f64> = rows
        .iter()
        .flat_map(|(_, x, _)| x.iter().copied())
        .collect();
    grid.sort_by(|a, b| a.partial_cmp(b).expect("finite outgoing energies"));
    grid.dedup();
    if grid.len() < 2 {
        return None;
    }

    let (grid, pdf) = match law {
        // Piecewise constant: the mixture's density on each union bin is the
        // weighted sum of the parts' densities on the bins that contain it, so
        // the union grid represents it exactly with nothing to interpolate. The
        // trailing entry is the unused right edge, which histogram sampling never
        // reads.
        TabulatedInterp::Histogram => {
            let pdf: Vec<f64> = grid
                .windows(2)
                .map(|b| {
                    rows.iter()
                        .map(|(w, x, p)| w * histogram_value_on(x, p, b[0]))
                        .sum()
                })
                .chain(std::iter::once(0.0))
                .collect();
            (grid, pdf)
        }
        // Piecewise linear: the mixture is DISCONTINUOUS wherever a group's
        // support starts or ends -- above a group's maximum energy it contributes
        // nothing, so the sum steps down there -- and a single value per grid
        // point cannot represent a step: interpolating across it smears the jump
        // and shifts the mean (mixing flat spectra on [0, 1] and [0, 3] MeV with
        // weights 0.25/0.75 gave 1.20 MeV instead of 1.25 MeV). So each point
        // carries its one-sided limits, and a point where they differ is emitted
        // TWICE. The duplicate spans a zero-width bin, which contributes nothing
        // to the cdf and is therefore never selected by the samplers' cdf search.
        TabulatedInterp::LinLin => {
            let limit = |at: f64, from_left: bool| -> f64 {
                rows.iter()
                    .filter(|(_, x, _)| {
                        let (lo, hi) = (x[0], x[x.len() - 1]);
                        if from_left {
                            at > lo && at <= hi
                        } else {
                            at >= lo && at < hi
                        }
                    })
                    .map(|(w, x, p)| w * interp_linear_at(x, p, at))
                    .sum()
            };
            let last = grid.len() - 1;
            let mut xs: Vec<f64> = Vec::with_capacity(grid.len() + rows.len() * 2);
            let mut pdf: Vec<f64> = Vec::with_capacity(xs.capacity());
            for (i, &e) in grid.iter().enumerate() {
                let (l, r) = (limit(e, true), limit(e, false));
                if i == 0 {
                    xs.push(e);
                    pdf.push(r);
                } else if i == last || l == r {
                    xs.push(e);
                    pdf.push(l);
                } else {
                    xs.push(e);
                    pdf.push(l);
                    xs.push(e);
                    pdf.push(r);
                }
            }
            (xs, pdf)
        }
    };

    let mut cdf = Vec::with_capacity(grid.len());
    let mut acc = 0.0;
    cdf.push(0.0);
    for i in 1..grid.len() {
        acc += match law {
            TabulatedInterp::Histogram => pdf[i - 1] * (grid[i] - grid[i - 1]),
            TabulatedInterp::LinLin => 0.5 * (pdf[i] + pdf[i - 1]) * (grid[i] - grid[i - 1]),
        };
        cdf.push(acc);
    }
    if acc <= 0.0 || !acc.is_finite() {
        return None;
    }
    let pdf: Vec<f64> = pdf.iter().map(|v| v / acc).collect();
    let cdf: Vec<f64> = cdf.iter().map(|v| v / acc).collect();

    let table = TabulatedProbability::Tabulated {
        x: grid,
        p: pdf,
        c: cdf,
        interp: law,
        n_discrete: 0,
    };
    // Two incident rows spanning the evaluation range, mirroring how ENDF stores
    // an energy-independent spectrum, so the flat sampler's bracket logic has a
    // bracket to find.
    let (e_lo, e_hi) = incident_range(parts)?;
    Some(EnergyDistribution::ContinuousTabular {
        energy: vec![e_lo, e_hi],
        energy_out: vec![table.clone(), table],
        histogram_interp: false,
    })
}

/// Integral of a tabulated pdf under its own interpolation law.
fn pdf_area(x: &[f64], p: &[f64], interp: TabulatedInterp) -> f64 {
    match interp {
        TabulatedInterp::Histogram => x
            .windows(2)
            .zip(p.iter())
            .map(|(xs, &v)| v * (xs[1] - xs[0]))
            .sum(),
        TabulatedInterp::LinLin => trapezoid(x, p),
    }
}

/// Piecewise-constant density at `at`: the value of the bin that contains it, or
/// zero outside the table's support.
fn histogram_value_on(x: &[f64], p: &[f64], at: f64) -> f64 {
    if at < x[0] || at >= x[x.len() - 1] {
        return 0.0;
    }
    // `partition_point` gives the count of edges at or below `at`; the bin index
    // is one less, and `at < x.last()` keeps it in range.
    p[x.partition_point(|&e| e <= at).max(1) - 1]
}

/// Widest incident-energy range covered by `parts`.
fn incident_range(parts: &[(f64, &EnergyDistribution)]) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (_, dist) in parts {
        let EnergyDistribution::ContinuousTabular { energy, .. } = dist else {
            return None;
        };
        lo = lo.min(*energy.first()?);
        hi = hi.max(*energy.last()?);
    }
    (hi > lo).then_some((lo, hi))
}

/// Trapezoid integral of `y` over `x`.
fn trapezoid(x: &[f64], y: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(xs, ys)| 0.5 * (ys[0] + ys[1]) * (xs[1] - xs[0]))
        .sum()
}

/// Linear interpolation of `(x, y)` at `at`, zero outside the tabulated range
/// (a delayed group contributes nothing above its own maximum energy).
fn interp_linear_at(x: &[f64], y: &[f64], at: f64) -> f64 {
    if at < x[0] || at > x[x.len() - 1] {
        return 0.0;
    }
    match x.binary_search_by(|v| v.partial_cmp(&at).expect("finite")) {
        Ok(i) => y[i],
        Err(0) => y[0],
        Err(i) if i >= x.len() => y[x.len() - 1],
        Err(i) => {
            let dx = x[i] - x[i - 1];
            if dx <= 0.0 {
                y[i - 1]
            } else {
                y[i - 1] + (y[i] - y[i - 1]) * (at - x[i - 1]) / dx
            }
        }
    }
}

#[cfg(test)]
mod fission_chi_cache_tests {
    use super::*;
    use crate::reaction_product::tabulated::Tabulated1D;

    /// Maxwell chi with a distinguishing `theta`, so two channels are tellable
    /// apart by the value that comes back out of the cache.
    fn maxwell(theta: f64) -> EnergyDistribution {
        EnergyDistribution::Maxwell {
            theta: Tabulated1D::Tabulated1D {
                x: vec![1.0, 2.0e7],
                y: vec![theta, theta],
                breakpoints: vec![2],
                interpolation: vec![2],
            },
            u: 0.0,
        }
    }

    fn theta_of(chi: &FissionChiFlat) -> f64 {
        match chi {
            FissionChiFlat::Maxwell { theta, .. } => theta[0],
            other => panic!("expected Maxwell, got {other:?}"),
        }
    }

    /// Issue #418. Each fission MT gets its own slot, so an evaluation whose
    /// partial channels carry different prompt spectra (U240 is the only one in
    /// ENDF/B-VIII.1) gets the spectrum of the channel that actually fissioned.
    /// The single-slot cache this replaced returned whichever channel the first
    /// fission event of the run happened to sample, for every fission after it.
    #[test]
    fn each_fission_mt_caches_its_own_chi() {
        let cache = FissionChiFlatCache::default();
        for (i, mt) in FISSION_CHI_MTS.iter().enumerate() {
            let built = cache.get_or_build(*mt, Some(&maxwell(i as f64 + 1.0)));
            assert_eq!(
                theta_of(built),
                i as f64 + 1.0,
                "MT {mt} took another channel's spectrum"
            );
        }
        // And re-reading a slot does not rebuild it: MT 19 keeps its own theta
        // even when handed MT 38's distribution.
        assert_eq!(
            theta_of(cache.get_or_build(19, Some(&maxwell(99.0)))),
            2.0,
            "a populated slot must stay cached"
        );
    }

    /// A nuclide with one fission channel is the overwhelmingly common case
    /// (87 of the 88 fissionable nuclides in ENDF/B-VIII.1), and it must behave
    /// exactly as the single-slot cache did.
    #[test]
    fn a_single_channel_nuclide_is_unaffected() {
        let cache = FissionChiFlatCache::default();
        assert_eq!(theta_of(cache.get_or_build(18, Some(&maxwell(1.5)))), 1.5);
        assert_eq!(theta_of(cache.get_or_build(18, Some(&maxwell(9.9)))), 1.5);
    }

    /// Absent product data caches as `None` for that channel without poisoning
    /// the others, which is what the MT 18 of a partial-fission evaluation needs
    /// (U240's MT 18 is redundant and carries no neutron product at all).
    #[test]
    fn a_channel_with_no_distribution_caches_none() {
        let cache = FissionChiFlatCache::default();
        assert!(matches!(cache.get_or_build(18, None), FissionChiFlat::None));
        assert_eq!(theta_of(cache.get_or_build(19, Some(&maxwell(3.0)))), 3.0);
    }

    /// `clone()` resets every slot, not just the first.
    #[test]
    fn clone_resets_all_slots() {
        let cache = FissionChiFlatCache::default();
        cache.get_or_build(19, Some(&maxwell(1.0)));
        cache.get_or_build(20, Some(&maxwell(2.0)));
        let fresh = cache.clone();
        assert_eq!(theta_of(fresh.get_or_build(19, Some(&maxwell(7.0)))), 7.0);
        assert_eq!(theta_of(fresh.get_or_build(20, Some(&maxwell(8.0)))), 8.0);
    }
}

#[cfg(test)]
mod delayed_mixture_tests {
    use super::*;

    /// Flat-pdf spectrum on `[lo, hi]`, normalised under `interp`, stored as two
    /// identical incident rows the way ENDF stores an energy-independent spectrum.
    fn flat(lo: f64, hi: f64, interp: TabulatedInterp) -> EnergyDistribution {
        let h = 1.0 / (hi - lo);
        // Under histogram interpolation the trailing pdf entry is the unused right
        // edge; under lin-lin both ends carry the density.
        let p = match interp {
            TabulatedInterp::Histogram => vec![h, 0.0],
            TabulatedInterp::LinLin => vec![h, h],
        };
        table_dist(vec![lo, hi], p, interp)
    }

    fn table_dist(x: Vec<f64>, p: Vec<f64>, interp: TabulatedInterp) -> EnergyDistribution {
        let mut c = vec![0.0];
        for i in 1..x.len() {
            let step = match interp {
                TabulatedInterp::Histogram => p[i - 1] * (x[i] - x[i - 1]),
                TabulatedInterp::LinLin => 0.5 * (p[i] + p[i - 1]) * (x[i] - x[i - 1]),
            };
            c.push(c[i - 1] + step);
        }
        let table = TabulatedProbability::Tabulated {
            x,
            p,
            c,
            interp,
            n_discrete: 0,
        };
        EnergyDistribution::ContinuousTabular {
            energy: vec![1.0e-5, 2.0e7],
            energy_out: vec![table.clone(), table],
            histogram_interp: false,
        }
    }

    fn row(dist: &EnergyDistribution) -> (&[f64], &[f64], TabulatedInterp) {
        let EnergyDistribution::ContinuousTabular { energy_out, .. } = dist else {
            panic!("expected ContinuousTabular");
        };
        let TabulatedProbability::Tabulated { x, p, interp, .. } = &energy_out[0];
        (x, p, *interp)
    }

    /// Mean outgoing energy under the table's own interpolation law.
    fn mean_of(dist: &EnergyDistribution) -> f64 {
        let (x, p, interp) = row(dist);
        let num: f64 = match interp {
            TabulatedInterp::Histogram => x
                .windows(2)
                .zip(p.iter())
                .map(|(xs, &v)| v * 0.5 * (xs[1] * xs[1] - xs[0] * xs[0]))
                .sum(),
            TabulatedInterp::LinLin => x
                .windows(2)
                .zip(p.windows(2))
                .map(|(xs, ps)| {
                    // integral of x*p over a linear bin
                    let (a, b) = (xs[0], xs[1]);
                    let (pa, pb) = (ps[0], ps[1]);
                    let h = b - a;
                    h * (a * (2.0 * pa + pb) + b * (pa + 2.0 * pb)) / 6.0
                })
                .sum(),
        };
        num / pdf_area(x, p, interp)
    }

    /// Density at `at` under the table's own law, zero outside its support.
    fn density_at(dist: &EnergyDistribution, at: f64) -> f64 {
        let (x, p, interp) = row(dist);
        let area = pdf_area(x, p, interp);
        let raw = match interp {
            TabulatedInterp::Histogram => histogram_value_on(x, p, at),
            TabulatedInterp::LinLin => {
                if at < x[0] || at > x[x.len() - 1] {
                    0.0
                } else {
                    interp_linear_at(x, p, at)
                }
            }
        };
        raw / area
    }

    /// The mixture's mean must be the weight-average of the parts' means, which is
    /// the property the delayed fold relies on. Checked for both laws, and with
    /// parts whose supports END at different energies -- the case that makes the
    /// mixture discontinuous.
    #[test]
    fn mixture_mean_is_the_weighted_mean() {
        for interp in [TabulatedInterp::Histogram, TabulatedInterp::LinLin] {
            let a = flat(0.0, 1.0e6, interp);
            let b = flat(0.0, 3.0e6, interp);
            for &wa in &[0.0_f64, 0.25, 0.5, 0.75, 1.0] {
                let parts = [(wa, &a), (1.0 - wa, &b)];
                let mix = weighted_energy_mixture(&parts).expect("mixture");
                let expected = wa * mean_of(&a) + (1.0 - wa) * mean_of(&b);
                let got = mean_of(&mix);
                assert!(
                    (got - expected).abs() <= 1e-9 * expected.max(1.0),
                    "{interp:?} wa={wa}: mixture mean {got:.6e} vs weighted {expected:.6e}"
                );
            }
        }
    }

    /// Pointwise exactness, not just the mean: the folded density must equal the
    /// weighted sum of the parts' densities everywhere, including inside the bin
    /// where one part stops contributing. Uses stair-step histograms with
    /// mismatched grids, like the real delayed groups (217 / 252 / 333 points over
    /// different ranges).
    #[test]
    fn histogram_fold_is_pointwise_exact() {
        let h = TabulatedInterp::Histogram;
        let a = table_dist(
            vec![0.0, 3.0e5, 9.0e5, 1.84e6],
            vec![2.0e-6, 5.0e-7, 1.0e-7, 0.0],
            h,
        );
        let b = table_dist(vec![0.0, 4.0e5, 2.19e6], vec![1.5e-6, 2.0e-7, 0.0], h);
        let c = table_dist(vec![0.0, 1.0e6, 3.0e6], vec![8.0e-7, 1.0e-7, 0.0], h);
        let (wa, wb, wc) = (0.2_f64, 0.5, 0.3);
        let mix = weighted_energy_mixture(&[(wa, &a), (wb, &b), (wc, &c)]).expect("mixture");
        for i in 0..600 {
            let e = i as f64 * 5.0e3; // 0 .. 3 MeV, crossing every breakpoint
            let expected = wa * density_at(&a, e) + wb * density_at(&b, e) + wc * density_at(&c, e);
            let got = density_at(&mix, e);
            assert!(
                (got - expected).abs() <= 1e-12 * expected.max(1e-9),
                "at {e:.4e} eV: folded {got:.12e} vs weighted {expected:.12e}"
            );
        }
    }

    /// The mixture is normalised and its cdf ends at exactly 1, under both laws.
    #[test]
    fn mixture_is_normalised() {
        for interp in [TabulatedInterp::Histogram, TabulatedInterp::LinLin] {
            let mix = weighted_energy_mixture(&[
                (0.3, &flat(0.0, 1.0e6, interp)),
                (0.7, &flat(0.0, 2.0e6, interp)),
            ])
            .expect("mixture");
            let EnergyDistribution::ContinuousTabular {
                energy_out, energy, ..
            } = &mix
            else {
                panic!()
            };
            assert_eq!(energy.len(), 2, "two incident rows so a bracket exists");
            for r in energy_out {
                let TabulatedProbability::Tabulated { x, p, c, .. } = r;
                assert!(
                    (pdf_area(x, p, interp) - 1.0).abs() < 1e-12,
                    "{interp:?} pdf must integrate to 1"
                );
                assert!(
                    (c[c.len() - 1] - 1.0).abs() < 1e-12,
                    "{interp:?} cdf must end at 1"
                );
                assert_eq!(c[0], 0.0, "cdf must start at 0");
            }
        }
    }

    /// Weights that do not sum to 1 are renormalised, so raw nu_d,g values can be
    /// handed straight in.
    #[test]
    fn unnormalised_weights_are_renormalised() {
        let a = flat(0.0, 1.0e6, TabulatedInterp::Histogram);
        let b = flat(0.0, 3.0e6, TabulatedInterp::Histogram);
        let raw = weighted_energy_mixture(&[(0.003, &a), (0.007, &b)]).expect("mixture");
        let norm = weighted_energy_mixture(&[(0.3, &a), (0.7, &b)]).expect("mixture");
        assert!((mean_of(&raw) - mean_of(&norm)).abs() < 1e-9 * mean_of(&norm));
    }

    /// A part whose stored pdf is unnormalised must still enter with its stated
    /// weight. The real delayed pdfs are histograms whose values integrate to 1 as
    /// histograms but to ~0.99 as trapezoids, so getting the law wrong here would
    /// mis-weight the groups by ~1%.
    #[test]
    fn parts_are_normalised_before_weighting() {
        let h = TabulatedInterp::Histogram;
        let a = flat(0.0, 1.0e6, h);
        let b = table_dist(vec![0.0, 3.0e6], vec![17.3, 0.0], h); // area 5.19e7, not 1
        let mix = weighted_energy_mixture(&[(0.5, &a), (0.5, &b)]).expect("mixture");
        let expected = 0.5 * mean_of(&a) + 0.5 * mean_of(&b);
        assert!(
            (mean_of(&mix) - expected).abs() <= 1e-9 * expected,
            "mixture mean {:.6e} vs {expected:.6e}",
            mean_of(&mix)
        );
    }

    /// Refusals: an unsupported kind, no parts, mixed interpolation laws, and
    /// discrete lines. Each would otherwise fold to a silently wrong spectrum.
    #[test]
    fn rejects_unfoldable_inputs() {
        let maxwell = EnergyDistribution::Maxwell {
            theta: Tabulated1D::Tabulated1D {
                x: vec![1.0, 2.0e7],
                y: vec![1.3e6, 1.3e6],
                breakpoints: vec![2],
                interpolation: vec![2],
            },
            u: 0.0,
        };
        assert!(weighted_energy_mixture(&[(1.0, &maxwell)]).is_none());
        assert!(weighted_energy_mixture(&[]).is_none());

        let hist = flat(0.0, 1.0e6, TabulatedInterp::Histogram);
        let linlin = flat(0.0, 2.0e6, TabulatedInterp::LinLin);
        assert!(
            weighted_energy_mixture(&[(0.5, &hist), (0.5, &linlin)]).is_none(),
            "a histogram part and a linear part have no common exact form"
        );

        let mut discrete = flat(0.0, 1.0e6, TabulatedInterp::Histogram);
        if let EnergyDistribution::ContinuousTabular { energy_out, .. } = &mut discrete {
            for r in energy_out {
                let TabulatedProbability::Tabulated { n_discrete, .. } = r;
                *n_discrete = 1;
            }
        }
        assert!(
            weighted_energy_mixture(&[(1.0, &discrete)]).is_none(),
            "discrete lines are not foldable as continuous density"
        );
    }
}
