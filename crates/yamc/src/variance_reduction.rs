//! Variance-reduction technique configuration.
//!
//! [`crate::model::Model::variance_reduction`] holds a list of technique
//! settings; an empty list (the default) is fully analog transport.
//! Techniques compose rather than compete: each hooks a different part of
//! transport (survival biasing at collisions, weight windows at
//! crossings, source biasing at birth), so the list may mix them. Order
//! carries no meaning. Multiplicity is validated per technique at run
//! start: at most one [`SurvivalBiasing`] entry; future weight windows
//! may legitimately have several entries (per particle type / energy
//! range).

use yamc_particle::ParticleType;
use yamc_tallies::RegularRectangularMesh;

/// Survival biasing (implicit capture) with weight-cutoff Russian
/// roulette.
///
/// Absorption is never sampled as a terminal event; every collision
/// multiplies the particle weight by the scattering probability
/// `sigma_s / sigma_t` and the particle always scatters. A particle whose
/// weight drops below `weight_cutoff` survives roulette with probability
/// `weight / weight_survive` (continuing at `weight_survive`) or is
/// killed, conserving weight in expectation. Defaults match the standard
/// `survival_biasing` setting (cutoff `weight` 0.25, `weight_avg` 1.0).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SurvivalBiasing {
    /// Russian-roulette trigger weight: a particle below it plays
    /// roulette. Default: 0.25.
    #[serde(default = "default_weight_cutoff")]
    pub weight_cutoff: f64,
    /// Weight assigned to roulette survivors. Default: 1.0.
    #[serde(default = "default_weight_survive")]
    pub weight_survive: f64,
}

impl Default for SurvivalBiasing {
    fn default() -> Self {
        Self {
            weight_cutoff: default_weight_cutoff(),
            weight_survive: default_weight_survive(),
        }
    }
}

/// Serde defaults (plain `#[serde(default)]` would give 0.0, an invalid
/// configuration).
fn default_weight_cutoff() -> f64 {
    0.25
}

fn default_weight_survive() -> f64 {
    1.0
}

impl SurvivalBiasing {
    /// Validate the weight parameters. `weight_survive` below
    /// `weight_cutoff` would hand every roulette survivor a weight that
    /// immediately re-triggers roulette, so it is rejected up front.
    pub fn validate(&self) -> Result<(), String> {
        if self.weight_cutoff <= 0.0 {
            return Err(format!(
                "SurvivalBiasing requires weight_cutoff > 0 (got {})",
                self.weight_cutoff
            ));
        }
        if self.weight_survive < self.weight_cutoff {
            return Err(format!(
                "SurvivalBiasing requires weight_survive >= weight_cutoff (got weight_survive {} < weight_cutoff {})",
                self.weight_survive, self.weight_cutoff
            ));
        }
        Ok(())
    }
}

/// Default roulette survivor factor (survivor weight = factor * lower).
fn default_survival_factor() -> f64 {
    3.0
}

/// Default maximum copies produced in one split event.
fn default_max_split() -> u32 {
    10
}

/// Default absolute weight floor below which a particle is killed.
fn default_weight_floor() -> f64 {
    1e-38
}

/// Default upper/lower window ratio.
fn default_ratio() -> f64 {
    5.0
}

/// DeGVR auto-N target reduced optical depth (mean free paths): when
/// `density_reduction` is left unset, `N` is chosen so the reduced-density
/// fiducial pass is roughly this many mfp thick (`N = tau / target`, continuous).
///
/// Empirical sweet spot ~7 mfp from the concrete-sphere sweep in
/// `examples/python/degvr_autoN_sweep.py` (fits best-N within +/-1 over
/// tau = 14..32). Larger than the naive ~1-2 mfp: an aggressive (large-N)
/// extrapolation `G = F*(H/F)^(N-1)` accumulates systematic error over a wide
/// density span, so a thicker fiducial with a gentler extrapolation wins; the
/// sweep also showed this optimum is generation-budget-independent (the limit
/// is extrapolation bias, not statistical noise).
pub(crate) const DEGVR_TAU_TARGET: f64 = 7.0;
/// Minimum auto-N (the fiducial is always at least a 2x density reduction).
pub(crate) const DEGVR_N_MIN: f64 = 2.0;

/// Fallback representative photon energy (eV) for the DeGVR auto-N ray-trace
/// when the photons are produced by neutrons (coupled secondary photons or D1S
/// decay photons) rather than emitted by a photon source: the auto ray-trace
/// integrates the photon attenuation at this energy, not at the driving neutron
/// energy (14 MeV attenuation is nothing like ~MeV-gamma attenuation). 1 MeV is
/// a typical secondary / decay gamma energy, near the photon attenuation minimum
/// for most materials, so the estimate is conservative (thin fiducial), which is
/// what photon windows prefer. Override per-problem via
/// [`WeightWindowGeneratorDeGVR::photon_energy`]. Deriving it from the material's
/// photon-production spectrum is a documented future refinement; auto-N only sets
/// generation efficiency, never bias.
pub(crate) const DEGVR_PHOTON_REP_ENERGY_EV: f64 = 1.0e6;

/// A mesh-based weight-window map: per (energy group, voxel) lower/upper
/// target-weight bounds for a single particle type.
///
/// Applied at collisions: a particle with weight above `upper` is split
/// into equal-weight copies (capped by `max_split`); one at or below
/// `lower` plays Russian roulette (survivor weight `survival_factor *
/// lower`); one below `weight_floor` is killed. A negative `lower` entry
/// is the "no window" sentinel, so that region is skipped. Bounds are
/// stored flat, indexed `group * num_voxels + voxel`. Multiple
/// `WeightWindowBounds` entries may coexist in the variance-reduction
/// list (for example one per particle type).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeightWindowBounds {
    /// Spatial mesh over which the windows are defined.
    pub mesh: RegularRectangularMesh,
    /// Particle type these windows apply to.
    pub particle: ParticleType,
    /// Energy group edges in eV, strictly ascending. `None` is a single
    /// group spanning all energies.
    pub energy_bins: Option<Vec<f64>>,
    /// Lower window bound per `(group, voxel)`, flat index
    /// `group * num_voxels + voxel`. A negative value is the "no window"
    /// sentinel.
    pub lower_bounds: Vec<f64>,
    /// Upper window bound per `(group, voxel)`, same layout as
    /// `lower_bounds`.
    pub upper_bounds: Vec<f64>,
    /// Roulette survivor weight is `survival_factor * lower`. Default 3.
    #[serde(default = "default_survival_factor")]
    pub survival_factor: f64,
    /// Maximum number of copies produced in one split event. Default 10.
    #[serde(default = "default_max_split")]
    pub max_split: u32,
    /// Absolute weight floor; a particle below it is killed. Default 1e-38.
    #[serde(default = "default_weight_floor")]
    pub weight_floor: f64,
}

impl WeightWindowBounds {
    /// Number of energy groups (`energy_bins.len() - 1`, or 1 when there
    /// is no energy binning).
    pub fn n_groups(&self) -> usize {
        self.energy_bins
            .as_ref()
            .map(|b| b.len().saturating_sub(1))
            .unwrap_or(1)
            .max(1)
    }

    /// Number of spatial voxels in the window mesh.
    pub fn num_voxels(&self) -> usize {
        self.mesh.num_voxels()
    }

    /// Energy group index for `energy`, or `None` if outside the binning
    /// (the region is then treated as having no window). Convention
    /// matches tallies: `[e0, e1], (e1, e2], ...`.
    fn group(&self, energy: f64) -> Option<usize> {
        match &self.energy_bins {
            None => Some(0),
            Some(bins) => {
                if bins.len() < 2 {
                    return Some(0);
                }
                if energy < bins[0] || energy > bins[bins.len() - 1] {
                    return None;
                }
                let g = bins.partition_point(|&e| e <= energy).max(1);
                Some((g - 1).min(bins.len() - 2))
            }
        }
    }

    /// Lower/upper window bounds for a particle at `position` with
    /// `energy`, or `None` when there is no window (outside the mesh,
    /// outside the energy binning, or a sentinel voxel).
    pub fn window(&self, position: [f64; 3], energy: f64) -> Option<(f64, f64)> {
        let voxel = self.mesh.get_bin(position)?;
        let group = self.group(energy)?;
        let idx = group * self.num_voxels() + voxel;
        let lower = *self.lower_bounds.get(idx)?;
        if lower < 0.0 {
            return None;
        }
        Some((lower, self.upper_bounds[idx]))
    }

    /// Validate bound-array sizes and parameter ranges.
    pub fn validate(&self) -> Result<(), String> {
        let n_v = self.num_voxels();
        let n_g = self.n_groups();
        let expected = n_v
            .checked_mul(n_g)
            .ok_or_else(|| "WeightWindowBounds size overflow".to_string())?;
        if self.lower_bounds.len() != expected || self.upper_bounds.len() != expected {
            return Err(format!(
                "WeightWindowBounds expects {expected} bounds (n_groups {n_g} x num_voxels {n_v}); got lower {} upper {}",
                self.lower_bounds.len(),
                self.upper_bounds.len()
            ));
        }
        if let Some(bins) = &self.energy_bins {
            if bins.len() < 2 {
                return Err("energy_bins requires at least 2 edges".to_string());
            }
            if bins.windows(2).any(|w| w[1] <= w[0]) {
                return Err("energy_bins must be strictly ascending".to_string());
            }
        }
        if self.survival_factor <= 1.0 {
            return Err(format!(
                "WeightWindowBounds requires survival_factor > 1 (got {})",
                self.survival_factor
            ));
        }
        if self.max_split < 1 {
            return Err("WeightWindowBounds requires max_split >= 1".to_string());
        }
        if !(self.weight_floor > 0.0 && self.weight_floor <= 1.0) {
            return Err(format!(
                "WeightWindowBounds requires 0 < weight_floor <= 1 (got {})",
                self.weight_floor
            ));
        }
        for (i, (&lo, &hi)) in self.lower_bounds.iter().zip(&self.upper_bounds).enumerate() {
            if lo >= 0.0 && hi < lo {
                return Err(format!(
                    "WeightWindowBounds upper {hi} < lower {lo} at index {i}"
                ));
            }
        }
        Ok(())
    }
}

/// Configuration for generating weight windows with the DeGVR method
/// (Density-extrapolation Global Variance Reduction, Pan et al. 2023).
///
/// DeGVR builds an importance map for a deep-penetration problem without an
/// iterative bootstrap. It reduces all material densities so particles reach
/// the whole mesh, runs a low-density "fiducial" fixed-source pass (density
/// original / `density_reduction`) and a half-reduction "asymptotic" pass
/// (density original / (`density_reduction` / 2), using the fiducial
/// windows), then extrapolates the two flux fields back to the original
/// density per voxel and energy group (`G = F * (H / F)^(N - 1)`, with
/// `N = density_reduction`) to obtain the approximate flux the final windows
/// are built from. Suited to bulk attenuation; weak on streaming / void /
/// near-source regions (use MAGIC there). Passed to
/// [`crate::model::Model::generate_weight_windows`], which returns one
/// [`WeightWindowBounds`] per requested particle type (in `particles` order).
///
/// Requesting several particles (`particles = [Neutron, Photon]`) builds all
/// the windows from a *single* coupled pair of reduced-density passes: the
/// fiducial pass tallies one flux mesh per particle, and the asymptotic pass
/// applies every fiducial window at once. This is how a coupled neutron ->
/// secondary-photon (or D1S decay-photon) problem gets a neutron window and a
/// photon window from one generation, ready to apply together in production.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeightWindowGeneratorDeGVR {
    /// Mesh the windows are generated on.
    pub mesh: RegularRectangularMesh,
    /// Energy group edges (eV), strictly ascending. `None` = a single group.
    pub energy_bins: Option<Vec<f64>>,
    /// Particle types to generate windows for (at least one, no duplicates).
    /// One [`WeightWindowBounds`] is produced per entry, in this order.
    pub particles: Vec<ParticleType>,
    /// Representative photon energy (eV) for the auto-N optical-depth ray-trace
    /// of a photon window. `None` (the default) means auto: the source energy
    /// when photons are emitted directly by a photon source, else a typical
    /// secondary / decay gamma energy ([`DEGVR_PHOTON_REP_ENERGY_EV`]). Only
    /// affects the auto `density_reduction` estimate, never bias; ignored when
    /// `density_reduction` is set or no photon window is requested.
    #[serde(default)]
    pub photon_energy: Option<f64>,
    /// Density-reduction factor `N`: the fiducial run uses `density / N` and
    /// the asymptotic run uses `density / (N / 2)`. `None` (the default) means
    /// auto: `N` is computed from a particle-free optical-depth ray-trace of the
    /// problem (source to deepest voxel), taking the max over requested
    /// particles so the fiducial is thin enough to seed every field. An explicit
    /// value overrides the auto estimate.
    #[serde(default)]
    pub density_reduction: Option<f64>,
    /// Upper/lower ratio of the generated windows. Default 5.
    #[serde(default = "default_ratio")]
    pub ratio: f64,
    /// Roulette survivor factor of the generated windows. Default 3.
    #[serde(default = "default_survival_factor")]
    pub survival_factor: f64,
    /// Maximum split of the generated windows. Default 10.
    #[serde(default = "default_max_split")]
    pub max_split: u32,
    /// Absolute weight floor of the generated windows. Default 1e-38.
    #[serde(default = "default_weight_floor")]
    pub weight_floor: f64,
}

impl WeightWindowGeneratorDeGVR {
    /// Number of energy groups (`energy_bins.len() - 1`, or 1 when there is
    /// no energy binning).
    pub fn n_groups(&self) -> usize {
        self.energy_bins
            .as_ref()
            .map(|b| b.len().saturating_sub(1))
            .unwrap_or(1)
            .max(1)
    }

    /// Auto density-reduction factor from an estimated optical depth `tau`:
    /// `N = tau / DEGVR_TAU_TARGET`, clamped to at least `DEGVR_N_MIN`. `N` is
    /// continuous: the density scaling (`density / N`, `density / (N/2)`) and the
    /// extrapolation exponent (`N - 1`) are all smooth in `N`, so there is no
    /// reason to round to an integer. The `>= 2` clamp is physical (it keeps the
    /// asymptotic pass at or below the true density).
    pub fn n_from_tau(tau: f64) -> f64 {
        (tau / DEGVR_TAU_TARGET).max(DEGVR_N_MIN)
    }

    /// Validate the generator parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.particles.is_empty() {
            return Err(
                "WeightWindowGeneratorDeGVR requires at least one particle type".to_string(),
            );
        }
        for (i, p) in self.particles.iter().enumerate() {
            if self.particles[..i].contains(p) {
                return Err(format!(
                    "WeightWindowGeneratorDeGVR particles must be unique (got {p:?} twice)"
                ));
            }
        }
        if let Some(e) = self.photon_energy {
            if !(e > 0.0 && e.is_finite()) {
                return Err(format!(
                    "WeightWindowGeneratorDeGVR requires photon_energy > 0 (got {e})"
                ));
            }
        }
        if let Some(n) = self.density_reduction {
            if n < 2.0 {
                return Err(format!(
                    "WeightWindowGeneratorDeGVR requires density_reduction >= 2 (got {n})"
                ));
            }
        }
        if self.ratio <= 1.0 {
            return Err(format!(
                "WeightWindowGeneratorDeGVR requires ratio > 1 (got {})",
                self.ratio
            ));
        }
        if self.survival_factor <= 1.0 {
            return Err(format!(
                "WeightWindowGeneratorDeGVR requires survival_factor > 1 (got {})",
                self.survival_factor
            ));
        }
        if self.max_split < 1 {
            return Err("WeightWindowGeneratorDeGVR requires max_split >= 1".to_string());
        }
        if !(self.weight_floor > 0.0 && self.weight_floor <= 1.0) {
            return Err(format!(
                "WeightWindowGeneratorDeGVR requires 0 < weight_floor <= 1 (got {})",
                self.weight_floor
            ));
        }
        if let Some(bins) = &self.energy_bins {
            if bins.len() < 2 {
                return Err("energy_bins requires at least 2 edges".to_string());
            }
            if bins.windows(2).any(|w| w[1] <= w[0]) {
                return Err("energy_bins must be strictly ascending".to_string());
            }
        }
        Ok(())
    }

    /// Build a [`WeightWindowBounds`] from a per-(group, voxel) flux field
    /// (flat, indexed `group * num_voxels + voxel`). Normalized per energy
    /// group so the peak-flux voxel gets a survival weight of 1.0 (matching a
    /// unit source birth weight, since source biasing is not applied): `lower
    /// = flux / (survival_factor * group_max)`, `upper = ratio * lower`. A
    /// voxel with non-positive/non-finite flux, or in an all-zero group, gets
    /// the negative "no window" sentinel. `particle` is the particle type the
    /// returned window applies to (one of `self.particles`).
    pub fn build_bounds(&self, flux: &[f64], particle: ParticleType) -> WeightWindowBounds {
        let nv = self.mesh.num_voxels();
        let ng = self.n_groups();
        let mut lower = vec![-1.0; ng * nv];
        let mut upper = vec![-1.0; ng * nv];
        for g in 0..ng {
            let base = g * nv;
            let group_max = flux[base..base + nv]
                .iter()
                .copied()
                .filter(|f| f.is_finite())
                .fold(0.0_f64, f64::max);
            if group_max <= 0.0 {
                continue;
            }
            let scale = 1.0 / (self.survival_factor * group_max);
            for v in 0..nv {
                let f = flux[base + v];
                if f > 0.0 && f.is_finite() {
                    let lo = f * scale;
                    lower[base + v] = lo;
                    upper[base + v] = self.ratio * lo;
                }
            }
        }
        WeightWindowBounds {
            mesh: self.mesh.clone(),
            particle,
            energy_bins: self.energy_bins.clone(),
            lower_bounds: lower,
            upper_bounds: upper,
            survival_factor: self.survival_factor,
            max_split: self.max_split,
            weight_floor: self.weight_floor,
        }
    }

    /// Extrapolate the original-density flux from the DeGVR fiducial field `F`
    /// (density / N) and asymptotic field `H` (density / (N / 2)), per bin:
    /// `G = F * (H / F)^(N - 1)`, evaluated in log space. A bin with
    /// non-positive/non-finite `F`, `H`, or result is set to 0 (which yields a
    /// sentinel window in [`build_bounds`], so the region is skipped rather
    /// than triggering a degenerate zero-width window).
    pub fn extrapolate(&self, fiducial: &[f64], asymptotic: &[f64], n: f64) -> Vec<f64> {
        fiducial
            .iter()
            .zip(asymptotic)
            .map(|(&f, &h)| {
                if f > 0.0 && h > 0.0 && f.is_finite() && h.is_finite() {
                    let ln_g = f.ln() + (n - 1.0) * (h.ln() - f.ln());
                    let g = ln_g.exp();
                    if g.is_finite() {
                        g
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// One variance-reduction technique's configuration. Future techniques
/// (weight windows, source biasing) are added as further variants.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum VarianceReduction {
    /// Survival biasing (implicit capture) + weight-cutoff Russian
    /// roulette.
    SurvivalBiasing(SurvivalBiasing),
    /// Mesh-based weight windows (split above `upper`, roulette below
    /// `lower`). Several entries are legal (for example per particle
    /// type).
    WeightWindowBounds(Box<WeightWindowBounds>),
}

#[cfg(test)]
mod tests {
    use super::{WeightWindowBounds, WeightWindowGeneratorDeGVR};
    use yamc_particle::ParticleType;
    use yamc_tallies::RegularRectangularMesh;

    fn mesh_2x1x2() -> RegularRectangularMesh {
        // 4 voxels: nx=2, ny=1, nz=2 over [0,2] x [0,1] x [0,2].
        RegularRectangularMesh::try_new([0.0, 0.0, 0.0], [2.0, 1.0, 2.0], [2, 1, 2]).unwrap()
    }

    fn wwb(lower: Vec<f64>, upper: Vec<f64>, energy_bins: Option<Vec<f64>>) -> WeightWindowBounds {
        WeightWindowBounds {
            mesh: mesh_2x1x2(),
            particle: ParticleType::Neutron,
            energy_bins,
            lower_bounds: lower,
            upper_bounds: upper,
            survival_factor: 3.0,
            max_split: 10,
            weight_floor: 1e-38,
        }
    }

    #[test]
    fn validate_accepts_correct_sizes() {
        let w = wwb(vec![0.1; 4], vec![0.5; 4], None);
        assert!(w.validate().is_ok());
        assert_eq!(w.num_voxels(), 4);
        assert_eq!(w.n_groups(), 1);
    }

    #[test]
    fn validate_rejects_wrong_length() {
        assert!(wwb(vec![0.1; 3], vec![0.5; 3], None).validate().is_err());
    }

    #[test]
    fn validate_rejects_bad_params() {
        let mut w = wwb(vec![0.1; 4], vec![0.5; 4], None);
        w.survival_factor = 1.0; // must be > 1
        assert!(w.validate().is_err());

        let mut w = wwb(vec![0.1; 4], vec![0.5; 4], None);
        w.weight_floor = 0.0; // must be in (0, 1]
        assert!(w.validate().is_err());

        let mut w = wwb(vec![0.1; 4], vec![0.5; 4], None);
        w.weight_floor = 2.0;
        assert!(w.validate().is_err());

        // upper < lower on a non-sentinel voxel
        assert!(wwb(vec![0.5; 4], vec![0.1; 4], None).validate().is_err());
    }

    #[test]
    fn validate_energy_bins_must_ascend() {
        // 2 groups -> 8 bounds.
        let w = wwb(vec![0.1; 8], vec![0.5; 8], Some(vec![0.0, 1e6, 2e6]));
        assert!(w.validate().is_ok());
        assert_eq!(w.n_groups(), 2);
        assert!(wwb(vec![0.1; 8], vec![0.5; 8], Some(vec![0.0, 2e6, 1e6]))
            .validate()
            .is_err());
    }

    #[test]
    fn window_lookup_and_sentinel() {
        let lower = vec![0.1, 0.2, 0.3, 0.4];
        let upper = vec![0.5, 1.0, 1.5, 2.0];
        let w = wwb(lower.clone(), upper.clone(), None);
        let v = w.mesh.get_bin([0.5, 0.5, 0.5]).unwrap();
        assert_eq!(w.window([0.5, 0.5, 0.5], 1.0), Some((lower[v], upper[v])));
        // outside the mesh -> no window
        assert_eq!(w.window([5.0, 0.5, 0.5], 1.0), None);
        // sentinel (negative lower) -> no window
        let vs = w.mesh.get_bin([1.5, 0.5, 0.5]).unwrap();
        let mut lower2 = lower.clone();
        lower2[vs] = -1.0;
        let w2 = wwb(lower2, upper, None);
        assert_eq!(w2.window([1.5, 0.5, 0.5], 1.0), None);
    }

    #[test]
    fn window_selects_energy_group() {
        // 2 groups [0,1e6],(1e6,2e6]; voxel 0 at index g*4+0.
        let mut lower = vec![0.1; 8];
        lower[0] = 0.11; // group 0, voxel 0
        lower[4] = 0.44; // group 1, voxel 0
        let w = wwb(lower, vec![9.0; 8], Some(vec![0.0, 1e6, 2e6]));
        assert_eq!(w.mesh.get_bin([0.5, 0.5, 0.5]).unwrap(), 0);
        assert_eq!(w.window([0.5, 0.5, 0.5], 5e5).map(|(lo, _)| lo), Some(0.11));
        assert_eq!(
            w.window([0.5, 0.5, 0.5], 1.5e6).map(|(lo, _)| lo),
            Some(0.44)
        );
        // energy outside the binning -> no window
        assert_eq!(w.window([0.5, 0.5, 0.5], 3e6), None);
    }

    fn degvr(survival_factor: f64, ratio: f64, n: f64) -> WeightWindowGeneratorDeGVR {
        WeightWindowGeneratorDeGVR {
            mesh: mesh_2x1x2(),
            energy_bins: None,
            particles: vec![ParticleType::Neutron],
            photon_energy: None,
            density_reduction: Some(n),
            ratio,
            survival_factor,
            max_split: 10,
            weight_floor: 1e-38,
        }
    }

    #[test]
    fn degvr_build_bounds_normalizes_and_sentinels() {
        let gen = degvr(3.0, 5.0, 32.0);
        // 4 voxels, single group; voxel 1 is the peak, voxel 3 unscored.
        let flux = vec![2.0, 8.0, 4.0, 0.0];
        let wwb = gen.build_bounds(&flux, ParticleType::Neutron);
        assert!(wwb.validate().is_ok());
        // peak voxel (flux 8): survival weight 1.0 -> lower = 1 / survival_factor
        assert!((wwb.lower_bounds[1] - 1.0 / 3.0).abs() < 1e-12);
        assert!((wwb.upper_bounds[1] - 5.0 / 3.0).abs() < 1e-12);
        // proportional elsewhere (voxel 0 is a quarter of the peak flux)
        assert!((wwb.lower_bounds[0] - (2.0 / 8.0) / 3.0).abs() < 1e-12);
        // unscored voxel -> sentinel
        assert!(wwb.lower_bounds[3] < 0.0);
    }

    #[test]
    fn degvr_extrapolate_matches_closed_form() {
        let f = vec![1.0, 0.5, 0.25];
        let h = vec![1.0, 0.25, 0.0625];
        // N = 2: asymptotic is already at original density, so G == H.
        let g = degvr(3.0, 5.0, 2.0).extrapolate(&f, &h, 2.0);
        for (gi, hi) in g.iter().zip(&h) {
            assert!((gi - hi).abs() < 1e-12, "N=2 should give G == H");
        }
        // N = 3: G = F * (H / F)^2.
        let g3 = degvr(3.0, 5.0, 3.0).extrapolate(&f, &h, 3.0);
        for i in 0..f.len() {
            let expected = f[i] * (h[i] / f[i]).powi(2);
            assert!((g3[i] - expected).abs() < 1e-9);
        }
        // non-positive inputs -> 0 (sentinel downstream)
        assert_eq!(
            degvr(3.0, 5.0, 2.0).extrapolate(&[0.0, 1.0], &[1.0, 0.0], 2.0),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn degvr_n_from_tau_scales_and_clamps() {
        use super::{WeightWindowGeneratorDeGVR, DEGVR_TAU_TARGET};
        // N = round(tau / target), so a tau of 3*target rounds to 3.
        assert_eq!(
            WeightWindowGeneratorDeGVR::n_from_tau(3.0 * DEGVR_TAU_TARGET),
            3.0
        );
        // Deep problem scales up.
        assert_eq!(
            WeightWindowGeneratorDeGVR::n_from_tau(20.0 * DEGVR_TAU_TARGET),
            20.0
        );
        // Shallow problem clamps to the 2x-reduction floor.
        assert_eq!(WeightWindowGeneratorDeGVR::n_from_tau(0.1), 2.0);
        // N is continuous, not rounded: a fractional multiple stays fractional.
        assert!(
            (WeightWindowGeneratorDeGVR::n_from_tau(4.5 * DEGVR_TAU_TARGET) - 4.5).abs() < 1e-12
        );
    }

    #[test]
    fn degvr_validate_particles_list() {
        // A valid multi-particle generator (neutron + photon) passes.
        let mut gen = degvr(3.0, 5.0, 4.0);
        gen.particles = vec![ParticleType::Neutron, ParticleType::Photon];
        assert!(gen.validate().is_ok());

        // Empty particle list is rejected.
        gen.particles = vec![];
        assert!(gen.validate().is_err());

        // Duplicate particle types are rejected.
        gen.particles = vec![ParticleType::Photon, ParticleType::Photon];
        assert!(gen.validate().is_err());

        // A non-positive photon_energy override is rejected.
        gen.particles = vec![ParticleType::Photon];
        gen.photon_energy = Some(0.0);
        assert!(gen.validate().is_err());
        gen.photon_energy = Some(2.5e6);
        assert!(gen.validate().is_ok());
    }

    #[test]
    fn degvr_build_bounds_stamps_requested_particle() {
        // build_bounds tags the window with the particle passed in, letting one
        // generator emit a window per requested particle type.
        let gen = degvr(3.0, 5.0, 4.0);
        let flux = vec![2.0, 8.0, 4.0, 1.0];
        let n = gen.build_bounds(&flux, ParticleType::Neutron);
        let p = gen.build_bounds(&flux, ParticleType::Photon);
        assert_eq!(n.particle, ParticleType::Neutron);
        assert_eq!(p.particle, ParticleType::Photon);
        // Same flux field -> identical bounds regardless of the tag.
        assert_eq!(n.lower_bounds, p.lower_bounds);
    }

    #[test]
    fn degvr_serde_round_trips_particles_and_photon_energy() {
        let mut gen = degvr(3.0, 5.0, 4.0);
        gen.particles = vec![ParticleType::Neutron, ParticleType::Photon];
        gen.photon_energy = Some(1.5e6);
        let json = serde_json::to_string(&gen).unwrap();
        let back: WeightWindowGeneratorDeGVR = serde_json::from_str(&json).unwrap();
        assert_eq!(back, gen);
        assert_eq!(
            back.particles,
            vec![ParticleType::Neutron, ParticleType::Photon]
        );
        assert_eq!(back.photon_energy, Some(1.5e6));
    }
}
