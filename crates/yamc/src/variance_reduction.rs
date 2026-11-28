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

/// Default absolute weight floor, below which a particle plays roulette.
fn default_weight_floor() -> f64 {
    1e-38
}

/// Default upper/lower window ratio.
fn default_ratio() -> f64 {
    5.0
}

/// DeGVR auto-N slope (mean free paths): above the floor [`DEGVR_N_MIN`],
/// `N = tau / DEGVR_TAU_TARGET` (continuous), so the reduced-density fiducial
/// pass is at most this many mean free paths thick at the source energy.
///
/// `N` trades generation cost against window quality. The histories the
/// fiducial pass needs grow as about `exp(1.2 * span)`, `span` being the
/// e-folds its flux falls across the mesh at density / `N`, so a smaller `N`
/// costs steeply more generation. A larger `N` multiplies the two passes'
/// errors by `N - 2` and `N - 1` in the extrapolated window and makes it
/// shallower at depth; production slows, and in thick concrete it stopped
/// reaching the deep region from about `N = tau / 3`.
pub(crate) const DEGVR_TAU_TARGET: f64 = 7.0;
/// Floor of the auto `N`.
///
/// `examples/python/degvr_density_reduction_study.py` costed generation plus a
/// production run to a 5% geometric-mean error over the lowest-flux quarter of
/// the mesh, in analog-history equivalents, for ten 1D shield cases (six
/// materials, neutron and photon sources, optical depths 10 to 33) and five 3D
/// models. The best `N` was 4 to 5.66 in 22 of the 30 models it could score and
/// did not grow with depth as `tau / 7` does. `max(5, tau / 7)` cost 1.18 times
/// the best `N` of each model on average (1.7 at worst) and `max(2, tau / 7)`
/// 1.39 (3.0). The gain came from the 3D models and the slabs at optical depths
/// of 10.5 and 21.5, where the earlier rule gave `N` of 2 to 3.2; at 32.5 the
/// new rule cost 8% more. A floor of 4 did as well in the slabs but worse in
/// 3D (1.41 against 1.19).
///
/// The floor assumes the default per-pass relative error of 10%, which the
/// extrapolation multiplies by up to `N - 1`. Part of the small-`N` production
/// penalty in 3D is splitting within each history, so revisit the floor if the
/// split control changes. The slope sets `N` above an optical depth of 35,
/// deeper than the study went.
pub(crate) const DEGVR_N_MIN: f64 = 5.0;

/// Default per-pass stop for DeGVR generation: each pass runs until 95% of the
/// window mesh's scored voxels (per energy group) reach this relative error.
/// The window only needs the flux to within its `ratio` (5), and the
/// extrapolation `G = F (H / F)^(N - 1)` scales each pass's log-flux error by
/// about `N - 1`, so a voxel error of 10% keeps the extrapolated flux within
/// a factor of about 2 for `N` up to 5.
pub const DEGVR_RELATIVE_ERROR: f64 = 0.1;

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
/// lower`); one below `weight_floor` plays roulette with the floor as its
/// survivor weight. A negative `lower` entry
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
    /// Absolute weight floor; a particle below it plays roulette with the floor
    /// as its survivor weight. Default 1e-38.
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

    /// The fraction of `geometry` this mesh covers, and the axes it falls short
    /// on, or `None` when it covers the whole thing.
    ///
    /// A window mesh smaller than the geometry is legal and unbiased: a
    /// particle outside it simply gets no window and behaves analog. The
    /// problem is that it is silent, and the symptom (the window "did not
    /// help") looks nothing like the cause. Checked once against the geometry
    /// bounds rather than counted per collision, so it costs nothing on the
    /// transport path and still names a mis-sized mesh before the run rather
    /// than after it.
    ///
    /// The fraction is by volume of the geometry's bounding box, which
    /// overstates coverage for a geometry that does not fill its own box. It is
    /// a prompt to look, not a metric.
    pub fn coverage_of(&self, geometry: &yamc_geo::BoundingBox) -> Option<(f64, Vec<&str>)> {
        let mesh_lo = self.mesh.lower_left();
        let mesh_hi = self.mesh.upper_right();
        let mut short = Vec::new();
        let mut covered = 1.0f64;
        for (axis, name) in ["x", "y", "z"].iter().enumerate().map(|(i, n)| (i, *n)) {
            let geo_lo = geometry.lower_left[axis];
            let geo_hi = geometry.upper_right[axis];
            let span = geo_hi - geo_lo;
            if span <= 0.0 {
                continue;
            }
            let overlap = (mesh_hi[axis].min(geo_hi) - mesh_lo[axis].max(geo_lo)).max(0.0);
            covered *= (overlap / span).min(1.0);
            if mesh_lo[axis] > geo_lo || mesh_hi[axis] < geo_hi {
                short.push(name);
            }
        }
        (!short.is_empty()).then_some((covered, short))
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
/// are built from. Passed to [`crate::model::Model::generate_weight_windows`],
/// which returns one [`WeightWindowBounds`] per requested particle type (in
/// `particles` order).
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
    /// secondary / decay gamma energy (1 MeV). Only
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
    /// `N = max(DEGVR_N_MIN, tau / DEGVR_TAU_TARGET)`. `N` is continuous: the
    /// density scaling (`density / N`, `density / (N/2)`) and the extrapolation
    /// exponent (`N - 1`) are all smooth in `N`, so there is no reason to round
    /// to an integer. An explicit `density_reduction` may go down to 2, which
    /// puts the asymptotic pass at full density.
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
        if let Some(e) = self.photon_energy
            && !(e > 0.0 && e.is_finite())
        {
            return Err(format!(
                "WeightWindowGeneratorDeGVR requires photon_energy > 0 (got {e})"
            ));
        }
        if let Some(n) = self.density_reduction
            && n < 2.0
        {
            return Err(format!(
                "WeightWindowGeneratorDeGVR requires density_reduction >= 2 (got {n})"
            ));
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
    /// sentinel window in [`Self::build_bounds`], so the region is skipped rather
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
    fn coverage_is_silent_when_the_mesh_covers_the_geometry() {
        let w = wwb(vec![0.1; 4], vec![0.5; 4], None);
        let lo = w.mesh.lower_left();
        let hi = w.mesh.upper_right();
        // Exactly coincident, and strictly inside, both count as covered.
        let exact = yamc_geo::BoundingBox::new(lo, hi);
        assert!(w.coverage_of(&exact).is_none());
        let inside = yamc_geo::BoundingBox::new(
            [lo[0] + 0.1, lo[1] + 0.1, lo[2] + 0.1],
            [hi[0] - 0.1, hi[1] - 0.1, hi[2] - 0.1],
        );
        assert!(w.coverage_of(&inside).is_none());
    }

    #[test]
    fn coverage_names_the_axes_a_short_mesh_falls_off() {
        let w = wwb(vec![0.1; 4], vec![0.5; 4], None);
        let lo = w.mesh.lower_left();
        let hi = w.mesh.upper_right();
        // Geometry twice the mesh on x alone: short on x, and half covered.
        let span_x = hi[0] - lo[0];
        let geometry = yamc_geo::BoundingBox::new(lo, [lo[0] + 2.0 * span_x, hi[1], hi[2]]);
        let (covered, short) = w
            .coverage_of(&geometry)
            .expect("a mesh half the geometry is not full coverage");
        assert_eq!(short, vec!["x"]);
        assert!(
            (covered - 0.5).abs() < 1e-12,
            "half the box on one axis is half the volume, got {covered}"
        );
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
        let generator = degvr(3.0, 5.0, 32.0);
        // 4 voxels, single group; voxel 1 is the peak, voxel 3 unscored.
        let flux = vec![2.0, 8.0, 4.0, 0.0];
        let wwb = generator.build_bounds(&flux, ParticleType::Neutron);
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
        use super::{WeightWindowGeneratorDeGVR, DEGVR_N_MIN, DEGVR_TAU_TARGET};
        // A deep problem scales with tau.
        assert_eq!(
            WeightWindowGeneratorDeGVR::n_from_tau(20.0 * DEGVR_TAU_TARGET),
            20.0
        );
        // Shallower problems sit on the floor.
        assert_eq!(
            WeightWindowGeneratorDeGVR::n_from_tau(3.0 * DEGVR_TAU_TARGET),
            DEGVR_N_MIN
        );
        assert_eq!(WeightWindowGeneratorDeGVR::n_from_tau(0.1), DEGVR_N_MIN);
        // N is continuous, not rounded: a fractional multiple stays fractional.
        assert!(
            (WeightWindowGeneratorDeGVR::n_from_tau(9.5 * DEGVR_TAU_TARGET) - 9.5).abs() < 1e-12
        );
    }

    #[test]
    fn degvr_validate_particles_list() {
        // A valid multi-particle generator (neutron + photon) passes.
        let mut generator = degvr(3.0, 5.0, 4.0);
        generator.particles = vec![ParticleType::Neutron, ParticleType::Photon];
        assert!(generator.validate().is_ok());

        // Empty particle list is rejected.
        generator.particles = vec![];
        assert!(generator.validate().is_err());

        // Duplicate particle types are rejected.
        generator.particles = vec![ParticleType::Photon, ParticleType::Photon];
        assert!(generator.validate().is_err());

        // A non-positive photon_energy override is rejected.
        generator.particles = vec![ParticleType::Photon];
        generator.photon_energy = Some(0.0);
        assert!(generator.validate().is_err());
        generator.photon_energy = Some(2.5e6);
        assert!(generator.validate().is_ok());
    }

    #[test]
    fn degvr_build_bounds_stamps_requested_particle() {
        // build_bounds tags the window with the particle passed in, letting one
        // generator emit a window per requested particle type.
        let generator = degvr(3.0, 5.0, 4.0);
        let flux = vec![2.0, 8.0, 4.0, 1.0];
        let n = generator.build_bounds(&flux, ParticleType::Neutron);
        let p = generator.build_bounds(&flux, ParticleType::Photon);
        assert_eq!(n.particle, ParticleType::Neutron);
        assert_eq!(p.particle, ParticleType::Photon);
        // Same flux field -> identical bounds regardless of the tag.
        assert_eq!(n.lower_bounds, p.lower_bounds);
    }

    #[test]
    fn degvr_serde_round_trips_particles_and_photon_energy() {
        let mut generator = degvr(3.0, 5.0, 4.0);
        generator.particles = vec![ParticleType::Neutron, ParticleType::Photon];
        generator.photon_energy = Some(1.5e6);
        let json = serde_json::to_string(&generator).unwrap();
        let back: WeightWindowGeneratorDeGVR = serde_json::from_str(&json).unwrap();
        assert_eq!(back, generator);
        assert_eq!(
            back.particles,
            vec![ParticleType::Neutron, ParticleType::Photon]
        );
        assert_eq!(back.photon_energy, Some(1.5e6));
    }
}
