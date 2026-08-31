// Photon interaction data structures and global element cache

use once_cell::sync::Lazy;
use rand::RngExt;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use yamc_nuclide::reaction_product::Tabulated1D;

// ============================================================================
// SUBSHELL DESIGNATORS
// ============================================================================

/// Subshell designator strings.
/// Index 0 is "K", index 1 is "L1", etc. Used for mapping file designators
/// to 1-based subshell indices.
pub const SUBSHELLS: &[&str] = &[
    "K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5", "N1", "N2", "N3", "N4", "N5", "N6", "N7",
    "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "P1", "P2", "P3", "P4", "P5", "P6", "P7",
    "P8", "P9", "P10", "P11", "Q1", "Q2", "Q3",
];

// ============================================================================
// MICRO CROSS SECTION CACHE (per-element, per-particle scratch)
// ============================================================================

/// Cached microscopic cross sections for a single element at a given energy.
/// Used during transport to avoid repeated interpolation.
pub struct ElementMicroXS {
    pub total: f64,
    pub coherent: f64,
    pub incoherent: f64,
    pub photoelectric: f64,
    pub pair_production: f64,
    pub heating: f64,
    pub last_energy: f64,
    pub index_grid: usize,
    pub interp_factor: f64,
}

/// Mean fraction of the incident photon energy transferred to the
/// Compton electron: sigma_tr / sigma_KN with the standard closed forms
/// (Attix, "Introduction to Radiological Physics", ch. 7; Hubbell 1977).
/// Verified against numerical integration of the Klein-Nishina
/// differential cross section to machine precision from 1 keV to 14 MeV.
/// (The previous inline expressions were wrong: the fraction went
/// NEGATIVE below ~400 keV, issue #358.)
///
/// `alpha` = E / (m_e c^2).
pub(crate) fn compton_energy_transfer_fraction(alpha: f64) -> f64 {
    if alpha < 1e-6 {
        // Low-energy limit: Thomson regime, negligible energy transfer.
        return 0.0;
    }
    let a = alpha;
    let one_2a = 1.0 + 2.0 * a;
    let l = one_2a.ln();
    let sigma_kn = (1.0 + a) / (a * a) * (2.0 * (1.0 + a) / one_2a - l / a) + l / (2.0 * a)
        - (1.0 + 3.0 * a) / (one_2a * one_2a);
    let sigma_tr = 2.0 * (1.0 + a) * (1.0 + a) / (a * a * one_2a)
        - (1.0 + 3.0 * a) / (one_2a * one_2a)
        - (1.0 + a) * (2.0 * a * a - 2.0 * a - 1.0) / (a * a * one_2a * one_2a)
        - 4.0 * a * a / (3.0 * one_2a * one_2a * one_2a)
        - ((1.0 + a) / (a * a * a) - 1.0 / (2.0 * a) + 1.0 / (2.0 * a * a * a)) * l;
    if sigma_kn > 1e-30 {
        sigma_tr / sigma_kn
    } else {
        0.0
    }
}

/// Expected radiative (fluorescence) photon energy emitted per initial vacancy
/// in each subshell, in eV, index-aligned with `shells`.
///
/// This is the expected value (by linearity of expectation over the branching
/// relaxation cascade) of the fluorescence photon energy that
/// [`PhotonInteraction::atomic_relaxation`] banks and transports for a vacancy
/// in shell `s`. It deliberately mirrors that function's behaviour EXACTLY so
/// the deterministic photon-KERMA coefficient equals the expected value of the
/// analog per-collision deposit:
/// - a radiative transition (`secondary_subshell == -1`) contributes its photon
///   `energy` this step, plus the cascade of the primary vacancy;
/// - a non-radiative (Auger) transition contributes no photon this step, plus
///   the cascades of the primary and secondary vacancies;
/// - a shell with no transition table emits its full `binding_energy` as a
///   fluorescence photon (matches the `transitions.is_empty()` fallback in
///   `atomic_relaxation`).
///
/// Transition probabilities are stored as a normalized CDF, so the incremental
/// probability of transition `t` is `t.probability - prev.probability`.
/// Transitions point to less-bound shells (a DAG), so the memoized recursion
/// terminates; a `visiting` guard keeps it safe against malformed data.
///
/// The caller must pass only shells for an element that actually has
/// atomic-relaxation data; when it does not, `atomic_relaxation` banks nothing
/// and the coefficient correction must be zero (handled at the call site).
pub(crate) fn compute_subshell_radiative_energy(shells: &[ElectronSubshell]) -> Vec<f64> {
    fn radiative_energy(
        s: usize,
        shells: &[ElectronSubshell],
        memo: &mut [Option<f64>],
        visiting: &mut [bool],
    ) -> f64 {
        if let Some(v) = memo[s] {
            return v;
        }
        if visiting[s] {
            // Cyclic transition data (should not happen): stop recursing.
            return 0.0;
        }
        visiting[s] = true;

        let shell = &shells[s];
        let value = if shell.transitions.is_empty() {
            // Mirrors atomic_relaxation: the full binding energy leaves as a
            // fluorescence photon when there is no transition table.
            shell.binding_energy
        } else {
            let mut expected = 0.0;
            let mut prev_cdf = 0.0;
            for t in &shell.transitions {
                let dp = (t.probability - prev_cdf).max(0.0);
                prev_cdf = t.probability;

                let mut step = 0.0;
                // The electron that fills the hole leaves a new vacancy in the
                // primary subshell (only recursed when it is a tracked shell).
                if t.primary_subshell >= 0 {
                    step += radiative_energy(t.primary_subshell as usize, shells, memo, visiting);
                }
                if t.secondary_subshell == -1 {
                    // Radiative: a fluorescence photon of `energy` is emitted.
                    step += t.energy;
                } else if t.secondary_subshell >= 0 {
                    // Non-radiative (Auger): a second vacancy, no photon here.
                    step += radiative_energy(t.secondary_subshell as usize, shells, memo, visiting);
                }
                expected += dp * step;
            }
            expected
        };

        visiting[s] = false;
        memo[s] = Some(value);
        value
    }

    let n = shells.len();
    let mut memo = vec![None; n];
    let mut visiting = vec![false; n];
    (0..n)
        .map(|s| radiative_energy(s, shells, &mut memo, &mut visiting))
        .collect()
}

impl Default for ElementMicroXS {
    fn default() -> Self {
        Self {
            total: 0.0,
            coherent: 0.0,
            incoherent: 0.0,
            photoelectric: 0.0,
            pair_production: 0.0,
            heating: 0.0,
            last_energy: 0.0,
            index_grid: 0,
            interp_factor: 0.0,
        }
    }
}

// ============================================================================
// ATOMIC TRANSITION DATA
// ============================================================================

/// A single atomic transition (fluorescence or Auger).
/// Used during atomic relaxation after photoelectric absorption.
#[derive(Debug, Clone)]
pub struct AtomicTransition {
    /// Index into the element's shells vec for the primary (vacancy) subshell,
    /// or -1 if the subshell is not tracked.
    pub primary_subshell: i32,
    /// Index into the element's shells vec for the secondary subshell,
    /// or -1 for radiative transitions (no secondary vacancy).
    pub secondary_subshell: i32,
    /// Transition energy in eV.
    pub energy: f64,
    /// Cumulative probability for this transition (CDF value).
    pub probability: f64,
}

// ============================================================================
// ELECTRON SUBSHELL DATA
// ============================================================================

/// Data for a single electron subshell of an element.
/// Contains binding energy, cross sections, and transition data.
#[derive(Debug, Clone)]
pub struct ElectronSubshell {
    /// SUBSHELL designator index (1-based: 1 = "K", 2 = "L1", ...).
    pub index_subshell: i32,
    /// Binding energy in eV.
    pub binding_energy: f64,
    /// Number of electrons in this subshell.
    pub num_electrons: f64,
    /// Index in the energy grid where the subshell XS becomes nonzero.
    pub threshold: usize,
    /// ln(sigma) values from the threshold index onwards.
    /// Length = energy.len() - threshold.
    pub cross_section: Vec<f64>,
    /// Atomic transitions originating from a vacancy in this subshell.
    /// Probabilities are stored as cumulative (CDF).
    pub transitions: Vec<AtomicTransition>,
}

/// One constituent atomic-relaxation subshell of a Compton-profile shell, with
/// its occupancy weight.
///
/// A Compton-profile shell is a Biggs Hartree-Fock (n, l) shell; the atomic
/// relaxation data uses (n, l, j) subshells, so a single Compton shell maps to
/// one or two relaxation subshells (the j = l +/- 1/2 split). When a Compton
/// event ionizes the (n, l) shell, the vacancy lands in one of the constituent
/// subshells with probability `weight` (its occupancy fraction). The mapping is
/// computed at data-generation time by occupancy grouping and shipped in
/// compton.arrow, so the load path no longer guesses it from binding energies.
#[derive(Debug, Clone)]
pub struct ComptonRelaxTarget {
    /// Index of the relaxation subshell in [`PhotonInteraction::shells`].
    pub shell_index: usize,
    /// Occupancy weight (constituent electrons / Compton-shell electrons); the
    /// weights of a Compton shell's targets sum to 1.
    pub weight: f64,
}

// ============================================================================
// PHOTON INTERACTION (main element data)
// ============================================================================

/// Complete photon interaction data for a single element.
///
/// All cross sections and the energy grid are stored in log form:
/// - energy[i] = ln(E_i)
/// - xs[i] = ln(sigma_i) if sigma > exp(-499), else -900.0
#[derive(Debug, Clone)]
pub struct PhotonInteraction {
    /// Element name (e.g., "Fe").
    pub name: String,
    /// Index in the global ELEMENTS vector.
    pub index: usize,
    /// Atomic number Z.
    pub atomic_number: u32,

    // ------------------------------------------------------------------
    // Energy grid stored as ln(E) in eV
    // ------------------------------------------------------------------
    pub energy: Vec<f64>,

    // ------------------------------------------------------------------
    // Cross sections stored as ln(sigma) in barns
    // Values below exp(-499) are stored as -900.0
    // ------------------------------------------------------------------
    pub coherent_xs: Vec<f64>,
    pub incoherent_xs: Vec<f64>,
    pub photoelectric_total_xs: Vec<f64>,
    pub pair_production_total_xs: Vec<f64>,
    pub pair_production_nuclear_xs: Vec<f64>,
    pub pair_production_electron_xs: Vec<f64>,
    pub heating_xs: Vec<f64>,

    // ------------------------------------------------------------------
    // Form factors (Tabulated1D from reaction_product)
    // ------------------------------------------------------------------
    /// Integrated coherent scattering form factor F(x, Z).
    pub coherent_int_form_factor: Tabulated1D,
    /// Incoherent scattering factor S(x, Z).
    pub incoherent_form_factor: Tabulated1D,

    // ------------------------------------------------------------------
    // Compton profile data
    // ------------------------------------------------------------------
    /// Number of electrons per Compton profile shell (from compton_profiles/num_electrons).
    pub electron_pdf: Vec<f64>,
    /// Binding energy per Compton profile shell in eV (from compton_profiles/binding_energy).
    pub binding_energy: Vec<f64>,
    /// Compton profile PDF J(pz) for each shell. profile_pdf[shell][pz_index].
    pub profile_pdf: Vec<Vec<f64>>,
    /// Compton profile CDF (computed by trapezoidal integration). profile_cdf[shell][pz_index].
    pub profile_cdf: Vec<Vec<f64>>,

    // ------------------------------------------------------------------
    // Subshell data
    // ------------------------------------------------------------------
    /// Electron subshell data (photoelectric subshells with XS and transitions).
    pub shells: Vec<ElectronSubshell>,

    /// 2D cross section storage for photoelectric subshell XS.
    /// cross_sections[i_grid][i_shell] = ln(sigma) at energy grid index i_grid for shell i_shell.
    /// For grid indices below a shell's threshold, the value is 0.0.
    pub cross_sections: Vec<Vec<f64>>,

    /// Expected radiative (fluorescence) photon energy emitted per initial
    /// vacancy in each subshell, in eV, index-aligned with `shells` and the
    /// `cross_sections` columns. This is the expected value (over the atomic
    /// relaxation cascade) of the fluorescence photon energy that
    /// [`PhotonInteraction::atomic_relaxation`] banks and transports as separate
    /// photons. Subtracting `sum_s sigma_PE_s(E) * subshell_radiative_energy[s]`
    /// from the full-energy photoelectric heating term yields the energy actually
    /// transferred to local charged particles (photoelectron + Auger), matching
    /// the OpenMC / NIST mu_tr convention and avoiding a double-count of the
    /// separately-transported fluorescence energy. Empty (or all-zero) when the
    /// element has no atomic-relaxation data. See
    /// [`compute_subshell_radiative_energy`].
    pub subshell_radiative_energy: Vec<f64>,

    /// Expected radiative (fluorescence) photon energy emitted per Compton
    /// (incoherent) scattering event, in eV: `sum_c electron_pdf[c] *
    /// sum_k w[c][k] * subshell_radiative_energy[compton_relax_map[c][k]]` over
    /// Compton-profile shells `c` and their constituent relaxation subshells `k`
    /// (occupancy weights `w`). A Compton event ionizes the shell sampled from
    /// `electron_pdf` (Doppler broadening) and then relaxes, banking +
    /// transporting that fluorescence, so the local Compton heating is
    /// `E*f_KN - R_compton`; subtracting `incoherent(E) * compton_radiative_energy`
    /// from the Compton heating term avoids double-counting that transported
    /// fluorescence (the Compton analogue of `subshell_radiative_energy`).
    /// Energy-independent because `electron_pdf`/`compton_relax_map` are, and
    /// because the transport relaxation guard is defeated by the `+ binding_energy`
    /// energy argument so every Compton event banks the full cascade. Zero when
    /// there is no Compton-profile or relaxation data.
    pub compton_radiative_energy: f64,

    /// Mapping from Compton-profile shell index to its constituent atomic
    /// relaxation subshells (with occupancy weights). Computed at data-generation
    /// time by occupancy grouping (see [`ComptonRelaxTarget`]) and read from
    /// compton.arrow. An empty entry means the Compton shell has no relaxation
    /// counterpart (outer/valence shells), so it banks no fluorescence.
    pub compton_relax_map: Vec<Vec<ComptonRelaxTarget>>,

    /// Whether atomic relaxation (transition) data is present.
    pub has_atomic_relaxation: bool,

    // ------------------------------------------------------------------
    // Bremsstrahlung data (for TTB electron treatment)
    // ------------------------------------------------------------------
    /// Scaled bremsstrahlung DCS: dcs[i_e][i_k] where i_e indexes electron
    /// energy and i_k indexes reduced photon energy (k = E_photon / E_electron).
    pub dcs: Vec<Vec<f64>>,
    /// Radiative stopping power S_rad(E) for this element, one value per
    /// TTB electron energy grid point.
    pub stopping_power_radiative: Vec<f64>,
    /// Per-shell ionization energies in eV (from bremsstrahlung group).
    pub ionization_energy: Vec<f64>,
    /// Per-shell electron counts (negative values indicate conduction electrons).
    pub n_electrons: Vec<f64>,
    /// Mean excitation energy I in eV (attribute from bremsstrahlung group).
    pub mean_excitation_energy: f64,
    /// TTB electron energy grid in LINEAR eV (from the Arrow data, same for all elements).
    pub ttb_electron_energy: Vec<f64>,
    /// TTB reduced photon energy grid k = E_photon / E_electron (same for all elements).
    pub ttb_photon_energy: Vec<f64>,
}

// ============================================================================
// PHYSICAL CONSTANTS
// ============================================================================

/// Electron mass energy equivalent in eV/c^2.
pub const MASS_ELECTRON_EV: f64 = 0.51099895000e6;

/// Inverse fine structure constant (dimensionless).
pub(crate) const FINE_STRUCTURE: f64 = 137.035999084;

/// Planck's constant times c in eV-Angstroms.
pub(crate) const PLANCK_C: f64 = 1.2398419839593942e4;

/// Maximum stack depth for atomic relaxation cascade.
const MAX_STACK_SIZE: usize = 7;

/// Reduced screening radii for pair production, indexed by atomic number Z.
/// Index 0 is unused (Z=0 does not exist); indices 1–98 correspond to Z=1–98.
/// From PENELOPE-2011 (Salvat, Fernandez-Varea, Sempau).
pub const REDUCED_SCREENING_RADII: [f64; 99] = [
    122.81, 73.167, 69.228, 67.301, 64.696, 61.228, 57.524, 54.033, 50.787, 47.851, 46.373, 45.401,
    44.503, 43.815, 43.074, 42.321, 41.586, 40.953, 40.524, 40.256, 39.756, 39.144, 38.462, 37.778,
    37.174, 36.663, 35.986, 35.317, 34.688, 34.197, 33.786, 33.422, 33.068, 32.740, 32.438, 32.143,
    31.884, 31.622, 31.438, 31.142, 30.950, 30.758, 30.561, 30.285, 30.097, 29.832, 29.581, 29.411,
    29.247, 29.085, 28.930, 28.721, 28.580, 28.442, 28.312, 28.139, 27.973, 27.819, 27.675, 27.496,
    27.285, 27.093, 26.911, 26.705, 26.516, 26.304, 26.108, 25.929, 25.730, 25.577, 25.403, 25.245,
    25.100, 24.941, 24.790, 24.655, 24.506, 24.391, 24.262, 24.145, 24.039, 23.922, 23.813, 23.712,
    23.621, 23.523, 23.430, 23.331, 23.238, 23.139, 23.048, 22.967, 22.833, 22.694, 22.624, 22.545,
    22.446, 22.358, 22.264,
];

// ============================================================================
// PHYSICS: helper functions
// ============================================================================

/// Generate an isotropic random unit direction vector.
/// Returns `[ux, uy, uz]`.
pub fn isotropic_direction<R: rand::Rng + ?Sized>(rng: &mut R) -> [f64; 3] {
    let mu: f64 = rng.random_range(-1.0..1.0);
    let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
    let sin_theta = (1.0 - mu * mu).sqrt();
    [sin_theta * phi.cos(), sin_theta * phi.sin(), mu]
}

// ============================================================================
// PHYSICS: Klein-Nishina sampling (free function)
// ============================================================================

/// Sample outgoing photon energy and scattering cosine from Klein-Nishina.
///
/// Uses Kahn's rejection method when `alpha < 3` and Koblinger's direct method
/// when `alpha >= 3`.
///
/// # Arguments
/// * `alpha` - Ratio of photon energy to electron rest mass: E / m_e c^2
/// * `rng` - Random number generator
///
/// # Returns
/// `(alpha_out, mu)` where `alpha_out` is the outgoing energy ratio and `mu`
/// is the cosine of the scattering angle.
pub fn klein_nishina<R: rand::Rng + ?Sized>(alpha: f64, rng: &mut R) -> (f64, f64) {
    let beta = 1.0 + 2.0 * alpha;

    if alpha < 3.0 {
        // Kahn's rejection method
        let t = beta / (beta + 8.0);
        loop {
            if rng.random::<f64>() < t {
                // Left branch
                let r: f64 = rng.random_range(0.0..2.0);
                let x = 1.0 + alpha * r;
                if rng.random::<f64>() < 4.0 / x * (1.0 - 1.0 / x) {
                    let mu = 1.0 - r;
                    let alpha_out = alpha / x;
                    return (alpha_out, mu);
                }
            } else {
                // Right branch
                let x = beta / (1.0 + 2.0 * alpha * rng.random::<f64>());
                let mu = 1.0 + (1.0 - x) / alpha;
                if rng.random::<f64>() < 0.5 * (mu * mu + 1.0 / x) {
                    let alpha_out = alpha / x;
                    return (alpha_out, mu);
                }
            }
        }
    } else {
        // Koblinger's direct method
        let gamma = 1.0 - beta.powi(-2);
        let s = rng.random::<f64>()
            * (4.0 / alpha + 0.5 * gamma + (1.0 - (1.0 + beta) / (alpha * alpha)) * beta.ln());

        let alpha_out = if s <= 2.0 / alpha {
            alpha / (1.0 + 2.0 * alpha * rng.random::<f64>())
        } else if s <= 4.0 / alpha {
            alpha * (1.0 + 2.0 * alpha * rng.random::<f64>()) / beta
        } else if s <= 4.0 / alpha + 0.5 * gamma {
            alpha * (1.0 - gamma * rng.random::<f64>()).sqrt()
        } else {
            alpha / beta.powf(rng.random::<f64>())
        };

        let mu = 1.0 + 1.0 / alpha - 1.0 / alpha_out;
        (alpha_out, mu)
    }
}

// ============================================================================
// PHYSICS: microscopic cross-section calculation
// ============================================================================

impl PhotonInteraction {
    /// Compute microscopic cross sections at the given energy via log-log interpolation.
    ///
    /// Uses binary search on the ln(E) grid, then linear interpolation in log space.
    /// Photoelectric XS is summed over subshells (not from a stored total).
    pub fn calculate_xs(&self, energy: f64) -> ElementMicroXS {
        let n_grid = self.energy.len();
        let log_e = energy.ln();

        // Binary search on ln(E) grid
        let i_grid = if log_e <= self.energy[0] {
            0
        } else if log_e >= self.energy[n_grid - 1] {
            n_grid - 2
        } else {
            // upper_bound - 1: find rightmost index where energy[i] <= log_e
            match self.energy.partition_point(|&e| e <= log_e) {
                0 => 0,
                i => i - 1,
            }
        };

        // Handle case where two energy points are the same
        let i_grid = if i_grid + 1 < n_grid && self.energy[i_grid] == self.energy[i_grid + 1] {
            (i_grid + 1).min(n_grid - 2)
        } else {
            i_grid
        };

        // Interpolation factor
        let denom = self.energy[i_grid + 1] - self.energy[i_grid];
        let f = if denom.abs() > 1e-30 {
            (log_e - self.energy[i_grid]) / denom
        } else {
            0.0
        };

        // Coherent XS (log-log interpolation)
        let coherent = (self.coherent_xs[i_grid]
            + f * (self.coherent_xs[i_grid + 1] - self.coherent_xs[i_grid]))
            .exp();

        // Incoherent XS
        let incoherent = (self.incoherent_xs[i_grid]
            + f * (self.incoherent_xs[i_grid + 1] - self.incoherent_xs[i_grid]))
            .exp();

        // Photoelectric XS: sum over subshells. In the same pass accumulate the
        // fluorescence-radiative deficit `sum_s sigma_PE_s * R_s`, the mean energy
        // that leaves a photoelectric absorption as separately-transported
        // fluorescence photons rather than as local charged-particle KE (used by
        // the heating coefficient below).
        let mut photoelectric = 0.0;
        let mut photoelectric_radiative_deficit = 0.0;
        let xs_lower = &self.cross_sections[i_grid];
        let xs_upper = &self.cross_sections[i_grid + 1];
        for i in 0..xs_upper.len() {
            if xs_lower[i] != 0.0 {
                let sigma_i = (xs_lower[i] + f * (xs_upper[i] - xs_lower[i])).exp();
                photoelectric += sigma_i;
                photoelectric_radiative_deficit += sigma_i
                    * self
                        .subshell_radiative_energy
                        .get(i)
                        .copied()
                        .unwrap_or(0.0);
            }
        }

        // Pair production XS
        let pair_production = (self.pair_production_total_xs[i_grid]
            + f * (self.pair_production_total_xs[i_grid + 1]
                - self.pair_production_total_xs[i_grid]))
            .exp();

        // Heating KERMA (eV·barn): average energy deposited locally in charged
        // particles per unit flux.  Since yamc does not transport electrons or
        // positrons, all charged-particle kinetic energy is deposited locally.
        //
        // Use tabulated heating XS from the Arrow data when available (computed by
        // NJOY HEATR); otherwise fall back to physics-based calculation.
        let heating_from_table = (self.heating_xs[i_grid]
            + f * (self.heating_xs[i_grid + 1] - self.heating_xs[i_grid]))
            .exp();
        let heating = if heating_from_table > 0.0 {
            heating_from_table
        } else {
            // Compton (incoherent): average electron recoil energy from the
            // Klein-Nishina formula.  α = E / (m_e c²).
            // Average scattered photon energy:
            //   <E'> / E = [1 + α(α-2-2/α)·ln(1+2α) + 2α/(1+2α)
            //               + α²(1+2α)/(2(1+2α)²) ]  ... complex.
            // Instead, use the exact Klein-Nishina energy-transfer coefficient:
            //   σ_tr = σ_KN * f(α),  where f(α) gives the fraction of energy
            //   transferred to the electron.
            let alpha = energy / MASS_ELECTRON_EV;

            // Klein-Nishina average fractional energy transfer to the electron.
            // Derived from integrating E_electron * dσ/dΩ over all angles,
            // divided by σ_KN * E.  Valid for free electrons; binding effects
            // are small for the energies where Compton dominates.
            let compton_frac = compton_energy_transfer_fraction(alpha);

            // Subtract the fluorescence banked when the Compton event ionizes a
            // bound shell (sampled from electron_pdf, then relaxed + transported),
            // so this coefficient equals the expected local charged-particle
            // deposit `E*f_KN - R_compton` per event rather than double-counting
            // that separately-transported fluorescence. Mirrors the photoelectric
            // correction above; `compton_radiative_energy` is 0 without relaxation
            // data.
            let compton_radiative_deficit = incoherent * self.compton_radiative_energy;
            let compton_heating =
                (energy * compton_frac * incoherent - compton_radiative_deficit).max(0.0);

            // Photoelectric: energy transferred to LOCAL charged particles
            // (photoelectron + Auger electrons). The fluorescence x-rays emitted
            // during atomic relaxation are banked and transported as separate
            // photons, so their energy (`photoelectric_radiative_deficit`) is
            // subtracted here to avoid double-counting it (once at the absorption
            // site, again when the fluorescence photon is later reabsorbed). This
            // is the NIST mu_tr / OpenMC (E - banked-photon-energy) convention and
            // makes this track-length KERMA coefficient equal the expected value
            // of the analog per-collision deposit. The deficit is zero when the
            // element carries no atomic-relaxation data (nothing is banked).
            let photo_heating = (energy * photoelectric - photoelectric_radiative_deficit).max(0.0);

            // Pair production: kinetic energy of e+/e− pair.
            let pair_heating = (energy - 2.0 * MASS_ELECTRON_EV).max(0.0) * pair_production;

            // Coherent: zero (elastic, no energy transfer).
            compton_heating + photo_heating + pair_heating
        };

        // Total = sum of all components
        let total = coherent + incoherent + photoelectric + pair_production;

        ElementMicroXS {
            total,
            coherent,
            incoherent,
            photoelectric,
            pair_production,
            heating,
            last_energy: energy,
            index_grid: i_grid,
            interp_factor: f,
        }
    }

    /// Rayleigh (coherent) scattering -- sample scattering cosine mu.
    ///
    /// Rejection sampling with the integrated coherent form factor F²(x, Z).
    /// The momentum transfer parameter x = (E / hc) * sqrt((1 - mu) / 2).
    /// No energy change (elastic).
    ///
    /// # Arguments
    /// * `alpha` - Ratio of photon energy to electron rest mass: E / m_e c^2
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Cosine of the scattering angle, mu in [-1, 1].
    pub fn rayleigh_scatter<R: rand::Rng + ?Sized>(&self, alpha: f64, rng: &mut R) -> f64 {
        let Tabulated1D::Tabulated1D { x, y, .. } = &self.coherent_int_form_factor;

        loop {
            // Maximum value of x^2
            let x2_max = (MASS_ELECTRON_EV / PLANCK_C * alpha).powi(2);

            // F(x^2_max, Z) -- evaluate the integrated form factor at x^2_max
            let f_max = self.coherent_int_form_factor.evaluate(x2_max);

            // Sample cumulative distribution
            let f_sample = rng.random::<f64>() * f_max;

            // Determine x^2 corresponding to F by inverse CDF lookup on y->x
            // y values are the CDF (integrated form factor), x values are x^2
            let i = match y.partition_point(|&val| val < f_sample) {
                0 => 0,
                i => (i - 1).min(y.len() - 2),
            };

            let denom = y[i + 1] - y[i];
            let r = if denom.abs() > 1e-30 {
                (f_sample - y[i]) / denom
            } else {
                0.0
            };
            let x2 = x[i] + r * (x[i + 1] - x[i]);

            // Calculate mu from x^2
            let mu = 1.0 - 2.0 * x2 / x2_max;

            // Rejection: accept with probability 0.5 * (1 + mu^2)
            if rng.random::<f64>() < 0.5 * (1.0 + mu * mu) {
                return mu;
            }
        }
    }

    /// Compton (incoherent) scattering -- sample outgoing energy and angle.
    ///
    /// Uses Klein-Nishina with rejection on the incoherent scattering factor
    /// S(x, Z). When `doppler=true`, applies Compton Doppler broadening using
    /// electron momentum profiles.
    ///
    /// # Arguments
    /// * `alpha` - Ratio of photon energy to electron rest mass: E / m_e c^2
    /// * `doppler` - Whether to apply Doppler broadening
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// `(alpha_out, mu, i_shell)` where:
    /// - `alpha_out` is the outgoing energy / m_e c^2
    /// - `mu` is the cosine of scattering angle
    /// - `i_shell` is the Compton profile shell index (-1 when doppler=false)
    pub fn compton_scatter<R: rand::Rng + ?Sized>(
        &self,
        alpha: f64,
        doppler: bool,
        rng: &mut R,
    ) -> (f64, f64, i32) {
        let mut form_factor_xmax = 0.0_f64;

        loop {
            // Sample Klein-Nishina for trial energy and angle
            let (alpha_out, mu) = klein_nishina(alpha, rng);

            // Momentum transfer parameter (Hubbell definition)
            let x = MASS_ELECTRON_EV / PLANCK_C * alpha * (0.5 * (1.0 - mu)).sqrt();

            // Calculate S(x, Z) and S(x_max, Z)
            let form_factor_x = self.incoherent_form_factor.evaluate(x);
            if form_factor_xmax == 0.0 {
                form_factor_xmax = self
                    .incoherent_form_factor
                    .evaluate(MASS_ELECTRON_EV / PLANCK_C * alpha);
            }

            // Rejection on form factor
            if form_factor_xmax > 0.0 && rng.random::<f64>() < form_factor_x / form_factor_xmax {
                if doppler && !self.electron_pdf.is_empty() {
                    // Apply Compton Doppler broadening
                    let (e_out, i_shell) = self.compton_doppler(alpha, mu, rng);
                    let alpha_out_doppler = e_out / MASS_ELECTRON_EV;
                    return (alpha_out_doppler, mu, i_shell);
                }
                // No Doppler broadening
                return (alpha_out, mu, -1);
            }
        }
    }

    /// Compton Doppler broadening -- sample outgoing photon energy from electron
    /// momentum distribution.
    ///
    /// Uses rejection sampling over Compton profile shells to account for
    /// electron binding energies and momentum distributions.
    ///
    ///
    /// # Arguments
    /// * `alpha` - Ratio of photon energy to electron rest mass: E / m_e c^2
    /// * `mu` - Cosine of scattering angle (from Klein-Nishina)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// `(E_out, i_shell)` where E_out is the outgoing photon energy in eV
    /// and i_shell is the Compton profile shell index.
    /// Public for GPU cross-validation (`yamc-gpu` tests compare the GPU
    /// Doppler sampler against this). Given an incident `alpha` and a sampled
    /// `mu`, returns the Doppler-broadened `(E_out, shell_index)`.
    pub fn compton_doppler<R: rand::Rng + ?Sized>(
        &self,
        alpha: f64,
        mu: f64,
        rng: &mut R,
    ) -> (f64, i32) {
        let e_in = alpha * MASS_ELECTRON_EV;
        let pz = compton_profile_pz();
        let n = pz.len();

        // Klein-Nishina free-electron result as fallback
        let e_out_kn = alpha / (1.0 + alpha * (1.0 - mu)) * MASS_ELECTRON_EV;

        let mut i_shell: usize;

        loop {
            // Sample shell from electron_pdf (cumulative sum sampling)
            let xi = rng.random::<f64>();
            let mut cumsum = 0.0;
            i_shell = 0;
            for (i, &pdf) in self.electron_pdf.iter().enumerate() {
                cumsum += pdf;
                if cumsum > xi {
                    i_shell = i;
                    break;
                }
                i_shell = i;
            }

            // Get binding energy for this shell
            let e_b = self.binding_energy[i_shell];

            // If photon energy < binding energy, use free-electron result
            if e_in < e_b {
                return (e_out_kn, i_shell as i32);
            }

            // Determine pz_max
            let pz_max = -FINE_STRUCTURE * (e_b - (e_in - e_b) * alpha * (1.0 - mu))
                / (2.0 * e_in * (e_in - e_b) * (1.0 - mu) + e_b * e_b).sqrt();

            if pz_max < 0.0 {
                return (e_out_kn, i_shell as i32);
            }

            // Determine profile CDF value at pz_max
            let profile = &self.profile_pdf[i_shell];
            let cdf = &self.profile_cdf[i_shell];

            let c_max = if pz_max > pz[n - 1] {
                cdf[n - 1]
            } else {
                // lower_bound_index: last index where pz[i] <= pz_max
                // Clamp to n-2 so that i_pz+1 is always valid
                let i_pz = match pz.partition_point(|&v| v <= pz_max) {
                    0 => 0,
                    i => (i - 1).min(n - 2),
                };

                let pz_l = pz[i_pz];
                let pz_r = pz[i_pz + 1];
                let p_l = profile[i_pz];
                let p_r = profile[i_pz + 1];
                let c_l = cdf[i_pz];

                if pz_l == pz_r {
                    c_l
                } else if p_l == p_r {
                    c_l + (pz_max - pz_l) * p_l
                } else {
                    let m = (p_l - p_r) / (pz_l - pz_r);
                    c_l + ((m * (pz_max - pz_l) + p_l).powi(2) - p_l * p_l) / (2.0 * m)
                }
            };

            if c_max <= 0.0 {
                return (e_out_kn, i_shell as i32);
            }

            // Sample c uniformly in [0, c_max]
            let c = rng.random::<f64>() * c_max;

            // Find interval in CDF containing c (lower_bound_index)
            // Clamp to n-2 so that i_c+1 is always valid
            let i_c = match cdf.partition_point(|&v| v <= c) {
                0 => 0,
                i => (i - 1).min(n - 2),
            };

            // Invert CDF to get pz_sample
            let pz_l = pz[i_c];
            let pz_r = pz[i_c + 1];
            let p_l = profile[i_c];
            let p_r = profile[i_c + 1];
            let c_l = cdf[i_c];

            let pz_sample = if pz_l == pz_r {
                pz_l
            } else if p_l == p_r {
                pz_l + (c - c_l) / p_l
            } else {
                let m = (p_l - p_r) / (pz_l - pz_r);
                pz_l + ((p_l * p_l + 2.0 * m * (c - c_l)).sqrt() - p_l) / m
            };

            // Solve quadratic for outgoing photon energy
            let momentum_sq = (pz_sample / FINE_STRUCTURE).powi(2);
            let f_val = 1.0 + alpha * (1.0 - mu);

            let a_coeff = momentum_sq - f_val * f_val;
            let b_coeff = 2.0 * e_in * (f_val - momentum_sq * mu);
            let c_coeff = e_in * e_in * (momentum_sq - 1.0);

            let quad = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
            if quad < 0.0 {
                // No real solution -- return KN result
                return (e_out_kn, i_shell as i32);
            }
            let sqrt_quad = quad.sqrt();

            let e_out_1 = -(b_coeff + sqrt_quad) / (2.0 * a_coeff);
            let e_out_2 = -(b_coeff - sqrt_quad) / (2.0 * a_coeff);

            // Determine positive solution
            let e_out = if e_out_1 > 0.0 {
                if e_out_2 > 0.0 {
                    // Both positive -- pick one at random
                    if rng.random::<f64>() < 0.5 {
                        e_out_1
                    } else {
                        e_out_2
                    }
                } else {
                    e_out_1
                }
            } else if e_out_2 > 0.0 {
                e_out_2
            } else {
                // No positive solution -- resample
                continue;
            };

            // Accept if E_out < E_in - E_b
            if e_out < e_in - e_b {
                return (e_out, i_shell as i32);
            }
            // Otherwise resample
        }
    }

    // ====================================================================
    // PR 5: Photoelectric Effect and Pair Production
    // ====================================================================

    /// Sample the subshell absorbing a photon in the photoelectric effect.
    ///
    /// Uses the cached grid index and interpolation factor from `micro_xs` to
    /// interpolate each subshell's photoionization cross section, then samples
    /// proportional to the subshell XS via cumulative comparison.
    ///
    ///
    /// # Returns
    /// Index into `self.shells` for the sampled subshell.
    pub fn sample_photoelectric_subshell<R: rand::Rng + ?Sized>(
        &self,
        micro_xs: &ElementMicroXS,
        rng: &mut R,
    ) -> usize {
        let i_grid = micro_xs.index_grid;
        let f = micro_xs.interp_factor;
        let cutoff = rng.random::<f64>() * micro_xs.photoelectric;

        let xs_lower = &self.cross_sections[i_grid];
        let xs_upper = &self.cross_sections[i_grid + 1];

        let mut prob = 0.0;
        for i_shell in 0..self.shells.len() {
            if xs_lower[i_shell] == 0.0 {
                continue;
            }
            prob += (xs_lower[i_shell] + f * (xs_upper[i_shell] - xs_lower[i_shell])).exp();
            if prob > cutoff {
                return i_shell;
            }
        }

        // Fallback: return last shell with nonzero XS
        self.shells.len() - 1
    }

    /// Atomic relaxation cascade after a vacancy is created in a subshell.
    ///
    /// Uses a stack-based approach to
    /// process vacancies. Each transition either produces a fluorescent X-ray
    /// photon (radiative, `secondary_subshell == -1`) or an Auger electron
    /// (non-radiative). Transition probabilities are stored as a cumulative
    /// distribution (CDF).
    ///
    /// # Arguments
    /// * `shell_idx` - Index into `self.shells` for the initial vacancy
    /// * `energy` - Incident photon energy in eV (used for threshold check)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Vec of `(energy, direction, is_photon)` for each secondary particle.
    pub fn atomic_relaxation<R: rand::Rng + ?Sized>(
        &self,
        shell_idx: usize,
        energy: f64,
        rng: &mut R,
    ) -> Vec<(f64, [f64; 3], bool)> {
        let mut secondaries = Vec::new();

        if !self.has_atomic_relaxation || self.shells[shell_idx].binding_energy > energy {
            return secondaries;
        }

        let mut holes: [usize; MAX_STACK_SIZE] = [0; MAX_STACK_SIZE];
        let mut n_holes: usize = 0;

        // Push initial vacancy
        holes[n_holes] = shell_idx;
        n_holes += 1;

        while n_holes > 0 {
            n_holes -= 1;
            let i_hole = holes[n_holes];
            let shell = &self.shells[i_hole];

            if shell.transitions.is_empty() {
                // No transitions -- assume fluorescent photon from captured free
                // electron.
                let dir = isotropic_direction(rng);
                secondaries.push((shell.binding_energy, dir, true));
                continue;
            }

            // Sample transition using CDF (our probabilities are cumulative)
            let r: f64 = rng.random();
            let i_trans = shell
                .transitions
                .iter()
                .position(|t| t.probability > r)
                .unwrap_or(shell.transitions.len() - 1);
            let transition = &shell.transitions[i_trans];

            let dir = isotropic_direction(rng);

            // Push primary vacancy (electron that filled the hole)
            if transition.primary_subshell >= 0 && n_holes < MAX_STACK_SIZE {
                holes[n_holes] = transition.primary_subshell as usize;
                n_holes += 1;
            }

            if transition.secondary_subshell != -1 {
                // Non-radiative (Auger/Coster-Kronig): emit electron
                if transition.secondary_subshell >= 0 && n_holes < MAX_STACK_SIZE {
                    holes[n_holes] = transition.secondary_subshell as usize;
                    n_holes += 1;
                }
                secondaries.push((transition.energy, dir, false));
            } else {
                // Radiative: emit fluorescent photon
                secondaries.push((transition.energy, dir, true));
            }
        }

        secondaries
    }

    /// Pair production -- sample electron and positron energies and angles.
    ///
    /// Full detailed pair production using reduced screening radii and the
    /// composition-rejection method from PENELOPE-2011.
    ///
    ///
    /// # Arguments
    /// * `alpha` - Ratio of photon energy to electron rest mass: E / m_e c^2
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// `(e_electron, e_positron, mu_electron, mu_positron)` where energies are
    /// in eV and mu values are cosines of scattering angles relative to the
    /// incident photon direction.
    pub fn pair_production<R: rand::Rng + ?Sized>(
        &self,
        alpha: f64,
        rng: &mut R,
    ) -> (f64, f64, f64, f64) {
        let z = self.atomic_number as usize;
        let r_z = REDUCED_SCREENING_RADII[z];

        // High-energy Coulomb correction
        let a = z as f64 / FINE_STRUCTURE;
        let c = a
            * a
            * (1.0 / (1.0 + a * a)
                + 0.202059
                + a * a
                    * (-0.03693
                        + a * a
                            * (0.00835
                                + a * a
                                    * (-0.00201
                                        + a * a
                                            * (0.00049 + a * a * (-0.00012 + a * a * 0.00003))))));

        // Low-energy correction factor
        let q = (2.0 / alpha).sqrt();
        let f_corr = q * (-0.1774 - 12.10 * a + 11.18 * a * a)
            + q * q * (8.523 + 73.26 * a - 44.41 * a * a)
            + q * q * q * (-13.52 - 121.1 * a + 96.41 * a * a)
            + q * q * q * q * (8.946 + 62.05 * a - 63.41 * a * a);

        // Calculate phi_1(1/2) and phi_2(1/2)
        let b = 2.0 * r_z / alpha;
        let t1 = 2.0 * (1.0 + b * b).ln();
        let t2 = b * (1.0 / b).atan();
        let t3 = b * b * (4.0 - 4.0 * t2 - 3.0 * (1.0 + 1.0 / (b * b)).ln());
        let t4 = 4.0 * r_z.ln() - 4.0 * c + f_corr;
        let phi1_max = 7.0 / 3.0 - t1 - 6.0 * t2 - t3 + t4;
        let phi2_max = 11.0 / 6.0 - t1 - 3.0 * t2 + 0.5 * t3 + t4;

        // Composition-rejection sampling for reduced energy e
        let u1 = 2.0 / 3.0 * (0.5 - 1.0 / alpha).powi(2) * phi1_max;
        let u2 = phi2_max;

        let e = loop {
            let rn: f64 = rng.random();

            let (i, e_trial) = if rng.random::<f64>() < u1 / (u1 + u2) {
                // Sample from pi_1 using inverse transform
                let e = if rn >= 0.5 {
                    0.5 + (0.5 - 1.0 / alpha) * (2.0 * rn - 1.0).cbrt()
                } else {
                    0.5 - (0.5 - 1.0 / alpha) * (1.0 - 2.0 * rn).cbrt()
                };
                (1, e)
            } else {
                // Sample from pi_2 using inverse transform
                let e = 1.0 / alpha + (0.5 - 1.0 / alpha) * 2.0 * rn;
                (2, e)
            };

            // Calculate phi_i(e) and accept/reject
            let b = r_z / (2.0 * alpha * e_trial * (1.0 - e_trial));
            let t1 = 2.0 * (1.0 + b * b).ln();
            let t2 = b * (1.0 / b).atan();
            let t3 = b * b * (4.0 - 4.0 * t2 - 3.0 * (1.0 + 1.0 / (b * b)).ln());

            if i == 1 {
                let phi1 = 7.0 / 3.0 - t1 - 6.0 * t2 - t3 + t4;
                if rng.random::<f64>() <= phi1 / phi1_max {
                    break e_trial;
                }
            } else {
                let phi2 = 11.0 / 6.0 - t1 - 3.0 * t2 + 0.5 * t3 + t4;
                if rng.random::<f64>() <= phi2 / phi2_max {
                    break e_trial;
                }
            }
        };

        // Kinetic energies of electron and positron
        let e_electron = (alpha * e - 1.0) * MASS_ELECTRON_EV;
        let e_positron = (alpha * (1.0 - e) - 1.0) * MASS_ELECTRON_EV;

        // Sample electron scattering angle from p(mu) = C/(1 - beta*mu)^2
        let beta_e = (e_electron * (e_electron + 2.0 * MASS_ELECTRON_EV)).sqrt()
            / (e_electron + MASS_ELECTRON_EV);
        let rn: f64 = rng.random_range(-1.0..1.0);
        let mu_electron = (rn + beta_e) / (rn * beta_e + 1.0);

        // Sample positron scattering angle
        let beta_p = (e_positron * (e_positron + 2.0 * MASS_ELECTRON_EV)).sqrt()
            / (e_positron + MASS_ELECTRON_EV);
        let rn: f64 = rng.random_range(-1.0..1.0);
        let mu_positron = (rn + beta_p) / (rn * beta_p + 1.0);

        (e_electron, e_positron, mu_electron, mu_positron)
    }
}

// ============================================================================
// GLOBAL ELEMENT CACHE
// ============================================================================

/// Map from element name to index in ELEMENTS vec.
static ELEMENT_MAP: Lazy<RwLock<HashMap<String, usize>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Global storage for loaded photon element data.
static ELEMENTS: Lazy<RwLock<Vec<Arc<PhotonInteraction>>>> = Lazy::new(|| RwLock::new(Vec::new()));

/// Shared Compton profile momentum grid (same for all elements).
/// Wrapped in Arc to allow cheap cloning from the hot path.
static COMPTON_PROFILE_PZ: Lazy<RwLock<Option<Arc<Vec<f64>>>>> = Lazy::new(|| RwLock::new(None));

/// Global TTB electron energy grid in LINEAR space (eV). Set once when
/// the first element with bremsstrahlung data loads, then never mutated
/// -- `init_bremsstrahlung` / `build_ttb_yields_table` read it in linear
/// space at prep time.
static TTB_E_GRID: Lazy<RwLock<Option<Vec<f64>>>> = Lazy::new(|| RwLock::new(None));

/// Log-space (ln(E)) copy of [`TTB_E_GRID`], built once by
/// [`ensure_ttb_e_grid_log`] and read at transport time by the
/// bremsstrahlung sampling. Kept separate from the linear grid (rather
/// than converting in place) so neither is ever mutated after first
/// build: concurrent photon simulations in one process previously raced
/// because one sim's transport read the grid while another sim's prep
/// re-cleared / re-converted it in place.
static TTB_E_GRID_LOG: Lazy<RwLock<Option<Vec<f64>>>> = Lazy::new(|| RwLock::new(None));

/// Global TTB reduced photon energy grid (k = E_photon / E_electron).
/// Shared across all elements (same grid from bremsstrahlung/photon_energy).
static TTB_K_GRID: Lazy<RwLock<Option<Vec<f64>>>> = Lazy::new(|| RwLock::new(None));

/// Lowest Z worth warning about when photon data carries no atomic relaxation.
///
/// Below this the K edge sits under the default 1 keV `photon_cutoff_energy`, so
/// a fluorescence photon could not be transported even if the data described
/// one, and the absence is unobservable rather than a gap. Sodium is the first
/// element whose K binding energy clears 1 keV (1.07 keV; neon is 0.87). Light
/// elements legitimately ship without relaxation -- the Be and Li fixtures both
/// do -- which is why this is not an unconditional warning.
const LOWEST_Z_NEEDING_RELAXATION: u32 = 11;

/// The warning to print when photon data arrives without atomic relaxation, or
/// `None` when its absence is expected (issue #41).
///
/// With no relaxation tables the photoelectric path emits no fluorescence and no
/// Auger electrons: the absorption just deposits locally, on both CPU and GPU.
/// That is a real fidelity difference and it is otherwise silent, which is how
/// it went unnoticed that the published fendl-3.2d photon set carries none while
/// the local `Fe.arrow` test fixture does. The tests passed with relaxation and
/// every downloaded element ran without it.
///
/// Split from the printing so the decision can be tested at any (Z, flag)
/// without a fixture: no fixture is both heavy enough to warn about and missing
/// relaxation, which is precisely the combination that reaches users.
fn relaxation_warning(
    name: &str,
    resolved_path: &str,
    z: u32,
    has_relaxation: bool,
) -> Option<String> {
    if has_relaxation || z < LOWEST_Z_NEEDING_RELAXATION {
        return None;
    }
    Some(format!(
        "Warning: photon data for {name} (Z={z}) carries no atomic-relaxation tables, so \
         photoelectric absorption will deposit locally and emit no fluorescence or Auger \
         electrons. Loaded from {resolved_path}. Another library may carry them; endf-b8.1 does."
    ))
}

/// Load an element from an Arrow directory, or return a cached copy if already loaded.
///
/// # Arguments
/// * `name` - Element name (e.g., "Fe"). Used as the cache key.
/// * `path` - Path to the Arrow directory containing photon data for this element.
///
/// # Returns
/// An `Arc<PhotonInteraction>` that can be shared across threads.
pub fn get_or_load_element(
    name: &str,
    path: &str,
) -> Result<Arc<PhotonInteraction>, Box<dyn std::error::Error>> {
    // Fast path: check if already cached
    {
        let map = ELEMENT_MAP
            .read()
            .map_err(|e| format!("ELEMENT_MAP read lock poisoned: {e}"))?;
        if let Some(&idx) = map.get(name) {
            let elems = ELEMENTS
                .read()
                .map_err(|e| format!("ELEMENTS read lock poisoned: {e}"))?;
            if let Some(elem) = elems.get(idx) {
                return Ok(Arc::clone(elem));
            }
        }
    }

    // Resolve keywords / URLs / parent directories to a concrete Arrow directory,
    // mirroring the nuclide loader (see `nuclide.rs::load_nuclide_for_python`). This
    // is what makes `material.read_nuclear_data("endf-b8.1")` autodownload the
    // element-level photon Arrow section set (e.g. Fe.arrow/) on first use.
    let resolved_path = yamc_nuclide::url_cache::resolve_data_path(
        path,
        Some(name),
        yamc_nuclide::url_cache::DataKind::Photon,
        // Photon data has no transmutation path, so it is always fetched whole.
        &yamc_nuclide::LoadScope::full(),
    )?;

    // Slow path: load from Arrow directory
    let p = std::path::Path::new(&resolved_path);
    let mut interaction = {
        #[cfg(feature = "arrow")]
        {
            crate::photon_arrow::read_photon_interaction_from_arrow(p)?
        }
        #[cfg(not(feature = "arrow"))]
        {
            return Err(format!(
                "Arrow directory '{}' found but arrow feature not enabled",
                resolved_path
            )
            .into());
        }
    };

    if let Some(message) = relaxation_warning(
        name,
        &resolved_path,
        interaction.atomic_number,
        interaction.has_atomic_relaxation,
    ) {
        eprintln!("{message}");
    }

    // Assign index and register in the global store. Lock ORDER matters:
    // every reader takes ELEMENT_MAP before ELEMENTS (the fast path above,
    // `get_element_by_name`), so the writer must acquire them in the same
    // order. Taking ELEMENTS first and then ELEMENT_MAP (the previous code)
    // is an ABBA inversion that deadlocks whenever a registering thread
    // races a fast-path reader: the writer holds ELEMENTS and waits on
    // ELEMENT_MAP while the reader holds ELEMENT_MAP and waits on ELEMENTS.
    // This hung multi-threaded test runs about half the time.
    let mut map = ELEMENT_MAP
        .write()
        .map_err(|e| format!("ELEMENT_MAP write lock poisoned: {e}"))?;

    // Double-check under the write lock: another thread may have loaded and
    // registered the same element while this one was reading the Arrow
    // directory. Without this, the loser registered a duplicate entry and
    // re-pointed the map at it.
    if let Some(&idx) = map.get(name) {
        let elems = ELEMENTS
            .read()
            .map_err(|e| format!("ELEMENTS read lock poisoned: {e}"))?;
        if let Some(elem) = elems.get(idx) {
            return Ok(Arc::clone(elem));
        }
    }

    let mut elems = ELEMENTS
        .write()
        .map_err(|e| format!("ELEMENTS write lock poisoned: {e}"))?;
    let index = elems.len();
    interaction.index = index;
    interaction.name = name.to_string();

    let arc = Arc::new(interaction);
    elems.push(Arc::clone(&arc));

    map.insert(name.to_string(), index);

    Ok(arc)
}

/// Retrieve a cached element by its global index.
pub fn get_element_by_index(index: usize) -> Option<Arc<PhotonInteraction>> {
    let elems = ELEMENTS.read().ok()?;
    elems.get(index).cloned()
}

/// Retrieve a cached element by name.
pub fn get_element_by_name(name: &str) -> Option<Arc<PhotonInteraction>> {
    let map = ELEMENT_MAP.read().ok()?;
    let &idx = map.get(name)?;
    let elems = ELEMENTS.read().ok()?;
    elems.get(idx).cloned()
}

/// Clear all cached element data. Useful for testing.
pub fn clear_element_cache() {
    if let Ok(mut elems) = ELEMENTS.write() {
        elems.clear();
    }
    if let Ok(mut map) = ELEMENT_MAP.write() {
        map.clear();
    }
    if let Ok(mut pz) = COMPTON_PROFILE_PZ.write() {
        *pz = None;
    }
}

thread_local! {
    /// Whether a read on this thread may publish the shared grids below.
    static SHARED_GRIDS_SUPPRESSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Stops reads on this thread from publishing the shared grids, until dropped.
///
/// The three grids below are process-wide and first-write-wins, so the FIRST
/// element read in a process decides the Compton momentum grid and the two TTB
/// grids that every later element load and every photon collision then uses.
/// That is right for a simulation, where one library is loaded and shared, and
/// wrong for anything that reads a directory to LOOK at it: a candidate
/// conversion held to one side would install its grids over the library's, and
/// a same-length grid with different values does it silently.
///
/// So an inspecting reader takes one of these first. Reading is unaffected:
/// nothing in the read path reads these back, they are consumed later by
/// transport.
///
/// Thread-local rather than global because the suppression belongs to one call,
/// and a simulation loading elements on other threads must be unaffected.
pub struct SharedGridsSuppressed(bool);

impl SharedGridsSuppressed {
    /// Suppress publication for this thread. Nests correctly.
    pub fn new() -> Self {
        Self(SHARED_GRIDS_SUPPRESSED.with(|f| f.replace(true)))
    }
}

impl Default for SharedGridsSuppressed {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SharedGridsSuppressed {
    fn drop(&mut self) {
        SHARED_GRIDS_SUPPRESSED.with(|f| f.set(self.0));
    }
}

fn shared_grids_suppressed() -> bool {
    SHARED_GRIDS_SUPPRESSED.with(|f| f.get())
}

/// Set the shared Compton profile momentum grid.
/// Called once during the first element load.
pub fn set_compton_profile_pz(pz: Vec<f64>) {
    if shared_grids_suppressed() {
        return;
    }
    if let Ok(mut stored) = COMPTON_PROFILE_PZ.write() {
        if stored.is_none() {
            *stored = Some(Arc::new(pz));
        }
    }
}

/// Get a cheap Arc clone of the shared Compton profile momentum grid.
pub fn compton_profile_pz() -> Arc<Vec<f64>> {
    COMPTON_PROFILE_PZ
        .read()
        .ok()
        .and_then(|opt| opt.as_ref().map(Arc::clone))
        .unwrap_or_else(|| Arc::new(Vec::new()))
}

/// Set the shared TTB electron energy grid (stored as ln(E)).
/// Called once during the first element load that has bremsstrahlung data.
pub fn set_ttb_e_grid(grid: Vec<f64>) {
    if shared_grids_suppressed() {
        return;
    }
    if let Ok(mut stored) = TTB_E_GRID.write() {
        if stored.is_none() {
            *stored = Some(grid);
        }
    }
}

/// Get a clone of the shared TTB electron energy grid (ln(E)).
pub fn ttb_e_grid() -> Vec<f64> {
    TTB_E_GRID
        .read()
        .ok()
        .and_then(|opt| opt.clone())
        .unwrap_or_default()
}

/// Set the shared TTB reduced photon energy grid.
/// Called once during the first element load that has bremsstrahlung data.
pub fn set_ttb_k_grid(grid: Vec<f64>) {
    if shared_grids_suppressed() {
        return;
    }
    if let Ok(mut stored) = TTB_K_GRID.write() {
        if stored.is_none() {
            *stored = Some(grid);
        }
    }
}

/// Get a clone of the shared TTB reduced photon energy grid.
pub fn ttb_k_grid() -> Vec<f64> {
    TTB_K_GRID
        .read()
        .ok()
        .and_then(|opt| opt.clone())
        .unwrap_or_default()
}

/// Build the log-space copy of the TTB electron energy grid, once.
///
/// Idempotent and non-destructive: the linear [`TTB_E_GRID`] is left
/// untouched (prep-time readers still need it linear) and
/// [`TTB_E_GRID_LOG`] is populated only if not already built. This is
/// safe to call at the start of every simulation, including from
/// concurrent simulations sharing the process -- it never mutates a grid
/// that another simulation's transport may be reading.
pub fn ensure_ttb_e_grid_log() {
    // Fast path: already built.
    if TTB_E_GRID_LOG
        .read()
        .ok()
        .map(|g| g.is_some())
        .unwrap_or(false)
    {
        return;
    }
    let linear = match TTB_E_GRID.read() {
        Ok(g) => g.clone(),
        Err(_) => return,
    };
    if let Some(grid) = linear {
        let log_grid: Vec<f64> = grid.iter().map(|v| v.ln()).collect();
        if let Ok(mut stored) = TTB_E_GRID_LOG.write() {
            // Re-check under the write lock in case of a race.
            if stored.is_none() {
                *stored = Some(log_grid);
            }
        }
    }
}

/// Get a clone of the log-space (ln(E)) TTB electron energy grid used at
/// transport time. Empty until [`ensure_ttb_e_grid_log`] has run.
pub fn ttb_e_grid_log() -> Vec<f64> {
    TTB_E_GRID_LOG
        .read()
        .ok()
        .and_then(|opt| opt.clone())
        .unwrap_or_default()
}

/// Clear TTB grids (for testing).
pub fn clear_ttb_grids() {
    if let Ok(mut g) = TTB_E_GRID.write() {
        *g = None;
    }
    if let Ok(mut g) = TTB_E_GRID_LOG.write() {
        *g = None;
    }
    if let Ok(mut g) = TTB_K_GRID.write() {
        *g = None;
    }
}

/// Look up the 1-based SUBSHELLS index for a designator string.
/// Returns None if the designator is not recognized.
pub fn subshell_index(designator: &str) -> Option<usize> {
    SUBSHELLS
        .iter()
        .position(|&s| s == designator)
        .map(|i| i + 1) // 1-based
}

// ============================================================================
// TESTS
// ============================================================================

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolve a path under `crates/yamc/tests/` from this crate.
    /// The Fe.arrow / similar photon test fixtures live there
    /// (shared with the yamc transport tests). `CARGO_MANIFEST_DIR`
    /// points at `crates/yamc-element/`, so we step up and across.
    fn yamc_test_path(rel: &str) -> String {
        format!("{}/../yamc/tests/{}", env!("CARGO_MANIFEST_DIR"), rel)
    }

    #[test]
    fn test_subshell_index() {
        assert_eq!(subshell_index("K"), Some(1));
        assert_eq!(subshell_index("L1"), Some(2));
        assert_eq!(subshell_index("L3"), Some(4));
        assert_eq!(subshell_index("Q3"), Some(39));
        assert_eq!(subshell_index("INVALID"), None);
    }

    #[test]
    fn test_element_micro_xs_default() {
        let xs = ElementMicroXS::default();
        assert_eq!(xs.total, 0.0);
        assert_eq!(xs.coherent, 0.0);
        assert_eq!(xs.incoherent, 0.0);
        assert_eq!(xs.photoelectric, 0.0);
        assert_eq!(xs.pair_production, 0.0);
        assert_eq!(xs.heating, 0.0);
        assert_eq!(xs.last_energy, 0.0);
        assert_eq!(xs.index_grid, 0);
        assert_eq!(xs.interp_factor, 0.0);
    }

    #[test]
    fn test_subshells_count() {
        assert_eq!(SUBSHELLS.len(), 39);
        assert_eq!(SUBSHELLS[0], "K");
        assert_eq!(SUBSHELLS[38], "Q3");
    }

    #[test]
    fn test_load_fe_from_arrow() {
        // Load Fe photon data from the test Arrow directory
        let path = yamc_test_path("Fe.arrow");
        let path = path.as_str();
        let result = get_or_load_element("Fe", path);
        assert!(
            result.is_ok(),
            "Failed to load Fe.arrow: {:?}",
            result.err()
        );

        let fe = result.unwrap();
        assert_eq!(fe.name, "Fe");
        assert_eq!(fe.atomic_number, 26);

        // Energy grid should be non-empty and in log form
        assert!(!fe.energy.is_empty(), "Energy grid should not be empty");
        // ln(E) values: the minimum photon energy is ~1 eV so ln(1) = 0
        // and the maximum is ~100 GeV so ln(1e11) ~ 25.3
        assert!(
            fe.energy[0] < fe.energy[fe.energy.len() - 1],
            "Energy grid should be monotonically increasing"
        );

        // Cross sections should have same length as energy grid
        let n = fe.energy.len();
        assert_eq!(fe.coherent_xs.len(), n);
        assert_eq!(fe.incoherent_xs.len(), n);
        assert_eq!(fe.photoelectric_total_xs.len(), n);
        assert_eq!(fe.pair_production_total_xs.len(), n);
        assert_eq!(fe.pair_production_nuclear_xs.len(), n);
        assert_eq!(fe.pair_production_electron_xs.len(), n);
        assert_eq!(fe.heating_xs.len(), n);

        // Coherent XS should have reasonable log values (not all -900)
        let non_trivial = fe.coherent_xs.iter().filter(|&&v| v > -900.0).count();
        assert!(
            non_trivial > 0,
            "Coherent XS should have some non-zero values"
        );

        // Form factors should have data
        match &fe.coherent_int_form_factor {
            Tabulated1D::Tabulated1D { x, y, .. } => {
                assert!(!x.is_empty(), "Coherent form factor x should not be empty");
                assert!(!y.is_empty(), "Coherent form factor y should not be empty");
            }
        }
        match &fe.incoherent_form_factor {
            Tabulated1D::Tabulated1D { x, y, .. } => {
                assert!(
                    !x.is_empty(),
                    "Incoherent form factor x should not be empty"
                );
                assert!(
                    !y.is_empty(),
                    "Incoherent form factor y should not be empty"
                );
            }
        }

        // Subshells: Fe (Z=26) should have several subshells
        assert!(!fe.shells.is_empty(), "Fe should have electron subshells");
        // K shell should be present
        let k_shell = fe.shells.iter().find(|s| s.index_subshell == 1);
        assert!(k_shell.is_some(), "Fe should have a K shell");
        let k = k_shell.unwrap();
        assert!(
            k.binding_energy > 7000.0,
            "Fe K-shell binding energy should be > 7 keV, got {}",
            k.binding_energy
        );
        assert!(k.num_electrons > 0.0, "K shell should have electrons");

        // Compton profile data
        assert!(
            !fe.electron_pdf.is_empty(),
            "Compton profile electron PDF should not be empty"
        );
        assert!(
            !fe.binding_energy.is_empty(),
            "Compton profile binding energy should not be empty"
        );
        assert!(
            !fe.profile_pdf.is_empty(),
            "Compton profile PDF should not be empty"
        );
        assert!(
            !fe.profile_cdf.is_empty(),
            "Compton profile CDF should not be empty"
        );
        // CDF should start near 0 and be monotonically increasing
        for (i, cdf) in fe.profile_cdf.iter().enumerate() {
            if !cdf.is_empty() {
                assert!(
                    cdf[0].abs() < 1e-10,
                    "CDF for shell {} should start at ~0, got {}",
                    i,
                    cdf[0]
                );
                assert!(
                    *cdf.last().unwrap() > 0.0,
                    "CDF for shell {} should end at a positive value, got {}",
                    i,
                    cdf.last().unwrap()
                );
            }
        }

        // electron_pdf should sum to ~1.0 (normalized)
        let pdf_sum: f64 = fe.electron_pdf.iter().sum();
        assert!(
            (pdf_sum - 1.0).abs() < 1e-10,
            "electron_pdf should sum to ~1.0, got {}",
            pdf_sum
        );

        // compton_relax_map should have same length as electron_pdf
        assert_eq!(
            fe.compton_relax_map.len(),
            fe.electron_pdf.len(),
            "compton_relax_map should have same length as electron_pdf"
        );

        // Cross sections 2D array
        assert_eq!(fe.cross_sections.len(), n);
        if !fe.shells.is_empty() {
            assert_eq!(fe.cross_sections[0].len(), fe.shells.len());
        }

        // Global pz grid should have been set
        let pz = compton_profile_pz();
        assert!(!pz.is_empty(), "Compton profile pz grid should be set");

        // Test caching: loading again should return the same data
        let fe2 = get_or_load_element("Fe", path).unwrap();
        assert_eq!(fe2.atomic_number, fe.atomic_number);
        assert_eq!(fe2.energy.len(), fe.energy.len());

        // Test lookup by name and index
        let fe_by_name = get_element_by_name("Fe");
        assert!(fe_by_name.is_some());
        assert_eq!(fe_by_name.unwrap().atomic_number, 26);

        let fe_by_idx = get_element_by_index(fe.index);
        assert!(fe_by_idx.is_some());
        assert_eq!(fe_by_idx.unwrap().atomic_number, 26);
    }

    #[test]
    fn test_calculate_xs_fe() {
        let path = yamc_test_path("Fe.arrow");
        let path = path.as_str();
        let fe = get_or_load_element("Fe", path).unwrap();

        // Test at 100 keV -- well above all thresholds
        let xs_100kev = fe.calculate_xs(100_000.0);
        assert!(
            xs_100kev.total > 0.0,
            "Total XS should be positive at 100 keV"
        );
        assert!(xs_100kev.coherent > 0.0, "Coherent XS should be positive");
        assert!(
            xs_100kev.incoherent > 0.0,
            "Incoherent XS should be positive"
        );
        assert!(
            xs_100kev.photoelectric > 0.0,
            "Photoelectric XS should be positive"
        );
        // Heating KERMA should be positive at 100 keV
        assert!(
            xs_100kev.heating > 0.0,
            "Heating KERMA should be positive at 100 keV"
        );
        // Pair production threshold is ~1.022 MeV, so should be ~0 at 100 keV
        assert!(
            xs_100kev.pair_production < 1e-10,
            "Pair production should be ~0 at 100 keV, got {}",
            xs_100kev.pair_production
        );
        // Total = sum of components
        let sum = xs_100kev.coherent
            + xs_100kev.incoherent
            + xs_100kev.photoelectric
            + xs_100kev.pair_production;
        assert!(
            (xs_100kev.total - sum).abs() / sum < 1e-10,
            "Total should equal sum of components"
        );

        // Test at 10 MeV -- above pair production threshold
        let xs_10mev = fe.calculate_xs(10_000_000.0);
        assert!(xs_10mev.total > 0.0);
        assert!(
            xs_10mev.pair_production > 0.0,
            "Pair production should be positive at 10 MeV"
        );

        // At low energy (1 keV), photoelectric should dominate
        let xs_1kev = fe.calculate_xs(1_000.0);
        assert!(xs_1kev.total > 0.0);
        assert!(
            xs_1kev.photoelectric > xs_1kev.coherent,
            "Photoelectric should dominate at 1 keV"
        );
        assert!(
            xs_1kev.photoelectric > xs_1kev.incoherent,
            "Photoelectric should exceed incoherent at 1 keV"
        );

        // Grid index and interp factor should be sensible
        assert!(xs_100kev.index_grid < fe.energy.len() - 1);
        assert!(xs_100kev.interp_factor >= 0.0 && xs_100kev.interp_factor <= 1.0);
    }

    #[test]
    fn test_klein_nishina_low_energy() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);

        // Low energy: alpha = 0.1 (51 keV) -- uses Kahn's method
        let alpha = 0.1;
        for _ in 0..1000 {
            let (alpha_out, mu) = klein_nishina(alpha, &mut rng);
            assert!(alpha_out > 0.0, "alpha_out must be positive");
            assert!(
                alpha_out <= alpha,
                "alpha_out ({alpha_out}) must be <= alpha ({alpha})"
            );
            assert!((-1.0..=1.0).contains(&mu), "mu ({mu}) must be in [-1, 1]");
        }
    }

    #[test]
    fn test_klein_nishina_high_energy() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(123);

        // High energy: alpha = 10.0 (5.1 MeV) -- uses Koblinger's method
        let alpha = 10.0;
        let mut forward_count = 0;
        let n = 5000;
        for _ in 0..n {
            let (alpha_out, mu) = klein_nishina(alpha, &mut rng);
            assert!(alpha_out > 0.0);
            assert!(alpha_out <= alpha);
            assert!((-1.0..=1.0).contains(&mu));
            if mu > 0.0 {
                forward_count += 1;
            }
        }
        // At high energy, scattering should be forward-peaked
        let forward_frac = forward_count as f64 / n as f64;
        assert!(
            forward_frac > 0.6,
            "Klein-Nishina at high energy should be forward-peaked, got {forward_frac}"
        );
    }

    #[test]
    fn test_rayleigh_scatter_fe() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);

        let fe = get_or_load_element("Fe", &yamc_test_path("Fe.arrow")).unwrap();

        // At 100 keV: alpha = 100000 / 511000 ≈ 0.196
        let alpha = 100_000.0 / MASS_ELECTRON_EV;
        let n = 2000;
        let mut sum_mu = 0.0;
        for _ in 0..n {
            let mu = fe.rayleigh_scatter(alpha, &mut rng);
            assert!(
                (-1.0..=1.0).contains(&mu),
                "Rayleigh mu ({mu}) must be in [-1, 1]"
            );
            sum_mu += mu;
        }
        // Rayleigh is forward-peaked (especially at higher energies)
        let mean_mu = sum_mu / n as f64;
        assert!(
            mean_mu > 0.0,
            "Rayleigh at 100 keV should be forward-peaked, mean mu = {mean_mu}"
        );

        // At very low energy (1 keV), scattering approaches isotropic
        let alpha_low = 1_000.0 / MASS_ELECTRON_EV;
        let mut count_backward = 0;
        for _ in 0..n {
            let mu = fe.rayleigh_scatter(alpha_low, &mut rng);
            assert!((-1.0..=1.0).contains(&mu));
            if mu < 0.0 {
                count_backward += 1;
            }
        }
        // At low energy, significant backward scattering should occur
        let backward_frac = count_backward as f64 / n as f64;
        assert!(
            backward_frac > 0.1,
            "Rayleigh at 1 keV should have some backward scattering, got {backward_frac}"
        );
    }

    #[test]
    fn test_compton_scatter_fe() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);

        let fe = get_or_load_element("Fe", &yamc_test_path("Fe.arrow")).unwrap();

        // At 1 MeV: alpha ≈ 1.957
        let energy = 1_000_000.0;
        let alpha = energy / MASS_ELECTRON_EV;
        let n = 5000;
        let mut sum_alpha_out = 0.0;

        for _ in 0..n {
            let (alpha_out, mu, i_shell) = fe.compton_scatter(alpha, false, &mut rng);
            assert!(alpha_out > 0.0, "alpha_out must be positive");
            assert!(
                alpha_out <= alpha,
                "Compton: E_out must be <= E_in ({alpha_out} > {alpha})"
            );
            assert!((-1.0..=1.0).contains(&mu), "mu ({mu}) out of range");
            assert_eq!(i_shell, -1, "Shell should be -1 when doppler=false");
            sum_alpha_out += alpha_out;
        }

        // Mean energy loss should be reasonable (not 0, not all energy)
        let mean_alpha_out = sum_alpha_out / n as f64;
        assert!(
            mean_alpha_out < alpha,
            "Mean Compton outgoing energy should be less than incoming"
        );
        assert!(
            mean_alpha_out > 0.1 * alpha,
            "Mean Compton outgoing energy should retain significant fraction"
        );

        // At high energy (10 MeV, alpha ≈ 19.6), scattering should be forward-peaked
        let alpha_high = 10_000_000.0 / MASS_ELECTRON_EV;
        let mut forward_count = 0;
        for _ in 0..n {
            let (_, mu, _) = fe.compton_scatter(alpha_high, false, &mut rng);
            if mu > 0.0 {
                forward_count += 1;
            }
        }
        let forward_frac = forward_count as f64 / n as f64;
        assert!(
            forward_frac > 0.5,
            "Compton at 10 MeV should be forward-peaked, got {forward_frac}"
        );
    }

    #[test]
    fn test_isotropic_direction() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);

        let n = 5000;
        let mut sum = [0.0_f64; 3];
        for _ in 0..n {
            let dir = isotropic_direction(&mut rng);
            // Check unit vector
            let mag = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-12,
                "Direction should be unit vector, got magnitude {mag}"
            );
            sum[0] += dir[0];
            sum[1] += dir[1];
            sum[2] += dir[2];
        }
        // Mean should be near zero for isotropic distribution
        for (i, &component) in sum.iter().enumerate() {
            let mean = component / n as f64;
            assert!(
                mean.abs() < 0.1,
                "Mean direction component {i} should be ~0, got {mean}"
            );
        }
    }

    #[test]
    fn test_sample_photoelectric_subshell_fe() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);

        let fe = get_or_load_element("Fe", &yamc_test_path("Fe.arrow")).unwrap();

        // At 100 keV, photoelectric effect is significant for Fe
        let xs = fe.calculate_xs(100_000.0);
        assert!(
            xs.photoelectric > 0.0,
            "Photoelectric XS should be positive at 100 keV"
        );

        let n = 5000;
        let mut shell_counts = vec![0usize; fe.shells.len()];
        for _ in 0..n {
            let i_shell = fe.sample_photoelectric_subshell(&xs, &mut rng);
            assert!(
                i_shell < fe.shells.len(),
                "Shell index {i_shell} out of range"
            );
            shell_counts[i_shell] += 1;
        }

        // K-shell (index 0) should be sampled most frequently at 100 keV
        // because K-shell has the largest photoelectric XS above its threshold
        let k_frac = shell_counts[0] as f64 / n as f64;
        assert!(
            k_frac > 0.5,
            "K-shell should dominate at 100 keV, got fraction {k_frac}"
        );

        // All sampled shells should be valid
        let total_sampled: usize = shell_counts.iter().sum();
        assert_eq!(total_sampled, n);
    }

    #[test]
    fn test_atomic_relaxation_fe() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);

        let fe = get_or_load_element("Fe", &yamc_test_path("Fe.arrow")).unwrap();

        // Trigger relaxation from K-shell vacancy at 100 keV
        let energy = 100_000.0;
        let k_shell_idx = 0; // K shell is the first shell

        // Run many relaxation cascades to verify statistical properties
        let n = 2000;
        let mut total_photons = 0;
        let mut _total_electrons = 0;
        let mut max_photon_energy = 0.0_f64;

        for _ in 0..n {
            let secondaries = fe.atomic_relaxation(k_shell_idx, energy, &mut rng);

            for &(e, dir, is_photon) in &secondaries {
                assert!(e > 0.0, "Secondary energy must be positive, got {e}");
                // Energy should not exceed K-shell binding energy
                assert!(
                    e <= fe.shells[k_shell_idx].binding_energy + 1.0,
                    "Secondary energy {e} exceeds K-shell binding energy {}",
                    fe.shells[k_shell_idx].binding_energy
                );
                // Direction should be unit vector
                let mag = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                assert!((mag - 1.0).abs() < 1e-12, "Direction should be unit vector");

                if is_photon {
                    total_photons += 1;
                    max_photon_energy = max_photon_energy.max(e);
                } else {
                    _total_electrons += 1;
                }
            }
        }

        // Should produce some fluorescent photons (K-alpha, K-beta X-rays)
        assert!(
            total_photons > 0,
            "Fe K-shell relaxation should produce fluorescent photons"
        );
        // Max fluorescent photon energy should be near K-shell binding energy
        // (Fe K-alpha ~ 6.4 keV)
        assert!(
            max_photon_energy > 5000.0 && max_photon_energy < 8000.0,
            "Max fluorescent photon energy should be near Fe K-alpha (~6.4 keV), got {max_photon_energy}"
        );

        // Test threshold: if incident energy < binding energy, no relaxation
        let low_energy_secondaries = fe.atomic_relaxation(k_shell_idx, 1.0, &mut rng);
        assert!(
            low_energy_secondaries.is_empty(),
            "No relaxation when energy < binding energy"
        );
    }

    #[test]
    fn test_pair_production_fe() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);

        let fe = get_or_load_element("Fe", &yamc_test_path("Fe.arrow")).unwrap();

        // At 10 MeV (well above 1.022 MeV threshold)
        let energy = 10_000_000.0;
        let alpha = energy / MASS_ELECTRON_EV;
        let threshold = 2.0 * MASS_ELECTRON_EV;

        let n = 5000;
        let mut sum_e_total = 0.0;
        for _ in 0..n {
            let (e_electron, e_positron, mu_electron, mu_positron) =
                fe.pair_production(alpha, &mut rng);

            // Both energies must be non-negative
            assert!(
                e_electron >= 0.0,
                "Electron energy must be >= 0, got {e_electron}"
            );
            assert!(
                e_positron >= 0.0,
                "Positron energy must be >= 0, got {e_positron}"
            );

            // Total kinetic energy = photon energy - 2*m_e*c^2
            let e_total = e_electron + e_positron;
            let expected_total = energy - threshold;
            assert!(
                (e_total - expected_total).abs() / expected_total < 1e-10,
                "E_electron + E_positron should equal E_photon - 2*m_e*c^2: \
                 {e_total} vs {expected_total}"
            );
            sum_e_total += e_total;

            // Cosines should be in [-1, 1]
            assert!(
                (-1.0..=1.0).contains(&mu_electron),
                "mu_electron ({mu_electron}) out of range"
            );
            assert!(
                (-1.0..=1.0).contains(&mu_positron),
                "mu_positron ({mu_positron}) out of range"
            );
        }

        // Mean total KE should match expected
        let mean_total = sum_e_total / n as f64;
        let expected = energy - threshold;
        assert!(
            (mean_total - expected).abs() / expected < 1e-6,
            "Mean total KE should match, got {mean_total} vs {expected}"
        );

        // At high energy, both electron and positron should be forward-peaked
        let mut forward_electron = 0;
        let mut forward_positron = 0;
        for _ in 0..n {
            let (_, _, mu_e, mu_p) = fe.pair_production(alpha, &mut rng);
            if mu_e > 0.0 {
                forward_electron += 1;
            }
            if mu_p > 0.0 {
                forward_positron += 1;
            }
        }
        let fwd_e = forward_electron as f64 / n as f64;
        let fwd_p = forward_positron as f64 / n as f64;
        assert!(
            fwd_e > 0.5,
            "Electron should be forward-peaked at 10 MeV, got {fwd_e}"
        );
        assert!(
            fwd_p > 0.5,
            "Positron should be forward-peaked at 10 MeV, got {fwd_p}"
        );
    }

    #[test]
    fn ensure_ttb_e_grid_log_is_idempotent_and_nondestructive() {
        // Regression test: a photon simulation builds the log TTB grid at
        // prep time; a second (or concurrent) simulation must not corrupt
        // it. The old in-place convert applied ln() on every call, so
        // repeated prep double-logged the grid. ensure_ttb_e_grid_log must
        // leave the linear grid untouched and produce the log grid exactly
        // once, no matter how many times it runs.
        clear_ttb_grids();
        let linear = vec![1.0_f64, 10.0, 100.0, 1000.0];
        set_ttb_e_grid(linear.clone());

        for _ in 0..3 {
            ensure_ttb_e_grid_log();
            // Linear grid is never mutated.
            assert_eq!(ttb_e_grid(), linear, "linear TTB grid was mutated");
            // Log grid is exactly ln(linear) -- not ln(ln(...)).
            let log_grid = ttb_e_grid_log();
            assert_eq!(log_grid.len(), linear.len());
            for (lg, lin) in log_grid.iter().zip(linear.iter()) {
                assert!(
                    (lg - lin.ln()).abs() < 1e-12,
                    "log grid entry {lg} != ln({lin}) = {} (double-converted?)",
                    lin.ln()
                );
            }
        }
        clear_ttb_grids();
    }

    #[test]
    fn compton_energy_transfer_fraction_matches_klein_nishina() {
        // Reference values from numerical integration of the
        // Klein-Nishina differential cross section (scipy quad,
        // agreement to ~1e-12; see issue #358). The broken closed form
        // this replaced returned -1.78 at 50 keV.
        const MEC2: f64 = crate::photon::MASS_ELECTRON_EV;
        let reference = [
            (1.0e3, 0.00195),
            (10.0e3, 0.01876),
            (50.0e3, 0.08071),
            (100.0e3, 0.13800),
            (200.0e3, 0.21635),
            (400.0e3, 0.30963),
            (1.0e6, 0.44004),
            (5.0e6, 0.62799),
            (14.0e6, 0.70581),
        ];
        for (energy, expected) in reference {
            let f = compton_energy_transfer_fraction(energy / MEC2);
            assert!(
                (f - expected).abs() < 5e-5,
                "f({energy:.1e} eV) = {f:.5}, expected {expected:.5}"
            );
            assert!(f > 0.0, "transfer fraction must be positive");
        }
        // Monotonically increasing over the transport range.
        let mut last = 0.0;
        for i in 1..=140 {
            let e = 1.0e5 * i as f64;
            let f = compton_energy_transfer_fraction(e / MEC2);
            assert!(f > last, "not increasing at {e:.1e} eV");
            last = f;
        }
    }

    fn shell_with(binding: f64, transitions: Vec<AtomicTransition>) -> ElectronSubshell {
        ElectronSubshell {
            index_subshell: 0,
            binding_energy: binding,
            num_electrons: 2.0,
            threshold: 0,
            cross_section: Vec::new(),
            transitions,
        }
    }

    #[test]
    fn compute_subshell_radiative_energy_matches_relaxation_cascade() {
        // Synthetic K/L/M shells pin the expected-value recursion against a
        // hand-computed cascade. A radiative transition (secondary_subshell == -1)
        // emits its photon `energy` and leaves a primary vacancy; an Auger
        // transition leaves primary + secondary vacancies and emits no photon; a
        // shell with no transitions emits its full binding energy. This mirrors
        // `atomic_relaxation` exactly so the KERMA coefficient equals the expected
        // analog deposit.
        let shells = vec![
            // 0 = K (binding 10000): 50% radiative K->L (photon 9000, L vacancy),
            //    50% Auger (L fills K, eject M => L and M vacancies, no photon).
            shell_with(
                10_000.0,
                vec![
                    AtomicTransition {
                        primary_subshell: 1,
                        secondary_subshell: -1,
                        energy: 9_000.0,
                        probability: 0.5,
                    },
                    AtomicTransition {
                        primary_subshell: 1,
                        secondary_subshell: 2,
                        energy: 8_900.0,
                        probability: 1.0,
                    },
                ],
            ),
            // 1 = L (binding 1000): radiative L->M (photon 900, M vacancy).
            shell_with(
                1_000.0,
                vec![AtomicTransition {
                    primary_subshell: 2,
                    secondary_subshell: -1,
                    energy: 900.0,
                    probability: 1.0,
                }],
            ),
            // 2 = M (binding 100): no transitions -> full binding emitted.
            shell_with(100.0, Vec::new()),
        ];

        let r = compute_subshell_radiative_energy(&shells);
        // R[M] = 100
        // R[L] = 900 + R[M] = 1000
        // R[K] = 0.5*(9000 + R[L]) + 0.5*(R[L] + R[M]) = 5000 + 550 = 5550
        assert!((r[2] - 100.0).abs() < 1e-9, "R[M] = {}", r[2]);
        assert!((r[1] - 1_000.0).abs() < 1e-9, "R[L] = {}", r[1]);
        assert!((r[0] - 5_550.0).abs() < 1e-9, "R[K] = {}", r[0]);
        // Radiative energy is non-negative and bounded by the shell binding energy.
        for (ri, s) in r.iter().zip(&shells) {
            assert!(*ri >= 0.0 && *ri <= s.binding_energy + 1e-9);
        }
    }

    #[test]
    fn compute_subshell_radiative_energy_untracked_primary() {
        // primary_subshell == -1 (electron from an untracked/valence shell): the
        // photon is still emitted but there is no further cascade.
        let shells = vec![shell_with(
            50.0,
            vec![AtomicTransition {
                primary_subshell: -1,
                secondary_subshell: -1,
                energy: 50.0,
                probability: 1.0,
            }],
        )];
        let r = compute_subshell_radiative_energy(&shells);
        assert!((r[0] - 50.0).abs() < 1e-9, "R = {}", r[0]);
    }

    /// Issue #41: photon data without atomic relaxation emits no fluorescence and
    /// no Auger electrons, and said nothing about it. The published fendl-3.2d
    /// photon set carries none while the local Fe fixture does, so the tests
    /// passed with relaxation and every downloaded element ran without it.
    ///
    /// Tested on the predicate rather than through a load, because no fixture is
    /// both heavy enough to warn about AND missing relaxation -- that pairing is
    /// exactly what reaches users and never reaches the suite.
    #[test]
    fn a_heavy_element_without_relaxation_warns() {
        for (name, z) in [("Fe", 26), ("W", 74), ("Pb", 82)] {
            let warning = relaxation_warning(name, "/cache/x.arrow", z, false)
                .unwrap_or_else(|| panic!("{name} (Z={z}) must warn"));
            assert!(warning.contains(name), "names the element: {warning}");
            assert!(warning.contains(&z.to_string()), "names Z: {warning}");
            assert!(
                warning.contains("/cache/x.arrow"),
                "names where it came from, which is how you tell which library: {warning}"
            );
        }
    }

    #[test]
    fn relaxation_present_never_warns() {
        for z in [1, 4, 11, 26, 92] {
            assert!(
                relaxation_warning("X", "/cache/x.arrow", z, true).is_none(),
                "Z={z} carries relaxation, so there is nothing to say"
            );
        }
    }

    /// Light elements legitimately ship without relaxation, so warning about them
    /// would be noise. The threshold is where the K edge clears the default 1 keV
    /// transport cutoff: sodium at 1.07 keV, neon at 0.87.
    #[test]
    fn light_elements_without_relaxation_are_silent() {
        for z in [1, 2, 3, 4, 9, 10] {
            assert!(
                relaxation_warning("X", "/cache/x.arrow", z, false).is_none(),
                "Z={z} is below the cutoff-relevant threshold and must stay quiet"
            );
        }
        assert!(
            relaxation_warning("Na", "/cache/x.arrow", 11, false).is_some(),
            "sodium is the first element whose K edge clears 1 keV"
        );
    }

    /// Issue #41 bullet 3: the flag on data as PUBLISHED, not as fixtured.
    ///
    /// The whole defect was that the local `Fe.arrow` fixture carries relaxation
    /// while what users downloaded did not, so every existing assertion passed
    /// against data no user ever loads. This one goes to the origin.
    ///
    /// Ignored by default because it downloads. Run:
    ///   cargo test -p yamc-element --features download-tls -- --ignored downloaded
    #[cfg(feature = "download")]
    #[test]
    #[ignore]
    fn a_downloaded_mid_z_element_carries_relaxation() {
        // Tungsten rather than iron: `get_or_load_element` keys the global store
        // on the element NAME, so asking for "Fe" would hand back whatever the
        // fixture tests already registered and prove nothing about the origin.
        // W is one of the elements the issue reported and no fixture loads it.
        let w = get_or_load_element("W", "endf-b8.1").expect("download W from endf-b8.1");
        assert_eq!(w.atomic_number, 74);
        assert!(
            w.has_atomic_relaxation,
            "endf-b8.1 W must publish atomic relaxation; without it the photoelectric \
             path emits no fluorescence and no Auger electrons"
        );
        assert!(
            w.shells.iter().any(|s| s.binding_energy > 0.0),
            "relaxation with all-zero binding energies is the shape the published \
             fendl-3.2d set has, and is not relaxation"
        );
        assert!(
            w.subshell_radiative_energy.iter().any(|&r| r > 0.0),
            "at least the K shell must carry positive fluorescence energy"
        );
        assert!(
            relaxation_warning("W", "p", w.atomic_number, w.has_atomic_relaxation).is_none(),
            "and therefore nothing to warn about"
        );
    }

    /// The two fixtures that exist, as a guard that the real flag still lines up
    /// with the predicate: Fe carries relaxation, Be does not and is light.
    #[test]
    fn the_fixtures_agree_with_the_predicate() {
        let fe = get_or_load_element("Fe", yamc_test_path("Fe.arrow").as_str()).unwrap();
        assert!(
            fe.has_atomic_relaxation,
            "the Fe fixture carries relaxation"
        );
        assert!(
            relaxation_warning("Fe", "p", fe.atomic_number, fe.has_atomic_relaxation).is_none()
        );

        let be = get_or_load_element("Be", yamc_test_path("Be.arrow").as_str()).unwrap();
        assert!(!be.has_atomic_relaxation, "the Be fixture carries none");
        assert!(
            relaxation_warning("Be", "p", be.atomic_number, be.has_atomic_relaxation).is_none(),
            "Be is Z=4, so its absence is expected rather than a gap"
        );
    }

    #[test]
    fn photoelectric_heating_subtracts_transported_fluorescence() {
        // End-to-end on Fe (whose fixture has no tabulated heating_xs, so the
        // physics fallback runs): the photoelectric photon-KERMA coefficient must
        // subtract the fluorescence energy that atomic_relaxation banks and
        // transports, leaving it strictly below the old "full incident energy"
        // value wherever photoelectric contributes.
        let fe = get_or_load_element("Fe", yamc_test_path("Fe.arrow").as_str()).unwrap();
        assert!(fe.has_atomic_relaxation);
        assert_eq!(fe.subshell_radiative_energy.len(), fe.shells.len());

        // Every R_s is non-negative and bounded by the shell binding energy; at
        // least one (the K shell, ~6.4 keV fluorescence) is strictly positive.
        let mut any_positive = false;
        for (ri, s) in fe.subshell_radiative_energy.iter().zip(&fe.shells) {
            assert!(
                *ri >= 0.0 && *ri <= s.binding_energy + 1.0,
                "R = {ri}, binding = {}",
                s.binding_energy
            );
            any_positive |= *ri > 0.0;
        }
        assert!(any_positive, "Fe should have positive fluorescence energy");

        // Just above the Fe K-edge (~7.11 keV) photoelectric dominates, so the
        // fluorescence subtraction must lower the heating below the full-E value.
        let e = 8_000.0;
        let micro = fe.calculate_xs(e);
        let alpha = e / MASS_ELECTRON_EV;
        let full_e_heating = e * micro.photoelectric
            + e * compton_energy_transfer_fraction(alpha) * micro.incoherent
            + (e - 2.0 * MASS_ELECTRON_EV).max(0.0) * micro.pair_production;
        assert!(micro.heating > 0.0, "heating must stay positive");
        assert!(
            micro.heating < full_e_heating,
            "corrected heating {} should be below full-E {}",
            micro.heating,
            full_e_heating
        );
    }

    #[test]
    fn compton_relax_map_splits_lumped_shells_by_occupancy() {
        // The Compton->relaxation map is read from compton.arrow (generated by
        // occupancy grouping). Each Compton shell maps to 0, 1, or 2 relaxation
        // subshells whose weights sum to 1; a lumped (n, l) Compton shell splits
        // across its two j-subshells by occupancy (Fe's 2p shell -> L2, L3 at
        // 1/3, 2/3).
        let fe = get_or_load_element("Fe", yamc_test_path("Fe.arrow").as_str()).unwrap();
        for targets in &fe.compton_relax_map {
            if targets.is_empty() {
                continue;
            }
            let wsum: f64 = targets.iter().map(|t| t.weight).sum();
            assert!(
                (wsum - 1.0).abs() < 1e-9,
                "Compton shell weights should sum to 1, got {}",
                wsum
            );
            for t in targets {
                assert!(t.shell_index < fe.shells.len(), "target index out of range");
            }
        }
        // Fe has lumped Compton shells (2p, 3p, 3d); the first one (2p) splits 1/3, 2/3.
        let two_target: Vec<&Vec<ComptonRelaxTarget>> = fe
            .compton_relax_map
            .iter()
            .filter(|t| t.len() == 2)
            .collect();
        assert!(
            !two_target.is_empty(),
            "Fe should have lumped (two-target) Compton shells"
        );
        let mut ws: Vec<f64> = two_target[0].iter().map(|t| t.weight).collect();
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (ws[0] - 1.0 / 3.0).abs() < 1e-6 && (ws[1] - 2.0 / 3.0).abs() < 1e-6,
            "Fe's first lumped Compton shell (2p) should split 1/3, 2/3, got {:?}",
            ws
        );
    }

    #[test]
    fn compton_heating_subtracts_shell_fluorescence() {
        // A Compton event ionizes the shell sampled from electron_pdf, then
        // relaxes + transports its fluorescence, so the Compton KERMA term must
        // subtract that fluorescence (the Compton analogue of the photoelectric
        // correction, issue #176).
        let fe = get_or_load_element("Fe", yamc_test_path("Fe.arrow").as_str()).unwrap();
        assert!(fe.has_atomic_relaxation);

        // compton_radiative_energy is the electron_pdf- and occupancy-weighted
        // per-shell R_s (reached via compton_relax_map). Positive for Fe (has K
        // fluorescence) and bounded by the K binding energy.
        assert!(
            fe.compton_radiative_energy > 0.0,
            "Fe should have positive Compton fluorescence energy"
        );
        let k_binding = fe
            .shells
            .iter()
            .map(|s| s.binding_energy)
            .fold(0.0_f64, f64::max);
        assert!(fe.compton_radiative_energy <= k_binding + 1.0);

        // Independently recompute and confirm it matches the loader.
        let expected: f64 = fe
            .electron_pdf
            .iter()
            .zip(&fe.compton_relax_map)
            .map(|(&pdf, targets)| {
                pdf * targets
                    .iter()
                    .map(|t| t.weight * fe.subshell_radiative_energy[t.shell_index])
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (fe.compton_radiative_energy - expected).abs() < 1e-12,
            "loader {} != recomputed {}",
            fe.compton_radiative_energy,
            expected
        );

        // At ~1 MeV Compton dominates; the fluorescence subtraction must lower the
        // heating below the un-subtracted value, and the gap must reflect the
        // Compton (not only photoelectric) deficit.
        let e = 1.0e6;
        let micro = fe.calculate_xs(e);
        let alpha = e / MASS_ELECTRON_EV;
        let uncorrected = e * micro.photoelectric
            + e * compton_energy_transfer_fraction(alpha) * micro.incoherent
            + (e - 2.0 * MASS_ELECTRON_EV).max(0.0) * micro.pair_production;
        assert!(micro.heating > 0.0);
        assert!(
            micro.heating < uncorrected,
            "corrected heating {} should be below un-subtracted {}",
            micro.heating,
            uncorrected
        );
        // The Compton deficit alone (incoherent * compton_radiative_energy) is a
        // real, positive part of the gap.
        assert!(micro.incoherent * fe.compton_radiative_energy > 0.0);
    }
}
