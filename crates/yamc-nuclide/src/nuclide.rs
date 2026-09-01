// Struct representing a nuclide - reads from Arrow files
use crate::buffer::F64Buffer;
use crate::load_scope::LoadScope;
use crate::reaction::Reaction;
use once_cell::sync::Lazy;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
#[cfg(feature = "debug_collision")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

// Reaction sampling methods for `Nuclide` live in a child module.
mod sampling;

// Global cache using Weak references: nuclides are automatically freed when
// no Material/Model holds them, preventing unbounded memory growth in loops.
static GLOBAL_NUCLIDE_CACHE: Lazy<Mutex<HashMap<String, Weak<Nuclide>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// =============================================================================
// Debug collision tracing for nuclide reaction sampling
// =============================================================================
#[cfg(feature = "debug_collision")]
static DEBUG_NUCLIDE_ENABLED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "debug_collision")]
static DEBUG_NUCLIDE_CHECKED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "debug_collision")]
static DEBUG_NUCLIDE_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "debug_collision")]
const DEBUG_NUCLIDE_MAX: u64 = 200;
#[cfg(feature = "debug_collision")]
const DEBUG_ENERGY_MIN: f64 = 2000.0;
#[cfg(feature = "debug_collision")]
const DEBUG_ENERGY_MAX: f64 = 5000.0;

#[cfg(feature = "debug_collision")]
fn is_debug_nuclide() -> bool {
    if !DEBUG_NUCLIDE_CHECKED.load(Ordering::Relaxed) {
        let is_set = std::env::var("YAMC_DEBUG_COLLISION").is_ok();
        DEBUG_NUCLIDE_ENABLED.store(is_set, Ordering::Relaxed);
        DEBUG_NUCLIDE_CHECKED.store(true, Ordering::Relaxed);
    }
    DEBUG_NUCLIDE_ENABLED.load(Ordering::Relaxed)
}

#[cfg(not(feature = "debug_collision"))]
#[inline(always)]
#[allow(dead_code)]
fn is_debug_nuclide() -> bool {
    false
}

#[cfg(feature = "debug_collision")]
#[inline]
fn energy_in_debug_range_nuc(energy: f64) -> bool {
    (DEBUG_ENERGY_MIN..=DEBUG_ENERGY_MAX).contains(&energy)
}

#[cfg(not(feature = "debug_collision"))]
#[inline(always)]
#[allow(dead_code)]
fn energy_in_debug_range_nuc(_energy: f64) -> bool {
    false
}

// Scattering MTs - EXCLUDING synthetic MT 4 (inelastic is represented by MT 50-91)
// Note: MT 16 is often redundant in libraries where level-specific (n,2n) reactions
// (MT 875-890) are available. The redundant flag (an Arrow column) should be checked.
pub const SCATTERING_MTS_NON_INELASTIC: &[i32] = &[
    2, // elastic
    5, 11, 16, 17, 22, 23, 24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 41, 42, 44, 45, 152, 153,
    154, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
    174, 175, 176, 177, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 190, 194, 195, 196,
    198, 199, 200,
    // Level-specific (n,2n) reactions - MT 875-890
    // These have proper product distributions (yield=2, energy distributions)
    875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890,
];

/// Canonical visiting order for the NON-elastic neutron-producing channels
/// when a collision samples *which* one of them occurs (issue #111).
///
/// The reaction-type draw is a cumulative walk: a single uniform `xi_mt` is
/// compared against the running sum of the candidate channels' partial cross
/// sections. The channel it lands on therefore depends on the ORDER the
/// candidates are visited in, so CPU and GPU must walk the same order to make
/// the same choice from the same `xi_mt`. This table is that order, and it is
/// also the slot layout of the GPU's flat per-MT buffers (slot `k` carries the
/// data for `INELASTIC_MT_SLOTS[k]`); yamc-gpu re-exports it as `MT_SLOTS`, so
/// there is exactly one definition.
///
/// Entries 0..=40 are the discrete-level + continuum inelastic series
/// MT 51..=91 (single neutron out). Entry 41 is MT 16 ((n,2n)) and 42 is MT 17
/// ((n,3n)); entries 43..=47 are the charged-particle-out + neutron MTs
/// MT 22 / 28 / 32 / 33 / 34; entries 48..=55 close the remaining
/// neutron-emitting coverage (MT 5, 23, 24, 25, 37, 41, 44, 45); entries
/// 56..=61 are the breakup channels MT 11 / 29 / 30 / 35 / 36 / 42 (issue
/// #106).
///
/// The order is historical (it grew by appending as GPU coverage was
/// extended) rather than meaningful: any fixed permutation samples each
/// channel with exactly the same probability, so this is a stream/realization
/// convention, not physics. It only has to be the SAME on both backends.
/// Entries 56..=61 are deliberately appended in ascending MT order, which is
/// the storage order the CPU used to visit them in when they were untabled, so
/// adding them renumbers no existing stream.
///
/// A nuclide may still carry non-elastic scattering MTs that are absent from
/// this table (MT 152..200 and 875..890, both in
/// [`SCATTERING_MTS_NON_INELASTIC`]). Those are appended after the tabled
/// entries by [`FastXSGrid::build_inelastic_walk_order`]; see its docs for the
/// coverage gap that leaves on the GPU. In ENDF/B-VIII.1 they appear in one
/// nuclide (La139); TENDL-2017 and TENDL-2025 carry neither.
pub const INELASTIC_MT_SLOTS: [i32; 62] = [
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    91, // 41 entries: MT 51..=91
    16, // (n,2n)
    17, // (n,3n)
    22, // (n,n'α)
    28, // (n,n'p)
    32, // (n,n'd)
    33, // (n,n't)
    34, // (n,n'³He)
    5,  // (n,misc / catch-all neutron-emitting)
    23, // (n,n'3α)
    24, // (n,2nα)
    25, // (n,3nα)
    37, // (n,4n)
    41, // (n,2np)
    44, // (n,n'2p)
    45, // (n,n'pα)
    // Breakup channels (issue #106). Rare in ENDF/B-VIII.1 (MT 11 / 29 in 26
    // nuclides each, MT 30 in 11, MT 42 in 20, MT 36 in La139 alone) but
    // near-universal in TENDL: MT 11 in 541 of 558 TENDL-2017 nuclides and
    // 1521 of 1649 in TENDL-2025, MT 42 in 525 and 1465. Without slots their
    // cross section fell into the GPU's derived absorption, so the GPU killed
    // neutrons the CPU scattered.
    11, // (n,2nd)
    29, // (n,n'3α)
    30, // (n,2n2α)
    35, // (n,n'd2α)
    36, // (n,n't2α)
    42, // (n,3np)
];

// Cross-section indices for FastXSGrid
const XS_TOTAL: usize = 0;
const XS_ABSORPTION: usize = 1;
const XS_SCATTERING: usize = 2;
const XS_FISSION: usize = 3;

/// Enum representing the sampled reaction type
/// Used to avoid synthetic MT numbers and provide a cleaner interface
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReactionType {
    /// Scattering reaction (elastic, inelastic, n2n, etc.)
    /// Caller should use sample_scattering_constituent to get the specific reaction
    Scattering,
    /// Absorption reaction (capture, charged particle production, etc.)
    Absorption,
    /// Fission reaction
    Fission,
}

/// Microscopic cross-section breakdown for the sampled collision nuclide
/// (URR-adjusted when applicable), as returned by [`Nuclide::collision_xs`].
///
/// Used by survival biasing (implicit capture): the transport loop consumes
/// the ratios `scatter / total` (survival factor) and `fission / total`
/// (expected fission progeny per collision), so microscopic values for the
/// nuclide picked by `Material::sample_collision_data` are exactly what is
/// needed; the expectation over nuclide selection reproduces the material
/// level ratios.
#[derive(Debug, Clone, Copy)]
pub struct CollisionXs {
    /// Total cross-section (barns).
    pub total: f64,
    /// Scattering cross-section (elastic + inelastic, including (n,xn)).
    pub scatter: f64,
    /// Fission cross-section (zero for non-fissionable nuclides).
    pub fission: f64,
}

/// Reaction-channel partial cross-sections at a collision site, URR-adjusted
/// exactly as [`Nuclide::sample_reaction_type`], for the shared GPU/CPU
/// reaction-type split (issue #111). The four partials sum to the (URR-adjusted)
/// total, so the analog split draws one uniform `xi2` and partitions
/// `[elastic | inelastic | fission | absorption]` against them, mirroring the
/// GPU twin's four-way branch.
///
/// The scattering bucket is split into elastic (`sigma_e`, MT 2) and everything
/// else (`sigma_i`: inelastic levels plus the neutron-producing (n,xn)/(n,n'x)
/// channels) by the *smooth* elastic/scatter ratio, mirroring the existing
/// two-step `sample_reaction_type` -> `sample_scattering_constituent` selection.
/// That keeps the marginal channel probabilities (and thus OpenMC parity)
/// identical; only the cumulative ordering of the draw changes.
#[derive(Debug, Clone, Copy)]
pub struct ReactionPartials {
    /// Elastic (MT 2) cross-section.
    pub sigma_e: f64,
    /// Absorption (capture; URR-adjusted) cross-section.
    pub sigma_a: f64,
    /// Inelastic + other neutron-producing scatter (scatter minus elastic).
    pub sigma_i: f64,
    /// Fission cross-section (zero for non-fissionable nuclides).
    pub sigma_f: f64,
}

/// Pre-computed cross-section data for fast lookup
///
/// Instead of doing binary search on the full energy grid for each cross-section lookup,
/// we pre-compute a logarithmic grid index that maps log(E) to a narrow range in the
/// energy grid, reducing binary search from O(log N) to O(1-2) iterations.
#[derive(Debug, Clone, Default)]
pub struct FastXSGrid {
    /// Logarithmic grid index: maps a log(E) bin to a starting index in
    /// `energy`. 32-bit because the column is `int32` on disk and the loader has
    /// range-checked it against the grid (issue #482); the largest index in any
    /// published library is 163,746.
    pub log_grid_index: Vec<u32>,
    /// Log of minimum energy in grid
    pub log_e_min: f64,
    /// Inverse of log bin width for fast bin calculation
    pub inv_log_delta: f64,
    /// Pre-computed cross-sections at each energy point: [total, absorption, scattering, fission]
    pub xs: Vec<[f64; 4]>,
    /// Energy grid for this temperature. `fast_xs.arrow` ships its own copy of
    /// the nuclide's union grid; where the two agree this is a zero-copy view of
    /// the nuclide's, and otherwise a copy of the accelerator's own column.
    pub energy: F64Buffer,
    /// MT numbers of scattering reactions, parallel to `scatter_mt_reactions`.
    /// Length = `n_scatter_mts`. `scatter_mt_xs` is indexed by energy-major
    /// then MT, same as the on-disk Arrow layout.
    pub scatter_mt_numbers: Vec<i32>,
    /// Row-major `[n_energies, n_scatter_mts]` XS matrix. Access XS at
    /// (energy index i, MT index j) via `scatter_mt_xs[i * n_scatter_mts + j]`.
    /// One contiguous buffer, matching the on-disk layout; uploadable to
    /// device memory as-is.
    pub scatter_mt_xs: F64Buffer,
    /// Reaction pointers for each scattering MT (parallel to
    /// `scatter_mt_numbers`). Only dereffed at sampling time; separated
    /// from the XS buffer so the latter stays pure data.
    pub scatter_mt_reactions: Vec<Arc<Reaction>>,
    /// Index of MT=2 (elastic) within `scatter_mt_numbers`, or `None` if absent.
    /// Cached at load time so the URR hot path avoids a linear scan on every
    /// collision in the URR energy range.
    pub elastic_idx: Option<usize>,
    /// Indices into `scatter_mt_numbers` of every NON-elastic scattering
    /// column, permuted into the canonical [`INELASTIC_MT_SLOTS`] order
    /// (issue #111). Built once at load time by
    /// [`FastXSGrid::build_inelastic_walk_order`] so the per-collision
    /// cumulative walk in [`FastXSGrid::sample_inelastic_scatter_reaction`] is
    /// a single pass over a precomputed permutation rather than a search
    /// through the slot table.
    pub inelastic_walk_order: Vec<usize>,
    /// Cached reaction for MT 101 (absorption) - avoids HashMap lookup in hot path
    pub reaction_absorption: Option<Arc<Reaction>>,
    /// MT numbers of fission reactions, parallel to `fission_mt_reactions`.
    /// Contains MT 18, 19, 20, 21, 38 (non-redundant).
    pub fission_mt_numbers: Vec<i32>,
    /// Row-major `[n_energies, n_fission_mts]` XS matrix for fission channels.
    pub fission_mt_xs: F64Buffer,
    /// Reaction pointers for each fission MT.
    pub fission_mt_reactions: Vec<Arc<Reaction>>,
    /// True if MT 19 (first-chance fission) exists, indicating partial fission data.
    /// When true, we need to sample which fission reaction occurs.
    pub has_partial_fission: bool,
    /// Pre-computed n_gamma (MT=102) cross-section at each energy point.
    /// Used by URR to correctly compute smooth_total_for_ratio when multiply_smooth=false.
    /// This separates n_gamma from other absorption (n,p), (n,alpha), etc.
    pub xs_ngamma: F64Buffer,
    /// Pre-computed photon production cross-section at each energy point.
    /// photon_prod[i] = SUM over reactions { SUM over photon products { reaction_xs[i] * yield(E[i]) } }
    pub photon_prod: F64Buffer,
    /// MT numbers of photon-producing reactions, parallel to `photon_rxn_reactions`.
    pub photon_rxn_mt_numbers: Vec<i32>,
    /// Row-major `[n_energies, n_photon_rxn_mts]` XS matrix for photon-producing
    /// reactions. Used by sample_photon_product for grid-consistent XS lookup.
    pub photon_rxn_xs: F64Buffer,
    /// Reaction pointers for each photon-producing MT.
    pub photon_rxn_reactions: Vec<Arc<Reaction>>,
    /// MT numbers for absorption-only (D1S decay-photon-lookup) channels,
    /// parallel to `absorption_mt_xs`. Contains non-redundant absorption
    /// reactions (MT 103-117 etc.) not already in photon_rxn_xs.
    pub absorption_mt_numbers: Vec<i32>,
    /// Row-major `[n_energies, n_absorption_mts]` XS matrix for absorption-only
    /// channels used by D1S decay photon lookup.
    pub absorption_mt_xs: F64Buffer,
    /// Pre-computed delayed photon scaling factor at each energy point.
    /// For fission reactions, photon production yield is multiplied by this factor
    /// to account for delayed photons: f = (prompt_energy + delayed_energy) / prompt_energy.
    /// Empty if nuclide has no fission_energy_release data.
    /// f = (prompt_energy + delayed_energy) / prompt_energy.
    pub delayed_photon_scaling: F64Buffer,
}

impl FastXSGrid {
    /// Permutation of the non-elastic `scatter_mt_numbers` columns into the
    /// canonical [`INELASTIC_MT_SLOTS`] order (issue #111). Columns whose MT is
    /// in the table come first, in table order; the elastic column is dropped
    /// (the elastic-vs-inelastic split is made earlier, by `xi2`).
    ///
    /// Columns whose MT is NOT in the table (MT 152..200 and 875..890) are
    /// appended afterwards in storage order. The GPU has no slot for those MTs
    /// at all: their cross section is not in its `xs_inelastic` aggregate, so
    /// it lands in the GPU's derived absorption instead and the GPU never
    /// samples them. That is a GPU *coverage* gap, not an ordering one -- for a
    /// nuclide carrying such an MT the two backends normalize their walks by
    /// different totals, so no permutation can make the selections agree.
    /// Putting them last keeps the CPU's physics complete (every
    /// neutron-producing channel is still sampled with probability proportional
    /// to its partial) while leaving the walk's leading segment aligned with
    /// the GPU's.
    ///
    /// MT 11 / 29 / 30 / 35 / 36 / 42 used to be in that trailing group and are
    /// now slotted (issue #106), which closed the gap for the TENDL libraries,
    /// where MT 11 and 42 appear in the great majority of nuclides. Because
    /// they were appended in ascending MT order before and sit in ascending MT
    /// order at the end of the table now, the walk they produce is unchanged.
    pub fn build_inelastic_walk_order(
        scatter_mt_numbers: &[i32],
        elastic_idx: Option<usize>,
    ) -> Vec<usize> {
        let mut order = Vec::with_capacity(scatter_mt_numbers.len());
        for &slot_mt in INELASTIC_MT_SLOTS.iter() {
            for (j, &mt) in scatter_mt_numbers.iter().enumerate() {
                if mt == slot_mt && Some(j) != elastic_idx {
                    order.push(j);
                }
            }
        }
        for (j, &mt) in scatter_mt_numbers.iter().enumerate() {
            if Some(j) != elastic_idx && !INELASTIC_MT_SLOTS.contains(&mt) {
                order.push(j);
            }
        }
        order
    }

    // ---- MT-XS accessors for the flat row-major per-group buffers ----------

    /// XS at (i_energy, mt_idx) for `scatter_mt_xs`; 0.0 if out of bounds.
    #[inline]
    pub fn scatter_xs_at(&self, i_energy: usize, mt_idx: usize) -> f64 {
        let n = self.scatter_mt_numbers.len();
        if n == 0 {
            return 0.0;
        }
        let idx = i_energy * n + mt_idx;
        self.scatter_mt_xs.get(idx).copied().unwrap_or(0.0)
    }

    /// Interpolated XS at (i_grid + f) for `scatter_mt_xs` MT slot `mt_idx`.
    #[inline]
    pub fn scatter_xs_interp(&self, i_grid: usize, f: f64, mt_idx: usize) -> f64 {
        let n = self.scatter_mt_numbers.len();
        if n == 0 {
            return 0.0;
        }
        let n_e = self.scatter_mt_xs.len() / n;
        if i_grid + 1 < n_e {
            let x0 = self.scatter_mt_xs[i_grid * n + mt_idx];
            let x1 = self.scatter_mt_xs[(i_grid + 1) * n + mt_idx];
            x0 + f * (x1 - x0)
        } else if i_grid < n_e {
            self.scatter_mt_xs[i_grid * n + mt_idx]
        } else if n_e > 0 {
            self.scatter_mt_xs[(n_e - 1) * n + mt_idx]
        } else {
            0.0
        }
    }

    /// XS at (i_energy, mt_idx) for `fission_mt_xs`; 0.0 if out of bounds.
    #[inline]
    pub fn fission_xs_at(&self, i_energy: usize, mt_idx: usize) -> f64 {
        let n = self.fission_mt_numbers.len();
        if n == 0 {
            return 0.0;
        }
        let idx = i_energy * n + mt_idx;
        self.fission_mt_xs.get(idx).copied().unwrap_or(0.0)
    }

    /// Interpolated XS at (i_grid + f) for `fission_mt_xs` MT slot `mt_idx`.
    #[inline]
    pub fn fission_xs_interp(&self, i_grid: usize, f: f64, mt_idx: usize) -> f64 {
        let n = self.fission_mt_numbers.len();
        if n == 0 {
            return 0.0;
        }
        let n_e = self.fission_mt_xs.len() / n;
        if i_grid + 1 < n_e {
            let x0 = self.fission_mt_xs[i_grid * n + mt_idx];
            let x1 = self.fission_mt_xs[(i_grid + 1) * n + mt_idx];
            x0 + f * (x1 - x0)
        } else if i_grid < n_e {
            self.fission_mt_xs[i_grid * n + mt_idx]
        } else if n_e > 0 {
            self.fission_mt_xs[(n_e - 1) * n + mt_idx]
        } else {
            0.0
        }
    }

    /// XS at (i_energy, mt_idx) for `photon_rxn_xs`; 0.0 if out of bounds.
    #[inline]
    pub fn photon_rxn_xs_at(&self, i_energy: usize, mt_idx: usize) -> f64 {
        let n = self.photon_rxn_mt_numbers.len();
        if n == 0 {
            return 0.0;
        }
        let idx = i_energy * n + mt_idx;
        self.photon_rxn_xs.get(idx).copied().unwrap_or(0.0)
    }

    /// Interpolated XS at (i_grid + f) for `photon_rxn_xs` MT slot `mt_idx`.
    #[inline]
    pub fn photon_rxn_xs_interp(&self, i_grid: usize, f: f64, mt_idx: usize) -> f64 {
        let n = self.photon_rxn_mt_numbers.len();
        if n == 0 {
            return 0.0;
        }
        let n_e = self.photon_rxn_xs.len() / n;
        if i_grid + 1 < n_e {
            let x0 = self.photon_rxn_xs[i_grid * n + mt_idx];
            let x1 = self.photon_rxn_xs[(i_grid + 1) * n + mt_idx];
            x0 + f * (x1 - x0)
        } else if i_grid < n_e {
            self.photon_rxn_xs[i_grid * n + mt_idx]
        } else if n_e > 0 {
            self.photon_rxn_xs[(n_e - 1) * n + mt_idx]
        } else {
            0.0
        }
    }

    /// Look up cross-sections at given energy, returning (total, absorption, scattering, fission, grid_idx, interp_factor)
    #[inline]
    pub fn lookup(&self, energy: f64) -> (f64, f64, f64, f64) {
        let n = self.energy.len();
        if n == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        // Handle boundary cases
        if energy <= self.energy[0] {
            let xs = &self.xs[0];
            return (
                xs[XS_TOTAL],
                xs[XS_ABSORPTION],
                xs[XS_SCATTERING],
                xs[XS_FISSION],
            );
        }
        if energy >= self.energy[n - 1] {
            let xs = &self.xs[n - 1];
            return (
                xs[XS_TOTAL],
                xs[XS_ABSORPTION],
                xs[XS_SCATTERING],
                xs[XS_FISSION],
            );
        }

        // Use logarithmic grid to find narrow search range
        let log_e = energy.ln();
        let bin = ((log_e - self.log_e_min) * self.inv_log_delta) as usize;
        let bin = bin.min(self.log_grid_index.len() - 2);

        let i_low = self.log_grid_index[bin] as usize;
        let i_high = self.log_grid_index[bin + 1] as usize + 1;

        // Binary search in narrow range
        let i_grid = i_low
            + self.energy[i_low..i_high]
                .partition_point(|&e| e <= energy)
                .saturating_sub(1);
        let i_grid = i_grid.min(n - 2);

        // Check for rare case where two energy points are the same
        let i_grid =
            if i_grid + 1 < self.energy.len() && self.energy[i_grid] == self.energy[i_grid + 1] {
                i_grid + 1
            } else {
                i_grid
            };
        let i_grid = i_grid.min(n - 2);

        // Linear interpolation
        let e0 = self.energy[i_grid];
        let e1 = self.energy[i_grid + 1];
        let f = (energy - e0) / (e1 - e0);

        let xs0 = &self.xs[i_grid];
        let xs1 = &self.xs[i_grid + 1];

        (
            xs0[XS_TOTAL] + f * (xs1[XS_TOTAL] - xs0[XS_TOTAL]),
            xs0[XS_ABSORPTION] + f * (xs1[XS_ABSORPTION] - xs0[XS_ABSORPTION]),
            xs0[XS_SCATTERING] + f * (xs1[XS_SCATTERING] - xs0[XS_SCATTERING]),
            xs0[XS_FISSION] + f * (xs1[XS_FISSION] - xs0[XS_FISSION]),
        )
    }

    /// Look up grid index and interpolation factor for given energy.
    /// Returns (grid_index, interp_factor) for use with scatter_mt_xs.
    #[inline]
    pub fn lookup_grid_index(&self, energy: f64) -> (usize, f64) {
        let n = self.energy.len();
        if n == 0 {
            return (0, 0.0);
        }

        // Handle boundary cases
        if energy <= self.energy[0] {
            return (0, 0.0);
        }
        if energy >= self.energy[n - 1] {
            return (n - 1, 0.0);
        }

        // Use logarithmic grid to find narrow search range
        let log_e = energy.ln();
        let bin = ((log_e - self.log_e_min) * self.inv_log_delta) as usize;
        let bin = bin.min(self.log_grid_index.len() - 2);

        let i_low = self.log_grid_index[bin] as usize;
        let i_high = self.log_grid_index[bin + 1] as usize + 1;

        // Binary search in narrow range
        let i_grid = i_low
            + self.energy[i_low..i_high]
                .partition_point(|&e| e <= energy)
                .saturating_sub(1);
        let i_grid = i_grid.min(n - 2);

        // Check for rare case where two energy points are the same
        let i_grid =
            if i_grid + 1 < self.energy.len() && self.energy[i_grid] == self.energy[i_grid + 1] {
                i_grid + 1
            } else {
                i_grid
            };
        let i_grid = i_grid.min(n - 2);

        // Compute interpolation factor
        let e0 = self.energy[i_grid];
        let e1 = self.energy[i_grid + 1];
        let f = (energy - e0) / (e1 - e0);

        (i_grid, f)
    }

    /// Look up n_gamma (MT=102) cross-section at given grid index and interpolation factor.
    /// Returns 0.0 if xs_ngamma is empty (MT=102 not available).
    #[inline]
    pub fn lookup_ngamma(&self, i_grid: usize, f: f64) -> f64 {
        if self.xs_ngamma.is_empty() {
            return 0.0;
        }
        if i_grid + 1 < self.xs_ngamma.len() {
            self.xs_ngamma[i_grid] + f * (self.xs_ngamma[i_grid + 1] - self.xs_ngamma[i_grid])
        } else if !self.xs_ngamma.is_empty() {
            self.xs_ngamma[i_grid.min(self.xs_ngamma.len() - 1)]
        } else {
            0.0
        }
    }

    /// Look up photon production cross-section at given grid index and interpolation factor.
    /// Returns 0.0 if photon_prod is empty.
    #[inline]
    pub fn lookup_photon_prod(&self, i_grid: usize, f: f64) -> f64 {
        if self.photon_prod.is_empty() {
            return 0.0;
        }
        if i_grid + 1 < self.photon_prod.len() {
            self.photon_prod[i_grid] + f * (self.photon_prod[i_grid + 1] - self.photon_prod[i_grid])
        } else if !self.photon_prod.is_empty() {
            self.photon_prod[i_grid.min(self.photon_prod.len() - 1)]
        } else {
            0.0
        }
    }

    /// Sample a scattering constituent using pre-computed XS data.
    /// Returns a reference to the sampled Reaction directly (no HashMap lookup needed).
    /// This is much faster than iterating and calling cross_section_at for each MT.
    #[inline]
    pub fn sample_scatter_reaction<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        rng: &mut R,
    ) -> Option<&Reaction> {
        let n_mts = self.scatter_mt_numbers.len();
        if n_mts == 0 {
            return None;
        }

        let (i_grid, f) = self.lookup_grid_index(energy);

        // First pass: compute total scattering XS
        let mut total_xs = 0.0;
        for j in 0..n_mts {
            let xs = self.scatter_xs_interp(i_grid, f, j);
            if xs > 0.0 {
                total_xs += xs;
            }
        }

        if total_xs <= 0.0 {
            return None;
        }

        // Second pass: sample and return reaction directly
        let xi = rng.random_range(0.0..total_xs);
        let mut accum = 0.0;
        for j in 0..n_mts {
            let xs = self.scatter_xs_interp(i_grid, f, j);
            if xs > 0.0 {
                accum += xs;
                if xi < accum {
                    return Some(self.scatter_mt_reactions[j].as_ref());
                }
            }
        }

        // Return first reaction as fallback
        self.scatter_mt_reactions.first().map(|r| r.as_ref())
    }

    /// The elastic (MT 2) reaction, if this grid carries one. Used by the
    /// analog reaction-type split (issue #111) once `xi2` has selected the
    /// elastic channel directly, so the elastic angular table can be fetched
    /// without re-sampling a constituent.
    #[inline]
    pub fn elastic_reaction(&self) -> Option<&Reaction> {
        self.elastic_idx
            .map(|idx| self.scatter_mt_reactions[idx].as_ref())
    }

    /// Select a *non-elastic* scattering constituent proportional to its smooth
    /// cross-section, driven by a pre-drawn PCG uniform `xi_mt` in `(0, 1]`
    /// (issue #111). The elastic-vs-inelastic split is made earlier by `xi2`, so
    /// the elastic (`elastic_idx`) column is excluded here. Returns `None` only
    /// when there is no non-elastic scattering at this energy.
    ///
    /// The cumulative walk visits the candidates in `inelastic_walk_order`, the
    /// canonical [`INELASTIC_MT_SLOTS`] order the GPU kernel sweeps its per-MT
    /// slots in, so the same `xi_mt` picks the same MT on both backends. (Both
    /// orders sample each channel with probability proportional to its partial
    /// cross section; only which channel a *given* `xi_mt` maps to changes.)
    /// The `accum >= target` test and the walk direction match the kernel's too,
    /// so the two agree even on an exact-equality tie.
    #[inline]
    pub fn sample_inelastic_scatter_reaction(&self, energy: f64, xi_mt: f64) -> Option<&Reaction> {
        if self.inelastic_walk_order.is_empty() {
            return None;
        }
        let (i_grid, f) = self.lookup_grid_index(energy);

        // First pass: total non-elastic scattering XS.
        let mut total_xs = 0.0;
        for &j in &self.inelastic_walk_order {
            let xs = self.scatter_xs_interp(i_grid, f, j);
            if xs > 0.0 {
                total_xs += xs;
            }
        }
        if total_xs <= 0.0 {
            return None;
        }

        // Second pass: cumulative walk against `xi_mt * total`.
        let target = xi_mt * total_xs;
        let mut accum = 0.0;
        for &j in &self.inelastic_walk_order {
            let xs = self.scatter_xs_interp(i_grid, f, j);
            if xs > 0.0 {
                accum += xs;
                if accum >= target {
                    return Some(self.scatter_mt_reactions[j].as_ref());
                }
            }
        }

        // Numerical fallback: the last candidate in the walk order.
        self.inelastic_walk_order
            .last()
            .map(|&j| self.scatter_mt_reactions[j].as_ref())
    }

    /// Sample which fission reaction occurs at `energy`, proportional to the
    /// partial fission cross sections, driven by a PCG uniform in `(0, 1]` that
    /// `draw_xi` supplies (issue #418). Returns the first fission reaction
    /// without drawing when the evaluation has no partial channels.
    ///
    /// `draw_xi` is a closure rather than a pre-drawn `f64` (the shape
    /// [`Self::sample_inelastic_scatter_reaction`] uses) because the draw has to
    /// be SKIPPED, not taken and discarded, whenever there is no choice to make.
    /// The GPU kernel sums MT 18/19/20/21/38 into a single fission cross section
    /// and makes no channel-selection draw at all, so a draw taken here for a
    /// single-channel nuclide would walk the CPU's stream off the kernel's
    /// schedule at every fission collision. `has_partial_fission` is a property
    /// of the evaluation rather than of the history, so gating on it keeps the
    /// two backends in lockstep for all 87 of ENDF/B-VIII.1's single-channel
    /// fissionables and costs a draw only on U240, the one nuclide that carries
    /// partial channels (and which the GPU cannot follow anyway, see #424).
    ///
    /// The `accum >= target` test, the walk direction and the `.last()` fallback
    /// match [`Self::sample_inelastic_scatter_reaction`], so the two neighbouring
    /// walks resolve an exact-equality tie the same way.
    #[inline]
    pub fn sample_fission_reaction(
        &self,
        energy: f64,
        draw_xi: impl FnOnce() -> f64,
    ) -> Option<&Reaction> {
        let n_mts = self.fission_mt_numbers.len();
        if n_mts == 0 {
            return None;
        }

        // No partial channels: the single fission MT is the answer, and a draw
        // taken for it would desynchronise the shared stream (see above).
        if !self.has_partial_fission {
            return self.fission_mt_reactions.first().map(|r| r.as_ref());
        }

        // Past this gate the nuclide always costs exactly one draw, so the stream
        // position depends only on `has_partial_fission` and not on the incident
        // energy: an early return below must not skip it.
        let xi_mt = draw_xi();

        let (i_grid, f) = self.lookup_grid_index(energy);

        // First pass: compute total fission XS at this energy
        let mut total_xs = 0.0;
        for j in 0..n_mts {
            let xs = self.fission_xs_interp(i_grid, f, j);
            if xs > 0.0 {
                total_xs += xs;
            }
        }

        if total_xs <= 0.0 {
            return self.fission_mt_reactions.first().map(|r| r.as_ref());
        }

        // Second pass: cumulative walk against `xi_mt * total`.
        let target = xi_mt * total_xs;
        let mut accum = 0.0;
        for j in 0..n_mts {
            let xs = self.fission_xs_interp(i_grid, f, j);
            if xs > 0.0 {
                accum += xs;
                if accum >= target {
                    return Some(self.fission_mt_reactions[j].as_ref());
                }
            }
        }

        // Numerical fallback: the last channel in the walk.
        self.fission_mt_reactions.last().map(|r| r.as_ref())
    }
}

/// Flatten a list of per-MT XS vectors into a row-major `[n_energies, n_mts]`
/// buffer, matching the on-disk Arrow layout and `FastXSGrid`'s in-memory
/// layout. Missing or short inner Vecs are padded with zeros.
pub(crate) fn flatten_row_major(per_mt: &[Vec<f64>], n_energies: usize) -> Vec<f64> {
    let n_mts = per_mt.len();
    if n_mts == 0 || n_energies == 0 {
        return Vec::new();
    }
    let mut flat = Vec::with_capacity(n_energies * n_mts);
    for i in 0..n_energies {
        for xs_vec in per_mt.iter() {
            flat.push(xs_vec.get(i).copied().unwrap_or(0.0));
        }
    }
    flat
}

/// Look up element name from atomic number Z.
pub fn element_name_from_z(z: u32) -> Option<String> {
    const ELEMENTS: &[(u32, &str)] = &[
        (1, "hydrogen"),
        (2, "helium"),
        (3, "lithium"),
        (4, "beryllium"),
        (5, "boron"),
        (6, "carbon"),
        (7, "nitrogen"),
        (8, "oxygen"),
        (9, "fluorine"),
        (10, "neon"),
        (11, "sodium"),
        (12, "magnesium"),
        (13, "aluminum"),
        (14, "silicon"),
        (15, "phosphorus"),
        (16, "sulfur"),
        (17, "chlorine"),
        (18, "argon"),
        (19, "potassium"),
        (20, "calcium"),
        (21, "scandium"),
        (22, "titanium"),
        (23, "vanadium"),
        (24, "chromium"),
        (25, "manganese"),
        (26, "iron"),
        (27, "cobalt"),
        (28, "nickel"),
        (29, "copper"),
        (30, "zinc"),
        (31, "gallium"),
        (32, "germanium"),
        (33, "arsenic"),
        (34, "selenium"),
        (35, "bromine"),
        (36, "krypton"),
        (37, "rubidium"),
        (38, "strontium"),
        (39, "yttrium"),
        (40, "zirconium"),
        (41, "niobium"),
        (42, "molybdenum"),
        (43, "technetium"),
        (44, "ruthenium"),
        (45, "rhodium"),
        (46, "palladium"),
        (47, "silver"),
        (48, "cadmium"),
        (49, "indium"),
        (50, "tin"),
        (51, "antimony"),
        (52, "tellurium"),
        (53, "iodine"),
        (54, "xenon"),
        (55, "cesium"),
        (56, "barium"),
        (57, "lanthanum"),
        (58, "cerium"),
        (59, "praseodymium"),
        (60, "neodymium"),
        (61, "promethium"),
        (62, "samarium"),
        (63, "europium"),
        (64, "gadolinium"),
        (65, "terbium"),
        (66, "dysprosium"),
        (67, "holmium"),
        (68, "erbium"),
        (69, "thulium"),
        (70, "ytterbium"),
        (71, "lutetium"),
        (72, "hafnium"),
        (73, "tantalum"),
        (74, "tungsten"),
        (75, "rhenium"),
        (76, "osmium"),
        (77, "iridium"),
        (78, "platinum"),
        (79, "gold"),
        (80, "mercury"),
        (81, "thallium"),
        (82, "lead"),
        (83, "bismuth"),
        (84, "polonium"),
        (85, "astatine"),
        (86, "radon"),
        (87, "francium"),
        (88, "radium"),
        (89, "actinium"),
        (90, "thorium"),
        (91, "protactinium"),
        (92, "uranium"),
        (93, "neptunium"),
        (94, "plutonium"),
        (95, "americium"),
        (96, "curium"),
    ];
    ELEMENTS
        .iter()
        .find(|(num, _)| *num == z)
        .map(|(_, name)| name.to_string())
}

/// Helper function to check if an MT number is a scattering reaction (excludes MT 4 synthetic)
#[inline]
pub fn is_scattering_mt(mt: i32) -> bool {
    // Inelastic constituent MTs (50-91)
    if (50..92).contains(&mt) {
        return true;
    }
    // Other scattering MTs (elastic and non-inelastic scattering)
    SCATTERING_MTS_NON_INELASTIC.contains(&mt)
}

/// Helper function to check if an MT number is a fission reaction
/// MT 18 = total fission, MT 19 = first-chance, MT 20 = second-chance,
/// MT 21 = third-chance, MT 38 = fourth-chance fission
///
/// Tests membership of [`crate::reaction_product::energy::FISSION_CHI_MTS`],
/// which is the one definition of the set, so this predicate and the per-channel
/// chi cache cannot come to disagree about which MTs are fission (issue #425).
#[inline]
pub fn is_fission_mt(mt: i32) -> bool {
    crate::reaction_product::energy::FISSION_CHI_MTS.contains(&mt)
}

/// Helper function to check if an MT number is an absorption reaction.
/// Absorption reactions include (n,gamma), (n,p), (n,alpha), and other
/// charged-particle-out or multi-particle-out reactions.
/// MT 102 = (n,gamma), 103-117 = charged particle emission,
/// 155, 182, 191-193, 197 = special absorption, 600-849 = lumped production.
#[inline]
pub fn is_absorption_mt(mt: i32) -> bool {
    matches!(
        mt,
        102 | 103
            | 104
            | 105
            | 106
            | 107
            | 108
            | 109
            | 111
            | 112
            | 113
            | 114
            | 115
            | 116
            | 117
            | 155
            | 182
            | 191
            | 192
            | 193
            | 197
    ) || (600..850).contains(&mt)
}

/// Enum to represent either an MT number or reaction name for flexible reaction identification
#[derive(Debug, Clone)]
pub enum ReactionIdentifier {
    Mt(i32),
    Name(String),
}

impl From<i32> for ReactionIdentifier {
    fn from(mt: i32) -> Self {
        ReactionIdentifier::Mt(mt)
    }
}

impl From<String> for ReactionIdentifier {
    fn from(name: String) -> Self {
        ReactionIdentifier::Name(name)
    }
}

impl From<&str> for ReactionIdentifier {
    fn from(name: &str) -> Self {
        ReactionIdentifier::Name(name.to_string())
    }
}

/// Clear the global nuclide cache.
#[allow(dead_code)]
pub fn clear_nuclide_cache() {
    match GLOBAL_NUCLIDE_CACHE.lock() {
        Ok(mut cache) => cache.clear(),
        Err(poisoned) => poisoned.into_inner().clear(),
    }
}

/// Data for fission neutron production (nu-bar)
/// Stores the average number of neutrons produced per fission as a function of energy
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FissionNuData {
    /// Incident neutron energies (eV)
    pub energy: Vec<f64>,
    /// Nu-bar values (average neutrons per fission)
    pub nu: Vec<f64>,
}

impl FissionNuData {
    /// Evaluate nu-bar at a given incident energy using linear interpolation
    pub fn evaluate(&self, energy: f64) -> f64 {
        if self.energy.is_empty() || self.nu.is_empty() {
            return 2.5; // Default value
        }

        // Clamp to bounds
        if energy <= self.energy[0] {
            return self.nu[0];
        }
        if energy >= *self.energy.last().unwrap() {
            return *self.nu.last().unwrap();
        }

        // Binary search for interpolation
        let idx = self.energy.partition_point(|&e| e < energy);
        if idx == 0 {
            return self.nu[0];
        }

        let e0 = self.energy[idx - 1];
        let e1 = self.energy[idx];
        let nu0 = self.nu[idx - 1];
        let nu1 = self.nu[idx];

        // Linear interpolation
        let f = (energy - e0) / (e1 - e0);
        nu0 + f * (nu1 - nu0)
    }
}

/// Core data model for a single nuclide and its reaction cross section data.
///
/// A `Nuclide` mirrors (and is loaded from) the Arrow IPC data containing
/// metadata plus reaction channel data at one or more temperatures. Reaction
/// data are organized by temperature key (e.g. "294") and then by ENDF/MT
/// number. Each [`Reaction`] holds its own threshold information and (possibly
/// truncated) energy grid relative to the top‑level temperature energy grid.
///
/// Temperatures:
/// * `available_temperatures` always lists every temperature present in the
///   source data – even if a filtered load only materialized a subset.
/// * `loaded_temperatures` tracks the subset actually parsed into `reactions`
///   and `energy` according to caller filtering semantics.
///
/// Performance optimization:
/// * `temperature_keys` provides O(1) lookup from temperature string to index
/// * `reactions` and `fast_xs` are Vec indexed by temp_idx for O(1) access
/// * Hot-path methods accept temp_idx directly to avoid string lookups
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Nuclide {
    /// Canonical nuclide name (e.g. "Li6"). May be derived when absent.
    pub name: Option<String>,
    /// Optional human readable element name (may be absent until nuclear data is loaded).
    pub element: Option<String>,
    /// Element symbol, e.g. "Li".
    pub atomic_symbol: Option<String>,
    /// Atomic (proton) number Z.
    pub atomic_number: Option<u32>,
    /// Neutron number N (may be computed from A - Z if missing).
    pub neutron_number: Option<u32>,
    /// Mass number A.
    pub mass_number: Option<u32>,
    /// Atomic weight ratio (target mass / neutron mass) from the Arrow data.
    pub atomic_weight_ratio: Option<f64>,
    /// Origin / library identifier (e.g. JEFF, ENDF, custom tag).
    pub library: Option<String>,
    /// Top‑level energy grid per temperature (full grid; per‑reaction grids may be threshold‑truncated).
    pub energy: Option<HashMap<String, F64Buffer>>,
    /// Reactions indexed by temperature index (use get_temp_idx to convert temp string to index).
    /// Vec index corresponds to temperature_keys index.
    #[serde(default)]
    pub reactions: Vec<HashMap<i32, Arc<Reaction>>>,
    /// True if any fission MT channel is present.
    pub fissionable: bool,
    /// All temperatures present in the Arrow data regardless of filtering.
    #[serde(skip, default)]
    pub available_temperatures: Vec<String>, // All temps listed in the Arrow data (even if not loaded)
    /// Subset of temperatures actually loaded into `reactions` / `energy`.
    /// Sorted for deterministic ordering and O(1) index lookup.
    #[serde(skip, default)]
    pub loaded_temperatures: Vec<String>, // Subset actually loaded into reactions/energy
    /// Optional path the Arrow data was read from (None for in‑memory sources / WASM).
    #[serde(skip, default)]
    pub data_path: Option<String>, // Path the Arrow data was loaded from (for potential future extension)
    /// Fission nu-bar data (average neutrons per fission as function of energy)
    #[serde(skip, default)]
    pub fission_nu: Option<FissionNuData>,
    /// Pre-computed fast cross-section lookup grid for O(1) energy lookup.
    /// Vec index corresponds to temperature_keys index for O(1) lookup.
    #[serde(skip, default)]
    pub fast_xs: Vec<FastXSGrid>,
    /// URR (Unresolved Resonance Range) probability table data per temperature.
    /// Vec index corresponds to temperature_keys index for O(1) lookup.
    /// None if nuclide has no URR data.
    #[serde(skip, default)]
    pub urr_data: Vec<Option<crate::urr::UrrData>>,
    /// True if any URR data is present for this nuclide.
    #[serde(skip, default)]
    pub urr_present: bool,
    /// Fission energy release terms behind the delayed-photon scaling
    /// `f(E) = (prompt + delayed) / prompt` (issue #369). `None` for the great
    /// majority of nuclides, which carry no `fission_energy_release` data.
    ///
    /// Stored as the evaluation's own functions rather than as values on an
    /// energy grid: neither term is necessarily a polynomial (U235, U238 and
    /// Pu239 tabulate the prompt term), and keeping the function means a
    /// regenerated grid can never silently mismatch a stale vector.
    #[serde(skip, default)]
    pub fission_photon_release: Option<crate::fission_photon::FissionPhotonRelease>,
    /// MF=33 cross-section covariance, when the load asked for it and the
    /// directory had it (issue #514).
    ///
    /// Three-way rather than two: `None` means this nuclide was not loaded with
    /// [`LoadScope::covariance`] set, or was loaded from a directory with no
    /// `covariance.arrow`; `Some(empty)` cannot occur, because an evaluation
    /// with no blocks writes no file. Distinguishing "not asked for" from "not
    /// available" is [`LoadScope::covariance`]'s job, and the pair has to be
    /// read together: a silent `None` on a nuclide nobody asked covariance for
    /// is not a gap in the data.
    ///
    /// Behind an `Arc` because the matrices are the largest thing a nuclide
    /// carries that is not already a shared buffer -- one LB=5 block on a
    /// 145-point grid is 10440 doubles -- and `Nuclide` is `Clone`.
    #[serde(skip, default)]
    pub covariance: Option<std::sync::Arc<Vec<crate::covariance::CovarianceBlock>>>,
    /// Lazily-built flat elastic angular table (issue #111). Routes the
    /// production CPU elastic scatter through the same
    /// `yamc_physics::gpu::flat::elastic_mu_cm` sampler the GPU kernel/twin
    /// use, so the two paths cannot drift. Built on the first elastic
    /// collision and shared read-only across transport threads; reset on
    /// clone and skipped by serde.
    #[serde(skip, default)]
    pub elastic_flat_cache: crate::reaction_product::ElasticFlatCache,
    /// Lazily-built flat fission outgoing-energy (chi) tables (issue #111
    /// fission sub-step). Routes the production CPU fission chi through the
    /// shared `yamc_physics::gpu::flat` fission-spectrum samplers. One slot per
    /// fission MT, each built from that channel's prompt fission neutron product
    /// on the channel's first fission and shared read-only across threads; reset
    /// on clone, skipped by serde.
    ///
    /// Per channel rather than per nuclide because an evaluation with partial
    /// fission channels carries a different prompt spectrum on each (issue
    /// #425).
    #[serde(skip, default)]
    pub fission_chi_flat_cache: crate::reaction_product::FissionChiFlatCache,
    /// Lazily-resolved delayed-neutron groups (issue #364): their yields, and the
    /// yield-weighted fold of their spectra. Same lifetime rules as the prompt chi
    /// cache above. `None` once resolved means the evaluation carries no delayed
    /// data, which is how a nuclide keeps the prompt-only behaviour.
    #[serde(skip, default)]
    pub delayed_neutron_cache: crate::delayed_neutrons::DelayedNeutronCache,
    /// Lazily-built per-MT flat angular tables for DISCRETE inelastic levels
    /// (issue #111 sub-step 3). Routes the production CPU discrete-level
    /// inelastic cosine through the shared `elastic_mu_cm` sampler (the GPU
    /// reuses it for inelastic), so the closed-form-Q level scatter is
    /// bit-identical between backends. Built per level on first use, shared
    /// read-only across threads; reset on clone, skipped by serde.
    #[serde(skip, default)]
    pub inelastic_angle_flat_cache: crate::reaction_product::InelasticAngleFlatCache,
    /// The subset of the Arrow data this nuclide was parsed from (issue #389).
    ///
    /// Defaults to [`LoadScope::full`], so a nuclide built by hand or revived
    /// from serde reads as complete. The global cache consults it to decide
    /// whether an entry it already holds is wide enough for a new request; a
    /// transmutation-scoped load is never handed to transport.
    #[serde(skip, default)]
    pub load_scope: crate::load_scope::LoadScope,
}

impl Nuclide {
    /// Get the temperature index for a given temperature string.
    /// This is used for O(1) indexed access to reactions and fast_xs.
    /// Returns None if temperature is not loaded.
    #[inline]
    pub fn get_temp_idx(&self, temperature: &str) -> Option<usize> {
        // Since loaded_temperatures is sorted and typically contains 1-3 elements,
        // a linear search is faster than binary search due to cache locality
        self.loaded_temperatures
            .iter()
            .position(|t| t == temperature)
    }

    /// Get reactions for a temperature by string key (convenience wrapper).
    /// For hot paths, prefer get_temp_idx + direct Vec indexing.
    #[inline]
    pub fn reactions_for_temp(&self, temperature: &str) -> Option<&HashMap<i32, Arc<Reaction>>> {
        self.get_temp_idx(temperature)
            .map(|idx| &self.reactions[idx])
    }

    /// Get fast_xs for a temperature by string key (convenience wrapper).
    /// For hot paths, prefer get_temp_idx + direct Vec indexing.
    #[inline]
    pub fn fast_xs_for_temp(&self, temperature: &str) -> Option<&FastXSGrid> {
        self.get_temp_idx(temperature)
            .and_then(|idx| self.fast_xs.get(idx))
    }

    /// Get URR data for a temperature by string key.
    /// Returns None if no URR data exists for this temperature.
    #[inline]
    pub fn urr_for_temp(&self, temperature: &str) -> Option<&crate::urr::UrrData> {
        if !self.urr_present {
            return None;
        }
        self.get_temp_idx(temperature)
            .and_then(|idx| self.urr_data.get(idx))
            .and_then(|opt| opt.as_ref())
    }

    /// Stable per-nuclide key used to decorrelate URR probability-table
    /// sampling across the isotopes of a material (issue #204). Each nuclide
    /// must draw an independent probability-table band, so the per-collision
    /// base seed is mixed with this key via [`crate::urr::urr_nuclide_random`].
    ///
    /// Prefers the intrinsic `ZA` (`Z * 1000 + A`); falls back to a non-zero
    /// hash of the canonical name so two distinct nuclides never collide onto
    /// a shared URR stream even if the ZA metadata is missing.
    #[inline]
    pub fn urr_stream_key(&self) -> u32 {
        match (self.atomic_number, self.mass_number) {
            (Some(z), Some(a)) => z * 1000 + a,
            _ => {
                // FNV-1a over the name; OR in the top bit so it can never be 0
                // and can never alias a real ZA (which is < 2^31).
                let mut h: u32 = 0x811c_9dc5;
                if let Some(name) = &self.name {
                    for b in name.as_bytes() {
                        h = (h ^ *b as u32).wrapping_mul(0x0100_0193);
                    }
                }
                h | 0x8000_0000
            }
        }
    }

    /// Check if energy is in the URR range for a given temperature.
    /// Returns true if URR probability tables should be used for this energy.
    #[inline]
    pub fn energy_in_urr(&self, energy: f64, temperature: &str) -> bool {
        self.urr_for_temp(temperature)
            .is_some_and(|urr| urr.energy_in_bounds(energy))
    }

    /// Get the element name with auto-loading if not available
    pub fn get_element(&mut self) -> Option<String> {
        if self.element.is_none() && self.name.is_some() {
            if let Some(name) = &self.name.clone() {
                let _ = self.auto_load_from_config(name, None);
            }
        }
        self.element.clone()
    }

    /// Get the atomic number with auto-loading if not available
    pub fn get_atomic_number(&mut self) -> Option<u32> {
        if self.atomic_number.is_none() && self.name.is_some() {
            if let Some(name) = &self.name.clone() {
                let _ = self.auto_load_from_config(name, None);
            }
        }
        self.atomic_number
    }

    /// Get the mass number with auto-loading if not available
    pub fn get_mass_number(&mut self) -> Option<u32> {
        if self.mass_number.is_none() && self.name.is_some() {
            if let Some(name) = &self.name.clone() {
                let _ = self.auto_load_from_config(name, None);
            }
        }
        self.mass_number
    }

    /// Get the atomic symbol with auto-loading if not available
    pub fn get_atomic_symbol(&mut self) -> Option<String> {
        if self.atomic_symbol.is_none() && self.name.is_some() {
            if let Some(name) = &self.name.clone() {
                let _ = self.auto_load_from_config(name, None);
            }
        }
        self.atomic_symbol.clone()
    }

    /// Get the neutron number with auto-loading if not available
    pub fn get_neutron_number(&mut self) -> Option<u32> {
        if self.neutron_number.is_none() && self.name.is_some() {
            if let Some(name) = &self.name.clone() {
                let _ = self.auto_load_from_config(name, None);
            }
        }
        self.neutron_number
    }

    /// Get available temperatures with auto-loading if not available
    pub fn get_available_temperatures(&mut self) -> Vec<String> {
        if self.available_temperatures.is_empty() && self.name.is_some() {
            if let Some(name) = &self.name.clone() {
                let _ = self.auto_load_from_config(name, None);
            }
        }
        self.available_temperatures.clone()
    }

    /// Get the energy grid for a specific temperature
    pub fn energy_grid(&self, temperature: &str) -> Option<&F64Buffer> {
        self.energy
            .as_ref()
            .and_then(|energy_map| energy_map.get(temperature))
    }

    /// Get a list of available temperatures
    pub fn temperatures(&self) -> Option<Vec<String>> {
        let mut temps = std::collections::HashSet::new();

        // Use loaded_temperatures as the source of truth for reaction temperatures
        for temp in &self.loaded_temperatures {
            temps.insert(temp.clone());
        }

        // Also check energy map
        if let Some(energy_map) = &self.energy {
            for temp in energy_map.keys() {
                temps.insert(temp.clone());
            }
        }

        if temps.is_empty() {
            None
        } else {
            let mut temps_vec: Vec<String> = temps.into_iter().collect();
            temps_vec.sort();
            Some(temps_vec)
        }
    }

    /// Get a list of available MT numbers
    pub fn reaction_mts(&self) -> Option<Vec<i32>> {
        let mut mts = std::collections::HashSet::new();
        for temp_reactions in self.reactions.iter() {
            for &mt in temp_reactions.keys() {
                mts.insert(mt);
            }
        }
        if mts.is_empty() {
            None
        } else {
            let mut mts_vec: Vec<i32> = mts.into_iter().collect();
            mts_vec.sort();
            Some(mts_vec)
        }
    }

    /// Get microscopic cross section data for a specific reaction and temperature.
    /// Returns a tuple of (cross_section_values, energy_grid).
    /// If temperature is None, uses the single loaded temperature if only one exists.
    /// Automatically loads data if not already loaded, using the nuclide name and config.
    ///
    /// # Arguments
    /// * `reaction` - Either an MT number (i32) or reaction name (String/&str) like "(n,gamma)" or "fission"
    /// * `temperature` - Optional temperature string
    pub fn microscopic_cross_section<R>(
        &mut self,
        reaction: R,
        temperature: Option<&str>,
        trim_trailing_zeros: bool,
    ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>>
    where
        R: Into<ReactionIdentifier>,
    {
        // Convert the reaction parameter to an MT number
        let mt = match reaction.into() {
            ReactionIdentifier::Mt(mt_num) => mt_num,
            ReactionIdentifier::Name(name) => {
                // Use the REACTION_MT mapping to convert string to MT number
                crate::data::REACTION_MT.get(name.as_str())
                    .copied()
                    .ok_or_else(|| format!("Unknown reaction name '{name}'. Available reactions can be found in REACTION_MT mapping."))?
            }
        };
        // Check if we need to load data automatically
        if self.loaded_temperatures.is_empty() {
            // No data loaded yet - try to load it automatically
            if let Some(name) = self.name.clone() {
                self.auto_load_from_config(&name, temperature)?;
            } else {
                return Err(
                    "No data loaded and no nuclide name available for automatic loading".into(),
                );
            }
        }

        // Check if we need to load additional temperature
        // Temperature keys are stored without 'K' suffix (e.g., "294" not "294K")
        if let Some(temp) = temperature {
            let temp_key = crate::temperature::strip_k(temp);

            // Whether the request is servable but not yet in memory. Not
            // membership of `available_temperatures`: a temperature the file
            // brackets is servable too, and the reload below is what fetches
            // its two neighbours so the loader can blend them.
            let needs_temp_load = !self.loaded_temperatures.contains(&temp_key.to_string())
                && crate::temperature::resolve(temp_key, &self.available_temperatures).is_ok();

            if needs_temp_load {
                if let Some(name) = self.name.clone() {
                    self.auto_load_additional_temperature(&name, temp)?;
                }
            }
        }

        // Now proceed with the original logic
        let (xs, energy) = self.get_microscopic_cross_section_data(mt, temperature)?;
        if trim_trailing_zeros {
            let mut last_nonzero = xs.len();
            for (i, &val) in xs.iter().enumerate().rev() {
                if val != 0.0 {
                    last_nonzero = i + 1;
                    break;
                }
            }
            Ok((xs[..last_nonzero].to_vec(), energy[..last_nonzero].to_vec()))
        } else {
            // Owned `Vec`s: this is the plotting / Python query surface, whose
            // callers expect data they can keep after the nuclide is dropped.
            Ok((xs.into(), energy.into()))
        }
    }

    /// Helper method to automatically load data from config
    pub fn auto_load_from_config(
        &mut self,
        nuclide_name: &str,
        temperature: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Try to load using the nuclide name and config
        let path_or_url = {
            let cfg = crate::config::CONFIG
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            cfg.get_cross_section(nuclide_name)
        };

        if let Some(path_or_url) = path_or_url {
            // Determine which temperatures to load
            let temps_to_load = if let Some(temp) = temperature {
                let mut temps = std::collections::HashSet::new();
                temps.insert(temp.to_string());
                Some(temps)
            } else {
                None // Load all temperatures
            };

            // Load the data (always pass nuclide_name for keyword/directory resolution)
            let loaded_nuclide = load_nuclide_for_python(
                Some(&path_or_url),
                Some(nuclide_name),
                temps_to_load.as_ref(),
            )?;

            // Update self with the loaded data
            *self = loaded_nuclide;
            Ok(())
        } else {
            Err(format!("No configuration found for nuclide '{nuclide_name}'. Set yamc.cross_section_data to configure data sources.").into())
        }
    }

    /// Helper method to load additional temperature
    fn auto_load_additional_temperature(
        &mut self,
        nuclide_name: &str,
        temperature: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path_or_url = {
            let cfg = crate::config::CONFIG
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            cfg.get_cross_section(nuclide_name)
        };

        if let Some(path_or_url) = path_or_url {
            // Create union of current loaded temperatures plus the new one
            let mut temps_to_load = std::collections::HashSet::new();
            for temp in &self.loaded_temperatures {
                temps_to_load.insert(temp.clone());
            }
            // Stripped, because the filter is matched against the file's
            // normalised labels. Inserting the raw "999K" spelling could never
            // match, and would leave `load_scope.temperatures` holding a label
            // nothing else in the tree spells that way.
            temps_to_load.insert(crate::temperature::strip_k(temperature).to_string());

            // Reload with the expanded temperature set (always pass nuclide_name for keyword/directory resolution)
            let loaded_nuclide = load_nuclide_for_python(
                Some(&path_or_url),
                Some(nuclide_name),
                Some(&temps_to_load),
            )?;

            // Update self with the reloaded data
            *self = loaded_nuclide;
            Ok(())
        } else {
            Err(format!(
                "No configuration found for nuclide '{nuclide_name}' to load additional temperature"
            )
            .into())
        }
    }

    /// Core method that extracts the cross section data (unchanged logic)
    fn get_microscopic_cross_section_data(
        &self,
        mt: i32,
        temperature: Option<&str>,
    ) -> Result<(F64Buffer, F64Buffer), Box<dyn std::error::Error>> {
        // Determine which temperature to use and get its index
        // Temperature keys are stored without 'K' suffix (e.g., "294" not "294K")
        let (temp_key, temp_idx) = if let Some(temp) = temperature {
            let temp_normalized = crate::temperature::strip_k(temp);
            if let Some(idx) = self.get_temp_idx(temp_normalized) {
                (temp_normalized, idx)
            } else {
                return Err(format!(
                    "Temperature '{}' not found in loaded data. Available temperatures: [{}]",
                    temp,
                    self.loaded_temperatures.join(", ")
                )
                .into());
            }
        } else {
            // No temperature provided - use single loaded temperature if available
            if self.loaded_temperatures.len() == 1 {
                (&self.loaded_temperatures[0] as &str, 0usize)
            } else if self.loaded_temperatures.is_empty() {
                return Err("No temperatures loaded in nuclide data".into());
            } else {
                return Err(format!(
                    "Multiple temperatures loaded [{}], must specify which one to use",
                    self.loaded_temperatures.join(", ")
                )
                .into());
            }
        };

        // Get the reaction data for this temperature and MT using index
        let temp_reactions = self.reactions.get(temp_idx).ok_or_else(|| {
            format!(
                "Temperature '{}' not found in reactions. Available temperatures: [{}]",
                temp_key,
                self.loaded_temperatures.join(", ")
            )
        })?;

        let reaction = temp_reactions.get(&mt).ok_or_else(|| {
            // Get available MTs for this temperature
            let available_mts: Vec<String> =
                temp_reactions.keys().map(|mt| mt.to_string()).collect();
            format!(
                "MT {} not found for temperature '{}'. Available MTs: [{}]",
                mt,
                temp_key,
                available_mts.join(", ")
            )
        })?;

        // Return the cross section and energy data
        if reaction.cross_section.is_empty() {
            return Err(format!(
                "No cross section data available for MT {mt} at temperature '{temp_key}'"
            )
            .into());
        }

        if reaction.energy.is_empty() {
            return Err(format!(
                "No energy grid available for MT {mt} at temperature '{temp_key}'"
            )
            .into());
        }

        Ok((reaction.cross_section.clone(), reaction.energy.clone()))
    }
}

// ============================================================================
// Nuclide Loading Functions
// ============================================================================

/// Load a nuclide from an Arrow IPC directory.
///
/// # Arguments
/// * `path` - Path to the Arrow nuclide directory (e.g., "Li6.arrow")
/// * `scope` - Which sections, MTs and temperatures to materialize
///
/// # Example
/// ```ignore
/// let nuclide = load_nuclide("Li6.arrow", &LoadScope::full())?;
/// ```
pub fn load_nuclide<P: AsRef<Path>>(
    path: P,
    scope: &LoadScope,
) -> Result<Nuclide, Box<dyn std::error::Error>> {
    crate::nuclide_loader::load_nuclide(path, scope)
}

/// Whether a cached nuclide can actually answer the temperatures a scope names.
///
/// [`LoadScope::covers`] treats an unfiltered load as universal, which is right
/// for every temperature the FILE carries and wrong for one it merely brackets.
/// An intermediate temperature is built by the Arrow loader only when a filter
/// names it, so an entry parsed WITHOUT a filter holds every rung and not the
/// temperature between two of them. Without this check the cache answers a
/// 450 K request with the unfiltered entry, the label is absent from
/// `loaded_temperatures`, and the request dies downstream as a missing MT.
fn serves_temperatures(existing: &Nuclide, scope: &LoadScope) -> bool {
    match &scope.temperatures {
        None => true,
        Some(wanted) => wanted
            .iter()
            .all(|t| existing.loaded_temperatures.contains(t)),
    }
}

/// Get or load a nuclide from cache, loading from file if needed.
///
/// Parameters:
/// - `nuclide_name`: Name of the nuclide (e.g., "Be9", "Li6")
/// - `path_map`: Map of nuclide names to file paths
/// - `temperatures_to_include`: Optional set of temperatures to load
///
/// Returns the cached nuclide when its recorded [`LoadScope`] already covers
/// what this caller asked for, otherwise loads from file.
pub fn get_or_load_nuclide(
    nuclide_name: &str,
    path_map: &HashMap<String, String>,
    scope: &LoadScope,
) -> Result<Arc<Nuclide>, Box<dyn std::error::Error>> {
    // Get path/keyword from map
    let path_or_keyword = path_map.get(nuclide_name).ok_or_else(|| {
        format!(
            "No data file provided for nuclide '{nuclide_name}'. Please supply a path for all nuclides."
        )
    })?;

    // Resolve keywords/URLs to local paths (when download feature is enabled).
    // This is also the download: only the sections `scope` needs are fetched.
    #[cfg(feature = "download")]
    let resolved_path = {
        let resolved = crate::url_cache::resolve_path_or_url(
            path_or_keyword,
            nuclide_name,
            crate::url_cache::DataKind::Neutron,
            scope,
        )?;
        resolved.to_string_lossy().to_string()
    };

    #[cfg(not(feature = "download"))]
    let resolved_path = path_or_keyword.clone();

    // Create cache key using resolved path
    let normalized_source = match std::fs::canonicalize(&resolved_path) {
        Ok(canonical) => canonical.to_string_lossy().to_string(),
        Err(_) => resolved_path.clone(),
    };
    let cache_key = format!("{nuclide_name}@{normalized_source}");

    // Fast path: upgrade Weak → Arc if the nuclide is still alive AND what it
    // holds is at least as wide as what this caller needs. A transmutation load
    // parses only the chain's MTs and none of the transport sections, so it can
    // never satisfy a transport request (issue #389).
    let mut cached_scope: Option<LoadScope> = None;
    {
        let cache = match GLOBAL_NUCLIDE_CACHE.lock() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(weak) = cache.get(&cache_key) {
            if let Some(existing) = weak.upgrade() {
                if existing.load_scope.covers(scope) && serves_temperatures(&existing, scope) {
                    return Ok(existing);
                }
                cached_scope = Some(existing.load_scope.clone());
            }
        }
    }

    // Cache miss or expired -- purge dead entries before loading new data.
    {
        let mut cache = match GLOBAL_NUCLIDE_CACHE.lock() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };
        cache.retain(|_, w| w.strong_count() > 0);
    }

    // Reload at the UNION of what was cached and what is now wanted, not at the
    // request alone. Two chains asking for MT {102, 16} and MT {102, 103} would
    // otherwise evict each other on every call.
    let load_at = match cached_scope {
        Some(cached) => cached.union(scope),
        None => scope.clone(),
    };

    // The fetch above was sized for `scope`. When the union widened it past
    // that, top the cache dir up first: the download is additive, so this only
    // pulls the sections that are genuinely missing.
    #[cfg(feature = "download")]
    if !scope.covers(&load_at) {
        crate::url_cache::resolve_path_or_url(
            path_or_keyword,
            nuclide_name,
            crate::url_cache::DataKind::Neutron,
            &load_at,
        )?;
    }

    let mut nuclide = crate::nuclide_loader::load_nuclide(&resolved_path, &load_at)?;
    nuclide.data_path = Some(resolved_path.clone());

    // Build any temperature the CALLER asked for that the file only brackets.
    //
    // The loader does this itself when it is handed a filter, and that covers
    // the request-shaped case. It does not cover this one: `load_at` is the
    // union with whatever was cached, and unioning a concrete temperature set
    // with an unfiltered load gives `None`, which the loader reads as "every
    // temperature the file carries" and which contains no intermediate rung. So
    // the widened read is correct about what to PARSE and has lost what to
    // BUILD, and this puts it back.
    //
    // Before the Arc, so the synthesised temperature is in the cached entry and
    // a second material at the same temperature reuses it rather than blending
    // the whole nuclide again.
    if let Some(wanted) = scope.temperatures.as_ref() {
        let mut missing: Vec<String> = wanted
            .iter()
            .filter(|t| !nuclide.loaded_temperatures.contains(t))
            .cloned()
            .collect();
        missing.sort();
        for label in missing {
            crate::blend::synthesise_temperature(&mut nuclide, &label)
                .map_err(|e| format!("{nuclide_name}: {e}"))?;
        }
    }

    let arc_nuclide = Arc::new(nuclide);
    {
        let mut cache = match GLOBAL_NUCLIDE_CACHE.lock() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };
        cache.insert(cache_key, Arc::downgrade(&arc_nuclide));
    }

    Ok(arc_nuclide)
}

/// Load a nuclide with Python wrapper semantics.
/// Handles both path and name parameters and preserves available_temperatures when filtering.
/// Supports keywords (e.g., "tendl-2025", "fendl-3.2d", "endf-b8.1"), directories, and URLs
/// when the download feature is enabled.
/// When `path` is None, falls back to Config using `nuclide_name`.
pub fn load_nuclide_for_python(
    path: Option<&str>,
    _nuclide_name: Option<&str>,
    temperatures: Option<&std::collections::HashSet<String>>,
) -> Result<Nuclide, Box<dyn std::error::Error>> {
    // If no path is provided, fall back to Config
    let path_str = match path {
        Some(p) => p.to_string(),
        None => {
            let name = _nuclide_name.ok_or("Either path or nuclide name is required")?;
            let config = crate::config::Config::global();
            config.get_cross_section(name).ok_or_else(|| {
                format!(
                    "No path provided and nuclide '{}' not found in Config. \
                     Set yamc.cross_section_data first or provide a path.",
                    name
                )
            })?
        }
    };

    // Python hands out whole nuclides (plotting, inspection, per-material
    // overrides), so this always loads the full section set.
    let scope = LoadScope::full().with_temperatures(temperatures.cloned());

    // Resolve the path (handles keywords, URLs, directories, and local paths)
    let resolved_path = crate::url_cache::resolve_data_path(
        &path_str,
        _nuclide_name,
        crate::url_cache::DataKind::Neutron,
        &scope,
    )?;

    // Load the nuclide using format-agnostic loader
    crate::nuclide_loader::load_nuclide(&resolved_path, &scope)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests;
