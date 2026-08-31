//! Internal scoring machinery for a `Tally`.
//!
//! `TallyAccumulator` owns the accumulation state that was previously
//! interleaved with `Tally`'s config fields: atomic per-bin values, the
//! finalized per-history Welford statistics, the per-tally score-index
//! caches populated once on first use, and the pre-loaded overlay XS cache.
//!
//! Keeping this separate from `Tally` isolates the accumulation state from
//! `Tally`'s config; public methods on `Tally` delegate to the accumulator
//! internally.
//!
//! The struct is `pub(crate)` on purpose: it is an implementation detail,
//! not part of the public Rust or Python API.
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64};
use std::sync::{Mutex, OnceLock};

use crate::tally::OverlayXsData;

/// Internal accumulation state for a `Tally`.
///
/// Holds the hot-path scoring buffers (atomic values per bin), the cross-batch
/// running sums, and the score-index / filter caches that are populated once
/// on first scoring.
///
/// This struct is owned by `Tally` via composition (`Tally.accumulator`).
/// The split lets future PRs move to `Vec<TallyAccumulator>` on `Model`
/// without touching scoring internals again.
#[derive(Debug)]
pub(crate) struct TallyAccumulator {
    /// Per-bin scratch atomic-f64 buffer used by the GPU writeback
    /// path (kernel produces a finalised per-bin value, host writes
    /// it via `store_bin_value`). Length = `num_bins` for the tally's
    /// shape; empty on tallies that haven't been initialised yet.
    pub(crate) values: Vec<AtomicU64>,

    /// Per-history Welford global state, installed once at the end
    /// of `Model::simulate_transport` after the rayon fold/reduce
    /// completes. `get_mean` / `get_std_dev` / `total_mean` /
    /// `total_std` read from this.
    pub(crate) welford_finalized: Mutex<Option<crate::welford::WelfordTallyStats>>,

    /// Snapshots of the tally's aggregate statistics versus number of
    /// histories, recorded at each batch checkpoint by the run loop. Drives
    /// the convergence/trend diagnostics. Empty until the run records it.
    pub(crate) convergence_history: Mutex<Vec<crate::result::ConvergencePoint>>,

    /// Number of batches accumulated so far. Kept on the accumulator
    /// rather than `Tally` so the simulation loop can update it
    /// without taking a mutable reference to the tally.
    pub(crate) n_realizations: AtomicU32,

    // --- One-shot caches populated by `update_cache()` on first score ---
    pub(crate) cache_initialized: AtomicBool,
    pub(crate) cached_num_energy_bins: OnceLock<usize>,
    pub(crate) cached_energy_bins: OnceLock<Option<Vec<f64>>>,
    pub(crate) cached_cell_filter_id: OnceLock<Option<u32>>,
    pub(crate) cached_has_flux_score: OnceLock<bool>,
    pub(crate) cached_flux_score_indices: OnceLock<Vec<usize>>,
    pub(crate) cached_has_heating_score: OnceLock<bool>,
    pub(crate) cached_heating_score_indices: OnceLock<Vec<usize>>,
    pub(crate) cached_has_heating_local_score: OnceLock<bool>,
    pub(crate) cached_heating_local_score_indices: OnceLock<Vec<usize>>,
    /// Production score indices: `(score_idx, mt_number)` for MTs 203..=207.
    pub(crate) cached_production_score_indices: OnceLock<Vec<(usize, i32)>>,
    pub(crate) cached_has_damage_energy_score: OnceLock<bool>,
    pub(crate) cached_damage_energy_score_indices: OnceLock<Vec<usize>>,
    pub(crate) cached_energy_function_filter: OnceLock<Option<crate::EnergyFunctionFilter>>,
    /// MT reaction score indices: `(score_idx, mt_number)`.
    pub(crate) cached_mt_score_indices: OnceLock<Vec<(usize, i32)>>,
    pub(crate) cached_particle_type_filter: OnceLock<Option<yamc_particle::ParticleType>>,
    pub(crate) cached_has_photon_score: OnceLock<bool>,
    /// Photon score indices: `(score_idx, component)` where component is
    /// 0=coherent, 1=incoherent, 2=photoelectric, 3=pair_production.
    pub(crate) cached_photon_score_indices: OnceLock<Vec<(usize, u8)>>,

    /// Pre-loaded overlay microscopic XS data. Populated by
    /// `Tally::prepare_overlay_xs()` when `multiply_density = false`.
    pub(crate) cached_overlay_xs: OnceLock<Option<OverlayXsData>>,
}

impl TallyAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            values: Vec::new(),
            welford_finalized: Mutex::new(None),
            convergence_history: Mutex::new(Vec::new()),
            n_realizations: AtomicU32::new(0),
            cache_initialized: AtomicBool::new(false),
            cached_num_energy_bins: OnceLock::new(),
            cached_energy_bins: OnceLock::new(),
            cached_cell_filter_id: OnceLock::new(),
            cached_has_flux_score: OnceLock::new(),
            cached_flux_score_indices: OnceLock::new(),
            cached_has_heating_score: OnceLock::new(),
            cached_heating_score_indices: OnceLock::new(),
            cached_has_heating_local_score: OnceLock::new(),
            cached_heating_local_score_indices: OnceLock::new(),
            cached_production_score_indices: OnceLock::new(),
            cached_has_damage_energy_score: OnceLock::new(),
            cached_damage_energy_score_indices: OnceLock::new(),
            cached_energy_function_filter: OnceLock::new(),
            cached_mt_score_indices: OnceLock::new(),
            cached_particle_type_filter: OnceLock::new(),
            cached_has_photon_score: OnceLock::new(),
            cached_photon_score_indices: OnceLock::new(),
            cached_overlay_xs: OnceLock::new(),
        }
    }
}

impl Default for TallyAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn new_accumulator_is_empty() {
        let acc = TallyAccumulator::new();
        assert!(acc.values.is_empty());
        assert_eq!(acc.n_realizations.load(Ordering::Relaxed), 0);
        assert!(!acc.cache_initialized.load(Ordering::Relaxed));
        assert!(acc.cached_num_energy_bins.get().is_none());
        assert!(acc.cached_overlay_xs.get().is_none());
    }

    #[test]
    fn tally_new_initializes_accumulator_empty() {
        use crate::tally::Tally;
        let tally = Tally::new();
        assert!(tally.accumulator.values.is_empty());
        assert_eq!(tally.accumulator.n_realizations.load(Ordering::Relaxed), 0);
        assert!(!tally.accumulator.cache_initialized.load(Ordering::Relaxed));
    }
}
