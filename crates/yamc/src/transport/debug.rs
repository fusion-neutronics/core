//! Debug-instrumentation cluster extracted verbatim from `model.rs`.
//!
//! This holds the `DEBUG_*` atomic statics and the dual-`#[cfg]` debug helper
//! pairs (a real `#[cfg(feature = "debug_*")]` body plus a matching
//! `#[cfg(not(...))]` no-op) for elastic-scatter and collision tracing. The
//! transport loop (`transport.rs`) and `Model::run_internal` (`model.rs`) call
//! into these via `use crate::transport::debug::*;` / explicit `use`. Compiled
//! away when the gating features are off.

#[cfg(any(
    feature = "debug_runtime",
    feature = "debug_collision",
    feature = "debug_diagnostics"
))]
use std::sync::atomic::AtomicU64;
// Gated on either feature rather than imported once per feature. Two `use`
// lines naming the same items are an E0252 redefinition as soon as both
// features are on, which is what `--all-features` does.
#[cfg(any(feature = "debug_runtime", feature = "debug_collision"))]
use std::sync::atomic::{AtomicBool, Ordering};
use yamc_nuclide::nuclide::ReactionType;

// Collisions in the current batch, for the `debug_diagnostics` batch report in
// `model.rs`. Counts every collision, neutron and photon alike.
//
// A module static rather than a local in `model.rs` because the increment
// happens in the transport loop, which was extracted out of `model.rs` and no
// longer shares that scope. `model.rs` zeroes it at the start of each batch, so
// it stays per-batch rather than cumulative.
#[cfg(feature = "debug_diagnostics")]
pub(crate) static BATCH_COLLISIONS: AtomicU64 = AtomicU64::new(0);

// Debug logging and counters (compiled only when feature "debug_runtime" is enabled)
#[cfg(feature = "debug_runtime")]
// Debug logging for elastic scattering (enabled by YAMC_DEBUG_ELASTIC env var)
static DEBUG_ELASTIC: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "debug_runtime")]
static DEBUG_ELASTIC_CHECKED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "debug_runtime")]
static DEBUG_ELASTIC_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "debug_runtime")]
static TOTAL_ELASTIC_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "debug_runtime")]
const DEBUG_ELASTIC_MAX: u64 = 1000; // Max number of elastic events to log

// Additional debug counters
#[cfg(feature = "debug_runtime")]
pub(crate) static TOTAL_COLLISIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "debug_runtime")]
pub(crate) static TOTAL_SCATTERING_REACTIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "debug_runtime")]
pub(crate) static TOTAL_ABSORPTION_REACTIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "debug_runtime")]
pub(crate) static TOTAL_FISSION_REACTIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "debug_runtime")]
pub(crate) static TOTAL_SCATTER_MT2: AtomicU64 = AtomicU64::new(0); // MT 2 = elastic
#[cfg(feature = "debug_runtime")]
pub(crate) static TOTAL_SCATTER_MT_OTHER: AtomicU64 = AtomicU64::new(0); // non-elastic scattering

#[cfg(feature = "debug_runtime")]
pub(crate) fn is_debug_elastic() -> bool {
    if !DEBUG_ELASTIC_CHECKED.load(Ordering::Relaxed) {
        let is_set = std::env::var("YAMC_DEBUG_ELASTIC").is_ok();
        DEBUG_ELASTIC.store(is_set, Ordering::Relaxed);
        DEBUG_ELASTIC_CHECKED.store(true, Ordering::Relaxed);
        if is_set {
            eprintln!("YAMC_DEBUG_ELASTIC is enabled - will log elastic scattering events");
        }
    }
    DEBUG_ELASTIC.load(Ordering::Relaxed)
}

#[cfg(not(feature = "debug_runtime"))]
#[inline(always)]
pub(crate) fn is_debug_elastic() -> bool {
    false
}

#[cfg(feature = "debug_runtime")]
pub(crate) fn log_elastic_scatter(e_in: f64, mu_cm: f64, e_out: f64, is_free_gas: bool) {
    TOTAL_ELASTIC_COUNT.fetch_add(1, Ordering::Relaxed); // Always count
    if !is_debug_elastic() {
        return;
    }
    let count = DEBUG_ELASTIC_COUNT.fetch_add(1, Ordering::Relaxed);
    if count < DEBUG_ELASTIC_MAX {
        let mode = if is_free_gas {
            "free-gas"
        } else {
            "target-at-rest"
        };
        eprintln!(
            "ELASTIC[{count}]: E_in={e_in:.6e} mu_cm={mu_cm:+.6} E_out={e_out:.6e} mode={mode}"
        );
    }
}

#[cfg(not(feature = "debug_runtime"))]
#[inline(always)]
pub(crate) fn log_elastic_scatter(_e_in: f64, _mu_cm: f64, _e_out: f64, _is_free_gas: bool) {}

/// Get total elastic scatter count (for diagnostics)
#[cfg(feature = "debug_runtime")]
pub fn get_elastic_scatter_count() -> u64 {
    TOTAL_ELASTIC_COUNT.load(Ordering::Relaxed)
}
#[cfg(not(feature = "debug_runtime"))]
pub fn get_elastic_scatter_count() -> u64 {
    0
}

/// Get all debug counters
#[cfg(feature = "debug_runtime")]
pub fn get_debug_counters() -> (u64, u64, u64, u64, u64, u64) {
    (
        TOTAL_COLLISIONS.load(Ordering::Relaxed),
        TOTAL_SCATTERING_REACTIONS.load(Ordering::Relaxed),
        TOTAL_ABSORPTION_REACTIONS.load(Ordering::Relaxed),
        TOTAL_FISSION_REACTIONS.load(Ordering::Relaxed),
        TOTAL_SCATTER_MT2.load(Ordering::Relaxed),
        TOTAL_SCATTER_MT_OTHER.load(Ordering::Relaxed),
    )
}
#[cfg(not(feature = "debug_runtime"))]
pub fn get_debug_counters() -> (u64, u64, u64, u64, u64, u64) {
    (0, 0, 0, 0, 0, 0)
}

/// Reset elastic scatter counters (for new simulations)
#[cfg(feature = "debug_runtime")]
pub fn reset_elastic_scatter_count() {
    TOTAL_ELASTIC_COUNT.store(0, Ordering::Relaxed);
    DEBUG_ELASTIC_COUNT.store(0, Ordering::Relaxed);
    TOTAL_COLLISIONS.store(0, Ordering::Relaxed);
    TOTAL_SCATTERING_REACTIONS.store(0, Ordering::Relaxed);
    TOTAL_ABSORPTION_REACTIONS.store(0, Ordering::Relaxed);
    TOTAL_FISSION_REACTIONS.store(0, Ordering::Relaxed);
    TOTAL_SCATTER_MT2.store(0, Ordering::Relaxed);
    TOTAL_SCATTER_MT_OTHER.store(0, Ordering::Relaxed);
    // Reset the checked flag so env var is re-read
    DEBUG_ELASTIC_CHECKED.store(false, Ordering::Relaxed);
}
#[cfg(not(feature = "debug_runtime"))]
pub fn reset_elastic_scatter_count() {}

// =============================================================================
// Debug collision tracing for 2000-5000 eV energy range
// Enabled by YAMC_DEBUG_COLLISION env var when compiled with debug_collision feature
// =============================================================================
#[cfg(feature = "debug_collision")]
static DEBUG_COLLISION_ENABLED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "debug_collision")]
static DEBUG_COLLISION_CHECKED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "debug_collision")]
static DEBUG_COLLISION_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "debug_collision")]
const DEBUG_COLLISION_MAX: u64 = 500; // Max number of collisions to log
#[cfg(feature = "debug_collision")]
const DEBUG_ENERGY_MIN: f64 = 2000.0; // eV
#[cfg(feature = "debug_collision")]
const DEBUG_ENERGY_MAX: f64 = 5000.0; // eV

/// Check if collision debugging is enabled
#[cfg(feature = "debug_collision")]
fn is_debug_collision() -> bool {
    if !DEBUG_COLLISION_CHECKED.load(Ordering::Relaxed) {
        let is_set = std::env::var("YAMC_DEBUG_COLLISION").is_ok();
        DEBUG_COLLISION_ENABLED.store(is_set, Ordering::Relaxed);
        DEBUG_COLLISION_CHECKED.store(true, Ordering::Relaxed);
        if is_set {
            eprintln!("=== YAMC_DEBUG_COLLISION enabled ===");
            eprintln!(
                "Tracing collisions in {:.0}-{:.0} eV range (max {} events)",
                DEBUG_ENERGY_MIN, DEBUG_ENERGY_MAX, DEBUG_COLLISION_MAX
            );
            eprintln!("=====================================");
        }
    }
    DEBUG_COLLISION_ENABLED.load(Ordering::Relaxed)
}

#[cfg(not(feature = "debug_collision"))]
#[inline(always)]
#[allow(dead_code)]
fn is_debug_collision() -> bool {
    false
}

/// Check if energy is in the debug range (2000-5000 eV)
#[cfg(feature = "debug_collision")]
#[inline]
fn energy_in_debug_range(energy: f64) -> bool {
    (DEBUG_ENERGY_MIN..=DEBUG_ENERGY_MAX).contains(&energy)
}

#[cfg(not(feature = "debug_collision"))]
#[inline(always)]
#[allow(dead_code)]
fn energy_in_debug_range(_energy: f64) -> bool {
    false
}

/// Log when particle enters the debug energy range
#[cfg(feature = "debug_collision")]
#[allow(dead_code)]
fn log_enter_debug_range(particle_idx: usize, energy: f64) {
    if !is_debug_collision() {
        return;
    }
    let count = DEBUG_COLLISION_COUNT.load(Ordering::Relaxed);
    if count < DEBUG_COLLISION_MAX {
        eprintln!(
            "[COLL_ENTER] P{} entered debug range at E={:.2} eV",
            particle_idx, energy
        );
    }
}

#[cfg(not(feature = "debug_collision"))]
#[inline(always)]
#[allow(dead_code)]
fn log_enter_debug_range(_particle_idx: usize, _energy: f64) {}

/// Log collision distance sampling
#[cfg(feature = "debug_collision")]
pub(crate) fn log_collision_distance(
    particle_idx: usize,
    energy: f64,
    sigma_t: f64,
    distance: f64,
    urr_random: Option<f64>,
) {
    if !is_debug_collision() || !energy_in_debug_range(energy) {
        return;
    }
    let count = DEBUG_COLLISION_COUNT.fetch_add(1, Ordering::Relaxed);
    if count < DEBUG_COLLISION_MAX {
        let urr_str = urr_random
            .map(|r| format!(" urr_r={:.6}", r))
            .unwrap_or_default();
        eprintln!(
            "[COLL_DIST] P{} E={:.2}eV sigma_t={:.6e}/cm dist={:.6e}cm{}",
            particle_idx, energy, sigma_t, distance, urr_str
        );
    }
}

#[cfg(not(feature = "debug_collision"))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn log_collision_distance(
    _particle_idx: usize,
    _energy: f64,
    _sigma_t: f64,
    _distance: f64,
    _urr_random: Option<f64>,
) {
}

/// Log reaction type selection
#[cfg(feature = "debug_collision")]
pub(crate) fn log_reaction_selection(
    particle_idx: usize,
    energy: f64,
    reaction_type: &ReactionType,
    nuclide_name: &str,
) {
    if !is_debug_collision() || !energy_in_debug_range(energy) {
        return;
    }
    let count = DEBUG_COLLISION_COUNT.load(Ordering::Relaxed);
    if count < DEBUG_COLLISION_MAX {
        let rxn_name = match reaction_type {
            ReactionType::Scattering => "SCATTER",
            ReactionType::Absorption => "ABSORB",
            ReactionType::Fission => "FISSION",
        };
        eprintln!(
            "[COLL_RXN] P{} E={:.2}eV nuclide={} reaction={}",
            particle_idx, energy, nuclide_name, rxn_name
        );
    }
}

#[cfg(not(feature = "debug_collision"))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn log_reaction_selection(
    _particle_idx: usize,
    _energy: f64,
    _reaction_type: &ReactionType,
    _nuclide_name: &str,
) {
}

/// Log scattering result (energy before and after)
#[cfg(feature = "debug_collision")]
pub(crate) fn log_scatter_result(particle_idx: usize, e_in: f64, e_out: f64, mt: i32) {
    if !is_debug_collision() || !energy_in_debug_range(e_in) {
        return;
    }
    let count = DEBUG_COLLISION_COUNT.load(Ordering::Relaxed);
    if count < DEBUG_COLLISION_MAX {
        let ratio = if e_in > 0.0 { e_out / e_in } else { 0.0 };
        eprintln!(
            "[COLL_SCAT] P{} MT={} E_in={:.2}eV E_out={:.2}eV ratio={:.4}",
            particle_idx, mt, e_in, e_out, ratio
        );
    }
}

#[cfg(not(feature = "debug_collision"))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn log_scatter_result(_particle_idx: usize, _e_in: f64, _e_out: f64, _mt: i32) {}

/// Reset collision debug counters
#[cfg(feature = "debug_collision")]
pub fn reset_collision_debug() {
    DEBUG_COLLISION_COUNT.store(0, Ordering::Relaxed);
    DEBUG_COLLISION_CHECKED.store(false, Ordering::Relaxed);
}

#[cfg(not(feature = "debug_collision"))]
pub fn reset_collision_debug() {}
