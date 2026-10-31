//! Collision physics for yamc.
//!
//! Pure functions and small data structures that turn an incoming particle
//! plus a sampled reaction into outgoing particles. The transport loop in
//! `yamc::model` and the GPU dispatch layer call into this crate; nothing
//! here depends on the engine, the geometry, or the Python bindings.
//!
//! Modules:
//! - [`bank`] -- per-history secondary-particle stack
//! - [`interaction`] -- direction rotation, target-velocity sampling,
//!   fission-neutron sampling, the `SecondaryNeutrons` smallvec type
//! - [`scatter`] -- elastic kinematics on a sampled angle
//! - [`inelastic`] -- inelastic kinematics dispatching on a sampled reaction
//!   product table
//! - [`bremsstrahlung`] -- thick-target bremsstrahlung tabulation and
//!   photon sampling for charged secondaries
//! - [`photon_production`] -- secondary photon sampling for (n,γ) and
//!   similar reactions that publish a photon-product table
//! - [`decay_photon_production`] -- D1S decay-photon sampling from
//!   activation chains
//!
//! Decay-chain enumeration, Bateman activity evolution, activity / decay heat
//! and the D1S shutdown-dose post-processing live in `yani-decay`, which needs
//! only a chain. `decay_photon_production` stays here because it banks
//! particles.

pub mod gpu;
pub mod neutron;
pub mod photon;
pub mod util;

/// Per-batch and per-event diagnostic counters for photon transport debugging.
/// Compiled away entirely unless `debug_diagnostics` is enabled, so there is
/// zero overhead in release builds.
#[cfg(feature = "debug_diagnostics")]
pub mod photon_diag {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static FLUORESCENCE_5_10: AtomicU64 = AtomicU64::new(0);
    pub static TTB_5_10: AtomicU64 = AtomicU64::new(0);
    pub static FLUORESCENCE_TOTAL: AtomicU64 = AtomicU64::new(0);
    pub static TTB_TOTAL: AtomicU64 = AtomicU64::new(0);
    pub static TTB_ABOVE_KEDGE: AtomicU64 = AtomicU64::new(0);
    pub static PE_KSHELL: AtomicU64 = AtomicU64::new(0);
    pub static PE_TOTAL: AtomicU64 = AtomicU64::new(0);
    pub static DECAY_PHOTONS: AtomicU64 = AtomicU64::new(0);
    pub static PROMPT_PHOTONS: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        FLUORESCENCE_5_10.store(0, Ordering::Relaxed);
        TTB_5_10.store(0, Ordering::Relaxed);
        FLUORESCENCE_TOTAL.store(0, Ordering::Relaxed);
        TTB_TOTAL.store(0, Ordering::Relaxed);
        TTB_ABOVE_KEDGE.store(0, Ordering::Relaxed);
        PE_KSHELL.store(0, Ordering::Relaxed);
        PE_TOTAL.store(0, Ordering::Relaxed);
        DECAY_PHOTONS.store(0, Ordering::Relaxed);
        PROMPT_PHOTONS.store(0, Ordering::Relaxed);
    }

    pub fn print_summary() {
        let f5 = FLUORESCENCE_5_10.load(Ordering::Relaxed);
        let t5 = TTB_5_10.load(Ordering::Relaxed);
        let ft = FLUORESCENCE_TOTAL.load(Ordering::Relaxed);
        let tt = TTB_TOTAL.load(Ordering::Relaxed);
        let tk = TTB_ABOVE_KEDGE.load(Ordering::Relaxed);
        let pk = PE_KSHELL.load(Ordering::Relaxed);
        let pt = PE_TOTAL.load(Ordering::Relaxed);
        eprintln!("\n[PHOTON DIAGNOSTIC] 5-10 keV photon sources:");
        eprintln!("  Fluorescence: {f5} (of {ft} total)");
        eprintln!("  TTB/Brems:    {t5} (of {tt} total, {tk} above K-edge)");
        eprintln!("  Total 5-10:   {}", f5 + t5);
        eprintln!(
            "  Photoelectric: {} total, {} K-shell (frac={:.4})",
            pt,
            pk,
            if pt > 0 { pk as f64 / pt as f64 } else { 0.0 }
        );
    }
}
