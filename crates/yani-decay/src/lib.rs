//! Decay-chain observables over a [`yani`] transmutation chain.
//!
//! Everything that reads half-lives, branchings and decay-source records out of
//! a chain and turns them into a number a user asks for:
//!
//! - [`decay_chain`] -- enumerate the photon-emitting descendants of a produced
//!   nuclide, and evolve their Bateman activity across an irradiation schedule
//! - [`decay`] -- activity [Bq] and decay heat [W] of an inventory
//! - [`contact_dose`] -- the contact dose rate [Gy/h or Sv/h] of an inventory,
//!   from its own decay photons through its own self-shielding
//! - [`decay_photons`] -- the D1S (Direct 1-Step) shutdown-dose-rate
//!   post-processing: radionuclide discovery, time-correction factors, and
//!   applying them to tally results
//!
//! A leaf over `yani` and the data tables in `yamc-nuclide` (atomic masses,
//! photon attenuation, dose coefficients): no cross sections, no materials, no
//! transport. The two consumers that DO need those sit above it.
//! `yamc-physics` calls [`decay_chain::descendant_paths`] when it emits D1S
//! photons at a collision, and the Python bindings call the rest.

pub mod contact_dose;
pub mod decay;
pub mod decay_chain;
pub mod decay_photons;

pub use contact_dose::{contact_dose_by_nuclide, contact_dose_total, DoseQuantity};
pub use decay::{
    activity_by_nuclide, activity_total, decay_heat_by_nuclide, decay_heat_total,
    decay_photon_lines, total,
};
pub use decay_chain::{
    build_emitter_paths, descendant_paths, evolve_chain_activity, DecayChainPath,
};
pub use decay_photons::{
    apply_time_correction, get_radionuclides_from_chain, time_correction_factors, CorrectedSteps,
};
