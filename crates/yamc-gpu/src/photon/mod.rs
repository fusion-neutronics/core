//! Photon transport on the GPU.
//!
//! - `xs` -- cross-section / interaction-data preparation (pure CPU data
//!   prep, always built).
//! - `transport` -- the production multi-cell photon transport kernel
//!   (cubecl, macOS-gated).

pub mod xs;

/// Per-collision element-selection inputs (macro-total weight table + the
/// per-material element-slab offset/count meta). Always compiled (referenced
/// by the un-gated `translate_photon.rs` / `dispatch.rs`), so it lives here
/// rather than inside the macOS-gated `transport` module -- mirrors
/// `neutron::nuclide_select_inputs`.
pub mod element_select_inputs;

#[cfg(not(target_os = "macos"))]
pub mod transport;

/// Host-side adapter: drain a populated device particle bank's photon
/// records into the photon transport kernel's structure-of-arrays source,
/// then transport them. The coupled neutron->photon pipeline's photon
/// drain (S5).
#[cfg(not(target_os = "macos"))]
pub mod from_bank;

/// Shared thick-target-bremsstrahlung per-photon energy sampler (used by
/// the transport kernel's TTB loops; tested against the CPU reference).
#[cfg(not(target_os = "macos"))]
pub mod ttb_energy;

/// On-device discrete photon-line energy sampling (intensity-weighted line
/// walk for D1S decay spectra / ENDF discrete-photon production; tested
/// bit-for-bit against the CPU twin).
#[cfg(not(target_os = "macos"))]
pub mod discrete_spectrum;

/// On-device coupled neutron->photon production sampling: the per-(reaction,
/// product) selection walk and the photon-count (yield floor + Bernoulli)
/// draw, with CPU twins and bit-for-bit GPU parity tests.
#[cfg(not(target_os = "macos"))]
pub mod production_select;

/// On-device coupled neutron->photon production KINEMATICS sampling: given a
/// selected photon product and the incident neutron energy, sample the emitted
/// photon's outgoing energy and scattering cosine (angle-first, energy-second,
/// matching the CPU `sample_uncorrelated` draw order), with a CPU twin and
/// bit/ULP-exact GPU parity tests.
#[cfg(not(target_os = "macos"))]
pub mod production_kinematics;
