//! Neutron transport on the GPU.
//!
//! - `xs` -- cross-section preparation (pure CPU data prep, always built).
//! - `transport` -- the production multi-cell multi-step transport kernel
//!   plus its bit-for-bit CPU mirrors (cubecl, macOS-gated).
//! - `probes` -- the incremental build-up kernels (one transport phase
//!   each) retained as a per-phase validation/benchmark harness.

pub mod xs;

/// Host-side GPU input-parameter structs (plain data, no cubecl). Always
/// built so the translate/dispatch layers can reference them on macOS, where
/// the kernel modules below are `#[cfg(not(target_os = "macos"))]`-gated.
pub mod fission_bank_inputs;
pub mod nuclide_select_inputs;
pub mod survival_biasing;

/// On-device collision-nuclide selection (the per-nuclide sampling walk;
/// prerequisite for per-nuclide secondary-photon production and D1S).
#[cfg(not(target_os = "macos"))]
pub mod nuclide_select;
#[cfg(not(target_os = "macos"))]
pub mod probes;
#[cfg(not(target_os = "macos"))]
pub mod transport;
