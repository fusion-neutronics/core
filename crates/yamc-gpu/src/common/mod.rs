//! Cross-cutting GPU primitives shared by neutron and photon transport.
//!
//! Split into always-compiled host data types (`rng`, `particle`,
//! `tallies` -- pure CPU, no cubecl) and the cubecl `#[cube]` helpers
//! (`polyfills`, `poly_solve`, `pcg32`, plus the `geometry`, `sampling`,
//! and `probes` groups) which are gated out on macOS along with the rest
//! of the GPU path.

pub mod particle;
pub mod rng;
pub mod tallies;

#[cfg(not(target_os = "macos"))]
pub mod lost_particles;
#[cfg(not(target_os = "macos"))]
pub mod particle_bank;
#[cfg(not(target_os = "macos"))]
pub mod pcg32;
#[cfg(not(target_os = "macos"))]
pub mod poly_solve;
#[cfg(not(target_os = "macos"))]
pub mod polyfills;
#[cfg(not(target_os = "macos"))]
pub mod urr;

#[cfg(not(target_os = "macos"))]
pub mod geometry;
#[cfg(not(target_os = "macos"))]
pub mod probes;
#[cfg(not(target_os = "macos"))]
pub mod sampling;
