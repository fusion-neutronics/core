//! Cross-cutting helpers: RNG, timing, diagnostics, model fingerprint, interpolation.

pub mod fast_rng;
pub mod fingerprint;
pub mod lost_particle;
pub mod timer;
pub(crate) mod utilities;
pub use utilities::{interpolate_linear, interpolate_log_log};
