//! Particle species classifier used by reaction-product code to encode
//! "this reaction emits a Neutron / Photon". Extracted out of yamc's
//! runtime `Particle` struct so reaction data can stand on its own
//! without dragging in transport state.
//!
//! `Particle` (the runtime state) lives in yamc and re-imports this
//! enum so existing call sites keep resolving against
//! `yamc::particle::ParticleType`.

use serde::{Deserialize, Serialize};

/// Particle types for transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ParticleType {
    #[serde(rename = "neutron")]
    #[default]
    Neutron,
    #[serde(rename = "photon")]
    Photon,
}
