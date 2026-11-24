//! Runtime particle state used by the yamc transport engine.
//!
//! The `Particle` struct holds the live state of a particle as it moves
//! through transport: position, direction, energy, current cell index,
//! collision history (debug-only), and so on. Extracted out of yamc into
//! its own crate so leaf crates that only need the runtime type
//! (currently `yamc-gpu` for its `From<&Particle>` conversion; later
//! tools like a slim WASM model viewer that wants Source + Particle
//! standalone) can depend on it without dragging the transport engine,
//! transmutation, MPI, or tally machinery.
//!
//! `ParticleType` (the `{Neutron, Photon}` enum) lives in `yamc-nuclide`
//! because reaction-product code needs it to encode emitted-particle
//! species. It's re-exported here so existing call sites that do
//! `yamc_particle::particle::ParticleType` keep resolving.

pub mod particle;

pub use particle::{
    cell_index_to_option, cell_index_to_u32, surface_id_to_option, surface_id_to_u32,
    urr_from_option, urr_to_option, CollisionEvent, Particle, ParticleType, NO_CELL, NO_SURFACE,
    NO_URR,
};
