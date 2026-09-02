//! Evaluated neutron nuclear data: `Nuclide`, `Reaction`, reaction-product
//! sampling, and the loaders that read them off disk.
//!
//! Extracted from `yamc` so downstream tools (cross-section plotters,
//! data converters, the GPU integration) can pull just nuclear data
//! without dragging in the transport engine. yamc itself depends on
//! this crate and re-exports its types under their original paths.

pub mod blend;
pub mod buffer;
pub mod composition;
pub mod config;
pub mod covariance;
pub mod data;
pub mod delayed_neutrons;
pub mod fission_photon;
pub mod group_structures;
pub mod interpolation;
pub mod load_scope;
pub mod nuclide;
pub mod nuclide_registry;
pub mod particle_type;
pub mod reaction;
pub mod reaction_product;
pub mod temperature;
pub mod urr;

// Grouped subsystems. Each subfolder's child modules are re-exported at the
// crate root below so the historic flat `yamc_nuclide::<module>` paths (and the
// internal `crate::<module>` paths) keep resolving unchanged after the move.
pub mod secondary;
pub use secondary::{sampling, secondary_correlated, secondary_kalbach, secondary_uncorrelated};

pub mod storage;
pub use storage::{in_memory_storage, nuclide_loader, url_cache};

#[cfg(feature = "arrow")]
pub mod arrow;
#[cfg(feature = "arrow")]
pub use arrow::{arrow_helpers, nuclide_arrow};

pub use buffer::F64Buffer;
pub use composition::{atomic_mass, expand_element, expand_element_enriched, expand_formula};
pub use config::{Config, CONFIG};

// ---- Nuclear-data load logging (process-global toggle) ----
// When enabled, each nuclide arrow file read prints one line (nuclide + path).
// The Model toggles this from `verbose=["nuclear_data"]` around a run; it is a
// process-global atomic so loads on rayon worker threads are visible too.
use std::sync::atomic::{AtomicBool, Ordering as LoadLogOrdering};
static LOG_NUCLIDE_LOADS: AtomicBool = AtomicBool::new(false);

/// Enable or disable per-file nuclear-data load logging (process-global).
pub fn set_load_logging(on: bool) {
    LOG_NUCLIDE_LOADS.store(on, LoadLogOrdering::Relaxed);
}

/// Whether per-file nuclear-data load logging is currently enabled.
pub fn load_logging_enabled() -> bool {
    LOG_NUCLIDE_LOADS.load(LoadLogOrdering::Relaxed)
}
pub use interpolation::interpolate_linear;
pub use load_scope::{LoadScope, SectionScope};
pub use nuclide::Nuclide;
pub use nuclide_registry::{NuclideId, NuclideRegistry};
pub use particle_type::ParticleType;
pub use reaction::Reaction;
pub use storage::set_storage;
