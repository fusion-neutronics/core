//! Incremental neutron build-up kernels -- each adds one transport phase
//! on top of the previous (free flight, XS lookup, single collision,
//! tally, multi-step). Superseded by `neutron::transport` for production,
//! but kept as a per-phase validation and benchmark harness.

pub mod collision_sampling;
pub mod elastic_scatter;
pub mod full_step;
pub mod full_step_multi_tally;
pub mod full_step_tally;
pub mod multi_step;
pub mod transport_step;
