//! Host-side dispatcher for inelastic-collision angle / E_out
//! sampling.
//!
//! The implementation moved to
//! `yamc_physics::gpu::flat::inelastic_dispatch` (issue #111, stream
//! unification) so the CPU production transport, which builds without
//! the `gpu` feature, and this host-side twin call one
//! single-source-of-truth dispatcher. Re-exported under the historical
//! name so the `shared.rs` and test call sites are unchanged.

pub(super) use yamc_physics::gpu::flat::inelastic_dispatch::sample_inelastic_kinematics as sample_inelastic_angle;
