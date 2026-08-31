//! Standalone validation and device-capability probe kernels. Not part of
//! production transport: they probe shader features (f64 atomics,
//! transcendentals, fixed-point tally) and validate individual primitives
//! against their CPU mirrors.

pub mod atomic_f64;
pub mod atomic_relaxation;
pub mod atomic_u64_add;
pub mod atomic_u64_cas;
pub mod compton_doppler;
pub mod compton_scatter;
pub mod f64_arith;
pub mod f64_sin_cos;
pub mod f64_transcendentals;
pub mod fixed_point_tally;
pub mod incoherent_sf;
pub mod pe_subshell;
pub mod photon_element_select;
pub mod photon_select;
pub mod rayleigh_scatter;
pub mod rotate_mu_phi;
pub mod sphere_distance;
