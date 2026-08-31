//! Shared `#[cube]` sampling helpers: free-flight distance to collision,
//! tabulated cross-section lookup/interpolation, the continuous-tabular
//! outgoing-energy law, the correlated (File-6 Law-1) outgoing-energy law, the
//! Kalbach-Mann (File-6 Law-4) correlated outgoing-energy + angle law, the
//! N-body phase-space (File-6 Law-6) outgoing-energy + isotropic-angle law, the
//! Evaporation/Maxwell/Watt outgoing-energy rejection draws, the Marsaglia
//! azimuthal-direction sampler, the tabulated angle-CDF inverter, and the
//! stochastic incident-energy bracket pick.

pub mod angle_cdf_invert;
pub mod distance_to_collision;
pub mod energy_bracket;
pub mod eout_continuous_tabular;
pub mod eout_correlated;
pub mod eout_rejection;
pub mod fission_chi;
pub mod kalbach_mann;
pub mod marsaglia_phi;
pub mod nbody_phase_space;
pub mod xs_lookup;
