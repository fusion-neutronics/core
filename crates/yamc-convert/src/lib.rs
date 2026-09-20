//! Build the transport Arrow format from ACE and ENDF evaluations.
//!
//! The yamc half of "yamc and yani should each convert ENDF into the format
//! they need". Unlike the transmutation half, the ENDF route here needs NJOY to
//! reconstruct resonances and Doppler broaden, so it cannot reach zero external
//! tools; the ACE route can.
//!
//! Writes the full transport section set: `nuclide`, `reactions`, `products`,
//! `distributions`, `fast_xs`, `urr`, `total_nu` and `fission_photon` for
//! neutrons, and `element`, `subshells`, `compton` and `bremsstrahlung` for
//! photons. The redundant MTs come from `yamc_nuclide::synthesis`, shared with
//! the loader, and are independently checkable without a second
//! implementation: the sums it builds must reproduce the evaluation's own
//! total.

pub mod covariance;
pub mod distributions;
pub mod energy_ranges;
pub mod entry;
pub mod fast_xs;
pub mod fission_nu;
pub mod heating;
pub mod marker;
pub mod nuclide;
pub mod photon;
pub mod products;
pub mod reaction_ranges;
pub mod reactions;
pub mod sections;
pub mod univariate_flat;
