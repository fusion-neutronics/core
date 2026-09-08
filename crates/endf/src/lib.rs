//! A reader for ENDF-6 formatted evaluated nuclear data files.
//!
//! The layout follows the format itself: [`records`] holds the fixed-width
//! record primitives, [`function`] the tabulated function types, [`mf`] one
//! module per ENDF file (MF), and [`material`] the splitting of a file into
//! materials and sections.
//!
//! ```no_run
//! let material = endf::Material::from_file("n-095_Am_244.endf")?;
//! let total = material.mf3(1).expect("total cross section");
//! println!("{} barns at {} eV", total.sigma.eval(0.0253), 0.0253);
//! # Ok::<(), endf::Error>(())
//! ```
//!
//! Everything here describes the file format and nothing more. A
//! simulation-ready projection of this data (reconstructed resonances,
//! summed reactions, unionised grids) belongs in a consumer built on top,
//! not in this crate.

#![forbid(unsafe_code)]

pub mod ace;
pub mod angle_energy;
pub mod chain;
pub mod data;
pub mod decay;
pub mod error;
pub mod fission_energy;
pub mod function;
pub mod incident_neutron;
pub mod incident_photon;
pub mod material;
pub mod mf;
pub mod njoy;
pub mod photon_data;
pub mod product;
pub mod radionuclide_production;
pub mod reaction;
pub mod records;
pub mod spline;

#[cfg(test)]
mod testdata;
pub mod univariate;
pub mod urr;

// The ACE table is where a transport code starts, so its types are at the
// root rather than behind `ace::`. `MetastableScheme` in particular is an
// argument to `IncidentNeutron::from_ace`, and needing a module path to call a
// re-exported method is the kind of friction that shows up immediately.
pub use ace::{get_table, get_tables, tables_from_str, MetastableScheme, Table, TableType};
pub use angle_energy::{
    AngleEnergy, CorrelatedAngleEnergy, KalbachMann, NBodyPhaseSpace, UncorrelatedAngleEnergy,
};
pub use chain::{collect_q_values, q_values, Chain, Nuclide, QValues, ReactionInfo, REACTIONS};
pub use data::{gnds_name, zam, EV_PER_MEV, K_BOLTZMANN};
pub use decay::{Decay, DecayMode, FissionProductYields, FissioningNuclide, ProductYield};
pub use error::{Error, Result};
pub use fission_energy::FissionEnergyRelease;
pub use function::{Polynomial, Tabulated1D, Tabulated2D};
pub use incident_neutron::IncidentNeutron;
pub use incident_photon::{AtomicRelaxation, IncidentPhoton, PhotonReaction};
pub use material::{get_materials, materials_from_str, Interpretation, Material, Section};
pub use photon_data::{ComptonProfile, PhotonData};
pub use product::{EmissionMode, Product, Yield};
pub use radionuclide_production::{
    isomer_table, isomer_table_from_materials, level_to_isomeric_state, radionuclide_production,
    Isomer, IsomerTable, RadionuclideProduction,
};
pub use reaction::{reaction_mt, reaction_name, Reaction, FISSION_MTS};
pub use records::{float_endf, int_endf, Cont, Head, ListRecord, Matrix, Reader, Tab1, Tab2};
pub use spline::CubicSpline;
pub use urr::ProbabilityTables;
