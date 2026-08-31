//! Tallies: scores, spatial / energy / particle filters, estimators, and
//! per-history variance accumulation.
pub(crate) mod accumulator;
#[cfg(feature = "arrow")]
pub mod arrow_io;
pub mod combine;
pub mod convergence_target;
pub mod estimator;
pub mod filter;
pub mod mesh;
pub mod mt;
pub mod result;
pub mod score;
pub mod simulation_results;
pub mod tally;

pub mod welford;

// Re-export main types for convenience
pub use convergence_target::{ConvergenceMetric, ConvergenceTarget, TallySelector};
pub use estimator::Estimator;
pub use filter::cell::CellFilter;
pub use filter::energy::EnergyFilter;
pub use filter::energy_function::EnergyFunctionFilter;
pub use filter::material::MaterialFilter;
pub use filter::mesh::{MeshFilter, MeshKind};
pub use filter::parent_nuclide::ParentNuclideFilter;
pub use filter::particle_type::ParticleTypeFilter;
#[cfg(feature = "mesh")]
pub use filter::unstructured_mesh::UnstructuredMeshFilter;
pub use mesh::{CylindricalMesh, RegularRectangularMesh};
pub use mt::Mt;
pub use result::{ConvergencePoint, StatisticalChecks, TallyResult};
pub use score::{
    DamageEnergyScore, FluxScore, HeatingLocalScore, HeatingScore, PhotonComponent, PhotonXSScore,
    ProductionScore, ReactionRateScore, Score, ScoreKind,
};
pub use simulation_results::SimulationResults;
pub use tally::{NuclideBin, Tally};
pub use welford::{AggMoments, ScorePdf};
