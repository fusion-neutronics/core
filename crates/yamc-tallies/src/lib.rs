//! Tallies: scores, spatial / energy / particle filters, estimators, and
//! per-history variance accumulation.
pub(crate) mod accumulator;
#[cfg(feature = "arrow")]
pub mod arrow_io;
pub mod combine;
pub mod convergence;
pub mod convergence_target;
pub mod estimator;
pub mod filter;
pub mod mesh;
pub mod mesh_parts;
pub mod mt;
pub mod result;
pub mod score;
pub mod tally;
pub mod transport_results;
pub mod units;

pub mod welford;

// Re-export main types for convenience
pub use convergence::{BinConvergence, ConvergenceLevel, ConvergenceStatus};
pub use convergence_target::{
    ConvergenceMetric, ConvergenceTarget, TallySelector, BIN_RELATIVE_ERROR_QUANTILE,
};
pub use estimator::Estimator;
pub use filter::cell::CellFilter;
pub use filter::energy::EnergyFilter;
pub use filter::energy_function::{EnergyFunctionFilter, Interpolation};
pub use filter::material::MaterialFilter;
pub use filter::mesh::{MeshFilter, MeshKind};
pub use filter::parent_nuclide::ParentNuclideFilter;
pub use filter::particle_type::ParticleTypeFilter;
pub use filter::surface::{CosineFilter, SurfaceFilter};
#[cfg(feature = "mesh")]
pub use filter::unstructured_mesh::UnstructuredMeshFilter;
pub use mesh::{CylindricalMesh, RegularRectangularMesh};
pub use mesh_parts::MeshParts;
pub use mt::Mt;
pub use result::{ConvergencePoint, Normalisation, StatisticalChecks, TallyResult};
pub use score::{
    CurrentDirection, CurrentScore, DamageEnergyScore, FluxScore, HeatingLocalScore, HeatingScore,
    PhotonComponent, PhotonXSScore, ProductionScore, ReactionRateScore, Score, ScoreKind,
};
pub use tally::{NuclideBin, Tally};
pub use transport_results::TransportResults;
pub use units::{SourceStrength, StrengthPer};
pub use welford::{AggMoments, ScorePdf};
