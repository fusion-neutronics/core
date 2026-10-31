// Per-kind filter implementations (one module each).
pub mod cell;
pub mod energy;
pub mod energy_function;
pub mod material;
pub mod mesh;
pub mod parent_nuclide;
pub mod particle_type;
#[cfg(feature = "mesh")]
pub mod unstructured_mesh;

use crate::{
    CellFilter, EnergyFilter, EnergyFunctionFilter, MaterialFilter, MeshFilter,
    ParentNuclideFilter, ParticleTypeFilter,
};

/// Unified filter enum for tallies
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Filter {
    Cell(CellFilter),
    Material(MaterialFilter),
    Energy(EnergyFilter),
    /// Energy function filter that multiplies scores by an energy-dependent function.
    /// Used for dose calculations with ICRP dose coefficients.
    EnergyFunction(EnergyFunctionFilter),
    Mesh(MeshFilter),
    /// Unstructured mesh filter for scoring on tetrahedral meshes (via yamt).
    #[cfg(feature = "mesh")]
    UnstructuredMesh(crate::UnstructuredMeshFilter),
    ParticleType(ParticleTypeFilter),
    /// Parent nuclide filter for D1S decay photon tallies.
    /// Bins photon scores by the nuclide that produced the decay photon.
    ParentNuclide(ParentNuclideFilter),
}

impl Filter {
    /// Get the type name of this filter for validation
    pub fn type_name(&self) -> &'static str {
        match self {
            Filter::Cell(_) => "CellFilter",
            Filter::Material(_) => "MaterialFilter",
            Filter::Energy(_) => "EnergyFilter",
            Filter::EnergyFunction(_) => "EnergyFunctionFilter",
            Filter::Mesh(_) => "MeshFilter",
            #[cfg(feature = "mesh")]
            Filter::UnstructuredMesh(_) => "UnstructuredMeshFilter",
            Filter::ParticleType(_) => "ParticleTypeFilter",
            Filter::ParentNuclide(_) => "ParentNuclideFilter",
        }
    }

    /// Get the number of bins for this filter.
    ///
    /// Cell and Material filters bin over their list of IDs (1 when the filter
    /// carries a single ID -- the common case -- otherwise `len`). EnergyFunction
    /// and ParticleType are always single-bin (they gate rather than bin).
    pub fn num_bins(&self) -> usize {
        match self {
            Filter::Cell(f) => f.num_bins(),
            Filter::Material(f) => f.num_bins(),
            Filter::Energy(f) => f.num_bins(),
            Filter::EnergyFunction(_) => 1, // Always 1 bin
            Filter::Mesh(f) => f.num_bins(),
            #[cfg(feature = "mesh")]
            Filter::UnstructuredMesh(f) => f.num_bins(),
            Filter::ParticleType(_) => 1,
            Filter::ParentNuclide(f) => f.num_bins(),
        }
    }
}
