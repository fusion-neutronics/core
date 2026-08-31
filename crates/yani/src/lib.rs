//! yani -- Yet Another Nuclide Inventory
//!
//! CRAM matrix exponential solver, transmutation chain reader (Apache Arrow),
//! and burnup matrix construction for nuclear transmutation.

pub mod chain;
pub mod chain_arrow;
pub mod cram;
pub mod matrix;
pub mod reactions;

pub use chain::{
    fission_yield_interp_weights, load_chain, load_chain_parts, populated_nuclides,
    reachable_nuclides, reduce_chain, BranchCurve, BranchQuantity, BranchTable, ChainNuclide,
    ChainParts, ChainReaction, DecaySource, DecaySourceDistribution, FissionYield, FissionYieldSet,
    LoadedChain,
};
pub use chain_arrow::{
    export_chain_arrow, export_chain_parts, parse_chain_arrow, parse_chain_parts,
    parse_chain_parts_from_bytes, ChainSections, SectionFiles,
};
pub use cram::{cram48, cram48_sparse};
pub use matrix::{
    build_matrix, build_matrix_triplets, per_edge_rates, EdgeRates, FissionYieldWeights,
};
pub use reactions::{mt_to_reaction_type, reaction_type_to_mt, REACTION_MT_MAP};

use std::collections::HashMap;

/// Reaction rates for a material.
/// Structure: nuclide_name -> reaction_type -> rate [1/s]
///
/// The rate is sigma * phi (microscopic cross section times scalar flux)
/// for neutron-induced reactions.
pub type ReactionRates = HashMap<String, HashMap<String, f64>>;
