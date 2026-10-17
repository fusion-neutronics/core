//! Transmutation and activation driver over the [`yani`] solver.
//!
//! Takes a material, an irradiation schedule and a neutron spectrum (supplied
//! by the caller, or accumulated during transport by [`TransmutationTallies`])
//! and steps the inventory forward: collapse the spectrum against the
//! pointwise cross sections, build the reaction rates, hand the matrix to
//! CRAM, write the new composition back.
//!
//! Nothing here knows about transport, geometry or tallies as such. The
//! transport-coupled entry point (`Model::transmute`, which runs a transport
//! solve per timestep) stays in `yamc`, as does the MPI communicator whose
//! collectives [`TransmutationTallies::reduce_across_ranks`] drives through
//! [`CollectiveOps`].

pub mod branching_rule;
pub mod covariance_fold;
pub mod covariance_provenance;
pub mod covariance_sample;
pub mod cross_material;
pub mod d1s_uncertainty;
pub mod damage;
mod decay_branching_uncertainty;
pub mod derived;
mod fission_yield_uncertainty;
pub mod flux_uncertainty;
mod gas;
mod history_statistics;
mod isomer_branching_uncertainty;
mod joint_fit;
mod material_transmute;
pub mod multigroup;
pub mod nearest_correlation;
pub mod resonance_rates;
pub mod resonance_sampling;
mod results;
mod schedule;
pub mod screening;
pub mod self_shielding;
pub mod statistical;
mod transmutation;
mod transmutation_stepper;
mod transmutation_tallies;
pub mod uncertainty;

/// The temperature label `material`'s cross sections are read at.
///
/// The material's own label. Every transmutation entry point resolves and
/// loads it before any rate is collapsed ([`preload_activation_data`]), by the
/// rule transport applies (`Material::try_resolve_temperature`): the
/// temperature the material was given, else the only one its data carries,
/// else 294 K. Each nuclide then holds reactions at it, blended from the two
/// temperatures either side where its data only brackets it, or the run is
/// refused naming the nuclide that cannot serve it. That is what makes the
/// choice the same on every run and backed by reactions.
///
/// A material handed straight to a collapse without going through an entry
/// point is read where the same rule puts it (`Material::resolved_temperature`),
/// and `None` where the rule cannot place it. There is no per-nuclide default.
/// The lowest temperature a nuclide had loaded used to stand in for an unset
/// one, which read ENDF/B-VIII.1 Fe56 and TENDL-2017 U235 at 250 K, the lowest
/// they carry, where transport runs the same material at 294 K.
pub(crate) fn temperature_of(material: &yamc_materials::Material) -> Option<String> {
    if material.temperature().is_empty() {
        material.resolved_temperature(None).ok()
    } else {
        Some(material.temperature().to_string())
    }
}
// Re-export from yani (cram, chain, matrix now live there)
pub use yani::{
    build_matrix, build_matrix_triplets, cram48, cram48_sparse, load_chain, load_chain_parts,
    ChainNuclide, ChainReaction, FissionYieldWeights, LoadedChain, ReactionRates,
};

pub use branching_rule::{
    BranchingChannel, BranchingReport, BranchingState, DroppedChannel, UnmodelledRate,
    BRANCHING_RATE_TOLERANCE,
};
pub use isomer_branching_uncertainty::IsomerIntervalProductions;
pub use material_transmute::{
    activation_mts, apply_coupled_branching, preload_activation_data, screen_activation_products,
    transmute_material, transmute_material_shielded, transmute_materials, transport_cross_material,
    transport_replicas, CoupledDiagnostics, MultigroupSpectrum, TransmuteCase, TransmuteStep,
    TransportTallied,
};

/// Load the transmutation chain assembled from the configured per-subsection
/// sources (`yamc.transmutation_decay_data` / `_reactions` /
/// `_fission_yields`). Each source (a library keyword or a local path) is
/// resolved independently (keyword -> downloaded subsection tar), then the
/// parts are merged.
///
/// `_reactions` and `_fission_yields` can each be set to `None`, which leaves
/// that subsection out: nothing is downloaded and the chain is built without
/// it. That is the honest setting for a decay-only calculation and for a
/// material nothing in which fissions, and it is also the only way to run a
/// library that publishes no yields of its own without quietly folding another
/// library's in. `LoadedChain::parts` records what was left out, so a solve
/// that turns out to need it is refused rather than answered without it.
/// `_decay_data` has no such state: half-lives come from it and a chain
/// without them is empty.
///
/// When `yamc.transmutation_branch_ratios` names branching sources, each
/// one's `branching/` subsection is also resolved, and they are laid over the
/// chain in the order given and returned in `LoadedChain::branch`: each
/// `(parent, reaction)` channel takes its curves, and its MF=40 covariance,
/// from the first source that evaluates it (see
/// [`yani::parse_chain_parts_from_bytes`]), and the branching report names
/// that source per channel. The chain then carries grafted `(n,n')` and
/// metastable channels. On the coupled path the
/// verbatim energy-dependent curves are scored directly at the collision
/// energy during transport (see `transmutation_tallies`); the
/// flux-given `Material::transmute` path folds them against the user's
/// multigroup spectrum (see `material_transmute`). When none is named, the
/// overlay holds only the reactions subsection's own MT=5 `(n,X)`
/// multiplicities, which are folded the same way whatever branching source is
/// set, and every other reaction's product lands where the plain three-part
/// chain puts it.
pub fn load_configured_chain() -> Result<yani::LoadedChain, Box<dyn std::error::Error>> {
    let (decay_src, reactions_src, fpy_src, branch_srcs) = {
        let cfg = yamc_nuclide::config::Config::global();
        (
            cfg.get_transmutation_decay_data(),
            cfg.get_transmutation_reactions(),
            cfg.get_transmutation_fission_yields(),
            cfg.get_transmutation_branch_ratios(),
        )
    };
    let decay_dir = yamc_nuclide::url_cache::resolve_subsection(&decay_src, "decay")?;
    // `None` here is the user having turned the subsection off, not an absent
    // default: nothing is downloaded and the chain is built without it. What
    // that costs is checked where the rates are known, not here, because a
    // material that never fissions is entitled to no fission yields.
    let resolve = |src: Option<String>, subsection: &str| {
        src.map(|src| yamc_nuclide::url_cache::resolve_subsection(&src, subsection))
            .transpose()
    };
    let reactions_dir = resolve(reactions_src, "reactions")?;
    let fpy_dir = resolve(fpy_src, "fission_yields")?;
    let branch_dirs: Vec<(String, String)> = branch_srcs
        .into_iter()
        .map(|src| {
            let dir = yamc_nuclide::url_cache::resolve_subsection(&src, "branching")?;
            Ok((src, dir.to_string_lossy().into_owned()))
        })
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;
    let branching: Vec<(&str, &str)> = branch_dirs
        .iter()
        .map(|(src, dir)| (src.as_str(), dir.as_str()))
        .collect();
    yani::load_chain_parts(
        &decay_dir.to_string_lossy(),
        reactions_dir
            .as_ref()
            .map(|p| p.to_string_lossy())
            .as_deref(),
        fpy_dir.as_ref().map(|p| p.to_string_lossy()).as_deref(),
        &branching,
    )
}
pub use damage::{DamageRequest, DisplacementDamage};
pub use derived::{Estimate, LineEstimate};
pub use gas::{GAS_KEYS, GAS_NUCLIDES};
pub use history_statistics::{HistoryCovariance, RateCovariance, RateLabel, YieldChannelLabel};
pub use multigroup::{
    compute_multigroup_reaction_rates, reaction_rate_spectrum, scale_rates, EnergyGroups,
};
pub use results::{
    CollapseInputs, CollapseReuse, RateSpectrum, TransmutationResults, UncertaintyBreakdown,
};
pub use schedule::{duration_to_seconds, Schedule, ScheduleStep};
pub use screening::{ProductScreening, DEFAULT_PRODUCT_THRESHOLD};
pub use self_shielding::{Shape, Shielding, ShieldingInfo};
pub use transmutation::TransmutationDriver;
pub use transmutation_stepper::{ForwardEulerStepper, TransmutationStepper, DENSITY_FLOOR};
pub use transmutation_tallies::{
    CollectiveOps, PartialRates, ReplicaChanges, ReplicaTallied, TransmutationTallies,
};

/// The chain reaction name to MT mapping, from its single definition in
/// [`yani::reactions`]. Re-exported at crate visibility so the modules here
/// keep referring to it as `crate::reaction_type_to_mt`.
pub(crate) use yani::reactions::{mt_to_reaction_type, reaction_type_to_mt};
