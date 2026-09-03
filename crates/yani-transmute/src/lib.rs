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

pub mod covariance_fold;
pub mod covariance_sample;
pub mod derived;
pub mod flux_uncertainty;
mod material_transmute;
pub mod multigroup;
mod results;
mod schedule;
pub mod self_shielding;
mod transmutation;
mod transmutation_stepper;
mod transmutation_tallies;
pub mod uncertainty;

/// The temperature to read a nuclide at when the material names none.
///
/// Not `nd.energy.keys().next()`, which is what the three call sites below used
/// to do. `energy` is keyed by every temperature the Arrow file carries, and
/// that is a superset of the ones whose reactions were loaded: ENDF/B-8.1 Fe56
/// has grids for 250, 294, 600, 900, 1200 and 2500 K, and also a `0` that no
/// reaction set backs. `keys().next()` returns an arbitrary one of them, and a
/// different one per parse, because every `HashMap` is seeded separately. Draw
/// the `0` and `reactions_for_temp` answers `None`, the caller gives up on that
/// nuclide, and it contributes no reaction rates at all -- an irradiation that
/// activates nothing and reports zero activity, for the same input, about one
/// run in seven.
///
/// `loaded_temperatures` lists exactly the temperatures that do have reactions,
/// sorted, so its first entry is both valid and the same on every run.
pub(crate) fn default_temperature(nd: &yamc_nuclide::Nuclide) -> Option<String> {
    // Prefer a temperature that actually carries reactions. `loaded_temperatures`
    // is in the order the file lists them, and nothing guarantees the first
    // entry is backed by a reaction set on every dataset; falling back to it
    // keeps callers that only want an energy grid working.
    nd.loaded_temperatures
        .iter()
        .find(|t| nd.reactions_for_temp(t).is_some_and(|r| !r.is_empty()))
        .or_else(|| nd.loaded_temperatures.first())
        .cloned()
}
// Re-export from yani (cram, chain, matrix now live there)
pub use yani::{
    build_matrix, build_matrix_triplets, cram48, cram48_sparse, load_chain, load_chain_parts,
    ChainNuclide, ChainReaction, FissionYieldWeights, LoadedChain, ReactionRates,
};

pub use material_transmute::{
    activation_mts, apply_coupled_branching, preload_activation_data, transmute_material,
    transmute_material_shielded, MultigroupSpectrum, TransmuteStep,
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
/// When `yamc.transmutation_branch_ratios` is set, its `branching/` subsection
/// is also resolved and returned in `LoadedChain::branch`: the chain then
/// carries grafted `(n,n')` metastable channels. On the coupled path the
/// verbatim energy-dependent curves are scored directly at the collision
/// energy during transport (see `transmutation_tallies`, issue #218); the
/// flux-given `Material::transmute` path folds them against the user's
/// multigroup spectrum (see `material_transmute`). When unset, the overlay is
/// empty and physics matches the plain three-part chain.
pub fn load_configured_chain() -> Result<yani::LoadedChain, Box<dyn std::error::Error>> {
    let (decay_src, reactions_src, fpy_src, branch_src) = {
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
    let branch_dir = resolve(branch_src, "branching")?;
    yani::load_chain_parts(
        &decay_dir.to_string_lossy(),
        reactions_dir
            .as_ref()
            .map(|p| p.to_string_lossy())
            .as_deref(),
        fpy_dir.as_ref().map(|p| p.to_string_lossy()).as_deref(),
        branch_dir.as_ref().map(|p| p.to_string_lossy()).as_deref(),
    )
}
pub use derived::{Estimate, LineEstimate};
pub use multigroup::{
    compute_multigroup_reaction_rates, reaction_rate_spectrum, scale_rates, EnergyGroups,
};
pub use results::{CollapseInputs, RateSpectrum, TransmutationResults};
pub use schedule::{duration_to_seconds, Schedule, ScheduleStep};
pub use self_shielding::{Shape, Shielding, ShieldingInfo};
pub use transmutation::TransmutationDriver;
pub use transmutation_stepper::{ForwardEulerStepper, TransmutationStepper, DENSITY_FLOOR};
pub use transmutation_tallies::{CollectiveOps, PartialRates, TransmutationTallies};

/// The chain reaction name to MT mapping, from its single definition in
/// [`yani::reactions`]. Re-exported at crate visibility so the modules here
/// keep referring to it as `crate::reaction_type_to_mt`.
pub(crate) use yani::reactions::{mt_to_reaction_type, reaction_type_to_mt};
