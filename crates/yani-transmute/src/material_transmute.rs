use crate::covariance_fold::fold_rate_covariance;
use crate::covariance_sample::Sampler;
use crate::multigroup::{compute_multigroup_reaction_rates_shielded, scale_rates};
use crate::results::TransmutationResults;
use crate::self_shielding::{Shielding, ShieldingInfo};
use crate::transmutation_tallies::PartialRates;
use crate::uncertainty::{
    densities_of, settled, DataUncertainty, Ensemble, Info, BLOCK, MAX_SAMPLES, MIN_SAMPLES,
};
use crate::{reaction_type_to_mt, ForwardEulerStepper, TransmutationStepper};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use yamc_materials::material::{DensityUnits, Material};
use yamc_nuclide::load_scope::LoadScope;
use yamc_nuclide::nuclide::get_or_load_nuclide;
use yani::{
    per_edge_rates, BranchQuantity, BranchTable, ChainNuclide, FissionYieldWeights, ReactionRates,
};

/// One spectrum's collapsed rates, fission-yield weights and folded chain.
///
/// The three travel together because they are computed together, once per
/// distinct spectrum, and every step and every replica reads all three.
type PerSpectrum = (
    ReactionRates,
    FissionYieldWeights,
    Arc<HashMap<String, ChainNuclide>>,
);

/// A multigroup neutron spectrum over energy bins.
///
/// `boundaries` are `masses.len() + 1` strictly ascending bin edges (eV);
/// `masses` are the per-group probability mass (shape). Only the shape matters
/// for collapsing cross sections (the per-step `rate` carries the magnitude),
/// so these are normalized by the caller.
#[derive(Clone, Debug)]
pub struct MultigroupSpectrum {
    pub boundaries: Vec<f64>,
    pub masses: Vec<f64>,
    /// Per-group RELATIVE standard deviation of the flux, when the caller knows
    /// it (issue #559).
    ///
    /// Relative rather than absolute because `masses` is a normalized shape and
    /// the pulse `rate` carries the magnitude, so an absolute sigma would be in
    /// units this struct no longer has. A relative one is invariant under that
    /// split and is what the perturbation multiplies by.
    ///
    /// `None` is the common case and is NOT zero. A spectrum taken from a
    /// published reference set has no stated error, and that has to stay
    /// distinguishable from one measured to be exact.
    pub relative_std_dev: Option<Vec<f64>>,
}

/// One step of a standalone-transmutation timeline.
#[derive(Clone, Debug)]
pub struct TransmuteStep {
    /// Step duration in seconds.
    pub dt: f64,
    /// `Some((spectrum_index, rate))` for an irradiation step, where `rate` is
    /// the total flux magnitude [n/cm^2/s]; `None` for a decay-only step.
    pub irradiation: Option<(usize, f64)>,
}

/// The MTs a transport-free collapse will ask for across a whole chain.
///
/// Two sources, both needed. The chain's own reaction kinds give the transport
/// cross sections the rate collapse folds against. The branching overlay adds
/// the parents whose MF=9 yield curves are weighted by that same transport cross
/// section in `fold_state_fractions`; miss those and the isomeric split silently
/// falls back to the base branching.
///
/// `(n,n')` is deliberately absent from `REACTION_MT_MAP` because it has no
/// transport total, so it contributes no MT here and its rate comes from the
/// overlay's MF=10 partials instead.
pub fn activation_mts(
    chain: &HashMap<String, yani::ChainNuclide>,
    branch: &BranchTable,
) -> HashSet<i32> {
    let mut mts = HashSet::new();
    for cn in chain.values() {
        for rxn in &cn.reactions {
            if let Some(mt) = reaction_type_to_mt(&rxn.kind) {
                mts.insert(mt);
            }
        }
    }
    for kinds in branch.values() {
        for kind in kinds.keys() {
            if let Some(mt) = reaction_type_to_mt(kind) {
                mts.insert(mt);
            }
        }
    }
    mts
}

/// Transmute a material over a timeline of per-step multigroup spectra.
///
/// Performs standalone transmutation without re-running transport. Each
/// irradiation step references one of `spectra` (the spectrum shape) and a
/// `rate` (the total flux magnitude); reaction rates are collapsed once per
/// distinct spectrum from the initial material and scaled by `rate` per step.
/// Decay-only steps (`irradiation = None`) just let the material decay.
///
/// # Arguments
/// * `material` - Material to transmute. The cross sections this needs are
///   loaded INTO it, so a second call finds them already there and does no
///   Arrow decoding at all; the composition is not touched. Call
///   [`Material::release_nuclear_data`] to give the memory back, which a sweep
///   over thousands of distinct compositions should do (issue #576, finding 3).
/// * `spectra` - Distinct multigroup spectra referenced by the steps
/// * `steps` - The irradiation/cooling timeline
/// * `chain` - Parsed transmutation chain data
///
/// # Returns
/// [`TransmutationResults`], the same type the coupled path returns, keyed by
/// the material's own `material_id` (0 when it has none). Index 0 of its
/// materials is the initial composition and index `i` is the state after step
/// `i - 1`, so it carries one entry more than this used to return as a bare
/// `Vec`. It also carries the per-edge reaction rates each step was solved
/// with, which the solve computes to build the burnup matrix and used to throw
/// away (issue #505).
///
/// `uncertainty` asks for nuclear-data uncertainty on the inventory, from the
/// MF=33 activation cross-section covariance (issue #514). `None` is the
/// default path: no covariance is read, nothing is folded or factorized, the
/// step loop runs once, and the result is bit-identical to a build without any
/// of it.
#[allow(clippy::too_many_arguments)]
pub fn transmute_material(
    material: &mut Material,
    spectra: &[MultigroupSpectrum],
    steps: &[TransmuteStep],
    chain: Arc<HashMap<String, yani::ChainNuclide>>,
    branch: &BranchTable,
    parts: yani::ChainParts,
    uncertainty: Option<&DataUncertainty>,
) -> Result<TransmutationResults, Box<dyn std::error::Error>> {
    transmute_material_shielded(
        material,
        spectra,
        steps,
        chain,
        branch,
        parts,
        uncertainty,
        None,
    )
}

/// As [`transmute_material`], optionally correcting the collapse for resonance
/// self-shielding.
///
/// `shielding` of `None` is [`transmute_material`] exactly: no flux shape is
/// built, MT=1 and MT=2 are not read, and the inventories are bit-identical to
/// a build without [`crate::self_shielding`].
#[allow(clippy::too_many_arguments)]
pub fn transmute_material_shielded(
    material: &mut Material,
    spectra: &[MultigroupSpectrum],
    steps: &[TransmuteStep],
    chain: Arc<HashMap<String, yani::ChainNuclide>>,
    branch: &BranchTable,
    parts: yani::ChainParts,
    uncertainty: Option<&DataUncertainty>,
    shielding: Option<&Shielding>,
) -> Result<TransmutationResults, Box<dyn std::error::Error>> {
    // Validate spectra.
    for (i, s) in spectra.iter().enumerate() {
        if s.boundaries.len() != s.masses.len() + 1 {
            return Err(format!(
                "spectrum {i}: boundaries ({}) must be one longer than masses ({})",
                s.boundaries.len(),
                s.masses.len()
            )
            .into());
        }
        // Before the ascending check below, which a NaN passes: every
        // comparison against a NaN is false. A NaN boundary reaching the
        // collapse gives a NaN group average and an inventory of NaNs, with no
        // error anywhere along the way. `Histogram::new` now refuses one at the
        // Python boundary too; this is the guard for the Rust entry point, and
        // it is also what leaves the collapse's point set a function of an
        // ascending grid alone (issue #576).
        if let Some(bad) = s.boundaries.iter().position(|e| !e.is_finite()) {
            return Err(format!(
                "spectrum {i}: boundary {bad} is {}, not a finite energy",
                s.boundaries[bad]
            )
            .into());
        }
        if let Some(bad) = s.boundaries.windows(2).position(|w| w[1] <= w[0]) {
            return Err(format!(
                "spectrum {i}: boundaries must ascend strictly, but boundary {bad} is {} and \
                 boundary {} is {}",
                s.boundaries[bad],
                bad + 1,
                s.boundaries[bad + 1]
            )
            .into());
        }
        if s.masses.iter().any(|&m| !m.is_finite() || m < 0.0) {
            return Err(format!("spectrum {i}: masses must be finite and non-negative").into());
        }
    }
    // Validate steps.
    for (i, st) in steps.iter().enumerate() {
        if st.dt < 0.0 {
            return Err(format!("step {i}: dt = {} is negative", st.dt).into());
        }
        if let Some((idx, rate)) = st.irradiation {
            if idx >= spectra.len() {
                return Err(format!(
                    "step {i}: spectrum index {idx} out of range ({} spectra)",
                    spectra.len()
                )
                .into());
            }
            if rate < 0.0 {
                return Err(format!("step {i}: rate = {rate} is negative").into());
            }
        }
    }

    // A chain that can drive none of this material's nuclides. Every reaction
    // it holds has a parent the material does not contain, so the irradiation
    // produces nothing and the schedule solves to the composition it started
    // with. That is a silent zero: no step fails, no rate is negative, and the
    // mistake surfaces only as a decay heat of exactly zero much later, which
    // names neither the chain nor what it was built for.
    //
    // It happens whenever chains are scoped per material and two of them share
    // a path, since the second conversion replaces the first and leaves a
    // directory whose name still says otherwise. The manifest now records the
    // parents each subsection covers; this is the check that acts on them.
    //
    // Stronger than the membership check in `preload_activation_data`, which
    // asks whether any of the material's nuclides appear in the chain at all.
    // Appearing is not enough: a nuclide can be in the chain purely as somebody
    // else's product, carrying no reactions of its own, and a material made
    // only of those is exactly as undrivable as one that is absent. The two
    // guards are kept separate because that one must also cover the decay-only
    // path, where having no reactions is not a fault.
    //
    // Only when something is actually irradiated. A decay-only schedule drives
    // no reactions by construction, and a chain holding none of this material's
    // parents is the right chain for it.
    if steps.iter().any(|st| st.irradiation.is_some()) {
        let has_reactions =
            |name: &String| chain.get(name).is_some_and(|cn| !cn.reactions.is_empty());
        // Already sorted by `get_nuclides`, which the message relies on.
        let held = material.get_nuclides();
        if !held.iter().any(has_reactions) {
            let mut parents: Vec<&str> = chain
                .values()
                .filter(|cn| !cn.reactions.is_empty())
                .map(|cn| cn.name.as_str())
                .collect();
            parents.sort_unstable();
            let covers = if parents.is_empty() {
                "nothing".to_string()
            } else {
                parents.join(" ")
            };
            return Err(format!(
                "this chain drives none of the material's nuclides, so the irradiation would \
                 produce nothing and every step would return the starting composition.\n  \
                 material holds: {}\n  chain has reactions for: {covers}\n\
                 A chain is scoped to the nuclides it was built from, so build one for this \
                 material rather than reusing one built for another.",
                held.join(" "),
            )
            .into());
        }
    }

    // Load nuclear data for all chain nuclides (not just initial composition)
    // so reaction rates can be computed for daughter products. Into the
    // CALLER's material, so a second call on it finds the data already there.
    preload_activation_data(material, &chain, branch, uncertainty, shielding)?;

    let mut current_material = material.clone();
    if current_material.density_units != DensityUnits::Sum {
        current_material = current_material.to_sum_mode()?;
    }

    // Collapse reaction rates once per distinct spectrum from the initial
    // material, at unit total flux (masses are normalized, so this is the rate
    // per n/cm^2/s). Each step then scales its spectrum's rates by its `rate`.
    //
    // When an isomeric-branching overlay is supplied, fold it against the same
    // spectrum shape here (rate-compute time): this injects the `(n,n')`
    // metastable-production rates and yields a per-spectrum chain whose
    // isomeric branching ratios are flux-weighted rather than fixed. Branching
    // fractions are magnitude-independent, so the per-step `scale_rates` (which
    // scales only the rates) leaves them correct.
    //
    // The report is merged across spectra rather than assigned. It used to be
    // `shielding_info = info` inside this closure, so a schedule naming two
    // spectra reported only the second one's nuclides -- and #564's whole point
    // is that report naming what a dilute run did not correct for (issue #576).
    let mut shielding_info = ShieldingInfo::default();
    let mut per_spectrum: Vec<PerSpectrum> = Vec::with_capacity(spectra.len());
    for s in spectra {
        // Where an evaluation ends the cross section is unknown, and the fold
        // treats it as zero. A sliver of flux there is a rounding matter; more
        // than that and every rate on the nuclide would be understated by data
        // that does not exist, so the run stops and says which nuclide and how
        // much rather than answering as if it knew.
        let above = crate::multigroup::spectrum_above_evaluation(
            &current_material,
            &s.masses,
            &s.boundaries,
        );
        if let Some((name, top, fraction)) = above
            .iter()
            .find(|(_, _, fraction)| *fraction > crate::multigroup::ABOVE_EVALUATION_TOLERANCE)
        {
            return Err(format!(
                "{:.3}% of the spectrum lies above {top:.4e} eV, the last energy point in \
                 {name}'s evaluation. No cross section exists there, so the rates on {name} \
                 would be understated by that share. Cut the spectrum at the evaluation's \
                 top energy, or use a library evaluated to higher energy.",
                fraction * 100.0
            )
            .into());
        }
        let (mut rates, fy_weights, info) = compute_multigroup_reaction_rates_shielded(
            &current_material,
            &chain,
            &s.masses,
            &s.boundaries,
            1.0,
            shielding,
        );
        shielding_info.merge(info);
        let folded_chain =
            fold_branching_into_chain(&current_material, &chain, branch, s, &mut rates);
        per_spectrum.push((rates, fy_weights, folded_chain));
    }
    // Left sequential deliberately: the collapse inside it is already a
    // per-nuclide `par_iter`, so a second level of parallelism here would only
    // contend with it.

    let stepper = ForwardEulerStepper;
    let no_fission: FissionYieldWeights = HashMap::new();

    // Keyed by the material's own id, the way `TransmutationResults` is keyed
    // throughout. A material that was never given one answers to 0, which is
    // unambiguous here: this entry point transmutes exactly one material.
    let material_id = current_material.material_id.unwrap_or(0);
    let mut results = TransmutationResults::new(
        steps.iter().map(|st| st.dt).collect(),
        steps
            .iter()
            .map(|st| st.irradiation.map_or(0.0, |(_, rate)| rate))
            .collect(),
    );
    results.add_initial(material_id, current_material.clone());

    // The state every replica starts from: the same composition the nominal run
    // starts from, with the same nuclear data already loaded. Captured before
    // the step loop, which mutates `current_material` in place.
    let initial = current_material.clone();

    for st in steps {
        // The fission-yield weights are a normalized shape, so `scale_rates`
        // (which carries the magnitude) leaves them correct as they are.
        let (step_rates, step_weights, step_chain) = match st.irradiation {
            Some((idx, rate)) => (
                scale_rates(&per_spectrum[idx].0, rate),
                &per_spectrum[idx].1,
                &per_spectrum[idx].2,
            ),
            // Decay-only step: no reaction rates, so the base chain suffices
            // and no fission rate can demand a fold.
            None => (HashMap::new(), &no_fission, &chain),
        };
        current_material = stepper.step(
            &current_material,
            step_chain,
            &step_rates,
            step_weights,
            parts,
            st.dt,
        )?;
        // The per-edge rates are the matrix's own products, which the solve
        // computes to build the burnup matrix and used to throw away. Taken
        // from the same chain and rates the step was solved with, isomeric
        // overlay included, so a decay-only step records an empty map (issue
        // #505).
        results.add_step_rates(material_id, per_edge_rates(step_chain, &step_rates));
        results.add_step(material_id, current_material.clone());
    }

    if let Some(request) = uncertainty {
        let (ensemble, info) = run_replicas(
            &initial,
            spectra,
            steps,
            &per_spectrum,
            &chain,
            parts,
            &stepper,
            request,
            shielding,
        )?;
        results.uncertainty.insert(material_id, ensemble);
        results.uncertainty_info = Some(info);
    }

    // Recorded whether or not shielding ran: "not shielded" and "shielded and
    // nothing moved" are different claims, and a dilute run that should have
    // been shielded is the case #564 is about.
    results.shielding_info = Some(shielding_info);

    // Kept so routes can be derived from the same topology the solve used,
    // rather than from whatever a second load of the chain path returns.
    results.chain = Some(Arc::clone(&chain));

    // And the spectra it collapsed against, for the same reason: an
    // energy-resolved view of a rate is a statement about the spectrum that
    // drove it, and re-deriving one channel's breakdown on demand costs a few
    // KB of stored spectrum rather than the tens of MB the whole breakdown
    // would (yani#27).
    results.collapse = Some(crate::results::CollapseInputs {
        spectra: spectra
            .iter()
            .map(|s| (s.boundaries.clone(), s.masses.clone()))
            .collect(),
        step_spectrum: steps
            .iter()
            .map(|st| st.irradiation.map(|(idx, _)| idx))
            .collect(),
        shielding: shielding.copied(),
    });

    Ok(results)
}

/// What one replica produces, and nothing else.
///
/// Held rather than written into the driver's counters as it goes, so a
/// replica reads only shared read-only state and the merge order is the
/// driver's to fix (issue #576, finding 4).
struct ReplicaOutcome {
    /// This replica's inventory at each schedule step.
    densities: Vec<HashMap<String, f64>>,
    /// Rates the sampler had to truncate, for this replica alone.
    truncations: crate::covariance_sample::Truncations,
    /// The two flux counters a replica actually produces. NOT the whole
    /// `FluxCoverage`: `Info::add_flux_coverage` assigns the spectrum counts,
    /// which are established before the loop, so folding a replica's zeros over
    /// them would erase them.
    flux_bins_sampled: usize,
    flux_bins_floored: usize,
}

/// Load the cross sections a transport-free solve of this material needs, into
/// the material itself.
///
/// Into the caller's own `Material` rather than a local clone, which is the
/// point (issue #576, finding 3). The loaded `Arc<Nuclide>`s used to survive a
/// call only inside the returned `TransmutationResults`, because
/// `GLOBAL_NUCLIDE_CACHE` holds `Weak` references and the stepper's per-step
/// materials drop `nuclide_data`. A sweep that dropped the previous results
/// therefore re-decoded every reachable nuclide's Arrow directory on every
/// call: 1.8 s of a 2.3 s call on a steel against ENDF/B-8.1.
///
/// The material now owns them, so this is a no-op on the second call and the
/// memory is the caller's to release with
/// [`Material::release_nuclear_data`] -- which a sweep over thousands of
/// DISTINCT compositions should do, and which is why the cache is not simply
/// made strong (that would undo #401 for every code path in the workspace).
///
/// Idempotent, and safe to call yourself before a batch of transmutes.
pub fn preload_activation_data(
    material: &mut Material,
    chain: &HashMap<String, yani::ChainNuclide>,
    branch: &BranchTable,
    uncertainty: Option<&DataUncertainty>,
    shielding: Option<&Shielding>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Covariance rides on the same load rather than a second pass over the
    // same directories, and it is its own axis of the scope so the cache
    // can tell an entry loaded for an ordinary transmute apart from one
    // loaded for an uncertainty run.
    let want_covariance =
        uncertainty.is_some_and(|u| u.wants(crate::uncertainty::Source::CrossSections));

    let temp_filter = if !material.temperature().is_empty() {
        let mut s = HashSet::new();
        s.insert(material.temperature().to_string());
        Some(s)
    } else {
        None
    };

    // Only nuclides this material can actually reach need cross sections.
    // Walking the whole chain loads 556 of 3820 nuclides on ENDF/B-8.1
    // whatever is being irradiated, which is where the multi-GB footprint
    // came from (issue #401).
    //
    // The closure is exact, not a truncation: the stepper follows a reaction
    // edge only when its parent has a loaded rate
    // (`transmutation_stepper.rs`, "Only follow reaction pathways with
    // non-zero rates"), and it walks a subset of these edges from a subset
    // of these seeds, so its visited set is a subset of this one at every
    // step. Every reaction that fires without the filter still fires.
    //
    // Seeds are the composition UNION whatever is already loaded. The
    // composition alone is not enough: `Material::transmute` is reached
    // directly from Python without `ensure_nuclides_loaded`, so
    // `nuclide_data` can be empty, and seeding from it would load nothing
    // and silently return pure decay.
    let seeds: Vec<String> = material
        .nuclides
        .keys()
        .chain(material.nuclide_data.keys())
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    if !seeds.is_empty() && !seeds.iter().any(|s| chain.contains_key(s)) {
        return Err(format!(
            "none of the material's nuclides ({}) appear in the transmutation chain; \
             check the chain covers this composition (metastables are spelled e.g. \
             Am241_m1, not Am241m)",
            seeds.join(", ")
        )
        .into());
    }

    let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
    let reachable = yani::reachable_nuclides(chain, &seed_refs);

    // Collect (name, path) pairs while holding CONFIG lock, then drop it.
    // Skip nuclides whose chain entry has no neutron reactions: they are
    // decay-only sinks (e.g. fission products pulled in by yield edges)
    // and don't need cross-section data -- loading them just triggers
    // hundreds of wasted downloads (issue #45).
    let to_load: Vec<(String, String)> = {
        let cfg = yamc_nuclide::config::CONFIG
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        chain
            .iter()
            .filter(|(name, cn)| !cn.reactions.is_empty() && reachable.contains(*name))
            .map(|(name, _)| name)
            // Already loaded is normally enough, but an entry loaded
            // WITHOUT covariance cannot serve a request for it: skipping
            // here would leave the nuclide reported as having no
            // uncertainty data when the directory has some.
            .filter(|name| match material.nuclide_data.get(*name) {
                None => true,
                Some(nd) => want_covariance && !nd.load_scope.covariance,
            })
            .filter_map(|name| cfg.get_cross_section(name).map(|path| (name.clone(), path)))
            .collect()
    };

    // Nothing to load is the common case on every call after the first, so the
    // MT set and the scope are not built for it. The reachability closure above
    // cannot be skipped the same way: `Config::get_cross_section` falls back to
    // the library keyword, so it resolves a path for every name in the chain
    // and the candidate list is not empty without the closure to narrow it.
    if !to_load.is_empty() {
        // This path runs no transport, so it needs the union energy grid and the
        // cross sections of the MTs the network names, and nothing else (issue
        // #389). On the TENDL-2025 conversion that is roughly a fifth of the
        // per-nuclide data, and Fe56's nine full-grid transport MTs (total,
        // elastic, nonelastic, inelastic, absorption, disappearance, heating,
        // damage) are 1.36 MB each against 0.14 MB for a threshold reaction.
        // Self-shielding needs two MTs the network never names: the total,
        // which is what depresses the flux, and elastic, which is the
        // in-scattering that fills the dips again. They are the expensive
        // full-grid kind the comment above is about, so they are read only when
        // a chord was actually given and the correction is going to be applied.
        let mut wanted = activation_mts(chain, branch);
        if shielding.is_some() {
            wanted.insert(1);
            wanted.insert(2);
        }
        let scope = LoadScope::activation(wanted)
            .with_temperatures(temp_filter)
            .with_covariance(want_covariance);

        // One Arrow decode per nuclide, and they are independent: `to_load`'s
        // names are distinct, so no two tasks want the same file, and the
        // global cache's mutex is held only for the lookup and the insert,
        // never across the parse. Each `Nuclide` is a pure function of its
        // bytes and the scope, and `scope` is one value for every item, so the
        // results do not depend on the order they are produced in -- nor does
        // the map they land in, which is keyed by name (issue #576, finding 5a).
        let decoded: Vec<(String, std::sync::Arc<yamc_nuclide::Nuclide>)> = {
            let load_one = |(name, path): (String, String)| {
                let sources = HashMap::from([(name.clone(), path)]);
                get_or_load_nuclide(&name, &sources, &scope)
                    .ok()
                    .map(|nd| (name, nd))
            };
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                to_load.into_par_iter().filter_map(load_one).collect()
            }
            #[cfg(target_arch = "wasm32")]
            {
                to_load.into_iter().filter_map(load_one).collect()
            }
        };
        for (name, nd) in decoded {
            material.nuclide_data.insert(name, nd);
        }
    }

    // `to_load` skips nuclides already in `nuclide_data`, so `temp_filter`
    // never reaches them. A material whose data was loaded at one
    // temperature and then relabelled keeps the narrow reaction set, and
    // every rate lookup below resolves the label through
    // `reactions_for_temp`, which answers `None` and is handled by
    // `return 0.0` / `continue`. That is a silent zero: an irradiation
    // reporting no activation, with no error anywhere (#481).
    //
    // Widening goes through the material rather than the loop above because
    // that loop resolves paths from CONFIG, which never sees the explicit
    // paths handed to `read_nuclear_data`; `ensure_temperature_loaded` uses
    // each nuclide's own `data_path`.
    let label = material.temperature().to_string();
    if !label.is_empty() {
        material.ensure_temperature_loaded(&label)?;
    }

    // And the same for covariance, for the same reason: a nuclide already
    // in `nuclide_data` never reaches the loop above, and one loaded
    // through an explicit path is invisible to the CONFIG lookup that loop
    // uses. Without this an uncertainty run on a material built by
    // `read_nuclear_data` would report every nuclide as having no
    // covariance, which reads exactly like an evaluation that has none.
    if want_covariance {
        material.ensure_covariance_loaded()?;
    }

    Ok(())
}

/// One replica's inventories, one per schedule step.
///
/// The nominal loop above is left exactly as it was and this is a second,
/// lighter copy of it rather than a shared helper both call. That is deliberate:
/// the means have to stay bit-identical to the build before any of this existed,
/// and the surest way to keep them so is not to touch the loop that produces
/// them. This one also skips `per_edge_rates`, which a replica has no use for.
fn replica_steps(
    material: &Material,
    steps: &[TransmuteStep],
    per_spectrum: &[PerSpectrum],
    chain: &Arc<HashMap<String, ChainNuclide>>,
    parts: yani::ChainParts,
    stepper: &ForwardEulerStepper,
) -> Result<Vec<Material>, Box<dyn std::error::Error>> {
    let no_fission: FissionYieldWeights = HashMap::new();
    // `None` until the first step has produced a state to carry. It used to
    // start as `material.clone()`, which clones the initial composition
    // INCLUDING its ~556 loaded `Arc<Nuclide>`, once per replica, only so that
    // there is something owned to reassign on the first iteration. The stepper
    // takes a reference, so borrowing the caller's material for step one costs
    // nothing (issue #576, finding 6).
    let mut current: Option<Material> = None;
    let mut out = Vec::with_capacity(steps.len());
    for st in steps {
        let (step_rates, step_weights, step_chain) = match st.irradiation {
            Some((idx, rate)) => (
                scale_rates(&per_spectrum[idx].0, rate),
                &per_spectrum[idx].1,
                &per_spectrum[idx].2,
            ),
            None => (HashMap::new(), &no_fission, chain),
        };
        let stepped = stepper.step(
            current.as_ref().unwrap_or(material),
            step_chain,
            &step_rates,
            step_weights,
            parts,
            st.dt,
        )?;
        out.push(stepped.clone());
        current = Some(stepped);
    }
    Ok(out)
}

/// Fold the covariance, factorize it, and re-solve until the sigmas settle.
///
/// Replicas are added in blocks and convergence is judged between blocks on the
/// worst significant nuclide, so a problem dominated by one well-known channel
/// stops early and a genuinely noisy one keeps going. There is no user-facing
/// sample count on the default path: a knob that trades accuracy for time is a
/// knob that gets turned the wrong way.
#[allow(clippy::too_many_arguments)]
fn run_replicas(
    initial: &Material,
    spectra: &[MultigroupSpectrum],
    steps: &[TransmuteStep],
    per_spectrum: &[PerSpectrum],
    chain: &Arc<HashMap<String, ChainNuclide>>,
    parts: yani::ChainParts,
    stepper: &ForwardEulerStepper,
    request: &DataUncertainty,
    shielding: Option<&Shielding>,
) -> Result<(Ensemble, Info), Box<dyn std::error::Error>> {
    // One fold and one factorization per distinct spectrum, not per replica.
    // The fold is relativized, so the per-step `scale_rates` leaves it correct:
    // a relative covariance does not move when the flux magnitude does.
    let mut coverage = crate::covariance_fold::Coverage::default();
    let mut samplers = Vec::with_capacity(per_spectrum.len());
    let mut clipping = crate::covariance_sample::Clipping::default();

    // Switched off, this folds nothing and every sampler is empty, so the run
    // reports zero uncertainty with `sources` saying why. That is what makes
    // "add a source and watch sigma grow" measurable from one baseline.
    let cross_sections = request.wants(crate::uncertainty::Source::CrossSections);

    // One sampler per spectrum whether or not cross sections are on, so the
    // index stays the spectrum's own. Switched off, each is built from an empty
    // covariance map and perturbs nothing, which is cheaper than folding and
    // keeps every other source indexable by the same `idx`.
    for (idx, (rates, _, folded_chain)) in per_spectrum.iter().enumerate() {
        let folded = if cross_sections {
            let spectrum = &spectra[idx];
            let (folded, spectrum_coverage) = fold_rate_covariance(
                initial,
                folded_chain,
                rates,
                &spectrum.masses,
                &spectrum.boundaries,
            );
            merge_coverage(&mut coverage, spectrum_coverage);
            folded
        } else {
            std::collections::BTreeMap::new()
        };

        let sampler = Sampler::new(&folded);
        clipping.matrices_clipped += sampler.clipping.matrices_clipped;
        clipping.worst_relative_clip = clipping
            .worst_relative_clip
            .max(sampler.clipping.worst_relative_clip);
        samplers.push(sampler);
    }

    // The flux is the caller's own input, so it needs no nuclear data: only a
    // per-bin sigma, which a spectrum lifted from a published reference set
    // does not have. Where it is absent the spectrum contributes nothing and
    // the report says so.
    let want_flux = request.wants(crate::uncertainty::Source::FluxSpectrum);
    let mut flux_coverage = crate::flux_uncertainty::FluxCoverage::default();
    let mut per_group: Vec<Option<(Vec<f64>, crate::flux_uncertainty::PerGroupRates)>> =
        Vec::with_capacity(per_spectrum.len());
    for (idx, spectrum) in spectra.iter().enumerate() {
        match spectrum.relative_std_dev.as_ref().filter(|_| want_flux) {
            Some(relative) => {
                flux_coverage.spectra_with_sigma += 1;
                per_group.push(Some((
                    relative.clone(),
                    crate::multigroup::per_group_reaction_rates(
                        initial,
                        &per_spectrum[idx].2,
                        &spectrum.masses,
                        &spectrum.boundaries,
                        shielding,
                    ),
                )));
            }
            None => {
                if want_flux {
                    flux_coverage.spectra_without_sigma += 1;
                }
                per_group.push(None);
            }
        }
    }

    let mut info = Info::from_fold(&coverage, &clipping);
    info.sources = request
        .sources
        .iter()
        .map(|s| s.name().to_string())
        .collect();
    if info.sources.is_empty() {
        info.sources = crate::uncertainty::Source::IMPLEMENTED
            .iter()
            .map(|s| s.name().to_string())
            .collect();
    }
    let mut ensemble = Ensemble::new(steps.len());

    // Nothing to perturb means nothing to sample. The ensemble stays empty and
    // every sigma reads zero, with `info` saying why: no covariance data, not a
    // confident zero.
    if samplers.iter().all(Sampler::is_empty) && per_group.iter().all(Option::is_none) {
        info.converged = true;
        info.add_flux_coverage(&flux_coverage);
        return Ok((ensemble, info));
    }

    let target = request.samples;
    let cap = target.unwrap_or(MAX_SAMPLES);
    let mut previous_probe = std::collections::BTreeMap::new();
    let mut replica: u64 = 0;

    // One replica, start to finish, reading nothing that is not shared
    // read-only and writing nothing outside its own return value. That is what
    // lets a block of them run at once (issue #576, finding 4).
    //
    // Its own `FluxCoverage`, not the driver's: `Info::add_flux_coverage`
    // ASSIGNS `spectra_with_sigma` / `spectra_without_sigma`, which were
    // counted before the loop, so folding a replica's zeros in would erase
    // them. Only the two counters a replica actually produces come back.
    let one_replica = |replica: u64| -> Result<ReplicaOutcome, String> {
        let mut flux_coverage = crate::flux_uncertainty::FluxCoverage::default();
        let mut truncations = crate::covariance_sample::Truncations::default();

        // Every spectrum's rates are perturbed by the SAME replica index,
        // so a nuclide irradiated under two spectra in one schedule moves
        // together in both. Perturbing them independently would treat one
        // evaluation as two.
        let mut perturbed = Vec::with_capacity(per_spectrum.len());
        for (idx, (rates, weights, folded_chain)) in per_spectrum.iter().enumerate() {
            // The flux moves first, because its perturbation is defined
            // against the nominal per-group terms; the cross-section one is
            // multiplicative on the resulting rate, and the two sources are
            // independent so the order does not change the distribution.
            let rates = match &per_group[idx] {
                Some((relative, terms)) => {
                    let delta = crate::flux_uncertainty::flux_deviates(
                        relative,
                        request.seed,
                        replica,
                        idx,
                        &mut flux_coverage,
                    );
                    crate::flux_uncertainty::perturb_rates(rates, terms, &delta)
                }
                None => rates.clone(),
            };
            let (rates, t) = samplers[idx].perturb(&rates, request.seed, replica);
            truncations.floored += t.floored;
            truncations.sampled += t.sampled;
            perturbed.push((rates, weights.clone(), Arc::clone(folded_chain)));
        }

        // `Box<dyn Error>` is not `Send`, so it cannot come back out of a
        // parallel map; the driver puts the message back into one.
        let materials = replica_steps(initial, steps, &perturbed, chain, parts, stepper)
            .map_err(|e| e.to_string())?;
        Ok(ReplicaOutcome {
            densities: densities_of(&materials),
            truncations,
            flux_bins_sampled: flux_coverage.bins_sampled,
            flux_bins_floored: flux_coverage.bins_floored,
        })
    };

    while (replica as usize) < cap {
        let block_end = ((replica as usize) + BLOCK).min(cap);
        let block: Vec<u64> = (replica..block_end as u64).collect();

        // Collected into an indexed `Vec`, so the merge below sees the replicas
        // in replica order however they finished. That is required, not
        // cosmetic: `Ensemble::push` is a Welford recurrence, and the index it
        // pushes at IS the replica index `inventories_at` exposes.
        let outcomes: Vec<Result<ReplicaOutcome, String>> = {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                block.par_iter().map(|&r| one_replica(r)).collect()
            }
            #[cfg(target_arch = "wasm32")]
            {
                block.iter().map(|&r| one_replica(r)).collect()
            }
        };

        for outcome in outcomes {
            let outcome = outcome?;
            info.add_truncations(&outcome.truncations);
            flux_coverage.bins_sampled += outcome.flux_bins_sampled;
            flux_coverage.bins_floored += outcome.flux_bins_floored;
            ensemble.push(outcome.densities);
        }
        replica = block_end as u64;

        // A fixed sample count was asked for, so convergence is not this
        // driver's decision to make.
        if target.is_some() {
            continue;
        }

        let probe = ensemble.convergence_probe();
        if (replica as usize) >= MIN_SAMPLES && settled(&previous_probe, &probe) {
            info.converged = true;
            break;
        }
        previous_probe = probe;
    }

    // A nuclide the stepper dropped below its density floor in one replica is a
    // zero there, not a missing sample; without this its variance would be
    // taken over a different number of replicas than its neighbours'.
    ensemble.fold_absences();
    info.add_flux_coverage(&flux_coverage);
    info.samples = ensemble.replicas();
    if target.is_some() {
        info.converged = true;
    }
    Ok((ensemble, info))
}

/// Merge one spectrum's coverage into the run's.
///
/// Counts add and sets union, but a rate fraction is kept at its SMALLEST over
/// the spectra: a channel well covered under one spectrum and barely covered
/// under another is only as well covered as the worse of the two, and reporting
/// the better one would overstate what the evaluation actually says.
fn merge_coverage(
    into: &mut crate::covariance_fold::Coverage,
    from: crate::covariance_fold::Coverage,
) {
    // Every field merges the way `absorb` merges it, and delegating is what
    // keeps that true: this function and `absorb` used to carry two copies of
    // the same rules, and a field added to one of them was silently dropped by
    // the other.
    into.absorb(from);
    // The one rule that is this function's own. `absorb` runs per nuclide
    // within one spectrum, where "no data" is final. Across spectra it is not:
    // a nuclide whose covariance was unusable against one spectrum and usable
    // against another has data, so the two sets are resolved here rather than
    // accumulated blindly.
    let covered = into.covered.clone();
    into.without_data.retain(|n| !covered.contains(n));
}

/// Fold the isomeric-branching overlay against one spectrum shape.
///
/// Returns a chain to use for this spectrum. If the overlay contributes
/// nothing (no `branch`, or no matching parents), the original `chain` is
/// returned unchanged (no clone) so behaviour matches the plain three-part
/// chain exactly. Otherwise a per-spectrum clone is returned in which:
///
///   * grafted `(n,n')` metastable channels get their branching set to the
///     flux-weighted fraction, and `rates` is augmented with the total
///     `(n,n')` metastable-production rate (base, at unit flux; the caller
///     scales per step); and
///   * every other reaction that has branching curves has its ground/metastable
///     split replaced by the flux-weighted (energy-dependent) fractions,
///     leaving the transport total rate in `rates` untouched (only the split
///     changes, so a library total-rate difference cannot leak in).
pub(crate) fn fold_branching_into_chain(
    material: &Material,
    chain: &Arc<HashMap<String, ChainNuclide>>,
    branch: &BranchTable,
    spectrum: &MultigroupSpectrum,
    rates: &mut ReactionRates,
) -> Arc<HashMap<String, ChainNuclide>> {
    if branch.is_empty() {
        return Arc::clone(chain);
    }

    let mut refine: Fractions = HashMap::new();
    build_fold_refine(material, chain, branch, spectrum, rates, &mut refine);
    refine_chain(chain, &refine)
}

/// Build the fold's per-target fraction map into `refine`.
fn build_fold_refine(
    material: &Material,
    chain: &Arc<HashMap<String, ChainNuclide>>,
    branch: &BranchTable,
    spectrum: &MultigroupSpectrum,
    rates: &mut ReactionRates,
    refine: &mut Fractions,
) {
    for (parent, kinds) in branch.iter() {
        if !chain.contains_key(parent) {
            continue;
        }
        for (kind, curves) in kinds.iter() {
            if kind == "(n,n')" {
                // Self-inelastic: production of the metastable is its own MF=10
                // partial cross section (no transport total needed; the ground
                // self-loop is a no-op and omitted). The total (n,n') rate is
                // the sum of the metastable partial rates.
                let mut per_target: Vec<(String, f64)> = Vec::new();
                for c in curves {
                    if c.quantity != BranchQuantity::CrossSection || &c.target == parent {
                        continue;
                    }
                    let rate = fold_curve_rate(&c.energy, &c.values, spectrum);
                    if rate > 0.0 {
                        per_target.push((c.target.clone(), rate));
                    }
                }
                let total: f64 = per_target.iter().map(|(_, r)| r).sum();
                if total > 0.0 {
                    rates
                        .entry(parent.clone())
                        .or_default()
                        .insert("(n,n')".to_string(), total);
                    let m = refine
                        .entry(parent.clone())
                        .or_default()
                        .entry("(n,n')".to_string())
                        .or_default();
                    for (t, r) in per_target {
                        // Duplicate curves for the same target (real in TENDL,
                        // e.g. two LFS levels mapped to one chain nuclide) must
                        // sum, not overwrite.
                        *m.entry(t).or_insert(0.0) += r / total;
                    }
                }
            } else if let Some(fr) = fold_state_fractions(material, parent, kind, curves, spectrum)
            {
                let m = refine
                    .entry(parent.clone())
                    .or_default()
                    .entry(kind.clone())
                    .or_default();
                for (t, f) in fr {
                    *m.entry(t).or_insert(0.0) += f;
                }
            }
        }
    }
}

/// `parent -> reaction kind -> (target -> new branching fraction)`.
type Fractions = HashMap<String, HashMap<String, HashMap<String, f64>>>;

/// Rewrite the chain's branching splits from per-target fractions. Shared by
/// the multigroup fold (`fold_branching_into_chain`) and the coupled path
/// (`apply_coupled_branching`); returns the original chain untouched (no
/// clone) when there is nothing to refine.
fn refine_chain(
    chain: &Arc<HashMap<String, ChainNuclide>>,
    refine: &Fractions,
) -> Arc<HashMap<String, ChainNuclide>> {
    if refine.is_empty() {
        return Arc::clone(chain);
    }

    let mut folded = (**chain).clone();
    for (parent, kinds) in refine {
        let nuc = match folded.get_mut(parent) {
            Some(n) => n,
            None => continue,
        };
        for (kind, tmap) in kinds {
            if kind.as_str() == "(n,n')" {
                // Grafted self-inelastic channels: the folded fraction is the
                // share of the total (n,n') rate (base branchings are 1.0
                // placeholders), so assign it directly.
                for rx in nuc.reactions.iter_mut().filter(|r| &r.kind == kind) {
                    if let Some(t) = &rx.target {
                        if let Some(&f) = tmap.get(t) {
                            rx.branching = f;
                        }
                    }
                }
            } else {
                // Nuclide-changing reactions already carry a fixed split. Only
                // re-partition the probability mass currently on the targets
                // that have folded fractions, preserving that mass (so the
                // reaction's total production still equals its rate even when
                // the branch data covers only a subset of the chain targets).
                let mut base_sum = 0.0;
                let mut f_sum = 0.0;
                for rx in nuc.reactions.iter().filter(|r| &r.kind == kind) {
                    if let Some(t) = &rx.target {
                        if let Some(&f) = tmap.get(t) {
                            base_sum += rx.branching;
                            f_sum += f;
                        }
                    }
                }
                if base_sum > 0.0 && f_sum > 0.0 {
                    for rx in nuc.reactions.iter_mut().filter(|r| &r.kind == kind) {
                        if let Some(t) = &rx.target {
                            if let Some(&f) = tmap.get(t) {
                                rx.branching = (f / f_sum) * base_sum;
                            }
                        }
                    }
                }
            }
        }
    }

    Arc::new(folded)
}

/// Apply the isomeric-branching overlay on the coupled path (issue #218).
///
/// All fractions come from continuous-energy rates: MF=10 partials are folded
/// exactly from the tally's union-grid flux moments (every chain parent,
/// including products that build up during the step) and MF=9 yields are
/// scored directly at the collision energies (`y_s(E) * sigma_MT(E) * TL`,
/// material nuclides), so no group approximation enters any split. Semantics
/// per kind mirror `fold_branching_into_chain`:
///
///   * `(n,n')`: the summed metastable partial rate is injected into `rates`
///     (there is no transport total for MT 4) and the grafted channels get the
///     rate shares; zero-rate targets are dropped.
///   * other kinds: the per-target rates are normalized to fractions and the
///     base branching mass on the covered targets is re-partitioned; when all
///     rates are zero (e.g. flux entirely below threshold) the base split is
///     kept, matching the fold's `None` behaviour.
pub fn apply_coupled_branching(
    chain: &Arc<HashMap<String, ChainNuclide>>,
    partial_rates: &PartialRates,
    rates: &mut ReactionRates,
) -> Arc<HashMap<String, ChainNuclide>> {
    let mut refine: Fractions = HashMap::new();
    for (parent, kinds) in partial_rates {
        if !chain.contains_key(parent) {
            continue;
        }
        for (kind, per_target) in kinds {
            if kind == "(n,n')" {
                let positive: Vec<(&String, f64)> = per_target
                    .iter()
                    .filter(|(_, r)| *r > 0.0)
                    .map(|(t, r)| (t, *r))
                    .collect();
                let total: f64 = positive.iter().map(|(_, r)| r).sum();
                if total > 0.0 {
                    rates
                        .entry(parent.clone())
                        .or_default()
                        .insert("(n,n')".to_string(), total);
                    let m = refine
                        .entry(parent.clone())
                        .or_default()
                        .entry("(n,n')".to_string())
                        .or_default();
                    for (t, r) in positive {
                        // Duplicate curves for the same target sum, not
                        // overwrite (real in TENDL branching data).
                        *m.entry(t.clone()).or_insert(0.0) += r / total;
                    }
                }
            } else {
                let total: f64 = per_target.iter().map(|(_, r)| r).sum();
                if total > 0.0 {
                    let m = refine
                        .entry(parent.clone())
                        .or_default()
                        .entry(kind.clone())
                        .or_default();
                    for (t, r) in per_target {
                        *m.entry(t.clone()).or_insert(0.0) += r / total;
                    }
                }
            }
        }
    }

    refine_chain(chain, &refine)
}

/// Flux-weighted reaction rate (base, unit flux) of a partial cross-section
/// curve: `(sum_g groupavg_g * mass_g) * 1e-24`, matching the convention in
/// `compute_multigroup_reaction_rates`.
fn fold_curve_rate(energy: &[f64], values: &[f64], spectrum: &MultigroupSpectrum) -> f64 {
    let mut sigma_phi = 0.0;
    for g in 0..spectrum.masses.len() {
        let sigma_g = curve_group_average(
            energy,
            values,
            spectrum.boundaries[g],
            spectrum.boundaries[g + 1],
        );
        sigma_phi += sigma_g * spectrum.masses[g];
    }
    sigma_phi * 1.0e-24
}

/// Flux-weighted branching fractions across the final states of a
/// nuclide-changing reaction, normalized to sum to 1.
///
/// Cross-section curves (MF=10 partials) are rate-weighted directly:
/// `f_s = integral(sigma_s * phi) / sum_s' integral(sigma_s' * phi)`, which
/// needs no cross-section data. Yield curves (MF=9 fractions) are weighted by the
/// reaction's transport cross section `sigma_MT(E)` so the fraction is a proper
/// reaction-rate average rather than a bare flux average; returns `None` (keep
/// the base fixed split) when that transport cross section is unavailable.
fn fold_state_fractions(
    material: &Material,
    parent: &str,
    kind: &str,
    curves: &[yani::BranchCurve],
    spectrum: &MultigroupSpectrum,
) -> Option<Vec<(String, f64)>> {
    let cross: Vec<&yani::BranchCurve> = curves
        .iter()
        .filter(|c| c.quantity == BranchQuantity::CrossSection)
        .collect();
    let n_groups = spectrum.masses.len();

    let mut per: Vec<(String, f64)> = Vec::new();
    if !cross.is_empty() {
        for c in &cross {
            per.push((
                c.target.clone(),
                fold_curve_rate(&c.energy, &c.values, spectrum),
            ));
        }
    } else {
        // Yield curves (MF=9): weight by the reaction's transport cross section
        // at sub-group (continuous-energy) resolution. Within each flux group
        // integrate `y(E) * sigma_MT(E)` over the union of the yield and
        // cross-section energy grids, rather than `<y>_g * <sigma>_g` (a product
        // of separate group averages). The product form preserves the
        // within-group correlation between the yield and the cross section that
        // `<y>_g * <sigma>_g` drops; it matters when a flux group is coarse
        // relative to structure in `y` or `sigma` (e.g. a broad thermal group
        // over a 1/v capture whose isomeric yield varies across the group).
        use crate::reaction_type_to_mt;
        let mt = reaction_type_to_mt(kind)?;
        let nd = material.nuclide_data.get(parent)?;
        let temperature = if material.temperature().is_empty() {
            crate::default_temperature(nd)?
        } else {
            material.temperature().to_string()
        };
        let reactions = nd.reactions_for_temp(&temperature)?;
        let reaction = reactions.get(&mt)?;
        for c in curves {
            if c.quantity != BranchQuantity::Yield {
                continue;
            }
            let mut num = 0.0;
            for g in 0..n_groups {
                let prod = curve_xs_product_group_average(
                    &c.energy,
                    &c.values,
                    &reaction.energy,
                    |e| reaction.cross_section_at(e).unwrap_or(0.0),
                    spectrum.boundaries[g],
                    spectrum.boundaries[g + 1],
                );
                num += prod * spectrum.masses[g];
            }
            per.push((c.target.clone(), num));
        }
    }

    let total: f64 = per.iter().map(|(_, v)| v).sum();
    if total <= 0.0 {
        return None;
    }
    Some(per.into_iter().map(|(t, v)| (t, v / total)).collect())
}

/// Group-average the product `curve(E) * sigma(E)` over `[e_lo, e_hi]`.
///
/// Integrates on the union of the branching-curve grid (`energy`) and the
/// cross-section grid (`xs_energy`) within the group, so the within-group
/// correlation between the two is preserved (unlike `<curve>_g * <sigma>_g`).
/// `sigma_at` evaluates the cross section at an arbitrary energy. The curve is
/// linearly interpolated (zero below its first grid point, flat above its last)
/// by `curve_interp`.
fn curve_xs_product_group_average(
    energy: &[f64],
    values: &[f64],
    xs_energy: &[f64],
    sigma_at: impl Fn(f64) -> f64,
    e_lo: f64,
    e_hi: f64,
) -> f64 {
    if e_lo >= e_hi || energy.is_empty() {
        return 0.0;
    }

    // On each union segment both `curve` and `sigma` are linear, so their
    // product is quadratic; integrate it exactly (a product of two linears)
    // rather than trapezoidally, which would miss a mid-segment peak.
    //
    // The union used to be built by scanning both whole grids per group and
    // sorting; both ascend, so it is a merge of two bisected slices instead
    // (issue #576, finding 1a, applied to the integrator the branching fold
    // has of its own). Duplicates survive the merge where `dedup` removed
    // them, and each one contributes a segment of exactly zero width, whose
    // term is `0.0 * (...)`: exactly `+0.0` into a finite accumulator that
    // starts at `+0.0`.
    let mut integral = 0.0;
    let mut prev_e = e_lo;
    let (mut prev_y, mut prev_s) = (curve_interp(energy, values, e_lo), sigma_at(e_lo));
    for e in crate::multigroup::group_points_after_merged(energy, xs_energy, e_lo, e_hi) {
        let (y, s) = (curve_interp(energy, values, e), sigma_at(e));
        integral +=
            (e - prev_e) * (2.0 * prev_y * prev_s + prev_y * s + y * prev_s + 2.0 * y * s) / 6.0;
        prev_e = e;
        prev_y = y;
        prev_s = s;
    }
    integral / (e_hi - e_lo)
}

/// Group-average a piecewise-linear curve `(energy, values)` over `[e_lo, e_hi]`.
///
/// Mirrors `multigroup::group_averaged_xs` but for an arbitrary branching curve:
/// integrates the linear interpolant (zero below the first grid point, flat
/// above the last) and divides by the group width.
fn curve_group_average(energy: &[f64], values: &[f64], e_lo: f64, e_hi: f64) -> f64 {
    if e_lo >= e_hi || energy.is_empty() {
        return 0.0;
    }

    // The interior points, bisected rather than scanned for (issue #576,
    // finding 1a). This one never sorted or deduplicated, so the sequence is
    // not merely equivalent to what the scan produced, it is the same
    // sequence: `interior` returns the ascending grid's in-range sub-slice, in
    // order, duplicates and all.
    let mut integral = 0.0;
    let mut prev_e = e_lo;
    let mut prev_v = curve_interp(energy, values, e_lo);
    for e in crate::multigroup::group_points_after(energy, e_lo, e_hi) {
        let v = curve_interp(energy, values, e);
        integral += 0.5 * (prev_v + v) * (e - prev_e);
        prev_e = e;
        prev_v = v;
    }
    integral / (e_hi - e_lo)
}

/// Linear interpolation of `(energy, values)` at `e`: zero below the first grid
/// point (threshold), flat above the last. Shared with the transmutation tally,
/// which evaluates branching partials at the collision energy (issue #218).
pub(crate) fn curve_interp(energy: &[f64], values: &[f64], e: f64) -> f64 {
    if e <= energy[0] {
        // At/just below threshold the value is the first point only if e ==
        // energy[0]; below the grid the curve is zero.
        return if e < energy[0] { 0.0 } else { values[0] };
    }
    let last = energy.len() - 1;
    if e >= energy[last] {
        return values[last];
    }
    // Binary search for the bracketing interval.
    let idx = match energy.binary_search_by(|x| x.partial_cmp(&e).unwrap()) {
        Ok(i) => return values[i],
        Err(i) => i, // energy[i-1] < e < energy[i]
    };
    let (e0, e1) = (energy[idx - 1], energy[idx]);
    let (v0, v1) = (values[idx - 1], values[idx]);
    if e1 == e0 {
        return v0;
    }
    v0 + (v1 - v0) * (e - e0) / (e1 - e0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use yamc_materials::material::Material;
    use yani::{BranchCurve, BranchQuantity, ChainNuclide, ChainReaction};

    /// A minimal material (no cross-section data); enough for the (n,n') path,
    /// which uses the branching partials directly rather than cross-section data.
    fn dummy_material() -> Material {
        Material::new(
            HashMap::from([("Pb204".to_string(), 1.0e-3)]),
            "atom",
            "sum",
            None,
        )
        .unwrap()
    }

    #[test]
    fn curve_interp_threshold_and_flat() {
        let e = [1.0e6, 2.0e6, 3.0e6];
        let v = [0.0, 4.0, 8.0];
        assert_eq!(curve_interp(&e, &v, 5.0e5), 0.0); // below threshold
        assert_eq!(curve_interp(&e, &v, 1.0e6), 0.0); // at first point
        assert_eq!(curve_interp(&e, &v, 1.5e6), 2.0); // linear midpoint
        assert_eq!(curve_interp(&e, &v, 3.0e6), 8.0); // last point
        assert_eq!(curve_interp(&e, &v, 9.0e6), 8.0); // flat above range
    }

    #[test]
    fn curve_group_average_flat_curve() {
        // Constant 5.0 barns over the whole grid -> group average 5.0.
        let e = [1.0, 1.0e8];
        let v = [5.0, 5.0];
        let avg = curve_group_average(&e, &v, 100.0, 1000.0);
        assert!((avg - 5.0).abs() < 1e-9, "got {avg}");
    }

    #[test]
    fn product_group_average_captures_within_group_correlation() {
        // Over the group [1,3], yield decreases 1->0 while the cross section
        // increases 0->1 (anti-correlated). <y>=0.5 and <sigma>=0.5, so the
        // product-of-averages is 0.25; but the true group average of y*sigma is
        // 1/6 ~= 0.1667 (the product peaks mid-group). The exact bilinear
        // integration must return 1/6 -- not 0.25 (product of averages) and not
        // 0 (endpoints-only trapezoid, since y*sigma is 0 at both ends).
        let y_e = [1.0, 3.0];
        let y_v = [1.0, 0.0];
        let avg =
            curve_xs_product_group_average(&y_e, &y_v, &[1.0, 3.0], |e| (e - 1.0) / 2.0, 1.0, 3.0);
        assert!((avg - 1.0 / 6.0).abs() < 1e-9, "expected 1/6, got {avg}");
        assert!((avg - 0.25).abs() > 0.05, "must not equal <y>*<sigma>=0.25");
    }

    /// A single-parent chain with a grafted `(n,n')` metastable channel and a
    /// flat 0.1 barn partial cross section folds to the expected rate, and the
    /// grafted channel's branching becomes 1.0 (single metastable target).
    #[test]
    fn fold_nnprime_injects_rate_and_branching() {
        let mut map: HashMap<String, ChainNuclide> = HashMap::new();
        map.insert(
            "Pb204".to_string(),
            ChainNuclide {
                name: "Pb204".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "(n,n')".to_string(),
                    target: Some("Pb204_m1".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                decays: vec![],
                fission_yields: None,
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        let chain = Arc::new(map);

        let mut branch: BranchTable = BranchTable::new();
        branch.entry("Pb204".to_string()).or_default().insert(
            "(n,n')".to_string(),
            vec![BranchCurve {
                target: "Pb204_m1".to_string(),
                quantity: BranchQuantity::CrossSection,
                energy: vec![1.0, 1.0e8],
                values: vec![0.1, 0.1], // flat 0.1 barn
            }],
        );

        // Single group spanning the whole grid, unit mass -> sigma_phi = 0.1.
        let spectrum = MultigroupSpectrum {
            boundaries: vec![1.0, 1.0e8],
            masses: vec![1.0],
            relative_std_dev: None,
        };
        let mut rates: ReactionRates = HashMap::new();
        let folded =
            fold_branching_into_chain(&dummy_material(), &chain, &branch, &spectrum, &mut rates);

        let rate = rates
            .get("Pb204")
            .and_then(|m| m.get("(n,n')"))
            .copied()
            .expect("(n,n') rate injected");
        assert!((rate - 0.1 * 1.0e-24).abs() < 1e-40, "got {rate}");

        let br = folded["Pb204"].reactions[0].branching;
        assert!(
            (br - 1.0).abs() < 1e-12,
            "single metastable -> branching 1.0, got {br}"
        );
    }

    /// Cross-section (MF=10) partials normalize to rate-weighted fractions
    /// across final states (no cross-section data needed for this path).
    #[test]
    fn fold_state_fractions_cross_section_normalizes() {
        let curves = vec![
            BranchCurve {
                target: "X_g".to_string(),
                quantity: BranchQuantity::CrossSection,
                energy: vec![1.0, 1.0e8],
                values: vec![3.0, 3.0],
            },
            BranchCurve {
                target: "X_m1".to_string(),
                quantity: BranchQuantity::CrossSection,
                energy: vec![1.0, 1.0e8],
                values: vec![1.0, 1.0],
            },
        ];
        let spectrum = MultigroupSpectrum {
            boundaries: vec![1.0, 1.0e8],
            masses: vec![1.0],
            relative_std_dev: None,
        };
        let fr =
            fold_state_fractions(&dummy_material(), "X", "(n,2n)", &curves, &spectrum).unwrap();
        let get = |t: &str| fr.iter().find(|(x, _)| x == t).unwrap().1;
        assert!((get("X_g") - 0.75).abs() < 1e-9, "{:?}", fr);
        assert!((get("X_m1") - 0.25).abs() < 1e-9, "{:?}", fr);
    }

    /// With no branching overlay the original chain Arc is returned unchanged.
    #[test]
    fn fold_no_overlay_is_identity() {
        let map: HashMap<String, ChainNuclide> = HashMap::new();
        let chain = Arc::new(map);
        let branch: BranchTable = BranchTable::new();
        let spectrum = MultigroupSpectrum {
            boundaries: vec![1.0, 1.0e8],
            masses: vec![1.0],
            relative_std_dev: None,
        };
        let mut rates: ReactionRates = HashMap::new();
        let folded =
            fold_branching_into_chain(&dummy_material(), &chain, &branch, &spectrum, &mut rates);
        assert!(Arc::ptr_eq(&chain, &folded), "empty overlay must not clone");
        assert!(rates.is_empty());
    }

    /// A chain with a two-target `(n,2n)` split for direct-rate tests
    /// (issue #218): base branching 0.7 ground / 0.3 metastable.
    fn split_chain() -> Arc<HashMap<String, ChainNuclide>> {
        let mut map: HashMap<String, ChainNuclide> = HashMap::new();
        map.insert(
            "X".to_string(),
            ChainNuclide {
                name: "X".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![
                    ChainReaction {
                        kind: "(n,2n)".to_string(),
                        target: Some("X_g".to_string()),
                        branching: 0.7,
                        q_value: None,
                    },
                    ChainReaction {
                        kind: "(n,2n)".to_string(),
                        target: Some("X_m1".to_string()),
                        branching: 0.3,
                        q_value: None,
                    },
                ],
                decays: vec![],
                fission_yields: None,
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        Arc::new(map)
    }

    /// Directly-scored `(n,n')` partial rates inject the summed rate and set
    /// the grafted branching, mirroring the fold's semantics (issue #218).
    #[test]
    fn apply_partials_nnprime_injects_rate_and_branching() {
        let mut map: HashMap<String, ChainNuclide> = HashMap::new();
        map.insert(
            "Pb204".to_string(),
            ChainNuclide {
                name: "Pb204".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "(n,n')".to_string(),
                    target: Some("Pb204_m1".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                decays: vec![],
                fission_yields: None,
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        let chain = Arc::new(map);

        let mut partials: PartialRates = HashMap::new();
        partials.entry("Pb204".to_string()).or_default().insert(
            "(n,n')".to_string(),
            vec![("Pb204_m1".to_string(), 1.0e-25)],
        );

        let mut rates: ReactionRates = HashMap::new();
        let folded = apply_coupled_branching(&chain, &partials, &mut rates);

        let rate = rates
            .get("Pb204")
            .and_then(|m| m.get("(n,n')"))
            .copied()
            .expect("(n,n') rate injected");
        assert!((rate - 1.0e-25).abs() < 1e-40, "got {rate}");
        let br = folded["Pb204"].reactions[0].branching;
        assert!((br - 1.0).abs() < 1e-12, "got {br}");
    }

    /// Per-final-state rates re-partition the base branching mass by their
    /// rate shares: 3:1 rates on a 0.7/0.3 base split become 0.75/0.25.
    #[test]
    fn apply_partials_repartitions_split() {
        let chain = split_chain();
        let mut partials: PartialRates = HashMap::new();
        partials.entry("X".to_string()).or_default().insert(
            "(n,2n)".to_string(),
            vec![("X_g".to_string(), 3.0e-24), ("X_m1".to_string(), 1.0e-24)],
        );

        let mut rates: ReactionRates = HashMap::new();
        let folded = apply_coupled_branching(&chain, &partials, &mut rates);
        assert!(rates.is_empty(), "non-(n,n') kinds must not inject rates");

        let get = |t: &str| {
            folded["X"]
                .reactions
                .iter()
                .find(|r| r.target.as_deref() == Some(t))
                .unwrap()
                .branching
        };
        assert!((get("X_g") - 0.75).abs() < 1e-12);
        assert!((get("X_m1") - 0.25).abs() < 1e-12);
    }

    /// All-zero partial rates (flux never reached the thresholds) keep the
    /// base split, matching the fold's `None` behaviour; nothing is cloned.
    #[test]
    fn apply_partials_zero_total_keeps_base_split() {
        let chain = split_chain();
        let mut partials: PartialRates = HashMap::new();
        partials.entry("X".to_string()).or_default().insert(
            "(n,2n)".to_string(),
            vec![("X_g".to_string(), 0.0), ("X_m1".to_string(), 0.0)],
        );
        let mut rates: ReactionRates = HashMap::new();
        let folded = apply_coupled_branching(&chain, &partials, &mut rates);
        assert!(Arc::ptr_eq(&chain, &folded), "zero rates must not refine");
        assert!(rates.is_empty());
    }

    /// Duplicate curves for the same (kind, target), which really occur in
    /// TENDL branching data (two LFS levels mapped to one chain nuclide), must
    /// have their rates summed into the fraction, not overwrite each other.
    #[test]
    fn apply_partials_sums_duplicate_targets() {
        let chain = split_chain();
        let mut partials: PartialRates = HashMap::new();
        partials.entry("X".to_string()).or_default().insert(
            "(n,2n)".to_string(),
            vec![
                ("X_g".to_string(), 1.0e-24),
                ("X_g".to_string(), 2.0e-24), // duplicate target
                ("X_m1".to_string(), 1.0e-24),
            ],
        );
        let mut rates: ReactionRates = HashMap::new();
        let folded = apply_coupled_branching(&chain, &partials, &mut rates);
        let get = |t: &str| {
            folded["X"]
                .reactions
                .iter()
                .find(|r| r.target.as_deref() == Some(t))
                .unwrap()
                .branching
        };
        // Ground share = (1 + 2) / 4 of the base mass 1.0.
        assert!((get("X_g") - 0.75).abs() < 1e-12, "got {}", get("X_g"));
        assert!((get("X_m1") - 0.25).abs() < 1e-12, "got {}", get("X_m1"));
    }

    /// Empty partial rates return the original chain Arc untouched.
    #[test]
    fn apply_partials_empty_is_identity() {
        let chain = split_chain();
        let partials: PartialRates = HashMap::new();
        let mut rates: ReactionRates = HashMap::new();
        let folded = apply_coupled_branching(&chain, &partials, &mut rates);
        assert!(Arc::ptr_eq(&chain, &folded));
        assert!(rates.is_empty());
    }
}
