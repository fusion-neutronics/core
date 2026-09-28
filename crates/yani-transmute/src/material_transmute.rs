use crate::branching_rule::{
    build_lists, measure_unmodelled_mt5, removal_rate, Bound, BranchingChannel, BranchingReport,
    BranchingState, Denominator, DroppedChannel, ListRates, ListRule, Lists,
    BRANCHING_RATE_TOLERANCE, INELASTIC, MT_ANYTHING, MT_INELASTIC,
};
use crate::covariance_fold::fold_rate_covariance;
use crate::covariance_sample::Sampler;
use crate::multigroup::{collapse_with_lists, fold_list, scale_rates, Collapsed};
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
use yani::{per_edge_rates, BranchTable, ChainNuclide, FissionYieldWeights, ReactionRates};

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
    /// The flux's stated error, when the caller knows it (issues #559, #140).
    ///
    /// Relative rather than absolute, as a per-group standard deviation or a
    /// full covariance, because `masses` is a normalized shape and the pulse
    /// `rate` carries the magnitude, so an absolute error would be in units
    /// this struct no longer has. A relative one is invariant under that split
    /// and is what the perturbation multiplies by.
    ///
    /// `None` is the common case and is NOT zero. A spectrum taken from a
    /// published reference set has no stated error, and that has to stay
    /// distinguishable from one measured to be exact.
    pub flux_error: Option<crate::flux_uncertainty::FluxError>,
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
/// Three sources, all needed. The chain's own reaction kinds give the transport
/// cross sections the rate collapse folds against. The branching overlay adds
/// the totals its lists split (see [`crate::branching_rule`]), MT=4 among them
/// for `(n,n')`, which has no chain total of its own. And MT=5, whose products
/// the chain does not model, is read so its share of each parent's removal
/// can be reported (see [`crate::branching_rule::measure_unmodelled_mt5`]).
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
            if kind == INELASTIC {
                mts.insert(MT_INELASTIC);
            } else if let Some(mt) = reaction_type_to_mt(kind) {
                mts.insert(mt);
            }
        }
    }
    if !mts.is_empty() {
        mts.insert(MT_ANYTHING);
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
    transmute_materials(
        vec![TransmuteCase {
            material,
            spectra: spectra.to_vec(),
            steps: steps.to_vec(),
            shielding: shielding.copied(),
        }],
        chain,
        branch,
        parts,
        uncertainty,
    )
}

/// One material's part of a [`transmute_materials`] solve: the material, the
/// spectra its irradiation steps reference, its timeline, and whether its
/// collapse is self-shielded.
pub struct TransmuteCase<'a> {
    /// Loaded with the cross sections it needs, as [`transmute_material`] does.
    pub material: &'a mut Material,
    /// Distinct multigroup spectra this material's steps reference.
    pub spectra: Vec<MultigroupSpectrum>,
    /// The timeline. Every case must share the step durations and which steps
    /// irradiate; the rates and spectra are the case's own.
    pub steps: Vec<TransmuteStep>,
    /// Self-shielding for this material's collapse, `None` for dilute.
    pub shielding: Option<Shielding>,
}

/// Transmute several materials over one timeline in one call.
///
/// Each material gets its own spectra and flux magnitudes, which is what a mesh
/// from a transport run needs, and they share the timeline, so the result has
/// one series of times and is keyed by each material's `material_id` exactly as
/// the transport-coupled path's is.
///
/// Work that is a property of the network rather than of one material is done
/// once: the chain is shared, a nuclide's cross sections are decoded once and
/// shared by every material that needs them, and a multigroup collapse is run
/// once for every distinct combination of spectrum, composition, temperature,
/// loaded data and shielding. Materials that repeat a combination, cells of
/// one steel that saw the same spectrum, reuse it, with identical results
/// including the reports. [`TransmutationResults::collapse_reuse`] says how
/// many were shared. The per-material step solves then run in parallel.
///
/// Errors if two materials share an id, since the result could not tell them
/// apart, or if the timelines differ. A single material with no id answers to
/// 0, as [`transmute_material`] always has.
pub fn transmute_materials(
    mut cases: Vec<TransmuteCase<'_>>,
    chain: Arc<HashMap<String, yani::ChainNuclide>>,
    branch: &BranchTable,
    parts: yani::ChainParts,
    uncertainty: Option<&DataUncertainty>,
) -> Result<TransmutationResults, Box<dyn std::error::Error>> {
    if cases.is_empty() {
        return Err("no materials to transmute".into());
    }
    let plural = cases.len() > 1;
    // Errors name the material only when there is more than one to confuse.
    let named = |c: usize, id: u32, e: Box<dyn std::error::Error>| -> Box<dyn std::error::Error> {
        if plural {
            format!("material {id} (position {c}): {e}").into()
        } else {
            e
        }
    };

    let ids: Vec<u32> = cases
        .iter()
        .map(|case| case.material.material_id.unwrap_or(0))
        .collect();
    if plural {
        let mut seen: HashMap<u32, usize> = HashMap::new();
        for (c, &id) in ids.iter().enumerate() {
            if let Some(first) = seen.insert(id, c) {
                return Err(format!(
                    "materials at positions {first} and {c} both have id {id}; the results \
                     are keyed by material id, so give each material a distinct one"
                )
                .into());
            }
        }
    }

    for (c, case) in cases.iter().enumerate() {
        validate_case(&case.spectra, &case.steps).map_err(|e| named(c, ids[c], e))?;
    }
    for (c, case) in cases.iter().enumerate().skip(1) {
        check_same_timeline(&cases[0].steps, &case.steps)
            .map_err(|e| format!("material {} (position {c}): {e}", ids[c]))?;
    }

    // Per material: the chain must drive something, the data is loaded into
    // the caller's material, and the solve starts from its sum-mode copy.
    let mut initial: Vec<Material> = Vec::with_capacity(cases.len());
    for (c, case) in cases.iter_mut().enumerate() {
        check_chain_drives(case.material, &case.steps, &chain).map_err(|e| named(c, ids[c], e))?;
        preload_activation_data(
            case.material,
            &chain,
            branch,
            uncertainty,
            case.shielding.as_ref(),
        )
        .map_err(|e| named(c, ids[c], e))?;
        let mut current = case.material.clone();
        if current.density_units != DensityUnits::Sum {
            current = current
                .to_sum_mode()
                .map_err(|e| named(c, ids[c], e.into()))?;
        }
        initial.push(current);
    }

    // Collapse once per distinct combination, at unit total flux (masses are
    // normalized, so this is the rate per n/cm^2/s). Each step then scales its
    // spectrum's rates by its `rate`.
    //
    // When an isomeric-branching overlay is supplied, fold it against the same
    // spectrum shape here (rate-compute time): this injects the `(n,n')`
    // metastable-production rates and yields a per-spectrum chain whose
    // isomeric branching ratios are flux-weighted rather than fixed. Branching
    // fractions are magnitude-independent, so the per-step `scale_rates` (which
    // scales only the rates) leaves them correct.
    //
    // The report is merged across a material's spectra rather than assigned,
    // so a schedule naming two spectra reports both spectra's nuclides (issue
    // #576).
    let lists = build_lists(&chain, branch)?;
    let mut shared_spectra: Vec<MultigroupSpectrum> = Vec::new();
    let mut per_spectrum: Vec<PerSpectrum> = Vec::new();
    let mut shared_info: Vec<ShieldingInfo> = Vec::new();
    let mut shared_reports: Vec<Arc<BranchingReport>> = Vec::new();
    let mut known: HashMap<CollapseKey, usize> = HashMap::new();
    let mut requested = 0usize;
    let mut case_steps: Vec<Vec<TransmuteStep>> = Vec::with_capacity(cases.len());
    let mut case_shared: Vec<Vec<usize>> = Vec::with_capacity(cases.len());
    let mut case_info: Vec<ShieldingInfo> = Vec::with_capacity(cases.len());
    for (c, case) in cases.iter().enumerate() {
        let current = &initial[c];
        let shielding = case.shielding.as_ref();
        let mut to_shared = Vec::with_capacity(case.spectra.len());
        let mut info = ShieldingInfo::default();
        for s in &case.spectra {
            requested += 1;
            let key = CollapseKey::new(current, s, shielding);
            let g = match known.get(&key) {
                Some(&g) => g,
                None => {
                    let (entry, entry_info, entry_report) =
                        collapse_one(current, s, &chain, &lists, shielding)
                            .map_err(|e| named(c, ids[c], e))?;
                    per_spectrum.push(entry);
                    shared_info.push(entry_info);
                    shared_reports.push(Arc::new(entry_report));
                    shared_spectra.push(s.clone());
                    known.insert(key, per_spectrum.len() - 1);
                    per_spectrum.len() - 1
                }
            };
            info.merge(shared_info[g].clone());
            to_shared.push(g);
        }
        case_steps.push(
            case.steps
                .iter()
                .map(|st| TransmuteStep {
                    dt: st.dt,
                    irradiation: st.irradiation.map(|(idx, rate)| (to_shared[idx], rate)),
                })
                .collect(),
        );
        case_info.push(info);
        case_shared.push(to_shared);
    }

    // The step solves are independent, one per material.
    // The nominal steps index the shared collapses. The replicas fold and
    // factorize every spectrum they are handed and count its coverage, so
    // they get this material's own spectra, in its own indexing.
    let solve = |c: usize| {
        let replicas = uncertainty.map(|request| Replicas {
            spectra: &cases[c].spectra,
            steps: &cases[c].steps,
            per_spectrum: case_shared[c]
                .iter()
                .map(|&g| per_spectrum[g].clone())
                .collect(),
            request,
            shielding: cases[c].shielding.as_ref(),
        });
        solve_case(
            &initial[c],
            &case_steps[c],
            &per_spectrum,
            &chain,
            parts,
            replicas,
        )
        .map_err(|e| {
            let e: Box<dyn std::error::Error> = named(c, ids[c], e);
            e.to_string()
        })
    };
    let outcomes: Vec<Result<CaseOutcome, String>> = {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            if plural {
                (0..cases.len()).into_par_iter().map(solve).collect()
            } else {
                (0..cases.len()).map(solve).collect()
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            (0..cases.len()).map(solve).collect()
        }
    };

    let mut results = TransmutationResults::new(cases[0].steps.iter().map(|st| st.dt).collect());
    for (c, outcome) in outcomes.into_iter().enumerate() {
        let outcome = outcome?;
        let id = ids[c];
        let case = &cases[c];
        results.add_initial(
            id,
            initial[c].clone(),
            case.steps
                .iter()
                .map(|st| st.irradiation.map_or(0.0, |(_, rate)| rate))
                .collect(),
        );
        for (state, rates) in outcome.states.into_iter().zip(outcome.rates) {
            results.add_step_rates(id, rates);
            results.add_step(id, state);
        }
        // One report per step, the one its spectrum's fold produced; a
        // decay-only step splits nothing and reports nothing.
        results.branching_report.insert(
            id,
            case_steps[c]
                .iter()
                .map(|st| match st.irradiation {
                    Some((g, _)) => Arc::clone(&shared_reports[g]),
                    None => Arc::new(BranchingReport::default()),
                })
                .collect(),
        );
        if let Some((ensemble, info)) = outcome.uncertainty {
            results.uncertainty.insert(id, ensemble);
            results.uncertainty_info.insert(id, info);
        }
        // Recorded whether or not shielding ran: "not shielded" and "shielded
        // and nothing moved" are different claims, and a dilute run that
        // should have been shielded is the case #564 is about.
        results.shielding_info.insert(id, case_info[c].clone());
        // The spectra it collapsed against, for re-deriving an energy-resolved
        // view of a rate afterwards: a few KB of stored spectrum rather than
        // the tens of MB the whole breakdown would be (yani#27).
        results.collapse.insert(
            id,
            crate::results::CollapseInputs {
                spectra: case
                    .spectra
                    .iter()
                    .map(|s| (s.boundaries.clone(), s.masses.clone()))
                    .collect(),
                step_spectrum: case
                    .steps
                    .iter()
                    .map(|st| st.irradiation.map(|(idx, _)| idx))
                    .collect(),
                shielding: case.shielding,
            },
        );
    }
    results.collapse_reuse = Some(crate::results::CollapseReuse {
        performed: per_spectrum.len(),
        requested,
    });
    // Kept so routes can be derived from the same topology the solve used,
    // rather than from whatever a second load of the chain path returns.
    results.chain = Some(Arc::clone(&chain));
    Ok(results)
}

/// The spectra and steps of one material, checked before anything is loaded.
fn validate_case(
    spectra: &[MultigroupSpectrum],
    steps: &[TransmuteStep],
) -> Result<(), Box<dyn std::error::Error>> {
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
    Ok(())
}

/// The result has one series of times, so every material must step through
/// the same durations and irradiate on the same steps. The rates and spectra
/// are each material's own.
fn check_same_timeline(first: &[TransmuteStep], other: &[TransmuteStep]) -> Result<(), String> {
    if first.len() != other.len() {
        return Err(format!(
            "its schedule has {} steps and the first material's has {}; every material \
             must share one timeline",
            other.len(),
            first.len()
        ));
    }
    for (i, (a, b)) in first.iter().zip(other).enumerate() {
        if a.dt != b.dt {
            return Err(format!(
                "step {i} lasts {} s and the first material's lasts {} s; every material \
                 must share one timeline",
                b.dt, a.dt
            ));
        }
        if a.irradiation.is_some() != b.irradiation.is_some() {
            let what = |st: &TransmuteStep| {
                if st.irradiation.is_some() {
                    "an irradiation"
                } else {
                    "a cooldown"
                }
            };
            return Err(format!(
                "step {i} is {} and the first material's is {}; every material must \
                 irradiate on the same steps",
                what(b),
                what(a)
            ));
        }
    }
    Ok(())
}

/// A chain that can drive none of this material's nuclides.
///
/// Every reaction it holds has a parent the material does not contain, so the
/// irradiation produces nothing and the schedule solves to the composition it
/// started with. That is a silent zero: no step fails, no rate is negative,
/// and the mistake surfaces only as a decay heat of exactly zero much later,
/// which names neither the chain nor what it was built for.
///
/// It happens whenever chains are scoped per material and two of them share a
/// path, since the second conversion replaces the first and leaves a directory
/// whose name still says otherwise. The manifest now records the parents each
/// subsection covers; this is the check that acts on them.
///
/// Stronger than the membership check in `preload_activation_data`, which asks
/// whether any of the material's nuclides appear in the chain at all.
/// Appearing is not enough: a nuclide can be in the chain purely as somebody
/// else's product, carrying no reactions of its own, and a material made only
/// of those is exactly as undrivable as one that is absent. The two guards are
/// kept separate because that one must also cover the decay-only path, where
/// having no reactions is not a fault.
///
/// Only when something is actually irradiated. A decay-only schedule drives no
/// reactions by construction, and a chain holding none of this material's
/// parents is the right chain for it.
fn check_chain_drives(
    material: &Material,
    steps: &[TransmuteStep],
    chain: &HashMap<String, ChainNuclide>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !steps.iter().any(|st| st.irradiation.is_some()) {
        return Ok(());
    }
    let has_reactions = |name: &String| chain.get(name).is_some_and(|cn| !cn.reactions.is_empty());
    // Already sorted by `get_nuclides`, which the message relies on.
    let held = material.get_nuclides();
    if held.iter().any(has_reactions) {
        return Ok(());
    }
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
    Err(format!(
        "this chain drives none of the material's nuclides, so the irradiation would \
         produce nothing and every step would return the starting composition.\n  \
         material holds: {}\n  chain has reactions for: {covers}\n\
         A chain is scoped to the nuclides it was built from, so build one for this \
         material rather than reusing one built for another.",
        held.join(" "),
    )
    .into())
}

/// Everything a collapse reads, so two materials with equal keys collapse to
/// identical rates, fission-yield weights, folded chain and shielding report.
///
/// The composition is in it although a dilute collapse does not read the
/// densities for the rates themselves: the report's would-shield indicator
/// does, and a shielded collapse depends on them outright. The loaded data is
/// compared by identity, so two materials share only when they hold the same
/// decoded nuclides, which the shared cache makes the normal case.
#[derive(PartialEq, Eq, Hash)]
struct CollapseKey {
    boundaries: Vec<u64>,
    masses: Vec<u64>,
    flux_error: Option<Vec<u64>>,
    temperature: String,
    composition: Vec<(String, u64)>,
    data: Vec<(String, usize)>,
    chord: Option<u64>,
}

impl CollapseKey {
    fn new(material: &Material, s: &MultigroupSpectrum, shielding: Option<&Shielding>) -> Self {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        let mut composition: Vec<(String, u64)> = material
            .nuclides
            .iter()
            .map(|(n, d)| (n.clone(), d.to_bits()))
            .collect();
        composition.sort_unstable();
        let mut data: Vec<(String, usize)> = material
            .nuclide_data
            .iter()
            .map(|(n, nd)| (n.clone(), Arc::as_ptr(nd) as usize))
            .collect();
        data.sort_unstable();
        CollapseKey {
            boundaries: bits(&s.boundaries),
            masses: bits(&s.masses),
            flux_error: s.flux_error.as_ref().map(|e| e.key_bits()),
            temperature: material.temperature().to_string(),
            composition,
            data,
            chord: shielding.map(|sh| sh.chord_cm.to_bits()),
        }
    }
}

/// One collapse: the rates, yield weights and folded chain for one spectrum
/// against one material, and what the shielding did.
fn collapse_one(
    material: &Material,
    s: &MultigroupSpectrum,
    chain: &Arc<HashMap<String, ChainNuclide>>,
    lists: &Lists<'_>,
    shielding: Option<&Shielding>,
) -> Result<(PerSpectrum, ShieldingInfo, BranchingReport), Box<dyn std::error::Error>> {
    // Where an evaluation ends the cross section is unknown, and the fold
    // treats it as zero. A sliver of flux there is a rounding matter; more
    // than that and every rate on the nuclide would be understated by data
    // that does not exist, so the run stops and says which nuclide and how
    // much rather than answering as if it knew.
    let above = crate::multigroup::spectrum_above_evaluation(material, &s.masses, &s.boundaries);
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
    Ok(collapse_and_fold(material, s, chain, lists, shielding)?)
}

/// What one material's step solve produces.
struct CaseOutcome {
    /// The state after each step.
    states: Vec<Material>,
    /// The per-edge rates each step was solved with.
    rates: Vec<yani::EdgeRates>,
    /// The perturbed ensemble and its report, when uncertainty was asked for.
    uncertainty: Option<(Ensemble, Info)>,
}

/// One material's inputs to the uncertainty replicas, in its own indexing.
struct Replicas<'a> {
    spectra: &'a [MultigroupSpectrum],
    steps: &'a [TransmuteStep],
    per_spectrum: Vec<PerSpectrum>,
    request: &'a DataUncertainty,
    shielding: Option<&'a Shielding>,
}

/// One material's timeline, stepped from `initial` with steps whose spectrum
/// indices point into the shared `per_spectrum`.
fn solve_case(
    initial: &Material,
    steps: &[TransmuteStep],
    per_spectrum: &[PerSpectrum],
    chain: &Arc<HashMap<String, ChainNuclide>>,
    parts: yani::ChainParts,
    replicas: Option<Replicas<'_>>,
) -> Result<CaseOutcome, Box<dyn std::error::Error>> {
    let stepper = ForwardEulerStepper;
    let no_fission: FissionYieldWeights = HashMap::new();
    let mut current_material = initial.clone();
    let mut states = Vec::with_capacity(steps.len());
    let mut edge_rates = Vec::with_capacity(steps.len());
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
            None => (HashMap::new(), &no_fission, chain),
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
        edge_rates.push(per_edge_rates(step_chain, &step_rates));
        states.push(current_material.clone());
    }

    let uncertainty = match replicas {
        Some(r) => Some(run_replicas(
            initial,
            r.spectra,
            r.steps,
            &r.per_spectrum,
            chain,
            parts,
            &stepper,
            r.request,
            r.shielding,
            None,
        )?),
        None => None,
    };
    Ok(CaseOutcome {
        states,
        rates: edge_rates,
        uncertainty,
    })
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
    /// The half-lives this replica was solved with, for the nuclides whose
    /// half-life was perturbed; empty when none were.
    half_lives: HashMap<String, f64>,
    /// Half-life draws that came out non-positive and were floored.
    half_lives_floored: usize,
    /// Statistically drawn rates that came out negative and were floored.
    statistical_floored: usize,
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
        // A nuclide the material is MADE of is not optional. `to_load` is the
        // reachable chain closure, most of which is daughters and
        // grand-daughters: a library may legitimately not publish one of those,
        // and dropping it understates a second-generation inventory. Dropping a
        // nuclide from the composition is different in kind, because the foil
        // then cannot activate at all and the solve returns a decay curve for
        // an unirradiated material with no indication that anything is wrong.
        //
        // Observed on fendl-3.2d, which publishes 61 elements: an osmium foil
        // solved in 0.4 s, reported C/E 0.000 at all 21 cooling points, and
        // said nothing. The precise error already exists one layer down
        // ("Nuclide 'Os190' is not available in 'fendl-3.2d'.") and was being
        // discarded by the `.ok()` this replaces.
        let composition: HashSet<&str> = material.nuclides.keys().map(|s| s.as_str()).collect();
        let decoded: Vec<LoadOutcome> = {
            let load_one = |(name, path): (String, String)| {
                let sources = HashMap::from([(name.clone(), path)]);
                match get_or_load_nuclide(&name, &sources, &scope) {
                    Ok(nd) => Ok(Some((name, nd))),
                    Err(error) if composition.contains(name.as_str()) => {
                        // Just the first line. The underlying error appends the
                        // library's whole published index, which is useful once
                        // and unreadable seven times over, and a foil element
                        // has one entry per natural isotope.
                        let text = error.to_string();
                        Err(text.lines().next().unwrap_or(&text).to_string())
                    }
                    // A daughter the library does not publish. Tolerated, as
                    // before, so a partial network still runs.
                    Err(_) => Ok(None),
                }
            };
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                to_load.into_par_iter().map(load_one).collect()
            }
            #[cfg(target_arch = "wasm32")]
            {
                to_load.into_iter().map(load_one).collect()
            }
        };
        let mut refused: Vec<String> = Vec::new();
        let mut loaded = Vec::new();
        for outcome in decoded {
            match outcome {
                Ok(Some(pair)) => loaded.push(pair),
                Ok(None) => {}
                Err(message) => refused.push(message),
            }
        }
        if !refused.is_empty() {
            refused.sort();
            return Err(format!(
                "cross sections could not be loaded for {} nuclide(s) the \
                 material is made of, so it cannot activate:\n  {}",
                refused.len(),
                refused.join("\n  ")
            )
            .into());
        }
        for (name, nd) in loaded {
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

/// A transport run's tallied rates with their statistical covariance, and
/// the unfolded chain their branching is refined from.
pub(crate) struct TransportStatistics {
    rates: crate::statistical::StatisticalRates,
    base_chain: Arc<HashMap<String, ChainNuclide>>,
    branch: Arc<BranchTable>,
}

/// One material's independent-mode transport result, per source particle.
///
/// What `Model::transmute` extracts from its single transport, and everything
/// [`transport_replicas`] needs to resample it.
pub struct TransportTallied {
    /// Reaction totals per atom per source particle, [1/s] at unit source rate.
    pub rates: ReactionRates,
    /// Isomeric partials at the same normalization.
    pub partials: PartialRates,
    /// Fission-yield spectrum weights.
    pub fy_weights: FissionYieldWeights,
    /// The tallied flux shape, for folding MF=33 covariance against. Its
    /// magnitude does not matter; the fold is relative.
    pub spectrum: MultigroupSpectrum,
    /// The statistical covariance of `rates` and `partials`, when the tally
    /// carried history statistics.
    pub statistics: Option<crate::history_statistics::RateCovariance>,
    /// The branching overlay the partials were scored from, which says what
    /// each of them means (see [`apply_coupled_branching`]).
    pub branch: Arc<BranchTable>,
    /// The clipped and held production and the MT=5 rates the tally measured
    /// beside the partials, at the same normalization.
    pub diagnostics: CoupledDiagnostics,
}

/// Uncertainty on an independent-mode transport transmutation (issue #140,
/// item 3), by resampling and re-solving exactly as [`transmute_material`]
/// does, with the tally's spectrum standing in for the supplied one.
///
/// The sources that apply: `statistical`, the Monte Carlo covariance of the
/// tallied rates; `cross_sections`, the MF=33 covariance folded against the
/// tally's own flux shape; and `half_life`. `flux_spectrum` does not: there is
/// no supplied spectrum, and the flux's error is the statistical one.
///
/// `source_rates` scale the per-source-particle rates per step, which is how
/// independent mode scales them. The coupled method re-runs transport per step
/// and is not covered: its step-to-step noise propagation is out of scope.
pub fn transport_replicas(
    initial: &Material,
    tallied: &TransportTallied,
    timesteps: &[f64],
    source_rates: &[f64],
    chain: &Arc<HashMap<String, ChainNuclide>>,
    parts: yani::ChainParts,
    request: &DataUncertainty,
) -> Result<(Ensemble, Info), Box<dyn std::error::Error>> {
    let mut initial = initial.clone();
    if request.wants(crate::uncertainty::Source::CrossSections) {
        initial.ensure_covariance_loaded()?;
    }
    // The nominal the replicas scatter around: the same branching fold the
    // step loop applies, at unit source rate, since fractions do not depend on
    // the magnitude.
    let mut rates = tallied.rates.clone();
    let folded = if tallied.partials.is_empty() {
        Arc::clone(chain)
    } else {
        apply_coupled_branching(
            chain,
            &tallied.branch,
            &tallied.partials,
            &mut rates,
            Some(&tallied.diagnostics),
        )?
        .0
    };
    let per_spectrum: Vec<PerSpectrum> = vec![(rates, tallied.fy_weights.clone(), folded)];
    let steps: Vec<TransmuteStep> = timesteps
        .iter()
        .zip(source_rates)
        .map(|(&dt, &rate)| TransmuteStep {
            dt,
            irradiation: (rate > 0.0).then_some((0, rate)),
        })
        .collect();
    let statistics = tallied.statistics.as_ref().map(|c| TransportStatistics {
        rates: crate::statistical::StatisticalRates::new(c),
        base_chain: Arc::clone(chain),
        branch: Arc::clone(&tallied.branch),
    });
    let no_statistics = TransportStatistics {
        rates: crate::statistical::StatisticalRates::new(
            &crate::history_statistics::RateCovariance::empty(),
        ),
        base_chain: Arc::clone(chain),
        branch: Arc::clone(&tallied.branch),
    };
    run_replicas(
        &initial,
        std::slice::from_ref(&tallied.spectrum),
        &steps,
        &per_spectrum,
        chain,
        parts,
        &ForwardEulerStepper,
        request,
        None,
        // Always the transport path, even with no statistics to sample: that
        // is what keeps `flux_spectrum` out of it.
        Some(statistics.as_ref().unwrap_or(&no_statistics)),
    )
}

/// What the half-life source needs for a run: the nuclides to perturb, those
/// with no stated sigma, and the chains pruned to what the material can reach,
/// ready to receive a replica's half-lives.
struct HalfLifeSampling {
    /// `(name, half-life, sigma)` for the reachable nuclides with a sigma.
    candidates: Vec<(String, f64, f64)>,
    /// Reachable unstable nuclides whose evaluation states no sigma.
    without: std::collections::BTreeSet<String>,
    /// The base chain, pruned.
    base: HashMap<String, ChainNuclide>,
    /// Each spectrum's folded chain, pruned, in `per_spectrum` order.
    folded: Vec<HashMap<String, ChainNuclide>>,
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
    statistical: Option<&TransportStatistics>,
) -> Result<(Ensemble, Info), Box<dyn std::error::Error>> {
    // The two paths each have one source the other lacks. A transport run has
    // tallied rates with a statistical covariance and no caller-supplied flux
    // error (its flux error IS the statistical one); a spectrum run has the
    // reverse. Each ignores the other's, and the report lists only what
    // applied.
    let transport = statistical.is_some();
    let applies = |s: crate::uncertainty::Source| match s {
        crate::uncertainty::Source::Statistical => transport,
        crate::uncertainty::Source::FluxSpectrum => !transport,
        _ => true,
    };
    let want_statistical = transport && request.wants(crate::uncertainty::Source::Statistical);
    let statistical_in = statistical;
    let statistical = statistical.filter(|s| want_statistical && !s.rates.is_empty());

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
    let want_flux = !transport && request.wants(crate::uncertainty::Source::FluxSpectrum);
    let mut flux_coverage = crate::flux_uncertainty::FluxCoverage::default();
    let mut per_group: Vec<
        Option<(
            crate::flux_uncertainty::FluxError,
            crate::flux_uncertainty::PerGroupRates,
        )>,
    > = Vec::with_capacity(per_spectrum.len());
    for (idx, spectrum) in spectra.iter().enumerate() {
        match spectrum.flux_error.as_ref().filter(|_| want_flux) {
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

    // Half-lives: sampled per replica from the evaluation's stated sigma, and
    // substituted into every chain the replica is solved with, the base one
    // for cooldowns and each spectrum's folded one for irradiations, so one
    // replica has one set of decay constants throughout. Only the nuclides
    // this material can reach are carried into those chains: the stepper walks
    // a subset of that closure, so pruning to it changes nothing, and it keeps
    // the per-replica copy to the part of a 3800-nuclide chain the solve uses.
    let half_life = if request.wants(crate::uncertainty::Source::HalfLife) {
        let seeds: Vec<&str> = initial
            .nuclides
            .keys()
            .chain(initial.nuclide_data.keys())
            .map(|s| s.as_str())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let mut reach = yani::reachable_nuclides(chain, &seeds);
        for (_, _, folded_chain) in per_spectrum {
            reach.extend(yani::reachable_nuclides(folded_chain, &seeds));
        }
        let prune = |c: &HashMap<String, ChainNuclide>| -> HashMap<String, ChainNuclide> {
            c.iter()
                .filter(|(name, _)| reach.contains(*name))
                .map(|(name, cn)| (name.clone(), cn.clone()))
                .collect()
        };
        let base = prune(chain);
        let (candidates, without) = crate::uncertainty::half_life_candidates(&base);
        let folded: Vec<HashMap<String, ChainNuclide>> =
            per_spectrum.iter().map(|(_, _, c)| prune(c)).collect();
        Some(HalfLifeSampling {
            candidates,
            without,
            base,
            folded,
        })
    } else {
        None
    };

    // Decay energies: no solve reads them, so they are drawn where decay heat
    // is evaluated from each replica. Here only who has a sigma to draw from.
    let want_decay_energy = request.wants(crate::uncertainty::Source::DecayEnergy);
    let (decay_energy_perturbed, no_decay_energy_sigma) = if want_decay_energy {
        let seeds: Vec<&str> = initial
            .nuclides
            .keys()
            .chain(initial.nuclide_data.keys())
            .map(|s| s.as_str())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let mut with = std::collections::BTreeSet::new();
        let mut without = std::collections::BTreeSet::new();
        for name in yani::reachable_nuclides(chain, &seeds) {
            let Some(cn) = chain.get(&name) else { continue };
            if cn.half_life.is_none_or(|t| t <= 0.0) || cn.decay_energy <= 0.0 {
                continue;
            }
            if crate::uncertainty::has_decay_energy_sigma(cn) {
                with.insert(name);
            } else {
                without.insert(name);
            }
        }
        (with, without)
    } else {
        Default::default()
    };

    let mut info = Info::from_fold(&coverage, &clipping);
    if half_life.is_none() {
        info.not_perturbed.insert(0, "half-life".to_string());
    }
    if !want_decay_energy {
        info.not_perturbed.insert(0, "decay energy".to_string());
    }
    info.decay_energies_perturbed = decay_energy_perturbed.clone();
    info.no_decay_energy_uncertainty = no_decay_energy_sigma;
    if let Some(h) = &half_life {
        info.half_lives_perturbed = h.candidates.iter().map(|(n, _, _)| n.clone()).collect();
        info.no_half_life_uncertainty = h.without.clone();
    }
    let requested: &[crate::uncertainty::Source] = if request.sources.is_empty() {
        crate::uncertainty::Source::IMPLEMENTED
    } else {
        &request.sources
    };
    info.sources = requested
        .iter()
        .filter(|s| applies(**s))
        .map(|s| s.name().to_string())
        .collect();
    if let Some(st) = statistical {
        info.statistical_rates = st.rates.len();
    }
    let mut ensemble = Ensemble::new(steps.len());
    if !decay_energy_perturbed.is_empty() {
        ensemble.decay_energy_seed = Some(request.seed);
    }

    // Nothing to perturb means nothing to sample. The ensemble stays empty and
    // every sigma reads zero, with `info` saying why: no covariance data, not a
    // confident zero.
    let no_half_lives = half_life.as_ref().is_none_or(|h| h.candidates.is_empty());
    if samplers.iter().all(Sampler::is_empty)
        && per_group.iter().all(Option::is_none)
        && no_half_lives
        && statistical.is_none()
    {
        info.converged = true;
        info.add_flux_coverage(&flux_coverage);
        // Decay energies alone leave every inventory at nominal, but the
        // decay heat of each still moves, so the replicas are the nominal
        // inventory repeated: no solve beyond the one.
        if !decay_energy_perturbed.is_empty() {
            let nominal = densities_of(
                &replica_steps(initial, steps, per_spectrum, chain, parts, stepper)
                    .map_err(|e| e.to_string())?,
            );
            for _ in 0..request.samples.unwrap_or(MIN_SAMPLES) {
                ensemble.push_with_half_lives(nominal.clone(), HashMap::new());
            }
            ensemble.fold_absences();
            info.samples = ensemble.replicas();
        }
        if request.attribution {
            ensemble.attribution = Some(Default::default());
        }
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
        let mut half_lives_floored = 0usize;
        // A statistical draw of the whole tallied rate vector, the partials
        // re-folded into the branching the way the nominal was, so an
        // isomeric split moves with the rates it is made of.
        let mut statistical_floored = 0usize;
        let drawn = match statistical {
            Some(st) => {
                let (mut totals, partials, floored) = st.rates.sample(request.seed, replica);
                statistical_floored = floored;
                // The nominal was guarded; a draw around it only moves the
                // productions it is made of.
                let chain_k = if partials.is_empty() {
                    Arc::clone(&st.base_chain)
                } else {
                    apply_coupled_branching(
                        &st.base_chain,
                        &st.branch,
                        &partials,
                        &mut totals,
                        None,
                    )?
                    .0
                };
                Some((totals, chain_k))
            }
            None => None,
        };
        let sampled_half_lives = match &half_life {
            Some(h) if !h.candidates.is_empty() => crate::uncertainty::sample_half_lives(
                &h.candidates,
                request.seed,
                replica,
                &mut half_lives_floored,
            ),
            _ => HashMap::new(),
        };

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
            // A transport run has one spectrum, the tally's own, whose rates
            // and branching this replica drew statistically.
            let (rates, folded_chain) = match (&drawn, idx) {
                (Some((totals, chain_k)), 0) => (totals, chain_k),
                _ => (rates, folded_chain),
            };
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
            let folded_chain = match &half_life {
                // The pruned nominal chain, unless this replica drew its own
                // branching, which then carries the half-lives instead.
                Some(h) if !sampled_half_lives.is_empty() => Arc::new(if drawn.is_some() {
                    crate::uncertainty::with_half_lives(folded_chain, &sampled_half_lives)
                } else {
                    crate::uncertainty::with_half_lives(&h.folded[idx], &sampled_half_lives)
                }),
                _ => Arc::clone(folded_chain),
            };
            perturbed.push((rates, weights.clone(), folded_chain));
        }
        let base_chain = match &half_life {
            Some(h) if !sampled_half_lives.is_empty() => Arc::new(
                crate::uncertainty::with_half_lives(&h.base, &sampled_half_lives),
            ),
            _ => Arc::clone(chain),
        };

        // `Box<dyn Error>` is not `Send`, so it cannot come back out of a
        // parallel map; the driver puts the message back into one.
        let materials = replica_steps(initial, steps, &perturbed, &base_chain, parts, stepper)
            .map_err(|e| e.to_string())?;
        Ok(ReplicaOutcome {
            densities: densities_of(&materials),
            truncations,
            flux_bins_sampled: flux_coverage.bins_sampled,
            flux_bins_floored: flux_coverage.bins_floored,
            half_lives: sampled_half_lives,
            half_lives_floored,
            statistical_floored,
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
            info.half_lives_sampled += outcome.half_lives.len();
            info.half_lives_floored += outcome.half_lives_floored;
            info.statistical_floored += outcome.statistical_floored;
            if statistical.is_some() {
                info.statistical_sampled += info.statistical_rates;
            }
            ensemble.push_with_half_lives(outcome.densities, outcome.half_lives);
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

    if request.attribution {
        let applied: Vec<crate::uncertainty::Source> = info
            .sources
            .iter()
            .filter_map(|n| crate::uncertainty::Source::parse(n).ok())
            .collect();
        let mut by_source = std::collections::BTreeMap::new();
        if applied.len() == 1 {
            by_source.insert(
                info.sources[0].clone(),
                variances_of(&ensemble, steps.len()),
            );
        } else {
            for &source in &applied {
                let alone = DataUncertainty {
                    seed: request.seed,
                    samples: request.samples,
                    sources: vec![source],
                    attribution: false,
                };
                let (sub, _) = run_replicas(
                    initial,
                    spectra,
                    steps,
                    per_spectrum,
                    chain,
                    parts,
                    stepper,
                    &alone,
                    shielding,
                    statistical_in,
                )?;
                by_source.insert(source.name().to_string(), variances_of(&sub, steps.len()));
            }
        }
        let contributors = first_order_contributors(
            initial,
            steps,
            per_spectrum,
            chain,
            parts,
            stepper,
            applied
                .contains(&crate::uncertainty::Source::CrossSections)
                .then_some(samplers.as_slice()),
            half_life.as_ref(),
        )?;
        ensemble.attribution = Some(crate::uncertainty::Attribution {
            by_source,
            contributors,
        });
    }
    Ok((ensemble, info))
}

/// Each step's per-nuclide variance in an ensemble.
fn variances_of(ensemble: &Ensemble, n_steps: usize) -> Vec<HashMap<String, f64>> {
    (0..n_steps)
        .map(|step| {
            ensemble
                .std_dev_at(step)
                .into_iter()
                .filter(|(_, s)| *s > 0.0)
                .map(|(n, s)| (n, s * s))
                .collect()
        })
        .collect()
}

/// Relative step for the first-order sensitivities. Small enough that the
/// response is linear to well within what a variance share is read to, large
/// enough to stand clear of the solver's own rounding.
const SENSITIVITY_STEP: f64 = 1.0e-3;

/// First-order contributions to the inventory variance, one deterministic
/// solve per contributor.
///
/// Cross sections: each nuclide's evaluation (its channels with their
/// correlations, `w^T w` with `w = L^T s` over the factor its sampler uses)
/// and each channel alone. Half-lives: each nuclide the solve populates, as
/// `(s sigma_T / T)^2`. A half-life of a nuclide that never appears in the
/// inventory cannot move it, which keeps this to the nuclides that matter
/// even in a fission chain.
#[allow(clippy::too_many_arguments)]
fn first_order_contributors(
    initial: &Material,
    steps: &[TransmuteStep],
    per_spectrum: &[PerSpectrum],
    chain: &Arc<HashMap<String, ChainNuclide>>,
    parts: yani::ChainParts,
    stepper: &ForwardEulerStepper,
    samplers: Option<&[Sampler]>,
    half_life: Option<&HalfLifeSampling>,
) -> Result<Vec<crate::uncertainty::Contributor>, Box<dyn std::error::Error>> {
    use crate::uncertainty::Contributor;
    let solve = |ps: &[PerSpectrum], base: &Arc<HashMap<String, ChainNuclide>>| {
        replica_steps(initial, steps, ps, base, parts, stepper)
            .map(|m| densities_of(&m))
            .map_err(|e| e.to_string())
    };
    let nominal = solve(per_spectrum, chain)?;
    // (N' - N) / h, per step and nuclide, over the union of both inventories.
    let sensitivity = |perturbed: &[HashMap<String, f64>]| -> Vec<HashMap<String, f64>> {
        nominal
            .iter()
            .zip(perturbed)
            .map(|(a, b)| {
                a.keys()
                    .chain(b.keys())
                    .map(|n| {
                        let d = b.get(n).copied().unwrap_or(0.0) - a.get(n).copied().unwrap_or(0.0);
                        (n.clone(), d / SENSITIVITY_STEP)
                    })
                    .collect()
            })
            .collect()
    };

    let mut out: Vec<Contributor> = Vec::new();

    // Cross sections: one solve per (spectrum, nuclide, channel).
    if let Some(samplers) = samplers {
        type Job = (usize, String, usize, String);
        let mut jobs: Vec<Job> = Vec::new();
        for (a, sampler) in samplers.iter().enumerate() {
            for (name, kinds, _) in sampler.factors() {
                for (i, kind) in kinds.iter().enumerate() {
                    let rate = per_spectrum[a].0.get(name).and_then(|r| r.get(kind));
                    if rate.is_some_and(|r| *r > 0.0) {
                        jobs.push((a, name.clone(), i, kind.clone()));
                    }
                }
            }
        }
        let run = |(a, name, _, kind): &Job| -> Result<Vec<HashMap<String, f64>>, String> {
            let mut ps: Vec<PerSpectrum> = per_spectrum.to_vec();
            if let Some(r) = ps[*a].0.get_mut(name).and_then(|r| r.get_mut(kind)) {
                *r *= 1.0 + SENSITIVITY_STEP;
            }
            Ok(sensitivity(&solve(&ps, chain)?))
        };
        let results: Vec<Result<Vec<HashMap<String, f64>>, String>> = {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                jobs.par_iter().map(run).collect()
            }
            #[cfg(target_arch = "wasm32")]
            {
                jobs.iter().map(run).collect()
            }
        };
        // w[nuclide or (nuclide, kind)][step][output] -> vector over the
        // factor's columns, accumulated over spectra (they share the deviate).
        type PerStep = Vec<HashMap<String, Vec<f64>>>;
        let mut block: HashMap<String, PerStep> = HashMap::new();
        let mut channel: HashMap<(String, String), PerStep> = HashMap::new();
        for ((a, name, i, kind), sens) in jobs.iter().zip(results) {
            let sens = sens?;
            let (_, kinds, l) = samplers[*a]
                .factors()
                .find(|(n, _, _)| *n == name)
                .expect("the job came from this factor");
            let n = kinds.len();
            let row = &l[i * n..(i + 1) * n];
            for target in [
                block
                    .entry(name.clone())
                    .or_insert_with(|| vec![HashMap::new(); steps.len()]),
                channel
                    .entry((name.clone(), kind.clone()))
                    .or_insert_with(|| vec![HashMap::new(); steps.len()]),
            ] {
                for (step, per) in sens.iter().enumerate() {
                    for (out_name, s) in per {
                        if *s == 0.0 {
                            continue;
                        }
                        let w = target[step]
                            .entry(out_name.clone())
                            .or_insert_with(|| vec![0.0; n]);
                        if w.len() < n {
                            w.resize(n, 0.0);
                        }
                        for (wj, lj) in w.iter_mut().zip(row) {
                            *wj += s * lj;
                        }
                    }
                }
            }
        }
        let collapse = |w: PerStep| -> Vec<HashMap<String, f64>> {
            w.into_iter()
                .map(|per| {
                    per.into_iter()
                        .map(|(k, v)| (k, v.iter().map(|x| x * x).sum::<f64>()))
                        .filter(|(_, v)| *v > 0.0)
                        .collect()
                })
                .collect()
        };
        for (name, w) in block {
            out.push(Contributor {
                source: "cross_sections".to_string(),
                nuclide: name,
                reaction: None,
                variance: collapse(w),
            });
        }
        for ((name, kind), w) in channel {
            out.push(Contributor {
                source: "cross_sections".to_string(),
                nuclide: name,
                reaction: Some(kind),
                variance: collapse(w),
            });
        }
    }

    // Half-lives: one solve per populated nuclide with a stated sigma.
    if let Some(h) = half_life {
        let populated: HashSet<&str> = nominal
            .iter()
            .flat_map(|m| m.keys())
            .chain(initial.nuclides.keys())
            .map(|s| s.as_str())
            .collect();
        let jobs: Vec<&(String, f64, f64)> = h
            .candidates
            .iter()
            .filter(|(n, _, _)| populated.contains(n.as_str()))
            .collect();
        let run =
            |(name, t, _): &&(String, f64, f64)| -> Result<Vec<HashMap<String, f64>>, String> {
                let sampled = HashMap::from([(name.clone(), t * (1.0 + SENSITIVITY_STEP))]);
                let base = Arc::new(crate::uncertainty::with_half_lives(&h.base, &sampled));
                let ps: Vec<PerSpectrum> = per_spectrum
                    .iter()
                    .zip(&h.folded)
                    .map(|((r, w, _), folded)| {
                        (
                            r.clone(),
                            w.clone(),
                            Arc::new(crate::uncertainty::with_half_lives(folded, &sampled)),
                        )
                    })
                    .collect();
                Ok(sensitivity(&solve(&ps, &base)?))
            };
        let results: Vec<Result<Vec<HashMap<String, f64>>, String>> = {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                jobs.par_iter().map(run).collect()
            }
            #[cfg(target_arch = "wasm32")]
            {
                jobs.iter().map(run).collect()
            }
        };
        for ((name, t, sigma), sens) in jobs.into_iter().zip(results) {
            let rel = sigma / t;
            let variance = sens?
                .into_iter()
                .map(|per| {
                    per.into_iter()
                        .map(|(k, s)| (k, (s * rel) * (s * rel)))
                        .filter(|(_, v)| *v > 0.0)
                        .collect()
                })
                .collect();
            out.push(Contributor {
                source: "half_life".to_string(),
                nuclide: name.clone(),
                reaction: None,
                variance,
            });
        }
    }

    // Largest reach first: by the variance summed over the last step.
    let reach = |c: &Contributor| -> f64 { c.variance.last().map_or(0.0, |m| m.values().sum()) };
    out.retain(|c| c.variance.iter().any(|m| !m.is_empty()));
    out.sort_by(|a, b| {
        reach(b)
            .partial_cmp(&reach(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                (&a.source, &a.nuclide, &a.reaction).cmp(&(&b.source, &b.nuclide, &b.reaction))
            })
    });
    Ok(out)
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

/// What one nuclide's cross-section load produced.
///
/// `Ok(Some(..))` loaded, `Ok(None)` absent and tolerable because it is a chain
/// daughter rather than something the material is made of, and `Err` refused.
type LoadOutcome = Result<Option<(String, std::sync::Arc<yamc_nuclide::Nuclide>)>, String>;

/// Collapse one material against one spectrum and fold its isomeric branching
/// into the chain: the rates, fission-yield weights and chain with this
/// spectrum's splits, what the shielding did, and what the branching rule did.
///
/// The rule is [`crate::branching_rule`]'s: each list is folded in the walk
/// that collapses its parent's transport total, so a state's production and
/// the total it comes out of are taken under the same weight and the same
/// shielded flux shape. A parent with no transport data has no rate to split,
/// and only its `(n,n')` lists, which carry their own rate, are folded, from
/// the partials alone.
///
/// Refuses the spectrum where the rule would rest more than
/// [`BRANCHING_RATE_TOLERANCE`] of a parent's removal rate on a clipped or
/// held value, and reports MT=5's share of each parent's removal (see
/// [`measure_unmodelled_mt5`]).
pub(crate) fn collapse_and_fold(
    material: &Material,
    spectrum: &MultigroupSpectrum,
    chain: &Arc<HashMap<String, ChainNuclide>>,
    lists: &Lists<'_>,
    shielding: Option<&Shielding>,
) -> Result<(PerSpectrum, ShieldingInfo, BranchingReport), String> {
    let Collapsed {
        mut rates,
        fy_weights,
        info,
        lists: walked,
        mt5,
    } = collapse_with_lists(
        material,
        chain,
        lists,
        &spectrum.masses,
        &spectrum.boundaries,
        1.0,
        shielding,
    );
    let mut folded: HashMap<String, Vec<Option<ListRates>>> = walked
        .into_iter()
        .map(|(parent, per)| (parent, per.into_iter().map(Some).collect()))
        .collect();
    let groups: Vec<usize> = (0..spectrum.masses.len())
        .filter(|&g| spectrum.masses[g] != 0.0)
        .collect();
    for (parent, rules) in lists {
        if folded.contains_key(parent) {
            continue;
        }
        let per: Vec<Option<ListRates>> = rules
            .iter()
            .map(|rule| {
                (rule.kind == INELASTIC).then(|| {
                    fold_list(
                        &Bound {
                            rule,
                            total: None,
                            tail: &[],
                        },
                        &spectrum.masses,
                        &spectrum.boundaries,
                        groups.iter().copied(),
                        None,
                        1.0,
                    )
                })
            })
            .collect();
        folded.insert(parent.clone(), per);
    }
    let (folded_chain, mut report) = fold_branching_into_chain(chain, lists, &folded, &mut rates)?;
    report.unmodelled_mt5 = measure_unmodelled_mt5(&rates, &mt5);
    Ok(((rates, fy_weights, folded_chain), info, report))
}

/// Apply folded branching lists to the chain and the rates.
///
/// Shared by the multigroup fold ([`collapse_and_fold`]) and the coupled path
/// ([`apply_coupled_branching`]), which differ only in where each list's
/// productions came from. Per list:
///
/// * `(n,n')`: the productions of the isomers are the rate, injected into
///   `rates` (there is no chain total for MT=4), and each grafted channel gets
///   its share of it;
/// * a complete list: each state's share is its production over the listed
///   states' productions, and re-partitions the mass those states carry in the
///   chain;
/// * a list of isomers only: each state's share is its production over the
///   transport total folded in the same walk, and the chain's other targets
///   for the reaction (the ground state) take the rest.
///
/// Then the guards: a list whose clipped or held production is more than
/// [`BRANCHING_RATE_TOLERANCE`] of its parent's removal rate refuses the run.
/// Below that it is reported, with everything else the report carries.
pub(crate) fn fold_branching_into_chain(
    chain: &Arc<HashMap<String, ChainNuclide>>,
    lists: &Lists<'_>,
    folded: &HashMap<String, Vec<Option<ListRates>>>,
    rates: &mut ReactionRates,
) -> Result<(Arc<HashMap<String, ChainNuclide>>, BranchingReport), String> {
    let mut refine: Fractions = HashMap::new();
    let mut channels: Vec<Resolved> = Vec::new();
    let mut dropped: Vec<(DroppedChannel, Option<f64>)> = Vec::new();
    let mut parents: Vec<&String> = lists.keys().collect();
    parents.sort();
    for parent in parents {
        let (Some(per), Some(nuc)) = (folded.get(parent), chain.get(parent)) else {
            continue;
        };
        for (rule, list_rates) in lists[parent].iter().zip(per) {
            // A parent with no transport data has no rate but `(n,n')` to
            // split, and its other lists were not folded.
            let Some(list_rates) = list_rates else {
                continue;
            };
            resolve_list(
                rule,
                list_rates,
                nuc,
                rates,
                &mut refine,
                &mut channels,
                &mut dropped,
            );
        }
    }

    // Measured against removal rates that include the `(n,n')` rates the
    // lists injected, so every list is judged against the same whole.
    let mut refusals: Vec<String> = Vec::new();
    let mut report = BranchingReport::default();
    for resolved in channels {
        let mut channel = resolved.channel;
        let removal = removal_rate(rates, &channel.parent);
        let share_of = |rate: f64| if removal > 0.0 { rate / removal } else { 0.0 };
        channel.removal_share = share_of(resolved.rate);
        channel.clipped_share = share_of(resolved.clipped);
        channel.extrapolated_share = share_of(resolved.extrapolated);
        if channel.clipped_share > BRANCHING_RATE_TOLERANCE {
            let excess = match channel.own_total_excess {
                Some((e, ratio)) if ratio.is_finite() => format!(
                    " (the listed values reach {ratio:.4} times the evaluation's own total, at \
                     {e:.4e} eV)"
                ),
                Some((e, _)) => {
                    format!(" (the listed values are non-zero where the evaluation's own total is zero, at {e:.4e} eV)")
                }
                None => String::new(),
            };
            refusals.push(format!(
                "{} {}: {:.3}% of {}'s neutron removal rate is production the evaluation gives \
                 above the reaction's transport total, or as a negative value{excess}",
                channel.parent,
                channel.reaction,
                100.0 * channel.clipped_share,
                channel.parent
            ));
        }
        if channel.extrapolated_share > BRANCHING_RATE_TOLERANCE {
            refusals.push(format!(
                "{} {}: {:.3}% of {}'s neutron removal rate lies where the branching \
                 evaluation tabulates no split, and would rest on a fraction held from the \
                 edge of its range",
                channel.parent,
                channel.reaction,
                100.0 * channel.extrapolated_share,
                channel.parent
            ));
        }
        report.channels.push(channel);
    }
    if !refusals.is_empty() {
        return Err(format!(
            "the isomeric branching cannot be applied to this spectrum without moving more \
             than {:.1}% of a parent's removal rate onto values the evaluation does not give. \
             {}. Nothing is clipped or extrapolated silently: use a branching evaluation that \
             covers this spectrum consistently with the cross-section library, or leave the \
             branching overlay out for these nuclides.",
            100.0 * BRANCHING_RATE_TOLERANCE,
            refusals.join("; ")
        ));
    }
    for (mut d, rate) in dropped {
        let removal = removal_rate(rates, &d.parent);
        d.removal_share = rate.map(|r| if removal > 0.0 { r / removal } else { 0.0 });
        report.dropped.push(d);
    }
    Ok((refine_chain(chain, &refine), report))
}

/// A channel's report, with the rates its shares are still to be taken of.
struct Resolved {
    channel: BranchingChannel,
    rate: f64,
    clipped: f64,
    extrapolated: f64,
}

/// Turn one folded list into its split, its report and whatever it drops.
fn resolve_list(
    rule: &ListRule<'_>,
    folded: &ListRates,
    nuc: &ChainNuclide,
    rates: &mut ReactionRates,
    refine: &mut Fractions,
    channels: &mut Vec<Resolved>,
    dropped: &mut Vec<(DroppedChannel, Option<f64>)>,
) {
    let (parent, kind) = (rule.parent.as_str(), rule.kind.as_str());
    let to_rate = folded.to_rate;
    let drop = |target: Option<&str>, reason: String, rate: Option<f64>| {
        (
            DroppedChannel {
                parent: parent.to_string(),
                reaction: kind.to_string(),
                target: target.map(str::to_string),
                reason,
                removal_share: None,
            },
            rate,
        )
    };
    for c in &rule.passed_over {
        dropped.push(drop(
            Some(&c.target),
            "an MF=9 yield passed over: the evaluation gives MF=10 partials for the same \
             reaction, which are used"
                .to_string(),
            None,
        ));
    }

    // Productions per target, duplicates summed (TENDL can book two levels to
    // one chain nuclide), in the order the list names them.
    let mut per_target: Vec<(String, f64)> = Vec::new();
    for (k, c) in rule.curves.iter().enumerate() {
        if !rule.produces[k] {
            continue;
        }
        let p = folded.production.get(k).copied().unwrap_or(0.0);
        match per_target.iter_mut().find(|(t, _)| *t == c.target) {
            Some((_, sum)) => *sum += p,
            None => per_target.push((c.target.clone(), p)),
        }
    }
    let produced: f64 = per_target.iter().map(|(_, p)| p).sum();

    let reaction_rate = if kind == INELASTIC {
        if produced <= 0.0 {
            return;
        }
        let rate = produced * to_rate;
        rates
            .entry(parent.to_string())
            .or_default()
            .insert(INELASTIC.to_string(), rate);
        rate
    } else {
        let rate = rates
            .get(parent)
            .and_then(|r| r.get(kind))
            .copied()
            .unwrap_or(0.0);
        if !nuc.reactions.iter().any(|r| r.kind == kind) {
            let size = (folded.has_total && folded.total > 0.0).then_some(folded.total * to_rate);
            dropped.push(drop(
                None,
                format!("the chain has no {kind} reaction for {parent}"),
                size,
            ));
            return;
        }
        if !folded.has_total {
            let why = match rule.mt {
                Some(mt) => format!("the cross-section library has no MT={mt} for {parent}"),
                None => format!("{kind} has no transport MT"),
            };
            dropped.push(drop(None, why, None));
            return;
        }
        if rate <= 0.0 {
            return;
        }
        rate
    };

    // The shares of the reaction this run drives.
    let whole = match rule.denominator {
        Denominator::ListedSum => produced,
        Denominator::TransportTotal if kind == INELASTIC => produced,
        Denominator::TransportTotal => folded.total,
    };
    if whole <= 0.0 {
        if kind != INELASTIC {
            dropped.push(drop(
                None,
                "the listed states produce nothing under this spectrum where the reaction \
                 does, so the chain's own split is kept"
                    .to_string(),
                Some(reaction_rate),
            ));
        }
        return;
    }
    let shares: Vec<(String, f64)> = per_target
        .iter()
        .map(|(t, p)| (t.clone(), (p / whole).min(1.0)))
        .collect();

    // What the chain can carry of it.
    let carried = |t: &str| {
        nuc.reactions
            .iter()
            .any(|r| r.kind == kind && r.target.as_deref() == Some(t))
    };
    for (t, f) in &shares {
        if !carried(t) {
            let went = match (kind, rule.has_remainder()) {
                (INELASTIC, _) => "it is not produced",
                (_, true) => "its share stays with the ground state",
                (_, false) => "its share goes to the other listed states",
            };
            dropped.push(drop(
                Some(t),
                format!("{t} is not in the chain, so {went}"),
                Some(f * reaction_rate),
            ));
        }
    }
    let remainder = rule.has_remainder() && kind != INELASTIC;
    if kind != INELASTIC {
        if remainder {
            let listed: f64 = shares
                .iter()
                .filter(|(t, _)| carried(t))
                .map(|(_, f)| f)
                .sum();
            let rest_carried = nuc.reactions.iter().any(|r| {
                r.kind == kind
                    && r.branching > 0.0
                    && r.target
                        .as_deref()
                        .is_some_and(|t| !shares.iter().any(|(s, _)| s == t))
            });
            if !rest_carried && listed < 1.0 {
                dropped.push(drop(
                    None,
                    "the rest of the reaction, the ground state's, has no target in the chain"
                        .to_string(),
                    Some((1.0 - listed) * reaction_rate),
                ));
            }
        } else if !nuc.reactions.iter().any(|r| {
            r.kind == kind
                && r.branching > 0.0
                && r.target.as_deref().is_some_and(carried_in(&shares))
        }) {
            dropped.push(drop(
                None,
                "the chain carries none of the listed states for this reaction, so its own \
                 split is kept"
                    .to_string(),
                Some(reaction_rate),
            ));
        }
    }

    let split = refine
        .entry(parent.to_string())
        .or_default()
        .entry(kind.to_string())
        .or_default();
    for (t, f) in &shares {
        *split.fractions.entry(t.clone()).or_insert(0.0) += f;
    }
    split.remainder = remainder;

    // The report: a share of the reaction, for `(n,n')` of the MT=4 total
    // where there is one.
    let report_whole = if kind == INELASTIC && folded.has_total && folded.total > 0.0 {
        folded.total
    } else {
        whole
    };
    let states: Vec<BranchingState> = per_target
        .iter()
        .map(|(t, p)| {
            let facts: Vec<&yani::BranchState> = rule
                .curves
                .iter()
                .filter(|c| c.target == *t)
                .flat_map(|c| c.states.iter())
                .collect();
            BranchingState {
                target: t.clone(),
                lfs: facts.iter().map(|s| s.lfs).collect(),
                level_route: facts.iter().map(|s| s.level_route.clone()).collect(),
                level_energy_difference: facts.iter().map(|s| s.level_energy_difference).collect(),
                share: p / report_whole,
            }
        })
        .collect();
    let denominator = match (rule.denominator, rule.yields) {
        (Denominator::ListedSum, false) => "sum of the listed partials",
        (Denominator::ListedSum, true) => "sum of the listed yields",
        (Denominator::TransportTotal, _) => "transport total",
    };
    channels.push(Resolved {
        channel: BranchingChannel {
            parent: parent.to_string(),
            reaction: kind.to_string(),
            mt: rule
                .curves
                .iter()
                .flat_map(|c| c.states.iter())
                .map(|s| s.mt)
                .next(),
            file: if rule.yields { 9 } else { 10 },
            representation: if rule.is_absolute() {
                "absolute"
            } else {
                "share"
            }
            .to_string(),
            complete: rule.complete,
            completeness_source: "converter (list_complete)".to_string(),
            denominator: denominator.to_string(),
            states,
            removal_share: 0.0,
            clipped_share: 0.0,
            extrapolated_share: 0.0,
            own_total_excess: rule.own_total_excess,
            normalisation: rule.curves.iter().find_map(|c| c.normalisation.clone()),
        },
        rate: reaction_rate,
        clipped: folded.clipped * to_rate,
        extrapolated: folded.extrapolated * to_rate,
    });
}

/// Whether a chain target is one of the listed states.
fn carried_in(shares: &[(String, f64)]) -> impl Fn(&str) -> bool + '_ {
    move |t| shares.iter().any(|(s, _)| s == t)
}

/// `parent -> reaction kind -> folded split`.
type Fractions = HashMap<String, HashMap<String, Split>>;

/// One reaction's folded fractions, and what they are fractions of.
#[derive(Default)]
struct Split {
    /// `target -> fraction`.
    fractions: HashMap<String, f64>,
    /// Whether `fractions` are shares of the whole reaction, the chain's other
    /// targets for it (the ground state) taking the rest, rather than shares
    /// among the listed states, which re-partition only the mass those states
    /// already carry in the chain (see [`refine_chain`]): a complete list may
    /// cover only some of the chain's targets, as ENDF/B-VIII.1's La139
    /// (n,d3He) lists Xe135 and Xe135_m1 where the chain goes to Xe134.
    remainder: bool,
}

/// Rewrite the chain's branching splits from per-target fractions; returns the
/// original chain untouched (no clone) when there is nothing to refine.
///
/// A split with a remainder sets each listed state to its share of the
/// reaction's mass and the chain's other targets to what is left, in the
/// proportions they carried. Every other split re-partitions the mass its
/// listed states already carry.
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
        for (kind, split) in kinds {
            let tmap = &split.fractions;
            if kind.as_str() == INELASTIC {
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
            } else if split.remainder {
                // Each fraction is a share of the whole reaction, and the
                // targets the list does not name take the rest in the
                // proportions they carried: the ground state, where the chain
                // books the reaction to it.
                let mut mass = 0.0;
                let mut rest_mass = 0.0;
                let mut listed = 0.0;
                for rx in nuc.reactions.iter().filter(|r| &r.kind == kind) {
                    mass += rx.branching;
                    match rx.target.as_deref().and_then(|t| tmap.get(t)) {
                        Some(&f) => listed += f,
                        None => rest_mass += rx.branching,
                    }
                }
                let left = mass * (1.0 - listed).max(0.0);
                for rx in nuc.reactions.iter_mut().filter(|r| &r.kind == kind) {
                    match rx.target.as_deref().and_then(|t| tmap.get(t)) {
                        Some(&f) => rx.branching = f * mass,
                        None if rest_mass > 0.0 => {
                            rx.branching = rx.branching / rest_mass * left;
                        }
                        None => {}
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

/// What the coupled tally measured beside the partials, per list, for the
/// guards and the report (see [`TransmutationTallies::get_branching_diagnostics`]).
#[derive(Clone, Debug, Default)]
pub struct CoupledDiagnostics {
    /// `parent -> kind -> (clipped, held)` production rates, on the footing of
    /// the partials.
    pub lists: HashMap<String, HashMap<String, (f64, f64)>>,
    /// `parent -> MT=4 total rate`, for reporting `(n,n')` shares of it.
    pub inelastic_totals: HashMap<String, f64>,
    /// `parent -> MT=5 rate`.
    pub mt5: HashMap<String, f64>,
}

/// Apply the isomeric-branching overlay on the coupled path (issue #218).
///
/// The tally scores each list the way [`crate::branching_rule`] defines it,
/// at the collision energies, so its partials are already the productions:
/// a state's share of its reaction's transport total for every list on one of
/// the material's own nuclides, and the `(n,n')` partials folded from the
/// union-grid flux moments for every other chain parent. What remains is what
/// the multigroup fold does with its folded lists ([`fold_branching_into_chain`]),
/// with the tallied totals in place of the collapsed ones.
///
/// `diagnostics` carries the clipped and held production, which the nominal
/// run is guarded with, and the MT=5 rates it reports; a statistical replica,
/// re-drawn from a nominal already guarded, passes `None`, and its report is
/// empty of them.
pub fn apply_coupled_branching(
    chain: &Arc<HashMap<String, ChainNuclide>>,
    branch: &BranchTable,
    partial_rates: &PartialRates,
    rates: &mut ReactionRates,
    diagnostics: Option<&CoupledDiagnostics>,
) -> Result<(Arc<HashMap<String, ChainNuclide>>, BranchingReport), String> {
    let mut lists: Lists<'_> = HashMap::new();
    let mut folded: HashMap<String, Vec<Option<ListRates>>> = HashMap::new();
    let mut parents: Vec<&String> = partial_rates.keys().collect();
    parents.sort();
    for parent in parents {
        let (Some(kinds), true) = (branch.get(parent), chain.contains_key(parent)) else {
            continue;
        };
        let tallied_kinds = &partial_rates[parent];
        let mut names: Vec<&String> = tallied_kinds.keys().collect();
        names.sort();
        for kind in names {
            let Some(curves) = kinds.get(kind) else {
                continue;
            };
            let Some(rule) = ListRule::new(parent, kind, curves)? else {
                continue;
            };
            let per_target = &tallied_kinds[kind];
            // Each target's tallied production on its first curve, so the
            // per-target sum in `resolve_list` counts it once.
            let mut production = vec![0.0; rule.curves.len()];
            for (t, p) in per_target {
                if let Some(k) = rule
                    .curves
                    .iter()
                    .zip(&rule.produces)
                    .position(|(c, produces)| *produces && c.target == *t)
                {
                    production[k] += p;
                }
            }
            let total = if kind == INELASTIC {
                diagnostics.and_then(|d| d.inelastic_totals.get(parent).copied())
            } else {
                rates.get(parent).and_then(|r| r.get(kind)).copied()
            };
            let (clipped, extrapolated) = diagnostics
                .and_then(|d| d.lists.get(parent))
                .and_then(|k| k.get(kind))
                .copied()
                .unwrap_or((0.0, 0.0));
            // The tally scores a list only where the parent has the transport
            // cross section, so every list but `(n,n')` has its total; a
            // missing rate is a zero one.
            let has_total = kind != INELASTIC || total.is_some();
            folded
                .entry(parent.clone())
                .or_default()
                .push(Some(ListRates {
                    production,
                    to_rate: 1.0,
                    total: total.unwrap_or(0.0),
                    clipped,
                    extrapolated,
                    has_total,
                }));
            lists.entry(parent.clone()).or_default().push(rule);
        }
    }
    let (folded_chain, mut report) = fold_branching_into_chain(chain, &lists, &folded, rates)?;
    if let Some(d) = diagnostics {
        report.unmodelled_mt5 = measure_unmodelled_mt5(rates, &d.mt5);
    }
    Ok((folded_chain, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branching_rule::curve_interp;
    use std::collections::HashMap;
    use yamc_materials::material::Material;
    use yamc_nuclide::nuclide::Nuclide;
    use yamc_nuclide::reaction::Reaction;
    use yani::{BranchCurve, BranchQuantity, BranchState, ChainNuclide, ChainReaction};

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

    /// The converter's facts for one state: `complete` says whether its list
    /// names the ground state.
    fn facts(mt: i32, lfs: i32, complete: bool) -> Arc<[BranchState]> {
        Arc::from(vec![BranchState {
            mt,
            lfs,
            lmf: Some(10),
            list_complete: complete,
            level_route: if lfs == 0 { "ground" } else { "energy" }.to_string(),
            level_energy: 0.0,
            level_energy_difference: Some(0.0),
            mf3_cross_section: None,
        }])
    }

    fn curve(
        target: &str,
        quantity: BranchQuantity,
        energy: &[f64],
        values: &[f64],
        complete: bool,
    ) -> BranchCurve {
        let lfs = if target.contains("_m") { 1 } else { 0 };
        BranchCurve {
            target: target.to_string(),
            quantity,
            energy: energy.to_vec(),
            values: values.to_vec(),
            states: facts(0, lfs, complete),
            normalisation: None,
        }
    }

    /// The rates, the folded chain and the report of one collapse.
    type Folded = (
        ReactionRates,
        Arc<HashMap<String, ChainNuclide>>,
        BranchingReport,
    );

    /// Collapse and fold one material against one spectrum, as a solve does.
    fn fold(
        material: &Material,
        chain: &Arc<HashMap<String, ChainNuclide>>,
        branch: &BranchTable,
        spectrum: &MultigroupSpectrum,
    ) -> Result<Folded, String> {
        let lists = build_lists(chain, branch)?;
        let ((rates, _, folded), _, report) =
            collapse_and_fold(material, spectrum, chain, &lists, None)?;
        Ok((rates, folded, report))
    }

    fn one_group(lo: f64, hi: f64) -> MultigroupSpectrum {
        MultigroupSpectrum {
            boundaries: vec![lo, hi],
            masses: vec![1.0],
            flux_error: None,
        }
    }

    fn nuclide_entry(name: &str, reactions: Vec<ChainReaction>) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life: None,
            decay_energy: 0.0,
            reactions,
            decays: vec![],
            fission_yields: None,
            sources: Vec::new(),
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
        }
    }

    fn edge(kind: &str, target: &str, branching: f64) -> ChainReaction {
        ChainReaction {
            kind: kind.to_string(),
            target: Some(target.to_string()),
            branching,
            q_value: None,
        }
    }

    /// A single-parent chain with a grafted `(n,n')` metastable channel and a
    /// flat 0.1 barn partial cross section folds to the expected rate, and the
    /// grafted channel's branching becomes 1.0 (single metastable target),
    /// from the partial alone where the parent has no cross sections.
    #[test]
    fn fold_nnprime_injects_rate_and_branching() {
        let chain = Arc::new(HashMap::from([(
            "Pb204".to_string(),
            nuclide_entry("Pb204", vec![edge("(n,n')", "Pb204_m1", 1.0)]),
        )]));
        let mut branch: BranchTable = BranchTable::new();
        branch.entry("Pb204".to_string()).or_default().insert(
            "(n,n')".to_string(),
            vec![curve(
                "Pb204_m1",
                BranchQuantity::CrossSection,
                &[1.0, 1.0e8],
                &[0.1, 0.1],
                true,
            )],
        );
        let (rates, folded, report) =
            fold(&dummy_material(), &chain, &branch, &one_group(1.0, 1.0e8)).unwrap();
        let rate = rates["Pb204"]["(n,n')"];
        assert!((rate - 0.1 * 1.0e-24).abs() < 1e-40, "got {rate}");
        let br = folded["Pb204"].reactions[0].branching;
        assert!((br - 1.0).abs() < 1e-12, "got {br}");
        let channel = &report.channels[0];
        assert_eq!(channel.representation, "absolute");
        assert_eq!(channel.denominator, "transport total");
        assert_eq!(channel.states[0].level_route, vec!["energy".to_string()]);
    }

    /// With no branching overlay the original chain Arc is returned unchanged.
    #[test]
    fn fold_no_overlay_is_identity() {
        let chain = Arc::new(HashMap::new());
        let (rates, folded, report) = fold(
            &dummy_material(),
            &chain,
            &BranchTable::new(),
            &one_group(1.0, 1.0e8),
        )
        .unwrap();
        assert!(Arc::ptr_eq(&chain, &folded), "empty overlay must not clone");
        assert!(rates.is_empty());
        assert!(report.is_empty());
    }

    /// A branching file written before the converter stored its facts cannot
    /// say what a partial means, so it is refused rather than guessed at.
    #[test]
    fn a_list_without_facts_is_refused() {
        let chain = indium_chain();
        let mut branch = BranchTable::new();
        let mut legacy = curve(
            "In114_m1",
            BranchQuantity::CrossSection,
            &[1.0e7, 2.0e7],
            &[0.0, 1.5],
            false,
        );
        legacy.states = Arc::from(Vec::new());
        branch
            .entry("In115".to_string())
            .or_default()
            .insert("(n,2n)".to_string(), vec![legacy]);
        let err = build_lists(&chain, &branch).err().expect("refused");
        assert!(err.contains("carries no list facts"), "{err}");
    }

    /// A chain with a two-target `(n,2n)` split for direct-rate tests
    /// (issue #218): base branching 0.7 ground / 0.3 metastable.
    fn split_chain() -> Arc<HashMap<String, ChainNuclide>> {
        Arc::new(HashMap::from([(
            "X".to_string(),
            nuclide_entry(
                "X",
                vec![edge("(n,2n)", "X_g", 0.7), edge("(n,2n)", "X_m1", 0.3)],
            ),
        )]))
    }

    /// X's `(n,2n)` as a complete MF=10 list naming both states.
    fn split_branch() -> BranchTable {
        let mut branch = BranchTable::new();
        branch.entry("X".to_string()).or_default().insert(
            "(n,2n)".to_string(),
            vec![
                curve(
                    "X_g",
                    BranchQuantity::CrossSection,
                    &[1.0, 1.0e8],
                    &[1.0, 1.0],
                    true,
                ),
                curve(
                    "X_m1",
                    BranchQuantity::CrossSection,
                    &[1.0, 1.0e8],
                    &[1.0, 1.0],
                    true,
                ),
            ],
        );
        branch
    }

    fn coupled(
        chain: &Arc<HashMap<String, ChainNuclide>>,
        branch: &BranchTable,
        partials: &PartialRates,
        rates: &mut ReactionRates,
    ) -> Arc<HashMap<String, ChainNuclide>> {
        apply_coupled_branching(chain, branch, partials, rates, None)
            .unwrap()
            .0
    }

    /// Directly-scored `(n,n')` partial rates inject the summed rate and set
    /// the grafted branching, mirroring the fold's semantics (issue #218).
    #[test]
    fn apply_partials_nnprime_injects_rate_and_branching() {
        let chain = Arc::new(HashMap::from([(
            "Pb204".to_string(),
            nuclide_entry("Pb204", vec![edge("(n,n')", "Pb204_m1", 1.0)]),
        )]));
        let mut branch = BranchTable::new();
        branch.entry("Pb204".to_string()).or_default().insert(
            "(n,n')".to_string(),
            vec![curve(
                "Pb204_m1",
                BranchQuantity::CrossSection,
                &[1.0, 1.0e8],
                &[0.1, 0.1],
                true,
            )],
        );
        let mut partials: PartialRates = HashMap::new();
        partials.entry("Pb204".to_string()).or_default().insert(
            "(n,n')".to_string(),
            vec![("Pb204_m1".to_string(), 1.0e-25)],
        );
        let mut rates: ReactionRates = HashMap::new();
        let folded = coupled(&chain, &branch, &partials, &mut rates);
        let rate = rates["Pb204"]["(n,n')"];
        assert!((rate - 1.0e-25).abs() < 1e-40, "got {rate}");
        let br = folded["Pb204"].reactions[0].branching;
        assert!((br - 1.0).abs() < 1e-12, "got {br}");
    }

    fn branching_in(chain: &HashMap<String, ChainNuclide>, parent: &str, target: &str) -> f64 {
        chain[parent]
            .reactions
            .iter()
            .find(|r| r.target.as_deref() == Some(target))
            .unwrap()
            .branching
    }

    /// Productions of a complete list re-partition the base branching mass by
    /// their shares: 3:1 on a 0.7/0.3 base split become 0.75/0.25.
    #[test]
    fn apply_partials_repartitions_split() {
        let chain = split_chain();
        let mut partials: PartialRates = HashMap::new();
        partials.entry("X".to_string()).or_default().insert(
            "(n,2n)".to_string(),
            vec![("X_g".to_string(), 3.0e-24), ("X_m1".to_string(), 1.0e-24)],
        );
        let tallied = HashMap::from([(
            "X".to_string(),
            HashMap::from([("(n,2n)".to_string(), 4.0e-24)]),
        )]);
        let mut rates = tallied.clone();
        let folded = coupled(&chain, &split_branch(), &partials, &mut rates);
        assert_eq!(rates, tallied, "non-(n,n') kinds must not inject rates");
        assert!((branching_in(&folded, "X", "X_g") - 0.75).abs() < 1e-12);
        assert!((branching_in(&folded, "X", "X_m1") - 0.25).abs() < 1e-12);
    }

    /// All-zero productions (flux never reached the thresholds) keep the base
    /// split; nothing is cloned.
    #[test]
    fn apply_partials_zero_total_keeps_base_split() {
        let chain = split_chain();
        let mut partials: PartialRates = HashMap::new();
        partials.entry("X".to_string()).or_default().insert(
            "(n,2n)".to_string(),
            vec![("X_g".to_string(), 0.0), ("X_m1".to_string(), 0.0)],
        );
        let mut rates: ReactionRates = HashMap::from([(
            "X".to_string(),
            HashMap::from([("(n,2n)".to_string(), 1.0e-24)]),
        )]);
        let folded = coupled(&chain, &split_branch(), &partials, &mut rates);
        assert!(Arc::ptr_eq(&chain, &folded), "zero rates must not refine");
    }

    /// Duplicate curves for the same (kind, target), which really occur in
    /// TENDL branching data (two LFS levels mapped to one chain nuclide), must
    /// have their rates summed into the fraction, not overwrite each other.
    #[test]
    fn apply_partials_sums_duplicate_targets() {
        let chain = split_chain();
        let mut branch = split_branch();
        let list = branch.get_mut("X").unwrap().get_mut("(n,2n)").unwrap();
        let duplicate = list[0].clone();
        list.insert(1, duplicate);
        let mut partials: PartialRates = HashMap::new();
        partials.entry("X".to_string()).or_default().insert(
            "(n,2n)".to_string(),
            vec![
                ("X_g".to_string(), 1.0e-24),
                ("X_g".to_string(), 2.0e-24), // duplicate target
                ("X_m1".to_string(), 1.0e-24),
            ],
        );
        let mut rates: ReactionRates = HashMap::from([(
            "X".to_string(),
            HashMap::from([("(n,2n)".to_string(), 4.0e-24)]),
        )]);
        let folded = coupled(&chain, &branch, &partials, &mut rates);
        // Ground share = (1 + 2) / 4 of the base mass 1.0.
        assert!((branching_in(&folded, "X", "X_g") - 0.75).abs() < 1e-12);
        assert!((branching_in(&folded, "X", "X_m1") - 0.25).abs() < 1e-12);
    }

    /// Empty partial rates return the original chain Arc untouched.
    #[test]
    fn apply_partials_empty_is_identity() {
        let chain = split_chain();
        let mut rates: ReactionRates = HashMap::new();
        let folded = coupled(&chain, &split_branch(), &HashMap::new(), &mut rates);
        assert!(Arc::ptr_eq(&chain, &folded));
        assert!(rates.is_empty());
    }

    /// A transport reaction on its own grid, for the folds that divide by one.
    fn reaction(mt: i32, energy: Vec<f64>, cross_section: Vec<f64>) -> Reaction {
        Reaction {
            cross_section: cross_section.into(),
            threshold_idx: 0,
            energy: energy.into(),
            mt_number: mt,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        }
    }

    /// In115 with only the given transport reactions loaded, at 294 K.
    fn indium(reactions: Vec<Reaction>) -> Material {
        let temperature = "294".to_string();
        let by_mt: HashMap<i32, Arc<Reaction>> = reactions
            .into_iter()
            .map(|r| (r.mt_number, Arc::new(r)))
            .collect();
        let nuclide = Nuclide {
            name: Some("In115".to_string()),
            element: None,
            atomic_symbol: Some("In".to_string()),
            atomic_number: Some(49),
            neutron_number: Some(66),
            mass_number: Some(115),
            atomic_weight_ratio: Some(113.9),
            library: None,
            energy: None,
            reactions: vec![by_mt],
            fissionable: false,
            available_temperatures: vec![temperature.clone()],
            loaded_temperatures: vec![temperature.clone()],
            data_path: None,
            fission_nu: None,
            fast_xs: vec![],
            urr_data: vec![],
            urr_present: false,
            fission_photon_release: None,
            covariance: None,
            elastic_flat_cache: Default::default(),
            fission_chi_flat_cache: Default::default(),
            delayed_neutron_cache: Default::default(),
            inelastic_angle_flat_cache: Default::default(),
            load_scope: Default::default(),
        };
        let mut m = Material::new(
            HashMap::from([("In115".to_string(), 1.0e-3)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        m.set_temperature(&temperature);
        m.nuclide_data
            .insert("In115".to_string(), Arc::new(nuclide));
        m
    }

    /// In115 as the ENDF/B-VIII.1 overlay leaves it: each reaction's ground
    /// state at 1.0 from the base chain and its isomer grafted at 0.0.
    fn indium_chain() -> Arc<HashMap<String, ChainNuclide>> {
        Arc::new(HashMap::from([(
            "In115".to_string(),
            nuclide_entry(
                "In115",
                vec![
                    edge("(n,gamma)", "In116", 1.0),
                    edge("(n,gamma)", "In116_m1", 0.0),
                    edge("(n,2n)", "In114", 1.0),
                    edge("(n,2n)", "In114_m1", 0.0),
                ],
            ),
        )]))
    }

    fn branching_of(chain: &HashMap<String, ChainNuclide>, kind: &str, target: &str) -> f64 {
        chain["In115"]
            .reactions
            .iter()
            .find(|r| r.kind == kind && r.target.as_deref() == Some(target))
            .unwrap()
            .branching
    }

    fn indium_branch(kind: &str, curves: Vec<BranchCurve>) -> BranchTable {
        let mut branch = BranchTable::new();
        branch
            .entry("In115".to_string())
            .or_default()
            .insert(kind.to_string(), curves);
        branch
    }

    /// ENDF/B-VIII.1 In115 (n,gamma): MF=9 lists In116_m1 alone, at a flat
    /// 0.79. The share is the weighted yield itself, whatever shape the capture
    /// cross section has, and In116 keeps the rest.
    #[test]
    fn an_isomer_only_yield_is_its_weighted_yield() {
        let material = indium(vec![reaction(102, vec![1.0e-5, 1.0e7], vec![100.0, 1.0])]);
        let branch = indium_branch(
            "(n,gamma)",
            vec![curve(
                "In116_m1",
                BranchQuantity::Yield,
                &[1.0e-5, 2.0e7],
                &[0.79, 0.79],
                false,
            )],
        );
        let spectrum = MultigroupSpectrum {
            boundaries: vec![1.0e-5, 1.0, 1.0e3, 1.0e7],
            masses: vec![0.5, 0.3, 0.2],
            flux_error: None,
        };
        let (rates, folded, report) = fold(&material, &indium_chain(), &branch, &spectrum).unwrap();
        let m = branching_of(&folded, "(n,gamma)", "In116_m1");
        let g = branching_of(&folded, "(n,gamma)", "In116");
        assert!((m - 0.79).abs() < 1e-12, "In116_m1 {m}");
        assert!((g - 0.21).abs() < 1e-12, "In116 {g}");
        assert_eq!(
            rates["In115"].len(),
            1,
            "only the split moves, never the rate"
        );
        let channel = &report.channels[0];
        assert_eq!(
            (
                channel.file,
                channel.representation.as_str(),
                channel.complete
            ),
            (9, "share", false)
        );
        assert_eq!(channel.denominator, "transport total");
    }

    /// An energy-dependent yield is weighted by the flux and the cross section
    /// and not renormalized: a linear 0.2 to 0.6 yield under a flat cross
    /// section averages 0.3 and 0.5 over two groups holding 1/4 and 3/4 of the
    /// flux, 0.45 in all.
    #[test]
    fn an_isomer_only_yield_is_weighted_not_renormalized() {
        let material = indium(vec![reaction(102, vec![0.0, 1.0e8], vec![2.0, 2.0])]);
        let branch = indium_branch(
            "(n,gamma)",
            vec![curve(
                "In116_m1",
                BranchQuantity::Yield,
                &[0.0, 1.0e8],
                &[0.2, 0.6],
                false,
            )],
        );
        let spectrum = MultigroupSpectrum {
            boundaries: vec![0.0, 5.0e7, 1.0e8],
            masses: vec![0.25, 0.75],
            flux_error: None,
        };
        let (_, folded, _) = fold(&material, &indium_chain(), &branch, &spectrum).unwrap();
        let m = branching_of(&folded, "(n,gamma)", "In116_m1");
        let g = branching_of(&folded, "(n,gamma)", "In116");
        assert!((m - 0.45).abs() < 1e-12, "In116_m1 {m}");
        assert!((g - 0.55).abs() < 1e-12, "In116 {g}");
    }

    /// ENDF/B-VIII.1 In115 (n,2n): MF=10 lists In114_m1 alone. Its share is
    /// its partial rate over the transport total's, here a partial at 3/4 of
    /// the (n,2n) cross section, and In114 keeps the other quarter.
    #[test]
    fn an_isomer_only_partial_is_a_share_of_the_transport_total() {
        let material = indium(vec![reaction(16, vec![1.0e7, 2.0e7], vec![0.0, 2.0])]);
        let branch = indium_branch(
            "(n,2n)",
            vec![curve(
                "In114_m1",
                BranchQuantity::CrossSection,
                &[1.0e7, 2.0e7],
                &[0.0, 1.5],
                false,
            )],
        );
        let spectrum = MultigroupSpectrum {
            boundaries: vec![1.0e6, 1.2e7, 2.0e7],
            masses: vec![0.5, 0.5],
            flux_error: None,
        };
        let (_, folded, report) = fold(&material, &indium_chain(), &branch, &spectrum).unwrap();
        let m = branching_of(&folded, "(n,2n)", "In114_m1");
        let g = branching_of(&folded, "(n,2n)", "In114");
        assert!((m - 0.75).abs() < 1e-12, "In114_m1 {m}");
        assert!((g - 0.25).abs() < 1e-12, "In114 {g}");
        assert_eq!(report.channels[0].representation, "absolute");

        // Without the transport total there is nothing to take a share of, so
        // the base split stands, and the report says why.
        let (_, folded, report) =
            fold(&indium(vec![]), &indium_chain(), &branch, &spectrum).unwrap();
        assert_eq!(branching_of(&folded, "(n,2n)", "In114_m1"), 0.0);
        assert_eq!(branching_of(&folded, "(n,2n)", "In114"), 1.0);
        assert!(
            report.dropped[0].reason.contains("no MT=16"),
            "{:?}",
            report.dropped
        );
    }

    /// In115 (n,2n) folded over `spectrum` from one isomer-only partial and
    /// the given transport total, as `(In114_m1, In114)`.
    fn n2n_split(
        total: (&[f64], &[f64]),
        partial: (&[f64], &[f64]),
        spectrum: MultigroupSpectrum,
    ) -> Result<(f64, f64, BranchingReport), String> {
        let material = indium(vec![reaction(16, total.0.to_vec(), total.1.to_vec())]);
        let branch = indium_branch(
            "(n,2n)",
            vec![curve(
                "In114_m1",
                BranchQuantity::CrossSection,
                partial.0,
                partial.1,
                false,
            )],
        );
        let (_, folded, report) = fold(&material, &indium_chain(), &branch, &spectrum)?;
        Ok((
            branching_of(&folded, "(n,2n)", "In114_m1"),
            branching_of(&folded, "(n,2n)", "In114"),
            report,
        ))
    }

    /// The share is a ratio of rate integrals, not an average of ratios. The
    /// partial runs 0.3 to 1.5 b under a total of 1 to 3 b, so its ratio to the
    /// total goes from 0.3 to 0.5, and two groups hold 1/4 and 3/4 of the flux.
    /// Group averages 0.6 and 1.2 b over 1.5 and 2.5 b give
    /// `(0.25 * 0.6 + 0.75 * 1.2) / (0.25 * 1.5 + 0.75 * 2.5) = 7/15`, where a
    /// flux-weighted average of the per-group ratios would give 0.46.
    #[test]
    fn an_isomer_only_partial_is_a_ratio_of_rate_integrals() {
        let (m, g, _) = n2n_split(
            (&[1.0e7, 2.0e7], &[1.0, 3.0]),
            (&[1.0e7, 2.0e7], &[0.3, 1.5]),
            MultigroupSpectrum {
                boundaries: vec![1.0e7, 1.5e7, 2.0e7],
                masses: vec![0.25, 0.75],
                flux_error: None,
            },
        )
        .unwrap();
        assert!((m - 7.0 / 15.0).abs() < 1e-12, "In114_m1 {m}");
        assert!((g - 8.0 / 15.0).abs() < 1e-12, "In114 {g}");
    }

    /// Past its last energy an isomer-only partial follows the transport total
    /// at the share it ends on, not its own last value, and that part of the
    /// production is the evaluation's fraction held, not its data: reported,
    /// and refused once it is more than the tolerance of the parent's removal.
    /// Here the partial ends at 20 MeV on 1.6 b, 0.8 of a 2 b total that then
    /// falls to nothing by 30 MeV, as ENDF/B-VIII.1's In115 (n,2n) partial
    /// against TENDL-2025's total does.
    #[test]
    fn an_isomer_only_partial_follows_the_total_past_its_last_energy() {
        let total: (&[f64], &[f64]) = (&[1.0e7, 2.0e7, 3.0e7], &[2.0, 2.0, 0.0]);
        let partial: (&[f64], &[f64]) = (&[1.0e7, 2.0e7], &[1.0, 1.6]);
        let spectrum = |boundaries: Vec<f64>, masses: Vec<f64>| MultigroupSpectrum {
            boundaries,
            masses,
            flux_error: None,
        };
        // Wholly above: every atom of it rests on the held fraction.
        let err = n2n_split(total, partial, spectrum(vec![2.0e7, 3.0e7], vec![1.0]))
            .expect_err("refused");
        assert!(err.contains("In115 (n,2n)"), "{err}");
        assert!(err.contains("tabulates no split"), "{err}");

        // A sliver above: 1e-5 of the flux in 20-30 MeV, where the partial is
        // `0.8 * sigma_MT`. The held production is reported, not refused, and
        // the share is the ratio of the two rate integrals,
        // `(w * 1.3 + v * 0.8 * 1.0) / (w * 2.0 + v * 1.0)`.
        let (w, v) = (1.0 - 1.0e-5, 1.0e-5);
        let (m, _, report) = n2n_split(
            total,
            partial,
            spectrum(vec![1.0e7, 2.0e7, 3.0e7], vec![w, v]),
        )
        .unwrap();
        let want = (w * 1.3 + v * 0.8) / (w * 2.0 + v);
        assert!((m - want).abs() < 1e-12, "In114_m1 {m} against {want}");
        let held = report.channels[0].extrapolated_share;
        let want_held = v * 0.8 / (w * 2.0 + v);
        assert!(
            (held - want_held).abs() < 1e-12 * want_held.max(1e-30),
            "{held}"
        );

        // A total that is zero where the partial ends leaves no share to hold,
        // and the partial ends there.
        let (m, g, _) = n2n_split(
            (&[1.0e7, 2.0e7, 2.5e7], &[2.0, 0.0, 1.0]),
            partial,
            spectrum(vec![2.0e7, 3.0e7], vec![1.0]),
        )
        .unwrap();
        assert!(m < 1e-12, "In114_m1 {m}");
        assert!((g - 1.0).abs() < 1e-12, "In114 {g}");
    }

    /// A partial above the transport total cannot all be made. The rule clips
    /// it at each energy and measures what it clipped against the parent's
    /// removal rate: a Pt-like list, a partial 5.4 times its total (as
    /// ENDF/B-VIII.1's Pt194 (n,d) is at 14 MeV), refuses the run when the
    /// reaction is the parent's removal, and is carried and reported when the
    /// reaction is a sliver of it.
    #[test]
    fn a_partial_above_its_total_refuses_only_when_it_matters() {
        let branch = indium_branch(
            "(n,2n)",
            vec![curve(
                "In114_m1",
                BranchQuantity::CrossSection,
                &[1.0e7, 2.0e7],
                &[5.4, 5.4],
                false,
            )],
        );
        let spectrum = one_group(1.35e7, 1.45e7);
        let alone = indium(vec![reaction(16, vec![1.0e7, 2.0e7], vec![1.0, 1.0])]);
        let err = fold(&alone, &indium_chain(), &branch, &spectrum).expect_err("refused");
        assert!(
            err.contains("above the reaction's transport total"),
            "{err}"
        );
        assert!(err.contains("In115 (n,2n)"), "{err}");

        // The same list beside a capture 1e5 times the (n,2n): the clipped
        // 4.4 b is 4.4e-5 of In115's removal, and is carried and reported.
        let beside = indium(vec![
            reaction(16, vec![1.0e7, 2.0e7], vec![1.0, 1.0]),
            reaction(102, vec![1.0e7, 2.0e7], vec![1.0e5, 1.0e5]),
        ]);
        let (_, folded, report) = fold(&beside, &indium_chain(), &branch, &spectrum).unwrap();
        assert_eq!(branching_of(&folded, "(n,2n)", "In114_m1"), 1.0);
        assert_eq!(branching_of(&folded, "(n,2n)", "In114"), 0.0);
        let channel = report
            .channels
            .iter()
            .find(|c| c.reaction == "(n,2n)")
            .unwrap();
        let want = 4.4 / (1.0 + 1.0e5);
        assert!(
            (channel.clipped_share - want).abs() < 1e-9 * want,
            "{}",
            channel.clipped_share
        );
    }

    /// MT=5, whose products no chain reaction carries, is measured against
    /// each parent's removal and reported: at 30 MeV it is half of In115's
    /// here, and a D-T spectrum below its 20 MeV threshold reports nothing.
    #[test]
    fn mt5_is_reported_against_the_removal() {
        let material = indium(vec![
            reaction(16, vec![1.0e7, 4.0e7], vec![1.0, 1.0]),
            reaction(5, vec![2.0e7, 4.0e7], vec![0.0, 2.0]),
        ]);
        let (_, _, report) = fold(
            &material,
            &indium_chain(),
            &BranchTable::new(),
            &one_group(2.9e7, 3.1e7),
        )
        .unwrap();
        let mt5 = &report.unmodelled_mt5[0];
        assert_eq!(mt5.nuclide, "In115");
        assert!((mt5.share - 0.5).abs() < 1e-3, "{mt5:?}");

        let (_, _, report) = fold(
            &material,
            &indium_chain(),
            &BranchTable::new(),
            &one_group(1.35e7, 1.45e7),
        )
        .unwrap();
        assert!(report.unmodelled_mt5.is_empty(), "{report:?}");
    }

    /// A complete list reproduces the evaluation's own partials where they
    /// add up to the transport total, as they do in a single-library run: each
    /// state's production is its partial folded as a cross section in its own
    /// right, under the same weight, group by group.
    #[test]
    fn a_consistent_complete_list_reproduces_its_partials() {
        let energy = vec![1.0e7, 1.2e7, 1.4e7, 1.7e7, 2.0e7];
        let ground = vec![0.0, 0.4, 0.3, 0.2, 0.1];
        let isomer = vec![0.0, 0.1, 0.5, 0.9, 1.1];
        let total: Vec<f64> = ground.iter().zip(&isomer).map(|(a, b)| a + b).collect();
        let material = indium(vec![reaction(16, energy.clone(), total)]);
        let branch = indium_branch(
            "(n,2n)",
            vec![
                curve(
                    "In114",
                    BranchQuantity::CrossSection,
                    &energy,
                    &ground,
                    true,
                ),
                curve(
                    "In114_m1",
                    BranchQuantity::CrossSection,
                    &energy,
                    &isomer,
                    true,
                ),
            ],
        );
        let spectrum = MultigroupSpectrum {
            boundaries: vec![1.0e7, 1.3e7, 1.5e7, 2.0e7],
            masses: vec![0.2, 0.5, 0.3],
            flux_error: None,
        };
        let (rates, folded, report) = fold(&material, &indium_chain(), &branch, &spectrum).unwrap();
        let own = |values: &[f64]| {
            let partial = reaction(16, energy.clone(), values.to_vec());
            (0..3)
                .map(|g| {
                    crate::multigroup::group_averaged_xs(
                        &partial,
                        spectrum.boundaries[g],
                        spectrum.boundaries[g + 1],
                    ) * spectrum.masses[g]
                })
                .sum::<f64>()
                * 1.0e-24
        };
        let r = rates["In115"]["(n,2n)"];
        for (target, values) in [("In114", &ground), ("In114_m1", &isomer)] {
            let got = branching_of(&folded, "(n,2n)", target) * r;
            let want = own(values);
            assert!(
                (got - want).abs() < 1e-12 * want,
                "{target}: {got} against its own partial's {want}"
            );
        }
        let channel = &report.channels[0];
        assert_eq!(channel.denominator, "sum of the listed partials");
        assert_eq!(channel.representation, "share");
    }

    /// The old fold and the rule agree on a complete MF=9 list whose yields
    /// sum to one: each state's weight is `integral(y_s sigma_MT phi)` in both,
    /// integrated on the same points in the same order, and normalized over
    /// the listed states. So the split is the same to the bit, spelled out
    /// here with the old arithmetic.
    #[test]
    fn a_complete_yield_list_splits_exactly_as_before() {
        let capture = reaction(
            102,
            vec![1.0e-5, 1.0, 1.0e3, 1.0e7],
            vec![300.0, 30.0, 3.0, 0.1],
        );
        let material = indium(vec![capture.clone()]);
        let spectrum = MultigroupSpectrum {
            boundaries: vec![1.0e-5, 0.5, 2.0e3, 1.0e7],
            masses: vec![0.3, 0.3, 0.4],
            flux_error: None,
        };
        let grid = [1.0e-5, 3.0, 5.0e4, 1.0e7];
        let yields = [
            curve("In116", BranchQuantity::Yield, &grid, &[0.25; 4], true),
            curve("In116_m1", BranchQuantity::Yield, &grid, &[0.75; 4], true),
        ];
        let branch = indium_branch("(n,gamma)", yields.to_vec());
        let (_, folded, _) = fold(&material, &indium_chain(), &branch, &spectrum).unwrap();

        // The old integrator: the product of the two linear pieces over the
        // union of the yield and cross-section grids, per group.
        let old_weight = |c: &BranchCurve| {
            let mut num = 0.0;
            for g in 0..spectrum.masses.len() {
                let (e_lo, e_hi) = (spectrum.boundaries[g], spectrum.boundaries[g + 1]);
                let mut points: Vec<f64> = c
                    .energy
                    .iter()
                    .chain(capture.energy.iter())
                    .copied()
                    .filter(|&e| e > e_lo && e < e_hi)
                    .collect();
                points.sort_by(f64::total_cmp);
                points.push(e_hi);
                let sigma = |e: f64| capture.cross_section_at(e).unwrap_or(0.0);
                let mut integral = 0.0;
                let (mut prev_e, mut prev_y, mut prev_s) =
                    (e_lo, curve_interp(&c.energy, &c.values, e_lo), sigma(e_lo));
                for e in points {
                    let (y, s) = (curve_interp(&c.energy, &c.values, e), sigma(e));
                    integral += (e - prev_e)
                        * (2.0 * prev_y * prev_s + prev_y * s + y * prev_s + 2.0 * y * s)
                        / 6.0;
                    (prev_e, prev_y, prev_s) = (e, y, s);
                }
                num += integral / (e_hi - e_lo) * spectrum.masses[g];
            }
            num
        };
        let weights = [old_weight(&yields[0]), old_weight(&yields[1])];
        let total = weights[0] + weights[1];
        let f = [weights[0] / total, weights[1] / total];
        let f_sum = f[0] + f[1];
        let expected = [(f[0] / f_sum) * 1.0, (f[1] / f_sum) * 1.0];
        assert_eq!(
            branching_of(&folded, "(n,gamma)", "In116").to_bits(),
            expected[0].to_bits()
        );
        assert_eq!(
            branching_of(&folded, "(n,gamma)", "In116_m1").to_bits(),
            expected[1].to_bits()
        );
    }

    /// The coupled path: a tallied production for the isomer alone is a share
    /// of the reaction's tallied total, MF=9 and MF=10 alike, and the ground
    /// state keeps the rest.
    #[test]
    fn coupled_isomer_only_rates_are_shares_of_the_tallied_total() {
        let chain = indium_chain();
        let mut branch = indium_branch(
            "(n,gamma)",
            vec![curve(
                "In116_m1",
                BranchQuantity::Yield,
                &[1.0e-5, 2.0e7],
                &[0.79, 0.79],
                false,
            )],
        );
        branch.get_mut("In115").unwrap().insert(
            "(n,2n)".to_string(),
            vec![curve(
                "In114_m1",
                BranchQuantity::CrossSection,
                &[1.0e7, 2.0e7],
                &[0.0, 1.8],
                false,
            )],
        );
        let mut partials: PartialRates = HashMap::new();
        let kinds = partials.entry("In115".to_string()).or_default();
        kinds.insert(
            "(n,gamma)".to_string(),
            vec![("In116_m1".to_string(), 0.79 * 4.0e-22)],
        );
        kinds.insert(
            "(n,2n)".to_string(),
            vec![("In114_m1".to_string(), 0.9 * 2.0e-25)],
        );
        let totals = HashMap::from([(
            "In115".to_string(),
            HashMap::from([
                ("(n,gamma)".to_string(), 4.0e-22),
                ("(n,2n)".to_string(), 2.0e-25),
            ]),
        )]);
        let mut rates = totals.clone();
        let folded = coupled(&chain, &branch, &partials, &mut rates);
        assert_eq!(rates, totals, "only the split moves, never the rate");
        let near = |kind: &str, target: &str, want: f64| {
            let got = branching_of(&folded, kind, target);
            assert!((got - want).abs() < 1e-12, "{kind} {target}: {got}");
        };
        near("(n,gamma)", "In116_m1", 0.79);
        near("(n,gamma)", "In116", 0.21);
        near("(n,2n)", "In114_m1", 0.9);
        near("(n,2n)", "In114", 0.1);

        // No tallied total, no reaction to split: the chain is left alone.
        let mut rates: ReactionRates = HashMap::new();
        let folded = coupled(&chain, &branch, &partials, &mut rates);
        assert!(Arc::ptr_eq(&chain, &folded));
    }

    /// Isomer-only partials that are all zero under a live reaction, the flux
    /// lying between the total's threshold and the partial's, give the isomer
    /// nothing and the ground all of the reaction, on both paths and whatever
    /// split the base chain carried. A full list of zeros, and a reaction with
    /// no tallied total, keep it.
    #[test]
    fn zero_isomer_only_rates_give_the_ground_the_reaction() {
        let mut chain = (*indium_chain()).clone();
        for rx in chain
            .get_mut("In115")
            .unwrap()
            .reactions
            .iter_mut()
            .filter(|r| r.kind == "(n,2n)")
        {
            rx.branching = 0.5;
        }
        let chain = Arc::new(chain);
        let ground_takes_all = |folded: &HashMap<String, ChainNuclide>| {
            assert_eq!(branching_of(folded, "(n,2n)", "In114_m1"), 0.0);
            assert_eq!(branching_of(folded, "(n,2n)", "In114"), 1.0);
        };

        // The spectrum path: one group at 11 to 13 MeV, above the total's
        // threshold and below the partial's.
        let material = indium(vec![reaction(16, vec![1.0e7, 3.0e7], vec![2.0, 2.0])]);
        let isomer_only = indium_branch(
            "(n,2n)",
            vec![curve(
                "In114_m1",
                BranchQuantity::CrossSection,
                &[1.5e7, 2.0e7],
                &[0.0, 1.6],
                false,
            )],
        );
        let (_, folded, _) =
            fold(&material, &chain, &isomer_only, &one_group(1.1e7, 1.3e7)).unwrap();
        ground_takes_all(&folded);

        // The coupled path, over the same reaction's tallied total.
        let only = |targets: &[&str]| -> PartialRates {
            let rates = targets.iter().map(|t| (t.to_string(), 0.0)).collect();
            HashMap::from([(
                "In115".to_string(),
                HashMap::from([("(n,2n)".to_string(), rates)]),
            )])
        };
        let tallied = || -> ReactionRates {
            HashMap::from([(
                "In115".to_string(),
                HashMap::from([("(n,2n)".to_string(), 2.0e-24)]),
            )])
        };
        ground_takes_all(&coupled(
            &chain,
            &isomer_only,
            &only(&["In114_m1"]),
            &mut tallied(),
        ));
        let complete = indium_branch(
            "(n,2n)",
            vec![
                curve(
                    "In114",
                    BranchQuantity::CrossSection,
                    &[1.5e7, 2.0e7],
                    &[0.0, 0.4],
                    true,
                ),
                curve(
                    "In114_m1",
                    BranchQuantity::CrossSection,
                    &[1.5e7, 2.0e7],
                    &[0.0, 1.6],
                    true,
                ),
            ],
        );
        let full = only(&["In114", "In114_m1"]);
        assert!(Arc::ptr_eq(
            &chain,
            &coupled(&chain, &complete, &full, &mut tallied())
        ));
        let mut untallied: ReactionRates = HashMap::new();
        assert!(Arc::ptr_eq(
            &chain,
            &coupled(&chain, &isomer_only, &only(&["In114_m1"]), &mut untallied)
        ));
    }
}
