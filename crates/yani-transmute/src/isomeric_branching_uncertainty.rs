//! The uncertainty of an isomeric split, from MF=40.
//!
//! The branching overlay splits a reaction between its product states by
//! folding each state's MF=10 partial cross section against the spectrum, and
//! MF=40 is the covariance of those partials. The two meet here: MF=40 is
//! folded against the same spectrum exactly as MF=33 is (`covariance_fold`),
//! giving the covariance of the partial RATES, and each replica perturbs those
//! rates and turns them into fractions the way the nominal fold does. The
//! fractions then split the reaction's transport total, which MF=33 perturbs
//! on its own, so the cross sections say how much reacts and MF=40 says how it
//! splits. `(n,n')` has no transport total, its rate being the sum of its
//! partials, so there MF=40 moves the production rate too.
//!
//! # Labels are fixed per parent
//!
//! A parent's partials are sampled as one vector, one deviate per partial,
//! labelled `"kind target"`. The labels are every partial the branch table has
//! a usable MF=40 block for in a list with a split to move, sorted, and they do
//! not depend on the spectrum: a partial below threshold in one spectrum keeps
//! its place with a zero row. Nor on the material: an isomer-only list whose
//! transport total a material does not load keeps its places the same way. That
//! is what makes one MF=40 partial move together across the spectra of a
//! schedule, as one evaluation must, the way MF=33's kinds come from the
//! chain's topology rather than from which rates are non-zero.
//!
//! # What the data do not say
//!
//! Every block in TENDL-2017, TENDL-2025 and ENDF/B-VIII.1 is a state's
//! covariance with itself, so the partials are sampled independently of one
//! another. Independent partials give a split a relative sigma of about
//! `(1 - f) sqrt(s_g^2 + s_m^2)`, an upper bound if the true ground-isomer
//! correlation is positive. No library publishes an MF=40 x MF=33 correlation
//! either, so the two sources draw from separate streams.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

use yamc_materials::Material;
use yamc_nuclide::covariance::expand::{expand_ni, ExpandedBlock, Unsupported};
use yamc_nuclide::covariance::{BranchingCovarianceBlock, CovarianceData};
use yani::{BranchQuantity, BranchTable, ChainNuclide, ReactionRates};

use crate::covariance_fold::{fold_partial_covariance, PartialBlock};
use crate::covariance_sample::Sampler;
use crate::material_transmute::{
    along_the_total, build_fold_refine, fold_curve_rate, remainder_state, transport_reaction,
    Fractions, MultigroupSpectrum, PerSpectrum, Split,
};
use crate::transmutation_tallies::PartialRates;

/// Keeps the MF=40 streams clear of the cross-section streams, which are keyed
/// on the same parent names without a tag.
pub(crate) const ISOMERIC_BRANCHING_STREAM: u32 = 0x150B_4A7C;

/// A branching subsection's MF=40 blocks, by parent and then reaction kind.
pub(crate) struct BranchingCovariance {
    by_parent: HashMap<String, BTreeMap<String, Vec<BranchingCovarianceBlock>>>,
}

impl BranchingCovariance {
    pub(crate) fn from_blocks(blocks: Vec<BranchingCovarianceBlock>) -> Self {
        let mut by_parent: HashMap<String, BTreeMap<String, Vec<BranchingCovarianceBlock>>> =
            HashMap::new();
        for b in blocks {
            by_parent
                .entry(b.nuclide.clone())
                .or_default()
                .entry(b.reaction.clone())
                .or_default()
                .push(b);
        }
        BranchingCovariance { by_parent }
    }
}

/// Read `branching_covariance.arrow`, once per process per path.
///
/// Every material of a run, and every source-alone rerun of an attribution,
/// asks for the same file, and it runs to tens of megabytes for a TENDL
/// library. Keyed by path alone, so a file rewritten under the same name in
/// one process is not reread.
pub(crate) fn load_branching_covariance(path: &Path) -> Result<Arc<BranchingCovariance>, String> {
    static CACHE: OnceLock<RwLock<HashMap<PathBuf, Arc<BranchingCovariance>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(hit) = cache.read().unwrap_or_else(|p| p.into_inner()).get(path) {
        return Ok(Arc::clone(hit));
    }
    let blocks =
        yamc_nuclide::arrow::covariance_arrow::read_branching_covariance(path).map_err(|e| {
            format!(
                "could not read the isomeric branching covariance {}: {e}. It is read \
                 because data_uncertainty asked for the \"isomeric_branching\" source; \
                 leave that source out to run without it.",
                path.display()
            )
        })?;
    let loaded = Arc::new(BranchingCovariance::from_blocks(blocks));
    cache
        .write()
        .unwrap_or_else(|p| p.into_inner())
        .insert(path.to_path_buf(), Arc::clone(&loaded));
    Ok(loaded)
}

fn label(kind: &str, target: &str) -> String {
    format!("{kind} {target}")
}

/// The curve a block's state is weighted by: its own partial where its target
/// sums several states, else the label's fold curves.
fn pieces<'a>(
    own: &'a Option<(Vec<f64>, Vec<f64>)>,
    curves: &'a [(Vec<f64>, Vec<f64>)],
) -> Vec<(&'a [f64], &'a [f64])> {
    match own {
        Some((e, v)) => vec![(e.as_slice(), v.as_slice())],
        None => curves
            .iter()
            .map(|(e, v)| (e.as_slice(), v.as_slice()))
            .collect(),
    }
}

/// How a reaction's partials become its split, mirroring the nominal fold.
enum Form {
    /// `(n,n')`: shares of the partials' own sum, which is also the rate.
    Inelastic,
    /// Shares among the listed states, of the mass they carry.
    Among,
    /// Metastable states only: each a share of the transport total, whose
    /// curve this is, and the ground state the rest.
    OfWhole {
        ground: String,
        whole: (Vec<f64>, Vec<f64>),
    },
    /// Metastable states only, with no transport total loaded to take shares
    /// of. The nominal fold leaves the chain's split in place and so does
    /// every replica, but the states keep their labels, at no rate, so which
    /// deviate a parent's partial draws does not depend on the material.
    NoTotal,
}

/// One product state of a reaction, with the curves the fold reads for it:
/// one normally, several where the table carries duplicates for one target,
/// each continued along the transport total for an isomer-only list as the
/// fold continues it, and none where there is no total to continue along.
struct State {
    target: String,
    label: String,
    curves: Vec<(Vec<f64>, Vec<f64>)>,
}

/// One reaction of a parent whose split is made of MF=10 partials.
struct Channel {
    kind: String,
    form: Form,
    /// Every partial of the reaction, in the branch table's order. One with no
    /// MF=40 keeps its nominal rate and still counts in the split.
    states: Vec<State>,
}

/// An MF=40 block the fold can use, with its two states as label indices.
struct UsableBlock {
    row: usize,
    col: usize,
    row_lfs: i32,
    col_lfs: i32,
    /// A state's own partial, where its target sums several states.
    row_own: Option<(Vec<f64>, Vec<f64>)>,
    col_own: Option<(Vec<f64>, Vec<f64>)>,
    expanded: ExpandedBlock,
}

/// Everything about one parent that does not depend on the spectrum.
struct Parent {
    /// Sorted `"kind target"`, one per partial with a usable block.
    labels: Vec<String>,
    /// `(kind, target)` of each label.
    keys: Vec<(String, String)>,
    /// The fold curves of each label, from its channel's state.
    curves: Vec<Vec<(Vec<f64>, Vec<f64>)>>,
    channels: Vec<Channel>,
    blocks: Vec<UsableBlock>,
}

/// What one spectrum's MF=40 fold produced.
pub(crate) struct IsomericSpectrum {
    /// The unit-flux partial rate of every partial of every channel, by parent
    /// and `"kind target"`, zero allowed. Only the labels are ever perturbed.
    partials: ReactionRates,
    /// The transport total's folded rate, for an isomer-only channel.
    wholes: HashMap<(String, String), f64>,
    /// The nominal fold of the whole overlay against this spectrum, from
    /// which a replica's split differs only where MF=40 moved a partial, so a
    /// split given by MF=9 yields keeps its nominal fractions.
    nominal: Fractions,
    sampler: Sampler,
    /// The partials each replica's draw moves, see [`draws`].
    draws: usize,
}

/// What the report says about the isomeric source.
#[derive(Default)]
pub(crate) struct IsomericReport {
    pub(crate) channels_perturbed: BTreeSet<String>,
    pub(crate) no_uncertainty: BTreeSet<String>,
    pub(crate) partials_without_covariance: BTreeSet<String>,
    pub(crate) rate_fraction_covered: BTreeMap<(String, String), f64>,
    pub(crate) blocks_skipped: BTreeMap<String, usize>,
    pub(crate) matrices_clipped: usize,
}

/// The isomeric source for one material's run: fixed labels, one fold and
/// sampler per spectrum, and the report.
pub(crate) struct IsomericSampling {
    parents: BTreeMap<String, Parent>,
    spectra: Vec<IsomericSpectrum>,
    pub(crate) report: IsomericReport,
}

impl IsomericSampling {
    /// Fold MF=40 against each spectrum for the parents the material reaches.
    ///
    /// `covariance` is `None` for a branching library without the file, which
    /// perturbs nothing and reports every channel with a rate as having no
    /// isomeric uncertainty. `transport` is the tally path, whose nominal split
    /// comes from the tallied partials, so no multigroup fold is built for it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        covariance: Option<&BranchingCovariance>,
        material: &Material,
        chain: &Arc<HashMap<String, ChainNuclide>>,
        reach: &HashMap<String, ChainNuclide>,
        branch: &BranchTable,
        spectra: &[MultigroupSpectrum],
        per_spectrum: &[PerSpectrum],
        transport: bool,
    ) -> Self {
        let mut report = IsomericReport::default();
        let mut parents: BTreeMap<String, Parent> = BTreeMap::new();
        // Every overlay channel with a split the material can drive, for the
        // report. One whose split MF=9 yields give has no partials, and no
        // channel here.
        let mut overlay: Vec<(String, String)> = Vec::new();
        // Overlay lists with no split to move, which the report treats as
        // reactions no overlay splits.
        let mut unsplit: HashSet<(&str, &str)> = HashSet::new();
        let mut names: Vec<&String> = branch
            .keys()
            .filter(|p| chain.contains_key(*p) && reach.contains_key(*p))
            .collect();
        names.sort();
        for parent in names {
            let mut kinds: Vec<&String> = branch[parent].keys().collect();
            kinds.sort();
            let mut channels = Vec::new();
            for kind in kinds {
                let curves = &branch[parent][kind];
                if one_state(&chain[parent], kind, curves) {
                    unsplit.insert((parent.as_str(), kind.as_str()));
                    continue;
                }
                overlay.push((parent.clone(), kind.clone()));
                if let Some(channel) = channel(material, chain, parent, kind, curves) {
                    channels.push(channel);
                }
            }
            let blocks = covariance.and_then(|c| c.by_parent.get(parent));
            let plan = plan(
                parent,
                channels,
                |kind| unsplit.contains(&(parent.as_str(), kind)),
                blocks,
                &mut report.blocks_skipped,
            );
            parents.insert(parent.clone(), plan);
        }

        // With no label anywhere there is nothing to fold, and the nominal
        // split and the partials would only be computed to be thrown away.
        let any_label = parents.values().any(|p| !p.labels.is_empty());
        let mut spectra_out = Vec::with_capacity(spectra.len());
        // Per (parent, label), the smallest covered share over the spectra.
        let mut covered: BTreeMap<(String, String), f64> = BTreeMap::new();
        // Which (parent, label) had a rate and a fold in some spectrum.
        let mut sampled: BTreeSet<(String, String)> = BTreeSet::new();
        // The blocks' views borrow only what `plan` fixed, so one set serves
        // every spectrum.
        let views: BTreeMap<&String, Vec<PartialBlock>> = parents
            .iter()
            .map(|(name, parent)| {
                let blocks = parent
                    .blocks
                    .iter()
                    .map(|b| PartialBlock {
                        row: b.row,
                        col: b.col,
                        row_lfs: b.row_lfs,
                        col_lfs: b.col_lfs,
                        row_curve: pieces(&b.row_own, &parent.curves[b.row]),
                        col_curve: pieces(&b.col_own, &parent.curves[b.col]),
                        expanded: &b.expanded,
                    })
                    .collect();
                (name, blocks)
            })
            .collect();
        for (idx, spectrum) in spectra.iter().enumerate() {
            let mut nominal = Fractions::new();
            if any_label && !transport {
                let mut scratch = per_spectrum[idx].0.clone();
                build_fold_refine(
                    material,
                    chain,
                    branch,
                    spectrum,
                    &mut scratch,
                    &mut nominal,
                );
            }
            let mut partials: ReactionRates = HashMap::new();
            let mut wholes = HashMap::new();
            let mut folded = BTreeMap::new();
            for (name, parent) in parents.iter().filter(|_| any_label) {
                let rates = partials.entry(name.clone()).or_default();
                for ch in &parent.channels {
                    for state in &ch.states {
                        let rate = state
                            .curves
                            .iter()
                            .map(|(e, v)| fold_curve_rate(e, v, spectrum))
                            .sum::<f64>();
                        rates.insert(state.label.clone(), rate);
                    }
                    if let Form::OfWhole { whole, .. } = &ch.form {
                        wholes.insert(
                            (name.clone(), ch.kind.clone()),
                            fold_curve_rate(&whole.0, &whole.1, spectrum),
                        );
                    }
                }
                if parent.labels.is_empty() {
                    continue;
                }
                let label_rates: Vec<f64> = parent.labels.iter().map(|l| rates[l]).collect();
                let (cov, coverage) = fold_partial_covariance(
                    &spectrum.boundaries,
                    &spectrum.masses,
                    &parent.labels,
                    &label_rates,
                    &views[name],
                );
                for (l, fraction) in coverage.rate_fraction_covered {
                    sampled.insert((name.clone(), l.clone()));
                    covered
                        .entry((name.clone(), l))
                        .and_modify(|f| *f = f.min(fraction))
                        .or_insert(fraction);
                }
                if let Some(cov) = cov {
                    folded.insert(name.clone(), cov);
                }
            }
            let sampler = Sampler::with_stream(&folded, ISOMERIC_BRANCHING_STREAM);
            report.matrices_clipped += sampler.clipping.matrices_clipped;
            let draws = draws(&sampler, &partials);
            spectra_out.push(IsomericSpectrum {
                partials,
                wholes,
                nominal,
                sampler,
                draws,
            });
        }

        // Every channel with a rate is one or the other, never neither.
        for (parent, kind) in overlay {
            let has_rate = per_spectrum.iter().any(|(rates, _, _)| {
                rates
                    .get(&parent)
                    .and_then(|r| r.get(&kind))
                    .is_some_and(|r| *r > 0.0)
            });
            if !has_rate {
                continue;
            }
            let plan = &parents[&parent];
            let moved =
                plan.keys.iter().zip(&plan.labels).any(|((k, _), l)| {
                    *k == kind && sampled.contains(&(parent.clone(), l.clone()))
                });
            let name = format!("{parent} {kind}");
            if !moved {
                report.no_uncertainty.insert(name);
                continue;
            }
            report.channels_perturbed.insert(name);
            if let Some(ch) = plan.channels.iter().find(|c| c.kind == kind) {
                for state in &ch.states {
                    let with_rate = spectra_out
                        .iter()
                        .any(|s| s.partials[&parent][&state.label] > 0.0);
                    if with_rate && !plan.labels.contains(&state.label) {
                        report
                            .partials_without_covariance
                            .insert(format!("{parent} {}", state.label));
                    }
                }
            }
        }
        // A split the chain fixes, on a reaction no overlay splits, is held at
        // the chain's number with nothing to sample it from. Without an overlay
        // that is every isomeric split the material drives, and it is said so
        // rather than left to read as exact.
        for (rates, _, _) in per_spectrum {
            for (parent, kinds) in rates {
                let Some(cn) = reach.get(parent) else {
                    continue;
                };
                let overlaid = branch.get(parent);
                for (kind, rate) in kinds {
                    let split = overlaid.is_some_and(|k| k.contains_key(kind))
                        && !unsplit.contains(&(parent.as_str(), kind.as_str()));
                    if *rate <= 0.0 || split {
                        continue;
                    }
                    let to_an_isomer = cn.reactions.iter().any(|r| {
                        r.kind == *kind
                            && r.branching > 0.0
                            && r.target
                                .as_deref()
                                .is_some_and(|t| endf::zam(t).is_ok_and(|(_, _, m)| m > 0))
                    });
                    if to_an_isomer {
                        report.no_uncertainty.insert(format!("{parent} {kind}"));
                    }
                }
            }
        }
        report.rate_fraction_covered = covered;

        IsomericSampling {
            parents,
            spectra: spectra_out,
            report,
        }
    }

    /// Whether any spectrum has anything to perturb.
    pub(crate) fn is_empty(&self) -> bool {
        self.spectra.iter().all(|s| s.sampler.is_empty())
    }

    /// Whether spectrum `idx` has anything to perturb.
    pub(crate) fn perturbs(&self, idx: usize) -> bool {
        !self.spectra[idx].sampler.is_empty()
    }

    /// One replica's split of every overlay channel under spectrum `idx`, with
    /// `rates` carrying its `(n,n')` production rates to match.
    pub(crate) fn replica_fractions(
        &self,
        idx: usize,
        seed: u64,
        replica: u64,
        rates: &mut ReactionRates,
        sampled: &mut usize,
    ) -> Fractions {
        let spectrum = &self.spectra[idx];
        let (perturbed, _) = spectrum.sampler.perturb(&spectrum.partials, seed, replica);
        *sampled += spectrum.draws;
        self.fractions_from(idx, &perturbed, rates)
    }

    /// The split under spectrum `idx` with the partials `partials` in place of
    /// the nominal ones, for every channel with a label; every other channel
    /// keeps its nominal split. `(n,n')` rates in `rates` are scaled by the
    /// ratio of the new partials' sum to the nominal one, which is what the
    /// fold injected as the rate.
    fn fractions_from(
        &self,
        idx: usize,
        partials: &ReactionRates,
        rates: &mut ReactionRates,
    ) -> Fractions {
        let spectrum = &self.spectra[idx];
        let mut fractions = spectrum.nominal.clone();
        for (name, parent) in &self.parents {
            let (Some(now), Some(nominal)) = (partials.get(name), spectrum.partials.get(name))
            else {
                continue;
            };
            for ch in &parent.channels {
                if !parent.keys.iter().any(|(k, _)| *k == ch.kind) {
                    continue;
                }
                let rate = |s: &State| now.get(&s.label).copied().unwrap_or(0.0);
                let mut split = Split::default();
                match &ch.form {
                    Form::Inelastic => {
                        let total: f64 = ch.states.iter().map(rate).filter(|r| *r > 0.0).sum();
                        if total <= 0.0 {
                            continue;
                        }
                        let before: f64 = ch
                            .states
                            .iter()
                            .map(|s| nominal[&s.label])
                            .filter(|r| *r > 0.0)
                            .sum();
                        if before > 0.0 {
                            if let Some(r) = rates.get_mut(name).and_then(|m| m.get_mut(&ch.kind)) {
                                *r *= total / before;
                            }
                        }
                        for s in &ch.states {
                            let r = rate(s);
                            if r > 0.0 {
                                *split.fractions.entry(s.target.clone()).or_insert(0.0) +=
                                    r / total;
                            }
                        }
                    }
                    Form::Among => {
                        let total: f64 = ch.states.iter().map(rate).sum();
                        if total <= 0.0 {
                            continue;
                        }
                        for s in &ch.states {
                            *split.fractions.entry(s.target.clone()).or_insert(0.0) +=
                                rate(s) / total;
                        }
                    }
                    Form::OfWhole { ground, .. } => {
                        let Some(&whole) = spectrum
                            .wholes
                            .get(&(name.clone(), ch.kind.clone()))
                            .filter(|w| **w > 0.0)
                        else {
                            continue;
                        };
                        for s in &ch.states {
                            *split.fractions.entry(s.target.clone()).or_insert(0.0) +=
                                rate(s) / whole;
                        }
                        split.remainder = Some(ground.clone());
                    }
                    Form::NoTotal => continue,
                }
                fractions
                    .entry(name.clone())
                    .or_default()
                    .insert(ch.kind.clone(), split);
            }
        }
        fractions
    }

    /// Scale a transport run's tallied partials by one replica's MF=40
    /// factors, folded against the tally's own multigroup flux shape.
    ///
    /// The tally scores the partials in continuous energy and the covariance
    /// is folded on groups; for a relative covariance that is the same
    /// approximation MF=33 makes on this path.
    pub(crate) fn scale_tallied_partials(
        &self,
        partials: &mut PartialRates,
        seed: u64,
        replica: u64,
        sampled: &mut usize,
    ) {
        let Some(spectrum) = self.spectra.first() else {
            return;
        };
        if spectrum.sampler.is_empty() {
            return;
        }
        let mut tallied: ReactionRates = HashMap::new();
        for (name, parent) in &self.parents {
            let Some(by_kind) = partials.get(name) else {
                continue;
            };
            for (l, (kind, target)) in parent.labels.iter().zip(&parent.keys) {
                let rate: f64 = by_kind
                    .get(kind)
                    .map(|list| {
                        list.iter()
                            .filter(|(t, _)| t == target)
                            .map(|(_, r)| r)
                            .sum()
                    })
                    .unwrap_or(0.0);
                if rate > 0.0 {
                    tallied
                        .entry(name.clone())
                        .or_default()
                        .insert(l.clone(), rate);
                }
            }
        }
        let (perturbed, _) = spectrum.sampler.perturb(&tallied, seed, replica);
        *sampled += draws(&spectrum.sampler, &tallied);
        for (name, by_label) in &tallied {
            let parent = &self.parents[name];
            for (l, rate) in by_label {
                let factor = perturbed[name][l] / rate;
                let i = parent
                    .labels
                    .binary_search(l)
                    .expect("a label of this parent");
                let (kind, target) = &parent.keys[i];
                if let Some(list) = partials.get_mut(name).and_then(|k| k.get_mut(kind)) {
                    for (t, r) in list.iter_mut() {
                        if t == target {
                            *r *= factor;
                        }
                    }
                }
            }
        }
    }

    /// The first-order jobs: `(spectrum, parent, label index)` for every label
    /// with a rate in that spectrum's fold.
    pub(crate) fn jobs(&self) -> Vec<(usize, String, usize)> {
        let mut out = Vec::new();
        for (a, spectrum) in self.spectra.iter().enumerate() {
            for (name, labels, _) in spectrum.sampler.factors() {
                for (i, l) in labels.iter().enumerate() {
                    if spectrum.partials[name][l] > 0.0 {
                        out.push((a, name.clone(), i));
                    }
                }
            }
        }
        out
    }

    /// Spectrum `a`'s factor for `parent`: its labels and row-major `L`.
    pub(crate) fn factor(&self, a: usize, parent: &str) -> Option<(&[String], &[f64])> {
        self.spectra[a]
            .sampler
            .factors()
            .find(|(n, _, _)| *n == parent)
            .map(|(_, labels, l)| (labels, l))
    }

    /// `(kind, target)` of a parent's label `i`.
    pub(crate) fn key(&self, parent: &str, i: usize) -> (&str, &str) {
        let (kind, target) = &self.parents[parent].keys[i];
        (kind, target)
    }

    /// Spectrum `a`'s rates and folded chain with one partial scaled, for a
    /// first-order job on the spectrum path. `base` is the pruned base chain.
    pub(crate) fn scaled(
        &self,
        a: usize,
        parent: &str,
        i: usize,
        scale: f64,
        nominal: &PerSpectrum,
        base: &Arc<HashMap<String, ChainNuclide>>,
    ) -> PerSpectrum {
        let mut partials = self.spectra[a].partials.clone();
        let l = &self.parents[parent].labels[i];
        if let Some(r) = partials.get_mut(parent).and_then(|m| m.get_mut(l)) {
            *r *= scale;
        }
        let mut rates = nominal.0.clone();
        let fractions = self.fractions_from(a, &partials, &mut rates);
        (
            rates,
            nominal.1.clone(),
            crate::material_transmute::refine_chain(base, &fractions),
        )
    }
}

/// How many of `rates`' partials one draw under `sampler` moves: those with a
/// rate and a non-zero row in their parent's factor. A label at no rate, below
/// threshold or with no total to take a share of, keeps its place in the
/// factor and is not a draw, and neither is one whose MF=40 grid misses where
/// its rate is.
fn draws(sampler: &Sampler, rates: &ReactionRates) -> usize {
    sampler
        .factors()
        .map(|(name, labels, l)| {
            let n = labels.len();
            labels
                .iter()
                .enumerate()
                .filter(|(i, label)| {
                    rates
                        .get(name)
                        .and_then(|r| r.get(*label))
                        .is_some_and(|r| *r > 0.0)
                        && l[i * n..(i + 1) * n].iter().any(|v| *v != 0.0)
                })
                .count()
        })
        .sum()
}

/// Whether a reaction's list shares it among fewer than two of the states the
/// chain takes it to, so that there is no split for MF=40 to move: one state
/// keeps all the mass the list covers in every replica, whatever its partial
/// does, and with none there is no mass to split. 451 of TENDL-2017's MF=10
/// lists with MF=40 name one state, a ground state each time (Os192 (n,3n)
/// lists Os190 alone). Of those lists besides `(n,n')` whose parent the
/// ENDF/B-VIII.1 chain has, 11574 of 12587 are for a reaction that chain does
/// not carry for the parent at all. An `(n,n')` list, or one of isomers whose
/// ground state takes the rest (see `remainder_state`), always has a split:
/// there a partial sets the rate or a share of the whole.
fn one_state(nuc: &ChainNuclide, kind: &str, curves: &[yani::BranchCurve]) -> bool {
    if kind == "(n,n')"
        || remainder_state(nuc, kind, curves.iter().map(|c| c.target.as_str())).is_some()
    {
        return false;
    }
    // The among-listed split re-partitions only the mass on the chain's own
    // edges, so a listed state the chain does not go to takes none of it.
    let carried: BTreeSet<&str> = curves
        .iter()
        .map(|c| c.target.as_str())
        .filter(|t| {
            nuc.reactions
                .iter()
                .any(|r| r.kind == kind && r.target.as_deref() == Some(*t))
        })
        .collect();
    carried.len() < 2
}

/// A reaction's partials, if its split is made of any: the MF=10 curves of
/// the reaction, grouped by target in the table's order. `None` for a split
/// only MF=9 yields give. The material decides only how an isomer-only list's
/// curves are continued, never which states there are, so a parent has the
/// same labels in every material.
fn channel(
    material: &Material,
    chain: &Arc<HashMap<String, ChainNuclide>>,
    parent: &str,
    kind: &str,
    curves: &[yani::BranchCurve],
) -> Option<Channel> {
    let inelastic = kind == "(n,n')";
    let cross: Vec<&yani::BranchCurve> = curves
        .iter()
        .filter(|c| c.quantity == BranchQuantity::CrossSection)
        .filter(|c| !(inelastic && c.target == parent))
        .collect();
    if cross.is_empty() {
        return None;
    }
    let form = if inelastic {
        Form::Inelastic
    } else {
        match remainder_state(
            &chain[parent],
            kind,
            curves.iter().map(|c| c.target.as_str()),
        ) {
            Some(ground) => match transport_reaction(material, parent, kind) {
                Some(r) => Form::OfWhole {
                    ground,
                    whole: (r.energy.to_vec(), r.cross_section.to_vec()),
                },
                None => Form::NoTotal,
            },
            None => Form::Among,
        }
    };
    let mut states: Vec<State> = Vec::new();
    for c in cross {
        let curve = match &form {
            Form::OfWhole { whole, .. } => {
                Some(along_the_total(&c.energy, &c.values, &whole.0, &whole.1))
            }
            Form::NoTotal => None,
            _ => Some((c.energy.clone(), c.values.clone())),
        };
        match states.iter_mut().find(|s| s.target == c.target) {
            Some(s) => s.curves.extend(curve),
            None => states.push(State {
                target: c.target.clone(),
                label: label(kind, &c.target),
                curves: curve.into_iter().collect(),
            }),
        }
    }
    Some(Channel {
        kind: kind.to_string(),
        form,
        states,
    })
}

/// The parent's labels and usable blocks, counting every block it cannot use.
/// The blocks of a kind `unsplit` names are passed over uncounted, as there
/// is no split for them to move.
fn plan(
    parent: &str,
    channels: Vec<Channel>,
    unsplit: impl Fn(&str) -> bool,
    blocks: Option<&BTreeMap<String, Vec<BranchingCovarianceBlock>>>,
    skipped: &mut BTreeMap<String, usize>,
) -> Parent {
    let mut skip = |why: String| *skipped.entry(why).or_insert(0) += 1;
    let state_of = |kind: &str, target: &str| {
        channels
            .iter()
            .find(|c| c.kind == kind)
            .and_then(|c| c.states.iter().find(|s| s.target == target))
    };
    // A state's own partial, looked up by (kind, target, level) among all the
    // parent's blocks, since a partner's is carried on the partner's rows.
    let own = |kind: &str, target: &str, lfs: i32| {
        blocks
            .and_then(|b| b.get(kind))
            .and_then(|rows| {
                rows.iter()
                    .find(|b| b.target == target && b.lfs == lfs && b.energy.is_some())
            })
            .and_then(|b| Some((b.energy.clone()?, b.values.clone()?)))
    };

    type Candidate = (String, String, i32, i32, ExpandedBlock, String);
    let mut candidates: Vec<Candidate> = Vec::new();
    for (kind, rows) in blocks.into_iter().flatten() {
        if unsplit(kind) {
            continue;
        }
        for b in rows {
            if b.block.is_cross_material() {
                skip("cross_material".to_string());
                continue;
            }
            if b.block.partner_mt() != b.block.mt {
                skip("cross_reaction".to_string());
                continue;
            }
            let ni = match &b.block.data {
                CovarianceData::Ni(ni) => ni,
                CovarianceData::Nc(_) => {
                    skip("nc".to_string());
                    continue;
                }
            };
            // The ground state's own (n,n') partial scatters the parent into
            // itself, which changes nothing, so neither does its covariance.
            if kind == "(n,n')" && (b.target == parent || b.target1.as_deref() == Some(parent)) {
                continue;
            }
            let Some(target1) = b.target1.as_deref() else {
                skip("unmatched".to_string());
                continue;
            };
            if state_of(kind, &b.target).is_none() || state_of(kind, target1).is_none() {
                skip("unmatched".to_string());
                continue;
            }
            let expanded = match expand_ni(ni) {
                Ok(e) if e.is_empty() => continue,
                Ok(e) => e,
                Err(Unsupported::Layout(lb)) => {
                    skip(format!("lb_{lb}"));
                    continue;
                }
                Err(Unsupported::Malformed) => {
                    skip("malformed".to_string());
                    continue;
                }
            };
            candidates.push((
                kind.clone(),
                b.target.clone(),
                b.lfs,
                b.lfs1,
                expanded,
                target1.to_string(),
            ));
        }
    }

    let mut keys: Vec<(String, String)> = candidates
        .iter()
        .flat_map(|(kind, target, _, _, _, target1)| {
            [
                (kind.clone(), target.clone()),
                (kind.clone(), target1.clone()),
            ]
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    // Sorted by the label string itself, which is what the sampler's rows and
    // a lookup by label both go by.
    keys.sort_by_key(|(k, t)| label(k, t));
    let labels: Vec<String> = keys.iter().map(|(k, t)| label(k, t)).collect();
    let curves = keys
        .iter()
        .map(|(k, t)| state_of(k, t).expect("a label is a state").curves.clone())
        .collect();
    let index = |kind: &str, target: &str| {
        labels
            .binary_search(&label(kind, target))
            .expect("a label of this parent")
    };
    let usable = candidates
        .into_iter()
        .map(|(kind, target, lfs, lfs1, expanded, target1)| UsableBlock {
            row: index(&kind, &target),
            col: index(&kind, &target1),
            row_lfs: lfs,
            col_lfs: lfs1,
            row_own: own(&kind, &target, lfs),
            col_own: own(&kind, &target1, lfs1),
            expanded,
        })
        .collect();
    Parent {
        labels,
        keys,
        curves,
        channels,
        blocks: usable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yamc_nuclide::covariance::CovarianceBlock;
    use yani::{BranchCurve, ChainReaction};

    fn nuclide(name: &str, reactions: Vec<(&str, &str, f64)>) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life: None,
            half_life_uncertainty: None,
            decay_energy: 0.0,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
            reactions: reactions
                .into_iter()
                .map(|(kind, target, branching)| ChainReaction {
                    kind: kind.to_string(),
                    target: Some(target.to_string()),
                    branching,
                    q_value: Some(0.0),
                })
                .collect(),
            decays: vec![],
            fission_yields: None,
            sources: Vec::new(),
        }
    }

    fn material(name: &str) -> Material {
        Material::new(
            HashMap::from([(name.to_string(), 1.0e-2)]),
            "atom",
            "sum",
            None,
        )
        .unwrap()
    }

    fn flat(target: &str, barns: f64) -> BranchCurve {
        BranchCurve {
            target: target.to_string(),
            quantity: BranchQuantity::CrossSection,
            energy: vec![7.4e6, 2.0e7],
            values: vec![barns, barns],
        }
    }

    /// An LB=5 LS=1 self block with one relative variance over `grid`.
    fn self_block(
        parent: &str,
        kind: &str,
        target: &str,
        variance: f64,
    ) -> BranchingCovarianceBlock {
        BranchingCovarianceBlock {
            nuclide: parent.to_string(),
            reaction: kind.to_string(),
            target: target.to_string(),
            lfs: 1,
            target1: Some(target.to_string()),
            lfs1: 1,
            energy: None,
            values: None,
            block: CovarianceBlock {
                mt: 16,
                subsection_idx: 0,
                block_idx: 0,
                mat1: 0,
                mt1: 16,
                xmf1: 10.0,
                xlfs1: 1.0,
                mtl: 0,
                data: CovarianceData::Ni(endf::mf::covariance::NiSubsection {
                    lb: 5,
                    ls: 1,
                    nt: 3,
                    ne: 2,
                    ek: vec![7.4e6, 2.0e7],
                    fkk: vec![variance],
                    ..Default::default()
                }),
            },
        }
    }

    fn fourteen_mev() -> MultigroupSpectrum {
        MultigroupSpectrum {
            boundaries: vec![1.3e7, 1.5e7],
            masses: vec![1.0],
            flux_error: None,
        }
    }

    /// Scaling one partial moves the split it sits in and leaves the
    /// reaction's rate alone: that rate is the transport total's, which
    /// MF=33 perturbs and MF=40 does not.
    #[test]
    fn a_partial_moves_the_split_and_not_the_total() {
        let chain = Arc::new(HashMap::from([
            (
                "Pb208".to_string(),
                nuclide(
                    "Pb208",
                    vec![("(n,2n)", "Pb207", 1.0), ("(n,2n)", "Pb207_m1", 0.0)],
                ),
            ),
            ("Pb207".to_string(), nuclide("Pb207", vec![])),
            ("Pb207_m1".to_string(), nuclide("Pb207_m1", vec![])),
        ]));
        let mut branch = BranchTable::new();
        branch.entry("Pb208".to_string()).or_default().insert(
            "(n,2n)".to_string(),
            vec![flat("Pb207", 1.5), flat("Pb207_m1", 0.5)],
        );
        let covariance =
            BranchingCovariance::from_blocks(vec![self_block("Pb208", "(n,2n)", "Pb207_m1", 0.01)]);
        let spectrum = fourteen_mev();
        let m = material("Pb208");
        let mut rates: ReactionRates = HashMap::from([(
            "Pb208".to_string(),
            HashMap::from([("(n,2n)".to_string(), 2.0e-24)]),
        )]);
        let folded = crate::material_transmute::fold_branching_into_chain(
            &m, &chain, &branch, &spectrum, &mut rates,
        );
        let per_spectrum = vec![(rates, HashMap::new(), folded)];
        let iso = IsomericSampling::new(
            Some(&covariance),
            &m,
            &chain,
            &chain,
            &branch,
            std::slice::from_ref(&spectrum),
            &per_spectrum,
            false,
        );
        assert_eq!(iso.parents["Pb208"].labels, vec!["(n,2n) Pb207_m1"]);
        assert!(iso.report.channels_perturbed.contains("Pb208 (n,2n)"));
        assert!(iso
            .report
            .partials_without_covariance
            .contains("Pb208 (n,2n) Pb207"));

        let branching = |c: &HashMap<String, ChainNuclide>, target: &str| {
            c["Pb208"]
                .reactions
                .iter()
                .find(|r| r.target.as_deref() == Some(target))
                .unwrap()
                .branching
        };
        let (rates, _, scaled) = iso.scaled(0, "Pb208", 0, 1.1, &per_spectrum[0], &chain);
        assert_eq!(
            rates["Pb208"]["(n,2n)"].to_bits(),
            2.0e-24f64.to_bits(),
            "the total is the transport total's, and MF=40 does not move it"
        );
        let f = 0.55 / 2.05;
        assert!((branching(&scaled, "Pb207_m1") - f).abs() < 1e-12);
        assert!((branching(&scaled, "Pb207") - (1.0 - f)).abs() < 1e-12);
        // A scale of one is the nominal split.
        let (_, _, same) = iso.scaled(0, "Pb208", 0, 1.0, &per_spectrum[0], &chain);
        assert!((branching(&same, "Pb207_m1") - 0.25).abs() < 1e-15);
    }

    /// A parent's labels come from the branch table and its MF=40, not from
    /// what the material loads, so each partial draws the same deviate in
    /// every material. Reached without its cross sections, Pb208 has no
    /// (n,2n) total for its isomer-only list to take a share of, and that
    /// label stays at no rate; loaded, the same label has one. (n,n') Pb208_m1
    /// is label 1 either way.
    #[test]
    fn a_parents_labels_do_not_depend_on_what_the_material_loads() {
        let chain = Arc::new(HashMap::from([
            (
                "Pb208".to_string(),
                nuclide(
                    "Pb208",
                    vec![
                        ("(n,2n)", "Pb207", 1.0),
                        ("(n,2n)", "Pb207_m1", 0.0),
                        ("(n,n')", "Pb208_m1", 1.0),
                    ],
                ),
            ),
            ("Pb207".to_string(), nuclide("Pb207", vec![])),
            ("Pb207_m1".to_string(), nuclide("Pb207_m1", vec![])),
            ("Pb208_m1".to_string(), nuclide("Pb208_m1", vec![])),
        ]));
        let mut branch = BranchTable::new();
        let kinds = branch.entry("Pb208".to_string()).or_default();
        kinds.insert("(n,2n)".to_string(), vec![flat("Pb207_m1", 0.5)]);
        kinds.insert("(n,n')".to_string(), vec![flat("Pb208_m1", 0.3)]);
        let mut inelastic = self_block("Pb208", "(n,n')", "Pb208_m1", 0.01);
        inelastic.block.mt = 4;
        inelastic.block.mt1 = 4;
        let covariance = BranchingCovariance::from_blocks(vec![
            self_block("Pb208", "(n,2n)", "Pb207_m1", 0.01),
            inelastic,
        ]);
        let spectrum = fourteen_mev();
        let sample = |m: &Material, mut rates: ReactionRates| {
            let folded = crate::material_transmute::fold_branching_into_chain(
                m, &chain, &branch, &spectrum, &mut rates,
            );
            let per_spectrum = vec![(rates, HashMap::new(), folded)];
            IsomericSampling::new(
                Some(&covariance),
                m,
                &chain,
                &chain,
                &branch,
                std::slice::from_ref(&spectrum),
                &per_spectrum,
                false,
            )
        };
        let labels = ["(n,2n) Pb207_m1", "(n,n') Pb208_m1"].map(String::from);

        let bare = sample(&material("Pb208"), HashMap::new());
        assert_eq!(bare.parents["Pb208"].labels, labels);
        assert!(
            bare.report.blocks_skipped.is_empty(),
            "{:?}",
            bare.report.blocks_skipped
        );
        assert_eq!(bare.spectra[0].partials["Pb208"][&labels[0]], 0.0);
        assert!(bare.report.channels_perturbed.contains("Pb208 (n,n')"));
        // A label at no rate keeps its place and is not a draw.
        assert_eq!(bare.spectra[0].draws, 1);

        let Some(path) = yamc_test_cache::nuclide("Pb208") else {
            eprintln!("skipping the loaded half -- Pb208 fixture absent");
            return;
        };
        let mut m = material("Pb208");
        m.set_temperature("294");
        m.read_nuclear_data(&HashMap::from([("Pb208".to_string(), path)]), None)
            .expect("read nuclear data");
        let loaded = sample(
            &m,
            HashMap::from([(
                "Pb208".to_string(),
                HashMap::from([("(n,2n)".to_string(), 2.0e-24)]),
            )]),
        );
        assert_eq!(loaded.parents["Pb208"].labels, labels);
        assert!(loaded.spectra[0].partials["Pb208"][&labels[0]] > 0.0);
        assert_eq!(loaded.spectra[0].draws, 2);
        let (_, l_bare) = bare.factor(0, "Pb208").expect("the (n,n') partial");
        let (_, l_loaded) = loaded.factor(0, "Pb208").expect("both partials");
        let sigma = |l: &[f64]| l[2..4].iter().map(|v| v * v).sum::<f64>().sqrt();
        for replica in 0..64 {
            let a = bare.spectra[0].sampler.deviates(9, replica)["Pb208"][1] / sigma(l_bare);
            let b = loaded.spectra[0].sampler.deviates(9, replica)["Pb208"][1] / sigma(l_loaded);
            assert!(
                (a - b).abs() <= 4.0 * f64::EPSILON * a.abs(),
                "replica {replica}: {a} against {b}"
            );
        }
    }

    /// A split the chain fixes, with no overlay to sample it from, is named as
    /// held at nominal; a reaction to the ground state alone is no split.
    #[test]
    fn a_split_the_chain_fixes_is_named_as_held() {
        let chain = Arc::new(HashMap::from([(
            "Co59".to_string(),
            nuclide(
                "Co59",
                vec![
                    ("(n,gamma)", "Co60", 0.444),
                    ("(n,gamma)", "Co60_m1", 0.556),
                    ("(n,2n)", "Co58", 1.0),
                ],
            ),
        )]));
        let rates: ReactionRates = HashMap::from([(
            "Co59".to_string(),
            HashMap::from([
                ("(n,gamma)".to_string(), 1.0e-24),
                ("(n,2n)".to_string(), 1.0e-26),
            ]),
        )]);
        let per_spectrum = vec![(rates, HashMap::new(), Arc::clone(&chain))];
        let iso = IsomericSampling::new(
            None,
            &material("Co59"),
            &chain,
            &chain,
            &BranchTable::new(),
            &[fourteen_mev()],
            &per_spectrum,
            false,
        );
        assert!(iso.is_empty());
        assert_eq!(
            iso.report.no_uncertainty,
            BTreeSet::from(["Co59 (n,gamma)".to_string()])
        );
    }

    /// A list shared among its states that names only one state the chain
    /// goes to has no split for MF=40 to move: its blocks are passed over
    /// uncounted, and the channel is neither perturbed nor a gap. Where the
    /// chain also sends the reaction to an isomer the list leaves alone, that
    /// split is the chain's own and is named as held, as it is with no overlay.
    #[test]
    fn a_list_naming_one_state_has_no_split() {
        let sample = |reactions: Vec<(&str, &str, f64)>, curves: Vec<BranchCurve>| {
            let chain = Arc::new(HashMap::from([
                ("Pb208".to_string(), nuclide("Pb208", reactions)),
                ("Pb207".to_string(), nuclide("Pb207", vec![])),
                ("Pb207_m1".to_string(), nuclide("Pb207_m1", vec![])),
            ]));
            let mut branch = BranchTable::new();
            branch
                .entry("Pb208".to_string())
                .or_default()
                .insert("(n,2n)".to_string(), curves);
            let covariance = BranchingCovariance::from_blocks(vec![
                self_block("Pb208", "(n,2n)", "Pb207", 0.01),
                self_block("Pb208", "(n,2n)", "Pb207_m1", 0.01),
            ]);
            let spectrum = fourteen_mev();
            let m = material("Pb208");
            let mut rates: ReactionRates = HashMap::from([(
                "Pb208".to_string(),
                HashMap::from([("(n,2n)".to_string(), 2.0e-24)]),
            )]);
            let folded = crate::material_transmute::fold_branching_into_chain(
                &m, &chain, &branch, &spectrum, &mut rates,
            );
            let per_spectrum = vec![(rates, HashMap::new(), folded)];
            IsomericSampling::new(
                Some(&covariance),
                &m,
                &chain,
                &chain,
                &branch,
                std::slice::from_ref(&spectrum),
                &per_spectrum,
                false,
            )
        };
        let no_split = |iso: &IsomericSampling| {
            assert!(iso.parents["Pb208"].labels.is_empty());
            assert!(iso.is_empty());
            assert!(iso.report.channels_perturbed.is_empty());
            assert!(
                iso.report.blocks_skipped.is_empty(),
                "{:?}",
                iso.report.blocks_skipped
            );
        };

        // The ground alone, the shape of TENDL-2017's Os192 (n,3n) to Os190.
        let alone = sample(vec![("(n,2n)", "Pb207", 1.0)], vec![flat("Pb207", 2.0)]);
        no_split(&alone);
        assert!(alone.report.no_uncertainty.is_empty());

        // Two states listed, but the chain goes to one of them, so the other
        // takes none of the mass and the one keeps all of it.
        let one_carried = sample(
            vec![("(n,2n)", "Pb207", 1.0)],
            vec![flat("Pb207", 1.5), flat("Pb207_m1", 0.5)],
        );
        no_split(&one_carried);
        assert!(one_carried.report.no_uncertainty.is_empty());

        // The ground alone beside a chain split to the isomer.
        let beside = sample(
            vec![("(n,2n)", "Pb207", 0.75), ("(n,2n)", "Pb207_m1", 0.25)],
            vec![flat("Pb207", 2.0)],
        );
        no_split(&beside);
        assert_eq!(
            beside.report.no_uncertainty,
            BTreeSet::from(["Pb208 (n,2n)".to_string()])
        );
    }

    /// One MF=40 partial draws one deviate per replica in every spectrum of
    /// a schedule, including one where another of the parent's partials has
    /// no rate at all.
    ///
    /// TENDL-2017 Nb93 under a 1-20 MeV spectrum and under a 0.1-5 MeV one,
    /// wholly below its (n,2n) threshold. Every block is a self block, so each
    /// factor is diagonal and the standardized deviate of a label is its
    /// stream's own normal: equal in both spectra to the last bits the two
    /// different square roots leave.
    #[test]
    fn a_labels_deviate_is_the_same_in_every_spectrum() {
        let text = |blob: &[u8]| {
            let mut out = Vec::new();
            lzma_rs::xz_decompress(&mut &blob[..], &mut out).unwrap();
            endf::Material::from_str(&String::from_utf8(out).unwrap()).unwrap()
        };
        let nb93 = text(include_bytes!(
            "../../endf/fixtures/n-041_Nb_093_tendl2017_trimmed.endf.xz"
        ));
        let decay = vec![
            text(include_bytes!(
                "../../endf/fixtures/dec-041_Nb_092m1.endf.xz"
            )),
            text(include_bytes!(
                "../../endf/fixtures/dec-041_Nb_093m1.endf.xz"
            )),
        ];
        let extracted = yani_convert::branching::extract_branching(
            &[nb93],
            &decay,
            endf::radionuclide_production::ISOMER_ENERGY_TOLERANCE,
            yani_convert::branching::DEFAULT_LINEARIZE_TOL,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        yani_convert::branching::write_branching_covariance(&extracted.covariance, dir.path())
            .unwrap();
        let blocks = yamc_nuclide::arrow::covariance_arrow::read_branching_covariance(
            &dir.path().join("branching_covariance.arrow"),
        )
        .unwrap();
        let covariance = BranchingCovariance::from_blocks(blocks);
        let mut branch = BranchTable::new();
        for row in &extracted.rows {
            branch
                .entry(row.nuclide.clone())
                .or_default()
                .entry(row.reaction.clone())
                .or_default()
                .push(BranchCurve {
                    target: row.target.clone(),
                    quantity: if row.quantity == "yield" {
                        BranchQuantity::Yield
                    } else {
                        BranchQuantity::CrossSection
                    },
                    energy: row.energy.clone(),
                    values: row.values.clone(),
                });
        }
        let chain = Arc::new(HashMap::from([
            (
                "Nb93".to_string(),
                nuclide(
                    "Nb93",
                    vec![
                        ("(n,2n)", "Nb92", 1.0),
                        ("(n,2n)", "Nb92_m1", 0.0),
                        ("(n,n')", "Nb93_m1", 1.0),
                    ],
                ),
            ),
            ("Nb92".to_string(), nuclide("Nb92", vec![])),
            ("Nb92_m1".to_string(), nuclide("Nb92_m1", vec![])),
            ("Nb93_m1".to_string(), nuclide("Nb93_m1", vec![])),
        ]));
        let m = material("Nb93");
        let spectra = [
            MultigroupSpectrum {
                boundaries: vec![1.0e6, 2.0e7],
                masses: vec![1.0],
                flux_error: None,
            },
            MultigroupSpectrum {
                boundaries: vec![1.0e5, 5.0e6],
                masses: vec![1.0],
                flux_error: None,
            },
        ];
        let per_spectrum: Vec<PerSpectrum> = spectra
            .iter()
            .map(|s| {
                let mut rates: ReactionRates = HashMap::from([(
                    "Nb93".to_string(),
                    HashMap::from([("(n,2n)".to_string(), 1.0e-24)]),
                )]);
                let folded = crate::material_transmute::fold_branching_into_chain(
                    &m, &chain, &branch, s, &mut rates,
                );
                (rates, HashMap::new(), folded)
            })
            .collect();
        let iso = IsomericSampling::new(
            Some(&covariance),
            &m,
            &chain,
            &chain,
            &branch,
            &spectra,
            &per_spectrum,
            false,
        );
        let (labels_a, l_a) = iso.factor(0, "Nb93").expect("a factor under 1-20 MeV");
        let (labels_b, l_b) = iso.factor(1, "Nb93").expect("a factor under 0.1-5 MeV");
        assert_eq!(labels_a, labels_b);
        assert_eq!(
            labels_a,
            ["(n,2n) Nb92", "(n,2n) Nb92_m1", "(n,n') Nb93_m1"].map(String::from)
        );
        let n = labels_a.len();
        let i = labels_a.iter().position(|l| l == "(n,n') Nb93_m1").unwrap();
        let sigma = |l: &[f64], i: usize| {
            l[i * n..(i + 1) * n]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
        };
        // Below threshold the (n,2n) partials have no rate, and no row, and
        // are not draws.
        assert!(l_b[..2 * n].iter().all(|v| *v == 0.0));
        assert_eq!((iso.spectra[0].draws, iso.spectra[1].draws), (3, 1));
        assert!(sigma(l_a, i) > 0.0 && sigma(l_b, i) > 0.0);
        for replica in 0..64 {
            let a = iso.spectra[0].sampler.deviates(9, replica)["Nb93"][i] / sigma(l_a, i);
            let b = iso.spectra[1].sampler.deviates(9, replica)["Nb93"][i] / sigma(l_b, i);
            assert!(
                (a - b).abs() <= 4.0 * f64::EPSILON * a.abs(),
                "replica {replica}: {a} against {b}"
            );
        }
    }

    /// The literal tag of every stream a replica draws from, and the two that
    /// are not keyed on a name.
    const TAGS: [u32; 6] = [
        0,
        0x4A1F_11FE,
        0xDEC4_E6E1,
        0xDB2A_0C5E,
        0xF155_10E1,
        0x150B_4A7C,
    ];
    const STATISTICAL: u32 = 0x57A7_1571;
    const FLUX: u32 = 0xF10D_5EED;

    /// No (name, tag) pair over the committed chain lands on another's
    /// ordinal, so no two sources ever share a stream by accident.
    #[test]
    fn the_stream_does_not_collide_with_any_other() {
        assert_eq!(ISOMERIC_BRANCHING_STREAM, 0x150B_4A7C);
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
        let chain = yani::parse_chain_arrow(&path).expect("parse chain");
        let mut seen: HashMap<u32, (String, u32)> = HashMap::new();
        let unnamed: Vec<u32> = std::iter::once(STATISTICAL)
            .chain((0..64).map(|i| FLUX ^ i))
            .collect();
        for name in chain.keys() {
            for tag in TAGS {
                let ordinal = crate::covariance_sample::name_ordinal(name) ^ tag;
                assert!(
                    !unnamed.contains(&ordinal),
                    "{name} under {tag:#x} is an unnamed stream"
                );
                if let Some(other) = seen.insert(ordinal, (name.clone(), tag)) {
                    panic!("{name} under {tag:#x} collides with {other:?}");
                }
            }
        }
    }
}
