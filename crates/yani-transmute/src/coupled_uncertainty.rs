//! Uncertainty on a coupled transport-transmutation, by replicas of the whole
//! coupled run.
//!
//! The coupled method transports every irradiation step on the compositions
//! the previous steps produced, so step k's rates carry the noise and the data
//! draws of steps 0 to k-1 through the compositions they were tallied in.
//! Resampling one step's rates on its own would leave that feedback out. Here
//! each replica is the coupled calculation again, start to finish: its own
//! transport seed when the statistical source is on, its own decay data when
//! those sources are on, its own cross sections in every transport when the
//! cross-section source is on, and every step's transport run on that
//! replica's own compositions. The spread of the replicas' inventories is then
//! the uncertainty of the coupled run, exactly and to all orders: nothing is
//! linearised and no step is treated as independent of another. The cost is
//! the transport of every irradiation step once per replica.
//!
//! The draws are those [`crate::transport_replicas`] makes for independent
//! mode, keyed on `(seed, replica, nuclide)`, so a nuclide's half-life, a
//! parent's branching, an evaluation's fission yields and a nuclide's cross
//! sections are the same in replica `k` of either method, and shared by every
//! material of the run. The transport applies the cross sections (see
//! `yamc::coupled_xs`); this module only says which replicas to run.
//!
//! The cross sections are carried only beside the statistical source. A
//! replica's transport on its own cross sections follows different histories
//! from the nominal one whatever its seed, so its spread holds that
//! transport's Monte Carlo noise as well as the data's, and no seed removes
//! it. With both sources on, that noise is the statistical source's
//! uncertainty and the spread is the sum of the two, as asked; with the cross
//! sections alone, it would be reported as the data's.
//!
//! The MF=40 isomer production covariance moves the branching table: every
//! transport of replica `k` tallies its isomeric partials, and every step
//! splits them, on production curves carrying replica `k`'s draw, the one an
//! independent replica of the same index makes (see
//! [`crate::isomer_branching_uncertainty`]). A curve only moves what is
//! tallied, not what is transported, so this source needs no seed of its
//! own, as the decay data need none.
//!
//! Attribution reruns the ensemble once per source, each drawing that source
//! alone on the same streams, so a source's share is measured as the total
//! is, at the cost of one more coupled ensemble per source.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use yamc_materials::Material;
use yani::ChainNuclide;

use crate::material_transmute::{decay_energy_candidates, decay_photon_candidates};
use crate::uncertainty::{
    check_convergence, Attribution, DataUncertainty, Ensemble, FluxResponseReport, Info, Linearity,
    Source, BLOCK, MAX_SAMPLES,
};

/// Keeps the replicas' transport seeds clear of every per-nuclide stream.
const TRANSPORT_STREAM: u32 = 0x7A45_C0DE;

/// One material's per-step inventories in one replica, entry `i` after step
/// `i`.
pub type Inventories = Vec<HashMap<String, f64>>;

/// What one replica drew of the decay data.
#[derive(Debug, Default, Clone)]
pub struct Drawn {
    /// Every perturbed half-life [s], by nuclide.
    pub half_lives: HashMap<String, f64>,
    /// Every parent whose decay branching was drawn.
    pub decay_branchings: BTreeSet<String>,
    /// The parents among them whose draw was pulled back onto the simplex.
    pub floored: BTreeSet<String>,
}

/// What a coupled run's replicas draw, and how many of them to run.
#[derive(Clone)]
pub struct CoupledReplicas {
    request: DataUncertainty,
    statistical: bool,
    cross_sections: bool,
    /// Every replica rides one transport per step as a likelihood-ratio
    /// weight ([`DataUncertainty::coupled_replica_weights`]).
    weights: bool,
    half_lives: Vec<crate::uncertainty::HalfLifeSigma>,
    decay_branchings: Option<crate::decay_branching_uncertainty::Candidates>,
    fission_yields: Option<crate::fission_yield_uncertainty::Candidates>,
    isomer: Option<Arc<crate::isomer_branching_uncertainty::CoupledIsomer>>,
}

impl CoupledReplicas {
    /// The draws `request` asks of a coupled run over `chain`, whose
    /// isomeric splits `branch` gives, or an error naming what a coupled
    /// replica does not yet carry.
    pub fn new(
        request: &DataUncertainty,
        chain: &HashMap<String, ChainNuclide>,
        branch: &yani::BranchTable,
    ) -> Result<Self, String> {
        check_convergence(request.convergence)?;
        let cross_sections = request.wants(Source::CrossSections);
        let statistical = request.wants(Source::Statistical);
        if request.coupled_replica_weights {
            let refuse = |what: &str, why: &str| -> Result<Self, String> {
                Err(format!(
                    "data_uncertainty with method=\"coupled\" and coupled_replica_weights \
                     does not carry {what}: {why}; leave it out, or turn \
                     coupled_replica_weights off to rerun the whole coupled calculation \
                     per replica"
                ))
            };
            if statistical {
                return refuse(
                    "\"statistical\"",
                    "every replica rides the nominal transport's histories, so none has \
                     a Monte Carlo realisation of its own to spread by",
                );
            }
            if request.wants(Source::IsomerBranching) {
                return refuse(
                    "\"isomer_branching\"",
                    "the weighted transport tallies every replica on the nominal \
                     production curves",
                );
            }
            if request.attribution {
                return refuse("attribution", "it needs one ensemble per source");
            }
        } else if cross_sections && !statistical {
            return Err(
                "data_uncertainty with method=\"coupled\" carries \"cross_sections\" only \
                 with \"statistical\": a replica transported on its own cross sections \
                 follows different histories from the nominal run whatever its seed, so \
                 its spread holds the transport's Monte Carlo noise as well as the data's, \
                 and without the statistical source that noise would be reported as the \
                 data's. Name both, or use method=\"independent\", whose replica weights \
                 remove the noise"
                    .to_string(),
            );
        }
        let half_lives = if request.wants(Source::HalfLife) {
            crate::uncertainty::half_life_candidates(chain).0
        } else {
            Vec::new()
        };
        Ok(Self {
            request: request.clone(),
            statistical,
            cross_sections,
            weights: request.coupled_replica_weights,
            half_lives,
            decay_branchings: request
                .wants(Source::DecayBranching)
                .then(|| crate::decay_branching_uncertainty::candidates(chain)),
            fission_yields: request
                .wants(Source::FissionYield)
                .then(|| crate::fission_yield_uncertainty::candidates(chain, chain)),
            isomer: if request.wants(Source::IsomerBranching) {
                Some(Arc::new(
                    crate::isomer_branching_uncertainty::CoupledIsomer::new(branch, chain)?,
                ))
            } else {
                None
            },
        })
    }

    /// `branch` with replica `replica`'s MF=40 draw on its stated
    /// production curves, or `branch` itself when nothing is drawn.
    pub fn branch(&self, branch: &Arc<yani::BranchTable>, replica: u64) -> Arc<yani::BranchTable> {
        match self.isomer.as_ref().filter(|i| !i.is_empty()) {
            Some(i) => Arc::new(i.branch(branch, self.request.seed, replica)),
            None => Arc::clone(branch),
        }
    }

    /// How many replicas each step's one transport carries as weights, or
    /// `None` when each replica is a coupled run of its own.
    pub fn replica_weights(&self) -> Option<usize> {
        self.weights
            .then(|| crate::uncertainty::transport_replicas_for(&self.request))
    }

    /// Each material's ensemble of the replicas a weighted coupled run
    /// stepped, `per_replica[k]` replica `k`'s per-step inventories and
    /// `drawn[k]` what it drew of the decay data.
    pub fn collect(
        &self,
        chain: &Arc<HashMap<String, ChainNuclide>>,
        initial: &HashMap<u32, Material>,
        nominal: &HashMap<u32, Inventories>,
        per_replica: &[HashMap<u32, Inventories>],
        drawn: &[Drawn],
    ) -> HashMap<u32, (Ensemble, Info)> {
        let mut reports: BTreeMap<u32, (Ensemble, Info)> = initial
            .iter()
            .map(|(&id, material)| {
                let steps = nominal.get(&id).map_or(0, Vec::len);
                let (e, mut info) = self.report(material, chain, Ensemble::new(steps));
                info.flux_response = FluxResponseReport::CoupledReplicaWeights {
                    replicas: per_replica.len(),
                };
                (id, (e, info))
            })
            .collect();
        for (inventories, drawn) in per_replica.iter().zip(drawn) {
            push(&mut reports, inventories, drawn);
        }
        for (ensemble, info) in reports.values_mut() {
            ensemble.fold_absences();
            info.conclude(ensemble, chain, &self.request);
        }
        reports.into_iter().collect()
    }

    /// Whether each replica's transports read its own cross sections.
    pub fn cross_sections(&self) -> bool {
        self.cross_sections
    }

    /// Whether a replica's inventory can differ from the nominal one: a
    /// fresh transport seed, its own cross sections, or decay data that
    /// enters the solve. Without any every replica would repeat the nominal
    /// run.
    pub fn moves_inventory(&self) -> bool {
        self.statistical
            || self.cross_sections
            || !self.half_lives.is_empty()
            || self
                .decay_branchings
                .as_ref()
                .is_some_and(|b| !b.splits.is_empty())
            || self.fission_yields.as_ref().is_some_and(|f| !f.is_empty())
            || self.isomer.as_ref().is_some_and(|i| !i.is_empty())
    }

    /// The transport seed replica `replica` runs every step with.
    ///
    /// With the statistical source on, a seed of its own, distinct from
    /// `base` and from every other replica's, so its transports are an
    /// independent realisation. Off, `base` itself: every replica follows the
    /// nominal run's random numbers, so its spread is the decay data's, with
    /// only the Monte Carlo noise of each step's difference from the nominal
    /// transport left in it.
    pub fn transport_seed(&self, base: u64, replica: u64) -> u64 {
        if !self.statistical {
            return base;
        }
        let key = yamc_rng::secondary_seed(
            yamc_rng::history_seed(self.request.seed, replica),
            TRANSPORT_STREAM,
        );
        yamc_rng::expand_seed(key) ^ base
    }

    /// `chain` with replica `replica`'s decay data, and what it drew.
    pub fn chain(
        &self,
        chain: &Arc<HashMap<String, ChainNuclide>>,
        replica: u64,
    ) -> (Arc<HashMap<String, ChainNuclide>>, Drawn) {
        let seed = self.request.seed;
        let mut floored = BTreeSet::new();
        let mut decay_branchings = HashMap::new();
        for split in self.decay_branchings.iter().flat_map(|b| &b.splits) {
            let mut n = 0;
            decay_branchings.extend(crate::decay_branching_uncertainty::sample(
                std::slice::from_ref(split),
                seed,
                replica,
                &mut n,
            ));
            if n > 0 {
                floored.insert(split.parent.clone());
            }
        }
        let edits = crate::material_transmute::ChainEdits {
            half_lives: crate::uncertainty::sample_half_lives(&self.half_lives, seed, replica),
            decay_branchings,
            fission_yields: self
                .fission_yields
                .as_ref()
                .filter(|f| !f.is_empty())
                .map_or_else(HashMap::new, |f| {
                    crate::fission_yield_uncertainty::sample(f, seed, replica)
                }),
        };
        let drawn = Drawn {
            half_lives: edits.half_lives.clone(),
            decay_branchings: edits.decay_branchings.keys().cloned().collect(),
            floored,
        };
        if edits.is_empty() {
            return (Arc::clone(chain), drawn);
        }
        (Arc::new(edits.apply(chain)), drawn)
    }

    /// Run replicas through `solve` until every material's tracked sigmas
    /// meet the request's convergence, or `samples` of them when it is set,
    /// and report each material's ensemble.
    ///
    /// `solve(replicas, replica, chain)` is replica `replica` of `replicas`'
    /// coupled run with that decay data, returning each material's per-step
    /// inventories; `replicas` is these or, for an attribution, these with
    /// one source alone, and says the transport seed, the cross sections and
    /// the branching table to run with. With `attribution`, each material's
    /// ensemble carries where its uncertainty comes from (see
    /// `attribute`).
    /// `initial` is each material's starting composition with its nuclear
    /// data, which says what the material reaches. `nominal` is each
    /// material's nominal per-step inventories: when nothing a replica draws
    /// moves an inventory, they stand in for every replica, no transport is
    /// run, and the decay energies and photon lines still move what is
    /// evaluated from them.
    pub fn run<E>(
        &self,
        chain: &Arc<HashMap<String, ChainNuclide>>,
        initial: &HashMap<u32, Material>,
        nominal: &HashMap<u32, Inventories>,
        mut solve: impl FnMut(
            &CoupledReplicas,
            u64,
            &Arc<HashMap<String, ChainNuclide>>,
        ) -> Result<HashMap<u32, Inventories>, E>,
    ) -> Result<HashMap<u32, (Ensemble, Info)>, E> {
        let mut reports = self.ensembles(chain, initial, nominal, &mut solve)?;
        if self.request.attribution {
            self.attribute(chain, initial, nominal, &mut solve, &mut reports)?;
        }
        Ok(reports.into_iter().collect())
    }

    /// The sources the request applies, in [`Source::IMPLEMENTED`] order:
    /// every one it names but the flux spectrum, which a transport tallies
    /// rather than draws.
    fn applied(&self) -> Vec<Source> {
        Source::IMPLEMENTED
            .iter()
            .copied()
            .filter(|s| *s != Source::FluxSpectrum && self.request.wants(*s))
            .collect()
    }

    /// These replicas with `source` alone drawn, `samples` of them, on the
    /// same streams: replica `k` of each draws what replica `k` of the whole
    /// request draws of that source. The cross sections keep the statistical
    /// source beside them, as a replica cannot carry them without it.
    fn alone(&self, source: Source, samples: usize) -> Self {
        let mut sources = vec![source];
        if source == Source::CrossSections {
            sources.insert(0, Source::Statistical);
        }
        let mut out = self.clone();
        out.request = DataUncertainty {
            sources,
            samples: Some(samples),
            attribution: false,
            ..self.request.clone()
        };
        out.statistical = out.request.wants(Source::Statistical);
        out.cross_sections = out.request.wants(Source::CrossSections);
        if !out.request.wants(Source::HalfLife) {
            out.half_lives.clear();
        }
        if !out.request.wants(Source::DecayBranching) {
            out.decay_branchings = None;
        }
        if !out.request.wants(Source::FissionYield) {
            out.fission_yields = None;
        }
        if !out.request.wants(Source::IsomerBranching) {
            out.isomer = None;
        }
        out
    }

    /// Where each material's uncertainty comes from: each applied source's
    /// variance from an ensemble of coupled runs drawing that source alone,
    /// as many replicas as the whole request ran and on the same streams.
    ///
    /// The cross sections are measured beside the statistical source, so
    /// their entry is the variance with both less the variance with the
    /// statistical source alone. The two ensembles share their transport
    /// seeds replica by replica, which keeps the difference's noise down; it
    /// is still the difference of two estimates, and where the cross
    /// sections move a nuclide by less than the transport's noise it can come
    /// out negative, and is reported so rather than hidden. A single applied
    /// source is the whole ensemble. There are no first-order contributors:
    /// a coupled replica's sensitivity is a coupled run of its own, and
    /// `linearity` is `None` throughout.
    fn attribute<E>(
        &self,
        chain: &Arc<HashMap<String, ChainNuclide>>,
        initial: &HashMap<u32, Material>,
        nominal: &HashMap<u32, Inventories>,
        solve: &mut impl FnMut(
            &CoupledReplicas,
            u64,
            &Arc<HashMap<String, ChainNuclide>>,
        ) -> Result<HashMap<u32, Inventories>, E>,
        reports: &mut BTreeMap<u32, (Ensemble, Info)>,
    ) -> Result<(), E> {
        type Variances = Vec<HashMap<String, f64>>;
        let variances = |r: &BTreeMap<u32, (Ensemble, Info)>| -> BTreeMap<u32, Variances> {
            r.iter()
                .map(|(id, (e, _))| {
                    let steps = nominal.get(id).map_or(0, Vec::len);
                    (*id, crate::material_transmute::variances_of(e, steps))
                })
                .collect()
        };
        let applied = self.applied();
        let samples = reports
            .values()
            .map(|(e, _)| e.replicas())
            .max()
            .unwrap_or(0);
        let mut by_source: BTreeMap<String, BTreeMap<u32, Variances>> = BTreeMap::new();
        if applied.len() == 1 {
            by_source.insert(applied[0].name().to_string(), variances(reports));
        } else {
            let mut statistical: Option<BTreeMap<u32, Variances>> = None;
            for &source in &applied {
                let v = variances(
                    &self
                        .alone(source, samples)
                        .ensembles(chain, initial, nominal, solve)?,
                );
                if source == Source::Statistical {
                    statistical = Some(v.clone());
                }
                by_source.insert(source.name().to_string(), v);
            }
            if let Some(both) = by_source.get_mut(Source::CrossSections.name()) {
                let stat = match statistical {
                    Some(s) => s,
                    None => variances(
                        &self
                            .alone(Source::Statistical, samples)
                            .ensembles(chain, initial, nominal, solve)?,
                    ),
                };
                for (id, steps) in both.iter_mut() {
                    for (step, per) in steps.iter_mut().enumerate() {
                        let alone = stat.get(id).and_then(|s| s.get(step));
                        let names: BTreeSet<String> = per
                            .keys()
                            .chain(alone.into_iter().flat_map(|a| a.keys()))
                            .cloned()
                            .collect();
                        for name in names {
                            let s = alone.and_then(|a| a.get(&name)).copied().unwrap_or(0.0);
                            *per.entry(name).or_insert(0.0) -= s;
                        }
                    }
                }
            }
        }
        let mut linearity: BTreeMap<String, Option<Vec<HashMap<String, Linearity>>>> = applied
            .iter()
            .map(|s| (s.name().to_string(), None))
            .collect();
        linearity.insert("all".to_string(), None);
        for (id, (ensemble, _)) in reports.iter_mut() {
            ensemble.attribution = Some(Attribution {
                by_source: by_source
                    .iter()
                    .map(|(s, v)| (s.clone(), v.get(id).cloned().unwrap_or_default()))
                    .collect(),
                contributors: Vec::new(),
                linearity: linearity.clone(),
            });
        }
        Ok(())
    }

    /// One ensemble per material of these replicas, run through `solve`.
    fn ensembles<E>(
        &self,
        chain: &Arc<HashMap<String, ChainNuclide>>,
        initial: &HashMap<u32, Material>,
        nominal: &HashMap<u32, Inventories>,
        solve: &mut impl FnMut(
            &CoupledReplicas,
            u64,
            &Arc<HashMap<String, ChainNuclide>>,
        ) -> Result<HashMap<u32, Inventories>, E>,
    ) -> Result<BTreeMap<u32, (Ensemble, Info)>, E> {
        let request = &self.request;
        let moves = self.moves_inventory();
        let mut reports: BTreeMap<u32, (Ensemble, Info)> = initial
            .iter()
            .map(|(&id, material)| {
                let steps = nominal.get(&id).map_or(0, Vec::len);
                (id, self.report(material, chain, Ensemble::new(steps)))
            })
            .collect();
        let decay_only = reports
            .values()
            .any(|(e, _)| e.decay_energy_seed.is_some() || e.decay_photon_seed.is_some());
        if !moves && !decay_only {
            for (_, info) in reports.values_mut() {
                info.converged = true;
                info.convergence = request.convergence;
            }
            return Ok(reports);
        }
        let cap = request.samples.unwrap_or(MAX_SAMPLES);
        let mut replica: u64 = 0;
        while (replica as usize) < cap {
            let end = (replica as usize + BLOCK).min(cap) as u64;
            while replica < end {
                let (inventories, drawn) = if moves {
                    let (edited, drawn) = self.chain(chain, replica);
                    (solve(self, replica, &edited)?, drawn)
                } else {
                    (nominal.clone(), Drawn::default())
                };
                push(&mut reports, &inventories, &drawn);
                replica += 1;
            }
            if request.samples.is_none()
                && reports
                    .values_mut()
                    .all(|(e, _)| e.may_stop(chain, request.convergence))
            {
                break;
            }
        }
        for (ensemble, info) in reports.values_mut() {
            ensemble.fold_absences();
            info.conclude(ensemble, chain, request);
        }
        Ok(reports)
    }

    /// What a replica of `initial` perturbs and what it holds, before any
    /// replica is run, and the empty ensemble with the decay seeds set.
    fn report(
        &self,
        initial: &Material,
        chain: &HashMap<String, ChainNuclide>,
        mut ensemble: Ensemble,
    ) -> (Ensemble, Info) {
        let request = &self.request;
        let mut info = Info::from_fold(&Default::default(), &Default::default(), false)
            .expect("an empty fold has no rate fraction to disagree with");
        info.flux_response = FluxResponseReport::CoupledReplicas;
        // Restricted to what this material reaches, as independent mode
        // reports it; the draws themselves cover the whole chain.
        let reach = crate::material_transmute::reachable_chain(initial, chain);
        if request.wants(Source::HalfLife) {
            let (with, without, not_carried) = crate::uncertainty::half_life_candidates(&reach);
            info.half_lives_perturbed = with.into_iter().map(|(n, _, _)| n).collect();
            info.no_half_life_uncertainty = without;
            info.half_life_uncertainty_not_carried = not_carried;
        } else {
            info.not_perturbed.insert(0, "half-life".to_string());
        }
        if request.wants(Source::DecayBranching) {
            let b = crate::decay_branching_uncertainty::candidates(&reach);
            if !b.splits.is_empty() {
                info.not_perturbed.insert(
                    0,
                    "decay emission per branch (line intensities and decay energy \
                     follow the nominal branching)"
                        .to_string(),
                );
            }
            info.decay_branchings_perturbed = b.splits.iter().map(|t| t.parent.clone()).collect();
            info.no_decay_branching_uncertainty = b.without;
            info.decay_branchings_too_wide = b.too_wide;
            info.decay_branchings_partly_held = b.partly_held;
            info.decay_branchings_sigma_adjusted = b.sigma_adjusted;
        } else {
            info.not_perturbed
                .insert(0, "decay branching ratio".to_string());
        }
        if request.wants(Source::DecayEnergy) {
            let (with, without, not_carried) = decay_energy_candidates(initial, chain);
            if !with.is_empty() {
                ensemble.decay_energy_seed = Some(request.seed);
            }
            info.decay_energies_perturbed = with;
            info.no_decay_energy_uncertainty = without;
            info.decay_energy_uncertainty_not_carried = not_carried;
        } else {
            info.not_perturbed.insert(0, "decay energy".to_string());
        }
        if request.wants(Source::FissionYield) {
            let f = crate::fission_yield_uncertainty::candidates(&reach, chain);
            info.fission_yields_perturbed = f.perturbed;
            info.no_fission_yield_uncertainty = f.without;
            info.fission_yield_uncertainty_not_carried = f.not_carried;
            info.fission_yields_mapping_mismatch = f.mapping_mismatch;
        } else {
            info.not_perturbed.insert(0, "fission yield".to_string());
        }
        if request.wants(Source::DecayPhotonLines) {
            let (with, without, not_carried) = decay_photon_candidates(initial, chain);
            if !with.is_empty() {
                ensemble.decay_photon_seed = Some(request.seed);
            }
            info.decay_photon_spectra_folded = with
                .iter()
                .filter_map(|name| {
                    let folded = crate::uncertainty::folded_photon_spectra(chain.get(name)?);
                    (!folded.is_empty()).then(|| (name.clone(), folded))
                })
                .collect();
            info.decay_photon_lines_perturbed = with;
            info.no_decay_photon_line_uncertainty = without;
            info.decay_photon_line_uncertainty_not_carried = not_carried;
        } else {
            info.not_perturbed.insert(
                0,
                "decay photon line energy and intensity, and continuum normalisation".to_string(),
            );
        }
        if !self.cross_sections {
            info.not_perturbed
                .insert(0, "activation cross section (MF=33)".to_string());
        }
        match &self.isomer {
            Some(isomer) => {
                let reachable = reach.keys().cloned().collect();
                let r = isomer.report(&reachable);
                info.isomer_branchings_perturbed = r.perturbed;
                info.isomer_branching_blocks_not_used = r.not_used;
                info.isomer_branching_covariance_repaired = r.repaired;
                info.isomer_branching_lognormal_not_carried = r.lognormal_not_carried;
            }
            None => info.not_perturbed.insert(
                0,
                "isomeric branching covariance (MF=40), every split held".to_string(),
            ),
        }
        if !self.statistical {
            info.not_perturbed
                .insert(0, "tallied-rate statistics".to_string());
        }
        // The screening that picks which products' reactions are tallied runs
        // once, on the nominal scouting transport, and every replica tallies
        // that same set.
        info.not_perturbed.push(
            "transmutation product screening (made once, from the nominal scouting \
             transport)"
                .to_string(),
        );
        let requested: &[Source] = if request.sources.is_empty() {
            Source::IMPLEMENTED
        } else {
            &request.sources
        };
        info.sources = requested
            .iter()
            .filter(|s| **s != Source::FluxSpectrum)
            .map(|s| s.name().to_string())
            .collect();
        (ensemble, info)
    }
}

/// One replica's per-step inventories and decay-data draw, into every
/// material's ensemble and report.
fn push(
    reports: &mut BTreeMap<u32, (Ensemble, Info)>,
    inventories: &HashMap<u32, Inventories>,
    drawn: &Drawn,
) {
    for (id, (ensemble, info)) in reports.iter_mut() {
        // Counted over what this material reaches, as independent mode
        // counts them; the draw covers the whole chain.
        let half_lives: HashMap<String, f64> = drawn
            .half_lives
            .iter()
            .filter(|(n, _)| info.half_lives_perturbed.contains(*n))
            .map(|(n, t)| (n.clone(), *t))
            .collect();
        let mine =
            |set: &BTreeSet<String>| set.intersection(&info.decay_branchings_perturbed).count();
        info.half_lives_sampled += half_lives.len();
        info.decay_branchings_sampled += mine(&drawn.decay_branchings);
        info.decay_branchings_floored += mine(&drawn.floored);
        let per_step = inventories.get(id).cloned().unwrap_or_default();
        ensemble.push_with_half_lives(per_step, half_lives);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(sources: Vec<Source>) -> DataUncertainty {
        DataUncertainty {
            sources,
            ..Default::default()
        }
    }

    /// Every source is carried, the default request and attribution
    /// included; the cross sections only beside the statistical source.
    #[test]
    fn what_a_replica_carries() {
        let chain = HashMap::new();
        let branch = yani::BranchTable::new();
        let new = |r: &DataUncertainty| CoupledReplicas::new(r, &chain, &branch);
        for sources in [
            vec![],
            vec![Source::Statistical],
            vec![Source::Statistical, Source::CrossSections],
            vec![Source::IsomerBranching],
        ] {
            assert!(new(&request(sources.clone())).is_ok(), "{sources:?}");
        }
        let both = new(&request(vec![Source::Statistical, Source::CrossSections])).unwrap();
        assert!(both.cross_sections() && both.moves_inventory());
        let err = new(&request(vec![Source::CrossSections]))
            .err()
            .expect("refused");
        assert!(err.contains("\"statistical\""), "{err}");
        let attribution = DataUncertainty {
            attribution: true,
            ..request(vec![Source::Statistical, Source::CrossSections])
        };
        assert!(new(&attribution).is_ok());
        let isomer = new(&request(vec![Source::IsomerBranching])).unwrap();
        assert!(
            !isomer.moves_inventory(),
            "no MF=40 to draw in an empty table"
        );
    }

    /// With the replicas riding one transport as weights, the sources a
    /// shared transport cannot carry are refused by name, the cross sections
    /// are carried without the statistical source, and every step's
    /// transport carries the request's replica count.
    #[test]
    fn what_weighted_replicas_carry() {
        let chain = HashMap::new();
        let branch = yani::BranchTable::new();
        let weighted = |sources: Vec<Source>| DataUncertainty {
            samples: Some(16),
            coupled_replica_weights: true,
            ..request(sources)
        };
        for (sources, named) in [
            (
                vec![Source::Statistical, Source::CrossSections],
                "\"statistical\"",
            ),
            (vec![Source::IsomerBranching], "\"isomer_branching\""),
            (vec![], "\"statistical\""),
        ] {
            let err = CoupledReplicas::new(&weighted(sources), &chain, &branch)
                .err()
                .expect("refused");
            assert!(
                err.contains(named) && err.contains("coupled_replica_weights"),
                "{err}"
            );
        }
        let attribution = DataUncertainty {
            attribution: true,
            ..weighted(vec![Source::CrossSections])
        };
        assert!(CoupledReplicas::new(&attribution, &chain, &branch).is_err());
        let xs = CoupledReplicas::new(&weighted(vec![Source::CrossSections]), &chain, &branch)
            .expect("carried");
        assert_eq!(xs.replica_weights(), Some(16));
        let reruns = CoupledReplicas::new(
            &request(vec![Source::Statistical, Source::CrossSections]),
            &chain,
            &branch,
        )
        .unwrap();
        assert_eq!(reruns.replica_weights(), None);
    }

    /// A source alone keeps the streams of the whole request and draws
    /// nothing else; the cross sections keep the statistical source.
    #[test]
    fn a_source_alone() {
        let chain = HashMap::new();
        let branch = yani::BranchTable::new();
        let all = CoupledReplicas::new(&request(vec![]), &chain, &branch).unwrap();
        let xs = all.alone(Source::CrossSections, 8);
        assert!(xs.statistical && xs.cross_sections);
        assert_eq!(xs.request.samples, Some(8));
        assert!(!xs.request.attribution);
        let stat = all.alone(Source::Statistical, 8);
        assert!(stat.statistical && !stat.cross_sections);
        for k in 0..8 {
            assert_eq!(xs.transport_seed(3, k), stat.transport_seed(3, k));
            assert_eq!(xs.transport_seed(3, k), all.transport_seed(3, k));
        }
        let half = all.alone(Source::HalfLife, 8);
        assert!(!half.statistical && !half.cross_sections && half.isomer.is_none());
        assert_eq!(
            all.applied(),
            Source::IMPLEMENTED
                .iter()
                .copied()
                .filter(|s| *s != Source::FluxSpectrum)
                .collect::<Vec<_>>()
        );
    }

    /// With the statistical source on every replica's transport seed is its
    /// own and none is the nominal's; off, every replica keeps the nominal
    /// random numbers.
    #[test]
    fn transport_seeds() {
        let chain = HashMap::new();
        let branch = yani::BranchTable::new();
        let on =
            CoupledReplicas::new(&request(vec![Source::Statistical]), &chain, &branch).unwrap();
        let seeds: std::collections::HashSet<u64> =
            (0..4096).map(|k| on.transport_seed(7, k)).collect();
        assert_eq!(seeds.len(), 4096);
        assert!(!seeds.contains(&7));
        assert!(on.moves_inventory());
        let off = CoupledReplicas::new(&request(vec![Source::HalfLife]), &chain, &branch).unwrap();
        assert!((0..8).all(|k| off.transport_seed(7, k) == 7));
        assert!(
            !off.moves_inventory(),
            "no half-life to draw in an empty chain"
        );
    }
}
