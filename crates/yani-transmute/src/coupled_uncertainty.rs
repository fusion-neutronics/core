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
//! What a replica cannot yet carry is refused rather than held at nominal:
//! `isomer_branching`, which would need every replica's transport to read its
//! own MF=40 production curves, and `attribution`, which would need one
//! coupled ensemble per source.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use yamc_materials::Material;
use yani::ChainNuclide;

use crate::material_transmute::{decay_energy_candidates, decay_photon_candidates};
use crate::uncertainty::{
    check_convergence, DataUncertainty, Ensemble, FluxResponseReport, Info, Source, BLOCK,
    MAX_SAMPLES,
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
pub struct CoupledReplicas {
    request: DataUncertainty,
    statistical: bool,
    cross_sections: bool,
    half_lives: Vec<crate::uncertainty::HalfLifeSigma>,
    decay_branchings: Option<crate::decay_branching_uncertainty::Candidates>,
    fission_yields: Option<crate::fission_yield_uncertainty::Candidates>,
}

impl CoupledReplicas {
    /// The draws `request` asks of a coupled run over `chain`, or an error
    /// naming what a coupled replica does not yet carry.
    pub fn new(
        request: &DataUncertainty,
        chain: &HashMap<String, ChainNuclide>,
    ) -> Result<Self, String> {
        check_convergence(request.convergence)?;
        if request.wants(Source::IsomerBranching) {
            let carried: Vec<&str> = Source::IMPLEMENTED
                .iter()
                .filter(|s| **s != Source::IsomerBranching)
                .map(|s| s.name())
                .collect();
            return Err(format!(
                "data_uncertainty with method=\"coupled\" runs the whole coupled \
                 calculation once per replica, and a replica's transport does not yet read \
                 its own MF=40 isomer production curves, so \"isomer_branching\" cannot be \
                 carried (an empty sources list asks for every source). Name the sources: \
                 with the coupled method these are {carried:?}; method=\"independent\" \
                 carries every source"
            ));
        }
        let cross_sections = request.wants(Source::CrossSections);
        let statistical = request.wants(Source::Statistical);
        if cross_sections && !statistical {
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
        if request.attribution {
            return Err(
                "data_uncertainty with method=\"coupled\" does not yet attribute \
                        the uncertainty by source, which would take one coupled ensemble per \
                        source; run one source at a time to measure each"
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
            half_lives,
            decay_branchings: request
                .wants(Source::DecayBranching)
                .then(|| crate::decay_branching_uncertainty::candidates(chain)),
            fission_yields: request
                .wants(Source::FissionYield)
                .then(|| crate::fission_yield_uncertainty::candidates(chain, chain)),
        })
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
    /// `solve(replica, chain)` is replica `replica` of the coupled run with
    /// that decay data, returning each material's per-step inventories.
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
            u64,
            &Arc<HashMap<String, ChainNuclide>>,
        ) -> Result<HashMap<u32, Inventories>, E>,
    ) -> Result<HashMap<u32, (Ensemble, Info)>, E> {
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
            return Ok(reports.into_iter().collect());
        }
        let cap = request.samples.unwrap_or(MAX_SAMPLES);
        let mut replica: u64 = 0;
        while (replica as usize) < cap {
            let end = (replica as usize + BLOCK).min(cap) as u64;
            while replica < end {
                let (inventories, drawn) = if moves {
                    let (edited, drawn) = self.chain(chain, replica);
                    (solve(replica, &edited)?, drawn)
                } else {
                    (nominal.clone(), Drawn::default())
                };
                for (id, (ensemble, info)) in reports.iter_mut() {
                    // Counted over what this material reaches, as independent
                    // mode counts them; the draw covers the whole chain.
                    let half_lives: HashMap<String, f64> = drawn
                        .half_lives
                        .iter()
                        .filter(|(n, _)| info.half_lives_perturbed.contains(*n))
                        .map(|(n, t)| (n.clone(), *t))
                        .collect();
                    let mine = |set: &BTreeSet<String>| {
                        set.intersection(&info.decay_branchings_perturbed).count()
                    };
                    info.half_lives_sampled += half_lives.len();
                    info.decay_branchings_sampled += mine(&drawn.decay_branchings);
                    info.decay_branchings_floored += mine(&drawn.floored);
                    let per_step = inventories.get(id).cloned().unwrap_or_default();
                    ensemble.push_with_half_lives(per_step, half_lives);
                }
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
        Ok(reports.into_iter().collect())
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
        info.not_perturbed.insert(
            0,
            "isomeric branching covariance (MF=40), every split held".to_string(),
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn request(sources: Vec<Source>) -> DataUncertainty {
        DataUncertainty {
            sources,
            ..Default::default()
        }
    }

    /// A source a coupled replica cannot carry is refused by name, and so is
    /// the default request, which asks for every source. The cross sections
    /// are carried beside the statistical source and refused without it.
    #[test]
    fn what_a_replica_cannot_carry_is_refused() {
        let chain = HashMap::new();
        for sources in [
            vec![],
            vec![Source::CrossSections],
            vec![Source::Statistical, Source::IsomerBranching],
        ] {
            let err = CoupledReplicas::new(&request(sources), &chain)
                .err()
                .expect("refused");
            assert!(err.contains("method=\"coupled\""), "{err}");
        }
        let attribution = DataUncertainty {
            attribution: true,
            ..request(vec![Source::Statistical])
        };
        assert!(CoupledReplicas::new(&attribution, &chain).is_err());
        assert!(CoupledReplicas::new(&request(vec![Source::Statistical]), &chain).is_ok());
        let both = request(vec![Source::Statistical, Source::CrossSections]);
        let carried = CoupledReplicas::new(&both, &chain).expect("carried");
        assert!(carried.cross_sections() && carried.moves_inventory());
        let err = CoupledReplicas::new(&request(vec![Source::CrossSections]), &chain)
            .err()
            .expect("refused");
        assert!(err.contains("\"statistical\""), "{err}");
    }

    /// With the statistical source on every replica's transport seed is its
    /// own and none is the nominal's; off, every replica keeps the nominal
    /// random numbers.
    #[test]
    fn transport_seeds() {
        let chain = HashMap::new();
        let on = CoupledReplicas::new(&request(vec![Source::Statistical]), &chain).unwrap();
        let seeds: std::collections::HashSet<u64> =
            (0..4096).map(|k| on.transport_seed(7, k)).collect();
        assert_eq!(seeds.len(), 4096);
        assert!(!seeds.contains(&7));
        assert!(on.moves_inventory());
        let off = CoupledReplicas::new(&request(vec![Source::HalfLife]), &chain).unwrap();
        assert!((0..8).all(|k| off.transport_seed(7, k) == 7));
        assert!(
            !off.moves_inventory(),
            "no half-life to draw in an empty chain"
        );
    }
}
