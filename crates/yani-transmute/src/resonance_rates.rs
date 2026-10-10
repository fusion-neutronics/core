//! Rates moved by sampled resonance parameters, exact in the parameters.
//!
//! Where a nuclide's library folder carries `resonance_parameters.arrow`
//! (MF=2 and MF=32), a replica draws each resonance range's parameter vector
//! from its covariance ([`ResonanceSampler`]), rebuilds the range's cross
//! sections from it ([`reconstruction_at`]) and moves each channel's rate by
//! what the rebuilt cross section changes it by. The first-order MF=32 rows
//! the converter writes into `covariance.arrow` (`subsection_idx = -1`) are
//! then left out of that nuclide's MF=33 draw, so the resonance range's
//! uncertainty is counted once, from the parameters.
//!
//! # Why this is exact
//!
//! A rate is linear in the cross section (the convention of
//! [`crate::covariance_fold`]):
//!
//! ```text
//! R = 1e-24 ∫ σ_T(E) ψ(E) dE
//! ```
//!
//! with `σ_T` the library's cross section at the temperature in use and `ψ`
//! the collapse's flux density: `φ_g / ΔE_g` flat in each group, `φ_g / (E
//! ln(E_hi / E_lo))` under the `1/E` within-group weight, and on a
//! self-shielded run `φ_g s(E) / ∫_g s dE` with the nominal shape `s`. The
//! library's `σ_T` is NJOY's BROADR applied to the 0 K reconstruction, which
//! is linear, so a sampled parameter vector changes the rate by exactly
//!
//! ```text
//! ΔR = 1e-24 ∫ B_T[σ'_0 - σ_0](E) ψ(E) dE = 1e-24 ∫ (σ'_0 - σ_0)(E) w_T(E) dE
//! ```
//!
//! where `σ'_0` and `σ_0` are the perturbed and nominal 0 K reconstructions
//! and `w_T` the broadened flux weight of [`broadened_weight`], the adjoint
//! of the broadening. Nothing in that is linearized in the parameters: the
//! cross section is rebuilt from the drawn vector, and the only
//! approximations are the integration grid ([`reconstruction_grid`], which
//! traces every nominal and every perturbed resonance) and the weight's
//! polynomial fit, both far below a sampling error. `w_T` depends on the
//! spectrum, the nuclide and the temperature only, so it is built once per
//! run and every replica costs a reconstruction and an integral.
//!
//! Broadening stops where BROADR stops: above `thnmax` the PENDF carries the
//! 0 K cross section and `w_T` is the flux itself. Every library yani reads
//! is processed with BROADR's card 3 giving only `errthn` (the deck
//! `endf::njoy` writes, and OpenMC's, which it mirrors), so `thnmax` is
//! BROADR's default since NJOY2012.75: the top of the resolved range, or the
//! bottom of the unresolved one where nothing is resolved, and at most
//! 6.5 MeV ([`thnmax`]). That is per nuclide, and read off its own MF=2.
//! The temperature is the label the rates were read at, in kelvin.
//!
//! The change is added to the rate the MF=33 draw gives, as a shift of the
//! same ratio the MF=33 draw multiplies it by: the replica's rate is
//! `R_flux (ratio_33 + ΔR / R)`, `R_flux` the rate after the flux draw, which
//! is the product the MF=33 path already takes between a flux and a
//! cross-section perturbation. ENDF-102 section 32 states the MF=32
//! contribution as independent of MF=33's and to be added to it, and the two
//! draws are independent: the parameters come off a stream of their own
//! ([`crate::resonance_sampling::resonance_deviates`]). A rate the two take
//! below zero is floored and counted in `rates_floored`, as the MF=33 path
//! floors its own.
//!
//! A range's reactions reach the channels the converter's first-order rows
//! reach: elastic is MT 2, capture MT 102, fission MT 18 and an R-matrix
//! limited range's other exit pairs their own MTs (600 for a proton, 51 for
//! an inelastic neutron). A channel reads a reaction whose MT is its own, or
//! a level of its own (`(n,n')` MT 4 reads MT 51, `(n,p)` MT 103 reads MT
//! 600). An isotope of a natural element contributes its abundance's share.
//!
//! # What stays first order
//!
//! A nuclide falls back to the first-order rows, and is reported with the
//! reason, when its parameters cannot be sampled and rebuilt: MF=32 that does
//! not match MF=2 (FENDL-3.2d La138 lists an unresolved `l` MF=2 does not
//! have), a covariance with a negative variance (JEFF-4.0 Pa233), a
//! formalism `endf` does not reconstruct (single-level Breit-Wigner), or a
//! temperature label that is not a number. The run goes on either way.
//!
//! Everything that reads the evaluation's statement rather than a replica
//! keeps the rows, as the first-order image of the same source: the fold's
//! coverage (so the resonance range counts as covered, which it is: its
//! uncertainty is sampled), the sigma report's evaluated and sampled channel
//! sigmas and repairs, and first-order attribution's sensitivities and
//! contributors. Attribution's linearity check reads each replica's actual
//! rate changes, parameters included, so it measures how far the inventory
//! is from linear in the rates this run drew.
//!
//! On a self-shielded run the weight is taken under the nominal shielded
//! shape, as the MF=33 fold takes its partials: a perturbed resonance does
//! not deepen its own flux dip here. A transport-coupled run keeps the
//! first-order rows for every nuclide.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use endf::doppler::{broadened_weight, BroadenedWeight, Flux};
use endf::mf::mf2::ResonanceRange;
use endf::resonance::{cross_sections_on, CrossSections, RangeReconstruction, REACTIONS};
use endf::resonance_covariance::{reconstruction_at, reconstruction_grid, RangeCovariance};
use yamc_materials::Material;
use yamc_nuclide::resonance_parameters::ResonanceParameters;
use yani::reactions::reaction_type_to_mt;
use yani::ReactionRates;

use crate::covariance_fold::level_sum;
use crate::covariance_sample::RateShifts;
use crate::multigroup::{CollapseShapes, Weighting};
use crate::resonance_sampling::{resonance_deviates, ResonanceSampler, SamplerReport};
use crate::self_shielding::{FluxShape, Shielding};

/// BROADR's ceiling on its default `thnmax`, 6.5 MeV.
pub const BROADR_CEILING_EV: f64 = 6.5e6;

/// How one nuclide's resonance-parameter covariance reached the replicas.
#[derive(Debug, Clone, PartialEq)]
pub enum ResonanceMethod {
    /// Every range's parameters drawn per replica and the cross sections
    /// rebuilt; the first-order MF=32 rows are out of its MF=33 draw. One
    /// report per range, in the order MF=32 lists them.
    Sampled { ranges: Vec<SamplerReport> },
    /// The first-order MF=32 rows of `covariance.arrow`, where the converter
    /// wrote any, sampled with MF=33; why the parameters were not.
    FirstOrder { reason: String },
}

impl ResonanceMethod {
    /// How far the correlations the draws have are from the evaluated ones,
    /// over every range: the largest single change and the root-sum-square
    /// of the ranges' Frobenius changes (see
    /// [`SamplerReport::correlation_change`]). `(0, 0)` for the first-order
    /// rows. Widths stay lognormal whatever this says: a pair of widths no
    /// lognormal carries is where it is largest.
    pub fn correlation_change(&self) -> (f64, f64) {
        let ResonanceMethod::Sampled { ranges } = self else {
            return (0.0, 0.0);
        };
        ranges
            .iter()
            .fold((0.0_f64, 0.0_f64), |(largest, frobenius), r| {
                (
                    largest.max(r.correlation_change),
                    frobenius.hypot(r.correlation_frobenius_change),
                )
            })
    }

    /// `"parameters sampled"` or `"first-order rows"`.
    pub fn name(&self) -> &'static str {
        match self {
            ResonanceMethod::Sampled { .. } => "parameters sampled",
            ResonanceMethod::FirstOrder { .. } => "first-order rows",
        }
    }
}

/// BROADR's default upper limit of broadening for an evaluation with these
/// ranges: the top of the resolved range, or the bottom of the unresolved one
/// where nothing is resolved, capped at [`BROADR_CEILING_EV`]. `None` with
/// neither, where BROADR broadens to the first threshold, which no range here
/// reaches.
pub fn thnmax<'a>(ranges: impl IntoIterator<Item = &'a ResonanceRange>) -> Option<f64> {
    let mut resolved_top: Option<f64> = None;
    let mut unresolved_bottom: Option<f64> = None;
    for r in ranges {
        match r.lru {
            1 => resolved_top = Some(resolved_top.map_or(r.eh, |t| t.max(r.eh))),
            2 => unresolved_bottom = Some(unresolved_bottom.map_or(r.el, |b| b.min(r.el))),
            _ => {}
        }
    }
    resolved_top
        .or(unresolved_bottom)
        .map(|e| e.min(BROADR_CEILING_EV))
}

/// One range ready to draw: its sampler, and the nominal reconstruction on
/// the grid that traces it alone, which every replica's grid holds.
struct PreparedRange {
    /// The range's position in the list `range_covariances` returns, which
    /// keys its deviates.
    index: usize,
    cov: RangeCovariance,
    range: ResonanceRange,
    abundance: f64,
    sampler: ResonanceSampler,
    report: SamplerReport,
    nominal: Box<dyn RangeReconstruction + Send + Sync>,
    edges: [f64; 2],
    /// The MT each of the reconstruction's reaction slots stands for.
    slot_mts: [Option<i32>; REACTIONS],
    nominal_grid: Vec<f64>,
    nominal_values: Vec<CrossSections>,
}

impl PreparedRange {
    /// The nominal cross sections on `grid`, read off the cached ones where
    /// a point is one of the nominal grid's and rebuilt elsewhere.
    fn nominal_on(&self, grid: &[f64]) -> Vec<CrossSections> {
        let mut j = 0;
        grid.iter()
            .map(|&e| {
                while j < self.nominal_grid.len() && self.nominal_grid[j] < e {
                    j += 1;
                }
                if j < self.nominal_grid.len() && self.nominal_grid[j] == e {
                    self.nominal_values[j]
                } else {
                    self.nominal.cross_sections(e)
                }
            })
            .collect()
    }
}

/// One nuclide's ranges, ready to draw, and what broadening needs.
struct PreparedNuclide {
    ranges: Vec<PreparedRange>,
    awr: f64,
    broaden_below: Option<f64>,
    /// The span of every range, for cutting the flux to what reaches it.
    span: (f64, f64),
}

type Preparation = Result<Arc<PreparedNuclide>, String>;

/// How many nuclides' preparations the process keeps before it starts over.
const PREPARED_CACHE: usize = 64;

/// Preparations already made, keyed by the data they were made from, so a
/// nuclide is prepared once per process however many materials, cases and
/// attribution sub-runs read it. Repairing a range can take most of a
/// minute (ENDF/B-VIII.1 Pu240).
static PREPARED: Mutex<Vec<(Arc<ResonanceParameters>, Preparation)>> = Mutex::new(Vec::new());

fn prepared(name: &str, data: &Arc<ResonanceParameters>) -> Preparation {
    {
        let cache = PREPARED.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, p)) = cache
            .iter()
            .find(|(d, _)| Arc::ptr_eq(d, data) || **d == **data)
        {
            return p.clone();
        }
    }
    let made = prepare(name, data).map(Arc::new);
    let mut cache = PREPARED.lock().unwrap_or_else(|p| p.into_inner());
    if cache.len() >= PREPARED_CACHE {
        cache.clear();
    }
    cache.push((Arc::clone(data), made.clone()));
    made
}

/// Parse, match, build every range's sampler and nominal reconstruction, and
/// rebuild one draw of each to find any refusal before a replica can.
fn prepare(name: &str, data: &ResonanceParameters) -> Result<PreparedNuclide, String> {
    let (mf2, mf32) = data
        .parse()
        .map_err(|e| format!("MF=2 or MF=32 does not parse: {e}"))?;
    let covariances = endf::resonance_covariance::range_covariances(&mf2, &mf32)
        .map_err(|e| format!("MF=32 does not match MF=2: {e}"))?;
    let all_ranges = mf2.isotopes.iter().flat_map(|i| i.ranges.iter());
    let broaden_below = thnmax(all_ranges);
    let mut ranges = Vec::new();
    for (index, cov) in covariances.into_iter().enumerate() {
        if cov.is_empty() {
            continue;
        }
        let isotope = &mf2.isotopes[cov.isotope];
        let range = isotope.ranges[cov.mf2_range].clone();
        let where_ = format!("range {:.4e} to {:.4e} eV", range.el, range.eh);
        let (sampler, report) =
            ResonanceSampler::new(&cov).map_err(|e| format!("{where_}: {e}"))?;
        let nominal = endf::resonance::reconstruction(&range)
            .map_err(|e| format!("{where_} (LRU={}, LRF={}): {e}", range.lru, range.lrf))?;
        // One draw rebuilt here, so a parameter with no place in the range is
        // found now rather than in a replica. Every parameter that is not
        // held moves in it, so it meets every field a later draw can.
        let probe = sampler.sample(&resonance_deviates(0, 0, name, index, sampler.dimension()));
        reconstruction_at(&range, &cov, &probe)
            .map_err(|e| format!("{where_}: a drawn parameter vector cannot be rebuilt: {e}"))?;
        let (lo, hi) = nominal.bounds();
        let edges = [lo, hi];
        let nominal_grid = reconstruction_grid(&[nominal.as_ref()], &edges)
            .map_err(|e| format!("{where_}: {e}"))?;
        let nominal_values = cross_sections_on(nominal.as_ref(), &nominal_grid);
        let others = nominal.other_reactions();
        ranges.push(PreparedRange {
            index,
            cov,
            abundance: isotope.abn,
            sampler,
            report,
            edges,
            slot_mts: [Some(2), Some(102), Some(18), others[0], others[1]],
            nominal_grid,
            nominal_values,
            nominal,
            range,
        });
    }
    if ranges.is_empty() {
        return Err("MF=32 states no parameter covariance".to_string());
    }
    let span = ranges.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), r| {
        (lo.min(r.edges[0]), hi.max(r.edges[1]))
    });
    Ok(PreparedNuclide {
        ranges,
        awr: mf2.awr,
        broaden_below,
        span,
    })
}

/// Whether a channel of reaction `channel_mt` could read a reaction some
/// range gives: elastic, capture, fission, or an R-matrix exit pair (a level
/// of `(n,n')` or of a charged-particle emission) or the sum it is a level of.
fn could_read(channel_mt: i32) -> bool {
    matches!(
        channel_mt,
        2 | 4 | 18 | 19..=21 | 38 | 51..=91 | 102..=107 | 600..=849
    )
}

/// Whether a channel of reaction `channel_mt` reads the reaction `mt` a
/// range gives: its own MT, or a level of it.
fn reads(channel_mt: i32, mt: i32) -> bool {
    mt == channel_mt || level_sum(mt) == Some(channel_mt)
}

/// One channel a spectrum reads off a sampled nuclide.
struct Channel {
    kind: String,
    mt: i32,
    /// The nominal rate the shift is relative to.
    rate: f64,
}

/// How one spectrum reads one sampled nuclide.
struct SpectrumRead {
    weight: BroadenedWeight,
    channels: Vec<Channel>,
}

struct RunNuclide {
    name: String,
    prepared: Arc<PreparedNuclide>,
    /// Per spectrum, `None` where it reads no channel the ranges give.
    spectra: Vec<Option<SpectrumRead>>,
}

/// Per spectrum, each sampled nuclide's relative rate shifts in one replica.
pub(crate) type ReplicaShifts = Vec<RateShifts>;

/// The resonance-parameter sampling of one run.
pub(crate) struct ResonanceRun {
    seed: u64,
    n_spectra: usize,
    nuclides: Vec<RunNuclide>,
    methods: BTreeMap<String, ResonanceMethod>,
    /// Shifts already computed, by replica, so attribution's linearity check
    /// reads a replica's draw without rebuilding it.
    computed: Mutex<HashMap<u64, Arc<ReplicaShifts>>>,
}

/// One spectrum as the weights read it.
pub(crate) struct WeightSpectrum<'a> {
    pub rates: &'a ReactionRates,
    pub masses: &'a [f64],
    pub boundaries: &'a [f64],
}

impl ResonanceRun {
    /// Prepare every nuclide of `material` that has resonance parameters and
    /// a rate under one of `spectra`, and build its weights.
    pub(crate) fn new(
        material: &Material,
        spectra: &[WeightSpectrum<'_>],
        shielding: Option<&Shielding>,
        seed: u64,
    ) -> Self {
        let mut names: Vec<&String> = material
            .nuclide_data
            .iter()
            .filter(|(name, nd)| {
                nd.resonance_parameters.is_some()
                    && spectra.iter().any(|s| s.rates.contains_key(*name))
            })
            .map(|(name, _)| name)
            .collect();
        names.sort();
        let shapes: Vec<Option<CollapseShapes>> = spectra
            .iter()
            .map(|s| {
                let valid = !s.masses.is_empty() && s.boundaries.len() == s.masses.len() + 1;
                valid
                    .then(|| CollapseShapes::new(material, s.masses, s.boundaries, shielding))
                    .flatten()
            })
            .collect();
        let one = |name: &&String| -> (String, Result<Option<RunNuclide>, String>) {
            let name = (*name).clone();
            let outcome = run_nuclide(material, &name, spectra, &shapes);
            (name, outcome)
        };
        let built: Vec<(String, Result<Option<RunNuclide>, String>)> = {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                names.par_iter().map(one).collect()
            }
            #[cfg(target_arch = "wasm32")]
            {
                names.iter().map(one).collect()
            }
        };
        let mut nuclides = Vec::new();
        let mut methods = BTreeMap::new();
        for (name, outcome) in built {
            match outcome {
                Ok(Some(n)) => {
                    methods.insert(
                        name,
                        ResonanceMethod::Sampled {
                            ranges: n.prepared.ranges.iter().map(|r| r.report.clone()).collect(),
                        },
                    );
                    nuclides.push(n);
                }
                Ok(None) => {
                    methods.insert(
                        name,
                        ResonanceMethod::FirstOrder {
                            reason: "no channel of this run reads a reaction its resonance \
                                     ranges give (elastic, capture, fission or an R-matrix \
                                     exit pair)"
                                .to_string(),
                        },
                    );
                }
                Err(reason) => {
                    methods.insert(name, ResonanceMethod::FirstOrder { reason });
                }
            }
        }
        ResonanceRun {
            seed,
            n_spectra: spectra.len(),
            nuclides,
            methods,
            computed: Mutex::new(HashMap::new()),
        }
    }

    /// Whether any nuclide's parameters are sampled.
    pub(crate) fn is_empty(&self) -> bool {
        self.nuclides.is_empty()
    }

    /// The nuclides whose parameters are sampled, whose first-order rows the
    /// MF=33 draw leaves out.
    pub(crate) fn sampled(&self) -> std::collections::BTreeSet<String> {
        self.nuclides.iter().map(|n| n.name.clone()).collect()
    }

    /// Every nuclide with resonance parameters, and how they were taken.
    pub(crate) fn methods(&self) -> &BTreeMap<String, ResonanceMethod> {
        &self.methods
    }

    /// Replica `replica`'s relative rate shifts, per spectrum.
    pub(crate) fn shifts(&self, replica: u64) -> Result<Arc<ReplicaShifts>, String> {
        if let Some(s) = self
            .computed
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&replica)
        {
            return Ok(Arc::clone(s));
        }
        let made = Arc::new(self.compute(replica)?);
        self.computed
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(replica, Arc::clone(&made));
        Ok(made)
    }

    fn compute(&self, replica: u64) -> Result<ReplicaShifts, String> {
        let mut out: ReplicaShifts = vec![RateShifts::new(); self.n_spectra];
        for n in &self.nuclides {
            // Per spectrum and channel, the change in rate.
            let mut delta: Vec<Vec<f64>> = n
                .spectra
                .iter()
                .map(|s| {
                    s.as_ref()
                        .map_or_else(Vec::new, |s| vec![0.0; s.channels.len()])
                })
                .collect();
            for r in &n.prepared.ranges {
                let fail = |e: endf::Error| {
                    format!(
                        "{} range {:.4e} to {:.4e} eV, replica {replica}: {e}",
                        n.name, r.edges[0], r.edges[1]
                    )
                };
                let values = r.sampler.draw(self.seed, replica, &n.name, r.index);
                let perturbed = reconstruction_at(&r.range, &r.cov, &values).map_err(fail)?;
                let grid = reconstruction_grid(&[r.nominal.as_ref(), perturbed.as_ref()], &r.edges)
                    .map_err(fail)?;
                let nominal = r.nominal_on(&grid);
                let drawn = cross_sections_on(perturbed.as_ref(), &grid);
                // The difference is zero outside the range: a repeated end
                // point steps it there, so neither continuation the weight's
                // integral applies past a table's ends reaches it.
                let mut energy = Vec::with_capacity(grid.len() + 2);
                energy.push(grid[0]);
                energy.extend_from_slice(&grid);
                energy.push(grid[grid.len() - 1]);
                let mut sigma = vec![0.0; energy.len()];
                for (slot, mt) in r.slot_mts.iter().enumerate() {
                    let Some(mt) = *mt else { continue };
                    let wanted = n
                        .spectra
                        .iter()
                        .flatten()
                        .any(|s| s.channels.iter().any(|c| reads(c.mt, mt)));
                    if !wanted {
                        continue;
                    }
                    for (i, (a, b)) in drawn.iter().zip(&nominal).enumerate() {
                        sigma[i + 1] = r.abundance * (a.slots()[slot] - b.slots()[slot]);
                    }
                    for (s, d) in n.spectra.iter().zip(delta.iter_mut()) {
                        let Some(s) = s else { continue };
                        if !s.channels.iter().any(|c| reads(c.mt, mt)) {
                            continue;
                        }
                        let change = 1e-24 * s.weight.integrate(&energy, &sigma).map_err(fail)?;
                        for (c, d) in s.channels.iter().zip(d.iter_mut()) {
                            if reads(c.mt, mt) {
                                *d += change;
                            }
                        }
                    }
                }
            }
            for ((s, d), shifts) in n.spectra.iter().zip(&delta).zip(out.iter_mut()) {
                let Some(s) = s else { continue };
                let entry = shifts.entry(n.name.clone()).or_default();
                for (c, d) in s.channels.iter().zip(d) {
                    entry.insert(c.kind.clone(), d / c.rate);
                }
            }
        }
        Ok(out)
    }
}

/// One nuclide's preparation and weights, `Ok(None)` where no spectrum reads
/// a channel its ranges give.
fn run_nuclide(
    material: &Material,
    name: &str,
    spectra: &[WeightSpectrum<'_>],
    shapes: &[Option<CollapseShapes>],
) -> Result<Option<RunNuclide>, String> {
    let nd = &material.nuclide_data[name];
    let data = nd
        .resonance_parameters
        .as_ref()
        .expect("only nuclides with resonance parameters are prepared");
    let label = if material.temperature().is_empty() {
        crate::default_temperature(nd).unwrap_or_default()
    } else {
        material.temperature().to_string()
    };
    let temperature = yamc_nuclide::temperature::label_to_kelvin(&label)
        .ok_or_else(|| format!("its temperature label {label:?} is not a number of kelvin"))?;

    // Channels first: preparing a nuclide no channel could read is wasted.
    let any_candidate = spectra.iter().any(|s| {
        s.rates.get(name).is_some_and(|k| {
            k.iter()
                .any(|(kind, r)| *r > 0.0 && reaction_type_to_mt(kind).is_some_and(could_read))
        })
    });
    if !any_candidate {
        return Ok(None);
    }
    let prepared = prepared(name, data)?;
    let mts: Vec<i32> = prepared
        .ranges
        .iter()
        .flat_map(|r| r.slot_mts.iter().flatten().copied())
        .collect();
    let mut reads_any = false;
    let mut read = Vec::with_capacity(spectra.len());
    for (s, shapes) in spectra.iter().zip(shapes) {
        let mut channels: Vec<Channel> = s
            .rates
            .get(name)
            .map(|kinds| {
                kinds
                    .iter()
                    .filter(|(_, r)| **r > 0.0)
                    .filter_map(|(kind, r)| {
                        let mt = reaction_type_to_mt(kind)?;
                        mts.iter().any(|&m| reads(mt, m)).then(|| Channel {
                            kind: kind.clone(),
                            mt,
                            rate: *r,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        if channels.is_empty() {
            read.push(None);
            continue;
        }
        channels.sort_by(|a, b| a.kind.cmp(&b.kind));
        let shape = shapes.as_ref().and_then(|c| c.shape_for(name));
        let flux = flux_density(s, shape.as_ref(), &prepared, temperature)?;
        let weight = broadened_weight(&flux, prepared.awr, temperature, prepared.broaden_below)
            .map_err(|e| format!("its broadened flux weight cannot be built: {e}"))?;
        reads_any = true;
        read.push(Some(SpectrumRead { weight, channels }));
    }
    Ok(reads_any.then(|| RunNuclide {
        name: name.to_string(),
        prepared,
        spectra: read,
    }))
}

/// The collapse's flux density over the groups whose broadening reaches the
/// nuclide's ranges: [`endf::doppler::WINDOW`] Doppler widths, plus one,
/// either side, outside which the kernel is below `exp(-49)` of its peak.
fn flux_density(
    s: &WeightSpectrum<'_>,
    shape: Option<&FluxShape>,
    prepared: &PreparedNuclide,
    temperature: f64,
) -> Result<Flux, String> {
    let (lo, hi) = prepared.span;
    let (lo, hi) = if temperature > 0.0 {
        let alpha = prepared.awr / (endf::K_BOLTZMANN * temperature);
        let reach = endf::doppler::WINDOW + 1.0;
        let x_lo = ((alpha * lo).sqrt() - reach).max(0.0);
        let x_hi = (alpha * hi).sqrt() + reach;
        (x_lo * x_lo / alpha, x_hi * x_hi / alpha)
    } else {
        (lo, hi)
    };
    let b = s.boundaries;
    let groups: Vec<usize> = (0..s.masses.len())
        .filter(|&g| b[g + 1] > lo && b[g] < hi)
        .collect();
    if groups.is_empty() {
        return Err("no flux group reaches its resonance ranges".to_string());
    }
    // A group starting at zero energy keeps its density and starts where the
    // broadening cannot see below.
    let floor = |e: f64| {
        if e > 0.0 {
            e
        } else {
            lo.max(1e-11).min(b[1] * 0.5)
        }
    };
    match shape {
        None => {
            let edges: Vec<f64> = groups
                .iter()
                .map(|&g| floor(b[g]))
                .chain(std::iter::once(b[groups[groups.len() - 1] + 1]))
                .collect();
            let flat = |g: usize| s.masses[g] / (b[g + 1] - b[g]);
            match crate::multigroup::within_group_weight() {
                Weighting::OneOverE if b[groups[0]] > 0.0 => Ok(Flux::Lethargy {
                    per_lethargy: groups
                        .iter()
                        .map(|&g| s.masses[g] / (b[g + 1] / b[g]).ln())
                        .collect(),
                    edges,
                }),
                _ => Ok(Flux::Histogram {
                    density: groups.iter().map(|&g| flat(g)).collect(),
                    edges,
                }),
            }
        }
        Some(shape) => {
            let (shape_energy, _) = shape.points();
            let mut energy = Vec::new();
            let mut value = Vec::new();
            for &g in &groups {
                let (glo, ghi) = (floor(b[g]), b[g + 1]);
                let mut points = vec![glo];
                // The shape's own points inside the group, ascending.
                points.extend(
                    shape_energy
                        .iter()
                        .rev()
                        .copied()
                        .filter(|&e| e > glo && e < ghi),
                );
                points.push(ghi);
                let s_at: Vec<f64> = points.iter().map(|&e| shape.at(e)).collect();
                let integral: f64 = points
                    .windows(2)
                    .zip(s_at.windows(2))
                    .map(|(e, v)| 0.5 * (v[0] + v[1]) * (e[1] - e[0]))
                    .sum();
                let scale = if integral > 0.0 {
                    s.masses[g] / integral
                } else {
                    0.0
                };
                energy.extend_from_slice(&points);
                value.extend(s_at.iter().map(|v| v * scale));
            }
            Ok(Flux::Pointwise { energy, value })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use endf::mf::mf2::ResonanceParameters as Parameters;

    fn range(el: f64, eh: f64, lru: i64) -> ResonanceRange {
        ResonanceRange {
            el,
            eh,
            lru,
            lrf: 3,
            nro: 0,
            naps: 0,
            parameters: Parameters::ScatteringRadius {
                spi: 0.0,
                ap: 0.0,
                nls: 0,
            },
        }
    }

    /// BROADR's default limit: the resolved range's top, else the unresolved
    /// range's bottom, never above 6.5 MeV.
    #[test]
    fn thnmax_is_broadr_default() {
        assert_eq!(
            thnmax(&[range(1e-5, 1e4, 1), range(1e4, 1e5, 2)]),
            Some(1e4)
        );
        assert_eq!(thnmax(&[range(2e3, 1e5, 2)]), Some(2e3));
        assert_eq!(thnmax(&[range(1e-5, 2e7, 1)]), Some(BROADR_CEILING_EV));
        assert_eq!(thnmax(std::iter::empty()), None);
    }

    /// A channel reads its own reaction and its levels, and nothing else.
    #[test]
    fn a_channel_reads_its_reaction_and_its_levels() {
        assert!(reads(102, 102));
        assert!(reads(4, 51));
        assert!(reads(103, 600));
        assert!(!reads(102, 2));
        assert!(!reads(16, 102));
    }
}
