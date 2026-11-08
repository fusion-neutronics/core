//! Transmutation orchestration: the `Model::transmute` driver and its private
//! chain-walking helper.
//!
//! The only part of the transmutation stack that needs transport. Everything
//! it drives (the stepper, the schedule, the multigroup collapse, the tallies)
//! lives in `yani-transmute`; this file is the coupled loop that runs a
//! transport solve per timestep and feeds the tallied rates back in.

use crate::model::Model;
use crate::track::NoOpTracker;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use yamc_materials::material::{DensityUnits, Material};
use yani_transmute::{
    apply_coupled_branching, ChainNuclide, FissionYieldWeights, ForwardEulerStepper, ReactionRates,
    TransmutationResults, TransmutationStepper, TransmutationTallies,
};

/// How small a share of the dominant nuclide's macroscopic contribution a
/// nuclide may hold and still be worth transport-scope cross sections.
///
/// Ten decades below the nuclide that dominates the material is seven below
/// anything a Monte Carlo solve resolves and eight below nuclear-data
/// uncertainty, so what it buys is memory rather than accuracy: a UO2 sphere
/// goes from 1262 promotions to 113.
const TRANSPORT_SHARE: f64 = 1.0e-10;

/// The largest genuine cross section [barns] a nuclide's loaded data shows, or
/// a bare elastic-scale stand-in when nothing is loaded for it yet.
///
/// MTs at or above 300 are excluded because they are not cross sections: the
/// heating and damage-energy numbers are in eV-barns and run to 1e7 for oxygen,
/// which would swamp any ranking built on them.
fn peak_cross_section(material: &Material, name: &str) -> f64 {
    /// Nothing loaded means nothing to rank on, so assume a plain scatterer.
    /// Also the floor for anything whose loaded MTs are all thresholds, since
    /// activation scope carries no elastic channel.
    const ELASTIC_SCALE: f64 = 20.0;

    let peak = material
        .nuclide_data
        .get(name)
        .and_then(|nuclide| nuclide.reactions_for_temp(material.temperature()))
        .map(|reactions| {
            reactions
                .iter()
                .filter(|(mt, _)| **mt < 300)
                .flat_map(|(_, reaction)| reaction.cross_section.iter().copied())
                .fold(0.0_f64, f64::max)
        })
        .unwrap_or(0.0);
    peak.max(ELASTIC_SCALE)
}

/// How the transport took each of `material`'s nuclides' resolved resonance
/// range: sampled per replica, or the first-order MF=32 rows and why.
fn resonance_methods(
    material: &Material,
) -> std::collections::BTreeMap<String, yani_transmute::resonance_rates::ResonanceMethod> {
    crate::xs_perturbation::resonance_sampling(material)
        .into_iter()
        .map(|(name, sampled)| {
            let method = match sampled {
                Ok(r) => r.method(),
                Err(reason) => {
                    yani_transmute::resonance_rates::ResonanceMethod::FirstOrder { reason }
                }
            };
            (name, method)
        })
        .collect()
}

impl Model {
    /// Run coupled transport-transmutation calculation (tally-based).
    ///
    /// Transport runs each timestep with updated compositions. Per-material tallies
    /// are used to extract flux and reaction rates, normalized by material volumes.
    ///
    /// For each timestep:
    /// 1. Set up tallies for transmutable materials (first iteration only)
    /// 2. If source_rate > 0: Run transport with current compositions
    /// 3. Extract tallies and normalize by volume and source_rate
    /// 4. Transmute all transmutable materials using CRAM16
    /// 5. Update material compositions in geometry
    ///
    /// When source_rate = 0 (cooling period), transport is skipped and
    /// only radioactive decay is applied.
    ///
    /// # Arguments
    /// * `method` - `"coupled"` (transport every step) or `"independent"` (transport once,
    ///   scale rates per step). Independent mode is faster for low-burnup / fusion scenarios.
    /// * `timesteps` - Time intervals [s] for each transmutation step
    /// * `source_rates` - Total source strength [n/s] for each step (0 = cooling)
    /// * `chain` - Parsed transmutation chain data
    /// * `parts` - Which optional subsections `chain` was built from
    /// * `settings` - Per-run transport settings (particle count, seed,
    ///   threads, `max_runtime`). Every transport step uses the same settings,
    ///   so `total_particles` / `max_runtime` are applied afresh to each
    ///   step's transport solve (a per-step budget, not a whole-run one).
    /// * `uncertainty` - Resample the single transport of independent mode
    ///   and re-solve, for every source [`yani_transmute::transport_replicas`]
    ///   applies. With the coupled method, run the whole coupled calculation
    ///   again per replica instead ([`yani_transmute::coupled_uncertainty`]),
    ///   each on its own transport seed, decay data and, with the
    ///   cross-section source, its own nuclear data in every transport
    ///   ([`crate::coupled_xs`]) and, with the MF=40 source, its own
    ///   isomer production curves; attribution reruns the ensemble once per
    ///   source. `None` is the default path, bit-identical to a build
    ///   without it, and so are the nominal results of either method with it.
    ///   With the cross-section source on in independent mode, the transport
    ///   carries a fixed number of replica weights
    ///   ([`yani_transmute::uncertainty::transport_replicas_for`]) and
    ///   tallies every rate per replica under them, so each replica's rates
    ///   carry the flux's response to its cross sections; its nominal rates
    ///   are bit-identical to a run without.
    ///
    /// * `product_threshold` - The atom fraction a product must reach, at any
    ///   point of the schedule, for its cross sections to be loaded and its
    ///   reactions tallied and solved
    ///   ([`yani_transmute::DEFAULT_PRODUCT_THRESHOLD`] is the usual choice).
    ///   A cheap scouting transport runs first to supply the flux spectrum the
    ///   screening solves against (see [`yani_transmute::screening`]); a
    ///   product left out still decays and is still produced. Product nuclide
    ///   paths are resolved from CONFIG.cross_sections or material data.
    ///
    /// * `processing` - Removals, transfers and feeds (see
    ///   [`yani_transmute::processing`]). The materials a transfer links are
    ///   solved together each step. A material it names that the geometry does
    ///   not hold, a storage tank say, is taken from `receivers`, carried
    ///   unirradiated and reported like the transmutable ones. `None` solves
    ///   every material independently, as before.
    ///
    /// # Returns
    /// TransmutationResults containing material compositions at each timestep
    #[allow(clippy::too_many_arguments)]
    pub fn transmute(
        &mut self,
        method: &str,
        schedule: &yani_transmute::Schedule,
        chain: Arc<HashMap<String, yani_transmute::ChainNuclide>>,
        branch: Arc<yani::BranchTable>,
        parts: yani::ChainParts,
        settings: &crate::model::TransportSettings,
        uncertainty: Option<&yani_transmute::uncertainty::DataUncertainty>,
        product_threshold: f64,
        processing: Option<yani_transmute::ProcessingRun<'_>>,
    ) -> Result<TransmutationResults, Box<dyn std::error::Error>> {
        yani_transmute::screening::check_threshold(product_threshold)?;
        if processing.is_some() && uncertainty.is_some() {
            return Err("data_uncertainty is not yet supported together with processing".into());
        }
        // Validate method
        let is_independent = match method {
            "independent" => true,
            "coupled" => false,
            _ => {
                return Err(format!(
                    "method must be \"independent\" or \"coupled\", got \"{method}\""
                )
                .into());
            }
        };
        // Coupled mode re-runs transport every step, so its rates carry noise
        // and data draws that feed forward through the compositions;
        // resampling one step's rates would not describe that. Each replica is
        // the whole coupled run again instead, and what a replica cannot carry
        // yet is refused here, before anything runs.
        let coupled_replicas = match uncertainty {
            Some(request) if !is_independent => {
                Some(yani_transmute::coupled_uncertainty::CoupledReplicas::new(
                    request, &chain, &branch,
                )?)
            }
            _ => None,
        };
        // The per-history covariance of one transport is what independent mode
        // resamples; coupled replicas measure the noise by rerunning instead.
        let want_statistics = is_independent
            && uncertainty
                .is_some_and(|u| u.wants(yani_transmute::uncertainty::Source::Statistical));
        // The interval productions independent mode re-splits the tallied
        // partials with. A coupled replica tallies its own partials on its own
        // production curves instead.
        let want_isomer = is_independent
            && uncertainty
                .is_some_and(|u| u.wants(yani_transmute::uncertainty::Source::IsomerBranching));
        let want_cross_sections = uncertainty
            .is_some_and(|u| u.wants(yani_transmute::uncertainty::Source::CrossSections));
        // With the cross sections perturbed, independent mode's single
        // transport carries one likelihood-ratio weight per replica and
        // tallies every rate per replica under it, so each replica's rates
        // answer its own data. A coupled replica transports on its own data
        // instead, so its transports carry no weights.
        let replica_weights = uncertainty
            .filter(|_| want_cross_sections && is_independent)
            .map(|u| crate::model::TransportDataUncertainty {
                seed: u.seed,
                replicas: yani_transmute::uncertainty::transport_replicas_for(u),
            });
        if let Some(data) = &replica_weights {
            // Refused before anything runs rather than after the scouting
            // transport.
            self.validate_data_uncertainty(data, true)?;
        }

        // Expand the schedule into per-step timesteps / source rates. The
        // schedule guarantees at least one step, and equal lengths by
        // construction (one rate and one duration per step).
        let timesteps = schedule.timesteps();
        let source_rates = schedule.source_rates();

        // Validate inputs
        for (i, &dt) in timesteps.iter().enumerate() {
            if dt <= 0.0 {
                return Err(format!("timestep[{i}] must be positive, got {dt}").into());
            }
        }
        for (i, &sr) in source_rates.iter().enumerate() {
            if sr < 0.0 {
                return Err(format!("source_rate[{i}] must be non-negative, got {sr}").into());
            }
        }

        // Independent mode requires at least one non-zero source rate for the initial transport
        if is_independent && !source_rates.iter().any(|&sr| sr > 0.0) {
            return Err(
                "independent mode requires at least one source_rate > 0 for initial transport"
                    .into(),
            );
        }

        // Find all transmutable materials and their cell indices
        let transmutable_cells = self.find_transmutable_cells()?;
        if transmutable_cells.is_empty() {
            return Err("No transmutable materials found in geometry".into());
        }

        let has_nuclide_tallies = self.tallies.iter().any(|t| !t.nuclides.is_empty());

        // Ensure nuclide data is loaded for all transmutable materials.
        // This must happen before chain-walking and TransmutationTallies creation,
        // both of which need nuclide_data to be populated.
        for cell_indices in transmutable_cells.values() {
            let cell_idx = cell_indices[0];
            let first_mat_idx = self.geometry.cells()[cell_idx]
                .material_idx
                .expect("transmutable cell must have a material");
            let material_arc = &self.geometry.materials()[first_mat_idx as usize];
            if material_arc.nuclide_data.is_empty() {
                let mut material = (**material_arc).clone();
                material.ensure_nuclides_loaded()?;
                let new_arc = Arc::new(material);
                let slots: Vec<u32> = cell_indices
                    .iter()
                    .filter_map(|&cidx| self.geometry.cells()[cidx].material_idx)
                    .collect();
                for slot in slots {
                    self.geometry.materials_mut()[slot as usize] = Arc::clone(&new_arc);
                }
            }
        }

        // The materials only processing names, and per transmutable material
        // the nuclides processing can bring in, whose cross sections its
        // rates need although its composition does not hold them.
        let mut receivers: Vec<Material> = Vec::new();
        let mut brought: HashMap<u32, std::collections::BTreeSet<String>> = HashMap::new();
        let processing_terms = processing.as_ref().map(|run| run.processing);
        if let Some(run) = &processing {
            let in_geometry: HashMap<u32, &Material> = self
                .geometry
                .materials()
                .iter()
                .filter_map(|m| m.get_material_id().map(|id| (id, m.as_ref())))
                .collect();
            for m in &run.receivers {
                let Some(id) = m.get_material_id() else {
                    continue;
                };
                if transmutable_cells.contains_key(&id) {
                    continue;
                }
                if in_geometry.contains_key(&id) {
                    return Err(format!(
                        "processing names material {id}, which is in the geometry but not \
                         transmutable; make it transmutable to process it"
                    )
                    .into());
                }
                receivers.push(match m.density_units {
                    DensityUnits::Sum => m.clone(),
                    _ => m.to_sum_mode()?,
                });
            }
            let all: Vec<&Material> = transmutable_cells
                .keys()
                .filter_map(|id| in_geometry.get(id).copied())
                .chain(&receivers)
                .collect();
            run.processing.check(&all, timesteps.len())?;
            for &id in transmutable_cells.keys() {
                brought.insert(id, run.processing.nuclides_into(id, &chain, &all));
            }
        }

        let peak_source_rate = source_rates.iter().copied().fold(0.0, f64::max);

        // Scouting transport: the flux spectrum the product screening solves
        // against.
        //
        // A product's reaction rate comes from continuous-energy scoring during
        // transport, but which products are worth scoring has to be known
        // before the tallies are built. The flux moments this tally
        // accumulates per union-grid bin (one binary search and two adds per
        // track segment, whatever the nuclide count) give a fine-group
        // spectrum, and a multigroup solve on it is all that deciding which
        // products to carry needs.
        //
        // This runs before the products are loaded, so its tally carries only
        // the material's own nuclides and costs a bare transport. In coupled
        // mode the resulting product set is reused by every step, so the extra
        // solve amortises; in independent mode it is the price of the single
        // pruned solve that follows, which is cheaper than the unpruned one it
        // replaces at any particle count worth running.
        #[cfg(feature = "debug_timing")]
        let scout_start = std::time::Instant::now();

        let scout: Option<Arc<TransmutationTallies>> = if peak_source_rate > 0.0 {
            println!("Scouting transport for the flux spectrum the product screening needs...");
            let scout = {
                let seeded: HashMap<u32, &Material> = transmutable_cells
                    .iter()
                    .filter_map(|(&mat_id, indices)| {
                        let idx = *indices.first()?;
                        let slot = self.geometry.cells()[idx].material_idx?;
                        let material = self.geometry.materials().get(slot as usize)?;
                        Some((mat_id, material.as_ref()))
                    })
                    .collect();
                Arc::new(TransmutationTallies::new(
                    &transmutable_cells,
                    &seeded,
                    &chain,
                    &branch,
                    &HashMap::new(),
                )?)
            };
            self.run_internal::<NoOpTracker>(settings, None, Some(Arc::clone(&scout)), false)?;
            Some(scout)
        } else {
            // Nothing is irradiated, so no reaction can fire and no product
            // needs a cross section: there is nothing to scout for.
            None
        };

        #[cfg(feature = "debug_timing")]
        eprintln!(
            "[TIMING] Scouting transport for the product screening: {:.3}s",
            scout_start.elapsed().as_secs_f64()
        );

        // Load cross sections for the products the irradiation populates, and
        // nothing else.
        //
        // The chain's reachable closure is no guide: it saturates on one large
        // strongly connected component, so Fe56, water and steel all reach the
        // same few hundred reactive nuclides while a real solve of an iron
        // sphere populates 49, and loading the closure is hundreds of MB. The
        // screening solves the schedule on the scouting spectrum instead,
        // loading a product once its atom fraction or its share of a decay
        // quantity reaches `product_threshold` and solving again until nothing
        // new crosses it (see
        // `yani_transmute::screening`). A product left out still decays and is
        // still produced; only its own reactions are dropped.
        //
        // These load at activation scope: the MTs the network names and no
        // transport sections. A product enters the run at zero density and so
        // cannot be collided on; if one later grows a density, the coupled
        // loop's `ensure_nuclide_loaded` widens it to transport scope on
        // demand.
        #[cfg(feature = "debug_timing")]
        let preload_start = std::time::Instant::now();

        // Per material, the nuclides the tallies score: the composition and
        // the screened products. A material missing from this map scores
        // everything it has, which is what happens when nothing was
        // irradiated.
        let mut carried: HashMap<u32, std::collections::HashSet<String>> = HashMap::new();
        let mut screenings: HashMap<u32, yani_transmute::ProductScreening> = HashMap::new();

        if let Some(scout) = &scout {
            for (&mat_id, cell_indices) in &transmutable_cells {
                let cell_idx = cell_indices[0];
                let first_mat_idx = self.geometry.cells()[cell_idx]
                    .material_idx
                    .expect("transmutable cell must have a material");
                let material_arc = &self.geometry.materials()[first_mat_idx as usize];
                let mut material = (**material_arc).clone();
                let volume = material.volume.unwrap_or(1.0);

                // The scouting flux on the tally's union grid, per unit source
                // rate, as the shape the collapse takes and the magnitude each
                // step scales it by. A material the scout never reached sees
                // no flux and screens to its composition.
                let Some(mut spectrum) = scout.flux_spectrum(mat_id, volume) else {
                    continue;
                };
                let per_source: f64 = spectrum.masses.iter().sum();
                if per_source.is_nan() || per_source <= 0.0 {
                    continue;
                }
                for m in spectrum.masses.iter_mut() {
                    *m /= per_source;
                }
                let steps: Vec<yani_transmute::TransmuteStep> = timesteps
                    .iter()
                    .zip(source_rates.iter())
                    .map(|(&dt, &sr)| yani_transmute::TransmuteStep {
                        dt,
                        irradiation: (sr > 0.0).then_some((0, sr * per_source)),
                    })
                    .collect();
                let before = material.nuclide_data.len();
                let screening = yani_transmute::screen_activation_products(
                    &mut material,
                    &[spectrum],
                    &steps,
                    &chain,
                    &branch,
                    parts,
                    None,
                    None,
                    product_threshold,
                )
                .map_err(|e| format!("material {mat_id}: {e}"))?;
                if self.verbose.nuclear_data {
                    println!(
                        "Transmutation products for material {mat_id}: cross sections for {} of \
                         the {} reactive nuclides the chain reaches, after {} screening \
                         solve(s); {} populated below an atom fraction of {:e} react no further \
                         (summed peak fraction {:.3e})",
                        screening.loaded().len(),
                        screening.closure,
                        screening.rounds,
                        screening.below_threshold.len(),
                        screening.threshold,
                        screening.dropped_fraction(),
                    );
                }
                let mut keep: std::collections::HashSet<String> =
                    screening.loaded().into_iter().collect();
                if let Some(extra) = brought.get(&mat_id).filter(|b| !b.is_empty()) {
                    yani_transmute::preload_activation_data(
                        &mut material,
                        &chain,
                        &branch,
                        None,
                        None,
                        extra,
                    )
                    .map_err(|e| format!("material {mat_id}: {e}"))?;
                    keep.extend(extra.iter().cloned());
                }
                // The transport composition is not the screening's to rule
                // on: those nuclides are collided on, not produced.
                keep.extend(material.nuclides.keys().cloned());
                carried.insert(mat_id, keep);
                screenings.insert(mat_id, screening);

                if material.nuclide_data.len() != before {
                    // Validate temperature consistency after loading new nuclides
                    if let Err(e) = material.validate_temperature_consistency() {
                        eprintln!(
                            "\x1b[33mWarning: Temperature inconsistency after loading transmutation products: {e}\x1b[0m"
                        );
                    }
                    // Products are deliberately NOT seeded into `nuclides` at
                    // zero density: that would put their energy grids into
                    // every lookup, and activation-scope data has no transport
                    // sections to offer transport anyway. Use full transport
                    // MT filter so the microscopic XS cache is valid for
                    // subsequent transport steps.
                    let mt_filter = self.transport_mt_filter();
                    material.invalidate_xs_cache();
                    material.calculate_macroscopic_xs(&mt_filter, true);
                    if has_nuclide_tallies {
                        material.populate_per_nuclide_xs();
                    }
                    let new_arc = Arc::new(material);
                    let slots: Vec<u32> = cell_indices
                        .iter()
                        .filter_map(|&cidx| self.geometry.cells()[cidx].material_idx)
                        .collect();
                    for slot in slots {
                        self.geometry.materials_mut()[slot as usize] = Arc::clone(&new_arc);
                    }
                }
            }
        }

        #[cfg(feature = "debug_timing")]
        eprintln!(
            "[TIMING] Pre-load daughter nuclides + initial XS: {:.3}s",
            preload_start.elapsed().as_secs_f64()
        );

        // Build materials map for TransmutationTallies initialization
        let materials_for_init: HashMap<u32, &Material> = transmutable_cells
            .keys()
            .filter_map(|&mat_id| {
                transmutable_cells
                    .get(&mat_id)
                    .and_then(|indices| indices.first())
                    .and_then(|&idx| {
                        self.geometry.cells()[idx]
                            .material_idx
                            .and_then(|slot| self.geometry.materials().get(slot as usize))
                    })
                    .map(|m| (mat_id, m.as_ref()))
            })
            .collect();

        // Create flux-weighted transmutation tallies. Normalization is by the
        // true total source-particle count, accumulated chunk by chunk during
        // transport, so no particle count is needed up front.
        let dep_tallies = {
            let tallies = TransmutationTallies::new(
                &transmutable_cells,
                &materials_for_init,
                &chain,
                &branch,
                &carried,
            )?;
            // The per-history covariance the statistical source samples from.
            // Off otherwise, so the default run allocates nothing for it.
            let tallies = if want_statistics {
                tallies.with_history_statistics()
            } else {
                tallies
            };
            // Each MF=40 state's production on its covariance intervals, which
            // the isomer source re-splits the tallied partials with. Off
            // without it. Before the replicas, which carry it per replica.
            let tallies = if want_isomer {
                tallies.with_isomer_intervals(&chain, &branch)?
            } else {
                tallies
            };
            // Every replica's tally under the transport's replica weights.
            // Off without the cross-section source.
            Arc::new(if let Some(data) = &replica_weights {
                tallies.with_replicas(data.replicas, data.seed)
            } else {
                tallies
            })
        };

        // Initialize results
        let mut results = TransmutationResults::new(timesteps.to_vec());
        // Before the first state, so each one is taken apart into the
        // inventory and what the decay data has no record for.
        results.set_chain(
            Arc::clone(&chain),
            yamc_nuclide::config::Config::global()
                .transmutation_decay_data
                .clone(),
        );
        results.product_screening = screenings;
        for receiver in &receivers {
            let id = receiver.get_material_id().unwrap_or(0);
            results.add_initial(id, receiver.clone(), vec![0.0; timesteps.len()]);
            results.product_screening.insert(id, Default::default());
        }

        // Track full compositions (including all chain products) across steps.
        // The cell material only has transport nuclides (with HDF5 data),
        // but transmutation needs the full composition to carry forward products.
        let mut full_compositions: HashMap<u32, Material> = HashMap::new();

        // Store initial compositions (converted to atom densities [atoms/b-cm])
        for (&mat_id, cell_indices) in &transmutable_cells {
            let cell_idx = cell_indices[0]; // Use first cell as representative
            let slot = self.geometry.cells()[cell_idx]
                .material_idx
                .expect("transmutable cell must have a material");
            let material = self.geometry.materials()[slot as usize].as_ref();

            // Convert initial material to "sum" mode (atom densities) for consistent results
            let mut initial_material = material.clone();
            let atoms_per_bcm = material.get_atoms_per_barn_cm()?;
            initial_material.nuclides = atoms_per_bcm;
            initial_material.density_units = DensityUnits::Sum;
            initial_material.density = None;

            #[cfg(feature = "debug_transmutation")]
            {
                eprintln!(
                    "[DEBUG transmutation] Initial composition for material {}:",
                    mat_id
                );
                let mut nucs: Vec<_> = initial_material.nuclides.iter().collect();
                nucs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
                for (name, density) in &nucs {
                    eprintln!("  {:10} {:.6e} atoms/b-cm", name, density);
                }
            }

            results.add_initial(mat_id, initial_material.clone(), source_rates.to_vec());
            full_compositions.insert(mat_id, initial_material);
        }

        // Per-material spectrum weights folding each fissionable nuclide's
        // tabulated fission yields against the flux it saw.
        // Re-extracted per step in coupled mode, once up front in independent
        // mode; empty for every material with nothing fissionable in it.
        let mut fy_weights: HashMap<u32, FissionYieldWeights> = HashMap::new();

        let mut tallied: HashMap<u32, yani_transmute::TransportTallied> = HashMap::new();
        // The initial compositions the replicas start from, sum mode, with
        // the nuclear data the fold reads.
        let initial_for_replicas: HashMap<u32, Material> = if uncertainty.is_some() {
            full_compositions.clone()
        } else {
            HashMap::new()
        };

        // Independent mode: run transport ONCE and extract micro rates (per-source-particle).
        // These are scaled by each step's source_rate during transmutation.
        let independent_micro_rates: Option<HashMap<u32, ReactionRates>> = if is_independent {
            println!("Independent mode: running single transport for micro reaction rates...");

            dep_tallies.reset();

            // With replica weights the one transport carries them; the
            // nominal tallies it scores are those of a run without, bit for
            // bit.
            let transport_settings = crate::model::TransportSettings {
                data_uncertainty: replica_weights,
                ..settings.clone()
            };
            self.run_internal::<NoOpTracker>(
                &transport_settings,
                None,
                Some(Arc::clone(&dep_tallies)),
                false,
            )?;

            // Extract rates with source_rate=1.0 to get per-source-particle micro rates
            let mut micro_rates: HashMap<u32, ReactionRates> = HashMap::new();
            // With uncertainty asked for, everything the replicas resample,
            // at the same unit source rate.
            if let Some(request) = uncertainty {
                for (&mat_id, cell_indices) in &transmutable_cells {
                    let slot = self.geometry.cells()[cell_indices[0]]
                        .material_idx
                        .expect("transmutable cell must have a material");
                    let material = self.geometry.materials()[slot as usize].as_ref();
                    let volume = material.volume.unwrap_or(1.0);
                    let Some(spectrum) = dep_tallies.flux_spectrum(mat_id, volume) else {
                        continue;
                    };
                    let statistics =
                        if request.wants(yani_transmute::uncertainty::Source::Statistical) {
                            dep_tallies.get_reaction_rate_covariance(mat_id, volume, 1.0)
                        } else {
                            None
                        };
                    tallied.insert(
                        mat_id,
                        yani_transmute::TransportTallied {
                            rates: dep_tallies.get_reaction_rates(mat_id, volume, 1.0),
                            partials: if branch.curves().is_empty() {
                                HashMap::new()
                            } else {
                                dep_tallies.get_partial_rates(mat_id, volume, 1.0)
                            },
                            fy_weights: dep_tallies.get_fission_yield_weights(mat_id),
                            spectrum,
                            statistics,
                            branch: Arc::clone(&branch),
                            diagnostics: dep_tallies
                                .get_branching_diagnostics(mat_id, volume, 1.0)
                                .with_material(material),
                            isomer: dep_tallies.get_isomer_productions(mat_id, volume, 1.0),
                            replicas: dep_tallies.replica_tallied(mat_id, volume).map(|mut t| {
                                t.resonance_parameters = resonance_methods(material);
                                t
                            }),
                            draw_key: slot,
                        },
                    );
                }
            }
            for (&mat_id, cell_indices) in &transmutable_cells {
                let slot = self.geometry.cells()[cell_indices[0]]
                    .material_idx
                    .expect("transmutable cell must have a material");
                let material = self.geometry.materials()[slot as usize].as_ref();
                let volume = material.volume.unwrap_or(1.0);
                let mat_rates = dep_tallies.get_reaction_rates(mat_id, volume, 1.0);

                // The fission-yield fold is a normalized shape, so the single
                // transport of independent mode fixes it for every step just as
                // it fixes the micro rates.
                let mat_weights = dep_tallies.get_fission_yield_weights(mat_id);
                if !mat_weights.is_empty() {
                    fy_weights.insert(mat_id, mat_weights);
                }

                if !mat_rates.is_empty() {
                    micro_rates.insert(mat_id, mat_rates);
                }
            }

            println!(
                "Independent mode: extracted micro rates for {} materials",
                micro_rates.len()
            );
            Some(micro_rates)
        } else {
            None
        };

        #[cfg(feature = "debug_timing")]
        let transmutation_loop_start = std::time::Instant::now();

        // The geometry every coupled replica starts from: the one the nominal
        // loop below starts from.
        let replica_start = coupled_replicas
            .as_ref()
            .map(|_| self.geometry.materials().to_vec());

        // Transmutation loop
        let mut t0 = 0.0;
        for (step, (&dt, &source_rate)) in timesteps.iter().zip(source_rates.iter()).enumerate() {
            println!(
                "Transmutation step {}/{}: dt={:.2e} s, source_rate={:.2e} n/s",
                step + 1,
                timesteps.len(),
                dt,
                source_rate
            );

            // Extract reaction rates
            let mut rates = if source_rate > 0.0 {
                if is_independent {
                    // Independent mode: scale pre-computed micro rates by this step's source_rate
                    let micro = independent_micro_rates.as_ref().unwrap();
                    let mut step_rates: HashMap<u32, ReactionRates> = HashMap::new();
                    for (&mat_id, micro_mat_rates) in micro {
                        let mut scaled: ReactionRates = HashMap::new();
                        for (nuclide, reactions) in micro_mat_rates {
                            let mut scaled_rxns: HashMap<String, f64> = HashMap::new();
                            for (rxn, &micro_rate) in reactions {
                                scaled_rxns.insert(rxn.clone(), micro_rate * source_rate);
                            }
                            scaled.insert(nuclide.clone(), scaled_rxns);
                        }
                        step_rates.insert(mat_id, scaled);
                    }
                    step_rates
                } else {
                    // Coupled mode: run transport and extract rates
                    // Reset transmutation tallies for this transport step
                    dep_tallies.reset();

                    // Run transport with transmutation tally scoring
                    #[cfg(feature = "debug_timing")]
                    let transport_start = std::time::Instant::now();

                    self.run_internal::<NoOpTracker>(
                        settings,
                        None,
                        Some(Arc::clone(&dep_tallies)),
                        false,
                    )?;

                    #[cfg(feature = "debug_timing")]
                    eprintln!(
                        "[TIMING] Step {} transport: {:.3}s",
                        step + 1,
                        transport_start.elapsed().as_secs_f64()
                    );

                    self.tallied_step_rates(
                        &dep_tallies,
                        &transmutable_cells,
                        source_rate,
                        step,
                        &mut fy_weights,
                    )
                }
            } else {
                // Cooling period - no reactions, decay only
                HashMap::new()
            };

            // Isomeric-branching overlay (see `fold_step_branching`).
            let (folded_chains, step_reports) = if source_rate > 0.0 {
                self.fold_step_branching(
                    &dep_tallies,
                    &transmutable_cells,
                    &chain,
                    &branch,
                    source_rate,
                    step,
                    &mut rates,
                )?
            } else {
                Default::default()
            };

            // Transmute all materials in parallel (each material is independent)
            #[cfg(feature = "debug_timing")]
            let transmute_solve_start = std::time::Instant::now();

            let transmuted_materials = match processing_terms {
                None => solve_step(
                    &transmutable_cells,
                    &full_compositions,
                    &rates,
                    &fy_weights,
                    &folded_chains,
                    &chain,
                    parts,
                    dt,
                ),
                Some(terms) => {
                    let (cells, tanks) = solve_processed_step(
                        terms,
                        &full_compositions,
                        &mut receivers,
                        &rates,
                        &fy_weights,
                        &folded_chains,
                        &chain,
                        parts,
                        (step, t0, dt),
                    )?;
                    for tank in tanks {
                        let id = tank.get_material_id().unwrap_or(0);
                        results.add_step(id, tank);
                        results.add_step_rates(id, Default::default());
                        results
                            .branching_report
                            .entry(id)
                            .or_default()
                            .push(Default::default());
                    }
                    cells
                }
            };
            t0 += dt;

            #[cfg(feature = "debug_timing")]
            eprintln!(
                "[TIMING] Step {} transmutation solve (CRAM): {:.3}s",
                step + 1,
                transmute_solve_start.elapsed().as_secs_f64()
            );

            // Apply results sequentially (updates shared state)
            #[cfg(feature = "debug_timing")]
            let material_update_start = std::time::Instant::now();

            for (mat_id, result) in transmuted_materials {
                let transmuted_material =
                    result.map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
                let cell_indices = transmutable_cells.get(&mat_id).unwrap();

                #[cfg(feature = "debug_transmutation")]
                {
                    eprintln!(
                        "[DEBUG transmutation] Material {} step {} transmuted compositions:",
                        mat_id, step
                    );
                    let mut nucs: Vec<_> = transmuted_material.nuclides.iter().collect();
                    nucs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
                    for (name, density) in &nucs {
                        if **density > 0.0 {
                            eprintln!("  {:10} {:.6e} atoms/b-cm", name, density);
                        }
                    }
                }

                // Store full composition in results and for next step
                results.add_step(mat_id, transmuted_material.clone());
                full_compositions.insert(mat_id, transmuted_material.clone());

                // The rates this step was actually solved with, split over the
                // edges they drove. Same two inputs the stepper
                // took, so the numbers are the burnup matrix's own, and both
                // methods report the same thing: only where `rates` came from
                // differs, and the chain is whichever this material was folded
                // onto. A cooldown passes no rates and records an empty map.
                let edge_chain = folded_chains.get(&mat_id).unwrap_or(&chain);
                let edges = rates
                    .get(&mat_id)
                    .map(|mat_rates| yani::per_edge_rates(edge_chain, mat_rates))
                    .unwrap_or_default();
                results.add_step_rates(mat_id, edges);
                results
                    .branching_report
                    .entry(mat_id)
                    .or_default()
                    .push(step_reports.get(&mat_id).cloned().unwrap_or_default());

                // In independent mode, skip transport material updates (geometry is frozen)
                if !is_independent {
                    self.update_transport_material(
                        step,
                        mat_id,
                        cell_indices,
                        transmuted_material,
                        has_nuclide_tallies,
                        &mut |_| Ok(false),
                    )?;
                }
            }

            #[cfg(feature = "debug_timing")]
            eprintln!(
                "[TIMING] Step {} material update total: {:.3}s",
                step + 1,
                material_update_start.elapsed().as_secs_f64()
            );
        }

        #[cfg(feature = "debug_timing")]
        eprintln!(
            "[TIMING] Total transmutation loop: {:.3}s",
            transmutation_loop_start.elapsed().as_secs_f64()
        );

        // Coupled uncertainty: the whole coupled run again per replica, from
        // the geometry the nominal loop started from, after which the model
        // is put back as the nominal loop left it, so the nominal results and
        // the model are what a run without uncertainty gives.
        if let (Some(replicas), Some(start)) = (&coupled_replicas, replica_start) {
            let end = self.geometry.materials().to_vec();
            let nominal: HashMap<u32, yani_transmute::coupled_uncertainty::Inventories> =
                transmutable_cells
                    .keys()
                    .map(|&id| {
                        let per_step = (1..=timesteps.len())
                            .map(|i| {
                                results
                                    .get_material(id, i)
                                    .map(|m| m.nuclides.clone())
                                    .unwrap_or_default()
                            })
                            .collect();
                        (id, per_step)
                    })
                    .collect();
            let run = CoupledRun {
                start: &start,
                initial: &initial_for_replicas,
                cells: &transmutable_cells,
                tallies: &dep_tallies,
                timesteps: &timesteps,
                source_rates: &source_rates,
                branch: &branch,
                carried: &carried,
                parts,
                has_nuclide_tallies,
            };
            // Every nuclide of every material read in full with its
            // covariance, and fielded, once for all the replicas.
            let mut library = match uncertainty.filter(|_| replicas.cross_sections()) {
                Some(request) => Some(crate::coupled_xs::CoupledCrossSections::new(
                    &start,
                    request.seed,
                )?),
                None => None,
            };
            let ensembles = replicas.run(
                &chain,
                &initial_for_replicas,
                &nominal,
                |these, k, edited| {
                    println!("Coupled uncertainty: replica {}...", k + 1);
                    let replica_settings = crate::model::TransportSettings {
                        seed: these.transport_seed(settings.seed, k),
                        ..settings.clone()
                    };
                    let mut data = library
                        .as_mut()
                        .filter(|_| these.cross_sections())
                        .map(|l| l.replica(k));
                    let branch = these.branch(&branch, k);
                    self.coupled_replica(&run, edited, &branch, &replica_settings, data.as_mut())
                },
            );
            self.geometry.materials_mut().clone_from_slice(&end);
            for (id, (ensemble, mut info)) in ensembles? {
                if let Some(l) = &library {
                    l.report(id, &mut info);
                }
                results.uncertainty.insert(id, ensemble);
                results.uncertainty_info.insert(id, info);
            }
        }

        // Uncertainty: resample what the single transport produced and re-solve
        // the same schedule, per material, independently of the nominal loop
        // above, which is untouched and so bit-identical either way.
        if let Some(request) = uncertainty.filter(|_| is_independent) {
            // The cross-material covariance links nuclides across the
            // materials, so it is read from all of them at once, in slot
            // order (each one's draw key): ids would make the order depend on
            // what else the process built.
            let mut resampled: Vec<(u32, &Material)> = initial_for_replicas
                .iter()
                .filter_map(|(id, m)| tallied.get(id).map(|t| (t.draw_key, m)))
                .collect();
            resampled.sort_by_key(|(key, _)| *key);
            let materials: Vec<&Material> = resampled.iter().map(|(_, m)| *m).collect();
            let cross = yani_transmute::transport_cross_material(&materials, &chain, request)?;
            for (mat_id, material) in &initial_for_replicas {
                let Some(transport) = tallied.get(mat_id) else {
                    continue;
                };
                let (ensemble, info) = yani_transmute::transport_replicas(
                    material,
                    transport,
                    &timesteps,
                    &source_rates,
                    &chain,
                    parts,
                    request,
                    &cross,
                )?;
                results.uncertainty.insert(*mat_id, ensemble);
                results.uncertainty_info.insert(*mat_id, info);
                if let Some(c) = transport.statistics.clone() {
                    results.rate_covariance.insert(*mat_id, c);
                }
            }
        }

        results.data_releases = yamc_nuclide::url_cache::data_releases();
        Ok(results)
    }
}

/// What every replica of a coupled run shares with the nominal one.
struct CoupledRun<'a> {
    /// The geometry's materials as the nominal loop started.
    start: &'a [Arc<Material>],
    /// Each transmutable material's starting composition, sum mode.
    initial: &'a HashMap<u32, Material>,
    cells: &'a HashMap<u32, Vec<usize>>,
    tallies: &'a Arc<TransmutationTallies>,
    timesteps: &'a [f64],
    source_rates: &'a [f64],
    branch: &'a Arc<yani::BranchTable>,
    /// Per material, the nuclides the tallies score.
    carried: &'a HashMap<u32, std::collections::HashSet<String>>,
    parts: yani::ChainParts,
    has_nuclide_tallies: bool,
}

/// Each material's next composition, or why its solve failed.
type Transmuted = Vec<(u32, Result<Material, String>)>;

/// Solve one step for every transmutable material, in parallel (each is
/// independent), from `compositions` under `rates`, each on its own
/// branch-folded chain where it has one and on `chain` otherwise.
#[allow(clippy::too_many_arguments)]
fn solve_step(
    cells: &HashMap<u32, Vec<usize>>,
    compositions: &HashMap<u32, Material>,
    rates: &HashMap<u32, ReactionRates>,
    fy_weights: &HashMap<u32, FissionYieldWeights>,
    folded_chains: &HashMap<u32, Arc<HashMap<String, ChainNuclide>>>,
    chain: &Arc<HashMap<String, ChainNuclide>>,
    parts: yani::ChainParts,
    dt: f64,
) -> Transmuted {
    cells
        .par_iter()
        .map(|(&mat_id, _)| {
            let input = compositions.get(&mat_id).unwrap();
            let mat_rates = rates.get(&mat_id).cloned().unwrap_or_default();
            let mat_fy = fy_weights.get(&mat_id).cloned().unwrap_or_default();
            let ch = folded_chains.get(&mat_id).unwrap_or(chain);
            let result = ForwardEulerStepper
                .step(input, ch, &mat_rates, &mat_fy, parts, dt)
                .map_err(|e| e.to_string());
            (mat_id, result)
        })
        .collect()
}

/// [`solve_step`] under `processing`: the transmutable materials and the
/// `receivers` stepped together through [`yani_transmute::Processing::step`],
/// which solves the materials a transfer links as one and the rest
/// independently. Returns the transmutable materials' results and the
/// receivers' new states, which also replace `receivers`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn solve_processed_step(
    processing: &yani_transmute::Processing,
    compositions: &HashMap<u32, Material>,
    receivers: &mut Vec<Material>,
    rates: &HashMap<u32, ReactionRates>,
    fy_weights: &HashMap<u32, FissionYieldWeights>,
    folded_chains: &HashMap<u32, Arc<HashMap<String, ChainNuclide>>>,
    chain: &Arc<HashMap<String, ChainNuclide>>,
    parts: yani::ChainParts,
    (step, t0, dt): (usize, f64, f64),
) -> Result<(Transmuted, Vec<Material>), String> {
    let no_rates = ReactionRates::new();
    let no_fission = FissionYieldWeights::new();
    let mut ids: Vec<u32> = compositions.keys().copied().collect();
    ids.sort_unstable();
    let members: Vec<yani_transmute::Member<'_>> = ids
        .iter()
        .map(|id| (*id, &compositions[id]))
        .chain(
            receivers
                .iter()
                .map(|m| (m.get_material_id().unwrap_or(0), m)),
        )
        .map(|(id, material)| yani_transmute::Member {
            id,
            material,
            chain: folded_chains.get(&id).unwrap_or(chain),
            rates: rates.get(&id).unwrap_or(&no_rates),
            fy_weights: fy_weights.get(&id).unwrap_or(&no_fission),
        })
        .collect();
    let mut solved = processing.step(&members, step, t0, dt, parts).into_iter();
    let cells: Transmuted = ids.iter().copied().zip(solved.by_ref()).collect();
    let tanks: Vec<Material> = solved
        .zip(receivers.iter())
        .map(|(r, m)| r.map_err(|e| format!("material {}: {e}", m.get_material_id().unwrap_or(0))))
        .collect::<Result<_, _>>()?;
    *receivers = tanks.clone();
    Ok((cells, tanks))
}

impl Model {
    /// Each material's rates from the transport just run on its current
    /// composition, at `source_rate`, with its fission-yield fold from that
    /// spectrum put into `fy_weights`.
    fn tallied_step_rates(
        &self,
        dep_tallies: &TransmutationTallies,
        cells: &HashMap<u32, Vec<usize>>,
        source_rate: f64,
        #[allow(unused_variables)] step: usize,
        fy_weights: &mut HashMap<u32, FissionYieldWeights>,
    ) -> HashMap<u32, ReactionRates> {
        // Extract rates from transmutation tallies for each material
        let mut step_rates: HashMap<u32, ReactionRates> = HashMap::new();
        for (&mat_id, cell_indices) in cells {
            let slot = self.geometry.cells()[cell_indices[0]]
                .material_idx
                .expect("transmutable cell must have a material");
            let material = self.geometry.materials()[slot as usize].as_ref();
            let volume = material.volume.unwrap_or(1.0);
            let mat_rates = dep_tallies.get_reaction_rates(mat_id, volume, source_rate);

            // This step's spectrum, so this step's yield fold.
            let mat_weights = dep_tallies.get_fission_yield_weights(mat_id);
            if mat_weights.is_empty() {
                fy_weights.remove(&mat_id);
            } else {
                fy_weights.insert(mat_id, mat_weights);
            }

            #[cfg(feature = "debug_transmutation")]
            if !mat_rates.is_empty() {
                eprintln!(
                    "\n[DEBUG transmutation] Reaction rates for material {} at step {}:",
                    mat_id, step
                );
                let mut nuclides: Vec<_> = mat_rates.keys().collect();
                nuclides.sort();
                for nuclide in nuclides {
                    let reactions = &mat_rates[nuclide.as_str()];
                    let mut rxns: Vec<_> = reactions.iter().collect();
                    rxns.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
                    for (rx, rate) in rxns {
                        eprintln!("  {} {:15} {:.6e} [1/s/atom]", nuclide, rx, rate);
                    }
                }
            }

            if !mat_rates.is_empty() {
                step_rates.insert(mat_id, mat_rates);
            }
        }
        step_rates
    }

    /// Isomeric-branching overlay: every list on a material nuclide is
    /// scored at the collision energies as `yani_transmute::branching_rule`
    /// defines it, and the (n,n') partials of every other chain parent are
    /// folded from the tally's union-grid flux moments (covering products
    /// that build up during the step too), their rates injected into `rates`.
    /// When no branching subsection is configured the overlay is empty and
    /// the physics is identical to before. MT=5's `(n,X)` multiplicities
    /// come from the reactions subsection and are folded either way; what of
    /// MT=5 the chain does not model is reported, and refused above the
    /// tolerance of the material's removal.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn fold_step_branching(
        &self,
        dep_tallies: &TransmutationTallies,
        cells: &HashMap<u32, Vec<usize>>,
        chain: &Arc<HashMap<String, ChainNuclide>>,
        branch: &Arc<yani::BranchTable>,
        source_rate: f64,
        step: usize,
        rates: &mut HashMap<u32, ReactionRates>,
    ) -> Result<
        (
            HashMap<u32, Arc<HashMap<String, ChainNuclide>>>,
            HashMap<u32, Arc<yani_transmute::BranchingReport>>,
        ),
        Box<dyn std::error::Error>,
    > {
        let mut folded_chains: HashMap<u32, Arc<HashMap<String, ChainNuclide>>> = HashMap::new();
        let mut step_reports: HashMap<u32, Arc<yani_transmute::BranchingReport>> = HashMap::new();
        for (&mat_id, cell_indices) in cells {
            let slot = self.geometry.cells()[cell_indices[0]]
                .material_idx
                .expect("transmutable cell must have a material");
            let cell_material = self.geometry.materials()[slot as usize].as_ref();
            let volume = cell_material.volume.unwrap_or(1.0);
            let diagnostics = dep_tallies
                .get_branching_diagnostics(mat_id, volume, source_rate)
                .with_material(cell_material);
            let partials = if branch.curves().is_empty() {
                HashMap::new()
            } else {
                dep_tallies.get_partial_rates(mat_id, volume, source_rate)
            };
            let mat_rates = rates.entry(mat_id).or_default();
            let named = |e: String| -> Box<dyn std::error::Error> {
                format!("material {mat_id}, step {}: {e}", step + 1).into()
            };
            let (folded, report) =
                apply_coupled_branching(chain, branch, &partials, mat_rates, Some(&diagnostics))
                    .map_err(named)?;
            if !partials.is_empty() {
                folded_chains.insert(mat_id, folded);
            }
            step_reports.insert(mat_id, Arc::new(report));
        }
        Ok((folded_chains, step_reports))
    }

    /// One replica of the coupled run: every step of `run`'s schedule from its
    /// starting geometry, solved on `chain` and transported with `settings`,
    /// exactly as the nominal loop steps, returning each material's per-step
    /// inventories. Leaves the geometry at the replica's last step.
    ///
    /// With `data`, every material of the geometry reads that replica's
    /// nuclear data from the start, and so does every nuclide a step loads
    /// into one later, before its cross sections are built.
    fn coupled_replica(
        &mut self,
        run: &CoupledRun,
        chain: &Arc<HashMap<String, ChainNuclide>>,
        branch: &Arc<yani::BranchTable>,
        settings: &crate::model::TransportSettings,
        mut data: Option<&mut crate::coupled_xs::Replica>,
    ) -> Result<
        HashMap<u32, yani_transmute::coupled_uncertainty::Inventories>,
        Box<dyn std::error::Error>,
    > {
        self.geometry.materials_mut().clone_from_slice(run.start);
        if let Some(data) = data.as_mut() {
            // A material several slots share is perturbed once and shared.
            let mut made: HashMap<usize, Arc<Material>> = HashMap::new();
            for slot in self.geometry.materials_mut().iter_mut() {
                let at = Arc::as_ptr(slot) as usize;
                if let Some(m) = made.get(&at) {
                    *slot = Arc::clone(m);
                    continue;
                }
                let mut material = (**slot).clone();
                if data.apply(&mut material)? {
                    *slot = Arc::new(material);
                }
                made.insert(at, Arc::clone(slot));
            }
        }
        // A replica with its own production curves tallies on them.
        let tallies = if Arc::ptr_eq(branch, run.branch) {
            Arc::clone(run.tallies)
        } else {
            let materials: HashMap<u32, &Material> = run
                .cells
                .iter()
                .filter_map(|(&id, cells)| {
                    let slot = self.geometry.cells()[*cells.first()?].material_idx?;
                    Some((id, self.geometry.materials().get(slot as usize)?.as_ref()))
                })
                .collect();
            Arc::new(TransmutationTallies::new(
                run.cells,
                &materials,
                chain,
                branch,
                run.carried,
            )?)
        };
        let mut compositions = run.initial.clone();
        let mut fy_weights: HashMap<u32, FissionYieldWeights> = HashMap::new();
        let mut out: HashMap<u32, yani_transmute::coupled_uncertainty::Inventories> = run
            .cells
            .keys()
            .map(|&id| (id, Vec::with_capacity(run.timesteps.len())))
            .collect();
        for (step, (&dt, &source_rate)) in run.timesteps.iter().zip(run.source_rates).enumerate() {
            let mut rates = HashMap::new();
            let mut folded = HashMap::new();
            if source_rate > 0.0 {
                tallies.reset();
                self.run_internal::<NoOpTracker>(
                    settings,
                    None,
                    Some(Arc::clone(&tallies)),
                    false,
                )?;
                rates = self.tallied_step_rates(
                    &tallies,
                    run.cells,
                    source_rate,
                    step,
                    &mut fy_weights,
                );
                folded = self
                    .fold_step_branching(
                        &tallies,
                        run.cells,
                        chain,
                        branch,
                        source_rate,
                        step,
                        &mut rates,
                    )?
                    .0;
            }
            let solved = solve_step(
                run.cells,
                &compositions,
                &rates,
                &fy_weights,
                &folded,
                chain,
                run.parts,
                dt,
            );
            for (mat_id, result) in solved {
                let material = result?;
                out.entry(mat_id)
                    .or_default()
                    .push(material.nuclides.clone());
                compositions.insert(mat_id, material.clone());
                self.update_transport_material(
                    step,
                    mat_id,
                    &run.cells[&mat_id],
                    material,
                    run.has_nuclide_tallies,
                    &mut |m| data.as_mut().map_or(Ok(false), |d| d.apply(m)),
                )?;
            }
        }
        Ok(out)
    }

    /// Hand `transmuted_material` to the next step's transport, in every
    /// cell of `cell_indices`.
    ///
    /// `adjust` sees the material once every nuclide it needs is loaded and
    /// before its cross sections are built, and says whether it changed its
    /// nuclear data: a coupled replica puts its own draw in place of what a
    /// step loaded here.
    fn update_transport_material(
        &mut self,
        #[allow(unused_variables)] step: usize,
        #[allow(unused_variables)] mat_id: u32,
        cell_indices: &[usize],
        transmuted_material: Material,
        has_nuclide_tallies: bool,
        adjust: &mut dyn FnMut(&mut Material) -> Result<bool, String>,
    ) -> Result<(), String> {
        let cell_idx = cell_indices[0];
        let slot = self.geometry.cells()[cell_idx]
            .material_idx
            .expect("transmutable cell must have a material");
        let cell_material = self.geometry.materials()[slot as usize].as_ref();

        // Build transport material: start from transmuted composition
        let mut transport_material = transmuted_material;
        // Carry forward existing nuclide data from previous transport step
        transport_material.nuclide_data = cell_material.nuclide_data.clone();

        // Make sure everything now in the composition carries
        // transport-scope data. This covers nuclides transmutation
        // has just created, and equally the preloaded products that
        // hold only activation-scope cross sections and have now
        // grown a density; `ensure_nuclide_loaded` is
        // the one place that decides which of those need reading.
        let new_nuclides_loaded = {
            // Promote to transport scope only what can reach the
            // transport answer. "Does this nuclide have a
            // density?" is too broad: after one step of fission
            // 1269 of them do, and reading full transport data for
            // all of them takes 21 GB, most of it for 895
            // decay-only sinks that the product loader had
            // deliberately declined to read at all.
            //
            // Density alone is the wrong test in the other
            // direction: Xe135 at 3e-11 atoms/b-cm still holds
            // 9e7 barns, so dropping it would move the flux. So
            // rank by `density x peak cross section`, both of which
            // are already in hand, and keep everything within
            // `TRANSPORT_SHARE` of the material's dominant nuclide.
            // That is where the strong absorbers sort themselves
            // out: on the UO2 sphere Xe135, Gd157, Sm149 and Sm151
            // are the four highest peaks in the whole composition.
            let names: Vec<String> = transport_material.nuclides.keys().cloned().collect();
            let share: Vec<f64> = names
                .iter()
                .map(|name| {
                    let density = transport_material
                        .nuclides
                        .get(name)
                        .copied()
                        .unwrap_or(0.0);
                    density * peak_cross_section(&transport_material, name)
                })
                .collect();
            let dominant = share.iter().copied().fold(0.0_f64, f64::max);
            let cut = dominant * TRANSPORT_SHARE;

            let mut loaded_any = false;
            for (name, &weight) in names.iter().zip(&share) {
                if weight < cut {
                    continue;
                }
                match transport_material.ensure_nuclide_loaded(name) {
                    Ok(loaded) => {
                        loaded_any |= loaded;
                    }
                    Err(_) => {
                        // No cross section data available for this nuclide;
                        // remove it from transport (still tracked in full_compositions)
                    }
                }
            }
            loaded_any
        };
        let new_nuclides_loaded = adjust(&mut transport_material)? || new_nuclides_loaded;

        // Drop from transport anything without transport-scope data.
        // Testing presence in `nuclide_data` is not enough now that
        // preloaded products sit there at activation scope: one whose
        // widening above failed would otherwise stay in the transport
        // composition with no secondary distributions to collide on.
        // It is still tracked in full_compositions either way.
        let transportable: Vec<String> = transport_material
            .nuclides
            .keys()
            .filter(|name| transport_material.has_transport_data(name))
            .cloned()
            .collect();
        transport_material
            .nuclides
            .retain(|name, _| transportable.contains(name));

        #[cfg(feature = "debug_timing")]
        {
            let new_names: Vec<String> = transport_material
                .nuclides
                .keys()
                .filter(|name| !transport_material.nuclide_data.contains_key(*name))
                .cloned()
                .collect();
            if !new_names.is_empty() {
                eprintln!(
                    "[TIMING] Step {} mat {}: nuclides without data: {:?}",
                    step + 1,
                    mat_id,
                    new_names
                );
            }
            eprintln!(
                "[TIMING] Step {} mat {}: new_nuclides_loaded={}, transport_nuclides={}, nuclide_data={}",
                step + 1,
                mat_id,
                new_nuclides_loaded,
                transport_material.nuclides.len(),
                transport_material.nuclide_data.len()
            );
        }

        #[cfg(feature = "debug_timing")]
        let xs_start = std::time::Instant::now();

        let mt_filter = self.transport_mt_filter();
        if new_nuclides_loaded {
            // New nuclides change the energy grid and microscopic XS;
            // must do a full recomputation
            #[cfg(feature = "debug_timing")]
            eprintln!(
                "[TIMING] Step {} mat {}: FULL recomputation (new nuclides loaded)",
                step + 1,
                mat_id
            );
            transport_material.invalidate_xs_cache();
            transport_material.calculate_macroscopic_xs(&mt_filter, true);
        } else {
            // Only atom densities changed; reuse cached unified grid and micro XS
            #[cfg(feature = "debug_timing")]
            eprintln!(
                "[TIMING] Step {} mat {}: CACHED path (density-only update)",
                step + 1,
                mat_id
            );
            transport_material.unified_energy_grid_neutron =
                cell_material.unified_energy_grid_neutron.clone();
            transport_material.cached_microscopic_xs = cell_material.cached_microscopic_xs.clone();
            transport_material.invalidate_density_caches();
            transport_material.calculate_macroscopic_xs(&mt_filter, true);
        }
        if has_nuclide_tallies {
            transport_material.populate_per_nuclide_xs();
        }

        #[cfg(feature = "debug_timing")]
        eprintln!(
            "[TIMING] Step {} mat {} XS update: {:.3}s",
            step + 1,
            mat_id,
            xs_start.elapsed().as_secs_f64()
        );

        // Update the material store for every cell slot owned by this mat_id.
        let new_arc = Arc::new(transport_material);
        let slots: Vec<u32> = cell_indices
            .iter()
            .filter_map(|&cidx| self.geometry.cells()[cidx].material_idx)
            .collect();
        for slot in slots {
            self.geometry.materials_mut()[slot as usize] = Arc::clone(&new_arc);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fe56 at 294 K from the offline fixture, cross sections loaded.
    fn fe56() -> Material {
        let mut material = Material::new(
            HashMap::from([("Fe56".to_string(), 1.0)]),
            "atom",
            "g/cc",
            Some(7.874),
        )
        .unwrap();
        material.set_temperature("294");
        material
            .read_nuclear_data(
                &HashMap::from([(
                    "Fe56".to_string(),
                    format!("{}/tests/Fe56.arrow", env!("CARGO_MANIFEST_DIR")),
                )]),
                None,
            )
            .unwrap();
        material
    }

    /// The ranking behind which nuclides earn transport-scope data is
    /// `density x peak cross section`, so the peak has to be in barns. The
    /// heating and damage-energy tables sit in the same reaction map in
    /// eV-barns and run millions of times higher; ranked on those, every
    /// nuclide's order would be set by its Q values.
    #[test]
    fn the_peak_is_barns_not_the_ev_barn_tables() {
        let material = fe56();
        let peak = peak_cross_section(&material, "Fe56");

        let reactions = material.nuclide_data["Fe56"]
            .reactions_for_temp("294")
            .expect("fixture carries 294 K");
        let energy_valued = reactions
            .iter()
            .filter(|(mt, _)| **mt >= 300)
            .flat_map(|(_, r)| r.cross_section.iter().copied())
            .fold(0.0_f64, f64::max);
        assert!(
            energy_valued > 0.0,
            "fixture must carry an MT >= 300 table for this test to mean anything"
        );
        assert!(
            peak < energy_valued,
            "peak {peak:e} reached the eV-barn tables (max {energy_valued:e})"
        );
        // Iron's largest real cross section is a resonance in the low keV,
        // hundreds of barns, nowhere near the 1e6+ an eV-barn table gives.
        assert!(
            (1.0..1.0e6).contains(&peak),
            "peak {peak:e} is not a plausible cross section in barns"
        );
    }

    /// Activation scope carries no elastic channel, and a nuclide with nothing
    /// loaded has no channels at all. Either must still rank as a scatterer, or
    /// the ranking would drop a nuclide that does scatter.
    #[test]
    fn an_unrated_nuclide_ranks_as_a_scatterer() {
        let material = fe56();
        assert_eq!(peak_cross_section(&material, "Cs137"), 20.0);
    }

    /// `RUNS` independent transport runs of a 14 MeV point source in a 10 cm
    /// Fe56 sphere, each with its own history-statistics tally carrying
    /// `reactions` (kind, product), on two threads so the per-worker scratch
    /// and the stripe fold both run for real. `None` without the Fe56
    /// transport fixture.
    fn fe56_statistics_runs(
        runs: u64,
        particles: usize,
        reactions: &[(&str, &str)],
    ) -> Option<(Material, Vec<Arc<TransmutationTallies>>)> {
        use crate::geo::{BoundaryType, HalfspaceType, Region, Surface};
        use crate::geometry::cell::Cell;
        use crate::geometry::Geometry;
        use yamc_source::distribution::angular::AngularDistribution;
        use yamc_source::distribution::energy::Discrete;
        use yamc_source::distribution::spatial::Point;
        use yamc_source::source::{
            ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
        };

        let Some(fe56_path) = yamc_test_cache::transport_nuclide("Fe56") else {
            eprintln!("skipping -- Fe56 transport fixture absent");
            return None;
        };
        let mut material = Material::new(
            HashMap::from([("Fe56".to_string(), 1.0)]),
            "atom",
            "g/cc",
            Some(7.874),
        )
        .unwrap();
        material.set_temperature("294");
        material
            .read_nuclear_data(&HashMap::from([("Fe56".to_string(), fe56_path)]), None)
            .unwrap();
        material.set_material_id(1);
        material.transmutable = true;
        let chain: HashMap<String, ChainNuclide> = HashMap::from([(
            "Fe56".to_string(),
            ChainNuclide {
                name: "Fe56".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: reactions
                    .iter()
                    .map(|(kind, target)| yani::ChainReaction {
                        kind: kind.to_string(),
                        target: Some(target.to_string()),
                        branching: 1.0,
                        q_value: None,
                        branching_uncertainty: None,
                        evaluated_branching: None,
                        multiplicity: None,
                    })
                    .collect(),
                decays: vec![],
                fission_yields: None,
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
                decay_evaluated: true,
                excitation_energy: None,
                level_index: None,
                decay_energy_components: Default::default(),
            },
        )]);

        let mut out = Vec::new();
        for run in 0..runs {
            let sphere = Arc::new(Surface::sphere(
                0.0,
                0.0,
                0.0,
                10.0,
                Some(1),
                Some(BoundaryType::Vacuum),
            ));
            let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
            let cell = Cell::new(Some(1), region, None, Some(0));
            let geometry = Geometry::new(vec![cell], vec![Arc::new(material.clone())]).unwrap();
            let source = ParticleSource::Neutron(Source {
                space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
                angle: AngularDistribution::Isotropic,
                energy: SourceEnergyDistribution::Discrete(
                    Discrete::new(vec![1.406e7], vec![1.0]).unwrap(),
                ),
                strength: 1.0,
            });
            let mut model = Model::new(geometry, vec![source], vec![]);
            model.verbose = crate::model::Verbose::silent();

            let cells: HashMap<u32, Vec<usize>> = HashMap::from([(1u32, vec![0usize])]);
            let for_init: HashMap<u32, &Material> = HashMap::from([(1u32, &material)]);
            let tallies = Arc::new(
                TransmutationTallies::new(
                    &cells,
                    &for_init,
                    &chain,
                    &yani::BranchTable::new(),
                    &HashMap::new(),
                )
                .expect("no branching to read")
                .with_history_statistics(),
            );
            let settings = crate::model::TransportSettings {
                total_particles: Some(particles),
                seed: 1000 + run,
                threads: Some(2),
                ..Default::default()
            };
            model
                .run_internal::<NoOpTracker>(&settings, None, Some(Arc::clone(&tallies)), false)
                .unwrap();
            out.push(tallies);
        }
        Some((material, out))
    }

    /// Measured over predicted variance, from each run's predicted variance
    /// of a quantity and the quantity's value in each run.
    fn spread_ratio(values: &[f64], predicted: &[f64]) -> f64 {
        let n = values.len() as f64;
        let avg = values.iter().sum::<f64>() / n;
        let measured = values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / (n - 1.0);
        measured / (predicted.iter().sum::<f64>() / n)
    }

    /// With 16 runs the measured variance is a chi-squared on 15 degrees of
    /// freedom over its expectation, so a correct prediction lands in
    /// [0.42, 1.83] 95% of the time. The bounds are a little wider and the
    /// seeds fixed, so the tests below are deterministic.
    const SPREAD_BOUNDS: std::ops::RangeInclusive<f64> = 0.35..=2.2;

    /// The statistical uncertainty the transmutation tally's per-history
    /// covariance predicts must match the actual spread between independent
    /// transport runs.
    ///
    /// Two quantities are checked: the total track length, which is the sum
    /// over every bin and so rests almost entirely on the cross-bin
    /// covariance, and the fast-band track length above 1 MeV.
    #[test]
    fn history_covariance_predicts_the_run_to_run_spread() {
        const PARTICLES: usize = 1500;
        let Some((_, runs)) = fe56_statistics_runs(16, PARTICLES, &[("(n,gamma)", "Fe57")]) else {
            return;
        };

        let mut predicted = [Vec::new(), Vec::new()];
        let mut means = [Vec::new(), Vec::new()];
        for tallies in &runs {
            let cov = tallies.history_covariance(1).unwrap();
            assert_eq!(cov.n_histories, PARTICLES as u64);
            let b = cov.n_bins();
            let total: Vec<f64> = (0..cov.dim()).map(|i| (i < b) as u8 as f64).collect();
            let fast: Vec<f64> = (0..cov.dim())
                .map(|i| (i < b && cov.grid[i] >= 1.0e6) as u8 as f64)
                .collect();
            let (m_total, v_total) = cov.linear_combination(&total);
            let (m_fast, v_fast) = cov.linear_combination(&fast);

            // The folded total is the tally's own flux numerator.
            let flux_tl = tallies.get_flux(1, 1.0, 1.0);
            assert!(
                (m_total - flux_tl).abs() <= 1e-9 * flux_tl,
                "summed s0 {m_total:e} vs flux track length {flux_tl:e}"
            );
            for (q, (m, v)) in [(m_total, v_total), (m_fast, v_fast)]
                .into_iter()
                .enumerate()
            {
                means[q].push(m);
                predicted[q].push(v);
            }
        }
        for (q, name) in [(0, "total"), (1, "fast")] {
            let ratio = spread_ratio(&means[q], &predicted[q]);
            eprintln!("{name}: measured / predicted variance = {ratio:.3}");
            assert!(
                SPREAD_BOUNDS.contains(&ratio),
                "{name}: measured over predicted variance {ratio:.3}"
            );
        }
    }

    /// Every reaction rate's predicted sigma must match its run-to-run
    /// spread, over rates spanning a 1/v capture, threshold reactions from a
    /// few MeV up, and one near the 14 MeV source. The rates reported must be
    /// the accessor's, and a rate difference must get its variance from the
    /// cross terms too: two fast threshold rates scored by the same histories
    /// are positively correlated, and their difference is checked against the
    /// runs as well.
    #[test]
    fn rate_covariance_predicts_the_run_to_run_spread() {
        const REACTIONS: &[(&str, &str)] = &[
            ("(n,gamma)", "Fe57"),
            ("(n,p)", "Mn56"),
            ("(n,a)", "Cr53"),
            ("(n,2n)", "Fe55"),
        ];
        let (volume, source_rate) = (4.0 / 3.0 * std::f64::consts::PI * 1000.0, 1.0e14);
        let Some((_, runs)) = fe56_statistics_runs(16, 1500, REACTIONS) else {
            return;
        };

        let kinds: Vec<&str> = REACTIONS.iter().map(|(k, _)| *k).collect();
        let mut values = vec![Vec::new(); kinds.len() + 1];
        let mut predicted = vec![Vec::new(); kinds.len() + 1];
        let mut correlation = Vec::new();
        for tallies in &runs {
            let rc = tallies
                .get_reaction_rate_covariance(1, volume, source_rate)
                .unwrap();
            let rates = tallies.get_reaction_rates(1, volume, source_rate);
            let idx: Vec<usize> = kinds
                .iter()
                .map(|k| rc.index_of("Fe56", k, None).expect("every tallied rate"))
                .collect();
            for (q, &i) in idx.iter().enumerate() {
                let expected = rates["Fe56"][kinds[q]];
                assert!((rc.rates[i] - expected).abs() <= 1e-12 * expected);
                values[q].push(rc.rates[i]);
                predicted[q].push(rc.covariance(i, i));
            }
            // (n,p) minus (n,a): its variance needs the cross term.
            let (p, a) = (idx[1], idx[2]);
            values[kinds.len()].push(rc.rates[p] - rc.rates[a]);
            predicted[kinds.len()]
                .push(rc.covariance(p, p) + rc.covariance(a, a) - 2.0 * rc.covariance(p, a));
            correlation.push(rc.correlation(p, a));
        }
        for (q, name) in kinds.iter().chain(&["(n,p) - (n,a)"]).enumerate() {
            let ratio = spread_ratio(&values[q], &predicted[q]);
            eprintln!("{name}: measured / predicted variance = {ratio:.3}");
            assert!(
                SPREAD_BOUNDS.contains(&ratio),
                "{name}: measured over predicted variance {ratio:.3}"
            );
        }
        let mean_corr = correlation.iter().sum::<f64>() / correlation.len() as f64;
        eprintln!("corr((n,p), (n,a)) = {mean_corr:.3}");
        assert!(
            mean_corr > 0.2,
            "fast threshold rates should correlate, got {mean_corr:.3}"
        );
    }

    /// One isomer-source comparison: the relative sigma of each product
    /// after a minute's irradiation, from the transport-coupled replicas and
    /// from the spectrum path fed the same tallied flux, and the run's report.
    struct IsomerComparison {
        sigmas: Vec<(String, f64, f64)>,
        info: yani_transmute::uncertainty::Info,
        replicas: Option<yani_transmute::ReplicaTallied>,
    }

    /// A sphere of `nuclide` (`radius` cm, `density` g/cc) under a 14.06 MeV
    /// point source, tallied with the MF=40 intervals on, then resampled
    /// with `sources` both ways: through `transport_replicas` and through
    /// `transmute_material` with the tally's flux as the spectrum. The draws
    /// are a function of the seed, the replica and the parent, so both paths
    /// draw the same multipliers and differ only in the weights. With
    /// `replicas`, the transport carries that many replica weights and the
    /// transport path reads each replica's own tally.
    #[allow(clippy::too_many_arguments)]
    fn isomer_transport_against_spectrum(
        nuclide: &str,
        path: String,
        chain_dir: &std::path::Path,
        branch_dir: &std::path::Path,
        density: f64,
        radius: f64,
        products: &[&str],
        sources: Vec<yani_transmute::uncertainty::Source>,
        replicas: Option<usize>,
    ) -> IsomerComparison {
        use crate::geo::{BoundaryType, HalfspaceType, Region, Surface};
        use crate::geometry::cell::Cell;
        use crate::geometry::Geometry;
        use yamc_source::distribution::angular::AngularDistribution;
        use yamc_source::distribution::energy::Discrete;
        use yamc_source::distribution::spatial::Point;
        use yamc_source::source::{
            ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
        };
        use yani_transmute::uncertainty::{DataUncertainty, Ensemble};

        let dir = |d: std::path::PathBuf| d.to_string_lossy().into_owned();
        let loaded = yani::load_chain_parts(
            &dir(chain_dir.join("decay")),
            Some(&dir(chain_dir.join("reactions"))),
            None,
            &[("branching", &dir(branch_dir.to_path_buf()))],
        )
        .expect("chain");
        let branch = Arc::new(loaded.branch);

        let mut material = Material::new(
            HashMap::from([(nuclide.to_string(), 1.0)]),
            "atom",
            "g/cc",
            Some(density),
        )
        .unwrap();
        material.set_temperature("294");
        material
            .read_nuclear_data(&HashMap::from([(nuclide.to_string(), path)]), None)
            .unwrap();
        material.set_material_id(1);
        material.transmutable = true;
        if replicas.is_some() {
            material.ensure_covariance_loaded().unwrap();
        }
        let volume = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        material.volume = Some(volume);

        let sphere = Arc::new(Surface::new_sphere(
            0.0,
            0.0,
            0.0,
            radius,
            Some(1),
            Some(BoundaryType::Vacuum),
        ));
        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
        let cell = Cell::new(Some(1), region, None, Some(0));
        let geometry = Geometry::new(vec![cell], vec![Arc::new(material.clone())]).unwrap();
        let source = ParticleSource::Neutron(Source {
            space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
            angle: AngularDistribution::Isotropic,
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![1.406e7], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        });
        let mut model = Model::new(geometry, vec![source], vec![]);
        model.verbose = crate::model::Verbose::silent();
        let cells: HashMap<u32, Vec<usize>> = HashMap::from([(1u32, vec![0usize])]);
        let for_init: HashMap<u32, &Material> = HashMap::from([(1u32, &material)]);
        const SEED: u64 = 11;
        let tallies =
            TransmutationTallies::new(&cells, &for_init, &loaded.chain, &branch, &HashMap::new())
                .unwrap()
                .with_isomer_intervals(&loaded.chain, &branch)
                .unwrap();
        let tallies = Arc::new(match replicas {
            Some(n) => tallies.with_replicas(n, SEED),
            None => tallies,
        });
        let settings = crate::model::TransportSettings {
            total_particles: Some(20_000),
            seed: 7,
            threads: Some(2),
            data_uncertainty: replicas.map(|replicas| crate::model::TransportDataUncertainty {
                seed: SEED,
                replicas,
            }),
            ..Default::default()
        };
        model
            .run_internal::<NoOpTracker>(&settings, None, Some(Arc::clone(&tallies)), false)
            .unwrap();

        let spectrum = tallies.flux_spectrum(1, volume).expect("flux");
        let tallied = yani_transmute::TransportTallied {
            rates: tallies.get_reaction_rates(1, volume, 1.0),
            partials: tallies.get_partial_rates(1, volume, 1.0),
            fy_weights: tallies.get_fission_yield_weights(1),
            spectrum: spectrum.clone(),
            statistics: None,
            branch: Arc::clone(&branch),
            diagnostics: tallies
                .get_branching_diagnostics(1, volume, 1.0)
                .with_material(&material),
            isomer: tallies.get_isomer_productions(1, volume, 1.0),
            replicas: tallies.replica_tallied(1, volume),
            draw_key: 0,
        };
        let mut initial = material.clone();
        initial.nuclides = material.get_atoms_per_barn_cm().unwrap();
        initial.density_units = DensityUnits::Sum;
        initial.density = None;
        let request = DataUncertainty {
            seed: SEED,
            samples: Some(replicas.unwrap_or(400)),
            sources,
            attribution: false,
            ..Default::default()
        };
        let (dt, rate) = (60.0, 1.0e14);
        let (transport, info) = yani_transmute::transport_replicas(
            &initial,
            &tallied,
            &[dt],
            &[rate],
            &loaded.chain,
            loaded.parts,
            &request,
            &Default::default(),
        )
        .unwrap();
        let mut spectral = initial.clone();
        let results = yani_transmute::transmute_material(
            &mut spectral,
            &[spectrum],
            &[yani_transmute::TransmuteStep {
                dt,
                irradiation: Some((0, rate)),
            }],
            Arc::clone(&loaded.chain),
            &branch,
            loaded.parts,
            Some(&request),
        )
        .unwrap();
        let relative = |e: &Ensemble, n: &str| {
            let x = e.samples_at(0, n);
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64;
            var.sqrt() / mean
        };
        let sigmas = products
            .iter()
            .map(|p| {
                let (t, s) = (
                    relative(&transport, p),
                    relative(&results.uncertainty[&1], p),
                );
                eprintln!("{nuclide} -> {p}: transport {t:.6}, spectrum path {s:.6}");
                (p.to_string(), t, s)
            })
            .collect();
        IsomerComparison {
            sigmas,
            info,
            replicas: tallied.replicas,
        }
    }

    /// The `isomer_branching` source on a transport run: an In115 sphere under
    /// a 14 MeV point source, ENDF/B-VIII.1 In115 with its own MF=40. The
    /// tally scores each stated state's production per MF=40 interval, and
    /// the replicas re-split the tallied partials with them. Fed the same
    /// tallied flux as a spectrum, the spectrum path draws the same
    /// multipliers, so the isomers' sigmas must agree to the small difference
    /// between the tally's continuous-energy weights and the spectrum path's
    /// group fold. Self-skips without the fixtures.
    #[test]
    fn in115_isomer_branching_on_transport_matches_the_spectrum_path() {
        let Some(in115) = yamc_test_cache::transport_nuclide("In115") else {
            eprintln!("skipping -- In115 transport fixture absent");
            return;
        };
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let chain = manifest.join("tests/transmutation-endf-b8.1-sfr.arrow");
        let branch =
            manifest.join("../yani-transmute/tests/fixtures/endf-b8.1-branching-In115-Mo92");
        if !chain.join("reactions").is_dir() || !branch.is_dir() {
            eprintln!("skipping -- chain or branching fixture absent");
            return;
        }
        let run = isomer_transport_against_spectrum(
            "In115",
            in115,
            &chain,
            &branch,
            7.31,
            3.0,
            &["In114_m1", "In115_m1"],
            vec![yani_transmute::uncertainty::Source::IsomerBranching],
            None,
        );
        for kind in ["(n,2n)", "(n,n')"] {
            assert!(
                run.info
                    .isomer_branchings_perturbed
                    .contains(&("In115".to_string(), kind.to_string())),
                "{kind}: {:?}",
                run.info.isomer_branchings_perturbed
            );
        }
        assert!(run.info.sources.contains(&"isomer_branching".to_string()));
        assert!(!run
            .info
            .not_perturbed
            .iter()
            .any(|s| s.contains("every split held")));
        for (product, t, s) in &run.sigmas {
            assert!(*t > 0.005, "{product}: the split moves, sigma {t}");
            assert!((t / s - 1.0).abs() < 0.01, "{product}: {t} against {s}");
        }
    }

    /// With replica weights each replica's MF=40 interval productions are
    /// scored under that replica's weights and cross sections, as its
    /// partials are: on every replica a state's intervals sum to the
    /// replica's own partial, so the weights `w_j / P` the draw folds with
    /// are the replica's. Self-skips without the fixtures.
    #[test]
    fn in115_isomer_productions_follow_each_replica() {
        use yani_transmute::uncertainty::Source as S;
        let Some(in115) = yamc_test_cache::transport_nuclide("In115") else {
            eprintln!("skipping -- In115 transport fixture absent");
            return;
        };
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let chain = manifest.join("tests/transmutation-endf-b8.1-sfr.arrow");
        let branch =
            manifest.join("../yani-transmute/tests/fixtures/endf-b8.1-branching-In115-Mo92");
        if !chain.join("reactions").is_dir() || !branch.is_dir() {
            eprintln!("skipping -- chain or branching fixture absent");
            return;
        }
        let run = isomer_transport_against_spectrum(
            "In115",
            in115,
            &chain,
            &branch,
            7.31,
            3.0,
            &["In114_m1", "In115_m1"],
            vec![S::CrossSections, S::IsomerBranching],
            Some(16),
        );
        let replicas = run.replicas.expect("replica tallies");
        assert_eq!(replicas.isomer.len(), 16);
        let mut sums = Vec::new();
        for k in 0..16 {
            for (kind, key, target) in [
                ("(n,2n)", (16, 1), "In114_m1"),
                ("(n,n')", (4, 1), "In115_m1"),
            ] {
                let w: f64 = replicas.isomer[k]
                    .interval_sum("In115", key)
                    .expect("tallied");
                let partial = replicas.partials[k]["In115"][kind]
                    .iter()
                    .find(|(t, _)| t == target)
                    .unwrap()
                    .1;
                assert!(
                    (w - partial).abs() <= 1e-9 * partial,
                    "replica {k} {target}: {w} vs {partial}"
                );
                sums.push(w);
            }
        }
        assert!(run.info.sources.contains(&"isomer_branching".to_string()));
        for (product, t, _) in &run.sigmas {
            assert!(*t > 0.0, "{product}");
        }
    }

    /// The same comparison on JEFF-4.0's Nb93 `(n,2n)` and Ni58 `(n,p)`, read
    /// from a local release tree: `YAMC_DATA_DIR` names the directory holding
    /// `jeff-4.0-arrow` and `transmutation-jeff-4.0.arrow`. Ignored by default.
    /// The MF=40 source alone must agree to the weights' difference; with the
    /// cross sections too it is printed for the record.
    #[test]
    #[ignore]
    fn jeff40_isomer_branching_on_transport_matches_the_spectrum_path() {
        use yani_transmute::uncertainty::Source as S;
        let Some(data) = std::env::var_os("YAMC_DATA_DIR").map(std::path::PathBuf::from) else {
            eprintln!("skipping -- YAMC_DATA_DIR unset");
            return;
        };
        let chain = data.join("transmutation-jeff-4.0.arrow");
        let neutron = data.join("jeff-4.0-arrow/neutron");
        for (nuclide, density, products) in [
            ("Nb93", 8.57, ["Nb92", "Nb92_m1"]),
            ("Ni58", 8.9, ["Co58", "Co58_m1"]),
        ] {
            for (sources, replicas) in [
                (vec![S::IsomerBranching], None),
                (vec![S::CrossSections, S::IsomerBranching], None),
                (vec![S::CrossSections], None),
                (vec![S::CrossSections, S::IsomerBranching], Some(128)),
                (vec![S::CrossSections], Some(128)),
            ] {
                eprintln!("{nuclide} {sources:?} replicas {replicas:?}");
                let alone = sources == [S::IsomerBranching];
                let path = neutron.join(format!("{nuclide}.arrow"));
                let run = isomer_transport_against_spectrum(
                    nuclide,
                    path.to_string_lossy().into_owned(),
                    &chain,
                    &chain.join("branching"),
                    density,
                    3.0,
                    &products,
                    sources,
                    replicas,
                );
                // With cross sections on, the two paths draw MF=33 (and
                // MF=32) differently, so the sigmas are printed, not compared.
                for (product, t, s) in &run.sigmas {
                    if alone {
                        assert!((t / s - 1.0).abs() < 0.02, "{product}: {t} against {s}");
                    }
                }
                // Each replica's interval productions are its own: they sum
                // to its partials, and they move from replica to replica.
                if let Some(r) = &run.replicas {
                    // The production-weighted change of the stated states'
                    // productions from replica 0, largest over the replicas.
                    let mut moved = 0.0f64;
                    for (k, isomer) in r.isomer.iter().enumerate() {
                        let (mut diff, mut base) = (0.0, 0.0);
                        for (parent, key, w) in isomer.iter() {
                            let a: f64 = w.iter().sum();
                            let b = r.isomer[0].interval_sum(parent, key).unwrap_or(0.0);
                            diff += (a - b).abs();
                            base += b;
                        }
                        assert!(isomer.states() > 0, "replica {k}");
                        moved = moved.max(diff / base);
                    }
                    eprintln!(
                        "  largest change of the stated productions from replica 0: {moved:.4}"
                    );
                    assert!(moved > 0.0);
                }
            }
        }
    }
}
