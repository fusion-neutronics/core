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
use yamc_nuclide::config::Config;
use yamc_nuclide::nuclide::get_or_load_nuclide;
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
    ///   applies. `None` is the default path, bit-identical to a build without
    ///   it; with the coupled method it is an error.
    ///
    /// Product cross sections are loaded for the material's whole reachable
    /// closure (`yani::reachable_nuclides`) and then narrowed to the products
    /// that irradiation can populate above the solver's density floor
    /// (`yani::populated_nuclides`), so which products are carried is derived
    /// from the composition and the fluence rather than from a caller-supplied
    /// depth. A cheap scouting transport runs first to supply the flux spectrum
    /// that bound is folded against.
    /// Product nuclide paths are resolved from CONFIG.cross_sections or material data.
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
    ) -> Result<TransmutationResults, Box<dyn std::error::Error>> {
        // Validate method
        let is_independent = match method {
            "independent" => true,
            "coupled" => false,
            _ => {
                return Err(format!(
                    "method must be \"independent\" or \"coupled\", got \"{method}\""
                )
                .into())
            }
        };
        // Coupled mode re-runs transport every step, so its rates carry noise
        // that feeds forward through the compositions; resampling one step's
        // rates would not describe that, and a number that quietly means
        // something else is worse than none.
        if uncertainty.is_some() && !is_independent {
            return Err(
                "data_uncertainty is supported with method=\"independent\" only: \
                        the coupled method re-runs transport every step, and propagating \
                        its step-to-step statistical noise is out of scope"
                    .into(),
            );
        }
        let want_statistics =
            uncertainty.is_some_and(|u| u.wants(yani_transmute::uncertainty::Source::Statistical));

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
                let _ = material.ensure_nuclides_loaded();
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

        // The whole irradiation, as the product bound below sees it: every
        // step's duration (decay runs during cooling steps too) and the largest
        // source rate any step reaches, so a rate folded at that rate bounds
        // every step's.
        let total_time: f64 = timesteps.iter().sum();
        let peak_source_rate = source_rates.iter().copied().fold(0.0, f64::max);

        // Scouting transport: the flux spectrum the product bound needs.
        //
        // A product's reaction rate comes from continuous-energy scoring during
        // transport, so it cannot be folded from a stored group flux after the
        // fact. What CAN be folded afterwards is a bound on it, from the flux
        // moments this tally accumulates per energy bin (one binary search and
        // two adds per track segment, whatever the nuclide count), and a bound
        // is all that deciding which products to carry needs.
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
            println!("Scouting transport for the flux spectrum the product bound needs...");
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
            // needs a cross section. The bound reaches the same conclusion from
            // rates that are all zero, which is exactly right here rather than
            // merely conservative, so there is nothing to scout for.
            None
        };

        #[cfg(feature = "debug_timing")]
        eprintln!(
            "[TIMING] Scouting transport for the product bound: {:.3}s",
            scout_start.elapsed().as_secs_f64()
        );

        // Load cross sections for every product the material can reach, then
        // keep the ones the irradiation can actually populate.
        //
        // A caller-supplied hop count cannot express the right answer: how far
        // a product chain runs depends on the fluence, not on graph distance.
        // The reachable closure is derived from the composition instead, so
        // there is nothing to set.
        //
        // Reachability is blunt, though: it saturates on one large strongly
        // connected component, so Fe56, water and steel all reach the same 461
        // reactive nuclides while a real solve of an iron sphere populates 49.
        // `yani::populated_nuclides` carries an upper bound on the density each
        // node could attain and drops those that stay under the solver's own
        // floor, so depth tracks fluence and there is still nothing to set.
        //
        // The closure is large, so these load at activation scope: the MTs the
        // network names and no transport sections, the same scope
        // `transmute_material` uses. A product enters the run at
        // zero density and so cannot be collided on; if one later grows a
        // density, the coupled loop's `ensure_nuclide_loaded` widens it to
        // transport scope on demand.
        #[cfg(feature = "debug_timing")]
        let preload_start = std::time::Instant::now();

        // Per material, the products the bound says are worth scoring. A
        // material missing from this map scores everything it has, which is
        // what happens when nothing was irradiated or nothing was loaded.
        let mut carried: HashMap<u32, std::collections::HashSet<String>> = HashMap::new();

        {
            let product_mts = yani_transmute::activation_mts(&chain, &branch);
            for (&mat_id, cell_indices) in &transmutable_cells {
                let cell_idx = cell_indices[0];
                let first_mat_idx = self.geometry.cells()[cell_idx]
                    .material_idx
                    .expect("transmutable cell must have a material");
                let material_arc = &self.geometry.materials()[first_mat_idx as usize];
                let mut material = (**material_arc).clone();

                // Walk the chain from the material's own nuclides. Seeds are the
                // composition UNION whatever is already loaded, since a material
                // built in Python may carry no `nuclide_data` yet.
                let seeds: Vec<String> = material
                    .nuclides
                    .keys()
                    .chain(material.nuclide_data.keys())
                    .cloned()
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
                let reachable = yani::reachable_nuclides(&chain, &seed_refs);

                // Filter to nuclides not already loaded, and skip decay-only
                // sinks: chain nodes with no neutron reactions don't need
                // cross-section data.
                let to_load: Vec<&String> = reachable
                    .iter()
                    .filter(|n| !material.nuclide_data.contains_key(*n))
                    .filter(|n| chain.get(*n).is_some_and(|cn| !cn.reactions.is_empty()))
                    .collect();

                if to_load.is_empty() {
                    continue;
                }

                // Load product nuclides from CONFIG - warn about missing ones
                // Collect paths from config, then drop the lock before loading
                let source_paths: Vec<(String, String)> = {
                    let config = Config::global();
                    to_load
                        .iter()
                        .filter_map(|nuclide_name| {
                            match config.get_cross_section(nuclide_name) {
                                Some(path) => {
                                    Some(((*nuclide_name).clone(), path))
                                }
                                None => {
                                    eprintln!(
                                        "\x1b[33mWarning: Transmutation product \"{nuclide_name}\" not found. \
                                         It will not participate in neutron transport reactions.\x1b[0m"
                                    );
                                    None
                                }
                            }
                        })
                        .collect()
                };

                // What this block put into `nuclide_data`, and so the only
                // thing the bound below may take back out again: whatever the
                // material arrived with is the caller's, not a product.
                let mut loaded: std::collections::HashSet<String> =
                    std::collections::HashSet::new();

                for (nuclide_name, h5_path) in &source_paths {
                    let mut source_map = HashMap::new();
                    source_map.insert(nuclide_name.clone(), h5_path.clone());

                    // Use material's temperature to ensure consistency
                    let temp_filter = if !material.temperature().is_empty() {
                        let mut temps = std::collections::HashSet::new();
                        temps.insert(material.temperature().to_string());
                        Some(temps)
                    } else {
                        None
                    };

                    let scope = yamc_nuclide::LoadScope::activation(product_mts.clone())
                        .with_temperatures(temp_filter.clone());
                    match get_or_load_nuclide(nuclide_name, &source_map, &scope) {
                        Ok(nuclide) => {
                            material.nuclide_data.insert(nuclide_name.clone(), nuclide);
                            loaded.insert(nuclide_name.clone());
                        }
                        Err(e) => {
                            eprintln!(
                                "\x1b[33mWarning: Failed to load chain nuclide \"{nuclide_name}\": {e}\x1b[0m"
                            );
                        }
                    }
                }

                if !loaded.is_empty() {
                    // Work out which of these products the irradiation can
                    // actually populate. The answer decides what the tallies
                    // score, not what the material carries: leave the loaded
                    // cross sections where they are.
                    //
                    // Scoring a product costs `n_MTs` cross-section lookups on
                    // every track segment of every step, so the bill is
                    // `n_nuclides x n_MTs x n_segments` and grows without bound
                    // in particle count: 461 products cost 12.1 s against 3.3 s
                    // for 50 at 100 000 particles on the Fe56 sphere.
                    //
                    // Every parent here is rated for real, from the scouting
                    // spectrum. That is what makes the bound a bound: rating an
                    // unloaded parent at zero is not sound (a node fed by
                    // enough sub-floor parents clears the floor, and zeroed
                    // edges hide that), and rating it at a cross-section
                    // ceiling instead is sound but only terminates once the
                    // whole closure is loaded anyway, returning this same set
                    // after thirteen rounds instead of one. Nothing is rated at
                    // zero except what this run would also rate at zero: a
                    // nuclide with no cross-section data cannot react here any
                    // more than it can in the solve.
                    if let Some(scout) = &scout {
                        let volume = material.volume.unwrap_or(1.0);
                        let bound_rates = scout.bounding_reaction_rates(
                            mat_id,
                            &material,
                            &chain,
                            volume,
                            peak_source_rate,
                        );
                        let initial = material.get_atoms_per_barn_cm()?;
                        let mut keep = yani::populated_nuclides(
                            &chain,
                            &branch,
                            &initial,
                            total_time,
                            yani_transmute::DENSITY_FLOOR,
                            |parent, kind| {
                                bound_rates
                                    .get(parent)
                                    .and_then(|kinds| kinds.get(kind))
                                    .copied()
                                    .unwrap_or(0.0)
                            },
                        );
                        // The transport composition is not the bound's to rule
                        // on: those nuclides are collided on, not produced.
                        keep.extend(material.nuclides.keys().cloned());
                        if self.verbose.nuclear_data {
                            let scored = material
                                .nuclide_data
                                .keys()
                                .filter(|n| chain.contains_key(*n) && keep.contains(*n))
                                .count();
                            println!(
                                "Transmutation products for material {mat_id}: {scored} of {} \
                                 can reach {:e} atoms/b-cm over {total_time:.3e} s",
                                material.nuclide_data.len(),
                                yani_transmute::DENSITY_FLOOR,
                            );
                        }
                        carried.insert(mat_id, keep);
                    }

                    // Validate temperature consistency after loading new nuclides
                    if let Err(e) = material.validate_temperature_consistency() {
                        eprintln!(
                            "\x1b[33mWarning: Temperature inconsistency after loading transmutation products: {e}\x1b[0m"
                        );
                    }
                    // Products are deliberately NOT seeded into `nuclides` at zero
                    // density. That used to pull their energy points into the
                    // unified grid up front to save a rebuild later, which was
                    // affordable for the handful of nuclides a hop count of 1 or 2
                    // reached. Over the full closure it would put hundreds of grids
                    // into every lookup, and activation-scope data has no transport
                    // sections to offer transport anyway.
                    // Use full transport MT filter so the microscopic XS cache
                    // is valid for subsequent transport steps
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
            Arc::new(if want_statistics {
                tallies.with_history_statistics()
            } else {
                tallies
            })
        };

        // Initialize results
        let mut results = TransmutationResults::new(timesteps.to_vec());

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

            self.run_internal::<NoOpTracker>(
                settings,
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
                            diagnostics: dep_tallies.get_branching_diagnostics(mat_id, volume, 1.0),
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

        // Transmutation loop
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

                    // Extract rates from transmutation tallies for each material
                    let mut step_rates: HashMap<u32, ReactionRates> = HashMap::new();
                    for (&mat_id, cell_indices) in &transmutable_cells {
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
            } else {
                // Cooling period - no reactions, decay only
                HashMap::new()
            };

            // Isomeric-branching overlay: every list on a material nuclide is
            // scored at the collision energies as `yani_transmute::
            // branching_rule` defines it, and the (n,n') partials
            // of every other chain parent are folded from the tally's
            // union-grid flux moments (covering products that build up during
            // the step too), their rates injected. When no branching
            // subsection is configured the overlay is empty and the physics is
            // identical to before. MT=5, whose products are not modelled, is
            // measured and reported either way.
            let mut folded_chains: HashMap<u32, Arc<HashMap<String, ChainNuclide>>> =
                HashMap::new();
            let mut step_reports: HashMap<u32, Arc<yani_transmute::BranchingReport>> =
                HashMap::new();
            if source_rate > 0.0 {
                for (&mat_id, cell_indices) in &transmutable_cells {
                    let slot = self.geometry.cells()[cell_indices[0]]
                        .material_idx
                        .expect("transmutable cell must have a material");
                    let cell_material = self.geometry.materials()[slot as usize].as_ref();
                    let volume = cell_material.volume.unwrap_or(1.0);
                    let diagnostics =
                        dep_tallies.get_branching_diagnostics(mat_id, volume, source_rate);
                    let partials = if branch.curves().is_empty() {
                        HashMap::new()
                    } else {
                        dep_tallies.get_partial_rates(mat_id, volume, source_rate)
                    };
                    let mat_rates = rates.entry(mat_id).or_default();
                    let named = |e: String| -> Box<dyn std::error::Error> {
                        format!("material {mat_id}, step {}: {e}", step + 1).into()
                    };
                    let (folded, report) = apply_coupled_branching(
                        &chain,
                        &branch,
                        &partials,
                        mat_rates,
                        Some(&diagnostics),
                    )
                    .map_err(named)?;
                    if !partials.is_empty() {
                        folded_chains.insert(mat_id, folded);
                    }
                    step_reports.insert(mat_id, Arc::new(report));
                }
            }

            // Transmute all materials in parallel (each material is independent)
            #[cfg(feature = "debug_timing")]
            let transmute_solve_start = std::time::Instant::now();

            let transmuted_materials: Vec<(u32, Result<Material, String>)> = transmutable_cells
                .par_iter()
                .map(|(&mat_id, _cell_indices)| {
                    let transmutation_input = full_compositions.get(&mat_id).unwrap();
                    let mat_rates = rates.get(&mat_id).cloned().unwrap_or_default();
                    let mat_fy = fy_weights.get(&mat_id).cloned().unwrap_or_default();
                    // Use the per-material branch-folded chain when present,
                    // else the base chain. (driver.transmute_material is just
                    // stepper.step over its fixed chain; this lets each material
                    // use its own spectrum-folded branching.)
                    let ch = folded_chains.get(&mat_id).unwrap_or(&chain);
                    let result = ForwardEulerStepper
                        .step(transmutation_input, ch, &mat_rates, &mat_fy, parts, dt)
                        .map_err(|e| e.to_string());
                    (mat_id, result)
                })
                .collect();

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
                        let names: Vec<String> =
                            transport_material.nuclides.keys().cloned().collect();
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
                        transport_material.cached_microscopic_xs =
                            cell_material.cached_microscopic_xs.clone();
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

        // Uncertainty: resample what the single transport produced and re-solve
        // the same schedule, per material, independently of the nominal loop
        // above, which is untouched and so bit-identical either way.
        if let Some(request) = uncertainty {
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
                )?;
                results.uncertainty.insert(*mat_id, ensemble);
                results.uncertainty_info.insert(*mat_id, info);
                if let Some(c) = transport.statistics.clone() {
                    results.rate_covariance.insert(*mat_id, c);
                }
            }
        }

        Ok(results)
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
                    })
                    .collect(),
                decays: vec![],
                fission_yields: None,
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
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
}
