/// Transmutation stepper trait and implementations.
///
/// Steppers solve the Bateman equations for one timestep given
/// material composition, chain data, and reaction rates.
use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;

use yamc_materials::material::Material;
use yani::matrix::{decay_particle_products, light_particle_products};
use yani::{
    build_matrix_triplets, cram48_sparse, ChainNuclide, ChainParts, FissionYieldWeights,
    ReactionRates,
};

/// The atom density [atoms/barn-cm] below which a solved nuclide is dropped.
///
/// One atom per cubic metre, near enough. This is the solver's own definition
/// of "populated", so it is also what a driver deciding which nuclides are
/// worth carrying should test a bound against
/// (`yani::populated_nuclides`, issue #404) rather than inventing a second
/// threshold that could disagree with this one.
pub const DENSITY_FLOOR: f64 = 1e-30;

/// Transmutation stepper trait.
///
/// Implementors solve the Bateman equations for a single material
/// over one timestep.
pub trait TransmutationStepper: Send + Sync {
    /// Integrate material composition over one timestep.
    ///
    /// # Arguments
    /// * `material` - Current material composition
    /// * `chain` - Transmutation chain data
    /// * `rates` - Reaction rates (sigma * phi) for this material
    /// * `fy_weights` - Spectrum weights over each fissionable nuclide's
    ///   tabulated fission-yield energies (issue #379). Required for every
    ///   nuclide with yields and a non-zero fission rate.
    /// * `parts` - Which optional subsections `chain` was built from, so a rate
    ///   needing one that was left out is refused rather than solved without it
    /// * `dt` - Timestep duration [s]
    ///
    /// # Returns
    /// New material with updated composition (in sum mode)
    fn step(
        &self,
        material: &Material,
        chain: &HashMap<String, ChainNuclide>,
        rates: &ReactionRates,
        fy_weights: &FissionYieldWeights,
        parts: ChainParts,
        dt: f64,
    ) -> Result<Material, Box<dyn Error>>;
}

/// Simple predictor stepper (1st order, forward Euler).
///
/// Uses beginning-of-step reaction rates for the entire timestep.
/// This is the simplest stepper - accurate for small timesteps
/// or when composition doesn't change significantly.
pub struct ForwardEulerStepper;

impl TransmutationStepper for ForwardEulerStepper {
    fn step(
        &self,
        material: &Material,
        chain: &HashMap<String, ChainNuclide>,
        rates: &ReactionRates,
        fy_weights: &FissionYieldWeights,
        parts: ChainParts,
        dt: f64,
    ) -> Result<Material, Box<dyn Error>> {
        // Get absolute atom densities [atoms/barn-cm]
        let atoms_per_bcm = material.get_atoms_per_barn_cm()?;

        // Collect all nuclides: existing in material + reachable products from chain
        let (names, n0) = collect_nuclides_and_densities_from_hashmap(&atoms_per_bcm, chain, rates);

        if names.is_empty() {
            return Ok(material.clone());
        }

        // Build transmutation matrix as sparse triplets
        let (triplets, mat_n) = build_matrix_triplets(chain, &names, rates, fy_weights, parts)?;

        #[cfg(feature = "debug_transmutation")]
        {
            eprintln!(
                "[DEBUG transmutation] Transmutation matrix ({}x{}, {} nonzeros):",
                mat_n,
                mat_n,
                triplets.len()
            );
            eprintln!("[DEBUG transmutation] Nuclide order: {:?}", names);
            eprintln!("[DEBUG transmutation] Initial densities n0:");
            for (i, name) in names.iter().enumerate() {
                if n0[i] > 0.0 {
                    eprintln!("  [{:3}] {:10} {:.6e}", i, name, n0[i]);
                }
            }
            eprintln!("[DEBUG transmutation] Non-zero matrix entries (A[row,col] = value):");
            for &(row, col, val) in &triplets {
                if val.abs() > 1e-30 {
                    eprintln!(
                        "  A[{},{}] ({} <- {}) = {:.6e}",
                        row, col, names[row], names[col], val
                    );
                }
            }
        }

        // Solve using sparse CRAM48
        let n_final = cram48_sparse(&triplets, mat_n, &n0, dt)
            .map_err(|e| format!("CRAM48 solver error: {e}"))?;

        #[cfg(feature = "debug_transmutation")]
        {
            eprintln!("[DEBUG transmutation] CRAM48 result (dt={:.6e} s):", dt);
            for (i, name) in names.iter().enumerate() {
                if n_final[i] > DENSITY_FLOOR {
                    eprintln!("  {:10} {:.6e} atoms/b-cm", name, n_final[i]);
                }
            }
        }

        // Create new material with updated composition (in sum mode).
        //
        // Where an exact answer exists, use it instead of the solver's. A
        // nuclide that nothing feeds over this step obeys `n(dt) = n0 *
        // exp(A_ii dt)` and nothing else, where `A_ii` is its own diagonal, so
        // its row needs no linear solve at all; CRAM48's LU fill couples every
        // row to the largest density in the solve, and what it returns for
        // those rows is a residue around 1e-22 of that largest density rather
        // than the answer.
        //
        // That residue is invisible in an inventory and loud in the outputs
        // that weight by a decay constant, which spans ten orders of magnitude
        // across the chain: cooled reactor graphite reported two thirds of its
        // decay heat from B12, a 20.2 ms emitter, on a density the solver had
        // invented after a cooldown that should have reduced it by exp(-2.96e6)
        // (issue #410).
        //
        // Being fed has to be traced, not just read off the matrix row: a
        // parent that starts at zero and grows during the step feeds its
        // daughter perfectly well. And being short-lived is not the test, which
        // is the trap this went through twice. Over a year-long pulse every
        // nuclide with a sub-hour half-life has a vanishing carry-over, and the
        // ones held in equilibrium by their own production are real
        // inventories; a stable nuclide at 1e-25 that nothing feeds is a real
        // inventory too, and comes through this untouched because its analytic
        // answer is its own starting density.
        //
        // The daughter of an unfed nuclide is itself fed, so it keeps the
        // solver's value while its parent takes the closed form. Both are the
        // same ODE from the same initial condition and only the row where the
        // closed form is exact gets replaced, so nothing is counted twice.
        let fed = fed_during_step(&triplets, &n0);
        let loss = total_loss_rate(&triplets, names.len());
        let mut new_densities = HashMap::new();
        for (i, name) in names.iter().enumerate() {
            let value = if fed[i] {
                n_final[i]
            } else {
                n0[i] * (loss[i] * dt).exp()
            };
            if value > DENSITY_FLOOR {
                new_densities.insert(name.clone(), value);
            }
        }

        Ok(Material::from_nuclide_densities_without_data(
            new_densities,
            material,
        ))
    }
}

/// Each nuclide's diagonal, the total rate at which it leaves: decay plus every
/// reaction channel the flux drives.
///
/// Taken from the matrix rather than from the half-life, because the half-life
/// is only half of it. An unfed nuclide that still burns (a seed with nothing
/// producing it, which is most seeds) would otherwise be handed back its
/// starting density and never deplete.
fn total_loss_rate(triplets: &[(usize, usize, f64)], n: usize) -> Vec<f64> {
    let mut loss = vec![0.0; n];
    for &(row, col, value) in triplets {
        if row == col {
            loss[row] += value;
        }
    }
    loss
}

/// Which nuclides anything can flow into over this step.
///
/// Reachability, not a single matrix row: a parent that starts empty and grows
/// during the step feeds its daughter, so the walk follows the off-diagonal
/// production edges outward from every nuclide that starts with a density.
/// Whatever it does not reach can only decay, and has an analytic answer.
fn fed_during_step(triplets: &[(usize, usize, f64)], n0: &[f64]) -> Vec<bool> {
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(row, col, value) in triplets {
        if row != col && value > 0.0 {
            children.entry(col).or_default().push(row);
        }
    }
    let mut fed = vec![false; n0.len()];
    let mut queue: VecDeque<usize> = n0
        .iter()
        .enumerate()
        .filter(|(_, &d)| d > 0.0)
        .map(|(i, _)| i)
        .collect();
    while let Some(parent) = queue.pop_front() {
        for &child in children.get(&parent).into_iter().flatten() {
            if !fed[child] {
                fed[child] = true;
                queue.push_back(child);
            }
        }
    }
    fed
}

/// Collect all nuclides relevant for transmutation from a HashMap of densities.
///
/// Performs a BFS from nuclides with non-zero density through the chain,
/// following all decay pathways (always active) and only reaction pathways
/// with non-zero rates. This avoids including unreachable nuclides that
/// would inflate the transmutation matrix.
fn collect_nuclides_and_densities_from_hashmap(
    nuclides: &HashMap<String, f64>,
    chain: &HashMap<String, ChainNuclide>,
    rates: &ReactionRates,
) -> (Vec<String>, Vec<f64>) {
    // BFS from nuclides with non-zero density through chain
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();

    for (name, &density) in nuclides.iter() {
        if density > 0.0 && chain.contains_key(name) {
            visited.insert(name.clone());
            queue.push_back(name.clone());
        }
    }

    while let Some(name) = queue.pop_front() {
        if let Some(nuc) = chain.get(&name) {
            // Always follow decay pathways (decays happen regardless of flux)
            for decay in &nuc.decays {
                if let Some(target) = &decay.target {
                    if chain.contains_key(target) && visited.insert(target.clone()) {
                        queue.push_back(target.clone());
                    }
                }
                for &(particle, _) in decay_particle_products(&decay.kind) {
                    let p = particle.to_string();
                    if chain.contains_key(&p) && visited.insert(p.clone()) {
                        queue.push_back(p);
                    }
                }
            }

            // Only follow reaction pathways with non-zero rates
            if let Some(rate_map) = rates.get(name.as_str()) {
                for rx in &nuc.reactions {
                    let rate = *rate_map.get(rx.kind.as_str()).unwrap_or(&0.0);
                    if rate > 0.0 {
                        if let Some(target) = &rx.target {
                            if chain.contains_key(target) && visited.insert(target.clone()) {
                                queue.push_back(target.clone());
                            }
                        }
                        for &(particle, _) in light_particle_products(&rx.kind) {
                            let p = particle.to_string();
                            if chain.contains_key(&p) && visited.insert(p.clone()) {
                                queue.push_back(p);
                            }
                        }
                        // Follow fission product nuclides. The union over every
                        // tabulated energy, not just one: which energies the
                        // fold actually weights depends on the spectrum, and a
                        // superset keeps network discovery independent of that
                        // (a product that ends up with zero yield only costs an
                        // empty matrix row).
                        if rx.kind.contains("fission") {
                            if let Some(fy_set) = &nuc.fission_yields {
                                for fy in &fy_set.yields {
                                    for (product, yield_val) in &fy.products {
                                        if *yield_val > 0.0
                                            && chain.contains_key(product)
                                            && visited.insert(product.clone())
                                        {
                                            queue.push_back(product.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "debug_transmutation")]
    {
        let total_chain = chain.len();
        let selected = visited.len();
        eprintln!(
            "[DEBUG transmutation] Matrix reduction: {selected}/{total_chain} nuclides selected \
             ({:.1}% reduction)",
            100.0 * (1.0 - selected as f64 / total_chain as f64)
        );
    }

    // Sort for deterministic ordering
    let mut names: Vec<String> = visited.into_iter().collect();
    names.sort();

    // Build initial density vector
    let n0: Vec<f64> = names
        .iter()
        .map(|name| *nuclides.get(name).unwrap_or(&0.0))
        .collect();

    (names, n0)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use yani::{ChainNuclide, ChainReaction, ReactionRates};

    fn simple_decay_chain() -> HashMap<String, ChainNuclide> {
        let mut chain = HashMap::new();
        // Co60 -> Ni60 (beta- decay, t_1/2 = 5.27 years)
        let half_life = 5.2714 * 365.25 * 24.0 * 3600.0; // seconds
        chain.insert(
            "Co60".to_string(),
            ChainNuclide {
                name: "Co60".to_string(),
                half_life: Some(half_life),
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![ChainReaction {
                    kind: "beta-".to_string(),
                    target: Some("Ni60".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                fission_yields: None,
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        chain.insert(
            "Ni60".to_string(),
            ChainNuclide {
                name: "Ni60".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        chain
    }

    #[test]
    fn test_predictor_decay_only() {
        let chain = simple_decay_chain();
        let half_life = 5.2714 * 365.25 * 24.0 * 3600.0;

        // Create material with Co60
        let material = Material::new(
            HashMap::from([("Co60".to_string(), 1.0e-3)]), // atoms/barn-cm
            "atom",
            "sum",
            None,
        )
        .unwrap();

        let stepper = ForwardEulerStepper;
        let rates: ReactionRates = HashMap::new(); // No reactions, decay only

        // Transmute for one half-life
        let result = stepper
            .step(
                &material,
                &chain,
                &rates,
                &HashMap::new(),
                ChainParts::default(),
                half_life,
            )
            .unwrap();

        // Co60 should be ~50% of initial
        let co60_final = *result.nuclides.get("Co60").unwrap_or(&0.0);
        let expected = 0.5 * 1.0e-3;
        let rel_error = (co60_final - expected).abs() / expected;
        assert!(
            rel_error < 1e-6,
            "Co60 relative error {rel_error} too large"
        );

        // Ni60 should be ~50% of initial Co60 (mass conservation)
        let ni60_final = *result.nuclides.get("Ni60").unwrap_or(&0.0);
        let expected_ni60 = 1.0e-3 - co60_final;
        let rel_error_ni = (ni60_final - expected_ni60).abs() / expected_ni60;
        assert!(
            rel_error_ni < 1e-6,
            "Ni60 relative error {rel_error_ni} too large"
        );
    }

    #[test]
    fn test_predictor_with_reaction() {
        let mut chain = HashMap::new();
        // Cr50 + n -> Cr51 (n,gamma)
        chain.insert(
            "Cr50".to_string(),
            ChainNuclide {
                name: "Cr50".to_string(),
                half_life: None, // stable
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("Cr51".to_string()),
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
        chain.insert(
            "Cr51".to_string(),
            ChainNuclide {
                name: "Cr51".to_string(),
                half_life: Some(2.3936e6), // ~27.7 days
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![ChainReaction {
                    kind: "ec/beta+".to_string(),
                    target: Some("V51".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                fission_yields: None,
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        chain.insert(
            "V51".to_string(),
            ChainNuclide {
                name: "V51".to_string(),
                half_life: None, // stable
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );

        let material = Material::new(
            HashMap::from([("Cr50".to_string(), 1.0e-3)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();

        let stepper = ForwardEulerStepper;
        let mut rates: ReactionRates = HashMap::new();
        let mut cr50_rates = HashMap::new();
        cr50_rates.insert("(n,gamma)".to_string(), 1.0e-6); // sigma*phi = 1e-6 /s
        rates.insert("Cr50".to_string(), cr50_rates);

        // Transmute for 1 day
        let dt = 86400.0;
        let result = stepper
            .step(
                &material,
                &chain,
                &rates,
                &HashMap::new(),
                ChainParts::default(),
                dt,
            )
            .unwrap();

        // Cr50 should decrease
        let cr50_final = *result.nuclides.get("Cr50").unwrap_or(&0.0);
        assert!(cr50_final < 1.0e-3, "Cr50 should decrease from activation");
        assert!(cr50_final > 0.0, "Cr50 shouldn't be zero after 1 day");

        // Cr51 should be produced
        let cr51_final = *result.nuclides.get("Cr51").unwrap_or(&0.0);
        assert!(cr51_final > 0.0, "Cr51 should be produced");
    }
}
