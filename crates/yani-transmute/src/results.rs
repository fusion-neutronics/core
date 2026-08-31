/// Transmutation results storage.
///
/// Stores material compositions at each timestep for all transmuted materials.
use std::collections::HashMap;
use yamc_materials::material::Material;
use yani::EdgeRates;

/// Results from a transmutation calculation.
///
/// Contains the evolution of all transmutable materials over the timesteps.
/// Index 0 is the initial composition, index i is after timestep[i-1].
#[derive(Debug, Clone)]
pub struct TransmutationResults {
    /// Material ID -> Vec of materials at each timestep.
    /// Index 0 is initial composition, index i is after timestep[i-1].
    pub materials: HashMap<u32, Vec<Material>>,

    /// Cumulative times [s] (index 0 = 0.0, index i = sum of first i timesteps)
    pub times: Vec<f64>,

    /// Timesteps used [s]
    pub timesteps: Vec<f64>,

    /// Source rates used [n/s]
    pub source_rates: Vec<f64>,

    /// Material ID -> per-step per-edge reaction rates, indexed as
    /// `timesteps`: entry i is the rates the solve drove step i with, so it is
    /// one shorter than that material's `materials` vector, whose index 0 is
    /// the initial composition. A decay-only step holds an empty map.
    ///
    /// The solve computes the rate of every production edge to build the
    /// burnup matrix, whose entries are sums over the edges into each product,
    /// and used to discard the edges themselves. They are the share of a
    /// product arriving down each route, which is the column a pathway
    /// analysis is written around. Issue #490.
    pub reaction_rates: HashMap<u32, Vec<EdgeRates>>,

    /// Material ID -> the ensemble of perturbed inventories, when
    /// nuclear-data uncertainty was asked for (issue #514).
    ///
    /// Empty on the default path, where nothing is sampled and nothing is
    /// allocated. Indexed by step the way `timesteps` is, so entry `i` is the
    /// inventory after step `i` and there is no entry for the initial
    /// composition, which carries no uncertainty.
    pub uncertainty: HashMap<u32, crate::uncertainty::Ensemble>,

    /// What was perturbed and what was not, when uncertainty was asked for.
    ///
    /// `None` on the default path. Present and possibly full of gaps
    /// otherwise: a nuclide with no covariance data must be reportable rather
    /// than showing up as a confidently small sigma.
    pub uncertainty_info: Option<crate::uncertainty::Info>,

    /// What the self-shielding did, or `None` when the run was not shielded.
    ///
    /// `None` and "shielded, but nothing moved" are different claims, and only
    /// this tells them apart.
    pub shielding_info: Option<crate::self_shielding::ShieldingInfo>,
}

impl TransmutationResults {
    /// Create a new empty results structure.
    pub fn new(timesteps: Vec<f64>, source_rates: Vec<f64>) -> Self {
        let mut times = Vec::with_capacity(timesteps.len() + 1);
        times.push(0.0);
        let mut cumulative = 0.0;
        for &dt in &timesteps {
            cumulative += dt;
            times.push(cumulative);
        }

        Self {
            materials: HashMap::new(),
            times,
            timesteps,
            source_rates,
            reaction_rates: HashMap::new(),
            uncertainty: HashMap::new(),
            uncertainty_info: None,
            shielding_info: None,
        }
    }

    /// Add initial material state.
    pub fn add_initial(&mut self, material_id: u32, material: Material) {
        self.materials
            .entry(material_id)
            .or_default()
            .push(material);
    }

    /// Add material state after a timestep.
    pub fn add_step(&mut self, material_id: u32, material: Material) {
        self.materials
            .entry(material_id)
            .or_default()
            .push(material);
    }

    /// Record the per-edge reaction rates a step was solved with.
    ///
    /// Called once per material per step, in step order, alongside
    /// [`Self::add_step`].
    pub fn add_step_rates(&mut self, material_id: u32, rates: EdgeRates) {
        self.reaction_rates
            .entry(material_id)
            .or_default()
            .push(rates);
    }

    /// Per-edge reaction rates for one material over one step.
    ///
    /// `step` indexes [`Self::timesteps`], so step 0 is the first schedule
    /// step. That is one less than the material index used by
    /// [`Self::get_material`], where 0 is the initial composition.
    ///
    /// Returns `None` when the material or the step is unknown, and
    /// `Some(&empty)` for a decay-only step.
    pub fn get_reaction_rates(&self, material_id: u32, step: usize) -> Option<&EdgeRates> {
        self.reaction_rates.get(&material_id)?.get(step)
    }

    /// Get material composition at a specific timestep.
    ///
    /// # Arguments
    /// * `material_id` - Material ID
    /// * `step` - Timestep index (0 = initial, 1 = after first step, etc.)
    pub fn get_material(&self, material_id: u32, step: usize) -> Option<&Material> {
        self.materials.get(&material_id)?.get(step)
    }

    /// Get the final composition for a material.
    pub fn get_final_material(&self, material_id: u32) -> Option<&Material> {
        let mats = self.materials.get(&material_id)?;
        mats.last()
    }

    /// Get final compositions for all materials.
    pub fn get_final_materials(&self) -> HashMap<u32, &Material> {
        self.materials
            .iter()
            .filter_map(|(id, mats)| mats.last().map(|m| (*id, m)))
            .collect()
    }

    /// One material per schedule step, without the initial composition.
    ///
    /// Indexed as [`Self::timesteps`], so entry `i` is the state at the end of
    /// step `i`. That is the shape the single-material entry point used to
    /// return on its own, and it pairs with [`Self::get_reaction_rates`], which
    /// takes the same index.
    ///
    /// Empty when the material is unknown or was never stepped.
    pub fn step_materials(&self, material_id: u32) -> &[Material] {
        match self.materials.get(&material_id) {
            // Index 0 is the initial composition, which is not a step.
            Some(mats) if !mats.is_empty() => &mats[1..],
            _ => &[],
        }
    }

    /// Get number of timesteps (not counting initial state).
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Get the nuclide density at a specific time for a specific material.
    ///
    /// # Arguments
    /// * `material_id` - Material ID
    /// * `nuclide` - Nuclide name (e.g., "U235")
    /// * `step` - Timestep index (0 = initial)
    pub fn get_nuclide_density(&self, material_id: u32, nuclide: &str, step: usize) -> Option<f64> {
        let mat = self.get_material(material_id, step)?;
        mat.nuclides.get(nuclide).copied()
    }

    /// Get the evolution of a specific nuclide over all timesteps.
    ///
    /// Returns a vector of densities, one per timestep (including initial).
    pub fn get_nuclide_evolution(&self, material_id: u32, nuclide: &str) -> Option<Vec<f64>> {
        let mats = self.materials.get(&material_id)?;
        Some(
            mats.iter()
                .map(|m| *m.nuclides.get(nuclide).unwrap_or(&0.0))
                .collect(),
        )
    }

    /// The nuclear-data standard deviation on a nuclide's density at `step`.
    ///
    /// `None` when uncertainty was not asked for. Indexed exactly like
    /// [`Self::get_nuclide_density`], so step 0 is the initial composition and
    /// answers zero: the starting inventory is an input, not a result, and
    /// perturbing cross sections does not move it.
    pub fn get_nuclide_uncertainty(
        &self,
        material_id: u32,
        nuclide: &str,
        step: usize,
    ) -> Option<f64> {
        let ensemble = self.uncertainty.get(&material_id)?;
        if step == 0 {
            return Some(0.0);
        }
        Some(
            ensemble
                .std_dev_evolution(nuclide)
                .get(step - 1)
                .copied()
                .unwrap_or(0.0),
        )
    }

    /// The standard deviation of a nuclide's density at every step.
    ///
    /// Parallel to [`Self::get_nuclide_evolution`], leading zero included, so
    /// the two can be zipped without an index correction.
    pub fn get_nuclide_uncertainty_evolution(
        &self,
        material_id: u32,
        nuclide: &str,
    ) -> Option<Vec<f64>> {
        let ensemble = self.uncertainty.get(&material_id)?;
        let mut out = vec![0.0];
        out.extend(ensemble.std_dev_evolution(nuclide));
        Some(out)
    }

    /// Every replica's inventory at `step`, for a derived quantity that must be
    /// evaluated per sample.
    ///
    /// Activity, decay heat and the decay-photon spectrum are all functions of
    /// a whole inventory. Evaluating one of them once per entry here and taking
    /// the spread of the results keeps the inter-nuclide correlations; taking
    /// the mean inventory and evaluating once throws them away, and summing
    /// per-nuclide sigmas in quadrature assumes an independence the resampling
    /// exists to avoid assuming.
    pub fn uncertainty_inventories(
        &self,
        material_id: u32,
        step: usize,
    ) -> Option<Vec<&HashMap<String, f64>>> {
        let ensemble = self.uncertainty.get(&material_id)?;
        if step == 0 {
            return Some(Vec::new());
        }
        Some(ensemble.inventories_at(step - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A material distinguishable only by its name: these tests are about
    /// where an entry lands in the results, not what is in it.
    fn material(name: &str) -> Material {
        let mut m = Material::new(
            HashMap::from([("Fe56".to_string(), 1.0)]),
            "atom",
            "g/cm3",
            Some(7.87),
        )
        .expect("build a material");
        m.name = Some(name.to_string());
        m
    }

    /// The initial composition sits at index 0 and is not a step, so the
    /// per-step view has to skip it or every step is off by one.
    #[test]
    fn step_materials_skips_the_initial_composition() {
        let mut results = TransmutationResults::new(vec![1.0, 2.0], vec![0.0, 0.0]);
        results.add_initial(7, material("initial"));
        results.add_step(7, material("after-one"));
        results.add_step(7, material("after-two"));

        let steps = results.step_materials(7);
        assert_eq!(steps.len(), results.timesteps.len());
        assert_eq!(steps[0].name.as_deref(), Some("after-one"));
        assert_eq!(steps[1].name.as_deref(), Some("after-two"));
    }

    #[test]
    fn step_materials_is_empty_for_an_unknown_material() {
        let results = TransmutationResults::new(vec![1.0], vec![0.0]);
        assert!(results.step_materials(7).is_empty());
    }

    /// A material that was recorded but never stepped has only its initial
    /// composition, which the slice must not report as a step.
    #[test]
    fn step_materials_is_empty_when_nothing_was_stepped() {
        let mut results = TransmutationResults::new(Vec::new(), Vec::new());
        results.add_initial(7, material("initial"));
        assert!(results.step_materials(7).is_empty());
    }

    /// `get_reaction_rates` indexes the schedule steps, one less than the
    /// material index, so the two must stay in step with each other.
    #[test]
    fn rates_and_step_materials_share_an_index() {
        let mut results = TransmutationResults::new(vec![1.0, 2.0], vec![1.0e14, 0.0]);
        results.add_initial(7, material("initial"));

        let mut irradiation = EdgeRates::new();
        irradiation.insert(
            "Fe56".to_string(),
            HashMap::from([(
                "(n,gamma)".to_string(),
                vec![(Some("Fe57".to_string()), 1.7e-9)],
            )]),
        );
        results.add_step_rates(7, irradiation);
        results.add_step(7, material("after-irradiation"));

        // A decay-only step drove no reactions, and records that rather than
        // shifting the indexing of the steps after it.
        results.add_step_rates(7, EdgeRates::new());
        results.add_step(7, material("after-cooling"));

        assert_eq!(
            results.step_materials(7)[0].name.as_deref(),
            Some("after-irradiation")
        );
        assert_eq!(
            results.get_reaction_rates(7, 0).expect("step 0")["Fe56"]["(n,gamma)"],
            vec![(Some("Fe57".to_string()), 1.7e-9)]
        );
        assert!(results.get_reaction_rates(7, 1).expect("step 1").is_empty());
        assert!(results.get_reaction_rates(7, 2).is_none());
    }
}
