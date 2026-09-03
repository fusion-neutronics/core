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

    /// The chain the solve was driven with, for deriving routes afterwards.
    ///
    /// An `Arc` clone of the one already held, so keeping it costs a refcount
    /// rather than a copy. It is here because a route is a statement about the
    /// topology AND about the rates, and without it every caller that wants
    /// [`Self::get_production_routes`] has to load the chain a second time and
    /// hope it is the same one the solve used. That is not a hypothetical: a
    /// chain is loaded from a path in a global, and the path can have been
    /// repointed between the solve and the question.
    ///
    /// `None` on results built by hand, which is what the tests do.
    pub chain: Option<std::sync::Arc<HashMap<String, yani::ChainNuclide>>>,

    /// What the multigroup collapse was driven with, for re-deriving an
    /// energy-resolved view of a rate afterwards.
    ///
    /// `None` when there was no multigroup collapse: a transport-coupled solve
    /// scores its rates at the collision energy and keeps no group structure to
    /// resolve them onto, and results built by hand have no spectrum at all.
    pub collapse: Option<CollapseInputs>,
}

/// The spectra a solve collapsed against, and how its steps used them.
///
/// Small on purpose. Two vectors per distinct spectrum is a few KB against the
/// tens of MB the full per-group rate breakdown would be, and it is enough to
/// re-derive any single channel's breakdown on demand
/// (see [`crate::multigroup::reaction_rate_spectrum`]).
#[derive(Debug, Clone)]
pub struct CollapseInputs {
    /// One entry per distinct spectrum: group boundaries [eV], ascending and
    /// one longer than the flux, and the flux shape the collapse weighted with.
    ///
    /// The shape is normalized, so the magnitude of a step's flux is its entry
    /// in [`TransmutationResults::source_rates`], exactly as the solve applies
    /// it.
    pub spectra: Vec<(Vec<f64>, Vec<f64>)>,

    /// Which spectrum each schedule step used, indexed as
    /// [`TransmutationResults::timesteps`]. `None` for a decay-only step, which
    /// drives no reactions.
    pub step_spectrum: Vec<Option<usize>>,

    /// The self-shielding request the collapse ran under, when there was one,
    /// so a re-derived breakdown is weighted the way the solve weighted it.
    pub shielding: Option<crate::self_shielding::Shielding>,
}

/// One channel's reaction rate, resolved onto the spectrum's own groups.
#[derive(Debug, Clone, PartialEq)]
pub struct RateSpectrum {
    /// Group boundaries [eV], ascending, one longer than `rates`.
    pub boundaries: Vec<f64>,
    /// Each group's contribution to the rate [1/s], summing to the rate
    /// [`TransmutationResults::get_reaction_rates`] reports for the channel.
    pub rates: Vec<f64>,
}

/// Flux-weighted isomeric branching: `parent -> kind -> [(target, fraction)]`.
///
/// Named for the same reason [`yani::EdgeRates`] is: it is three levels deep
/// and appears in a signature, a return and a binding, and spelling it out
/// three times is three chances to spell it differently.
pub type IsomericBranching = HashMap<String, HashMap<String, Vec<(String, f64)>>>;

/// One way a product is made: the steps, and the share of it arriving this way.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductionRoute {
    /// `(parent, kind, target)` per step, the neutron reactions first and then
    /// the decays that carry the product on. `kind` is the chain's own
    /// spelling, so `"(n,2n)"` for a reaction and `"it"` or `"beta-"` for a
    /// decay.
    pub steps: Vec<(String, String, String)>,
    /// Share of this product's production over the step that arrived down this
    /// route, in `[0, 1]` and summing to one across the routes returned.
    pub share: f64,
    /// Atoms of the product this route made per barn-cm over the step, before
    /// normalising. Kept because a share of 100% of almost nothing and a share
    /// of 100% of the whole inventory read the same otherwise.
    pub production: f64,
}

impl ProductionRoute {
    /// The route as the published pathway tables write it, e.g.
    /// `"W186(n,2n)W185_m1(IT)W185"`.
    ///
    /// Decay kinds are upper-cased in that convention and reaction kinds are
    /// not, which is the only reason this is not `format!` at the call site.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for (i, (parent, kind, target)) in self.steps.iter().enumerate() {
            if i == 0 {
                out.push_str(parent);
            }
            if kind.starts_with('(') {
                out.push_str(kind);
            } else {
                out.push('(');
                out.push_str(&kind.to_uppercase());
                out.push(')');
            }
            out.push_str(target);
        }
        out
    }
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
            chain: None,
            collapse: None,
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

    /// One channel's reaction rate over one step, resolved onto the groups of
    /// the spectrum that drove it.
    ///
    /// [`Self::get_reaction_rates`] answers with one number per edge, already
    /// collapsed. That number cannot say which part of the spectrum made it,
    /// and on a channel whose cross section spans decades the two readings are
    /// different physics: an effective `W186(n,gamma)` of 57 mb against a
    /// spectrum 89% of which sits in 12-16 MeV and 0.7% below 100 keV is
    /// either a fast-capture rate or a resonance-region rate, and only the
    /// breakdown says which (yani#27). A disagreement can then be attributed
    /// to resonance processing rather than guessed at, and a covariance grid
    /// that stops short of the spectrum can be checked against where the rate
    /// actually is.
    ///
    /// The entries sum to the collapsed rate for the same `(nuclide, kind)`,
    /// up to floating-point rounding, because both come from the same walk of
    /// the same cross sections, self-shielding included.
    ///
    /// Nothing is stored for this: the breakdown is re-derived from the
    /// spectrum and the step's initial composition when asked for, which is one
    /// reaction over the group structure. Keeping it for every channel would be
    /// tens of MB on a 709-group structure, and the default path should not pay
    /// that.
    ///
    /// `step` indexes [`Self::timesteps`], as [`Self::get_reaction_rates`]
    /// does.
    ///
    /// `None` when the material, the step, the nuclide or the channel is
    /// unknown; when the step drove no flux, whether by being decay-only or by
    /// carrying a zero rate; and on a transport-coupled solve, which scores its
    /// rates at the collision energy and keeps no group structure to resolve
    /// them onto.
    pub fn get_reaction_rate_spectrum(
        &self,
        material_id: u32,
        nuclide: &str,
        kind: &str,
        step: usize,
    ) -> Option<RateSpectrum> {
        let inputs = self.collapse.as_ref()?;
        let spectrum = (*inputs.step_spectrum.get(step)?)?;
        let (boundaries, flux) = inputs.spectra.get(spectrum)?;
        let source_rate = self.source_rates.get(step).copied()?;
        // The collapse is done once from the initial composition and scaled per
        // step, so this is the material it read, and `source_rate` is the
        // scaling the step applied.
        let material = self.get_material(material_id, 0)?;
        let rates = crate::multigroup::reaction_rate_spectrum(
            material,
            flux,
            boundaries,
            source_rate,
            inputs.shielding.as_ref(),
            nuclide,
            kind,
        )?;
        Some(RateSpectrum {
            boundaries: boundaries.clone(),
            rates,
        })
    }

    /// Flux-weighted isomeric branching for one material over one step:
    /// `parent -> kind -> [(target, fraction)]`, fractions summing to one.
    ///
    /// Which state a reaction leaves its product in is energy dependent, so the
    /// one number describing a given spectrum is the branching collapsed
    /// against it, and that number exists only inside a solve. The chain file
    /// carries the unweighted ratios, and where the overlay supplies the split
    /// it carries a placeholder: on TENDL-2025 the dominant tungsten channel is
    /// `W186 (n,2n) -> W185 1.000000` and `W186 (n,2n) -> W185_m1 0.000000` in
    /// the file, and the overlay replaces both at solve time with roughly the
    /// 54/46 that spectrum actually gives. Reading the file answers a different
    /// question from reading this, and on a foil whose decay heat comes from an
    /// isomer the difference is the whole answer.
    ///
    /// Only channels landing in more than one final state are returned: a
    /// channel with a single product has no branching to report, and listing it
    /// at 1.0 buries the ones that do. [`Self::get_reaction_rates`] has the
    /// unnormalised edges if the rest is wanted.
    ///
    /// This is what says whether a disagreement belongs to a cross section or
    /// to a branching ratio, which are different data and different fixes.
    ///
    /// `step` indexes [`Self::timesteps`], exactly as
    /// [`Self::get_reaction_rates`] does. `None` when the material or the step
    /// is unknown, and empty for a decay-only step, which splits nothing.
    pub fn get_isomeric_branching(
        &self,
        material_id: u32,
        step: usize,
    ) -> Option<IsomericBranching> {
        let edges = self.get_reaction_rates(material_id, step)?;
        let mut out: IsomericBranching = HashMap::new();
        for (parent, kinds) in edges {
            for (kind, targets) in kinds {
                // A channel naming no single product (fission) has no split to
                // report: its products come from the yields, not from an edge.
                let named: Vec<(&String, f64)> = targets
                    .iter()
                    .filter_map(|(t, r)| t.as_ref().map(|t| (t, *r)))
                    .collect();
                if named.len() < 2 {
                    continue;
                }
                let total: f64 = named.iter().map(|(_, r)| r).sum();
                if total <= 0.0 {
                    continue;
                }
                let mut split: Vec<(String, f64)> = named
                    .into_iter()
                    .map(|(t, r)| (t.clone(), r / total))
                    .collect();
                // Largest share first, so the state the channel mostly makes is
                // read first; ties by name, so the order is stable run to run.
                split.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                out.entry(parent.clone())
                    .or_default()
                    .insert(kind.clone(), split);
            }
        }
        Some(out)
    }

    /// Every way `product` was made over one step, weighted by how much of it
    /// arrived down each.
    ///
    /// Enumerating routes is easy and weighting them is not: the chain will
    /// happily say that W187 is made by `Os190(n,a)` and `Ir192(n,npa)` as
    /// readily as by `W186(n,gamma)`, and nothing in a tungsten foil is osmium.
    /// So the walk starts from the nuclides the material actually began with,
    /// and each route is weighted by what its own reactions drove.
    ///
    /// A route is `reaction_depth` neutron reactions and then any number of
    /// decays, up to `decay_depth`. Decays are followed regardless of depth
    /// budget because a decay is not a fluence-dependent step: it moves what a
    /// reaction already made rather than making more of it.
    ///
    /// The weight of a route is the atoms of the nuclide it starts from, times
    /// each reaction step's per-atom production over the step, times the
    /// branching of each decay it passes through. Reaction steps carry the step
    /// duration, so a two-reaction route is in the same units as a one-reaction
    /// route and is smaller by roughly a factor of the fluence, which is the
    /// honest answer for irradiations short enough that the products barely
    /// burn. Decay branchings are included because a route through a 1% branch
    /// delivers 1% of what the reaction made.
    ///
    /// Routes are returned largest share first. `None` when the material, the
    /// step or the chain is unknown; empty when nothing in this material makes
    /// `product` at all, which is a real answer and not a failure.
    ///
    /// `step` indexes [`Self::timesteps`], as [`Self::get_reaction_rates`] does.
    pub fn get_production_routes(
        &self,
        material_id: u32,
        product: &str,
        step: usize,
        reaction_depth: usize,
        decay_depth: usize,
    ) -> Option<Vec<ProductionRoute>> {
        let chain = self.chain.as_ref()?;
        let edges = self.get_reaction_rates(material_id, step)?;
        let dt = self.timesteps.get(step).copied().unwrap_or(0.0);
        let densities = self
            .get_material(material_id, 0)?
            .get_atoms_per_barn_cm()
            .unwrap_or_default();

        // The rate of one production edge, per atom of its parent, over this
        // step. `None` when the solve drove no such edge, which prunes the walk
        // to what actually happened rather than to what the chain permits.
        let edge_rate = |parent: &str, kind: &str, target: &str| -> Option<f64> {
            edges
                .get(parent)?
                .get(kind)?
                .iter()
                .find(|(t, _)| t.as_deref() == Some(target))
                .map(|(_, r)| *r)
        };

        /// One partly-walked route: where it has got to, how it got there, what
        /// it carries, and how much of its reaction budget it has spent.
        struct Partial {
            here: String,
            steps: Vec<(String, String, String)>,
            weight: f64,
            reactions_used: usize,
        }

        let mut found: Vec<ProductionRoute> = Vec::new();
        let mut stack: Vec<Partial> = densities
            .iter()
            .filter(|(_, &n)| n > 0.0)
            .map(|(name, &n)| Partial {
                here: name.clone(),
                steps: Vec::new(),
                weight: n,
                reactions_used: 0,
            })
            .collect();

        while let Some(Partial {
            here,
            steps: path,
            weight,
            reactions_used: used,
        }) = stack.pop()
        {
            if !path.is_empty() && here == product {
                found.push(ProductionRoute {
                    steps: path.clone(),
                    share: 0.0,
                    production: weight,
                });
                // Not returned early: a route can pass through its own product
                // on the way to making more of it, and stopping here would drop
                // the longer one.
            }
            if path.len() >= reaction_depth + decay_depth {
                continue;
            }
            let Some(node) = chain.get(&here) else {
                continue;
            };
            if used < reaction_depth {
                for rx in &node.reactions {
                    let Some(target) = rx.target.as_deref() else {
                        continue;
                    };
                    let Some(rate) = edge_rate(&here, &rx.kind, target) else {
                        continue;
                    };
                    if rate <= 0.0 {
                        continue;
                    }
                    let mut next = path.clone();
                    next.push((here.clone(), rx.kind.clone(), target.to_string()));
                    stack.push(Partial {
                        here: target.to_string(),
                        steps: next,
                        weight: weight * rate * dt,
                        reactions_used: used + 1,
                    });
                }
            }
            // Decays are followed only after something has been made: a route
            // is a production route, and a nuclide the material started with
            // decaying is not production.
            if !path.is_empty() {
                for dk in &node.decays {
                    let Some(target) = dk.target.as_deref() else {
                        continue;
                    };
                    if dk.branching <= 0.0 {
                        continue;
                    }
                    let mut next = path.clone();
                    next.push((here.clone(), dk.kind.clone(), target.to_string()));
                    stack.push(Partial {
                        here: target.to_string(),
                        steps: next,
                        weight: weight * dk.branching,
                        reactions_used: used,
                    });
                }
            }
        }

        let total: f64 = found.iter().map(|r| r.production).sum();
        if total > 0.0 {
            for r in &mut found {
                r.share = r.production / total;
            }
        }
        found.sort_by(|a, b| {
            b.share
                .partial_cmp(&a.share)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.steps.len().cmp(&b.steps.len()))
                .then_with(|| a.text().cmp(&b.text()))
        });
        Some(found)
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

    /// W185 is made two ways: straight off the (n,2n), and via the isomer that
    /// then decays to it. Both must be found, and weighted by what each drove.
    #[test]
    fn production_routes_find_both_the_direct_and_the_isomeric_path() {
        use std::sync::Arc;
        let nuclide =
            |name: &str, reactions: Vec<yani::ChainReaction>, decays: Vec<yani::ChainReaction>| {
                yani::ChainNuclide {
                    name: name.to_string(),
                    half_life: None,
                    half_life_uncertainty: None,
                    decay_energy: 0.0,
                    decay_energy_uncertainty: None,
                    reactions,
                    decays,
                    fission_yields: None,
                    sources: Vec::new(),
                }
            };
        let rx = |kind: &str, target: &str| yani::ChainReaction {
            kind: kind.to_string(),
            target: Some(target.to_string()),
            branching: 1.0,
            q_value: Some(0.0),
        };

        let chain = Arc::new(HashMap::from([
            (
                "W186".to_string(),
                nuclide(
                    "W186",
                    vec![rx("(n,2n)", "W185"), rx("(n,2n)", "W185_m1")],
                    vec![],
                ),
            ),
            (
                "W185_m1".to_string(),
                nuclide("W185_m1", vec![], vec![rx("it", "W185")]),
            ),
            ("W185".to_string(), nuclide("W185", vec![], vec![])),
        ]));

        let mut m = Material::new(
            HashMap::from([("W186".to_string(), 1.0)]),
            "atom",
            "g/cm3",
            Some(19.3),
        )
        .expect("tungsten");
        m.name = Some("foil".to_string());

        let mut results = TransmutationResults::new(vec![300.0], vec![1.0e10]);
        results.add_initial(7, m);
        let mut irradiation = EdgeRates::new();
        irradiation.insert(
            "W186".to_string(),
            HashMap::from([(
                "(n,2n)".to_string(),
                vec![
                    (Some("W185".to_string()), 1.0e-12),
                    (Some("W185_m1".to_string()), 3.0e-12),
                ],
            )]),
        );
        results.add_step_rates(7, irradiation);
        results.add_step(7, material("after"));
        results.chain = Some(chain);

        let routes = results
            .get_production_routes(7, "W185", 0, 1, 3)
            .expect("routes");
        let text: Vec<String> = routes.iter().map(|r| r.text()).collect();
        assert_eq!(
            text,
            vec!["W186(n,2n)W185_m1(IT)W185", "W186(n,2n)W185"],
            "largest share first, and the decay rendered as the tables write it"
        );
        // 3:1 in the rates, and the IT branch carries all of its share on.
        assert!((routes[0].share - 0.75).abs() < 1e-12);
        assert!((routes[1].share - 0.25).abs() < 1e-12);
    }

    /// A product nothing in this material makes has no routes, which is an
    /// answer rather than a failure.
    #[test]
    fn production_routes_are_empty_for_an_unreachable_product() {
        let mut results = TransmutationResults::new(vec![300.0], vec![1.0e10]);
        results.add_initial(7, material("initial"));
        results.add_step_rates(7, EdgeRates::new());
        results.add_step(7, material("after"));
        results.chain = Some(std::sync::Arc::new(HashMap::new()));
        assert!(results
            .get_production_routes(7, "Pu239", 0, 1, 3)
            .expect("a known material and step")
            .is_empty());
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

    /// A channel that splits is reported as fractions; one that does not is
    /// left out entirely rather than reported at 1.0.
    #[test]
    fn isomeric_branching_normalises_splits_and_omits_single_product_channels() {
        let mut results = TransmutationResults::new(vec![1.0], vec![1.0e14]);
        results.add_initial(7, material("initial"));

        let mut irradiation = EdgeRates::new();
        irradiation.insert(
            "W186".to_string(),
            HashMap::from([
                // Splits: 3 to the ground state and 1 to the isomer.
                (
                    "(n,2n)".to_string(),
                    vec![
                        (Some("W185".to_string()), 3.0e-12),
                        (Some("W185_m1".to_string()), 1.0e-12),
                    ],
                ),
                // Does not split, so it has no branching to report.
                (
                    "(n,gamma)".to_string(),
                    vec![(Some("W187".to_string()), 5.0e-13)],
                ),
            ]),
        );
        results.add_step_rates(7, irradiation);
        results.add_step(7, material("after"));

        let split = results.get_isomeric_branching(7, 0).expect("step 0");
        let w186 = &split["W186"];
        assert!(
            !w186.contains_key("(n,gamma)"),
            "a channel with one product has no branching to report"
        );
        // Largest share first, so the state the channel mostly makes leads.
        assert_eq!(w186["(n,2n)"][0].0, "W185");
        assert!((w186["(n,2n)"][0].1 - 0.75).abs() < 1e-12);
        assert_eq!(w186["(n,2n)"][1].0, "W185_m1");
        assert!((w186["(n,2n)"][1].1 - 0.25).abs() < 1e-12);
    }

    /// A decay-only step splits nothing, and says so rather than being absent.
    #[test]
    fn isomeric_branching_is_empty_for_a_decay_only_step() {
        let mut results = TransmutationResults::new(vec![1.0], vec![0.0]);
        results.add_initial(7, material("initial"));
        results.add_step_rates(7, EdgeRates::new());
        results.add_step(7, material("after"));
        assert!(results
            .get_isomeric_branching(7, 0)
            .expect("step 0")
            .is_empty());
        assert!(results.get_isomeric_branching(7, 9).is_none());
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

    /// A rate that came from somewhere other than a multigroup collapse has no
    /// group structure to resolve onto, and must say so rather than answer with
    /// a spectrum it invented.
    #[test]
    fn a_rate_from_no_spectrum_has_no_spectrum() {
        let mut results = TransmutationResults::new(vec![1.0], vec![1.0e14]);
        results.add_initial(7, material("initial"));
        results.add_step_rates(7, EdgeRates::new());
        results.add_step(7, material("after"));

        // This is the transport-coupled case: the rates are scored at the
        // collision energy and no spectrum is kept.
        assert!(results.collapse.is_none());
        assert!(results
            .get_reaction_rate_spectrum(7, "Fe56", "(n,gamma)", 0)
            .is_none());
    }

    /// A decay-only step drives no reactions, so there is nothing to resolve,
    /// and the step after it must still be answerable.
    #[test]
    fn a_decay_only_step_has_no_rate_spectrum() {
        let mut results = TransmutationResults::new(vec![1.0, 2.0], vec![0.0, 1.0e14]);
        results.add_initial(7, material("initial"));
        results.collapse = Some(CollapseInputs {
            spectra: vec![(vec![1.0e-5, 1.0e5, 2.0e7], vec![1.0, 1.0])],
            step_spectrum: vec![None, Some(0)],
            shielding: None,
        });

        assert!(results
            .get_reaction_rate_spectrum(7, "Fe56", "(n,gamma)", 0)
            .is_none());
        // The irradiation step is reachable, and stops on the material's data
        // rather than on the indexing: this one was built by hand and carries
        // no cross sections.
        assert!(results
            .get_reaction_rate_spectrum(7, "Fe56", "(n,gamma)", 1)
            .is_none());
        assert!(results
            .get_reaction_rate_spectrum(7, "Fe56", "(n,gamma)", 2)
            .is_none());
    }
}
