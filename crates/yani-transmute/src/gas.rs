//! Hydrogen and helium gas production, in appm.
//!
//! A convenience over the inventory, which already holds the gas: the light
//! particles a reaction emits (`yani::light_particle_products`) and a decay
//! emits (`yani::decay_particle_products`) are added to it as H1, H2, H3, He3
//! and He4. What this module adds is the arithmetic the inventory leaves to
//! the caller, done the same way every time:
//!
//! - appm is atoms per million of the material's **initial** atoms, the
//!   convention for embrittlement and swelling, so the denominator does not
//!   move as the material transmutes;
//! - the gas present at the start (water, polymers, lithium compounds) is
//!   subtracted unless the caller asks for the total;
//! - a chain that cannot hold the gas is refused rather than reported as zero.
//!
//! The last one matters because the network follows an emitted particle only
//! when the chain has an entry for it. A chain without He4 loses every alpha
//! without a word, and the inventory then reads exactly like a material that
//! made no helium.

use std::collections::{BTreeMap, HashMap};

use crate::derived::{estimate, Estimate};
use crate::results::TransmutationResults;

/// The gas nuclides the stepper follows, in the order they are reported.
pub const GAS_NUCLIDES: [&str; 5] = ["H1", "H2", "H3", "He3", "He4"];

/// Every key of a gas production result: the five nuclides, then the
/// hydrogen (H1 + H2 + H3) and helium (He3 + He4) totals.
pub const GAS_KEYS: [&str; 7] = ["H1", "H2", "H3", "He3", "He4", "H", "He"];

/// Atoms per million.
const APPM: f64 = 1.0e6;

/// A material's initial atom densities and their sum, the denominator of appm.
type Basis = (HashMap<String, f64>, f64);

/// One inventory's gas in appm, keyed as [`GAS_KEYS`].
///
/// `initial` is subtracted when `produced` is set, so a nuclide the material
/// started with and then burned reads negative: the net change, not a
/// clipped one.
fn gas_appm(
    densities: &HashMap<String, f64>,
    initial: &HashMap<String, f64>,
    initial_atoms: f64,
    produced: bool,
) -> BTreeMap<String, f64> {
    let appm = |name: &str| {
        let now = densities.get(name).copied().unwrap_or(0.0);
        let start = if produced {
            initial.get(name).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        (now - start) / initial_atoms * APPM
    };
    let [h1, h2, h3, he3, he4] = GAS_NUCLIDES.map(appm);
    BTreeMap::from([
        ("H1".to_string(), h1),
        ("H2".to_string(), h2),
        ("H3".to_string(), h3),
        ("He3".to_string(), he3),
        ("He4".to_string(), he4),
        ("H".to_string(), h1 + h2 + h3),
        ("He".to_string(), he3 + he4),
    ])
}

impl TransmutationResults {
    /// Refuse when the chain the solve was driven with cannot hold the gas.
    ///
    /// Asked of the stored chain rather than the configured one, since the
    /// configuration can have been repointed since the solve.
    fn require_gas_in_chain(&self) -> Result<(), String> {
        let chain = self.chain.as_ref().ok_or(
            "gas production needs the chain the solve was driven with, \
             and these results carry none",
        )?;
        let missing: Vec<&str> = GAS_NUCLIDES
            .into_iter()
            .filter(|name| !chain.contains_key(*name))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        let missing = missing.join(", ");
        Err(format!(
            "gas production cannot be reported: the transmutation chain has no \
             entry for {missing}, so the solve did not follow {missing} emitted by \
             reactions and decays and the inventory reads as if none were made. \
             Use decay data whose chain holds all five gas nuclides"
        ))
    }

    /// The initial atom densities and their sum, or `None` for a material not
    /// in the results.
    fn gas_basis(&self, material_id: u32) -> Result<Option<Basis>, String> {
        let Some(initial) = self.get_material(material_id, 0) else {
            return Ok(None);
        };
        let initial = initial.get_atoms_per_barn_cm()?;
        // Summed in name order, so the denominator is the same to the last bit
        // on every run rather than following `HashMap` order.
        let mut names: Vec<&String> = initial.keys().collect();
        names.sort();
        let total: f64 = names.into_iter().map(|name| initial[name]).sum();
        if total <= 0.0 {
            return Err(format!(
                "material {material_id} starts with no atoms, so appm has no denominator"
            ));
        }
        Ok(Some((initial, total)))
    }

    /// Hydrogen and helium gas in appm at every time point, the initial one
    /// included, keyed as [`GAS_KEYS`].
    ///
    /// appm is atoms per million initial atoms of the material. With
    /// `produced` the gas the material started with is subtracted, so entry 0
    /// is zero and the rest is what the irradiation and the decays made; a
    /// nuclide consumed faster than it is made reads negative. Without it the
    /// result is the gas present, starting inventory included.
    ///
    /// Each list is parallel to `times`. `Ok(None)` when the material is not
    /// in the results. `Err` when the chain lacks any of [`GAS_NUCLIDES`],
    /// since the solve then dropped that gas and a zero would be a wrong
    /// answer rather than a measured one.
    pub fn gas_production(
        &self,
        material_id: u32,
        produced: bool,
    ) -> Result<Option<BTreeMap<String, Vec<f64>>>, String> {
        self.require_gas_in_chain()?;
        let Some((initial, total)) = self.gas_basis(material_id)? else {
            return Ok(None);
        };
        let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for material in &self.materials[&material_id] {
            let densities = material.get_atoms_per_barn_cm()?;
            for (key, value) in gas_appm(&densities, &initial, total, produced) {
                out.entry(key).or_default().push(value);
            }
        }
        Ok(Some(out))
    }

    /// Gas production in appm at `step`, keyed as [`GAS_KEYS`], each with the
    /// nuclear-data ensemble's spread.
    ///
    /// Evaluated on every replica's inventory against the one initial
    /// inventory, which is an input and the same in all of them. Each total is
    /// summed within a replica before the spread is taken, the same rule the
    /// activity follows. Step 0 has no spread: it is the starting inventory
    /// in every replica.
    ///
    /// `Ok(None)` when the run carried no uncertainty for this material. See
    /// [`Self::gas_production`] for `produced` and for when this refuses.
    pub fn gas_production_uncertainty(
        &self,
        material_id: u32,
        step: usize,
        produced: bool,
    ) -> Result<Option<BTreeMap<String, Estimate>>, String> {
        self.require_gas_in_chain()?;
        let Some(ensemble) = self.uncertainty.get(&material_id) else {
            return Ok(None);
        };
        let (initial, total) = self
            .gas_basis(material_id)?
            .ok_or_else(|| format!("no material {material_id} in the results"))?;
        let nominal = self
            .get_material(material_id, step)
            .ok_or_else(|| format!("no material {material_id} at step {step}"))?
            .get_atoms_per_barn_cm()?;
        let inventories: Vec<&HashMap<String, f64>> = if step == 0 {
            vec![&nominal; ensemble.replicas()]
        } else {
            self.uncertainty_inventories(material_id, step)
                .unwrap_or_default()
        };
        let per_replica: Vec<BTreeMap<String, f64>> = inventories
            .into_iter()
            .map(|inventory| gas_appm(inventory, &initial, total, produced))
            .collect();
        let mut values = Vec::with_capacity(per_replica.len());
        Ok(Some(
            gas_appm(&nominal, &initial, total, produced)
                .into_iter()
                .map(|(key, nominal)| {
                    values.clear();
                    values.extend(per_replica.iter().map(|r| r[&key]));
                    (key, estimate(nominal, &values))
                })
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uncertainty::Ensemble;
    use std::sync::Arc;
    use yamc_materials::material::Material;
    use yani::ChainNuclide;

    fn nuclide(name: &str) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life: None,
            half_life_uncertainty: None,
            decay_energy: 0.0,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
            reactions: vec![],
            decays: vec![],
            fission_yields: None,
            sources: vec![],
        }
    }

    fn chain(names: &[&str]) -> Arc<HashMap<String, ChainNuclide>> {
        Arc::new(names.iter().map(|n| (n.to_string(), nuclide(n))).collect())
    }

    fn densities(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(n, v)| (n.to_string(), *v)).collect()
    }

    fn material(pairs: &[(&str, f64)]) -> Material {
        Material::new(densities(pairs), "atom", "sum", None).expect("build a material")
    }

    const FULL: [&str; 6] = ["Fe56", "H1", "H2", "H3", "He3", "He4"];

    /// Material 1 through one irradiation step and one cooling step.
    fn results(steps: &[&[(&str, f64)]]) -> TransmutationResults {
        let mut results = TransmutationResults::new(vec![1.0; steps.len() - 1]);
        results.add_initial(1, material(steps[0]), vec![0.0; steps.len() - 1]);
        for step in &steps[1..] {
            results.add_step(1, material(step));
        }
        results.chain = Some(chain(&FULL));
        results
    }

    /// The arithmetic the issue writes out by hand, for one step: change in
    /// each gas density over the initial total, times a million.
    #[test]
    fn appm_is_the_change_over_the_initial_atoms() {
        let start: &[(&str, f64)] = &[("Fe56", 0.08), ("H1", 0.002)];
        let end: &[(&str, f64)] = &[
            ("Fe56", 0.0799),
            ("H1", 0.002 + 4.0e-5),
            ("H2", 1.0e-6),
            ("H3", 2.0e-7),
            ("He3", 3.0e-8),
            ("He4", 1.2e-5),
        ];
        let results = results(&[start, end]);
        let total = 0.08 + 0.002;
        let got = results.gas_production(1, true).unwrap().unwrap();
        let by_hand = |n: &str| {
            let a1 = densities(end).get(n).copied().unwrap_or(0.0);
            let a0 = densities(start).get(n).copied().unwrap_or(0.0);
            (a1 - a0) / total * 1e6
        };
        for n in GAS_NUCLIDES {
            assert_eq!(got[n], vec![0.0, by_hand(n)], "{n}");
        }
        assert_eq!(got["H"][1], by_hand("H1") + by_hand("H2") + by_hand("H3"));
        assert_eq!(got["He"][1], by_hand("He3") + by_hand("He4"));
        let keys: Vec<&str> = got.keys().map(String::as_str).collect();
        let mut expected = GAS_KEYS.to_vec();
        expected.sort();
        assert_eq!(keys, expected);
    }

    /// The total adds the starting gas back; produced does not have it.
    #[test]
    fn total_includes_the_starting_gas() {
        let start: &[(&str, f64)] = &[("O16", 0.03), ("H1", 0.06)];
        let end: &[(&str, f64)] = &[("O16", 0.0299), ("H1", 0.06), ("He4", 1.0e-4)];
        let results = results(&[start, end]);
        let produced = results.gas_production(1, true).unwrap().unwrap();
        let total = results.gas_production(1, false).unwrap().unwrap();
        assert_eq!(produced["H1"], vec![0.0, 0.0]);
        let h1_start = 0.06 / 0.09 * 1e6;
        assert_eq!(total["H1"], vec![h1_start, h1_start]);
        assert_eq!(produced["He4"], total["He4"]);
    }

    /// Water gives the same produced He4 as the gas-free material with the
    /// same oxygen, since the starting hydrogen is subtracted and the
    /// denominator is each material's own initial atoms.
    #[test]
    fn starting_hydrogen_does_not_read_as_produced() {
        let oxide = results(&[&[("O16", 0.03)], &[("O16", 0.0299), ("He4", 1.0e-4)]]);
        let water = results(&[
            &[("O16", 0.03), ("H1", 0.06)],
            &[("O16", 0.0299), ("H1", 0.06), ("He4", 1.0e-4)],
        ]);
        let oxide = oxide.gas_production(1, true).unwrap().unwrap();
        let water = water.gas_production(1, true).unwrap().unwrap();
        // Per initial atom the water is diluted by its hydrogen, so per
        // oxygen atom the two agree.
        let per_oxygen = |appm: f64, atoms: f64| appm * atoms / 0.03;
        assert!(
            (per_oxygen(water["He4"][1], 0.09) - per_oxygen(oxide["He4"][1], 0.03)).abs() < 1e-9
        );
        assert_eq!(water["H"], vec![0.0, 0.0]);
    }

    /// One list entry per time point, the cooling step included, and gas
    /// made by decay during cooling shows up there.
    #[test]
    fn every_step_of_a_schedule_with_cooling() {
        let results = results(&[
            &[("Fe56", 1.0)],
            &[("Fe56", 0.99), ("H3", 1.0e-6)],
            &[("Fe56", 0.99), ("H3", 0.9e-6), ("He3", 0.1e-6)],
        ]);
        let got = results.gas_production(1, true).unwrap().unwrap();
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9);
        assert!(close(&got["H3"], &[0.0, 1.0, 0.9]), "{:?}", got["H3"]);
        assert!(close(&got["He3"], &[0.0, 0.0, 0.1]), "{:?}", got["He3"]);
        for (key, series) in &got {
            assert_eq!(series.len(), results.times.len(), "{key}");
        }
    }

    /// A chain without He4 dropped every alpha: refuse, naming it.
    #[test]
    fn a_chain_without_a_gas_nuclide_refuses() {
        let mut results = results(&[&[("Fe56", 1.0)], &[("Fe56", 0.99)]]);
        results.chain = Some(chain(&["Fe56", "H1", "H2", "H3", "He3"]));
        let err = results.gas_production(1, true).unwrap_err();
        assert!(err.contains("He4"), "{err}");
        assert!(!err.contains("H1,"), "{err}");

        results.chain = Some(chain(&["Fe56", "H1", "H3"]));
        let err = results.gas_production_uncertainty(1, 1, true).unwrap_err();
        assert!(err.contains("H2, He3, He4"), "{err}");

        results.chain = None;
        assert!(results.gas_production(1, true).is_err());
    }

    #[test]
    fn an_unknown_material_is_none() {
        let results = results(&[&[("Fe56", 1.0)], &[("Fe56", 0.99)]]);
        assert_eq!(results.gas_production(2, true).unwrap(), None);
    }

    #[test]
    fn uncertainty_is_none_without_an_ensemble() {
        let results = results(&[&[("Fe56", 1.0)], &[("Fe56", 0.99)]]);
        assert_eq!(
            results.gas_production_uncertainty(1, 1, true).unwrap(),
            None
        );
    }

    /// The spread is taken over the replicas' appm, totals summed within each.
    #[test]
    fn uncertainty_is_the_spread_over_the_replicas() {
        let mut results = results(&[
            &[("Fe56", 1.0), ("H1", 1.0e-6)],
            &[("Fe56", 0.99), ("H1", 3.0e-6), ("He4", 2.0e-6)],
        ]);
        let mut ensemble = Ensemble::new(1);
        let replicas = [
            (2.0e-6, 3.0e-6),
            (4.0e-6, 1.0e-6),
            (3.0e-6, 2.0e-6),
            (3.0e-6, 2.0e-6),
        ];
        for (h1, he4) in replicas {
            ensemble.push(vec![densities(&[("Fe56", 0.99), ("H1", h1), ("He4", he4)])]);
        }
        results.uncertainty.insert(1, ensemble);

        let got = results
            .gas_production_uncertainty(1, 1, true)
            .unwrap()
            .unwrap();
        let total = 1.0 + 1.0e-6;
        let h1: Vec<f64> = replicas
            .iter()
            .map(|r| (r.0 - 1.0e-6) / total * 1e6)
            .collect();
        let he4: Vec<f64> = replicas.iter().map(|r| r.1 / total * 1e6).collect();
        assert_eq!(got["H1"], estimate((3.0e-6 - 1.0e-6) / total * 1e6, &h1));
        assert_eq!(got["He4"], estimate(2.0e-6 / total * 1e6, &he4));
        assert_eq!(got["H1"].replicas, 4);
        assert!(got["H1"].std_dev.unwrap() > 0.0);
        assert!(got["H1"].std_dev_standard_error.is_some());
        // H1 and He4 trade one-for-one in every replica here, so the sum H + He
        // does not move: a total is summed inside each replica, not in
        // quadrature.
        let h_plus_he: Vec<f64> = h1.iter().zip(&he4).map(|(a, b)| a + b).collect();
        assert!(h_plus_he.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9));
        assert_eq!(got["He"].nominal, got["He4"].nominal);

        // Step 0 is the starting inventory in every replica: a measured zero.
        let start = results
            .gas_production_uncertainty(1, 0, true)
            .unwrap()
            .unwrap();
        assert_eq!(start["H1"].nominal, 0.0);
        assert_eq!(start["H1"].std_dev, Some(0.0));
    }
}
