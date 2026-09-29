//! Decay-inventory post-processing.
//!
//! Computes the radioactive activity and decay heat of a material inventory
//! from its atom densities and a transmutation chain that supplies half-lives
//! and mean decay energies. Activity is `A_i = N_i · λ_i` (Bq); decay heat is
//! the standard `P = Σ_i A_i · Ē_i` (W), where `Ē_i` is the mean recoverable
//! decay energy per decay.

use std::collections::HashMap;

use yani::{ChainNuclide, Continuum, DecaySourceDistribution, Interpolation, UnreadableContinuum};

/// Electron-volt to joule conversion (2019 SI redefinition).
const EV_TO_J: f64 = 1.602_176_634e-19;

/// Barns per cm²: converts atom density in atoms/(barn·cm) to atoms/cm³.
const BARN_PER_CM_SQ: f64 = 1.0e24;

/// Activity of each nuclide in `atom_densities` [Bq].
///
/// `atom_densities` maps nuclide name -> atoms/(barn·cm); `volume` is in cm³.
/// Total atoms are `density · 1e24 · volume` and the activity is `N · ln2 /
/// t_half`. Nuclides absent from `chain` or stable (no or non-positive
/// half-life) contribute nothing and are omitted from the returned map.
pub fn activity_by_nuclide(
    atom_densities: &HashMap<String, f64>,
    volume: f64,
    chain: &HashMap<String, ChainNuclide>,
) -> HashMap<String, f64> {
    let mut activities = HashMap::new();
    for (nuclide, &density) in atom_densities {
        let Some(chain_nuclide) = chain.get(nuclide) else {
            continue;
        };
        let Some(half_life) = chain_nuclide.half_life else {
            continue;
        };
        if half_life <= 0.0 {
            continue;
        }
        let atoms = density * BARN_PER_CM_SQ * volume;
        let activity = atoms * std::f64::consts::LN_2 / half_life;
        if activity > 0.0 {
            activities.insert(nuclide.clone(), activity);
        }
    }
    activities
}

/// Sum a per-nuclide map in nuclide-name order.
///
/// `HashMap` iteration order is not stable between two maps or between two
/// runs, so a total summed in it moves in the last bit. In a single answer that
/// is invisible; over an ensemble it is not, because two replicas holding the
/// identical inventory then disagree and manufacture a spread where there is
/// none (issue #558). Same reason [`decay_photon_lines`] walks its nuclides
/// sorted.
pub fn total(by_nuclide: &HashMap<String, f64>) -> f64 {
    let mut names: Vec<&String> = by_nuclide.keys().collect();
    names.sort();
    names.into_iter().map(|name| by_nuclide[name]).sum()
}

/// Total activity of a material inventory [Bq]. See [`activity_by_nuclide`].
pub fn activity_total(
    atom_densities: &HashMap<String, f64>,
    volume: f64,
    chain: &HashMap<String, ChainNuclide>,
) -> f64 {
    total(&activity_by_nuclide(atom_densities, volume, chain))
}

/// Decay heat contribution of each nuclide in `atom_densities` [W].
///
/// Multiplies each nuclide's activity (see [`activity_by_nuclide`]) by its mean
/// decay energy and the eV→J factor. Nuclides with non-positive decay energy
/// contribute nothing and are omitted from the returned map.
pub fn decay_heat_by_nuclide(
    atom_densities: &HashMap<String, f64>,
    volume: f64,
    chain: &HashMap<String, ChainNuclide>,
) -> HashMap<String, f64> {
    let mut heats = HashMap::new();
    for (nuclide, activity) in activity_by_nuclide(atom_densities, volume, chain) {
        let decay_energy = chain.get(&nuclide).map(|cn| cn.decay_energy).unwrap_or(0.0);
        let heat = activity * decay_energy * EV_TO_J;
        if heat > 0.0 {
            heats.insert(nuclide, heat);
        }
    }
    heats
}

/// One component of each nuclide's decay heat [W]: its activity times that
/// component's mean energy (`component` indexes
/// [`yani::DECAY_ENERGY_COMPONENTS`]: beta, gamma, alpha).
///
/// The components answer questions the total cannot: the gamma heat is what
/// leaves a thin component and deposits in its neighbours, the beta and alpha
/// heat stays where it is made, and each has its own uncertainty.
///
/// `Err` names every nuclide that makes decay heat but whose data carries no
/// split, which a file written before the split does for every nuclide.
/// Reporting a component heat without them would understate it by an unknown
/// amount, so there is no partial answer.
pub fn decay_heat_component_by_nuclide(
    atom_densities: &HashMap<String, f64>,
    volume: f64,
    chain: &HashMap<String, ChainNuclide>,
    component: usize,
) -> Result<HashMap<String, f64>, Vec<String>> {
    let mut heats = HashMap::new();
    let mut missing = Vec::new();
    for (nuclide, activity) in activity_by_nuclide(atom_densities, volume, chain) {
        let Some(cn) = chain.get(&nuclide) else {
            continue;
        };
        if cn.decay_energy <= 0.0 {
            continue;
        }
        let Some(part) = cn.decay_energy_components.get(component).copied().flatten() else {
            missing.push(nuclide);
            continue;
        };
        let heat = activity * part.energy * EV_TO_J;
        if heat > 0.0 {
            heats.insert(nuclide, heat);
        }
    }
    if missing.is_empty() {
        Ok(heats)
    } else {
        missing.sort();
        Err(missing)
    }
}

/// Total decay heat of a material inventory [W]. See [`decay_heat_by_nuclide`].
pub fn decay_heat_total(
    atom_densities: &HashMap<String, f64>,
    volume: f64,
    chain: &HashMap<String, ChainNuclide>,
) -> f64 {
    total(&decay_heat_by_nuclide(atom_densities, volume, chain))
}

/// The discrete decay photon lines an inventory emits: `(energy [eV], photons
/// per second)`, ascending in energy, coincident energies summed.
///
/// Lines only. A continuum is a density per eV rather than a set of rates, so
/// it has no place in this list, and summing its tabulated values as lines is
/// the defect issue #163 found. [`decay_photon_continua`] returns it.
///
/// The chain records each line's intensity **per atom per second**, not per
/// decay: it is the emission probability already multiplied by the nuclide's
/// decay constant. Co60's 1332 keV line, emitted on 99.98% of decays, is stored
/// as 4.166e-9, which is `0.9998 * ln2 / 1.663e8 s`. So the emission rate is
/// the line intensity times the nuclide's ATOM COUNT, and multiplying by its
/// activity instead would count the decay constant twice.
///
/// Feed the two columns to a `Discrete` photon source to transport the shape,
/// and keep the sum for the absolute rate that scales the tallies. Nuclides the
/// chain does not know and non-photon sources are skipped. Contributions are
/// summed in nuclide-name order so two runs over the same inventory produce the
/// same last bit.
pub fn decay_photon_lines(
    atom_densities: &HashMap<String, f64>,
    volume: f64,
    chain: &HashMap<String, ChainNuclide>,
) -> Vec<(f64, f64)> {
    let mut names: Vec<&String> = atom_densities.keys().collect();
    names.sort();

    // Keyed on the energy's bits so coincident lines merge exactly; a BTreeMap
    // so the walk out is ascending and the summation order is fixed.
    let mut lines: std::collections::BTreeMap<u64, f64> = std::collections::BTreeMap::new();
    for name in names {
        let atoms = atom_densities[name] * BARN_PER_CM_SQ * volume;
        if atoms <= 0.0 {
            continue;
        }
        let Some(chain_nuclide) = chain.get(name.as_str()) else {
            continue;
        };
        for source in &chain_nuclide.sources {
            if source.particle != "photon" {
                continue;
            }
            let DecaySourceDistribution::Discrete {
                energies,
                intensities,
            } = &source.distribution
            else {
                continue;
            };
            for (energy, intensity) in energies.iter().zip(intensities) {
                if *intensity > 0.0 {
                    *lines.entry(energy.to_bits()).or_insert(0.0) += atoms * intensity;
                }
            }
        }
    }
    lines
        .into_iter()
        .map(|(bits, rate)| (f64::from_bits(bits), rate))
        .collect()
}

/// One nuclide's decay photon continuum within an inventory.
#[derive(Clone, Debug, PartialEq)]
pub struct PhotonContinuum {
    /// The emitting nuclide.
    pub nuclide: String,
    /// Tabulated energies [eV], ascending.
    pub energies: Vec<f64>,
    /// The emission-rate density at each energy [photons/s/eV]: the chain's
    /// per-atom density times the nuclide's atom count.
    pub rates: Vec<f64>,
    /// How `rates` is read between the energies, `None` where the chain
    /// states no law.
    pub interpolation: Option<Interpolation>,
}

impl PhotonContinuum {
    /// Photons per second over the whole continuum, its integral under its
    /// law.
    pub fn emission_rate(&self) -> Result<f64, UnreadableContinuum> {
        Ok(Continuum::new(&self.energies, &self.rates, self.interpolation)?.integral())
    }
}

/// The decay photon continua an inventory emits, one per nuclide and
/// continuum, in nuclide-name order.
///
/// The counterpart of [`decay_photon_lines`] for the part of a decay spectrum
/// ENDF gives as a density. Each continuum stays on its own grid and keeps its
/// own law: two continua cannot be summed point by point unless they share
/// both, so merging them would mean resampling one onto the other.
///
/// Atom counts scale the chain's per-atom densities exactly as they scale line
/// intensities, and nuclides the chain does not know are skipped. A continuum
/// whose law the chain does not state is still returned, with `None` for it,
/// so the caller can see what it cannot integrate.
pub fn decay_photon_continua(
    atom_densities: &HashMap<String, f64>,
    volume: f64,
    chain: &HashMap<String, ChainNuclide>,
) -> Vec<PhotonContinuum> {
    let mut names: Vec<&String> = atom_densities.keys().collect();
    names.sort();

    let mut continua = Vec::new();
    for name in names {
        let atoms = atom_densities[name] * BARN_PER_CM_SQ * volume;
        if atoms <= 0.0 {
            continue;
        }
        let Some(chain_nuclide) = chain.get(name.as_str()) else {
            continue;
        };
        for source in &chain_nuclide.sources {
            if source.particle != "photon" {
                continue;
            }
            let DecaySourceDistribution::Tabular {
                energies,
                intensities,
                interpolation,
            } = &source.distribution
            else {
                continue;
            };
            continua.push(PhotonContinuum {
                nuclide: name.clone(),
                energies: energies.clone(),
                rates: intensities.iter().map(|d| atoms * d).collect(),
                interpolation: *interpolation,
            });
        }
    }
    continua
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nuclide(name: &str, half_life: Option<f64>, decay_energy: f64) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life,
            decay_energy,
            reactions: vec![],
            decays: vec![],
            fission_yields: None,
            sources: vec![],
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
        }
    }

    fn test_chain() -> HashMap<String, ChainNuclide> {
        let mut chain = HashMap::new();
        chain.insert(
            "Mn56".to_string(),
            nuclide("Mn56", Some(9284.04), 2_522_640.3),
        );
        chain.insert("Fe56".to_string(), nuclide("Fe56", None, 0.0));
        chain
    }

    #[test]
    fn activity_matches_hand_calculation() {
        let chain = test_chain();
        let mut densities = HashMap::new();
        densities.insert("Mn56".to_string(), 1.0e-12);

        let activity = activity_by_nuclide(&densities, 2.0, &chain);

        let atoms = 1.0e-12 * 1.0e24 * 2.0;
        let expected = atoms * std::f64::consts::LN_2 / 9284.04;

        assert_eq!(activity.len(), 1);
        assert!((activity["Mn56"] - expected).abs() < expected * 1e-12);
        assert!((activity_total(&densities, 2.0, &chain) - expected).abs() < expected * 1e-12);
    }

    #[test]
    fn decay_heat_matches_hand_calculation() {
        let chain = test_chain();
        let mut densities = HashMap::new();
        densities.insert("Mn56".to_string(), 1.0e-12);

        let heat = decay_heat_by_nuclide(&densities, 2.0, &chain);

        let atoms = 1.0e-12 * 1.0e24 * 2.0;
        let activity = atoms * std::f64::consts::LN_2 / 9284.04;
        let expected = activity * 2_522_640.3 * EV_TO_J;

        assert_eq!(heat.len(), 1);
        assert!((heat["Mn56"] - expected).abs() < expected * 1e-12);
        assert!((decay_heat_total(&densities, 2.0, &chain) - expected).abs() < expected * 1e-12);
    }

    /// Two maps of the same content iterate in different orders, and a total
    /// summed in that order moves in the last bit. Invisible in one answer;
    /// over an ensemble it makes two identical inventories disagree and puts a
    /// spread on a quantity that has none (issue #558).
    #[test]
    fn a_total_does_not_depend_on_the_map_it_came_out_of() {
        let pairs = [
            ("Fe56", 0.1),
            ("Mn56", 0.2),
            ("Cr51", 0.3),
            ("Co60", 0.7),
            ("Ni57", 1.3),
            ("Zn65", 2.9),
        ];
        let first: HashMap<String, f64> = pairs.iter().map(|(n, v)| (n.to_string(), *v)).collect();
        // Independently seeded maps, so at least one iterates differently.
        for _ in 0..64 {
            let again: HashMap<String, f64> = pairs
                .iter()
                .rev()
                .map(|(n, v)| (n.to_string(), *v))
                .collect();
            assert_eq!(total(&first), total(&again));
        }
        assert_eq!(total(&HashMap::new()), -0.0, "Rust's additive identity");
    }

    #[test]
    fn skips_stable_and_unknown_nuclides() {
        let chain = test_chain();
        let mut densities = HashMap::new();
        densities.insert("Mn56".to_string(), 1.0e-12);
        densities.insert("Fe56".to_string(), 1.0e-12); // stable -> omitted
        densities.insert("Xx99".to_string(), 1.0e-12); // absent from chain -> omitted

        assert_eq!(
            activity_by_nuclide(&densities, 1.0, &chain)
                .keys()
                .collect::<Vec<_>>(),
            vec!["Mn56"]
        );
        assert_eq!(
            decay_heat_by_nuclide(&densities, 1.0, &chain)
                .keys()
                .collect::<Vec<_>>(),
            vec!["Mn56"]
        );
    }

    /// The components of a split decay energy sum to the whole decay heat,
    /// and each is its own share of it.
    #[test]
    fn component_heats_sum_to_the_decay_heat() {
        let mut chain = test_chain();
        let parts = [0.8e6, 1.6e6, 0.12264e6];
        let mn56 = chain.get_mut("Mn56").unwrap();
        mn56.decay_energy = parts.iter().sum();
        for (slot, energy) in mn56.decay_energy_components.iter_mut().zip(parts) {
            *slot = Some(yani::DecayEnergyComponent {
                energy,
                uncertainty: None,
            });
        }
        let densities = HashMap::from([("Mn56".to_string(), 1.0e-12)]);
        let whole = decay_heat_total(&densities, 2.0, &chain);
        let by_part: Vec<f64> = (0..3)
            .map(|c| total(&decay_heat_component_by_nuclide(&densities, 2.0, &chain, c).unwrap()))
            .collect();
        assert!((by_part.iter().sum::<f64>() - whole).abs() < 1e-12 * whole);
        assert!((by_part[1] / whole - 1.6e6 / mn56_total(&chain)).abs() < 1e-12);
    }

    fn mn56_total(chain: &HashMap<String, ChainNuclide>) -> f64 {
        chain["Mn56"].decay_energy
    }

    /// Sm158 as ENDF/B-VIII.1 gives it has no lines at all, only a continuum;
    /// here a coarse histogram of the same kind beside one line.
    fn chain_with_a_continuum(
        interpolation: Option<Interpolation>,
    ) -> HashMap<String, ChainNuclide> {
        let lambda = std::f64::consts::LN_2 / 318.0;
        let mut sm158 = nuclide("Sm158", Some(318.0), 1.0e6);
        sm158.sources = vec![
            yani::DecaySource {
                particle: "photon".to_string(),
                distribution: DecaySourceDistribution::Discrete {
                    energies: vec![2.0e5],
                    intensities: vec![0.5 * lambda],
                },
            },
            yani::DecaySource {
                particle: "photon".to_string(),
                distribution: DecaySourceDistribution::Tabular {
                    energies: vec![1.0e4, 1.0e5, 1.0e6],
                    intensities: vec![2.0e-5 * lambda, 1.0e-6 * lambda, 0.0],
                    interpolation,
                },
            },
            // A continuum of another particle is not a photon source.
            yani::DecaySource {
                particle: "electron".to_string(),
                distribution: DecaySourceDistribution::Tabular {
                    energies: vec![1.0e4, 1.0e6],
                    intensities: vec![1.0e-6 * lambda, 0.0],
                    interpolation,
                },
            },
        ];
        HashMap::from([("Sm158".to_string(), sm158)])
    }

    /// Lines and continuum come back apart, each in its own units: the line in
    /// photons/s, the continuum in photons/s/eV with its law, and the lines
    /// list never holds a continuum's values (issue #163).
    #[test]
    fn a_continuum_is_returned_apart_from_the_lines() {
        let chain = chain_with_a_continuum(Some(Interpolation::Histogram));
        let densities = HashMap::from([("Sm158".to_string(), 1.0e-12)]);
        let atoms = 1.0e-12 * 1.0e24 * 2.0;
        let lambda = std::f64::consts::LN_2 / 318.0;

        assert_eq!(
            decay_photon_lines(&densities, 2.0, &chain),
            vec![(2.0e5, atoms * 0.5 * lambda)]
        );

        let continua = decay_photon_continua(&densities, 2.0, &chain);
        assert_eq!(
            continua.len(),
            1,
            "the electron continuum is not a photon one"
        );
        let c = &continua[0];
        assert_eq!(c.nuclide, "Sm158");
        assert_eq!(c.energies, vec![1.0e4, 1.0e5, 1.0e6]);
        assert_eq!(c.interpolation, Some(Interpolation::Histogram));
        assert_eq!(c.rates[1], atoms * 1.0e-6 * lambda);
        // 2e-5 per eV over 9e4 eV, then 1e-6 per eV over 9e5 eV: 2.7 photons
        // per decay.
        let expected = atoms * lambda * (2.0e-5 * 9.0e4 + 1.0e-6 * 9.0e5);
        let rate = c.emission_rate().unwrap();
        assert!(
            (rate / expected - 1.0).abs() < 1e-14,
            "{rate} != {expected}"
        );
    }

    /// A continuum with no stated law still comes back, so a caller sees it,
    /// but has no emission rate to give.
    #[test]
    fn a_continuum_without_a_law_is_returned_but_not_integrated() {
        let chain = chain_with_a_continuum(None);
        let densities = HashMap::from([("Sm158".to_string(), 1.0e-12)]);
        let continua = decay_photon_continua(&densities, 1.0, &chain);
        assert_eq!(continua[0].interpolation, None);
        assert_eq!(continua[0].emission_rate(), Err(UnreadableContinuum::NoLaw));
    }

    /// Data without the split cannot give a component, and says which
    /// nuclides it could not split rather than reporting a partial heat.
    #[test]
    fn a_component_heat_without_the_split_names_what_is_missing() {
        let chain = test_chain();
        let densities = HashMap::from([("Mn56".to_string(), 1.0e-12), ("Fe56".to_string(), 1.0)]);
        let err = decay_heat_component_by_nuclide(&densities, 2.0, &chain, 1).unwrap_err();
        // Fe56 is stable and makes no heat, so only Mn56 is missing.
        assert_eq!(err, vec!["Mn56".to_string()]);
    }
}
