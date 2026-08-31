//! Decay-inventory post-processing.
//!
//! Computes the radioactive activity and decay heat of a material inventory
//! from its atom densities and a transmutation chain that supplies half-lives
//! and mean decay energies. Activity is `A_i = N_i · λ_i` (Bq); decay heat is
//! the standard `P = Σ_i A_i · Ē_i` (W), where `Ē_i` is the mean recoverable
//! decay energy per decay.

use std::collections::HashMap;

use yani::{ChainNuclide, DecaySourceDistribution};

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
            } = &source.distribution;
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
}
