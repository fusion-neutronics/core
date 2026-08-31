//! Contact dose rate of an activated inventory.
//!
//! The dose someone receives with a hand on the material, from its own decay
//! photons. It is a slab estimate, not a transport result: the material is
//! taken to be a half-space, so half the photons emitted at any depth head
//! towards the surface, and the ones that reach it are the ones the material
//! did not attenuate. Integrating that over depth cancels the geometry
//! entirely, leaving the emission rate divided by the material's own linear
//! attenuation coefficient -- a self-absorption estimate with no distance in it.
//!
//! Per photon line of energy `E` and per-atom emission rate `S`, the estimate is
//!
//! ```text
//!     (B / 2) * (response(E) / mu_material(E)) * S * E     [absorbed dose in air]
//!     (B / 2) * (response(E) / mu_material(E)) * S         [effective dose]
//! ```
//!
//! summed over lines and over nuclides, where `mu_material` is the linear
//! attenuation coefficient [1/cm] of the material itself, `response` is the
//! mass energy-absorption coefficient of air [cm^2/g] or the ICRP-116
//! effective-dose coefficient [pSv cm^2], and `B` is a build-up factor
//! standing in for the photons that scatter in the slab and still arrive.
//!
//! This follows the FISPACT-II manual (UKAEA-CCFE-RE(21)02, Appendix C.7.1) for
//! the absorbed-air quantity, and matches what OpenMC's
//! `Material.get_photon_contact_dose_rate` computes.
//!
//! Two things it does not model: bremsstrahlung from decay electrons, which
//! matters at contact for strong beta emitters, and any nuclide whose radiation
//! the chain file does not describe, which contributes nothing rather than
//! raising.

use std::collections::{BTreeMap, HashMap};

use yamc_nuclide::composition::{atomic_mass, atomic_number};
use yamc_nuclide::data::effective_dose::{
    dose_coefficients, DoseDataSource, DoseGeometry, DoseParticle,
};
use yamc_nuclide::data::photon_attenuation::{
    mass_attenuation_coefficient, mass_energy_absorption_air, CoefficientTable,
};
use yani::{ChainNuclide, DecaySourceDistribution};

/// Electron-volt to joule conversion (2019 SI redefinition).
const EV_TO_J: f64 = 1.602_176_634e-19;

/// Barns per cm²: converts atom density in atoms/(barn·cm) to atoms/cm³.
const BARN_PER_CM_SQ: f64 = 1.0e24;

/// Avogadro constant (2019 SI redefinition) [1/mol].
const AVOGADRO: f64 = 6.022_140_76e23;

/// Seconds per hour: the dose rates are reported per hour.
const SECONDS_PER_HOUR: f64 = 3600.0;

/// Grams per kilogram: absorbed dose is per kilogram, mu_en/rho is per gram.
const GRAMS_PER_KG: f64 = 1000.0;

/// Sieverts per picosievert: the ICRP coefficients are tabulated in pSv cm².
const SV_PER_PSV: f64 = 1.0e-12;

/// Half the photons emitted anywhere in a half-space head towards its surface.
const SLAB_GEOMETRY_FACTOR: f64 = 0.5;

/// The dose quantity a contact dose rate is expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoseQuantity {
    /// Absorbed dose in air [Gy/h], following FISPACT-II. The response is the
    /// mass energy-absorption coefficient of air.
    AbsorbedAir,
    /// Effective dose [Sv/h]. The response is the ICRP-116 photon
    /// effective-dose coefficient for anterior-posterior irradiation.
    Effective,
}

impl DoseQuantity {
    /// The unit the quantity comes out in, for labelling a result.
    pub fn units(&self) -> &'static str {
        match self {
            DoseQuantity::AbsorbedAir => "Gy/h",
            DoseQuantity::Effective => "Sv/h",
        }
    }
}

/// The linear attenuation coefficient of a material, evaluated per element
/// rather than off a pre-summed grid.
///
/// Each element's mu/rho is tabulated for log-log reading, so summing the
/// interpolated values is what the tabulations ask for; summing first onto a
/// union grid and interpolating that would read a curve nobody published.
struct LinearAttenuation {
    /// `(partial mass density [g/cm^3], mu/rho table)` per element present.
    terms: Vec<(f64, &'static CoefficientTable)>,
}

impl LinearAttenuation {
    /// mu_material at `energy` [eV], in 1/cm.
    fn at(&self, energy: f64) -> f64 {
        self.terms
            .iter()
            .map(|(density, table)| density * table.interpolate(energy))
            .sum()
    }
}

/// Build the material's linear attenuation coefficient from its atom densities.
///
/// Each nuclide contributes a partial mass density `N * 1e24 * M / N_A` to its
/// element, and the element's mu/rho carries it to 1/cm.
///
/// Both orders here are fixed, and both have to be. The nuclides are walked in
/// name order, because an element with several isotopes -- iron in anything
/// activated -- sums them into one bucket, and `HashMap` iteration order is a
/// property of the map INSTANCE rather than of its contents: two maps built from
/// the same material walk differently, and so did two calls on the same one.
/// The buckets are then a `BTreeMap`, so `terms` comes out in atomic-number
/// order for `LinearAttenuation::at` to sum.
///
/// Sorting only the buckets is what this used to do, and it is not enough: it
/// fixes the order the elements are added in and leaves the order the isotopes
/// of one element were added in to the map's seed. `Material.contact_dose()`
/// returned three different values across six runs, and three across eight
/// calls in one process, because `mu` divides every photon-line term. Same
/// defect as issue #502 and issue #576, in the denominator of a dose.
fn linear_attenuation(atom_densities: &HashMap<String, f64>) -> Result<LinearAttenuation, String> {
    let mut names: Vec<&String> = atom_densities.keys().collect();
    names.sort();

    let mut by_element: BTreeMap<u32, f64> = BTreeMap::new();
    for nuclide in names {
        let density = atom_densities[nuclide];
        if density <= 0.0 {
            continue;
        }
        let z = atomic_number(nuclide)?;
        let mass = atomic_mass(nuclide)?;
        *by_element.entry(z).or_insert(0.0) += density * BARN_PER_CM_SQ * mass / AVOGADRO;
    }
    if by_element.is_empty() {
        return Err(
            "Material has no nuclides at a positive density; cannot compute the linear \
             attenuation coefficient a contact dose divides by"
                .to_string(),
        );
    }

    let mut terms = Vec::with_capacity(by_element.len());
    for (z, density) in by_element {
        let table = mass_attenuation_coefficient(z).ok_or_else(|| {
            format!("No photon mass attenuation data for Z={z}; the tabulation covers Z = 1 to 100")
        })?;
        terms.push((density, table));
    }
    Ok(LinearAttenuation { terms })
}

/// The response function a dose quantity folds the photon lines against.
fn response_function(quantity: DoseQuantity) -> CoefficientTable {
    match quantity {
        DoseQuantity::AbsorbedAir => mass_energy_absorption_air().clone(),
        DoseQuantity::Effective => {
            let (energy, coefficients) = dose_coefficients(
                DoseParticle::Photon,
                DoseGeometry::AP,
                DoseDataSource::ICRP116,
            );
            CoefficientTable::new(energy, coefficients)
        }
    }
}

/// The unit conversion that turns the folded integral into a dose rate.
fn multiplier(quantity: DoseQuantity, build_up: f64) -> f64 {
    let common = build_up * SLAB_GEOMETRY_FACTOR * SECONDS_PER_HOUR * BARN_PER_CM_SQ;
    match quantity {
        // [eV cm^2 / (b g s)] -> [Gy/h]
        DoseQuantity::AbsorbedAir => common * GRAMS_PER_KG * EV_TO_J,
        // [pSv cm^2 / (b s)] -> [Sv/h]
        DoseQuantity::Effective => common * SV_PER_PSV,
    }
}

/// Contact dose rate contribution of each nuclide in `atom_densities`.
///
/// `atom_densities` maps nuclide name -> atoms/(barn·cm). No volume is needed:
/// the slab estimate is intensive, so a bigger lump of the same material reads
/// the same at contact.
///
/// The chain records each photon line's intensity **per atom per second** (the
/// emission probability already multiplied by the decay constant), so a
/// nuclide's contribution scales with its atom density and not with its
/// activity -- multiplying by the activity would count the decay constant
/// twice.
///
/// Lines outside the range both the attenuation and the response tabulation
/// cover are dropped rather than extrapolated; for the absorbed-air quantity
/// that range is 1 keV to 20 MeV, and for the effective-dose quantity 10 keV to
/// 20 MeV. Nuclides that contribute nothing -- stable ones, ones the chain does
/// not know, ones with no photon lines in range -- are omitted from the map
/// rather than returned as zero.
///
/// Units follow `quantity`: Gy/h for [`DoseQuantity::AbsorbedAir`], Sv/h for
/// [`DoseQuantity::Effective`].
pub fn contact_dose_by_nuclide(
    atom_densities: &HashMap<String, f64>,
    chain: &HashMap<String, ChainNuclide>,
    quantity: DoseQuantity,
    build_up: f64,
) -> Result<HashMap<String, f64>, String> {
    if build_up <= 0.0 || build_up.is_nan() {
        return Err(format!("build_up must be positive, got {build_up}"));
    }

    let attenuation = linear_attenuation(atom_densities)?;
    let response = response_function(quantity);

    // The lines a fold can be taken over: inside both tabulations.
    let lowest = response.min_energy().max(
        attenuation
            .terms
            .iter()
            .map(|(_, table)| table.min_energy())
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let highest = response.max_energy().min(
        attenuation
            .terms
            .iter()
            .map(|(_, table)| table.max_energy())
            .fold(f64::INFINITY, f64::min),
    );

    let multiplier = multiplier(quantity, build_up);
    let weigh_by_energy = quantity == DoseQuantity::AbsorbedAir;

    let mut names: Vec<&String> = atom_densities.keys().collect();
    names.sort();

    let mut doses = HashMap::new();
    for name in names {
        let density = atom_densities[name];
        if density <= 0.0 {
            continue;
        }
        let Some(chain_nuclide) = chain.get(name.as_str()) else {
            continue;
        };

        let mut folded = 0.0;
        for source in &chain_nuclide.sources {
            if source.particle != "photon" {
                continue;
            }
            let DecaySourceDistribution::Discrete {
                energies,
                intensities,
            } = &source.distribution;
            for (&energy, &intensity) in energies.iter().zip(intensities) {
                if intensity <= 0.0 || energy < lowest || energy > highest {
                    continue;
                }
                let mut term = response.interpolate(energy) / attenuation.at(energy) * intensity;
                if weigh_by_energy {
                    term *= energy;
                }
                folded += term;
            }
        }

        let dose = folded * density * multiplier;
        if dose > 0.0 {
            doses.insert(name.clone(), dose);
        }
    }

    Ok(doses)
}

/// Total contact dose rate of an inventory. See [`contact_dose_by_nuclide`].
///
/// Contributions are summed in nuclide-name order, so two runs over the same
/// inventory agree to the last bit.
pub fn contact_dose_total(
    atom_densities: &HashMap<String, f64>,
    chain: &HashMap<String, ChainNuclide>,
    quantity: DoseQuantity,
    build_up: f64,
) -> Result<f64, String> {
    let doses = contact_dose_by_nuclide(atom_densities, chain, quantity, build_up)?;
    Ok(crate::decay::total(&doses))
}

#[cfg(test)]
mod tests {
    use super::*;
    use yani::DecaySource;

    /// Co60: the 1173 keV and 1332 keV lines, at the intensities a chain file
    /// records -- per atom per second, so `0.9985 * ln2 / 1.663e8 s` and
    /// `0.9998 * ln2 / 1.663e8 s`.
    fn cobalt60() -> ChainNuclide {
        let half_life = 1.663_2e8;
        let lambda = std::f64::consts::LN_2 / half_life;
        ChainNuclide {
            name: "Co60".to_string(),
            half_life: Some(half_life),
            decay_energy: 2_503_000.0,
            reactions: vec![],
            decays: vec![],
            fission_yields: None,
            sources: vec![DecaySource {
                particle: "photon".to_string(),
                distribution: DecaySourceDistribution::Discrete {
                    energies: vec![1_173_228.0, 1_332_492.0],
                    intensities: vec![0.9985 * lambda, 0.9998 * lambda],
                },
            }],
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
        }
    }

    fn chain_with_cobalt() -> HashMap<String, ChainNuclide> {
        let mut chain = HashMap::new();
        chain.insert("Co60".to_string(), cobalt60());
        chain.insert(
            "Fe56".to_string(),
            ChainNuclide {
                name: "Fe56".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        chain
    }

    /// Iron at 7.874 g/cm3 with a trace of Co60, in atoms/(barn·cm).
    fn activated_iron() -> HashMap<String, f64> {
        HashMap::from([
            ("Fe56".to_string(), 0.084_912),
            ("Co60".to_string(), 1.0e-6),
        ])
    }

    #[test]
    fn absorbed_air_dose_matches_a_hand_calculation() {
        let chain = chain_with_cobalt();
        let densities = activated_iron();

        let dose = contact_dose_total(&densities, &chain, DoseQuantity::AbsorbedAir, 2.0).unwrap();

        // Redo the fold by hand over Co60's two lines.
        let lambda = std::f64::consts::LN_2 / 1.663_2e8;
        let air = mass_energy_absorption_air();
        let attenuation = linear_attenuation(&densities).unwrap();
        let mut expected = 0.0;
        for (energy, probability) in [(1_173_228.0, 0.9985), (1_332_492.0, 0.9998)] {
            expected +=
                air.interpolate(energy) / attenuation.at(energy) * probability * lambda * energy;
        }
        expected *= 1.0e-6 * 2.0 * 0.5 * 3600.0 * 1000.0 * 1.0e24 * EV_TO_J;

        assert!(
            (dose - expected).abs() < expected * 1e-12,
            "{dose} != {expected}"
        );
    }

    #[test]
    fn only_photon_emitters_contribute() {
        let doses = contact_dose_by_nuclide(
            &activated_iron(),
            &chain_with_cobalt(),
            DoseQuantity::AbsorbedAir,
            2.0,
        )
        .unwrap();
        assert_eq!(doses.keys().collect::<Vec<_>>(), vec!["Co60"]);
    }

    #[test]
    fn stable_iron_alone_reads_zero() {
        let densities = HashMap::from([("Fe56".to_string(), 0.084_912)]);
        let dose = contact_dose_total(
            &densities,
            &chain_with_cobalt(),
            DoseQuantity::AbsorbedAir,
            2.0,
        )
        .unwrap();
        assert_eq!(dose, 0.0);
    }

    #[test]
    fn dose_scales_with_the_emitter_density_and_the_build_up_factor() {
        let chain = chain_with_cobalt();
        let mut densities = activated_iron();
        let base = contact_dose_total(&densities, &chain, DoseQuantity::AbsorbedAir, 2.0).unwrap();

        // Twice the build-up, twice the dose: it is a plain multiplier.
        let doubled_build_up =
            contact_dose_total(&densities, &chain, DoseQuantity::AbsorbedAir, 4.0).unwrap();
        assert!((doubled_build_up - 2.0 * base).abs() < base * 1e-12);

        // Twice the Co60 and the same iron: twice the dose, since the trace
        // emitter does not measurably change the attenuation it sits in.
        densities.insert("Co60".to_string(), 2.0e-6);
        let doubled_emitter =
            contact_dose_total(&densities, &chain, DoseQuantity::AbsorbedAir, 2.0).unwrap();
        assert!((doubled_emitter / base - 2.0).abs() < 1e-4);
    }

    #[test]
    fn self_shielding_makes_a_denser_host_read_lower() {
        // The same Co60 in lead rather than iron. Lead attenuates its own
        // photons harder, so less of them get out.
        let mut chain = chain_with_cobalt();
        chain.insert(
            "Pb208".to_string(),
            ChainNuclide {
                name: "Pb208".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        let in_lead = HashMap::from([
            ("Pb208".to_string(), 0.032_991),
            ("Co60".to_string(), 1.0e-6),
        ]);

        let iron =
            contact_dose_total(&activated_iron(), &chain, DoseQuantity::AbsorbedAir, 2.0).unwrap();
        let lead = contact_dose_total(&in_lead, &chain, DoseQuantity::AbsorbedAir, 2.0).unwrap();
        assert!(lead < iron, "lead {lead} should read below iron {iron}");
    }

    #[test]
    fn effective_dose_is_a_different_quantity_in_different_units() {
        let chain = chain_with_cobalt();
        let densities = activated_iron();
        let absorbed =
            contact_dose_total(&densities, &chain, DoseQuantity::AbsorbedAir, 2.0).unwrap();
        let effective =
            contact_dose_total(&densities, &chain, DoseQuantity::Effective, 2.0).unwrap();

        assert!(absorbed > 0.0 && effective > 0.0);
        // Around Co60's lines the ICRP-116 AP coefficient is roughly 1.2 Sv per
        // Gy of air kerma, so the two land within a factor of a few.
        let ratio = effective / absorbed;
        assert!((0.5..5.0).contains(&ratio), "unexpected ratio {ratio}");
    }

    #[test]
    fn a_line_outside_the_tabulated_range_is_dropped() {
        let mut chain = chain_with_cobalt();
        // A 30 MeV line is past the 20 MeV top of both tabulations.
        chain.get_mut("Co60").unwrap().sources[0].distribution =
            DecaySourceDistribution::Discrete {
                energies: vec![3.0e7],
                intensities: vec![1.0e-8],
            };
        let doses =
            contact_dose_by_nuclide(&activated_iron(), &chain, DoseQuantity::AbsorbedAir, 2.0)
                .unwrap();
        assert!(doses.is_empty());
    }

    #[test]
    fn matches_openmc_on_cobalt60_in_iron() {
        // 1e-6 atoms/(b·cm) of Co60 in iron at 7.874 g/cm3, run through
        // OpenMC's `Material.get_photon_contact_dose_rate` against its own
        // NIST tabulations. Both codes read the same XCOM mu/rho, the same
        // NIST-126 air mu_en/rho and the same ICRP-116 photon coefficients, so
        // the only thing left to disagree about is the arithmetic.
        //
        // OpenMC sums mu/rho onto a union energy grid and log-log interpolates
        // that; this crate log-log interpolates each element and sums the
        // results, which is what the per-element tabulations are fitted for.
        // On this material the two agree to well inside the 1e-9 checked here.
        let chain = chain_with_cobalt();
        let densities = activated_iron();

        let absorbed =
            contact_dose_total(&densities, &chain, DoseQuantity::AbsorbedAir, 2.0).unwrap();
        let effective =
            contact_dose_total(&densities, &chain, DoseQuantity::Effective, 2.0).unwrap();

        assert!(
            (absorbed / 3.800_881_528_7e2 - 1.0).abs() < 1e-9,
            "absorbed dose {absorbed} Gy/h differs from OpenMC's 380.08815287"
        );
        assert!(
            (effective / 3.807_282_670_9e2 - 1.0).abs() < 1e-9,
            "effective dose {effective} Sv/h differs from OpenMC's 380.72826709"
        );
    }

    #[test]
    fn a_non_positive_build_up_is_rejected() {
        let error = contact_dose_total(
            &activated_iron(),
            &chain_with_cobalt(),
            DoseQuantity::AbsorbedAir,
            0.0,
        )
        .unwrap_err();
        assert!(error.contains("build_up must be positive"), "{error}");
    }

    #[test]
    fn an_empty_inventory_is_rejected() {
        let error = contact_dose_total(
            &HashMap::new(),
            &chain_with_cobalt(),
            DoseQuantity::AbsorbedAir,
            2.0,
        )
        .unwrap_err();
        assert!(error.contains("no nuclides"), "{error}");
    }
}
