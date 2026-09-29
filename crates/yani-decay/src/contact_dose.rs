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
//! summed over lines and over nuclides, and integrated over energy for a
//! continuum, where `mu_material` is the linear attenuation coefficient [1/cm]
//! of the material itself, `response` is the mass energy-absorption
//! coefficient of air [cm^2/g] or the ICRP-116 effective-dose coefficient
//! [pSv cm^2], and `B` is a build-up factor standing in for the photons that
//! scatter in the slab and still arrive.
//!
//! This follows the FISPACT-II manual (UKAEA-CCFE-RE(21)02, Appendix C.7.1) for
//! the absorbed-air quantity. For photon lines it matches what OpenMC's
//! `Material.get_photon_contact_dose_rate` computes. A continuum is integrated
//! exactly under its evaluated interpolation law.
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
use yani::{ChainNuclide, Continuum, DecaySourceDistribution};

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

/// The eight-point Gauss-Legendre rule on [-1, 1], as (node, weight) for the
/// positive nodes; the negative ones mirror them. Exact for polynomials up to
/// degree 15.
const GAUSS_LEGENDRE_8: [(f64, f64); 4] = [
    (0.183_434_642_495_649_8, 0.362_683_783_378_362),
    (0.525_532_409_916_329, 0.313_706_645_877_887_3),
    (0.796_666_477_413_626_7, 0.222_381_034_453_374_5),
    (0.960_289_856_497_536_3, 0.101_228_536_290_376_3),
];

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

/// The dose weight's zeroth and first moments over each interval of a grid:
/// `(midpoint, integral of w, integral of (E - midpoint) w)`.
///
/// A continuum is constant or linear between its own points, so on an interval
/// inside one of them its density is exactly `value + slope * (E - midpoint)`,
/// and its fold against the weight is `value * m0 + slope * m1`. The law is
/// then read exactly and only the weight is left to a quadrature. The weight
/// is a ratio of log-log tables, each smooth between its own points, and the
/// grid holds every one of those points, so on each interval it is smooth and
/// the eight-point rule converges to rounding.
fn weight_moments(grid: &[f64], weight: impl Fn(f64) -> f64) -> Vec<(f64, f64, f64)> {
    grid.windows(2)
        .map(|interval| {
            let (mid, half) = (
                0.5 * (interval[0] + interval[1]),
                0.5 * (interval[1] - interval[0]),
            );
            let (mut m0, mut m1) = (0.0, 0.0);
            for (node, w) in GAUSS_LEGENDRE_8 {
                for offset in [-half * node, half * node] {
                    let value = w * weight(mid + offset);
                    m0 += value;
                    m1 += value * offset;
                }
            }
            (mid, half * m0, half * m1)
        })
        .collect()
}

/// Every energy a continuum fold has to break at inside `[lowest, highest]`:
/// the response's and each element's tabulated points, and each continuum's
/// own points and the ends of its part of the range.
fn fold_grid(
    response: &CoefficientTable,
    attenuation: &LinearAttenuation,
    continua: &[(usize, Continuum<'_>)],
    lowest: f64,
    highest: f64,
) -> Vec<f64> {
    let tables = std::iter::once(response.energy())
        .chain(attenuation.terms.iter().map(|(_, table)| table.energy()))
        .chain(continua.iter().map(|(_, c)| c.energies()));
    let mut grid: Vec<f64> = tables
        .flatten()
        .copied()
        .filter(|e| *e > lowest && *e < highest)
        .chain(
            continua
                .iter()
                .flat_map(|(_, c)| clipped_range(c, lowest, highest)),
        )
        .collect();
    grid.push(lowest);
    grid.push(highest);
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    grid
}

/// The part of a continuum's range inside `[lowest, highest]`, as its two ends.
fn clipped_range(continuum: &Continuum<'_>, lowest: f64, highest: f64) -> [f64; 2] {
    let energies = continuum.energies();
    [
        energies[0].max(lowest),
        energies[energies.len() - 1].min(highest),
    ]
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
/// A continuum is integrated over its part of that range, its density read
/// exactly under its law (see [`weight_moments`]). The part below or above the
/// range is left out, as a line there is, and nothing reports it: for the
/// JEFF-4.0 Cf252 continuum, which starts at 0 eV, it is 3.3e-4 of the
/// continuum's emission below 10 keV and 3.3e-6 below 1 keV (see
/// `yani-convert/tests/decay_continuum.rs`). A continuum this build cannot
/// integrate is an `Err` naming the nuclide rather than a smaller dose: one
/// with no stated law, one tabulated under a law other than histogram or
/// linear-linear, and a hand-built one whose lists are unpaired, whose
/// energies are not finite or descend, or whose densities are negative or
/// not finite. Its integral is unknown, and leaving it out would
/// understate the answer by an unknown amount. A continuum wholly outside the
/// range needs no law and adds nothing, as a line there does.
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

    // The line fold of each nuclide, and the continua to fold after it. The
    // continua wait because their weight moments are taken once, over the
    // union of every continuum's grid, rather than once per nuclide.
    let mut folds: Vec<(&String, f64, f64)> = Vec::new();
    let mut continua: Vec<(usize, Continuum<'_>)> = Vec::new();
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
            match &source.distribution {
                DecaySourceDistribution::Discrete {
                    energies,
                    intensities,
                } => {
                    for (&energy, &intensity) in energies.iter().zip(intensities) {
                        if intensity <= 0.0 || energy < lowest || energy > highest {
                            continue;
                        }
                        let mut term =
                            response.interpolate(energy) / attenuation.at(energy) * intensity;
                        if weigh_by_energy {
                            term *= energy;
                        }
                        folded += term;
                    }
                }
                DecaySourceDistribution::Tabular {
                    energies,
                    intensities,
                    interpolation,
                } => {
                    let (Some(&first), Some(&last)) = (energies.first(), energies.last()) else {
                        continue;
                    };
                    if last <= lowest || first >= highest {
                        continue;
                    }
                    let continuum =
                        Continuum::new(energies, intensities, *interpolation).map_err(|why| {
                            format!(
                                "The decay photon continuum of {name} {why}. A contact dose \
                                 without it would be understated by an unknown amount."
                            )
                        })?;
                    continua.push((folds.len(), continuum));
                }
            }
        }
        folds.push((name, density, folded));
    }

    if !continua.is_empty() {
        let grid = fold_grid(&response, &attenuation, &continua, lowest, highest);
        let moments = weight_moments(&grid, |energy| {
            let w = response.interpolate(energy) / attenuation.at(energy);
            if weigh_by_energy {
                w * energy
            } else {
                w
            }
        });
        for (fold, continuum) in &continua {
            // Both ends are grid points, so the intervals between them are
            // exactly the continuum's part of the range.
            let [start, end] = clipped_range(continuum, lowest, highest);
            let first = grid.partition_point(|&e| e < start);
            let last = grid.partition_point(|&e| e < end);
            let mut integral = 0.0;
            for &(mid, m0, m1) in &moments[first..last] {
                let (value, slope) = continuum.value_and_slope(mid);
                integral += value * m0 + slope * m1;
            }
            folds[*fold].2 += integral;
        }
    }

    let mut doses = HashMap::new();
    for (name, density, folded) in folds {
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
            decay_energy_components: Default::default(),
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
                decay_energy_components: Default::default(),
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
                decay_energy_components: Default::default(),
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

    /// Iron with a trace of Fe59 carrying only a synthetic continuum, so the
    /// material and the emitter are one element and the weight on every grid
    /// interval is a single power law, which integrates in closed form.
    fn iron_with_a_continuum(
        energies: Vec<f64>,
        densities: Vec<f64>,
        interpolation: Option<yani::Interpolation>,
    ) -> (HashMap<String, f64>, HashMap<String, ChainNuclide>) {
        let mut chain = HashMap::new();
        for (name, sources) in [
            ("Fe56", vec![]),
            (
                "Fe59",
                vec![DecaySource {
                    particle: "photon".to_string(),
                    distribution: DecaySourceDistribution::Tabular {
                        energies,
                        intensities: densities,
                        interpolation,
                    },
                }],
            ),
        ] {
            chain.insert(
                name.to_string(),
                ChainNuclide {
                    name: name.to_string(),
                    half_life: (name == "Fe59").then_some(3.84e6),
                    decay_energy: 0.0,
                    reactions: vec![],
                    decays: vec![],
                    fission_yields: None,
                    sources,
                    half_life_uncertainty: None,
                    decay_energy_uncertainty: None,
                    decay_energy_components: Default::default(),
                },
            );
        }
        let densities = HashMap::from([
            ("Fe56".to_string(), 0.084_912),
            ("Fe59".to_string(), 1.0e-6),
        ]);
        (densities, chain)
    }

    /// Irregular points across the whole tabulated range, with a jump, and
    /// reaching past both ends of it.
    fn synthetic_continuum() -> (Vec<f64>, Vec<f64>) {
        (
            vec![
                500.0, 2.5e3, 7.3e3, 7.3e3, 2.0e4, 8.8e4, 3.1e5, 1.0e6, 2.7e6, 9.0e6, 3.0e7,
            ],
            vec![
                1.0e-9, 4.0e-8, 2.0e-8, 6.0e-8, 3.0e-8, 1.0e-8, 4.0e-9, 1.5e-9, 3.0e-10, 2.0e-11,
                0.0,
            ],
        )
    }

    /// The fold in closed form: on each interval of the union grid the weight
    /// is the power law through its end values and the density is linear, so
    /// the integral of their product is two power-law integrals.
    fn closed_form_fold(
        densities: &HashMap<String, f64>,
        chain: &HashMap<String, ChainNuclide>,
        quantity: DoseQuantity,
    ) -> f64 {
        let DecaySourceDistribution::Tabular {
            energies,
            intensities,
            interpolation,
        } = &chain["Fe59"].sources[0].distribution
        else {
            unreachable!()
        };
        let continuum = Continuum::new(energies, intensities, *interpolation).unwrap();
        let attenuation = linear_attenuation(densities).unwrap();
        let response = response_function(quantity);
        let (lowest, highest) = (
            response
                .min_energy()
                .max(attenuation.terms[0].1.min_energy()),
            response
                .max_energy()
                .min(attenuation.terms[0].1.max_energy()),
        );
        let weight = |e: f64| {
            let w = response.interpolate(e) / attenuation.at(e);
            if quantity == DoseQuantity::AbsorbedAir {
                w * e
            } else {
                w
            }
        };
        let grid = fold_grid(&response, &attenuation, &[(0, continuum)], lowest, highest);
        let mut total = 0.0;
        for interval in grid.windows(2) {
            let (a, b) = (interval[0], interval[1]);
            // An absorption edge's two energies are one ulp apart, and the
            // interval between them holds nothing.
            if b / a - 1.0 < 1e-12 {
                continue;
            }
            let (wa, r) = (weight(a), b / a);
            let p = (weight(b) / wa).ln() / r.ln();
            let (value, slope) = continuum.value_and_slope(0.5 * (a + b));
            let at_a = value - slope * 0.5 * (b - a);
            let power = |k: f64| a * (r.powf(p + k) - 1.0) / (p + k);
            // density = at_a + slope (E - a) = (at_a - slope a) + slope E.
            total += wa * ((at_a - slope * a) * power(1.0) + slope * a * power(2.0));
        }
        total * densities["Fe59"] * multiplier(quantity, 2.0)
    }

    /// The continuum fold agrees with the closed form under both laws and for
    /// both quantities, which is what "read exactly" means here: the law is
    /// taken as the evaluation states it, and the quadrature of the weight
    /// leaves nothing above rounding.
    #[test]
    fn a_continuum_folds_to_its_closed_form() {
        for law in [
            yani::Interpolation::Histogram,
            yani::Interpolation::LinearLinear,
        ] {
            for quantity in [DoseQuantity::AbsorbedAir, DoseQuantity::Effective] {
                let (energies, values) = synthetic_continuum();
                let (densities, chain) = iron_with_a_continuum(energies, values, Some(law));
                let dose = contact_dose_total(&densities, &chain, quantity, 2.0).unwrap();
                let expected = closed_form_fold(&densities, &chain, quantity);
                assert!(
                    (dose / expected - 1.0).abs() < 1e-14,
                    "{law:?} {quantity:?}: {dose} != {expected}"
                );
            }
        }
    }

    /// The eight-point rule is exact through degree 15, which pins every node
    /// and weight to rounding.
    #[test]
    fn the_quadrature_rule_is_exact_through_degree_fifteen() {
        for degree in 0..16 {
            let integral: f64 = GAUSS_LEGENDRE_8
                .iter()
                .map(|(x, w)| w * (x.powi(degree) + (-x).powi(degree)))
                .sum();
            let exact = if degree % 2 == 0 {
                2.0 / (degree as f64 + 1.0)
            } else {
                0.0
            };
            assert!(
                (integral - exact).abs() < 4e-16,
                "degree {degree}: {integral} != {exact}"
            );
        }
    }

    /// Halving every interval moves nothing: the rule has converged on a
    /// material whose weight is a sum over elements, where no closed form
    /// exists.
    #[test]
    fn the_weight_moments_have_converged_on_a_mixture() {
        let densities = HashMap::from([
            ("Fe56".to_string(), 0.06),
            ("Cr52".to_string(), 0.016),
            ("Ni58".to_string(), 0.008),
            ("W184".to_string(), 1.0e-4),
        ]);
        let attenuation = linear_attenuation(&densities).unwrap();
        let response = response_function(DoseQuantity::AbsorbedAir);
        let (energies, values) = synthetic_continuum();
        let continuum =
            Continuum::new(&energies, &values, Some(yani::Interpolation::LinearLinear)).unwrap();
        let (lowest, highest) = (1.0e3, 2.0e7);
        let grid = fold_grid(&response, &attenuation, &[(0, continuum)], lowest, highest);
        let halved: Vec<f64> = grid
            .windows(2)
            .flat_map(|w| [w[0], 0.5 * (w[0] + w[1])])
            .chain([highest])
            .collect();
        let weight = |e: f64| response.interpolate(e) / attenuation.at(e) * e;
        let fold = |grid: &[f64]| -> f64 {
            weight_moments(grid, weight)
                .iter()
                .map(|&(mid, m0, m1)| {
                    let (value, slope) = continuum.value_and_slope(mid);
                    value * m0 + slope * m1
                })
                .sum()
        };
        let (once, twice) = (fold(&grid), fold(&halved));
        assert!(
            (once / twice - 1.0).abs() < 1e-14,
            "{once} vs {twice} on the halved grid"
        );
    }

    /// A continuum whose law the chain does not state stops the dose rather
    /// than leaving itself out of it, and says which nuclide and why.
    #[test]
    fn a_continuum_without_a_law_is_an_error_naming_it() {
        let (energies, values) = synthetic_continuum();
        let (densities, chain) = iron_with_a_continuum(energies, values, None);
        let error =
            contact_dose_total(&densities, &chain, DoseQuantity::AbsorbedAir, 2.0).unwrap_err();
        assert!(error.contains("Fe59"), "{error}");
        assert!(error.contains("interpolation"), "{error}");
    }

    /// Outside the tabulated range a continuum adds nothing, as a line there
    /// does, and needs no law to add nothing.
    #[test]
    fn a_continuum_outside_the_tabulated_range_needs_no_law() {
        let (densities, chain) = iron_with_a_continuum(vec![2.5e7, 3.0e7], vec![1.0, 1.0], None);
        let doses =
            contact_dose_by_nuclide(&densities, &chain, DoseQuantity::AbsorbedAir, 2.0).unwrap();
        assert!(doses.is_empty(), "{doses:?}");
    }

    /// The per-eV values of a continuum are not line intensities. Read as lines
    /// they gave a dose smaller by roughly the grid spacing in eV, which is the
    /// defect issue #163 describes.
    #[test]
    fn a_continuum_is_not_its_values_read_as_lines() {
        let (energies, values) = synthetic_continuum();
        let (densities, mut chain) = iron_with_a_continuum(
            energies.clone(),
            values.clone(),
            Some(yani::Interpolation::Histogram),
        );
        let as_continuum =
            contact_dose_total(&densities, &chain, DoseQuantity::AbsorbedAir, 2.0).unwrap();
        chain.get_mut("Fe59").unwrap().sources[0].distribution =
            DecaySourceDistribution::Discrete {
                energies,
                intensities: values,
            };
        let as_lines =
            contact_dose_total(&densities, &chain, DoseQuantity::AbsorbedAir, 2.0).unwrap();
        assert!(
            as_continuum > 1e3 * as_lines,
            "continuum {as_continuum}, lines {as_lines}"
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
