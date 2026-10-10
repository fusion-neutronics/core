//! Displacement damage on a spectrum solve, against the Fe56 fixture.
//!
//! Checks the four things a damage request promises: the fold of MT=444 is
//! the collapse's own group average (compared here with a trapezoid written
//! out independently), the NRT conversion uses the element's E_d and reports
//! where it came from, a solve that does not ask for damage neither fetches
//! MT=444 nor changes, and a composition element with no E_d is refused
//! before anything runs.
//!
//! Self-skips when the Fe56 fixture is missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_element::displacement::DisplacementEnergySource;
use yamc_materials::Material;
use yani_transmute::damage::MT_DAMAGE_ENERGY;
use yani_transmute::{
    transmute_materials, DamageRequest, MultigroupSpectrum, TransmutationResults, TransmuteCase,
    TransmuteStep,
};

const BOUNDARIES: [f64; 3] = [1.0e5, 1.0e6, 1.4e7];
const MASSES: [f64; 2] = [0.3, 0.7];
const FLUX: f64 = 1.0e10;
const IRRADIATION: f64 = 300.0;

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

/// Iron with CONFIG pointed at the fixture, so the driver's own activation
/// load (the narrow one) reads it.
fn iron() -> Option<Material> {
    let path = yamc_test_cache::nuclide("Fe56")?;
    {
        let mut cfg = yamc_nuclide::config::CONFIG
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        cfg.set_cross_section("Fe56", Some(&path));
    }
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.nuclides.insert("Fe56".to_string(), 8.5e-2);
    m.volume = Some(1.0);
    m.set_temperature("294");
    Some(m)
}

fn spectrum() -> MultigroupSpectrum {
    MultigroupSpectrum {
        boundaries: BOUNDARIES.to_vec(),
        masses: MASSES.to_vec(),
        flux_error: None,
    }
}

/// Five minutes of irradiation, then an hour of cooling.
fn steps() -> Vec<TransmuteStep> {
    vec![
        TransmuteStep {
            dt: IRRADIATION,
            irradiation: Some((0, FLUX)),
        },
        TransmuteStep {
            dt: 3600.0,
            irradiation: None,
        },
    ]
}

fn run(material: &mut Material, damage: Option<&DamageRequest>) -> TransmutationResults {
    transmute_materials(
        vec![TransmuteCase {
            material,
            spectra: vec![spectrum()],
            steps: steps(),
            shielding: None,
        }],
        chain(),
        &Default::default(),
        Default::default(),
        None,
        damage,
    )
    .expect("transmute")
}

/// The group average of a lin-lin tabulation, written out on its own: the
/// trapezoid over the group's end points and every tabulated point between.
fn group_average(reaction: &yamc_nuclide::reaction::Reaction, lo: f64, hi: f64) -> f64 {
    let mut points = vec![(lo, reaction.cross_section_at(lo).unwrap())];
    for (&e, &xs) in reaction.energy.iter().zip(reaction.cross_section.iter()) {
        if e > lo && e < hi {
            points.push((e, xs));
        }
    }
    points.push((hi, reaction.cross_section_at(hi).unwrap()));
    let integral: f64 = points
        .windows(2)
        .map(|w| 0.5 * (w[0].1 + w[1].1) * (w[1].0 - w[0].0))
        .sum();
    integral / (hi - lo)
}

/// One test rather than several, because the order matters: the process-wide
/// nuclide cache hands a later load whatever wider entry an earlier one made,
/// so "MT=444 was not fetched" can only be checked before anything asks for it.
#[test]
fn damage_energy_and_dpa_on_iron() {
    let Some(base) = iron() else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };

    // Not asked for: no MT=444 loaded, and no damage reported.
    let mut plain = base.clone();
    let without = run(&mut plain, None);
    assert!(
        !plain.nuclide_data["Fe56"]
            .load_scope
            .wants_mt(MT_DAMAGE_ENERGY),
        "MT=444 was loaded for a solve that did not ask for damage"
    );
    assert!(without.displacement_damage.is_empty());

    // Asked for: the inventories are bit-identical, and MT=444 is now held.
    let mut damaged = base.clone();
    let with = run(&mut damaged, Some(&DamageRequest::default()));
    for step in 0..=2 {
        let a = &without.get_material(0, step).unwrap().nuclides;
        let b = &with.get_material(0, step).unwrap().nuclides;
        assert_eq!(a.len(), b.len());
        for (name, n) in a {
            assert_eq!(n.to_bits(), b[name].to_bits(), "step {step} {name}");
        }
    }
    let fe56 = &damaged.nuclide_data["Fe56"];
    assert!(fe56.load_scope.wants_mt(MT_DAMAGE_ENERGY));
    let reaction = fe56.reactions_for_temp("294").unwrap()[&MT_DAMAGE_ENERGY].clone();

    // The fold, against the independent trapezoid. eV barn to eV cm^2 is the
    // 1e-24, exactly as for a reaction rate.
    let per_atom_per_flux: f64 = BOUNDARIES
        .windows(2)
        .zip(MASSES)
        .map(|(w, m)| group_average(&reaction, w[0], w[1]) * m)
        .sum::<f64>()
        * 1.0e-24;
    assert!(per_atom_per_flux > 0.0);
    let damage = with.get_displacement_damage(0).expect("damage reported");
    let expected = per_atom_per_flux * FLUX * IRRADIATION;
    // The products made over five minutes are a few parts in 1e12 of the
    // atoms, so the element's average is Fe56's to far below this tolerance.
    let fe = &damage.element_damage_energy["Fe"];
    assert!(
        (fe[1] - expected).abs() <= 1e-9 * expected,
        "{} vs {expected}",
        fe[1]
    );
    // Cooling adds nothing.
    assert_eq!(fe[2], fe[1]);
    assert_eq!(damage.damage_energy.len(), 3);
    assert_eq!(damage.damage_energy[0], 0.0);

    // NRT at the ASTM E521 value for iron.
    let ed = damage.displacement_energies["Fe"];
    assert_eq!(ed.energy_ev, 40.0);
    assert_eq!(ed.source, DisplacementEnergySource::AstmE521);
    let dpa = &damage.element_dpa["Fe"];
    assert!((dpa[1] - 0.8 * fe[1] / 80.0).abs() <= 1e-15 * dpa[1]);

    // An override is used and reported as the user's.
    let mut overridden = base.clone();
    let request = DamageRequest::new([("Fe".to_string(), 50.0)]).unwrap();
    let user = run(&mut overridden, Some(&request));
    let user = user.get_displacement_damage(0).unwrap();
    assert_eq!(
        user.displacement_energies["Fe"].source,
        DisplacementEnergySource::User
    );
    assert!((user.element_dpa["Fe"][1] - dpa[1] * 40.0 / 50.0).abs() <= 1e-12 * dpa[1]);
}

/// An element of the composition with no E_d is refused before the solve, and
/// the message says how to supply it.
#[test]
fn an_element_without_a_displacement_energy_is_refused() {
    let mut m = Material::new(
        HashMap::from([("Li7".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .unwrap();
    m.nuclides.insert("Li7".to_string(), 0.05);
    let err = transmute_materials(
        vec![TransmuteCase {
            material: &mut m,
            spectra: vec![spectrum()],
            steps: steps(),
            shielding: None,
        }],
        chain(),
        &Default::default(),
        Default::default(),
        None,
        Some(&DamageRequest::default()),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("Li") && err.contains("displacement_energies"),
        "{err}"
    );
    assert!(m.nuclide_data.is_empty(), "nothing should have been loaded");
}
