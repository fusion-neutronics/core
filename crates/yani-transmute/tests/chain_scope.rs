//! A chain that drives none of the material's nuclides is refused.
//!
//! Chains are routinely scoped to the nuclides they were built from, because
//! walking a whole library to activate one foil is wasted work. Scoping them
//! makes them substitutable by mistake: two conversions sharing an output path
//! leave the second one's chain behind under a name that still says the first,
//! and the material that name refers to then solves against reactions whose
//! parents it does not contain.
//!
//! Nothing about that fails. Every step runs, no rate is negative, and the
//! composition comes back exactly as it went in, so the mistake is a decay heat
//! of precisely zero discovered much later with nothing to point at.
//!
//! Needs no nuclear data for the refusal case: it is checked before any cross
//! section is read.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

/// A chain that can be driven from `parent` and from nothing else.
///
/// Built here rather than loaded, because the point is what the chain does NOT
/// contain and a published chain contains almost everything. `Fe57` is present
/// as a product so the chain is not trivially empty, and carries no reactions
/// of its own, which is what makes it a product rather than a parent.
fn chain_driven_by(parent: &str) -> Arc<HashMap<String, yani::ChainNuclide>> {
    let nuclide = |name: &str, reactions: Vec<yani::ChainReaction>| yani::ChainNuclide {
        name: name.to_string(),
        half_life: None,
        half_life_uncertainty: None,
        decay_energy: 0.0,
        decay_energy_uncertainty: None,
        reactions,
        decays: Vec::new(),
        fission_yields: None,
        sources: Vec::new(),
    };
    let capture = yani::ChainReaction {
        kind: "(n,gamma)".to_string(),
        target: Some("Fe57".to_string()),
        branching: 1.0,
        q_value: Some(0.0),
    };
    Arc::new(HashMap::from([
        (parent.to_string(), nuclide(parent, vec![capture])),
        ("Fe57".to_string(), nuclide("Fe57", Vec::new())),
    ]))
}

fn material(nuclide: &str, density: f64) -> Material {
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    m.density = Some(density);
    m.set_temperature("294");
    m
}

fn spectrum() -> MultigroupSpectrum {
    MultigroupSpectrum {
        boundaries: vec![1.0e-5, 1.0e6, 2.0e7],
        masses: vec![0.5, 0.5],
        relative_std_dev: None,
    }
}

fn run(mut m: Material, driver: &str, irradiation: Option<(usize, f64)>) -> Result<(), String> {
    let steps = vec![TransmuteStep {
        dt: 3600.0,
        irradiation,
    }];
    transmute_material(
        &mut m,
        &[spectrum()],
        &steps,
        chain_driven_by(driver),
        &Default::default(),
        Default::default(),
        None,
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

#[test]
fn a_chain_with_no_parent_in_the_material_is_refused_rather_than_solving_to_zero() {
    // Fe57 IS in this chain, as the product of Fe56 capture, and carries no
    // reactions of its own. So the older membership check passes it and there
    // is still nothing here to drive: this is the gap between "appears in the
    // chain" and "can be driven from".
    let err = run(material("Fe57", 7.87), "Fe56", Some((0, 1.0e14)))
        .expect_err("a chain that drives nothing must be refused");
    assert!(
        err.contains("drives none of the material's nuclides"),
        "the error must say what is wrong: {err}"
    );
    // Both halves have to be named. Which chain was loaded is exactly what the
    // silent-zero version of this could not tell anyone.
    assert!(
        err.contains("Fe57"),
        "must name what the material holds: {err}"
    );
    assert!(
        err.contains("chain has reactions for"),
        "must name what the chain covers: {err}"
    );
}

#[test]
fn a_decay_only_schedule_is_allowed_against_any_chain() {
    // Nothing is irradiated, so no reaction was ever going to be driven and a
    // chain that cannot drive this material is the right chain for it. Refusing
    // here would break every cooldown of an already-activated inventory.
    run(material("Fe57", 7.87), "Fe56", None).expect("a decay-only schedule needs no parents");
}

#[test]
fn a_chain_that_does_drive_the_material_still_runs() {
    run(material("Fe56", 7.87), "Fe56", Some((0, 1.0e14))).expect("iron against an iron chain");
}
