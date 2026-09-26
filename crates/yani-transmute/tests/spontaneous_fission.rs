//! A spontaneous-fission emitter decays at its full decay constant.
//!
//! ENDF/B-VIII.1 gives Cf252 two modes: alpha to Cm248 at 0.96908 and
//! spontaneous fission at 0.03092. The chain file writes the `sf` target as
//! Cf252 itself, because fission moves neither Z nor A in the mode table, and
//! read as an edge that put 3.1% of the decay constant back on Cf252's own
//! diagonal. One half-life left 0.5108 of it rather than 0.5, and Cf254, 99.69%
//! sf, barely decayed at all.
//!
//! The chain carries no spontaneous-fission product yields, so the fission
//! branch's atoms leave it. What has to hold is that the parent follows
//! `N0 exp(-lambda t)`, the alpha branch still feeds Cm248, an isomer that
//! fissions does not feed its ground state, and a nuclide with no fission mode
//! comes out as it always did.
//!
//! Needs no nuclear data: nothing is irradiated.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const N0: f64 = 1.0e-3;

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

fn lambda(chain: &HashMap<String, yani::ChainNuclide>, name: &str) -> f64 {
    std::f64::consts::LN_2 / chain[name].half_life.expect("unstable")
}

/// Densities after each cooling step of `durations`, starting from `N0` of
/// `parent` alone.
fn cool(parent: &str, durations: &[f64]) -> Vec<HashMap<String, f64>> {
    let mut material = Material::new(
        HashMap::from([(parent.to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    material.nuclides.insert(parent.to_string(), N0);
    material.set_temperature("294");
    // Never folded: a decay-only schedule reads no cross section.
    let spectrum = MultigroupSpectrum {
        boundaries: vec![1.0e-5, 2.0e7],
        masses: vec![1.0],
        flux_error: None,
    };
    let steps: Vec<TransmuteStep> = durations
        .iter()
        .map(|&dt| TransmuteStep {
            dt,
            irradiation: None,
        })
        .collect();
    let results = transmute_material(
        &mut material,
        &[spectrum],
        &steps,
        chain(),
        &Default::default(),
        Default::default(),
        None,
    )
    .expect("transmute");
    (1..=durations.len())
        .map(|step| {
            results
                .get_material(0, step)
                .expect("a state per step")
                .nuclides
                .clone()
        })
        .collect()
}

fn relative(got: f64, want: f64) -> f64 {
    ((got - want) / want).abs()
}

#[test]
fn cf252_is_removed_at_its_full_decay_constant() {
    let chain = chain();
    let l1 = lambda(&chain, "Cf252");
    let l2 = lambda(&chain, "Cm248");
    let half_life = std::f64::consts::LN_2 / l1;
    let durations = [half_life, 2.0 * half_life];
    let states = cool("Cf252", &durations);

    let alpha = chain["Cf252"]
        .decays
        .iter()
        .find(|d| d.kind == "alpha")
        .expect("Cf252 alpha")
        .branching;
    let mut t = 0.0;
    for (state, dt) in states.iter().zip(durations) {
        t += dt;
        let want = N0 * (-l1 * t).exp();
        let got = state["Cf252"];
        assert!(
            relative(got, want) < 1e-12,
            "Cf252 at t = {t:e} s: {got:e}, want N0 exp(-lambda t) = {want:e}"
        );
        // Two-member Bateman: the alpha branch alone feeds Cm248.
        let cm248 = alpha * l1 * N0 / (l2 - l1) * ((-l1 * t).exp() - (-l2 * t).exp());
        assert!(
            relative(state["Cm248"], cm248) < 1e-12,
            "Cm248 at t = {t:e} s: {:e}, want {cm248:e}",
            state["Cm248"]
        );
    }
}

#[test]
fn an_isomer_that_fissions_does_not_feed_its_ground_state() {
    // Am242_m2 is a fission isomer: 99.995% sf, which the chain file writes as
    // going to Am242. Ten half-lives leave essentially none of it, and none of
    // that may have become Am242.
    let chain = chain();
    let half_life = chain["Am242_m2"].half_life.unwrap();
    let state = &cool("Am242_m2", &[10.0 * half_life])[0];
    let want = N0 * 2f64.powi(-10);
    assert!(relative(state["Am242_m2"], want) < 1e-12, "{state:?}");
    assert_eq!(
        state.get("Am242").copied().unwrap_or(0.0),
        0.0,
        "spontaneous fission makes no Am242: {state:?}"
    );
}

#[test]
fn a_nuclide_with_no_fission_mode_decays_as_before() {
    let chain = chain();
    let half_life = chain["Co60"].half_life.unwrap();
    let state = &cool("Co60", &[half_life])[0];
    assert!(relative(state["Co60"], 0.5 * N0) < 1e-12, "{state:?}");
    assert!(relative(state["Ni60"], 0.5 * N0) < 1e-12, "{state:?}");
}
