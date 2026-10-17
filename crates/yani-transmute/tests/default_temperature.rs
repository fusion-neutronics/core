//! A material with no temperature transmutes where it transports: at the only
//! temperature its data carries, else at the 294 K default.
//!
//! Transport has resolved an unset temperature that way since core #355, and
//! the Python constructors document it. The flux-given transmutation did not:
//! it read each nuclide at the lowest temperature it had loaded, so an
//! unlabelled material activated at 250 K against ENDF/B-VIII.1, whose ladder
//! starts there (FENDL-3.2d, TENDL-2017 and TENDL-2025 start there too). Now
//! it resolves through the same `Material::try_resolve_temperature`, loads
//! what that picks, and refuses a default the data cannot serve rather than
//! reading somewhere else.
//!
//! Data: endf-b8.1 Fe56 in the fixture cache, which carries 250, 294, 600,
//! 900, 1200 and 2500 K. Self-skips when absent.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

/// Fe56 with its data read from `cache`, at `temperature` when given. With
/// none the load keeps every temperature the file carries, as it does for an
/// unlabelled material on any path.
fn fe56(cache: &str, temperature: Option<&str>) -> Material {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .expect("Fe56 material");
    if let Some(t) = temperature {
        m.set_temperature(t);
    }
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), cache.to_string())]),
        None,
    )
    .expect("read Fe56");
    m
}

/// A day at 1e14 n/cm2/s with a 1/E spectrum on CCFE-709 up to 20 MeV, so the
/// capture rate reads Fe56's resonances group by group and moves with their
/// Doppler broadening.
fn irradiate(
    m: &mut Material,
    chain: Arc<HashMap<String, yani::ChainNuclide>>,
) -> Result<yani_transmute::TransmutationResults, String> {
    let boundaries: Vec<f64> = yamc_nuclide::group_structures::get_group_structure("CCFE-709")
        .expect("CCFE-709")
        .iter()
        .copied()
        .take_while(|&e| e <= 2.0e7)
        .collect();
    let masses: Vec<f64> = boundaries.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let total: f64 = masses.iter().sum();
    let spectra = [MultigroupSpectrum {
        boundaries,
        masses: masses.iter().map(|m| m / total).collect(),
        flux_error: None,
    }];
    let steps = [TransmuteStep {
        dt: 86_400.0,
        irradiation: Some((0, 1.0e14)),
    }];
    transmute_material(
        m,
        &spectra,
        &steps,
        chain,
        &yani::BranchTable::new(),
        Default::default(),
        None,
    )
    .map_err(|e| e.to_string())
}

/// Fe56's capture rate over the step, per edge.
fn capture(results: &yani_transmute::TransmutationResults) -> Vec<(Option<String>, f64)> {
    results
        .get_reaction_rates(0, 0)
        .and_then(|r| r.get("Fe56"))
        .and_then(|r| r.get("(n,gamma)"))
        .cloned()
        .expect("Fe56 (n,gamma) was solved")
}

#[test]
fn an_unset_temperature_transmutes_at_294_k_like_transport() {
    let Some(cache) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("an_unset_temperature_transmutes_at_294_k_like_transport: skip (Fe56 missing)");
        return;
    };
    let chain = chain();

    let mut unset = fe56(&cache, None);
    // The state the bug needs: more than one temperature loaded, the lowest
    // of them below 294 K.
    let loaded = unset.nuclide_data["Fe56"].loaded_temperatures.clone();
    assert!(
        loaded.len() > 1 && loaded[0] == "250" && loaded.contains(&"294".to_string()),
        "needs the fixture's 250 K and 294 K both loaded, got {loaded:?}"
    );

    let at_294 = capture(&irradiate(&mut fe56(&cache, Some("294")), Arc::clone(&chain)).unwrap());
    let at_250 = capture(&irradiate(&mut fe56(&cache, Some("250")), Arc::clone(&chain)).unwrap());
    // Otherwise this test could not tell the two temperatures apart.
    assert_ne!(
        at_294, at_250,
        "the spectrum must make Fe56 capture differ between 250 K and 294 K"
    );

    let results = irradiate(&mut unset, chain).expect("an unset temperature must transmute");
    assert_eq!(
        capture(&results),
        at_294,
        "an unset temperature must collapse at 294 K, as transport runs it (250 K gives {at_250:?})"
    );
    // Adopted, as transport adopts it, so what the results carry says where
    // the rates were read.
    assert_eq!(unset.temperature(), "294");
    assert_eq!(results.get_material(0, 0).unwrap().temperature(), "294");
}

#[test]
fn an_unset_temperature_takes_the_only_one_the_data_carries() {
    let Some(cache) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("an_unset_temperature_takes_the_only_one_the_data_carries: skip (Fe56 missing)");
        return;
    };
    let chain = chain();

    // A single-temperature library, as the resolution sees one: the nuclide
    // advertises 600 K alone. The loaded rungs are left as the file gave them,
    // so a lowest-loaded rule would still find 250 K.
    let mut single = fe56(&cache, None);
    let mut nuclide = (*single.nuclide_data["Fe56"]).clone();
    nuclide.available_temperatures = vec!["600".to_string()];
    single
        .nuclide_data
        .insert("Fe56".to_string(), Arc::new(nuclide));

    let at_600 = capture(&irradiate(&mut fe56(&cache, Some("600")), Arc::clone(&chain)).unwrap());
    let results = irradiate(&mut single, chain).expect("one temperature must transmute");
    assert_eq!(
        capture(&results),
        at_600,
        "data carrying one temperature must be read at it"
    );
    assert_eq!(single.temperature(), "600");
}

#[test]
fn an_unset_temperature_the_data_cannot_serve_is_refused() {
    let Some(cache) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("an_unset_temperature_the_data_cannot_serve_is_refused: skip (Fe56 missing)");
        return;
    };

    // No published library lacks 294 K, so narrow the advertised ladder to
    // one wholly above it: 294 K is then neither listed nor bracketed, and
    // there is more than one temperature to choose from. The 250 K rung is
    // still loaded, which is what the run used to read without a word.
    let mut material = fe56(&cache, None);
    let mut nuclide = (*material.nuclide_data["Fe56"]).clone();
    nuclide.available_temperatures = vec!["600".to_string(), "900".to_string()];
    material
        .nuclide_data
        .insert("Fe56".to_string(), Arc::new(nuclide));

    let Err(err) = irradiate(&mut material, chain()) else {
        panic!("a default the data cannot serve must be refused, not read elsewhere");
    };
    assert!(
        err.contains("294") && err.contains("Fe56"),
        "the error must name the default and the nuclide that cannot serve it: {err}"
    );
    assert!(
        material.temperature().is_empty(),
        "a refused default must not be adopted"
    );
}
