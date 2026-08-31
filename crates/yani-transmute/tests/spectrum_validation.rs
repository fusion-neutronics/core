//! A spectrum that cannot be integrated is refused, not integrated anyway.
//!
//! `Histogram::new` rejected boundaries that were not strictly ascending with
//! `w[1] <= w[0]`, which a NaN passes: every comparison against a NaN is false.
//! Such a boundary reached the multigroup collapse, gave a NaN group average, a
//! NaN reaction rate and an inventory of NaNs, with no error anywhere along the
//! way. Issue #576.
//!
//! It is also the only input under which the collapse's point set could depend
//! on anything but an ascending grid, which is what finding 1a's bisection
//! rests on -- so it is checked here rather than assumed.
//!
//! Needs no nuclear data: every case is refused before a cross section is read.

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

fn iron() -> Material {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.density = Some(7.87);
    m.set_temperature("294");
    m
}

fn run(spectrum: MultigroupSpectrum) -> Result<(), String> {
    let steps = vec![TransmuteStep {
        dt: 3600.0,
        irradiation: Some((0, 1.0e14)),
    }];
    transmute_material(
        &mut iron(),
        &[spectrum],
        &steps,
        chain(),
        &Default::default(),
        Default::default(),
        None,
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

fn spectrum(boundaries: Vec<f64>, masses: Vec<f64>) -> MultigroupSpectrum {
    MultigroupSpectrum {
        boundaries,
        masses,
        relative_std_dev: None,
    }
}

#[test]
fn a_nan_boundary_is_refused_rather_than_producing_a_nan_inventory() {
    let err = run(spectrum(vec![1.0e-5, f64::NAN, 2.0e7], vec![0.5, 0.5]))
        .expect_err("a NaN boundary must be refused");
    assert!(
        err.contains("not a finite energy"),
        "the error must name the boundary as the problem, got {err:?}"
    );
}

#[test]
fn an_infinite_boundary_is_refused_too() {
    let err = run(spectrum(vec![1.0e-5, 1.0e5, f64::INFINITY], vec![0.5, 0.5]))
        .expect_err("an infinite boundary must be refused");
    assert!(err.contains("not a finite energy"), "got {err:?}");
}

#[test]
fn a_descending_boundary_pair_is_refused_and_named() {
    let err = run(spectrum(vec![1.0e-5, 2.0e7, 1.0e5], vec![0.5, 0.5]))
        .expect_err("boundaries that go backwards must be refused");
    assert!(
        err.contains("ascend strictly"),
        "the error must say what is wrong, got {err:?}"
    );
}

#[test]
fn a_repeated_boundary_is_refused() {
    // A zero-width group is not integrable and the collapse silently returned
    // 0.0 for it, which reads as "this energy range contributes nothing".
    let err = run(spectrum(
        vec![1.0e-5, 1.0e5, 1.0e5, 2.0e7],
        vec![0.3, 0.3, 0.4],
    ))
    .expect_err("a repeated boundary must be refused");
    assert!(err.contains("ascend strictly"), "got {err:?}");
}

#[test]
fn a_nan_mass_is_refused() {
    let err = run(spectrum(vec![1.0e-5, 1.0e5, 2.0e7], vec![0.5, f64::NAN]))
        .expect_err("a NaN mass must be refused");
    assert!(err.contains("finite and non-negative"), "got {err:?}");
}

#[test]
fn an_ordinary_spectrum_still_passes_validation() {
    // Whatever it goes on to do without nuclear data loaded, it must not fail
    // in the validation above: a guard that refuses valid input is worse than
    // no guard.
    let ok = run(spectrum(
        vec![1.0e-5, 0.625, 1.0e5, 2.0e7],
        vec![0.1, 0.3, 0.6],
    ));
    if let Err(e) = &ok {
        assert!(
            !e.contains("finite") && !e.contains("ascend"),
            "a valid spectrum was refused by validation: {e}"
        );
    }
}
