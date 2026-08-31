//! Nuclear-data uncertainty on `Material.transmute`, end to end.
//!
//! Nothing published carries `covariance.arrow` yet, so the fixture is built:
//! a cached Fe56 directory is copied and the real converter writes the MF=33
//! section into it from the committed ENDF evaluation. The trimmed Fe56 fixture
//! keeps covariance for MT=103, which is `(n,p)` to Mn56 -- the dominant
//! activation channel of irradiated iron, so the one nuclide whose sigma is
//! worth asserting on is exactly the one the evaluation covers.
//!
//! Self-skips when the nuclear-data fixtures are missing, the way every other
//! test that needs them does.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::uncertainty::DataUncertainty;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const FE56_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-026_Fe_056_trimmed.endf.xz");

/// Three groups: thermal, epithermal, fast. Small enough to read, wide enough
/// that `(n,p)` (a threshold reaction) actually fires.
const GROUPS: [f64; 4] = [1.0e-5, 0.625, 1.0e5, 2.0e7];
const FLUX: [f64; 3] = [1.0e12, 5.0e12, 1.0e14];

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

/// A copy of the cached Fe56 directory with `covariance.arrow` written into it.
fn fe56_with_covariance(tmp: &Path) -> Option<PathBuf> {
    let cached = PathBuf::from(yamc_test_cache::nuclide("Fe56")?);
    let dir = tmp.join("Fe56.arrow");
    std::fs::create_dir_all(&dir).expect("mkdir");
    for entry in std::fs::read_dir(&cached).expect("read cached Fe56") {
        let entry = entry.expect("dir entry");
        if entry.path().is_file() {
            std::fs::copy(entry.path(), dir.join(entry.file_name())).expect("copy section");
        }
    }

    let evaluation = tmp.join("fe56.endf");
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &FE56_ENDF[..], &mut raw).expect("fixture decompresses");
    std::fs::write(&evaluation, raw).expect("write evaluation");
    let material = endf::Material::from_file(&evaluation).expect("Fe56 parses");
    assert!(
        yamc_convert::covariance::write_covariance(&material, &dir).expect("covariance writes"),
        "the fixture must carry MF=33 for this test to mean anything"
    );
    Some(dir)
}

fn iron(data: &Path) -> Material {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.density = Some(7.87);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), data.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read Fe56");
    m
}

/// One hour of irradiation, then one hour of cooling.
fn schedule() -> Vec<TransmuteStep> {
    vec![
        TransmuteStep {
            dt: 3600.0,
            irradiation: Some((0, FLUX.iter().sum())),
        },
        TransmuteStep {
            dt: 3600.0,
            irradiation: None,
        },
    ]
}

fn spectra() -> Vec<MultigroupSpectrum> {
    let total: f64 = FLUX.iter().sum();
    vec![MultigroupSpectrum {
        boundaries: GROUPS.to_vec(),
        masses: FLUX.iter().map(|f| f / total).collect(),
        relative_std_dev: None,
    }]
}

fn run(
    material: &mut Material,
    uncertainty: Option<&DataUncertainty>,
) -> yani_transmute::TransmutationResults {
    transmute_material(
        material,
        &spectra(),
        &schedule(),
        chain(),
        &Default::default(),
        Default::default(),
        uncertainty,
    )
    .expect("transmute")
}

fn skip(what: &str) {
    eprintln!("skipping: {what} (nuclear-data fixtures missing)");
}

/// The headline: an irradiated iron foil's Mn56 comes back with a sigma on it.
#[test]
fn mn56_gets_a_nuclear_data_uncertainty() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("mn56_gets_a_nuclear_data_uncertainty");
    };
    let mut material = iron(&dir);
    let id = material.material_id.unwrap_or(0);

    let results = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 42,
            samples: Some(96),
            ..Default::default()
        }),
    );

    let mean = results
        .get_nuclide_density(id, "Mn56", 1)
        .expect("Mn56 is produced");
    let sigma = results
        .get_nuclide_uncertainty(id, "Mn56", 1)
        .expect("uncertainty was requested");

    assert!(mean > 0.0, "Fe56(n,p) must produce some Mn56");
    assert!(sigma > 0.0, "a perturbed (n,p) rate must move Mn56");
    assert!(
        sigma < mean,
        "a relative uncertainty of more than 100% on the dominant channel would \
         mean the fold or the sampler is wrong, not that the evaluation is: \
         mean {mean:e}, sigma {sigma:e}"
    );

    let info = results.uncertainty_info.expect("info is reported");
    assert!(
        info.perturbed.contains("Fe56"),
        "Fe56 carries the covariance, so it must be the perturbed nuclide"
    );
    assert_eq!(info.samples, 96);
}

/// Asking for uncertainty must not move the answer.
///
/// The means come from the nominal pass, which is the same loop it always was;
/// the replicas run beside it and never feed back. Bit-identical, not close.
#[test]
fn the_means_are_bit_identical_with_and_without_uncertainty() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("the_means_are_bit_identical_with_and_without_uncertainty");
    };
    let mut material = iron(&dir);
    let id = material.material_id.unwrap_or(0);

    let plain = run(&mut material, None);
    let with = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 7,
            samples: Some(64),
            ..Default::default()
        }),
    );

    let a = plain
        .get_material(id, 1)
        .expect("nominal step")
        .nuclides
        .clone();
    let b = with.get_material(id, 1).expect("step").nuclides.clone();
    assert_eq!(a.len(), b.len(), "the nuclide sets must match");
    for (name, value) in &a {
        assert_eq!(
            b.get(name),
            Some(value),
            "{name} moved when uncertainty was switched on"
        );
    }
}

/// Off by default: nothing is accumulated and nothing is reported.
#[test]
fn nothing_is_allocated_when_uncertainty_is_not_requested() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("nothing_is_allocated_when_uncertainty_is_not_requested");
    };
    let mut material = iron(&dir);
    let id = material.material_id.unwrap_or(0);

    let results = run(&mut material, None);
    assert!(results.uncertainty.is_empty(), "no ensemble is built");
    assert!(results.uncertainty_info.is_none(), "no report is made");
    assert_eq!(results.get_nuclide_uncertainty(id, "Mn56", 1), None);
}

/// The same seed gives the same sigma, and a different seed does not.
#[test]
fn the_same_seed_reproduces_the_same_uncertainty() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("the_same_seed_reproduces_the_same_uncertainty");
    };
    let mut material = iron(&dir);
    let id = material.material_id.unwrap_or(0);
    let mut at = |seed| {
        run(
            &mut material,
            Some(&DataUncertainty {
                seed,
                samples: Some(64),
                ..Default::default()
            }),
        )
        .get_nuclide_uncertainty(id, "Mn56", 1)
        .expect("uncertainty")
    };

    assert_eq!(at(5), at(5), "same seed, same answer, to the last bit");
    assert_ne!(at(5), at(6), "a different seed must actually resample");
}

/// The initial composition is an input, so it carries no uncertainty.
#[test]
fn the_starting_inventory_has_no_uncertainty() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("the_starting_inventory_has_no_uncertainty");
    };
    let mut material = iron(&dir);
    let id = material.material_id.unwrap_or(0);
    let results = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 1,
            samples: Some(64),
            ..Default::default()
        }),
    );
    assert_eq!(results.get_nuclide_uncertainty(id, "Fe56", 0), Some(0.0));
}

/// A nuclide with no covariance is reported, not silently given a zero sigma.
///
/// The chain reaches nuclides whose directories carry no MF=33 at all, and the
/// difference between "well known" and "nothing published" has to survive to
/// the caller.
#[test]
fn nuclides_without_covariance_are_named_in_the_report() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("nuclides_without_covariance_are_named_in_the_report");
    };
    let mut material = iron(&dir);

    let results = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 1,
            samples: Some(32),
            ..Default::default()
        }),
    );
    let info = results.uncertainty_info.expect("info is reported");

    assert!(
        !info.perturbed.is_empty(),
        "at least Fe56 must have been perturbed"
    );
    assert!(
        info.perturbed
            .intersection(&info.no_covariance_data)
            .next()
            .is_none(),
        "a nuclide cannot be both perturbed and lacking data"
    );
    assert!(
        info.not_perturbed.iter().any(|s| s.contains("half-life")),
        "the sources this does not propagate must be stated: {:?}",
        info.not_perturbed
    );
}

/// The ensemble is kept, so a derived quantity can be evaluated per sample.
#[test]
fn the_per_replica_inventories_are_available() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("the_per_replica_inventories_are_available");
    };
    let mut material = iron(&dir);
    let id = material.material_id.unwrap_or(0);
    let results = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 3,
            samples: Some(48),
            ..Default::default()
        }),
    );

    let inventories = results
        .uncertainty_inventories(id, 1)
        .expect("uncertainty was requested");
    assert_eq!(inventories.len(), 48, "one inventory per replica");

    // The spread of Mn56 over the ensemble must be the sigma that was reported,
    // which is what makes a per-sample derived quantity consistent with the
    // per-nuclide number beside it.
    let values: Vec<f64> = inventories
        .iter()
        .map(|inv| inv.get("Mn56").copied().unwrap_or(0.0))
        .collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let reported = results
        .get_nuclide_uncertainty(id, "Mn56", 1)
        .expect("uncertainty");
    assert!(
        (var.sqrt() - reported).abs() <= 1e-9 * reported.max(1e-30),
        "the ensemble and the reported sigma must be the same number: \
         {} vs {reported}",
        var.sqrt()
    );
}

/// Left to itself the driver stops when the sigmas settle.
#[test]
fn the_adaptive_driver_converges_and_says_so() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("the_adaptive_driver_converges_and_says_so");
    };
    let mut material = iron(&dir);
    let results = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 11,
            samples: None,
            ..Default::default()
        }),
    );
    let info = results.uncertainty_info.expect("info is reported");
    assert!(
        info.samples >= 128,
        "at least the minimum: {}",
        info.samples
    );
    assert!(info.samples <= 1024, "and no more than the cap");
    assert!(
        info.converged,
        "a single well-behaved channel must settle inside the cap"
    );
}

/// Switching the only source off leaves nothing to sample, and says so.
///
/// The baseline for "add a source, watch sigma grow": with every source off the
/// answer must be zero uncertainty and a report naming what was on, not a run
/// that quietly perturbs everything anyway.
#[test]
fn a_source_switched_off_contributes_nothing() {
    use yani_transmute::uncertainty::Source;

    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("a_source_switched_off_contributes_nothing");
    };
    let mut material = iron(&dir);
    let id = material.material_id.unwrap_or(0);

    let with = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 4,
            samples: Some(64),
            sources: vec![Source::CrossSections],
        }),
    );
    let without = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 4,
            samples: Some(64),
            sources: vec![],
        }),
    );

    let on = with
        .get_nuclide_uncertainty(id, "Mn56", 1)
        .expect("requested");
    assert!(on > 0.0, "cross sections on must move Mn56");

    // An EMPTY set means "every implemented source", which is what a
    // default-constructed request does. It is not "none": a request that
    // perturbed nothing by accident would be indistinguishable from an
    // evaluation with no covariance.
    let all = without
        .get_nuclide_uncertainty(id, "Mn56", 1)
        .expect("requested");
    assert_eq!(on, all, "an empty set is every source, not no source");

    let info = with.uncertainty_info.expect("info");
    assert_eq!(info.sources, vec!["cross_sections".to_string()]);
}

/// Sources are independent, so widening the set cannot shrink the answer.
///
/// The property the progressive workflow rests on: each source added is a new
/// independent contribution, so the inventory sigma must be monotonic in the
/// set. With one source implemented this can only check the trivial direction,
/// but it is the assertion the next source has to keep passing.
#[test]
fn adding_a_source_never_decreases_the_uncertainty() {
    use yani_transmute::uncertainty::Source;

    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("adding_a_source_never_decreases_the_uncertainty");
    };
    let mut material = iron(&dir);
    let id = material.material_id.unwrap_or(0);

    let mut sigma = |sources: Vec<Source>| {
        run(
            &mut material,
            Some(&DataUncertainty {
                seed: 8,
                samples: Some(96),
                sources,
            }),
        )
        .get_nuclide_uncertainty(id, "Mn56", 1)
        .expect("requested")
    };

    let one = sigma(vec![Source::CrossSections]);
    let all = sigma(Source::IMPLEMENTED.to_vec());
    assert!(
        all >= one,
        "widening the source set must not shrink sigma: {all:e} < {one:e}"
    );
}

// --- flux spectrum uncertainty (issue #559) -----------------------------------

/// A spectrum given with a per-bin error moves the inventory.
///
/// Needs no covariance at all: the flux is the caller's own input, so this runs
/// against the plain cached Fe56 with only the flux source enabled.
#[test]
fn a_flux_error_moves_the_inventory_on_its_own() {
    use yani_transmute::uncertainty::Source;

    let Some(dir) = yamc_test_cache::nuclide("Fe56") else {
        return skip("a_flux_error_moves_the_inventory_on_its_own");
    };
    let mut material = iron(std::path::Path::new(&dir));
    let id = material.material_id.unwrap_or(0);

    // 10% on every bin.
    let total: f64 = FLUX.iter().sum();
    let spectra = vec![MultigroupSpectrum {
        boundaries: GROUPS.to_vec(),
        masses: FLUX.iter().map(|f| f / total).collect(),
        relative_std_dev: Some(vec![0.10; FLUX.len()]),
    }];

    let results = transmute_material(
        &mut material,
        &spectra,
        &schedule(),
        chain(),
        &Default::default(),
        Default::default(),
        Some(&DataUncertainty {
            seed: 77,
            samples: Some(256),
            sources: vec![Source::FluxSpectrum],
        }),
    )
    .expect("transmute");

    let mean = results.get_nuclide_density(id, "Mn56", 1).expect("Mn56");
    let sigma = results
        .get_nuclide_uncertainty(id, "Mn56", 1)
        .expect("requested");
    assert!(sigma > 0.0, "a 10% flux error must move Mn56");

    // Mn56 production is linear in the flux and it is near saturation after an
    // hour, so its relative spread should track the flux's 10% rather than
    // being some unrelated size. A wide band, because the burnup and the decay
    // during the step both damp it.
    let rel = sigma / mean;
    assert!(
        (0.02..0.20).contains(&rel),
        "a 10% flux error should give Mn56 a spread of that order, got {:.3}%",
        100.0 * rel
    );

    let info = results.uncertainty_info.expect("info");
    assert_eq!(info.spectra_with_flux_sigma, 1);
    assert_eq!(info.spectra_without_flux_sigma, 0);
    assert!(info.flux_bins_sampled > 0);
}

/// A spectrum with no stated error contributes nothing, and is reported.
///
/// The FISPACT reference-spectra case: the user has a flux and no error for it.
/// That must not read as a flux known exactly.
#[test]
fn a_spectrum_without_an_error_is_reported_not_assumed_exact() {
    use yani_transmute::uncertainty::Source;

    let Some(dir) = yamc_test_cache::nuclide("Fe56") else {
        return skip("a_spectrum_without_an_error_is_reported_not_assumed_exact");
    };
    let mut material = iron(std::path::Path::new(&dir));
    let id = material.material_id.unwrap_or(0);

    let results = transmute_material(
        &mut material,
        &spectra(), // no relative_std_dev
        &schedule(),
        chain(),
        &Default::default(),
        Default::default(),
        Some(&DataUncertainty {
            seed: 5,
            samples: Some(32),
            sources: vec![Source::FluxSpectrum],
        }),
    )
    .expect("transmute");

    let info = results.uncertainty_info.clone().expect("info");
    assert_eq!(info.spectra_without_flux_sigma, 1);
    assert_eq!(info.spectra_with_flux_sigma, 0);
    assert!(
        info.has_gaps(),
        "a spectrum with no stated error is a gap, not a certainty"
    );
    assert_eq!(
        results.get_nuclide_uncertainty(id, "Mn56", 1),
        Some(0.0),
        "nothing to sample means zero, and the report says why"
    );
}

/// A bigger flux error gives a bigger inventory error.
#[test]
fn the_inventory_spread_scales_with_the_flux_error() {
    use yani_transmute::uncertainty::Source;

    let Some(dir) = yamc_test_cache::nuclide("Fe56") else {
        return skip("the_inventory_spread_scales_with_the_flux_error");
    };
    let mut material = iron(std::path::Path::new(&dir));
    let id = material.material_id.unwrap_or(0);
    let total: f64 = FLUX.iter().sum();

    let mut sigma_at = |rel: f64| {
        let spectra = vec![MultigroupSpectrum {
            boundaries: GROUPS.to_vec(),
            masses: FLUX.iter().map(|f| f / total).collect(),
            relative_std_dev: Some(vec![rel; FLUX.len()]),
        }];
        transmute_material(
            &mut material,
            &spectra,
            &schedule(),
            chain(),
            &Default::default(),
            Default::default(),
            Some(&DataUncertainty {
                seed: 31,
                samples: Some(256),
                sources: vec![Source::FluxSpectrum],
            }),
        )
        .expect("transmute")
        .get_nuclide_uncertainty(id, "Mn56", 1)
        .expect("requested")
    };

    let small = sigma_at(0.05);
    let large = sigma_at(0.15);
    assert!(
        large > small * 2.0,
        "tripling the flux error should roughly triple the inventory spread, \
         got {small:e} -> {large:e}"
    );
}
