//! Nuclear-data uncertainty on `Material.transmute`, end to end.
//!
//! The fixture is built rather than downloaded: a cached Fe56 directory is
//! copied and the real converter writes the MF=33 section into it from the
//! committed ENDF evaluation. The published libraries do now carry
//! `covariance.arrow`, but a test that fetched it would assert on whatever the
//! CDN currently holds and would need a network, so the section is produced
//! here from an evaluation that is in the tree. The trimmed Fe56 fixture keeps
//! covariance for MT=103, which is `(n,p)` to Mn56, the dominant activation
//! channel of irradiated iron, so the one nuclide whose sigma is worth
//! asserting on is exactly the one the evaluation covers.
//!
//! Self-skips when the nuclear-data fixtures are missing, the way every other
//! test that needs them does.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod common;
use common::fe56_with_covariance;

use yamc_materials::Material;
use yani_transmute::uncertainty::DataUncertainty;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

/// Three groups: thermal, epithermal, fast. Small enough to read, wide enough
/// that `(n,p)` (a threshold reaction) actually fires.
const GROUPS: [f64; 4] = [1.0e-5, 0.625, 1.0e5, 2.0e7];
const FLUX: [f64; 3] = [1.0e12, 5.0e12, 1.0e14];

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
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
        flux_error: None,
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

    let info = results
        .uncertainty_info
        .get(&0)
        .cloned()
        .expect("info is reported");
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
    assert!(results.uncertainty_info.is_empty(), "no report is made");
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
    let info = results
        .uncertainty_info
        .get(&0)
        .cloned()
        .expect("info is reported");

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
    for source in [
        "fission yield",
        "isomeric branching",
        "cross-material covariance",
    ] {
        assert!(
            info.not_perturbed.iter().any(|s| s.contains(source)),
            "{source} must be stated as not propagated: {:?}",
            info.not_perturbed
        );
    }
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
    let info = results
        .uncertainty_info
        .get(&0)
        .cloned()
        .expect("info is reported");
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
            attribution: false,
        }),
    );
    let without = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 4,
            samples: Some(64),
            sources: vec![],
            attribution: false,
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

    let info = with.uncertainty_info.get(&0).cloned().expect("info");
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
                attribution: false,
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
        flux_error: Some(yani_transmute::flux_uncertainty::FluxError::RelativeStdDev(
            vec![0.10; FLUX.len()],
        )),
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
            attribution: false,
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

    let info = results.uncertainty_info.get(&0).cloned().expect("info");
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
            attribution: false,
        }),
    )
    .expect("transmute");

    let info = results.uncertainty_info.get(&0).cloned().expect("info");
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

/// A source switched off is held at nominal and the report names it, the way
/// it names an input no source can sample. Switched back on with something to
/// act on, the entry goes.
#[test]
fn a_source_switched_off_is_named_as_held() {
    use yani_transmute::uncertainty::Source;

    let Some(dir) = yamc_test_cache::nuclide("Fe56") else {
        return skip("a_source_switched_off_is_named_as_held");
    };
    let total: f64 = FLUX.iter().sum();
    let with_sigma = vec![MultigroupSpectrum {
        boundaries: GROUPS.to_vec(),
        masses: FLUX.iter().map(|f| f / total).collect(),
        flux_error: Some(yani_transmute::flux_uncertainty::FluxError::RelativeStdDev(
            vec![0.05; FLUX.len()],
        )),
    }];
    let held = |spectra: &[MultigroupSpectrum], sources: Vec<Source>| {
        let mut material = iron(std::path::Path::new(&dir));
        transmute_material(
            &mut material,
            spectra,
            &schedule(),
            chain(),
            &Default::default(),
            Default::default(),
            Some(&DataUncertainty {
                seed: 7,
                samples: Some(8),
                sources,
                attribution: false,
            }),
        )
        .expect("transmute")
        .uncertainty_info
        .get(&0)
        .cloned()
        .expect("info")
        .not_perturbed
    };

    let narrowed = held(&with_sigma, vec![Source::HalfLife]);
    for off in [
        "activation cross section (MF=33)",
        "flux spectrum",
        "decay energy",
    ] {
        assert!(
            narrowed.iter().any(|s| s == off),
            "{off:?} missing from {narrowed:?}"
        );
    }
    assert!(!narrowed.iter().any(|s| s == "half-life"));

    let everything = held(&with_sigma, vec![]);
    for on in [
        "activation cross section (MF=33)",
        "flux spectrum",
        "half-life",
        "decay energy",
    ] {
        assert!(
            !everything.iter().any(|s| s == on),
            "{on:?} was sampled but listed as held: {everything:?}"
        );
    }

    // A spectrum with no sigma is used as given even with its source on.
    let without_sigma = held(&spectra(), vec![Source::FluxSpectrum]);
    assert!(without_sigma.iter().any(|s| s == "flux spectrum"));

    // One spectrum with a sigma and one without: the flux was partly sampled,
    // so the entry says only the spectra without a sigma were held.
    let mixed: Vec<MultigroupSpectrum> = with_sigma.iter().cloned().chain(spectra()).collect();
    let partial = held(&mixed, vec![Source::FluxSpectrum]);
    assert!(!partial.iter().any(|s| s == "flux spectrum"), "{partial:?}");
    assert!(
        partial
            .iter()
            .any(|s| s == "flux spectrum (spectra without a sigma only)"),
        "{partial:?}"
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
            flux_error: Some(yani_transmute::flux_uncertainty::FluxError::RelativeStdDev(
                vec![rel; FLUX.len()],
            )),
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
                attribution: false,
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

/// Attribution with the cross-section source alone: the first-order
/// contribution of Fe56's evaluation, every channel with its correlations,
/// must account for Mn56's variance, since Mn56 is made from Fe56 alone. And
/// each of Fe56's channels is reported on its own too.
#[test]
fn the_cross_section_attribution_names_the_evaluation() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("the_cross_section_attribution_names_the_evaluation");
    };
    let mut material = iron(&dir);
    let id = material.material_id.unwrap_or(0);
    let results = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 1,
            samples: Some(512),
            sources: vec![yani_transmute::uncertainty::Source::CrossSections],
            attribution: true,
        }),
    );
    let b = results
        .uncertainty_breakdown(id, "Mn56", 1)
        .expect("attribution was asked for");
    assert!(b.variance > 0.0);
    // One source: it is the total.
    assert_eq!(b.by_source["cross_sections"], b.variance);
    let block = b
        .contributors
        .iter()
        .find(|(s, n, r, _)| s == "cross_sections" && n == "Fe56" && r.is_none())
        .expect("Fe56's evaluation contributes")
        .3;
    assert!(
        (block / b.variance - 1.0).abs() < 0.15,
        "first order {block:.3e} against the resampled {:.3e}",
        b.variance
    );
    assert!(
        b.contributors
            .iter()
            .any(|(s, n, r, _)| s == "cross_sections"
                && n == "Fe56"
                && r.as_deref() == Some("(n,p)")),
        "the channel making Mn56 is named: {:?}",
        b.contributors
    );
}

/// Fe56 `(n,p)` shows the zero-variance rule on a real evaluation. Its MT=103
/// grid runs from 1e-5 eV, but the variance is zero on every interval below
/// 4.3 MeV, so the sliver of rate between the 2.97 MeV threshold and there is
/// not covered and the share falls just short of one, where spanning the grid
/// alone would call it fully covered.
#[test]
fn fe56_np_is_not_covered_where_its_variance_is_zero() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("fe56_np_is_not_covered_where_its_variance_is_zero");
    };
    let mut material = iron(&dir);
    let results = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 1,
            samples: Some(32),
            sources: vec![yani_transmute::uncertainty::Source::CrossSections],
            ..Default::default()
        }),
    );
    let info = results
        .uncertainty_info
        .get(&0)
        .cloned()
        .expect("info is reported");

    let np = info.rate_fraction_covered[&("Fe56".to_string(), "(n,p)".to_string())];
    assert!(
        np > 0.9999 && np < 1.0,
        "Fe56 (n,p) is covered above 4.3 MeV only, which is nearly all of its rate: {np}"
    );
}

/// On a dilute collapse the fold's partial rates and the rate they are divided
/// by are the same integral, so no channel may report partials above or below
/// its rate, and every share the fold reports lies in [0, 1].
#[test]
fn a_dilute_fold_reports_no_partials_off_the_rate() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("a_dilute_fold_reports_no_partials_off_the_rate");
    };
    let mut material = iron(&dir);
    let results = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 1,
            samples: Some(32),
            sources: vec![yani_transmute::uncertainty::Source::CrossSections],
            ..Default::default()
        }),
    );
    let info = results
        .uncertainty_info
        .get(&0)
        .cloned()
        .expect("info is reported");

    assert!(
        info.partials_above_rate.is_empty(),
        "a dilute fold is consistent: {:?}",
        info.partials_above_rate
    );
    assert!(
        info.partials_below_rate.is_empty(),
        "a dilute fold is consistent: {:?}",
        info.partials_below_rate
    );
    assert!(
        info.rate_fraction_covered_total.is_some(),
        "a dilute run drove production, so it has a total"
    );
    for (key, fraction) in &info.rate_fraction_covered {
        assert!(
            (0.0..=1.0).contains(fraction),
            "{key:?} reads {fraction}, which is not a share"
        );
    }
}
