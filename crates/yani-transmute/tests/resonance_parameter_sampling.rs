//! Resonance parameters sampled per replica, end to end.
//!
//! The fixture is the cached ENDF/B-VIII.1 W186 directory with its
//! `covariance.arrow` and `resonance_parameters.arrow` written by the real
//! converter from the committed MF=2 and MF=32 evaluation, which has no
//! MF=33: the resonance range's uncertainty is all there is, once as the
//! first-order rows the converter derives (`subsection_idx = -1`) and once as
//! the parameters. A run without `resonance_parameters.arrow` samples the
//! rows; a run with it samples the parameters and leaves the rows out.
//!
//! The observable is W187 after a one-second irradiation, which is the
//! capture rate times the fluence to well below a sampling error, so its
//! relative sigma is the capture rate's.
//!
//! Self-skips when the nuclear-data fixtures are missing, the way every other
//! test that needs them does.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::resonance_rates::ResonanceMethod;
use yani_transmute::uncertainty::DataUncertainty;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const W186_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-074_W_186_mf2_mf32.endf.xz");

/// What `resonance_parameters.arrow` holds in a fixture directory.
#[derive(Clone, Copy, PartialEq)]
enum Parameters {
    /// No file: the first-order rows alone.
    Absent,
    /// The evaluation's MF=2 and MF=32.
    Evaluated,
    /// MF=32 cut off half way, so it does not parse.
    Truncated,
}

/// A copy of the cached W186 directory with the converter's
/// `covariance.arrow` from the committed evaluation, and its
/// `resonance_parameters.arrow` as `parameters` says, or `None` when the
/// nuclear-data fixtures are missing.
fn w186(tmp: &Path, parameters: Parameters) -> Option<PathBuf> {
    let cached = PathBuf::from(yamc_test_cache::nuclide("W186")?);
    let dir = tmp.join("W186.arrow");
    std::fs::create_dir_all(&dir).expect("mkdir");
    for entry in std::fs::read_dir(&cached).expect("read cached W186") {
        let entry = entry.expect("dir entry");
        let name = entry.file_name();
        if entry.path().is_file() && name != "resonance_parameters.arrow" {
            std::fs::copy(entry.path(), dir.join(name)).expect("copy section");
        }
    }
    let evaluation = tmp.join("w186.endf");
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &W186_ENDF[..], &mut raw).expect("fixture decompresses");
    std::fs::write(&evaluation, raw).expect("write evaluation");
    let mut material = endf::Material::from_file(&evaluation).expect("W186 parses");
    assert!(
        yamc_convert::covariance::write_covariance(&material, &dir).expect("covariance writes"),
        "the fixture's MF=32 must give first-order rows"
    );
    match parameters {
        Parameters::Absent => {}
        Parameters::Evaluated => {
            assert!(
                yamc_convert::resonance_parameters::write_resonance_parameters(&material, &dir)
                    .expect("resonance parameters write")
            );
        }
        Parameters::Truncated => {
            let text = material.section_text.get_mut(&(32, 151)).expect("MF=32");
            let lines: Vec<&str> = text.lines().collect();
            *text = lines[..lines.len() / 2].join("\n") + "\n";
            assert!(
                yamc_convert::resonance_parameters::write_resonance_parameters(&material, &dir)
                    .expect("resonance parameters write")
            );
        }
    }
    Some(dir)
}

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

fn tungsten(data: &Path, temperature: &str) -> Material {
    let mut m = Material::new(
        HashMap::from([("W186".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("W186 material");
    m.density = Some(19.3);
    m.set_temperature(temperature);
    m.read_nuclear_data(
        &HashMap::from([("W186".to_string(), data.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read W186");
    m
}

/// A `1/E` flux on 80 logarithmic groups from 1e-5 eV to 20 MeV: smooth, and
/// most of a capture rate from the resolved range.
fn spectra() -> Vec<MultigroupSpectrum> {
    let n = 80;
    let (lo, hi): (f64, f64) = (1.0e-5, 2.0e7);
    let boundaries: Vec<f64> = (0..=n)
        .map(|i| lo * (hi / lo).powf(i as f64 / n as f64))
        .collect();
    // Equal lethargy widths, so a 1/E flux puts the same mass in each.
    vec![MultigroupSpectrum {
        masses: vec![1.0 / n as f64; n],
        boundaries,
        flux_error: None,
    }]
}

/// One second at 1e10 n/cm2/s: W187 is the capture rate times the fluence.
fn schedule() -> Vec<TransmuteStep> {
    vec![TransmuteStep {
        dt: 1.0,
        irradiation: Some((0, 1.0e10)),
    }]
}

fn run(material: &mut Material, samples: usize) -> yani_transmute::TransmutationResults {
    transmute_material(
        material,
        &spectra(),
        &schedule(),
        chain(),
        &Default::default(),
        Default::default(),
        Some(&DataUncertainty {
            seed: 20260,
            samples: Some(samples),
            sources: vec![yani_transmute::uncertainty::Source::CrossSections],
            ..Default::default()
        }),
    )
    .expect("transmute")
}

/// W187 in every replica, in replica order.
fn w187(results: &yani_transmute::TransmutationResults, id: u32) -> Vec<f64> {
    results
        .uncertainty_inventories(id, 1)
        .expect("uncertainty was requested")
        .iter()
        .map(|inventory| inventory.get("W187").copied().unwrap_or(0.0))
        .collect()
}

fn mean_and_sigma(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

fn method(results: &yani_transmute::TransmutationResults) -> ResonanceMethod {
    results
        .uncertainty_info
        .get(&0)
        .expect("info is reported")
        .resonance_parameters
        .get("W186")
        .cloned()
        .expect("W186 has resonance parameters")
}

fn skip(what: &str) {
    eprintln!("skipping: {what} (nuclear-data fixtures missing)");
}

/// How far the sampled sigma may sit from the first-order one beyond the
/// two estimates' sampling errors. First order in the parameters is not
/// exact: a lognormal width or a moved resonance energy changes the capture
/// integral nonlinearly, and the rows are at 0 K and infinite dilution where
/// the sampled rate is broadened to the temperature in use. Five percent of
/// the sigma holds both for W186 under a smooth spectrum.
const NONLINEARITY_ALLOWANCE: f64 = 0.05;

/// The sampled relative sigma of the capture rate agrees with the
/// first-order rows', the mean shift is reported, the draws are reproducible
/// and independent of the MF=33 draws.
#[test]
fn sampled_capture_sigma_agrees_with_first_order() {
    let tmp_rows = tempfile::tempdir().expect("temp dir");
    let tmp_parameters = tempfile::tempdir().expect("temp dir");
    let (Some(rows), Some(parameters)) = (
        w186(tmp_rows.path(), Parameters::Absent),
        w186(tmp_parameters.path(), Parameters::Evaluated),
    ) else {
        return skip("sampled_capture_sigma_agrees_with_first_order");
    };
    const N: usize = 192;

    let mut first_order = tungsten(&rows, "294");
    let first = run(&mut first_order, N);
    assert!(
        first.uncertainty_info[&0].resonance_parameters.is_empty(),
        "a folder without resonance_parameters.arrow reports none"
    );
    let first_values = w187(&first, 0);

    let mut sampled_material = tungsten(&parameters, "294");
    let started = std::time::Instant::now();
    let sampled = run(&mut sampled_material, N);
    let elapsed = started.elapsed();
    match method(&sampled) {
        ResonanceMethod::Sampled { ranges } => assert!(!ranges.is_empty()),
        other => panic!("W186's parameters must be sampled, got {other:?}"),
    }
    let sampled_values = w187(&sampled, 0);
    let nominal = sampled
        .get_nuclide_density(0, "W187", 1)
        .expect("W187 is made");

    let (m1, s1) = mean_and_sigma(&first_values);
    let (m2, s2) = mean_and_sigma(&sampled_values);
    let (r1, r2) = (s1 / m1, s2 / m2);
    // The standard error of a sigma from N normal draws, relative.
    let se = (1.0 / (2.0 * (N as f64 - 1.0))).sqrt();
    let allowed = 3.0 * se * (r1 * r1 + r2 * r2).sqrt() + NONLINEARITY_ALLOWANCE * r1;
    let shift = m2 / nominal - 1.0;
    eprintln!(
        "W186 capture under 1/E at 294 K, {N} replicas: first order {:.3}%, \
         sampled {:.3}% ({:+.1}%), allowed difference {:.3}%; sampled mean \
         shift {:+.3}% of nominal ({:+.2} standard errors); {:.2} s per replica",
        100.0 * r1,
        100.0 * r2,
        100.0 * (r2 / r1 - 1.0),
        100.0 * allowed,
        100.0 * shift,
        shift / (r2 / (N as f64).sqrt()),
        elapsed.as_secs_f64() / N as f64,
    );
    assert!(r1 > 0.0 && r2 > 0.0);
    assert!(
        (r2 - r1).abs() < allowed,
        "sampled {r2} against first order {r1}, allowed {allowed}"
    );
    // A shift in the mean is what the exact sampling can show and first order
    // cannot, but on W186 capture it is well inside the spread.
    assert!(shift.abs() < r2, "mean shift {shift} against sigma {r2}");

    // The parameter draws come off a stream of their own: the replicas of the
    // two runs, made on the same seed, are uncorrelated.
    let correlation = {
        let cov = first_values
            .iter()
            .zip(&sampled_values)
            .map(|(a, b)| (a - m1) * (b - m2))
            .sum::<f64>()
            / (N as f64 - 1.0);
        cov / (s1 * s2)
    };
    assert!(
        correlation.abs() < 4.0 / (N as f64).sqrt(),
        "the parameter draws follow the MF=33 draws: correlation {correlation}"
    );

    // A replica is a pure function of the seed and its index, so a shorter
    // run on the same seed repeats the first replicas bit for bit.
    let mut again = tungsten(&parameters, "294");
    let repeated = w187(&run(&mut again, 8), 0);
    assert_eq!(repeated, sampled_values[..8]);
}

/// Parameters that cannot be read fall back to the first-order rows, are
/// reported with the reason, and leave the run exactly as it is without them.
#[test]
fn parameters_that_fail_fall_back_to_the_rows() {
    let tmp_rows = tempfile::tempdir().expect("temp dir");
    let tmp_broken = tempfile::tempdir().expect("temp dir");
    let (Some(rows), Some(broken)) = (
        w186(tmp_rows.path(), Parameters::Absent),
        w186(tmp_broken.path(), Parameters::Truncated),
    ) else {
        return skip("parameters_that_fail_fall_back_to_the_rows");
    };
    let results = run(&mut tungsten(&broken, "294"), 16);
    match method(&results) {
        ResonanceMethod::FirstOrder { reason } => {
            assert!(reason.contains("does not parse"), "{reason}")
        }
        other => panic!("truncated MF=32 must fall back, got {other:?}"),
    }
    let rows_only = run(&mut tungsten(&rows, "294"), 16);
    assert_eq!(w187(&results, 0), w187(&rows_only, 0));
}

/// The broadened weight is built per temperature: a run at 900 K samples the
/// parameters as one at 294 K does, and its capture sigma is close to it but
/// not the same draw read the same way.
#[test]
fn rates_at_two_temperatures_are_both_sampled() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(parameters) = w186(tmp.path(), Parameters::Evaluated) else {
        return skip("rates_at_two_temperatures_are_both_sampled");
    };
    let mut relative = Vec::new();
    for temperature in ["294", "900"] {
        let results = run(&mut tungsten(&parameters, temperature), 32);
        assert!(
            matches!(method(&results), ResonanceMethod::Sampled { .. }),
            "{temperature} K"
        );
        let (m, s) = mean_and_sigma(&w187(&results, 0));
        relative.push(s / m);
    }
    eprintln!(
        "W186 capture sigma at 294 K {:.3}%, at 900 K {:.3}%",
        100.0 * relative[0],
        100.0 * relative[1]
    );
    assert!(relative.iter().all(|r| r.is_finite() && *r > 0.0));
    assert_ne!(relative[0], relative[1]);
    assert!((relative[1] / relative[0] - 1.0).abs() < 0.2);
}
