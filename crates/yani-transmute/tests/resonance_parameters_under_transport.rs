//! Resonance parameters sampled on a transport-coupled run.
//!
//! A transport run's rates are its tally's, and the resonance range is weighted
//! with the flux the transport tallied on
//! [`resolved_flux_grid`](yani_transmute::resonance_rates::resolved_flux_grid).
//! Given the same flux and rates, a transport-coupled run and a spectrum run
//! must draw the same parameters and move the rate by the same amount, and
//! that amount must be the drawn cross-section change that
//! [`TransportResonance`] tabulates for a transport that applies the draws
//! itself, folded against the flux.
//!
//! The fixture is the cached ENDF/B-VIII.1 W186 directory with its
//! `covariance.arrow` and `resonance_parameters.arrow` written by the real
//! converter from the committed MF=2 and MF=32 evaluation, which has no MF=33.
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::resonance_rates::{resolved_flux_grid, ResonanceMethod, TransportResonance};
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{
    multigroup::compute_multigroup_reaction_rates_shielded, transmute_material, transport_replicas,
    MultigroupSpectrum, TransmuteStep, TransportTallied,
};

const W186_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-074_W_186_mf2_mf32.endf.xz");
const SEED: u64 = 4242;
const REPLICAS: usize = 24;
/// The flux the rates are scaled to, n/cm2/s, for one second.
const FLUX: f64 = 1.0e10;

/// A copy of the cached W186 directory with the converter's
/// `covariance.arrow` and `resonance_parameters.arrow` from the committed
/// evaluation, or `None` when the fixtures are missing.
fn w186(tmp: &Path) -> Option<PathBuf> {
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
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &W186_ENDF[..], &mut raw).expect("fixture decompresses");
    let evaluation = tmp.join("w186.endf");
    std::fs::write(&evaluation, raw).expect("write evaluation");
    let material = endf::Material::from_file(&evaluation).expect("W186 parses");
    assert!(yamc_convert::covariance::write_covariance(&material, &dir).expect("covariance"));
    assert!(
        yamc_convert::resonance_parameters::write_resonance_parameters(&material, &dir)
            .expect("resonance parameters")
    );
    Some(dir)
}

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

fn tungsten(data: &Path) -> Material {
    let mut m = Material::new(
        HashMap::from([("W186".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("W186 material");
    m.density = Some(19.3);
    m.set_material_id(3);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("W186".to_string(), data.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read W186");
    m
}

/// A flux on the resolved grid: `1/E` with a dip at each of the 18.8 eV
/// resonance's flanks, so it is not one a collapse shape could stand for.
fn resolved_flux(grid: &[f64]) -> MultigroupSpectrum {
    let masses = grid
        .windows(2)
        .map(|b| {
            let mid = 0.5 * (b[0] + b[1]);
            let dip = 1.0 - 0.8 * (-((mid - 18.8) / 0.5).powi(2)).exp();
            dip * (b[1] / b[0]).ln()
        })
        .collect();
    MultigroupSpectrum {
        boundaries: grid.to_vec(),
        masses,
        flux_error: None,
    }
}

fn request() -> DataUncertainty {
    DataUncertainty {
        seed: SEED,
        samples: Some(REPLICAS),
        sources: vec![Source::CrossSections],
        ..Default::default()
    }
}

/// W187 per replica, and its nominal, after one second.
fn w187(results: &yani_transmute::TransmutationResults, id: u32) -> (Vec<f64>, f64) {
    let replicas = results
        .uncertainty_inventories(id, 1)
        .expect("uncertainty was requested")
        .iter()
        .map(|inventory| inventory.get("W187").copied().unwrap_or(0.0))
        .collect();
    let nominal = results.get_nuclide_density(id, "W187", 1).expect("W187");
    (replicas, nominal)
}

fn method(results: &yani_transmute::TransmutationResults, id: u32) -> ResonanceMethod {
    results.uncertainty_info[&id].resonance_parameters["W186"].clone()
}

/// The transport-coupled run with the tallied flux through the resonances
/// samples the parameters, and moves every replica's rate exactly as the
/// spectrum run does on the same flux. Without that flux it keeps the rows
/// and says why.
#[test]
fn transport_samples_the_parameters_as_the_spectrum_run_does() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = w186(tmp.path()) else {
        eprintln!("skipping: nuclear-data fixtures missing");
        return;
    };
    let material = tungsten(&dir);
    let grid = resolved_flux_grid(&material).expect("grid");
    assert!(
        grid.len() > 1000,
        "the grid traces the resonances: {}",
        grid.len()
    );
    let spectrum = resolved_flux(&grid);
    let chain = chain();

    // The spectrum run on the resolved flux.
    let mut on_spectrum = material.clone();
    let spectrum_results = transmute_material(
        &mut on_spectrum,
        std::slice::from_ref(&spectrum),
        &[TransmuteStep {
            dt: 1.0,
            irradiation: Some((0, FLUX)),
        }],
        Arc::clone(&chain),
        &Default::default(),
        Default::default(),
        Some(&request()),
    )
    .expect("spectrum run");
    assert!(matches!(
        method(&spectrum_results, 3),
        ResonanceMethod::Sampled { .. }
    ));

    // The transport-coupled run, its rates the collapse of the same flux per
    // unit source, the resolved flux its own.
    let mut loaded = material.clone();
    loaded.ensure_covariance_loaded().expect("covariance");
    let (rates, fy_weights, _) = compute_multigroup_reaction_rates_shielded(
        &loaded,
        &chain,
        &spectrum.masses,
        &spectrum.boundaries,
        1.0,
        None,
    );
    let tallied = |resolved: Option<MultigroupSpectrum>| TransportTallied {
        rates: rates.clone(),
        partials: HashMap::new(),
        fy_weights: fy_weights.clone(),
        spectrum: spectrum.clone(),
        resolved_spectrum: resolved,
        statistics: None,
        branch: Arc::new(yani::BranchTable::default()),
        diagnostics: Default::default(),
        replicas: None,
    };
    let transport = |resolved: Option<MultigroupSpectrum>| {
        let (ensemble, info) = transport_replicas(
            &material,
            &tallied(resolved),
            &[1.0],
            &[FLUX],
            &chain,
            Default::default(),
            &request(),
            &Default::default(),
        )
        .expect("transport replicas");
        (ensemble, info)
    };
    let (ensemble, info) = transport(Some(spectrum.clone()));
    let ResonanceMethod::Sampled { ranges } = &info.resonance_parameters["W186"] else {
        panic!(
            "W186 must be sampled on the transport run: {:?}",
            info.resonance_parameters["W186"]
        );
    };
    assert!(!ranges.is_empty());

    let (on_spectrum_w187, spectrum_nominal) = w187(&spectrum_results, 3);
    let on_transport = ensemble.samples_at(0, "W187");
    assert_eq!(on_transport.len(), on_spectrum_w187.len());
    let mut spread = 0.0_f64;
    for (k, (a, b)) in on_transport.iter().zip(&on_spectrum_w187).enumerate() {
        let shift = b / spectrum_nominal - 1.0;
        spread = spread.max(shift.abs());
        assert!(
            (a / b - 1.0).abs() < 1e-9,
            "replica {k}: transport W187 {a:e}, spectrum {b:e}"
        );
    }
    assert!(spread > 1e-3, "the parameters move the capture: {spread}");

    // Without the resolved flux the rows are kept, and the report says why.
    let (_, info) = transport(None);
    match &info.resonance_parameters["W186"] {
        ResonanceMethod::FirstOrder { reason } => {
            assert!(reason.contains("resolved_spectrum"), "{reason}")
        }
        other => panic!("without the resolved flux W186 keeps the rows, got {other:?}"),
    }
}

/// What the fold moves a replica's rate by is the cross-section change a
/// transport applies for that replica, `TransportResonance::changes`,
/// integrated against the same flux: the broadened change on the library's
/// grid against the fold's 0 K change through the broadened weight, the two
/// sides of the adjoint identity.
#[test]
fn the_tabulated_change_is_the_folded_one() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = w186(tmp.path()) else {
        eprintln!("skipping: nuclear-data fixtures missing");
        return;
    };
    let material = tungsten(&dir);
    let grid = resolved_flux_grid(&material).expect("grid");
    let spectrum = resolved_flux(&grid);
    let chain = chain();
    let mut run_material = material.clone();
    let results = transmute_material(
        &mut run_material,
        std::slice::from_ref(&spectrum),
        &[TransmuteStep {
            dt: 1.0,
            irradiation: Some((0, FLUX)),
        }],
        Arc::clone(&chain),
        &Default::default(),
        Default::default(),
        Some(&request()),
    )
    .expect("spectrum run");
    let (replicas, nominal) = w187(&results, 3);
    assert!(matches!(
        method(&results, 3),
        ResonanceMethod::Sampled { .. }
    ));

    let mut loaded = material.clone();
    loaded
        .ensure_resonance_parameters_loaded()
        .expect("parameters");
    let nd = &loaded.nuclide_data["W186"];
    let resonance = TransportResonance::new(
        "W186",
        nd.resonance_parameters.as_ref().expect("parameters"),
        "294",
    )
    .expect("prepared");
    let capture = nd.reactions_for_temp("294").expect("294 K")[&102].clone();
    let energies = nd.energy.as_ref().expect("grid")["294"].as_slice().to_vec();
    // A table on the library's grid at `e`, linear between its points.
    let at = |e: f64, values: &[f64]| -> f64 {
        let i = energies
            .partition_point(|&x| x <= e)
            .clamp(1, energies.len() - 1)
            - 1;
        let f = (e - energies[i]) / (energies[i + 1] - energies[i]);
        values[i] + f * (values[i + 1] - values[i])
    };
    // Every bin integrated on the library's own points inside it, which is
    // where both tables are linear.
    let fold = |values: &[f64]| -> f64 {
        let mut total = 0.0;
        for (g, b) in spectrum.boundaries.windows(2).enumerate() {
            let density = spectrum.masses[g] / (b[1] - b[0]);
            let mut points = vec![b[0]];
            points.extend(energies.iter().copied().filter(|&e| e > b[0] && e < b[1]));
            points.push(b[1]);
            for w in points.windows(2) {
                total += density * 0.5 * (at(w[0], values) + at(w[1], values)) * (w[1] - w[0]);
            }
        }
        1e-24 * total
    };
    let mut sigma = vec![0.0; energies.len()];
    let start = capture.threshold_idx;
    sigma[start..start + capture.cross_section.len()]
        .copy_from_slice(capture.cross_section.as_slice());
    let rate = fold(&sigma);
    for (k, w) in replicas.iter().enumerate().take(6) {
        let changes = resonance
            .changes(SEED, k as u64, &energies)
            .expect("changes");
        let delta = fold(&changes[&102]);
        let folded = w / nominal - 1.0;
        let tabulated = delta / rate;
        eprintln!("replica {k}: folded {folded:+.5e}, tabulated {tabulated:+.5e}");
        assert!(
            (tabulated - folded).abs() < 2e-3 * folded.abs() + 1e-6,
            "replica {k}: the fold moved capture by {folded:e}, the table by {tabulated:e}"
        );
    }
}
