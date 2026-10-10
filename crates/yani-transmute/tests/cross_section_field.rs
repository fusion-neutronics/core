//! The cross-section field: one draw of a nuclide's cross sections, read by
//! every spectrum of a schedule.
//!
//! These check the field against the fold it has to agree with, on every
//! evaluation this machine has cached, and that every spectrum really reads
//! one draw: a spectrum that is the mean of two others reads, replica by
//! replica, the mean of their perturbed rates. That holds only if the draw is
//! of the cross sections themselves rather than of each spectrum's rates.
//!
//! The always-present fixture is `common`'s Fe56 with the committed
//! evaluation's MF=33. The cached libraries add `lb = 8`, absolute and
//! cross-reaction blocks, and are skipped where absent.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod common;
use common::fe56_with_covariance;

use yamc_materials::Material;
use yani_transmute::compute_multigroup_reaction_rates;
use yani_transmute::covariance_fold::{cell_fields, fold_rate_covariance, FoldSpectrum};
use yani_transmute::covariance_sample::Sampler;
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

fn chain() -> HashMap<String, yani::ChainNuclide> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    yani::parse_chain_arrow(&path).expect("parse chain")
}

fn material(nuclide: &str, data: &Path) -> Material {
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    m.density = Some(8.0);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([(nuclide.to_string(), data.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read nuclear data");
    m.ensure_covariance_loaded().expect("read covariance");
    m
}

/// `n` groups log-spaced from 1e-5 eV to 20 MeV.
fn log_groups(n: usize) -> Vec<f64> {
    let (lo, hi) = (1.0e-5_f64.ln(), 2.0e7_f64.ln());
    (0..=n)
        .map(|i| (lo + (hi - lo) * i as f64 / n as f64).exp())
        .collect()
}

/// A fusion-like spectrum on `groups`: a 1/E slowing-down continuum with a
/// 14 MeV peak, normalized.
fn fusion(groups: &[f64]) -> Vec<f64> {
    let raw: Vec<f64> = groups
        .windows(2)
        .map(|w| {
            let mid = (w[0] * w[1]).sqrt();
            let continuum = (w[1] / w[0]).ln();
            let peak = if (13.5e6..15.0e6).contains(&mid) {
                30.0
            } else {
                0.0
            };
            continuum + peak
        })
        .collect();
    let total: f64 = raw.iter().sum();
    raw.iter().map(|v| v / total).collect()
}

/// A softer, fission-like spectrum on `groups`, normalized.
fn soft(groups: &[f64]) -> Vec<f64> {
    let raw: Vec<f64> = groups
        .windows(2)
        .map(|w| {
            let mid = (w[0] * w[1]).sqrt();
            let maxwell = (mid / 1.0e6).sqrt() * (-mid / 1.3e6).exp();
            maxwell * (w[1] - w[0]) / 1.0e6 + 0.02 * (w[1] / w[0]).ln()
        })
        .collect();
    let total: f64 = raw.iter().sum();
    raw.iter().map(|v| v / total).collect()
}

/// Every (label, nuclide, data directory) this machine can test: the fixture,
/// plus each cached evaluation of a few fusion-relevant nuclides.
fn evaluations(tmp: &Path) -> Vec<(String, String, PathBuf)> {
    let mut out = Vec::new();
    if let Some(dir) = fe56_with_covariance(tmp) {
        out.push(("fixture Fe56".to_string(), "Fe56".to_string(), dir));
    }
    for library in [
        "endf-b8.1",
        "jeff-4.0",
        "tendl-2017",
        "tendl-2025",
        "fendl-3.2d",
    ] {
        for nuclide in ["Fe56", "Cr52", "Co59", "W186", "Pb208", "Li6"] {
            let dir = yamc_test_cache::root().join(format!("{library}-{nuclide}.arrow"));
            if dir.join("covariance.arrow").is_file() && dir.join("reactions.arrow").is_file() {
                out.push((format!("{library} {nuclide}"), nuclide.to_string(), dir));
            }
        }
    }
    out
}

/// Under every spectrum, every channel's sampled variance is the fold's.
///
/// The fold contracts each block on its own grid against one spectrum; the
/// field refines every block onto the union of its reactions' grids and the
/// spectrum reads it there. Those are the same number by construction, so
/// this checks the refinement, the consumption rules and the short-range and
/// absolute paths against an independent implementation. Where the
/// evaluation needed a repair, or a channel is wide enough that a lognormal
/// cannot carry its correlations exactly, the two may differ, and those are
/// skipped.
#[test]
fn the_field_reproduces_the_fold_under_every_spectrum() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let evaluations = evaluations(tmp.path());
    if evaluations.is_empty() {
        eprintln!("skipping: no evaluation with covariance available");
        return;
    }
    let chain = chain();
    let (fine, coarse) = (log_groups(175), vec![1.0e-5, 0.625, 1.0e5, 5.0e6, 2.0e7]);
    let spectra = [
        (fine.clone(), fusion(&fine)),
        (coarse.clone(), soft(&coarse)),
    ];
    let mut checked = 0;
    for (label, nuclide, dir) in &evaluations {
        let m = material(nuclide, dir);
        let rates: Vec<_> = spectra
            .iter()
            .map(|(g, f)| compute_multigroup_reaction_rates(&m, &chain, f, g, 1.0).0)
            .collect();
        let folds: Vec<_> = spectra
            .iter()
            .zip(&rates)
            .map(|((g, f), r)| fold_rate_covariance(&m, &chain, r, f, g, None).0)
            .collect();
        let fold_spectra: Vec<FoldSpectrum> = spectra
            .iter()
            .zip(&rates)
            .map(|((g, f), r)| FoldSpectrum {
                chain: &chain,
                rates: r,
                multigroup_flux: f,
                group_boundaries: g,
            })
            .collect();
        let started = std::time::Instant::now();
        let fields = cell_fields(&m, &chain, &fold_spectra, None, &BTreeSet::new());
        let sampler = Sampler::new(&fields, &folds);
        let elapsed = started.elapsed();
        if let Some(field) = fields.get(nuclide.as_str()) {
            eprintln!(
                "{label}: {} relative, {} absolute cells, {} short-range blocks, built in {elapsed:?}",
                field.relative_cells.len(),
                field.absolute_cells.len(),
                field.short.len()
            );
        }
        for a in 0..spectra.len() {
            let repaired = !sampler.repairs(a).is_empty();
            for (n, kind, evaluated, sampled) in sampler.channel_sigmas(a) {
                if n != nuclide || evaluated == 0.0 || evaluated > 0.3 {
                    continue;
                }
                if repaired {
                    // Clipping removes negative eigenvalues, so `C+ - C` is
                    // PSD and no rate's variance can drop.
                    assert!(
                        sampled >= evaluated * (1.0 - 1e-6),
                        "{label} {kind} under spectrum {a}: repaired to {sampled}, below the \
                         folded {evaluated}"
                    );
                    eprintln!(
                        "{label} {kind} spectrum {a}: repaired, sigma {evaluated:.4} -> {sampled:.4}"
                    );
                } else {
                    assert!(
                        (sampled / evaluated - 1.0).abs() < 1e-6,
                        "{label} {kind} under spectrum {a}: sampled {sampled}, folded {evaluated}"
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no channel was compared");
}

/// A spectrum that is the mean of two others reads, replica by replica, the
/// mean of their perturbed rates.
///
/// A unit-flux rate is linear in the spectrum, and a draw of the cross
/// sections perturbs it linearly, so this holds exactly when, and only when,
/// all three spectra read the same draw of the same cross sections. Sampling
/// each spectrum's rates from its own folded covariance, however correctly
/// correlated, would not satisfy it.
#[test]
fn every_spectrum_reads_one_draw_of_the_cross_sections() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let evaluations = evaluations(tmp.path());
    if evaluations.is_empty() {
        eprintln!("skipping: no evaluation with covariance available");
        return;
    }
    let chain = chain();
    let groups = log_groups(80);
    let (a, b) = (fusion(&groups), soft(&groups));
    let mean: Vec<f64> = a.iter().zip(&b).map(|(x, y)| 0.5 * (x + y)).collect();
    let mut checked = 0;
    for (label, nuclide, dir) in &evaluations {
        let m = material(nuclide, dir);
        let fluxes = [&a, &b, &mean];
        let rates: Vec<_> = fluxes
            .iter()
            .map(|f| compute_multigroup_reaction_rates(&m, &chain, f, &groups, 1.0).0)
            .collect();
        let folds: Vec<_> = fluxes
            .iter()
            .zip(&rates)
            .map(|(f, r)| fold_rate_covariance(&m, &chain, r, f, &groups, None).0)
            .collect();
        let fold_spectra: Vec<FoldSpectrum> = fluxes
            .iter()
            .zip(&rates)
            .map(|(f, r)| FoldSpectrum {
                chain: &chain,
                rates: r,
                multigroup_flux: f,
                group_boundaries: &groups,
            })
            .collect();
        let sampler = Sampler::new(
            &cell_fields(&m, &chain, &fold_spectra, None, &BTreeSet::new()),
            &folds,
        );
        for replica in 0..16 {
            let draw = sampler.draw(7, replica);
            let perturbed: Vec<_> = (0..3)
                .map(|s| sampler.perturb_with(&draw, s, &rates[s], None).0)
                .collect();
            for (kind, r_mean) in &perturbed[2][nuclide.as_str()] {
                let (Some(ra), Some(rb)) = (
                    perturbed[0][nuclide.as_str()].get(kind),
                    perturbed[1][nuclide.as_str()].get(kind),
                ) else {
                    continue;
                };
                let want = 0.5 * (ra + rb);
                assert!(
                    (r_mean - want).abs() <= 1e-9 * want.abs().max(1e-300),
                    "{label} {kind} replica {replica}: {r_mean:e} against {want:e}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no rate was compared");
}

fn schedule_on(spectra: [usize; 2], flux: f64) -> Vec<TransmuteStep> {
    vec![
        TransmuteStep {
            dt: 3600.0,
            irradiation: Some((spectra[0], flux)),
        },
        TransmuteStep {
            dt: 3600.0,
            irradiation: Some((spectra[1], flux)),
        },
        TransmuteStep {
            dt: 3600.0,
            irradiation: None,
        },
    ]
}

/// Fe56's first-order contribution to Mn56, from a run whose two irradiation
/// steps use `spectra` out of `n` copies of one spectrum.
fn fe56_contribution(data: &Path, n: usize, spectra: [usize; 2]) -> f64 {
    let mut m = material("Fe56", data);
    let id = m.material_id.unwrap_or(0);
    let groups = vec![1.0e-5, 0.625, 1.0e5, 2.0e7];
    let spectrum = MultigroupSpectrum {
        masses: fusion(&groups),
        boundaries: groups,
        flux_error: None,
    };
    let results = transmute_material(
        &mut m,
        &vec![spectrum; n],
        &schedule_on(spectra, 1.0e14),
        Arc::new(chain()),
        &Default::default(),
        Default::default(),
        Some(&DataUncertainty {
            seed: 5,
            samples: Some(128),
            sources: vec![Source::CrossSections],
            attribution: true,
            ..Default::default()
        }),
    )
    .expect("transmute");
    let b = results
        .uncertainty_breakdown(id, "Mn56", 2)
        .expect("asked for");
    b.contributors
        .iter()
        .find(|(s, n, r, _)| s == "cross_sections" && n == "Fe56" && r.is_none())
        .expect("Fe56's evaluation contributes")
        .3
}

/// Splitting a schedule over two copies of one spectrum changes nothing
/// physical, so Fe56's first-order contribution must come out the same as
/// with both steps on one spectrum: both copies read the same field.
#[test]
fn a_schedule_split_over_two_copies_of_a_spectrum_attributes_the_same() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        eprintln!("skipping: nuclear-data fixtures missing");
        return;
    };
    let one = fe56_contribution(&dir, 1, [0, 0]);
    let two = fe56_contribution(&dir, 2, [0, 1]);
    assert!(one > 0.0);
    // Forward differences at h = 1e-3 per spectrum, so agreement is to about
    // that, not to rounding.
    assert!(
        (two / one - 1.0).abs() < 5e-3,
        "one spectrum {one:e}, the same split over two copies {two:e}"
    );
}
