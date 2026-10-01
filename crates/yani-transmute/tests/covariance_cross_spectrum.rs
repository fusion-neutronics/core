//! One nuclide's cross-section uncertainty under two spectra of one schedule.
//!
//! One evaluation states one uncertainty, so the rates it drives under two
//! spectra are correlated, and the fold gives that correlation as a cross
//! block between the spectra. These check the cross block against the
//! diagonal fold it must agree with, that the joint matrix is a covariance at
//! all, and that the first-order attribution of a schedule split over two
//! identical spectra matches the same schedule on one.
//!
//! The fixture is `common`'s: cached Fe56 with the committed evaluation's
//! MF=33 written into it. Self-skips when the nuclear-data fixtures are
//! missing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod common;
use common::fe56_with_covariance;

use yamc_materials::Material;
use yani_transmute::compute_multigroup_reaction_rates;
use yani_transmute::covariance_fold::{fold_cross_covariance, fold_rate_covariance, FoldSpectrum};
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const GROUPS: [f64; 4] = [1.0e-5, 0.625, 1.0e5, 2.0e7];
/// A hard spectrum, most of its flux fast.
const HARD: [f64; 3] = [1.0e12, 5.0e12, 1.0e14];
/// A softer one on a different group structure, so the cross block's
/// partials are taken on two grids.
const SOFT_GROUPS: [f64; 5] = [1.0e-5, 1.0, 1.0e5, 5.0e6, 2.0e7];
const SOFT: [f64; 4] = [4.0e13, 4.0e13, 1.5e13, 5.0e12];

fn chain() -> HashMap<String, yani::ChainNuclide> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    yani::parse_chain_arrow(&path).expect("parse chain")
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
    m.ensure_covariance_loaded().expect("read Fe56 covariance");
    m
}

fn normalized(flux: &[f64]) -> Vec<f64> {
    let total: f64 = flux.iter().sum();
    flux.iter().map(|f| f / total).collect()
}

fn skip(what: &str) {
    eprintln!("skipping: {what} (nuclear-data fixtures missing)");
}

/// The same spectrum twice: the cross block is the diagonal fold's matrix,
/// since `Σ C[k,l] r_i[k] r_j[l]` does not know which side a partial came
/// from.
#[test]
fn the_cross_block_of_a_spectrum_with_itself_is_its_own_fold() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("the_cross_block_of_a_spectrum_with_itself_is_its_own_fold");
    };
    let material = iron(&dir);
    let chain = chain();
    let flux = normalized(&HARD);
    let (rates, _) = compute_multigroup_reaction_rates(&material, &chain, &flux, &GROUPS, 1.0);
    let (folded, _) = fold_rate_covariance(&material, &chain, &rates, &flux, &GROUPS, None);
    let own = folded.get("Fe56").expect("Fe56 folds");

    let side = FoldSpectrum {
        chain: &chain,
        rates: &rates,
        multigroup_flux: &flux,
        group_boundaries: &GROUPS,
    };
    let cross = fold_cross_covariance(&material, &side, &side, None);
    let block = cross.get("Fe56").expect("Fe56 has a cross block");
    assert_eq!(block.len(), own.relative.len());
    let scale = own.relative.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "the fixture's (n,p) must carry a variance");
    for (x, c) in block.iter().zip(&own.relative) {
        assert!(
            (x - c).abs() <= 1e-12 * scale,
            "cross {x:e} against diagonal {c:e}"
        );
    }
}

/// Two different spectra: the stacked matrix must be a covariance, so every
/// cross entry is bounded by the two variances it sits between, and the
/// `(n,p)` rates driven by one evaluation under two spectra that both reach
/// its threshold are positively correlated.
#[test]
fn the_joint_matrix_of_two_spectra_is_a_covariance() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("the_joint_matrix_of_two_spectra_is_a_covariance");
    };
    let material = iron(&dir);
    let chain = chain();
    let (hard, soft) = (normalized(&HARD), normalized(&SOFT));
    let (rates_a, _) = compute_multigroup_reaction_rates(&material, &chain, &hard, &GROUPS, 1.0);
    let (rates_b, _) =
        compute_multigroup_reaction_rates(&material, &chain, &soft, &SOFT_GROUPS, 1.0);
    let (fold_a, _) = fold_rate_covariance(&material, &chain, &rates_a, &hard, &GROUPS, None);
    let (fold_b, _) = fold_rate_covariance(&material, &chain, &rates_b, &soft, &SOFT_GROUPS, None);
    let (ca, cb) = (&fold_a["Fe56"], &fold_b["Fe56"]);

    let cross = fold_cross_covariance(
        &material,
        &FoldSpectrum {
            chain: &chain,
            rates: &rates_a,
            multigroup_flux: &hard,
            group_boundaries: &GROUPS,
        },
        &FoldSpectrum {
            chain: &chain,
            rates: &rates_b,
            multigroup_flux: &soft,
            group_boundaries: &SOFT_GROUPS,
        },
        None,
    );
    let x = &cross["Fe56"];
    let (na, nb) = (ca.n(), cb.n());
    assert_eq!(x.len(), na * nb);
    for i in 0..na {
        for j in 0..nb {
            let bound = (ca.get(i, i).max(0.0) * cb.get(j, j).max(0.0)).sqrt();
            assert!(
                x[i * nb + j].abs() <= bound * (1.0 + 1e-9) + 1e-300,
                "{} / {}: |{:e}| above sqrt(var_a var_b) = {bound:e}",
                ca.kinds[i],
                cb.kinds[j],
                x[i * nb + j]
            );
        }
    }
    let i = ca.kinds.iter().position(|k| k == "(n,p)").expect("(n,p)");
    let j = cb.kinds.iter().position(|k| k == "(n,p)").expect("(n,p)");
    let rho = x[i * nb + j] / (ca.get(i, i) * cb.get(j, j)).sqrt();
    assert!(
        rho > 0.0 && rho <= 1.0 + 1e-9,
        "(n,p) under two spectra from one evaluation: correlation {rho}"
    );
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
/// steps use `spectra` out of `n` copies of the hard spectrum.
fn fe56_contribution(data: &Path, n: usize, spectra: [usize; 2]) -> f64 {
    let mut material = iron(data);
    let id = material.material_id.unwrap_or(0);
    let spectrum = MultigroupSpectrum {
        boundaries: GROUPS.to_vec(),
        masses: normalized(&HARD),
        flux_error: None,
    };
    let results = transmute_material(
        &mut material,
        &vec![spectrum; n],
        &schedule_on(spectra, HARD.iter().sum()),
        Arc::new(chain()),
        &Default::default(),
        Default::default(),
        Some(&DataUncertainty {
            seed: 5,
            samples: Some(128),
            sources: vec![Source::CrossSections],
            attribution: true,
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
/// with both steps on one spectrum. That holds only if the two copies'
/// perturbations are fully correlated, which is what the joint factor gives:
/// `(s_a + s_b)ᵀ C (s_a + s_b)` rather than a sum over two unrelated bases.
#[test]
fn a_schedule_split_over_two_copies_of_a_spectrum_attributes_the_same() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        return skip("a_schedule_split_over_two_copies_of_a_spectrum_attributes_the_same");
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
