//! The MF=33 fold of a self-shielded run must weight with shielded partials.
//!
//! The relative covariance of a rate is `rᵀ C r / R²`, with `r` the rate's
//! parts over the covariance intervals and `R` the rate. On a shielded run `R`
//! is the shielded rate, and `r` used to be the dilute parts regardless, so
//! the numerator and the denominator were two different integrals and the
//! relative sigma came out high by about one over the shielding factor. This
//! is the MF=33 counterpart of `flux_uncertainty_under_shielding.rs`, which
//! fixed the same mismatch for the flux source.
//!
//! The covariances here are synthetic, written into the cached Fe56 in place
//! of what it carries, so the expected sigma is known in closed form. One test
//! also folds the evaluation's own MF=33. The material is Fe56 alone at 1.0
//! atoms/b-cm with a 2 cm chord, enough for its keV resonances to shield
//! capture by a few percent on these three groups.
//!
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

mod common;
use common::fe56_with_covariance;

use endf::mf::covariance::NiSubsection;
use yamc_materials::Material;
use yamc_nuclide::covariance::{CovarianceBlock, CovarianceData};
use yani::ReactionRates;
use yani_transmute::covariance_fold::{fold_rate_covariance, RateCovariance};
use yani_transmute::multigroup::{
    compute_multigroup_reaction_rates_shielded, per_group_reaction_rates,
};
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material_shielded, MultigroupSpectrum, Shielding, TransmuteStep};

/// Three groups: thermal, epithermal, fast.
const GROUPS: [f64; 4] = [1.0e-5, 0.625, 1.0e5, 2.0e7];
const FLUX: [f64; 3] = [1.0e12, 5.0e12, 1.0e14];
const CHORD_CM: f64 = 2.0;
/// Fe56 channels the chain drives, by MT and by the kind the rates key them.
const CHANNELS: [(i32, &str); 4] = [
    (16, "(n,2n)"),
    (102, "(n,gamma)"),
    (103, "(n,p)"),
    (107, "(n,a)"),
];

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

fn masses() -> Vec<f64> {
    let total: f64 = FLUX.iter().sum();
    FLUX.iter().map(|f| f / total).collect()
}

fn fe56(data: &str) -> Material {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), data.to_string())]),
        None,
    )
    .expect("read Fe56");
    m
}

/// Cached Fe56 with `blocks` as its whole MF=33, marked as loaded so the
/// transmutation does not read the file's own covariance over them.
fn fe56_with(data: &str, blocks: Vec<CovarianceBlock>) -> Material {
    let mut m = fe56(data);
    let nuclide = Arc::make_mut(m.nuclide_data.get_mut("Fe56").expect("Fe56 loaded"));
    nuclide.covariance = Some(Arc::new(blocks));
    nuclide.load_scope.covariance = true;
    m
}

fn block(mt: i32, ni: NiSubsection) -> CovarianceBlock {
    CovarianceBlock {
        mt,
        subsection_idx: 0,
        block_idx: 0,
        mat1: 0,
        mt1: mt,
        xmf1: 0.0,
        xlfs1: 0.0,
        mtl: 0,
        data: CovarianceData::Ni(ni),
    }
}

/// `lb = 5`, `ls = 1`: every entry of the upper triangle is `v`, so the
/// intervals are fully correlated and `rᵀ C r = v (Σ r)²`.
fn correlated(energies: &[f64], v: f64) -> NiSubsection {
    let n = energies.len() - 1;
    NiSubsection {
        lb: 5,
        ls: 1,
        ne: energies.len() as i64,
        ek: energies.to_vec(),
        fkk: vec![v; n * (n + 1) / 2],
        ..Default::default()
    }
}

/// `lb = 1`: one relative variance per interval, uncorrelated between them.
fn diagonal(energies: &[f64], variances: &[f64]) -> NiSubsection {
    NiSubsection {
        lb: 1,
        ek: energies.to_vec(),
        fk: variances.to_vec(),
        ..Default::default()
    }
}

fn rates(m: &Material, shielding: Option<&Shielding>) -> ReactionRates {
    compute_multigroup_reaction_rates_shielded(m, &chain(), &masses(), &GROUPS, 1.0, shielding).0
}

fn variance(folded: &BTreeMap<String, RateCovariance>, kind: &str) -> f64 {
    let cov = &folded["Fe56"];
    let i = cov
        .kinds
        .iter()
        .position(|k| k == kind)
        .expect("the channel is folded");
    cov.get(i, i)
}

/// A covariance stated fully correlated across the whole flux range gives
/// every rate the same relative variance `v`, whatever its shape in energy,
/// exactly when the partial rates sum to the rate. So `v` back, to rounding,
/// is that sum checked. The grid's interior edges all fall inside groups,
/// which is where the shielded partials have to split a group's term. How a
/// cut group is shared between intervals does not reach this sum; the unit
/// tests in `covariance_fold.rs` check the split itself.
#[test]
fn the_shielded_partials_sum_to_the_shielded_rate() {
    let Some(data) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    };
    const V: f64 = 0.01;
    let grid = [1.0e-5, 1.0, 1.0e3, 3.0e4, 1.0e6, 2.0e7];
    let m = fe56_with(
        &data,
        CHANNELS
            .iter()
            .map(|&(mt, _)| block(mt, correlated(&grid, V)))
            .collect(),
    );
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let shielded = rates(&m, Some(&shielding));

    let (folded, coverage) = fold_rate_covariance(
        &m,
        &chain(),
        &shielded,
        &masses(),
        &GROUPS,
        Some(&shielding),
    );
    for (_, kind) in CHANNELS {
        let got = variance(&folded, kind);
        assert!(
            (got / V - 1.0).abs() < 1.0e-12,
            "{kind}: shielded partials over the shielded rate give {got}, not {V}"
        );
    }
    assert!(
        coverage.partials_above_rate.is_empty(),
        "{:?}",
        coverage.partials_above_rate
    );

    // What the fold did before: dilute partials, which sum to the dilute rate,
    // over the shielded rate. The variance is then v (R_dilute / R_shielded)^2,
    // which is what the check above tells apart.
    let dilute = rates(&m, None);
    let factor = shielded["Fe56"]["(n,gamma)"] / dilute["Fe56"]["(n,gamma)"];
    assert!(
        factor < 0.98,
        "capture must shield for this to discriminate: {factor}"
    );
    let (unshielded_fold, stale) =
        fold_rate_covariance(&m, &chain(), &shielded, &masses(), &GROUPS, None);
    let got = variance(&unshielded_fold, "(n,gamma)");
    let expected = V / (factor * factor);
    assert!(
        (got / expected - 1.0).abs() < 1.0e-12,
        "{got} against {expected}"
    );
    assert!(stale
        .partials_above_rate
        .contains_key(&("Fe56".to_string(), "(n,gamma)".to_string())));
}

/// Uncorrelated variances on intervals that are whole groups, against the
/// closed form `Σ v_k r_k² / R²` with `r_k` the shielded per-group terms of
/// the collapse. Those come from the walk the flux source perturbs, a
/// different path from the fold's, and they sum to the rate the fold divides
/// by.
#[test]
fn the_shielded_sigma_is_the_closed_form_value() {
    let Some(data) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    };
    let variances = [0.09, 0.04, 0.0025];
    let m = fe56_with(&data, vec![block(102, diagonal(&GROUPS, &variances))]);
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let shielded = rates(&m, Some(&shielding));
    let (folded, _) = fold_rate_covariance(
        &m,
        &chain(),
        &shielded,
        &masses(),
        &GROUPS,
        Some(&shielding),
    );

    let terms = per_group_reaction_rates(&m, &chain(), &masses(), &GROUPS, Some(&shielding));
    let r = &terms["Fe56"]["(n,gamma)"];
    let rate = shielded["Fe56"]["(n,gamma)"];
    let expected: f64 =
        variances.iter().zip(r).map(|(v, r)| v * r * r).sum::<f64>() / (rate * rate);
    let got = variance(&folded, "(n,gamma)");
    assert!(
        (got / expected - 1.0).abs() < 1.0e-12,
        "shielded (n,gamma) relative variance {got} against {expected}"
    );

    // And it is not the dilute answer, which weights the groups differently.
    let dilute = rates(&m, None);
    let (dilute_fold, _) = fold_rate_covariance(&m, &chain(), &dilute, &masses(), &GROUPS, None);
    let d = variance(&dilute_fold, "(n,gamma)");
    assert!(
        (got / d - 1.0).abs() > 1.0e-3,
        "shielding must move the answer for this to discriminate: {got} against {d}"
    );
}

/// The evaluation's own MF=33 folds consistently under shielding: every
/// partial sum is within rounding of the rate it is divided by, where the
/// dilute partials sat above the shielded rate.
#[test]
fn the_evaluations_covariance_folds_consistently_under_shielding() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = fe56_with_covariance(tmp.path()) else {
        eprintln!("skipping: nuclear-data fixtures missing");
        return;
    };
    let mut m = fe56(&dir.to_string_lossy());
    m.ensure_covariance_loaded().expect("covariance loads");
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let shielded = rates(&m, Some(&shielding));

    let (_, stale) = fold_rate_covariance(&m, &chain(), &shielded, &masses(), &GROUPS, None);
    assert!(
        !stale.partials_above_rate.is_empty(),
        "dilute partials must sit above a shielded rate for this to discriminate"
    );

    let (folded, coverage) = fold_rate_covariance(
        &m,
        &chain(),
        &shielded,
        &masses(),
        &GROUPS,
        Some(&shielding),
    );
    assert!(folded.contains_key("Fe56"));
    assert!(
        coverage.partials_above_rate.is_empty(),
        "{:?}",
        coverage.partials_above_rate
    );
    for (key, share) in &coverage.rate_fraction_covered {
        assert!((0.0..=1.0).contains(share), "{key:?}: {share}");
    }
}

/// A covariance of zero perturbs nothing on a shielded run either: every
/// replica is the nominal inventory, bit for bit.
#[test]
fn a_zero_covariance_gives_zero_spread_under_shielding() {
    let Some(data) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    };
    let grid = [1.0e-5, 1.0e3, 2.0e7];
    let mut m = fe56_with(
        &data,
        CHANNELS
            .iter()
            .map(|&(mt, _)| block(mt, correlated(&grid, 0.0)))
            .collect(),
    );
    let id = m.material_id.unwrap_or(0);
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let total: f64 = FLUX.iter().sum();
    let results = transmute_material_shielded(
        &mut m,
        &[MultigroupSpectrum {
            boundaries: GROUPS.to_vec(),
            masses: masses(),
            flux_error: None,
        }],
        &[TransmuteStep {
            dt: 86400.0,
            irradiation: Some((0, total)),
        }],
        chain(),
        &Default::default(),
        Default::default(),
        Some(&DataUncertainty {
            seed: 1,
            samples: Some(4),
            sources: vec![Source::CrossSections],
            attribution: false,
        }),
        Some(&shielding),
    )
    .expect("transmute");

    let info = &results.uncertainty_info[&id];
    assert!(info.perturbed.contains("Fe56"), "{:?}", info.perturbed);
    assert!(
        info.partials_above_rate.is_empty(),
        "{:?}",
        info.partials_above_rate
    );
    for product in ["Fe57", "Mn56"] {
        let nominal = results
            .get_nuclide_density(id, product, 1)
            .expect("the product exists after irradiation");
        let ensemble = results.uncertainty[&id].samples_at(0, product);
        assert_eq!(ensemble.len(), 4);
        for (replica, sample) in ensemble.iter().enumerate() {
            assert_eq!(
                *sample, nominal,
                "{product} replica {replica} perturbed nothing and must be the nominal run"
            );
        }
    }
}
