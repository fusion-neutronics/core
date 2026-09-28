//! Channels whose MF=33 covariance is stated only as a sum of other reactions'.
//!
//! ENDF/B-VIII.1 O16 gives `(n,p)` no covariance of its own: its MT 103
//! section is one NC block with LTY=0 saying `σ_103 = σ_600 + ... + σ_603`
//! from 1e-5 eV to 150 MeV, and the partials carry the NI blocks. B10 `(n,a)`
//! is `800 + 801` the same way, with a cross block between the two. The fold
//! used to skip every NC block, so these channels folded to nothing. Folded on
//! the cached evaluation over one narrow group, the nuclide must now be covered
//! for them with no NC block left over, and the relative sigma must be the
//! sandwich of the partials' own blocks written out by hand.
//!
//! Self-skips when the nuclear-data fixtures or their covariance are missing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc_materials::Material;
use yamc_nuclide::covariance::expand::expand_ni;
use yamc_nuclide::covariance::{CovarianceBlock, CovarianceData};
use yani_transmute::covariance_fold::{fold_rate_covariance, Coverage};
use yani_transmute::multigroup::compute_multigroup_reaction_rates_shielded;

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

/// Three groups with all the flux in the middle one, `[lo, hi]`, the top
/// one ending at 20 MeV or, for a group above that, at twice its top.
fn groups(lo: f64, hi: f64) -> ([f64; 4], [f64; 3]) {
    let top = if hi < 2.0e7 { 2.0e7 } else { 2.0 * hi };
    ([1.0e-5, lo, hi, top], [0.0, 1.0e14, 0.0])
}

/// The cached ENDF/B-VIII.1 evaluation with its own MF=33, or `None` when
/// either is missing.
fn material(name: &str, mat: i32) -> Option<(Material, Vec<CovarianceBlock>)> {
    material_in(yamc_test_cache::nuclide(name)?, name, mat)
}

/// The evaluation in `dir` with its own MF=33, or `None` when it has none.
///
/// `mat` is the evaluation's MAT, given to every block that does not carry
/// it, as a covariance.arrow written before the `mat` column does not. The
/// converter now writes it, and without it a cross block naming the own MAT
/// reads as another evaluation's.
fn material_in(dir: String, name: &str, mat: i32) -> Option<(Material, Vec<CovarianceBlock>)> {
    let mut blocks = yamc_nuclide::arrow::covariance_arrow::read_covariance(Path::new(&dir), name)
        .expect("covariance reads")?;
    for block in blocks.iter_mut().filter(|b| b.mat == 0) {
        block.mat = mat;
    }
    let mut m = Material::new(
        HashMap::from([(name.to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    m.set_temperature("294");
    m.read_nuclear_data(&HashMap::from([(name.to_string(), dir)]), None)
        .expect("read nuclide");
    let nuclide = Arc::make_mut(m.nuclide_data.get_mut(name).expect("loaded"));
    nuclide.covariance = Some(Arc::new(blocks.clone()));
    nuclide.load_scope.covariance = true;
    Some((m, blocks))
}

/// The relative sigma of `kind` on `name` with all the flux in `[lo, hi]`,
/// and the fold's report.
fn sigma(m: &Material, name: &str, kind: &str, lo: f64, hi: f64) -> (f64, Coverage) {
    let (bounds, flux) = groups(lo, hi);
    let chain = chain();
    let rates = compute_multigroup_reaction_rates_shielded(m, &chain, &flux, &bounds, 1.0, None).0;
    let (folded, coverage) = fold_rate_covariance(m, &chain, &rates, &flux, &bounds, None);
    let cov = &folded[name];
    let i = cov
        .kinds
        .iter()
        .position(|k| k == kind)
        .expect("channel folded");
    (cov.relative_std_devs()[i], coverage)
}

/// The NC block the tape states `mt`'s covariance with, as `(c_i, MT_i)`.
fn derivation(blocks: &[CovarianceBlock], mt: i32) -> Vec<(f64, i32)> {
    let nc = blocks
        .iter()
        .find_map(|b| match &b.data {
            CovarianceData::Nc(nc) if b.mt == mt => Some(nc),
            _ => None,
        })
        .expect("the tape derives this reaction");
    assert_eq!(nc.lty, 0);
    nc.ci
        .iter()
        .zip(&nc.xmti)
        .map(|(&c, &x)| (c, x as i32))
        .collect()
}

/// `∫ σ dE` of `mt` over `[lo, hi]`, by the trapezoid on its own points,
/// which is what a flat flux in one group weights with.
fn integral(m: &Material, name: &str, mt: i32, lo: f64, hi: f64) -> f64 {
    let by_mt = m.nuclide_data[name]
        .reactions_for_temp("294")
        .expect("294 K");
    let rx = &by_mt[&mt];
    let mut points = vec![lo];
    points.extend(rx.energy.iter().copied().filter(|&e| e > lo && e < hi));
    points.push(hi);
    let xs = |e: f64| rx.cross_section_at(e).unwrap_or(0.0);
    points
        .windows(2)
        .map(|w| 0.5 * (xs(w[0]) + xs(w[1])) * (w[1] - w[0]))
        .sum()
}

/// The relative covariance of `a` with `b` on the one cell of their block
/// that holds `[lo, hi]`, from whichever section the tape writes it in, or 0
/// when it writes none. Every block the test reads is relative and holds the
/// group inside one interval of each grid.
fn cell(blocks: &[CovarianceBlock], a: i32, b: i32, lo: f64, hi: f64) -> f64 {
    let holding = |grid: &[f64]| {
        grid.windows(2)
            .position(|w| w[0] <= lo && hi <= w[1])
            .expect("one interval holds the group")
    };
    // Both indices hold the same group, so the (b, a) block's cell is the
    // (a, b) covariance as it stands.
    for (row, col) in [(a, b), (b, a)] {
        let found: Vec<f64> = blocks
            .iter()
            .filter(|x| x.mt == row && x.partner_mt() == col && x.is_same_evaluation())
            .map(|x| {
                let CovarianceData::Ni(ni) = &x.data else {
                    panic!("MT {row} x {col} is not NI")
                };
                let e = expand_ni(ni).expect("expands");
                let (i, j) = (holding(&e.row_energies), holding(&e.col_energies));
                e.get(i, j)
            })
            .collect();
        if !found.is_empty() {
            return found.iter().sum();
        }
    }
    0.0
}

/// `sqrt(Σ_ij c_i c_j C_ij r_i r_j) / R`, the channel's relative sigma as
/// ENDF-102 33.3.2 a.3 has it derived, with every term by hand.
fn hand_sigma(
    m: &Material,
    blocks: &[CovarianceBlock],
    name: &str,
    mt: i32,
    lo: f64,
    hi: f64,
) -> f64 {
    let terms = derivation(blocks, mt);
    let mut variance = 0.0;
    for &(ci, a) in &terms {
        for &(cj, b) in &terms {
            variance += ci
                * cj
                * cell(blocks, a, b, lo, hi)
                * integral(m, name, a, lo, hi)
                * integral(m, name, b, lo, hi);
        }
    }
    variance.sqrt() / integral(m, name, mt, lo, hi)
}

fn close(got: f64, want: f64, what: &str) {
    assert!(want > 0.0, "{what}: nothing to compare with");
    assert!(
        ((got - want) / want).abs() < 1.0e-9,
        "{what}: fold {got}, hand {want}"
    );
}

/// ENDF/B-VIII.1 O16 `(n,p)` at 14 MeV.
#[test]
fn o16_np_is_derived_from_its_partials() {
    let Some((m, blocks)) = material("O16", 825) else {
        eprintln!("skipping: O16 fixture or its covariance.arrow missing");
        return;
    };
    assert_eq!(
        derivation(&blocks, 103),
        [(1.0, 600), (1.0, 601), (1.0, 602), (1.0, 603)]
    );
    let (lo, hi) = (1.39e7, 1.41e7);
    let (np, coverage) = sigma(&m, "O16", "(n,p)", lo, hi);
    assert!(
        !coverage.skipped_nc.contains_key("O16"),
        "{:?}",
        coverage.skipped_nc
    );
    let key = ("O16".to_string(), "(n,p)".to_string());
    assert_eq!(coverage.rate_fraction_covered[&key], 1.0);
    // 600 to 603 add up to MT 103 here, so the derivation holds.
    assert!(!coverage.partials_above_rate.contains_key(&key));
    assert!(!coverage.partials_below_rate.contains_key(&key));
    close(np, hand_sigma(&m, &blocks, "O16", 103, lo, hi), "O16 (n,p)");
    eprintln!(
        "ENDF/B-VIII.1 O16 (n,p), 13.9 to 14.1 MeV: {:.4}%",
        100.0 * np
    );
}

/// TENDL-2017 O16 states `(n,p)` the same way, as `600 + ... + 603`.
#[test]
fn tendl_2017_o16_np_is_derived_from_its_partials() {
    let dir = yamc_test_cache::root().join("tendl-2017-O16.arrow");
    if !yamc_test_cache::format_version_is_readable(&dir) {
        eprintln!("skipping: TENDL-2017 O16 fixture missing or stale");
        return;
    }
    let Some((m, blocks)) = material_in(dir.to_string_lossy().into_owned(), "O16", 825) else {
        eprintln!("skipping: TENDL-2017 O16 has no covariance.arrow");
        return;
    };
    assert_eq!(
        derivation(&blocks, 103),
        [(1.0, 600), (1.0, 601), (1.0, 602), (1.0, 603)]
    );
    let (lo, hi) = (1.39e7, 1.41e7);
    let (np, coverage) = sigma(&m, "O16", "(n,p)", lo, hi);
    assert!(
        !coverage.skipped_nc.contains_key("O16"),
        "{:?}",
        coverage.skipped_nc
    );
    close(np, hand_sigma(&m, &blocks, "O16", 103, lo, hi), "O16 (n,p)");
    eprintln!("TENDL-2017 O16 (n,p), 13.9 to 14.1 MeV: {:.4}%", 100.0 * np);
}

/// Above 20 MeV the ENDF/B-VIII.1 O16 `(n,d)` cross section holds MT 660 to
/// 669 as well, and its NC block names only 650 to 659, which carry the
/// covariance. The shortfall is the named reactions' rate over the
/// channel's, and is reported rather than read as a covered rate.
#[test]
fn o16_nd_above_20_mev_is_reported_short_of_its_rate() {
    let Some((m, blocks)) = material("O16", 825) else {
        eprintln!("skipping: O16 fixture or its covariance.arrow missing");
        return;
    };
    let named = derivation(&blocks, 104);
    assert!(named
        .iter()
        .all(|&(c, mt)| c == 1.0 && (650..=659).contains(&mt)));
    let (lo, hi) = (2.4e7, 2.6e7);
    let (_, coverage) = sigma(&m, "O16", "(n,d)", lo, hi);
    let want = named
        .iter()
        .map(|&(_, mt)| integral(&m, "O16", mt, lo, hi))
        .sum::<f64>()
        / integral(&m, "O16", 104, lo, hi);
    assert!(want < 0.99, "the named partials cover {want} of (n,d)");
    let got = coverage.partials_below_rate[&("O16".to_string(), "(n,d)".to_string())];
    assert!(
        ((got - want) / want).abs() < 1.0e-9,
        "fold {got}, hand {want}"
    );
    eprintln!("ENDF/B-VIII.1 O16 (n,d), 24 to 26 MeV: named partials are {got:.4} of the rate");
}

/// ENDF/B-VIII.1 B10 `(n,a)` is `800 + 801`, correlated by a cross block the
/// tape writes with `mat1` as B10's own MAT. Its MF=33 stops at 1.025 MeV, so
/// the group is thermal.
#[test]
fn b10_na_is_derived_from_its_correlated_partials() {
    let Some((m, blocks)) = material("B10", 525) else {
        eprintln!("skipping: B10 fixture or its covariance.arrow missing");
        return;
    };
    assert_eq!(derivation(&blocks, 107), [(1.0, 800), (1.0, 801)]);
    let (lo, hi) = (0.02, 0.03);
    assert_ne!(cell(&blocks, 800, 801, lo, hi), 0.0);
    let (na, coverage) = sigma(&m, "B10", "(n,a)", lo, hi);
    assert!(!coverage.skipped_nc.contains_key("B10"));
    close(na, hand_sigma(&m, &blocks, "B10", 107, lo, hi), "B10 (n,a)");
    eprintln!(
        "ENDF/B-VIII.1 B10 (n,a), 0.02 to 0.03 eV: {:.4}%",
        100.0 * na
    );
}
