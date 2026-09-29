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
/// when it writes none. A block whose grid ends below the group states
/// nothing there. Every block the test reads is relative and holds the group
/// inside one interval of each grid or outside all of them.
fn cell(blocks: &[CovarianceBlock], a: i32, b: i32, lo: f64, hi: f64) -> f64 {
    let holding = |grid: &[f64]| {
        let (first, last) = (grid[0], grid[grid.len() - 1]);
        if hi <= first || last <= lo {
            return None;
        }
        let k = grid
            .windows(2)
            .position(|w| w[0] <= lo && hi <= w[1])
            .expect("one interval holds the group");
        Some(k)
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
                match (holding(&e.row_energies), holding(&e.col_energies)) {
                    (Some(i), Some(j)) => e.get(i, j),
                    _ => 0.0,
                }
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

/// The O16 `(n,d)` shortfall is rate with no stated variance, so the share
/// of the rate the covariance covers falls with it rather than reading one.
#[test]
fn o16_nd_above_20_mev_is_not_counted_as_covered() {
    let Some((m, _)) = material("O16", 825) else {
        eprintln!("skipping: O16 fixture or its covariance.arrow missing");
        return;
    };
    let (_, coverage) = sigma(&m, "O16", "(n,d)", 2.4e7, 2.6e7);
    let key = ("O16".to_string(), "(n,d)".to_string());
    let ratio = coverage.partials_below_rate[&key];
    let share = coverage.rate_fraction_covered[&key];
    // 650 to 659 state a variance across the whole group, so what is left
    // uncovered is exactly the rate they miss.
    assert!(
        (share - ratio).abs() < 1.0e-9,
        "share {share}, named partials {ratio}"
    );
}

/// FENDL-3.2d and TENDL-2017 H2 `(n,2n)` is `σ_1 - σ_2 - σ_102` from
/// 3.339 MeV, a derived rate of millibarns from barns each rounded to its own
/// digits. The tape's derivation holds to that rounding, so it is not
/// reported as inconsistent.
///
/// The two tapes give MT 1, 2 and 102 self blocks and no cross blocks, and an
/// absent block states a zero covariance (ENDF-102 33.2), so the derived
/// variance is `Var(1) + Var(2) + Var(102)` of barn reactions over a
/// millibarn difference: about 2400% near threshold and 22% at 14 MeV on
/// TENDL-2017. That is the tape's literal statement, set by the correlations
/// it leaves out rather than by the fold, so it is pinned to the hand
/// sandwich rather than clamped. ENDF/B-VIII.1 and JEFF-4.0 give MT 16 its
/// own block instead.
#[test]
fn h2_n2n_cancelling_derivation_is_the_hand_sandwich() {
    for library in ["fendl-3.2d", "tendl-2017"] {
        // Cached as the activation subset of MT 1, 2, 16 and 102, which only
        // a load asking for some MTs reads.
        let dir = yamc_test_cache::root().join(format!("{library}-H2.arrow"));
        if !yamc_test_cache::format_version_is_readable(&dir) {
            eprintln!("skipping: {library} H2 fixture missing or stale");
            continue;
        }
        let scope = yamc_nuclide::load_scope::LoadScope::activation([1, 2, 16, 102].into())
            .with_temperatures(Some(["294".to_string()].into()))
            .with_covariance(true);
        let loaded = yamc_nuclide::nuclide::get_or_load_nuclide(
            "H2",
            &HashMap::from([("H2".to_string(), dir.to_string_lossy().into_owned())]),
            &scope,
        );
        let nd = loaded.expect("the cached subset loads");
        let Some(blocks) = nd.covariance.as_ref().map(|b| b.to_vec()) else {
            eprintln!("skipping: {library} H2 has no covariance.arrow");
            continue;
        };
        let mut m = Material::new(
            HashMap::from([("H2".to_string(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .expect("material");
        m.set_temperature("294");
        m.nuclide_data.insert("H2".to_string(), nd);
        assert_eq!(derivation(&blocks, 16), [(1.0, 1), (-1.0, 2), (-1.0, 102)]);
        let key = ("H2".to_string(), "(n,2n)".to_string());
        for (lo, hi) in [(3.4e6, 3.5e6), (1.39e7, 1.41e7)] {
            let (n2n, coverage) = sigma(&m, "H2", "(n,2n)", lo, hi);
            assert!(!coverage.skipped_nc.contains_key("H2"));
            assert!(
                !coverage.partials_above_rate.contains_key(&key)
                    && !coverage.partials_below_rate.contains_key(&key),
                "{lo} to {hi} eV: above {:?}, below {:?}",
                coverage.partials_above_rate.get(&key),
                coverage.partials_below_rate.get(&key)
            );
            close(n2n, hand_sigma(&m, &blocks, "H2", 16, lo, hi), "H2 (n,2n)");
            eprintln!("{library} H2 (n,2n), {lo} to {hi} eV: {:.4}%", 100.0 * n2n);
        }
    }
}

/// ENDF/B-VIII.1 O16 `(n,a)` is `800 + ... + 803` from 1e-5 eV to 150 MeV,
/// and MT 800's own grid ends at 20.5 MeV while 801 to 803 state variances to
/// 30 MeV. At 22 MeV the covered share is 801 to 803's rate only: 800 carries
/// rate there and states no variance for it.
#[test]
fn o16_na_above_800s_grid_covers_only_the_stating_partials() {
    let Some((m, blocks)) = material("O16", 825) else {
        eprintln!("skipping: O16 fixture or its covariance.arrow missing");
        return;
    };
    assert_eq!(
        derivation(&blocks, 107),
        [(1.0, 800), (1.0, 801), (1.0, 802), (1.0, 803)]
    );
    let (lo, hi) = (2.19e7, 2.21e7);
    let top_800 = blocks
        .iter()
        .filter(|b| b.mt == 800 && b.partner_mt() == 800)
        .filter_map(|b| match &b.data {
            CovarianceData::Ni(ni) => expand_ni(ni).ok(),
            CovarianceData::Nc(_) => None,
        })
        .flat_map(|e| e.row_energies.last().copied())
        .fold(0.0, f64::max);
    assert!(top_800 <= lo, "800's grid reaches {top_800} eV");
    assert!((801..=803).all(|mt| cell(&blocks, mt, mt, lo, hi) != 0.0));
    let (na, coverage) = sigma(&m, "O16", "(n,a)", lo, hi);
    let key = ("O16".to_string(), "(n,a)".to_string());
    let share = coverage.rate_fraction_covered[&key];
    let want = (801..=803)
        .map(|mt| integral(&m, "O16", mt, lo, hi))
        .sum::<f64>()
        / integral(&m, "O16", 107, lo, hi);
    assert!(want < 0.95, "801 to 803 carry {want} of (n,a)");
    assert!(
        ((share - want) / want).abs() < 1.0e-6,
        "share {share}, stating partials {want}"
    );
    close(na, hand_sigma(&m, &blocks, "O16", 107, lo, hi), "O16 (n,a)");
    eprintln!(
        "ENDF/B-VIII.1 O16 (n,a), 21.9 to 22.1 MeV: share {share:.4}, sigma {:.4}%",
        100.0 * na
    );
}

/// Through the load an ordinary standalone run makes, which asks for the
/// chain's MTs only: the partials O16 `(n,p)` and B10 `(n,a)` are derived
/// from must be read as well, or both channels fold to nothing.
#[test]
fn the_activation_load_reads_the_reactions_a_channel_is_derived_from() {
    let chain = chain();
    let branch = yani::BranchTable::new();
    let mut m = Material::new(
        HashMap::from([("O16".to_string(), 0.5), ("B10".to_string(), 0.5)]),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    m.set_temperature("294");
    let scope = yamc_nuclide::load_scope::LoadScope::activation(yani_transmute::activation_mts(
        &chain, &branch,
    ))
    .with_temperatures(Some(["294".to_string()].into()))
    .with_covariance(true);
    for name in ["O16", "B10"] {
        let Some(dir) = yamc_test_cache::nuclide(name) else {
            eprintln!("skipping: {name} fixture missing");
            return;
        };
        let nd = yamc_nuclide::nuclide::get_or_load_nuclide(
            name,
            &HashMap::from([(name.to_string(), dir)]),
            &scope,
        )
        .expect("loads");
        if nd.covariance.is_none() {
            eprintln!("skipping: {name} has no covariance.arrow");
            return;
        }
        m.nuclide_data.insert(name.to_string(), nd);
    }
    let request = yani_transmute::uncertainty::DataUncertainty {
        sources: vec![yani_transmute::uncertainty::Source::CrossSections],
        ..Default::default()
    };
    yani_transmute::preload_activation_data(&mut m, &chain, &branch, Some(&request), None)
        .expect("preload");
    let held = |name: &str, mt: i32| {
        m.nuclide_data[name]
            .reactions_for_temp("294")
            .expect("294 K")
            .contains_key(&mt)
    };
    assert!((600..=603).all(|mt| held("O16", mt)));
    assert!(held("B10", 800) && held("B10", 801));

    let (lo, hi) = (1.39e7, 1.41e7);
    let (np, coverage) = sigma(&m, "O16", "(n,p)", lo, hi);
    assert!(
        !coverage.skipped_nc.contains_key("O16"),
        "{:?}",
        coverage.skipped_nc
    );
    assert_eq!(
        coverage.rate_fraction_covered[&("O16".to_string(), "(n,p)".to_string())],
        1.0
    );
    assert!(np > 0.0);
    eprintln!(
        "ENDF/B-VIII.1 O16 (n,p) on the activation load, 13.9 to 14.1 MeV: {:.4}%",
        100.0 * np
    );
    let (na, coverage) = sigma(&m, "B10", "(n,a)", 0.02, 0.03);
    assert!(
        !coverage.skipped_nc.contains_key("B10"),
        "{:?}",
        coverage.skipped_nc
    );
    assert!(na > 0.0);
}

/// The refusal of a spectrum above the evaluation reads the top of the MTs a
/// collapse loads, so a partial an uncertainty run adds for a derivation
/// cannot move it even where the partial runs past every chain MT.
#[test]
fn a_derivation_partial_leaves_the_evaluation_top_alone() {
    let Some((mut m, _)) = material("O16", 825) else {
        eprintln!("skipping: O16 fixture or its covariance.arrow missing");
        return;
    };
    let mts = yani_transmute::activation_mts(&chain(), &yani::BranchTable::new());
    assert!(!mts.contains(&600));
    let top = |m: &Material| {
        m.nuclide_data["O16"]
            .reactions_for_temp("294")
            .expect("294 K")
            .iter()
            .filter(|(mt, _)| mts.contains(mt))
            .filter_map(|(_, r)| r.energy.last().copied())
            .fold(0.0, f64::max)
    };
    let evaluated = top(&m);
    // All the flux in one group above the evaluation's top.
    let bounds = [1.0e-5, evaluated, 4.0 * evaluated];
    let flux = [0.0, 1.0];
    let above = |m: &Material| {
        yani_transmute::multigroup::spectrum_above_evaluation(m, &flux, &bounds, &mts)
    };
    let before = above(&m);
    assert_eq!(before.len(), 1, "{before:?}");
    let nuclide = Arc::make_mut(m.nuclide_data.get_mut("O16").expect("loaded"));
    let by_mt = nuclide.reactions.first_mut().expect("a temperature");
    let mut partial = (*by_mt[&600]).clone();
    let mut energy = partial.energy.to_vec();
    let mut xs = partial.cross_section.to_vec();
    energy.push(8.0 * evaluated);
    xs.push(*xs.last().expect("points"));
    partial.energy = energy.into();
    partial.cross_section = xs.into();
    by_mt.insert(600, Arc::new(partial));
    assert_eq!(above(&m), before);
}
