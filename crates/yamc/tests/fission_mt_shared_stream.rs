//! Issue #418, divergence (a): selecting among the partial fission MTs
//! (18/19/20/21/38) was the last collision-path draw still coming off `FastRng`
//! instead of the per-particle PCG state the rest of the path moved to.
//!
//! The draw is GATED on `has_partial_fission`, and that gate is the load-bearing
//! part. The GPU kernel sums all five fission MTs into one cross section
//! (`crates/yamc-gpu/src/neutron/xs/extract.rs`) and makes no channel-selection
//! draw at all, so a uniform drawn here for a single-channel nuclide would walk
//! the CPU's stream one step off the kernel's schedule at every fission
//! collision, for no physics. `has_partial_fission` is a property of the
//! evaluation rather than of the history, so gating on it keeps the CPU stream
//! position deterministic while leaving all 87 single-channel fissionables in
//! ENDF/B-VIII.1 byte-identical.
//!
//! U240 is the only nuclide that exercises the draw: a scan of the 553 cached
//! endf-b8.1 nuclides finds 88 with a fission MT and exactly one with
//! non-redundant partial channels.

use std::cell::Cell;
use yamc_nuclide::arrow::nuclide_arrow::read_nuclide_from_arrow;
use yamc_nuclide::nuclide::Nuclide;
use yamc_nuclide::LoadScope;

const DATA: &str = "tests/U240.arrow";
const E_14MEV: f64 = 14.06e6;

fn load() -> Option<Nuclide> {
    if !std::path::Path::new(DATA).exists() {
        eprintln!("skipping: {DATA} absent (run scripts/fetch_test_fixtures.py)");
        return None;
    }
    Some(read_nuclide_from_arrow(std::path::Path::new(DATA), &LoadScope::full()).expect("U240"))
}

/// A counting draw source, so a test can assert how many uniforms a call spends
/// as well as which channel it lands on.
fn counting(value: f64, calls: &Cell<usize>) -> impl FnOnce() -> f64 + '_ {
    move || {
        calls.set(calls.get() + 1);
        value
    }
}

/// A nuclide with partial channels spends exactly one uniform per fission.
#[test]
fn a_partial_fission_nuclide_spends_one_draw() {
    let Some(n) = load() else { return };
    let ti = n.get_temp_idx("294").expect("294 K");
    let grid = &n.fast_xs[ti];
    assert!(grid.has_partial_fission, "U240 carries partial fission");

    let calls = Cell::new(0);
    let picked = grid.sample_fission_reaction(E_14MEV, counting(0.5, &calls));
    assert_eq!(calls.get(), 1, "exactly one uniform per partial fission");
    assert!(picked.is_some());
}

/// The gate: with no partial channels the call spends NO uniform, which is what
/// keeps the 87 single-channel fissionables on the kernel's draw schedule. U240's
/// own grid with the flag cleared stands in for them, since the fixture set has
/// no other fissionable.
#[test]
fn a_single_channel_nuclide_spends_no_draw() {
    let Some(n) = load() else { return };
    let ti = n.get_temp_idx("294").expect("294 K");
    let mut grid = n.fast_xs[ti].clone();
    grid.has_partial_fission = false;

    let calls = Cell::new(0);
    let picked = grid.sample_fission_reaction(E_14MEV, counting(0.5, &calls));
    assert_eq!(
        calls.get(),
        0,
        "a single-channel fissionable must not touch the shared stream"
    );
    assert_eq!(
        picked.expect("first channel").mt_number,
        grid.fission_mt_numbers[0],
        "the single channel is returned without sampling"
    );
}

/// A non-fissionable nuclide spends no uniform either, so the stream position
/// does not depend on which nuclide a collision landed on.
#[test]
fn a_non_fissionable_nuclide_spends_no_draw() {
    let Some(n) = load() else { return };
    let ti = n.get_temp_idx("294").expect("294 K");
    let mut grid = n.fast_xs[ti].clone();
    grid.fission_mt_numbers.clear();
    grid.fission_mt_reactions.clear();

    let calls = Cell::new(0);
    assert!(grid
        .sample_fission_reaction(E_14MEV, counting(0.5, &calls))
        .is_none());
    assert_eq!(calls.get(), 0);
}

/// The walk is proportional and monotone in the uniform: the smallest uniform
/// lands on the first open channel and the largest on the last, so the whole
/// open set is reachable and none of it is dead.
#[test]
fn the_walk_spans_the_open_channels() {
    let Some(n) = load() else { return };
    let ti = n.get_temp_idx("294").expect("294 K");
    let grid = &n.fast_xs[ti];
    let calls = Cell::new(0);

    let lowest = grid
        .sample_fission_reaction(E_14MEV, counting(f64::MIN_POSITIVE, &calls))
        .expect("open at 14.06 MeV")
        .mt_number;
    let highest = grid
        .sample_fission_reaction(E_14MEV, counting(1.0, &calls))
        .expect("open at 14.06 MeV")
        .mt_number;

    assert_eq!(
        lowest, 19,
        "the smallest uniform takes first-chance fission"
    );
    assert!(
        highest > lowest,
        "the largest uniform must reach a later channel, got {highest} against {lowest}"
    );
    assert_eq!(calls.get(), 2);
}

/// Sampling is proportional to the partial cross sections. At 14.06 MeV MT 38 is
/// still shut (it opens at 14.34 MeV), so it must never be selected however the
/// uniform falls, and every channel that is open must be reachable.
#[test]
fn selection_is_proportional_and_skips_shut_channels() {
    let Some(n) = load() else { return };
    let ti = n.get_temp_idx("294").expect("294 K");
    let grid = &n.fast_xs[ti];
    let calls = Cell::new(0);

    let mut seen = std::collections::BTreeMap::new();
    const N: usize = 2001;
    for i in 1..=N {
        let xi = i as f64 / (N as f64 + 1.0);
        let mt = grid
            .sample_fission_reaction(E_14MEV, counting(xi, &calls))
            .expect("open at 14.06 MeV")
            .mt_number;
        *seen.entry(mt).or_insert(0usize) += 1;
    }
    assert_eq!(calls.get(), N);

    println!("U240 channel shares at 14.06 MeV: {seen:?}");
    assert!(
        !seen.contains_key(&38),
        "MT 38 opens at 14.34 MeV and must not be sampled at 14.06, got {seen:?}"
    );
    assert!(
        seen.len() > 1,
        "more than one channel is open at 14.06 MeV, got {seen:?}"
    );

    // The shares must match the partial cross sections, which is the property
    // the cumulative walk exists to give. A stratified sweep of the unit
    // interval reproduces them to within one bin.
    let (i_grid, f) = grid.lookup_grid_index(E_14MEV);
    let total: f64 = (0..grid.fission_mt_numbers.len())
        .map(|j| grid.fission_xs_interp(i_grid, f, j).max(0.0))
        .sum();
    for (j, &mt) in grid.fission_mt_numbers.iter().enumerate() {
        let expected = grid.fission_xs_interp(i_grid, f, j).max(0.0) / total;
        let got = *seen.get(&mt).unwrap_or(&0) as f64 / N as f64;
        assert!(
            (got - expected).abs() < 2.0 / N as f64,
            "MT {mt} share {got:.5} against cross-section fraction {expected:.5}"
        );
    }
}

/// A fixed uniform picks a fixed channel regardless of how many times the grid
/// has been sampled before, so a run is reproducible from its seed.
#[test]
fn selection_is_reproducible() {
    let Some(n) = load() else { return };
    let ti = n.get_temp_idx("294").expect("294 K");
    let grid = &n.fast_xs[ti];
    let calls = Cell::new(0);

    let first = grid
        .sample_fission_reaction(E_14MEV, counting(0.73, &calls))
        .map(|r| r.mt_number);
    for _ in 0..64 {
        let again = grid
            .sample_fission_reaction(E_14MEV, counting(0.73, &calls))
            .map(|r| r.mt_number);
        assert_eq!(again, first, "a fixed uniform must pick a fixed channel");
    }
}
