//! Real two-pass variance guard for the per-history Welford estimator.
//!
//! Feeds a known, hand-authored set of per-history per-bin samples
//! through the *production* Welford path
//! (`WelfordWorkerState::add_contribution` -> `finish_history` ->
//! `combine` -> `finalize` -> `Tally::install_finalized`) and then
//! checks the tally read accessors (`get_mean`, `get_std_dev`,
//! `total_std`) against a genuinely independent two-pass computation
//! over the same samples:
//!
//!   pass 1: mean = Σx / N
//!   pass 2: M2   = Σ(x − mean)²       → var_of_mean = M2 / ((N−1)·N)
//!
//! This is independent of Welford's online `m2`, unlike the old
//! sum/sum_sq cross-check (which derived its "reference" from the same
//! `m2` it was meant to validate, and so could only ever confirm
//! floating-point associativity). The samples deliberately include a
//! sparse bin (untouched in some histories) so the lazy zero-fold in
//! `finalize` is exercised, and the histories are split across two
//! workers so the Chen pairwise `combine` is exercised too.

use std::sync::Arc;
use yamc_tallies::tally::Tally;
use yamc_tallies::welford::WelfordWorkerState;

/// Two-pass mean over a full per-history sample vector.
fn two_pass_mean(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Two-pass sum of squared deviations over a full per-history sample
/// vector (the textbook second pass).
fn two_pass_m2(samples: &[f64]) -> f64 {
    let mean = two_pass_mean(samples);
    samples.iter().map(|&x| (x - mean) * (x - mean)).sum()
}

/// Two-pass standard error of the mean: `sqrt(M2 / ((N−1)·N))`.
fn two_pass_std_err(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n <= 1 {
        return 0.0;
    }
    let nf = n as f64;
    (two_pass_m2(samples) / ((nf - 1.0) * nf)).sqrt()
}

/// Feed a slice of histories into a fresh single worker. Each history
/// is a list of `(bin, value)` contributions; bins absent from a
/// history are left untouched (implicit zero, folded in at finalize).
fn feed(histories: &[Vec<(usize, f64)>], num_bins: usize) -> WelfordWorkerState {
    let mut w = WelfordWorkerState::new(&[num_bins]);
    for contribs in histories {
        for &(bin, value) in contribs {
            w.add_contribution(0, bin, value);
        }
        w.finish_history();
    }
    w
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    let scale = expected.abs().max(actual.abs()).max(1e-18);
    let rel = (actual - expected).abs() / scale;
    assert!(
        rel < 1e-12,
        "{what}: got {actual:e}, two-pass reference {expected:e} (rel diff {rel:e})"
    );
}

#[test]
fn welford_getters_match_two_pass_reference() {
    const NUM_BINS: usize = 3;

    // Hand-authored per-history contributions. Bins 0 and 1 are touched
    // every history; bin 2 is sparse (untouched in histories 1, 3, 5),
    // so its full sample vector carries implicit zeros there.
    let histories: Vec<Vec<(usize, f64)>> = vec![
        vec![(0, 2.0), (1, 1.5), (2, 7.0)],
        vec![(0, 4.0), (1, 1.0)],
        vec![(0, 1.0), (1, 2.5), (2, 3.0)],
        vec![(0, 3.0), (1, 2.0)],
        vec![(0, 5.0), (1, 0.5), (2, 5.0)],
        vec![(0, 2.5), (1, 1.0)],
    ];
    let n = histories.len();

    // Independent reference: full per-bin sample vectors (one entry per
    // history, zero where the bin was untouched).
    let mut samples = vec![vec![0.0_f64; n]; NUM_BINS];
    for (h, contribs) in histories.iter().enumerate() {
        for &(bin, value) in contribs {
            samples[bin][h] += value;
        }
    }

    // Production path: split the histories across two workers (3 + 3),
    // Chen-combine, finalize (lazy zero-fold), and install on a tally.
    let combined = feed(&histories[..3], NUM_BINS)
        .combine(feed(&histories[3..], NUM_BINS))
        .finalize();
    let stats = combined
        .per_tally
        .into_iter()
        .next()
        .expect("one tally in worker state");

    let tally = Arc::new(Tally::new());
    tally.install_finalized(stats);

    assert_eq!(tally.get_n_realizations() as usize, n);

    // --- Per-bin mean ---
    let mean = tally.get_mean();
    assert_eq!(mean.len(), NUM_BINS);
    for bin in 0..NUM_BINS {
        assert_close(
            mean[bin],
            two_pass_mean(&samples[bin]),
            &format!("mean[{bin}]"),
        );
    }

    // --- Per-bin standard error of the mean ---
    let std_dev = tally.get_std_dev();
    assert_eq!(std_dev.len(), NUM_BINS);
    for bin in 0..NUM_BINS {
        assert_close(
            std_dev[bin],
            two_pass_std_err(&samples[bin]),
            &format!("std_dev[{bin}]"),
        );
    }

    // --- Aggregate standard deviation of the summed bin total ---
    // var(total) = Σ_bins var_of_mean(bin), assuming bins independent.
    let nf = n as f64;
    let total_std_reference = (0..NUM_BINS)
        .map(|bin| two_pass_m2(&samples[bin]) / ((nf - 1.0) * nf))
        .sum::<f64>()
        .sqrt();
    assert_close(tally.total_std(), total_std_reference, "total_std");
}
