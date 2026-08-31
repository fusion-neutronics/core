//! Issues #378 / #382: a collision-estimator photon heating score must apply an
//! energy function as a WEIGHT, not merely as a gate.
//!
//! The analog photon-heat arm of `score_collision` used to return the deposited
//! eV straight through, so an `EnergyFunctionFilter` on such a tally dropped
//! out-of-range collisions (the gate) but never scaled the ones it kept (the
//! weight). The track-length arm applied both, so the same tally answered
//! differently based only on the estimator.
//!
//! The fix separates the two factors that were previously fused into
//! `base_weight`: `weight / Sigma_t`, which converts a collision into a
//! track-length equivalent and must NOT touch an already-deposited eV value, and
//! `f(E)`, the user's response function, which applies to every score.
//!
//! These tests drive `score_collision` directly. `total_xs = 0` with a
//! `photon_heat_score_ev` is exactly how `transport/mod.rs:720` dispatches the
//! analog deposit, so this is the real arm rather than a stand-in.

use yamc_particle::ParticleType;
use yamc_tallies::filter::energy_function::EnergyFunctionFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::welford::WelfordWorkerState;
use yamc_tallies::{Estimator, Tally};

const DEPOSIT_EV: f64 = 1234.5;
/// Inside the table below and away from its knots, so interpolation is exercised.
const TEST_ENERGY: f64 = 5.0e4;

/// A response table that is strongly energy dependent, so a missing or misplaced
/// `f(E)` cannot pass by being close to constant.
fn sloped_table() -> Filter {
    Filter::EnergyFunction(EnergyFunctionFilter::new(
        vec![1.0e3, 1.0e4, 1.0e5, 1.0e6],
        vec![1.0, 10.0, 100.0, 1000.0],
    ))
}

/// A flat response, whose exact effect is known independently of any
/// interpolation scheme: every score must scale by precisely this factor.
fn flat_table(value: f64) -> Filter {
    Filter::EnergyFunction(EnergyFunctionFilter::new(
        vec![1.0e3, 1.0e4, 1.0e5, 1.0e6],
        vec![value; 4],
    ))
}

fn heating_tally(filters: Vec<Filter>) -> Tally {
    let mut t = Tally::new();
    t.scores = vec!["heating".parse().expect("heating score")];
    t.estimator = Estimator::Collision;
    t.filters = filters;
    t
}

/// Score one analog photon-heat deposit and return what landed in the tally.
///
/// Contributions accumulate into the worker's per-history scratch map, so
/// summing it gives the score for this single event.
fn score_analog_heat(tally: &Tally, energy: f64) -> f64 {
    let mut worker = WelfordWorkerState::new(&[tally.num_bins()]);
    tally.score_collision(
        &mut worker,
        0,
        1.0, // particle weight
        0.0, // total_xs = 0, exactly how the analog deposit is dispatched
        [0.0, 0.0, 0.0],
        None, // no cell filter
        None, // no material
        None,
        None, // no URR
        None, // no photon XS
        energy,
        ParticleType::Photon,
        None,
        Some(DEPOSIT_EV),
    );
    worker.tallies[0].scratch_map.values().sum()
}

/// A flat response of `c` must scale the deposit by exactly `c`. This is the
/// core of the fix and does not depend on the interpolation scheme at all.
#[test]
fn a_flat_energy_function_scales_the_deposit_exactly() {
    for c in [0.5_f64, 2.0, 137.0] {
        let scored = score_analog_heat(&heating_tally(vec![flat_table(c)]), TEST_ENERGY);
        let expected = DEPOSIT_EV * c;
        assert!(
            (scored - expected).abs() <= 1.0e-9 * expected.abs(),
            "a flat energy function of {c} must scale the analog photon-heat deposit \
             by exactly {c}: expected {expected}, scored {scored} (#382)"
        );
    }
}

/// Without an energy function the deposit is unchanged, so the weighting cannot
/// be leaking into tallies that never asked for it.
#[test]
fn no_energy_function_leaves_the_deposit_alone() {
    let scored = score_analog_heat(&heating_tally(Vec::new()), TEST_ENERGY);
    assert!(
        (scored - DEPOSIT_EV).abs() <= 1.0e-9 * DEPOSIT_EV,
        "with no energy function the analog deposit must pass through unchanged: \
         expected {DEPOSIT_EV}, scored {scored}"
    );
}

/// `f` must be evaluated at the incident energy, so two different energies must
/// give two different factors that track the table. This is what makes the
/// collision arm an estimator of the same integral as the track-length arm: both
/// evaluate `f` at the pre-collision energy.
#[test]
fn the_weight_follows_the_incident_energy() {
    let tally = heating_tally(vec![sloped_table()]);
    // Both are table knots, where the expected value is unambiguous.
    let at_1e4 = score_analog_heat(&tally, 1.0e4) / DEPOSIT_EV;
    let at_1e5 = score_analog_heat(&tally, 1.0e5) / DEPOSIT_EV;
    assert!(
        (at_1e4 - 10.0).abs() < 1.0e-6,
        "at a table knot the weight must be the tabulated value 10, got {at_1e4}"
    );
    assert!(
        (at_1e5 - 100.0).abs() < 1.0e-6,
        "at a table knot the weight must be the tabulated value 100, got {at_1e5}"
    );
}

/// The deposit must take `f(E)` but NOT `weight / Sigma_t`. That is the whole
/// reason the two factors had to be separated: the analog value is already an
/// energy deposit, so dividing it by a cross section would be a unit error.
/// Scoring with `total_xs = 0` (as the real dispatch does) and a flat response
/// pins this: the result is `deposit * c` and nothing else.
#[test]
fn the_deposit_does_not_take_the_flux_factor() {
    let scored = score_analog_heat(&heating_tally(vec![flat_table(3.0)]), TEST_ENERGY);
    assert!(
        (scored - DEPOSIT_EV * 3.0).abs() <= 1.0e-9 * DEPOSIT_EV * 3.0,
        "the analog deposit must be scaled by f(E) alone, with no weight/Sigma_t \
         factor: expected {}, scored {scored} (#382)",
        DEPOSIT_EV * 3.0
    );
}

/// Out of the table's range the whole event is still dropped. The gate was the
/// one half of this that already worked, and it must survive the fix.
#[test]
fn out_of_range_still_drops_the_event() {
    let tally = heating_tally(vec![sloped_table()]);
    for energy in [1.0_f64, 1.0e9] {
        let scored = score_analog_heat(&tally, energy);
        assert_eq!(
            scored, 0.0,
            "an energy of {energy} eV is outside the table, so the event must score \
             nothing at all, got {scored}"
        );
    }
}

/// `heating-local` shares the arm and must be weighted identically.
#[test]
fn heating_local_is_weighted_too() {
    let mut t = heating_tally(vec![flat_table(4.0)]);
    t.scores = vec!["heating-local".parse().expect("heating-local score")];
    let scored = score_analog_heat(&t, TEST_ENERGY);
    assert!(
        (scored - DEPOSIT_EV * 4.0).abs() <= 1.0e-9 * DEPOSIT_EV * 4.0,
        "heating-local shares the analog photon-heat arm, so it must take the same \
         weighting: expected {}, scored {scored}",
        DEPOSIT_EV * 4.0
    );
}
