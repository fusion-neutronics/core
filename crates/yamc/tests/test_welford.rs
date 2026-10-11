//! Welford of the variance-algorithm migration: per-history Welford
//! with per-rayon-worker private accumulator + lazy zero-fold at
//! finalize. See auto-memory `project-variance-welford-decision`.
//!
//! These tests run ONLY under `--features welford`. They verify:
//!
//! 1. `mean` matches the default-features (per-batch) build to ~1e-12
//!    relative. Algebraically the same total contributions per bin
//!    divided by the same total particle count, only the summation
//!    order differs.
//! 2. `std_dev` / `rel_err` differ from the default by sampling
//!    variance (per-history estimator vs per-batch estimator) but
//!    both converge to the same `Var(mean_estimate)`. We use a 5σ
//!    tolerance band based on the per-batch std-error-of-std-error.
//! 3. `get_n_realizations()` returns the count of source histories
//!    processed (`particles × batches`), not the batch count. This
//!    is the realization-unit change Welford introduces.
use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{Mt, ReactionRateScore, Score, Tally};
use yamc_tallies::CellFilter;

fn run_li6_sim(seed: u64, particles: usize, batches: usize) -> Arc<Tally> {
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 200.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            surf.clone(),
        )))),
    };
    let mut material = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.534),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();

    let mat_arc = Arc::new(material);
    let cell = Cell::new(Some(1), region, Some("test_cell".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![mat_arc]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let cell_filter = Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap()));
    let mut tally = Tally::new();
    tally.filters = vec![cell_filter];
    tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        105,
    )))];
    tally.name = Some("welford_tally".to_string());
    tally.reset_accumulation();

    let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles * batches),
            seed,
            ..Default::default()
        })
        .unwrap();
    Arc::clone(&model.tallies[0])
}

#[test]
fn realization_unit_is_per_history() {
    let particles = 1000;
    let batches = 10;
    let tally = run_li6_sim(42, particles, batches);
    // n_realizations is the count of source histories, not the count of
    // batches.
    let expected = (particles * batches) as u32;
    assert_eq!(
        tally.get_n_realizations(),
        expected,
        "n_realizations should equal particles × batches (={expected})"
    );
}

#[test]
fn mean_is_nonzero_and_reasonable() {
    let tally = run_li6_sim(42, 1000, 10);
    let mean = tally.get_mean();
    assert_eq!(mean.len(), 1);
    // For 1 MeV neutrons in pure Li6 with MT 105 (n,t), expect a
    // reaction probability ~O(1) per source particle (Li6 is highly
    // tritium-producing in this energy range).
    assert!(
        mean[0] > 0.5 && mean[0] < 2.0,
        "mean is wildly out of range: {}",
        mean[0]
    );
}

#[test]
fn std_dev_is_positive_and_reasonable() {
    let tally = run_li6_sim(42, 1000, 10);
    let mean = tally.get_mean()[0];
    let std = tally.get_std_dev()[0];
    let rel = std / mean;
    assert!(
        std > 0.0,
        "std should be positive for a stochastic simulation"
    );
    // Relative error for 10k histories on a high-probability event
    // should be small.
    assert!(rel > 0.0 && rel < 0.1, "relative error out of range: {rel}");
}

/// Welford mean should match a same-seed same-particles default-features
/// run to floating-point roundoff (~1e-12 relative).
///
/// We can't directly run the default-features path from this test (the
/// binary's `welford` cfg is build-time), but we can compare
/// against a hard-coded reference taken from the default build of the
/// same Li6 fixed-seed sim.
///
/// The reference is one realization of the fixed-seed stream (`std_dev` here
/// is 7.8e-3), so any change to draw order or stream seeding (the free-flight
/// draw, the per-history collision PCG and its seeding) renumbers the streams
/// and needs a recalibration, as does a change to the published Li6 data
/// (even a last-ulp change to the energy grid moves the mean by ~3e-10, above
/// the 1e-12 asserted here). Before recalibrating, confirm the shift is
/// statistical with long independent runs (e.g. 8 x 2 000 000 histories on
/// this model and tally), not a multi-seed spread of short runs.
#[test]
fn mean_matches_default_reference() {
    let tally = run_li6_sim(42, 1000, 10);
    let mean = tally.get_mean()[0];
    // Default-build reference (published endf-b8.1 Li6 data, per-history
    // seeding).
    let reference = 1.0101426555071467;
    let diff = (mean - reference).abs();
    let rel = diff / reference.abs().max(mean.abs()).max(1e-18);
    assert!(
        rel < 1e-12,
        "Welford mean = {mean:e} vs reference {reference:e}, rel diff {rel:e} > 1e-12"
    );
}

/// Welford std_dev is a different statistical estimator than the
/// default per-batch std_dev (different sample size, different
/// realization unit). They estimate the same population quantity
/// `Var(mean_estimate)` and should agree within statistical tolerance.
/// Tolerance: 5σ on the per-batch estimator's own standard error.
#[test]
fn std_dev_within_statistical_tolerance_of_default() {
    let tally = run_li6_sim(42, 1000, 10);
    let welford_std = tally.get_std_dev()[0];
    let default_std = 0.007761421038803751_f64;
    // Standard error of the std_dev estimator on a sample of size n:
    // approximately std_dev / sqrt(2(n-1)). The reference value was
    // estimated from n=10 per-batch samples; that's the looser bound.
    let n_ref_samples: f64 = 10.0;
    let std_err_of_std = default_std / (2.0_f64 * (n_ref_samples - 1.0)).sqrt();
    let diff = (welford_std - default_std).abs();
    let n_sigma = diff / std_err_of_std;
    assert!(
        n_sigma < 5.0,
        "Welford std_dev = {welford_std:e} vs default {default_std:e}; \
         deviation {n_sigma:.2}σ exceeds 5σ tolerance"
    );
}
