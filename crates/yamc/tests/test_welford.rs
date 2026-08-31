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
    tally.initialize_batches(batches);

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
    // Under Stage 3, n_realizations is the count of source histories,
    // not the count of batches.
    let expected = (particles * batches) as u32;
    assert_eq!(
        tally.get_n_realizations(),
        expected,
        "Stage 3: n_realizations should equal particles × batches (={expected})"
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
/// Reference recomputed for the issue #111 free-flight migration: the neutron
/// free-flight distance for a survival-off, single-nuclide, non-URR collision
/// now draws its uniform from the shared per-particle PCG stream instead of the
/// legacy FastRng stream (matching the GPU kernel's first per-step sample).
/// That renumbers the FastRng stream this fixed-seed Li6 sim consumes, shifting
/// the (n,t)-rate realization by ~0.49% (a draw-order change, not a physics
/// change -- the flight is still `-ln(xi) / sigma_t`). The mean is independent
/// of the variance estimator, so this is the default-build value. (The prior
/// value 0.9978143913613567 was the endf-b8.1 fixture's pre-flight-migration
/// result.)
///
/// Recomputed again for issue #274: the shared collision PCG widened from
/// 32-bit to 64-bit state (PCG-XSH-RR 64/32, per-history seed expanded via
/// splitmix64), which renumbers every fixed-seed stream. Same unbiased
/// physics, different realization (the prior 32-bit-stream value was
/// 1.0027406040118252).
///
/// Recomputed a third time for issue #315: the per-history collision seed is
/// now `history_seed(base_seed, global_index)` instead of `global_index *
/// golden`, i.e. the base seed finally reaches the collision stream, so every
/// fixed-seed stream is renumbered again. Nothing about the sampling changed,
/// only which stream each history draws from, and this 10 000-history mean is
/// one realization of it (`std_dev` here is 7.8e-3, so a re-draw moves the
/// value by ~1 sigma routinely).
///
/// Confirmed unbiased before recalibrating, on this exact model and tally:
/// 8 runs x 2 000 000 histories give a pooled mean of 0.99970860 before the
/// change and 1.00015215 after, a shift of +4.4e-4 against a standard error of
/// 2.5e-4 on each, i.e. 1.3 combined sigma. (Deliberately NOT checked with a
/// multi-seed spread, which is the estimator #315 fixed: before the fix those
/// 8 seeds spread by only 2.8e-6, a 250x under-estimate of the true 7.1e-4
/// per-run error, because they all shared one collision realization.)
/// The prior value was 0.9857592350441496.
///
/// Recomputed a fourth time for issue #126: the Li6 fixture is now the
/// PUBLISHED endf-b8.1 data (downloaded into `~/.cache/yamc`) instead of the
/// April-2026 copy that used to be committed here. The two differ only in the
/// last ulp of the energy grid (the published grid starts at exactly 1e-5 eV,
/// the retired one at 9.999999999999999e-6), which moves this mean by 3.1e-10
/// relative -- far below the 7.8e-3 statistical spread, but above the 1e-12
/// this test asserts. The prior value was 1.0101426555071467.
///
/// Recomputed a fifth time for the August-2026 re-upload of endf-b8.1, which
/// lands on exactly the pre-#126 value: 1.0101426555071467, bit for bit. The
/// grid ulp #126 recalibrated for has gone back to what the retired committed
/// copy had, so this is the same 3.1e-10 step in reverse rather than a new
/// realization. Nothing else in the tally moves.
#[test]
fn mean_matches_default_reference() {
    let tally = run_li6_sim(42, 1000, 10);
    let mean = tally.get_mean()[0];
    // Default-build reference (published endf-b8.1 Li6 data, #315 per-history
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
    // approximately std_dev / sqrt(2(n-1)). The default uses
    // n=10 batches; that's the looser bound.
    let n_batches: f64 = 10.0;
    let std_err_of_std = default_std / (2.0_f64 * (n_batches - 1.0)).sqrt();
    let diff = (welford_std - default_std).abs();
    let n_sigma = diff / std_err_of_std;
    assert!(
        n_sigma < 5.0,
        "Welford std_dev = {welford_std:e} vs default {default_std:e}; \
         deviation {n_sigma:.2}σ exceeds 5σ tolerance"
    );
}
