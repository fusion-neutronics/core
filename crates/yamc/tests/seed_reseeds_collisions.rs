//! Issue #315 regression: `TransportSettings::seed` must reseed the COLLISION
//! physics, not just the source sampling.
//!
//! The per-history collision PCG used to be seeded from the history index
//! alone (`expand_seed(global_index * golden)`), with no base-seed term. So
//! re-running a model with a different `seed` left every history's collision
//! stream untouched and only re-sampled the source birth. A multi-seed spread
//! then measured a fraction of the true variance, which is not a valid error
//! estimate. The seed is now `history_seed(base_seed, global_index)`, the one
//! shared definition both the CPU transport loop and the GPU seed buffer call.
//!
//! The fixture is chosen so the SOURCE contributes nothing to a seed-to-seed
//! difference: a point + monodirectional + single-energy neutron source samples
//! deterministically (zero RNG draws at birth), so every history is born
//! identical whatever the seed. Anything that differs between two seeds here
//! therefore comes from the collision stream alone. Run against the pre-fix
//! seeding, `different_seeds_change_the_collision_realisation` finds 1342 of
//! 1342 colliding histories (100.000%) identical across the two seeds and
//! fails; after the fix 1 of 1315 (0.08%) is, and that one is a coincidence
//! rather than a shared stream: a history whose single collision is a capture
//! scores `energy_out == energy_in == source energy` whatever the stream, so
//! two independent realisations can produce the same one-entry trace.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc::track::{HistorySelection, TrackEventType};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{Mt, ReactionRateScore, Score, Tally};
use yamc_tallies::CellFilter;

const DATA: &str = "tests/Fe56.arrow";
/// 14 MeV: reaches elastic, discrete + continuum inelastic and (n,2n), so the
/// trace exercises every sampler on the shared stream.
const ENERGY_EV: f64 = 14.06e6;
const RADIUS: f64 = 5.0;
const HISTORIES: usize = 2000;

fn data_present() -> bool {
    std::path::Path::new(DATA).exists()
}

/// One collision as the per-history signature compares them: the incoming and
/// outgoing energies as raw bit patterns, so the comparison is exact.
///
/// The sampled reaction MT is deliberately NOT part of the signature. The
/// terminal absorption's constituent MT is drawn by walking a
/// `HashMap<i32, Reaction>` (`Nuclide::sample_absorption_constituent`), whose
/// iteration order differs between two separately loaded copies of the same
/// nuclide, so the LABEL of an absorption can differ between two model
/// instances even though the draw, the energies and the transport are
/// identical. That is a pre-existing labelling wobble unrelated to seeding;
/// the energy sequence is the physics this test is about.
type Col = (u64, u64);

/// Single-nuclide Fe56 sphere with a deterministic monodirectional +z point
/// source at the origin, plus an (n,gamma) reaction-rate tally.
fn build_model() -> Model {
    let surface = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region::new_from_halfspace(HalfspaceType::Below(surface));

    let mut material = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material
        .read_nuclear_data(
            &HashMap::from([("Fe56".to_string(), DATA.to_string())]),
            None,
        )
        .unwrap();

    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::new_monodirectional(0.0, 0.0, 1.0),
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![ENERGY_EV], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let mut tally = Tally::new();
    tally.filters = vec![Filter::Cell(CellFilter::from_id(1))];
    tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        102,
    )))];
    tally.name = Some("capture".to_string());

    let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model
}

/// Per-history collision trace of the source walk: the `(energy_in, energy_out)`
/// bit patterns of every collision, in order.
fn collision_trace(model: &mut Model, seed: u64) -> Vec<Vec<Col>> {
    let settings = TransportSettings {
        total_particles: Some(HISTORIES),
        seed,
        ..Default::default()
    };
    let storage = model
        .run_with_tracking(&settings, HistorySelection::first(HISTORIES as u64))
        .expect("tracked run");
    let mut trace: Vec<Vec<Col>> = vec![Vec::new(); HISTORIES];
    for track in &storage.tracks {
        if track.generation != 0 {
            continue;
        }
        for ev in &track.events {
            if ev.event_type == TrackEventType::Collision && ev.history < HISTORIES {
                trace[ev.history].push((ev.energy_in.to_bits(), ev.energy_out.to_bits()));
            }
        }
    }
    trace
}

/// `(mean, std_dev)` of the (n,gamma) rate for one seed, from a fresh model.
///
/// Single-threaded so the comparison can be bit-exact. The per-history Welford
/// accumulators are private per rayon worker and merged in completion order, so
/// a multi-threaded run reproduces the same seed only to floating-point
/// rounding (a summation-order effect, checked separately in
/// [`same_seed_reproduces_the_run_exactly`]).
fn capture_rate(seed: u64, histories: usize, threads: Option<usize>) -> (f64, f64) {
    let mut model = build_model();
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(histories),
            seed,
            threads,
            ..Default::default()
        })
        .unwrap();
    let tally = &model.tallies[0];
    (tally.get_mean()[0], tally.get_std_dev()[0])
}

/// REPRODUCIBILITY: the seed still fully determines the run. Same seed twice
/// must give a bit-identical collision trace and a bit-identical tally result.
#[test]
fn same_seed_reproduces_the_run_exactly() {
    if !data_present() {
        eprintln!("skipping seed_reseeds_collisions -- {DATA} absent");
        return;
    }
    let mut model = build_model();
    let a = collision_trace(&mut model, 7);
    let b = collision_trace(&mut model, 7);
    assert_eq!(a, b, "same seed must reproduce the collision trace exactly");

    let (mean_a, std_a) = capture_rate(7, HISTORIES, Some(1));
    let (mean_b, std_b) = capture_rate(7, HISTORIES, Some(1));
    assert_eq!(
        mean_a.to_bits(),
        mean_b.to_bits(),
        "same seed must reproduce the tally mean bit-for-bit ({mean_a} vs {mean_b})"
    );
    assert_eq!(
        std_a.to_bits(),
        std_b.to_bits(),
        "same seed must reproduce the tally std_dev bit-for-bit ({std_a} vs {std_b})"
    );

    // Multi-threaded: the same seed transports the same histories, but the
    // per-worker Welford partials are merged in whatever order the workers
    // finish, so the result reproduces only to floating-point rounding. That is
    // a summation-order property of the parallel fold, not of the seeding.
    let (par_a, _) = capture_rate(7, HISTORIES, None);
    let (par_b, _) = capture_rate(7, HISTORIES, None);
    let rel = (par_a - par_b).abs() / par_a.abs().max(par_b.abs()).max(1e-300);
    assert!(
        rel < 1e-12,
        "same seed on the default thread pool must reproduce to rounding: \
         {par_a:e} vs {par_b:e} (rel {rel:e})"
    );
}

/// THE #315 REGRESSION: a different seed must give a different COLLISION
/// realisation. The source is deterministic here, so against the pre-fix
/// seeding the two traces were identical for 100.000% of the colliding
/// histories and this failed.
#[test]
fn different_seeds_change_the_collision_realisation() {
    if !data_present() {
        eprintln!("skipping seed_reseeds_collisions -- {DATA} absent");
        return;
    }
    // ONE model instance for both seeds, so the only difference between the two
    // runs is `TransportSettings::seed`.
    let mut model = build_model();
    let a = collision_trace(&mut model, 7);
    let b = collision_trace(&mut model, 8);

    // Guard the premise: the source is deterministic, so every history is born
    // at the same energy whichever seed ran. If this ever fails the test is no
    // longer isolating the collision stream from the source sampling.
    for (i, hist) in a.iter().enumerate() {
        if let Some(first) = hist.first() {
            assert_eq!(
                first.0,
                ENERGY_EV.to_bits(),
                "history {i} was not born at the source energy"
            );
        }
    }

    let with_collisions = a.iter().filter(|h| !h.is_empty()).count();
    assert!(
        with_collisions > HISTORIES / 4,
        "fixture must actually collide ({with_collisions} / {HISTORIES})"
    );
    let identical = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| !x.is_empty() && x == y)
        .count();
    let frac = identical as f64 / with_collisions as f64;
    eprintln!(
        "seed 7 vs seed 8: {identical} / {with_collisions} colliding histories kept an \
         identical collision trace ({:.3}%)",
        100.0 * frac
    );
    assert!(
        frac < 0.01,
        "{identical} of {with_collisions} colliding histories kept an IDENTICAL collision \
         trace across seeds ({:.3}%) -- the base seed is not reaching the per-history \
         collision PCG (issue #315)",
        100.0 * frac
    );
}

/// The two realisations are different but consistent: their (n,gamma) rates
/// agree within the reported statistics. A seed that changed the answer rather
/// than the realisation would show up here.
#[test]
fn different_seeds_agree_within_statistics() {
    if !data_present() {
        eprintln!("skipping seed_reseeds_collisions -- {DATA} absent");
        return;
    }
    // Single-threaded, so that "the two means differ" means the physics moved
    // rather than the parallel Welford merge order: with a deterministic source
    // and the pre-#315 seeding these two runs produced a BIT-IDENTICAL mean.
    let n = 20_000;
    let (mean_a, std_a) = capture_rate(7, n, Some(1));
    let (mean_b, std_b) = capture_rate(8, n, Some(1));
    assert_ne!(
        mean_a.to_bits(),
        mean_b.to_bits(),
        "different seeds gave a bit-identical mean -- the collision realisation did not move"
    );
    let sigma = (std_a * std_a + std_b * std_b).sqrt();
    let n_sigma = (mean_a - mean_b).abs() / sigma;
    eprintln!("capture rate {mean_a:e} (seed 7) vs {mean_b:e} (seed 8): {n_sigma:.2} sigma apart");
    assert!(
        n_sigma < 5.0,
        "capture rate {mean_a:e} (seed 7) vs {mean_b:e} (seed 8) differ by {n_sigma:.2} sigma"
    );
}
