//! Transport-loop performance benchmark.
//!
//! Establishes a stable, statistically-comparable baseline for the
//! `Model::simulate` inner-loop hot path. The point of this bench is to
//! catch <5% regressions during Phase 1f -- the inner transport loop is
//! being progressively factored into helpers (handle_lost_particle,
//! find_or_lose_cell, …, transport_particle) and a one-shot `cargo run
//! --example tbr` is too noisy to detect small per-step regressions.
//!
//! Geometry mirrors `examples/tbr.rs`: a 1-cm Li-Be sphere inside a
//! 200-cm vacuum sphere, with a 14.06 MeV monoenergetic point source at
//! the origin and a tritium-production reaction-rate tally on the outer
//! annulus. Each iteration runs `simulate(Some(1))` (single thread, for
//! reproducibility) on a fresh `model.clone()` for a fixed workload of
//! 5000 particles × 1 batch. Criterion repeats it many times and reports
//! a confidence interval; in practice this gives a 95% CI half-width of
//! ~0.5%, tight enough to detect real <1% regressions above noise.
//!
//! Run with:
//!     cargo bench --bench transport

use criterion::{criterion_group, criterion_main, Criterion};
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
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{Mt, ReactionRateScore, Score, Tally};
use yamc_tallies::CellFilter;

/// Build the standard tbr-like geometry and source. Same shape as
/// `examples/tbr.rs` but with smaller particle counts for quick
/// iterations under criterion.
fn build_tbr_model(batches: usize) -> Model {
    let sphere1 = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 1.0,
        },
        boundary: BoundaryType::Transmission,
        name: None,
    });
    let sphere2 = Arc::new(Surface {
        surface_id: Some(2),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 200.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });

    let region1 = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            sphere1.clone(),
        )))),
    };
    let region2 = Region {
        expr: RegionExpr::Intersection(
            Box::new(RegionExpr::Halfspace(HalfspaceType::Above(sphere1.clone()))),
            Box::new(RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                HalfspaceType::Above(sphere2.clone()),
            )))),
        ),
    };

    let mut material1 = Material::new(
        HashMap::from([
            ("Li6".into(), 0.07 / 2.0),
            ("Li7".into(), 0.93 / 2.0),
            ("Be9".into(), 0.5),
        ]),
        "atom",
        "g/cm3",
        Some(2.0),
    )
    .unwrap();
    material1.set_material_id(1);
    material1.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Be9".to_string(), "tests/Be9.arrow".to_string());
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    nuclide_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
    material1.read_nuclear_data(&nuclide_map, None).unwrap();
    let mat_arc = Arc::new(material1);

    let cell1 = Cell::new(Some(1), region1, Some("inner_sphere".to_string()), None);
    let cell2 = Cell::new(Some(2), region2, Some("outer_annular".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell1, cell2.clone()], vec![mat_arc]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let cell_filter2 = Filter::Cell(CellFilter::from_id(cell2.cell_id.unwrap()));
    let mut tally = Tally::new();
    tally.filters = vec![cell_filter2];
    tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        105,
    )))];
    tally.name = Some("tbr".to_string());
    tally.initialize_batches(batches);

    Model::new(geometry, vec![source], vec![Arc::new(tally)])
}

fn bench_transport(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport");
    // Pin to a single thread so we measure serial transport throughput
    // rather than rayon-scaling artefacts that depend on the host's
    // available cores. Multi-thread scaling is a separate concern.
    let threads = Some(1);

    // ~5000 particles per simulate call. Larger workloads tighten the
    // confidence interval (per-particle variance averages out) at a
    // modest cost in total bench runtime. Tuned to keep the per-iter
    // time near 100 ms so criterion's default measurement window fits
    // dozens of samples and a real <5% regression is detectable above
    // the noise floor.
    let particles = 5000;
    let batches = 1;

    // Build the model once outside the iter loop -- Arrow nuclide load
    // + XS cache build dominate model construction (~hundred ms) and
    // would dwarf the actual transport. Using `iter_batched` with a
    // clone-per-iter as the setup gives criterion a fresh model each
    // run (so tally state doesn't accumulate and bias later iterations)
    // while excluding the clone cost from the timing band.
    let template = build_tbr_model(batches);

    let settings = TransportSettings {
        total_particles: Some(particles * batches),
        threads,
        ..Default::default()
    };

    group.bench_function("tbr_5000p_1batch", |b| {
        b.iter_batched(
            || template.clone(),
            |mut model| model.simulate_transport(&settings).unwrap(),
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_transport);
criterion_main!(benches);
