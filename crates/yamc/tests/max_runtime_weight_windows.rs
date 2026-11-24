//! A timed run with strongly splitting weight windows stops near its budget.
//!
//! `max_runtime` is checked between chunks, and with weight windows one source
//! history can carry thousands of split particles, so a chunk sized by history
//! count alone can run for minutes past the budget. Self-skips when the Fe56
//! fixture is absent.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
use yamc::variance_reduction::{VarianceReduction, WeightWindowBounds};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;

const RADIUS: f64 = 50.0;

/// A 14 MeV point source at the centre of an iron sphere, with a window whose
/// lower bound falls tenfold per shell outward, so particles split at every
/// shell they reach.
fn model(path: &str) -> Model {
    let sphere = Arc::new(Surface {
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
    let mut iron = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.87),
    )
    .unwrap();
    iron.set_material_id(1);
    iron.set_temperature("294");
    iron.read_nuclear_data(&HashMap::from([("Fe56".into(), path.into())]), None)
        .unwrap();
    let cell = Cell::new(
        Some(1),
        Region::new_from_halfspace(HalfspaceType::Below(sphere)),
        None,
        Some(0),
    );
    let geometry = Geometry::new(vec![cell], vec![Arc::new(iron)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0; 3])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut tally = Tally::new();
    tally.filters.push(Filter::Cell(CellFilter::from_id(1)));
    tally.scores = vec!["flux".parse::<Score>().unwrap()];
    tally.reset_accumulation();
    let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
    model.verbose = Verbose::silent();

    let n = 10;
    let mesh = RegularRectangularMesh::new([-RADIUS; 3], [RADIUS; 3], [n, n, n]);
    let shell = |i: usize| (2 * i).abs_diff(n - 1) / 2;
    let mut lower = Vec::with_capacity(n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let s = shell(i).max(shell(j)).max(shell(k)) as i32;
                lower.push(0.5 * 10f64.powi(-s));
            }
        }
    }
    let upper = lower.iter().map(|l| 5.0 * l).collect();
    model
        .variance_reduction
        .push(VarianceReduction::WeightWindowBounds(Box::new(
            WeightWindowBounds {
                mesh,
                particle: ParticleType::Neutron,
                energy_bins: None,
                lower_bounds: lower,
                upper_bounds: upper,
                survival_factor: 3.0,
                max_split: 10,
                weight_floor: 1e-38,
            },
        )));
    model
}

/// Wall time of one `simulate_transport` call on a fresh model.
fn timed(path: &str, settings: &TransportSettings) -> f64 {
    let mut model = model(path);
    let start = std::time::Instant::now();
    model.simulate_transport(settings).expect("transport");
    start.elapsed().as_secs_f64()
}

/// The budget is one and a half times what the first 100 histories take
/// (seeds are keyed to the history index, so a capped run of 100 is the timed
/// run's first chunk), with the set-up a one-history run takes left out of
/// both. A fixed schedule then starts a second chunk of 100 with a third of
/// the budget left and ends a third over; sized from the rate, the run stops
/// near the budget. Scaled to the measured times, so the check holds on any
/// build or machine.
#[test]
fn a_timed_weight_window_run_stops_near_its_budget() {
    let Some(path) = yamc_test_cache::transport_nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture not fetched");
        return;
    };
    let settings = |total_particles, max_runtime| TransportSettings {
        total_particles,
        max_runtime,
        seed: 2,
        threads: Some(2),
        ..Default::default()
    };
    let setup = timed(&path, &settings(Some(1), None));
    let budget = 1.5 * (timed(&path, &settings(Some(100), None)) - setup);
    let ran = timed(&path, &settings(None, Some(budget))) - setup;
    assert!(
        ran < 1.15 * budget,
        "a {budget:.2} s budget ran for {ran:.2} s"
    );
}
