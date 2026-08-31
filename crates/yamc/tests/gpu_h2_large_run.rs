//! Regression test for #93: large GPU runs on a heavy-scattering nuclide must
//! not lose the device.
//!
//! The GPU path splits `total_particles` into equal per-dispatch batches. The
//! original schedule was always `total / 10`, so the particle count in a SINGLE
//! kernel launch grew without bound as `total_particles` rose. On the lightest,
//! least-absorbing scatterer (deuterium) a neutron undergoes very many
//! collisions before it leaks, so each thread runs a long inner loop; once a
//! launch reached ~200k particles (2M histories at total/10) the dispatch ran
//! long enough to trip the GPU watchdog (TDR). The device was then lost,
//! surfacing as an *uncatchable* `BufferAsyncError` panic on buffer readback
//! (it unwinds through a background wgpu thread, so it cannot be caught at the
//! `run_on_gpu` Result boundary -- it aborts the run).
//!
//! The fix caps per-dispatch particles at a watchdog-safe size, so 2M histories
//! now run as more launches of a proven-safe size. This test runs the
//! previously-crashing scenario (a thick H2 sphere, 2M histories) and asserts
//! the run completes with finite, positive flux. Pre-fix this aborted; the unit
//! tests in `model.rs` (`particles_per_chunk_caps_gpu_dispatch_size`) guard the
//! mechanism in CI where no GPU is present.
//!
//! Self-skips without an f64 GPU adapter or the cached H2 data.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

#[test]
fn gpu_h2_two_million_histories_does_not_lose_device() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no f64 GPU adapter");
        return;
    }
    let h2 = yamc_test_cache::nuclide_path("H2");
    if !std::path::Path::new(&h2).exists() {
        eprintln!("skipping: H2 cache data absent ({h2})");
        return;
    }

    // A thick deuterium sphere: low absorption + large elastic xs => many
    // collisions per history => the long-running kernel that lost the device
    // pre-fix once a single launch held ~200k particles.
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 15.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([("H2".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.0),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material
        .read_nuclear_data(&HashMap::from([("H2".to_string(), h2)]), None)
        .unwrap();

    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut tally = Tally::new();
    tally.filters.push(Filter::Cell(CellFilter::from_id(1)));
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.initialize_batches(1);
    let tally = Arc::new(tally);

    // 2M histories: pre-fix this dispatched 200k/launch (total/10) and lost the
    // device; post-fix it caps at 100k/launch (20 launches).
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = 20_000;
    model.tracking_mode = TrackingMode::Surface;

    let settings = TransportSettings {
        total_particles: Some(2_000_000),
        seed: 20240618,
        ..Default::default()
    };
    yamc::gpu::run_on_gpu(&mut model, &settings).expect("2M-history H2 GPU run should complete");

    let flux: f64 = tally.get_mean().iter().sum();
    assert!(
        flux.is_finite() && flux > 0.0,
        "expected finite positive flux from the 2M-history H2 run, got {flux}"
    );
}
