//! GPU lost-particle detection (issue #289), on-hardware.
//!
//! A particle whose position is inside no cell is *lost*: the geometry does not
//! cover the space it reached. The CPU refuses to run such a model past
//! `max_lost_particles`. Before this test the GPU kernels just ended the
//! history, so a model with a geometry gap "succeeded" on `compute='gpu'` with
//! a quietly truncated tally while the same model aborted on `compute='cpu'`.
//!
//! Three cells of the matrix:
//! - sound geometry: zero losses on both backends (guards against a false
//!   positive turning healthy leaks into lost particles),
//! - geometry with a gap and a raised cap: the GPU reports losses and returns,
//! - geometry with a gap at the default cap: the GPU aborts, and the
//!   diagnostics are on the model.
//!
//! Self-skips if the endf-b8.1 Fe56 cache or an f64 GPU is absent. Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_lost_particles -- --nocapture --test-threads=1
#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::gpu::{GpuDispatchError, GpuRunResult};
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
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 4242;
const SOURCE_E: f64 = 14_060_000.0;
const INNER_R: f64 = 5.0;
const OUTER_R: f64 = 15.0;
const NUCLIDE: &str = "Fe56";
const TOTAL: usize = 5_000;

fn cache_dir(nuclide: &str) -> String {
    yamc_test_cache::nuclide_path(nuclide)
}

fn data_present(nuclide: &str) -> bool {
    std::path::Path::new(&cache_dir(nuclide)).is_dir()
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

fn sphere(id: usize, radius: f64, boundary: BoundaryType) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: Some(id),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary,
        name: None,
    })
}

fn iron() -> Material {
    let mut material = Material::new(
        HashMap::from([(NUCLIDE.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([(NUCLIDE.to_string(), cache_dir(NUCLIDE))]);
    material.read_nuclear_data(&nm, None).unwrap();
    material
}

/// Iron core (`r < INNER_R`) inside an iron shell that starts at
/// `INNER_R + gap`. With `gap > 0` the band between the two is covered by no
/// cell, so every particle that reaches it is lost. `gap == 0.0` is the same
/// model with the two cells flush, which must lose nothing.
fn shells_with_gap(gap: f64) -> Geometry {
    let inner = sphere(1, INNER_R, BoundaryType::Transmission);
    let shell_in = sphere(2, INNER_R + gap, BoundaryType::Transmission);
    let outer = sphere(3, OUTER_R, BoundaryType::Vacuum);

    let core = Region::new_from_halfspace(HalfspaceType::Below(inner));
    let shell = Region::new_from_halfspace(HalfspaceType::Below(outer))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Above(shell_in)));

    let cells = vec![
        Cell::new(Some(1), core, Some("iron".into()), Some(0)),
        Cell::new(Some(2), shell, Some("iron".into()), Some(0)),
    ];
    Geometry::new(cells, vec![Arc::new(iron())]).unwrap()
}

fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn cell_flux_tally() -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters
        .push(Filter::Cell(CellFilter::from_ids(vec![1, 2])));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

fn build(gap: f64, max_lost: usize) -> (Model, TransportSettings) {
    let mut model = Model::new(
        shells_with_gap(gap),
        vec![neutron_source()],
        vec![cell_flux_tally()],
    );
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model.max_lost_particles = max_lost;
    let settings = TransportSettings {
        total_particles: Some(TOTAL),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (model, settings)
}

fn run_gpu_retry(
    model: &mut Model,
    settings: &TransportSettings,
) -> Result<GpuRunResult, GpuDispatchError> {
    let mut attempt = 0;
    loop {
        match yamc::gpu::run_on_gpu(model, settings) {
            Ok(r) => return Ok(r),
            Err(e) => {
                let msg = e.to_string();
                let transient = msg.contains("BufferAsync") || msg.contains("buffer async");
                if !transient || attempt >= 4 {
                    return Err(e);
                }
                eprintln!("GPU transient error (attempt {attempt}): {msg}; retrying");
                attempt += 1;
            }
        }
    }
}

/// A sound geometry must report zero losses: particles leak through the
/// `Vacuum` outer surface, which is a normal termination and must NOT be
/// counted as lost.
#[test]
fn sound_geometry_loses_nothing_on_gpu() {
    if !data_present(NUCLIDE) || !gpu_available() {
        eprintln!("skipping sound_geometry_loses_nothing_on_gpu: data/GPU absent");
        return;
    }
    let (mut model, settings) = build(0.0, 10);
    let run = run_gpu_retry(&mut model, &settings).expect("sound geometry must run on GPU");
    assert_eq!(
        run.lost_count, 0,
        "a gapless geometry with a vacuum boundary must lose no particles, got {}",
        run.lost_count
    );
    assert!(model.lost_particles.is_empty());
}

/// With the cap raised, a gapped geometry runs but reports the losses, so the
/// user can see the geometry is broken.
#[test]
fn gap_reports_losses_on_gpu() {
    if !data_present(NUCLIDE) || !gpu_available() {
        eprintln!("skipping gap_reports_losses_on_gpu: data/GPU absent");
        return;
    }
    let (mut model, settings) = build(0.1, usize::MAX);
    let run = run_gpu_retry(&mut model, &settings).expect("raised cap must not abort");
    assert!(
        run.lost_count > 0,
        "a 0.1 cm band covered by no cell must lose particles, got {}",
        run.lost_count
    );
    assert!(
        !model.lost_particles.is_empty(),
        "diagnostics must reach Model::lost_particles"
    );
    // Every record must sit in the uncovered band, which is what makes the
    // diagnostic actionable.
    for lp in &model.lost_particles {
        let r = (lp.position[0].powi(2) + lp.position[1].powi(2) + lp.position[2].powi(2)).sqrt();
        assert!(
            r > INNER_R - 1e-6 && r < INNER_R + 0.1 + 1e-6,
            "lost at r = {r}, outside the [{INNER_R}, {}] gap",
            INNER_R + 0.1
        );
        assert!(lp.energy > 0.0);
    }
}

/// At the default cap the run must fail, exactly as the CPU refuses to run the
/// same model, instead of returning a truncated tally.
#[test]
fn gap_aborts_on_gpu_at_default_cap() {
    if !data_present(NUCLIDE) || !gpu_available() {
        eprintln!("skipping gap_aborts_on_gpu_at_default_cap: data/GPU absent");
        return;
    }
    let (mut model, settings) = build(0.1, 10);
    let err = run_gpu_retry(&mut model, &settings)
        .expect_err("a gapped geometry must not silently succeed on GPU");
    match err {
        GpuDispatchError::MaxLostParticlesExceeded { count, max, .. } => {
            assert_eq!(max, 10);
            assert!(count > 10, "count {count} must exceed the cap");
        }
        other => panic!("expected MaxLostParticlesExceeded, got {other:?}"),
    }
    assert!(
        !model.lost_particles.is_empty(),
        "diagnostics must survive the abort on Model::lost_particles"
    );
}
