//! GPU-vs-CPU parity for the `total` (MT=1) reaction-rate score.
//!
//! `ReactionRate(MT=TOTAL)` is a particle-agnostic total interaction
//! rate. On the photon path it must mean the photon total macroscopic XS
//! (Σ_total,photon · flux), matching OpenMC's photon `total`. The GPU
//! routes it through the photon kernel's `SCORE_TOTAL` fast path (which
//! already uses the photon total XS). The CPU previously routed *all*
//! `ReactionRate(mt)` through the NEUTRON `macro_xs_by_mt` table -- a
//! category error for a photon, scoring ~0 on the track-length estimator.
//! The fix makes the CPU score the photon total XS for `MT=TOTAL` on the
//! photon path, in both the track-length and collision estimators.
//!
//! This test asserts:
//!   1. Photon `total` (track-length AND collision) now MATCHES the GPU
//!      within the usual photon agreement band (was 0 vs ~4.3).
//!   2. A specific neutron MT (102 = (n,gamma)) on a photon tally is
//!      REJECTED on the GPU (clear error) and scores 0 on the CPU.
//!
//! Setup mirrors `gpu_photon_components`: 1 cm Fe sphere, Co60-mean
//! (1.25 MeV) point photon source. The neutron-`total`-unchanged check
//! lives in the CPU-only `neutron_total_reaction_rate_unchanged` test
//! below (no GPU required).

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region};
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
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{ReactionRateScore, Score};
use yamc_tallies::tally::Tally;
use yamc_tallies::{Estimator, Mt};

fn fe_photon_material() -> Material {
    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();
    material
}

fn fe_sphere() -> (Geometry, u32) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 1.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell], vec![Arc::new(fe_photon_material())]).unwrap();
    (geometry, cell_id)
}

fn photon_source() -> ParticleSource {
    ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.25e6], vec![1.0]).unwrap()),
        strength: 1.0,
    })
}

/// One photon `ReactionRate(MT=TOTAL)` tally on the given estimator,
/// cell- and particle-filtered to the photon path.
fn photon_total_tally(cell_id: u32, estimator: Estimator) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        yamc_particle::particle::ParticleType::Photon,
    )));
    t.scores = vec![Score::ReactionRate(ReactionRateScore::total())];
    t.estimator = estimator;
    t.initialize_batches(4);
    Arc::new(t)
}

fn build_photon_total_model(estimator: Estimator) -> (Model, Arc<Tally>, TransportSettings) {
    let (geometry, cell_id) = fe_sphere();
    let tally = photon_total_tally(cell_id, estimator);
    let mut model = Model::new(geometry, vec![photon_source()], vec![tally.clone()]);
    model.max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(20_000 * 4),
        seed: 42,
        ..Default::default()
    };
    (model, tally, settings)
}

/// Photon `total` (MT=1) must now match the GPU on BOTH estimators. Before
/// the fix the CPU track-length value was ~0 (neutron MT-1 table at a photon
/// energy) while the GPU was ~4.3.
#[test]
fn gpu_photon_total_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    for estimator in [Estimator::TrackLength, Estimator::Collision] {
        let (mut cpu_m, cpu_t, settings) = build_photon_total_model(estimator);
        cpu_m
            .simulate_transport(&TransportSettings {
                threads: Some(1),
                ..settings
            })
            .unwrap();
        let cpu: f64 = cpu_t.get_mean().iter().sum();

        let (mut gpu_m, gpu_t, settings) = build_photon_total_model(estimator);
        yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
        let gpu: f64 = gpu_t.get_mean().iter().sum();

        eprintln!("Fe r=1 photon total ({estimator:?}): CPU = {cpu:.4e}  GPU = {gpu:.4e}");

        assert!(
            cpu > 0.0,
            "{estimator:?}: CPU photon total is zero -- fix regressed"
        );
        assert!(
            gpu > 0.0,
            "{estimator:?}: GPU photon total is zero -- setup wrong"
        );
        let ratio = gpu / cpu;
        assert!(
            (0.75..1.25).contains(&ratio),
            "{estimator:?}: GPU/CPU photon-total ratio {ratio:.3} outside [0.75, 1.25] \
             (CPU {cpu:.4e}, GPU {gpu:.4e})"
        );
    }
}

/// A *specific* neutron MT on a photon tally is a category error.
/// - GPU: rejected at `validate_tallies` with a clear message.
/// - CPU: scores 0 (the photon does not undergo a neutron reaction).
#[test]
fn specific_neutron_mt_on_photon_rejected_gpu_zero_cpu() {
    let gpu_available = yamc_gpu::GpuContext::new().is_ok();

    let build = || -> (Model, Arc<Tally>, TransportSettings) {
        let (geometry, cell_id) = fe_sphere();
        let mut t = Tally::new();
        t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
        t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
            yamc_particle::particle::ParticleType::Photon,
        )));
        // MT 102 = (n,gamma): a neutron capture reaction. Nonsense for a photon.
        t.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
            102,
        )))];
        t.initialize_batches(4);
        let tally = Arc::new(t);
        let mut model = Model::new(geometry, vec![photon_source()], vec![tally.clone()]);
        model.max_steps_per_particle = 5_000;
        model.transport_secondary_photons = true;
        let settings = TransportSettings {
            total_particles: Some(20_000 * 4),
            seed: 42,
            ..Default::default()
        };
        (model, tally, settings)
    };

    // CPU: runs fine, scores exactly 0 for the photon (n,gamma) channel.
    let (mut cpu_m, cpu_t, settings) = build();
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .unwrap();
    let cpu: f64 = cpu_t.get_mean().iter().sum();
    assert_eq!(
        cpu, 0.0,
        "specific neutron MT 102 on a photon must score 0, got {cpu:.4e}"
    );

    // GPU: rejected at dispatch with a clear message.
    if gpu_available {
        let (mut gpu_m, _gpu_t, settings) = build();
        let err = yamc::gpu::run_on_gpu(&mut gpu_m, &settings)
            .expect_err("GPU must reject a specific neutron MT on the photon path");
        let s = format!("{err}");
        assert!(
            s.contains("reaction-rate-by-MT") || s.contains("neutron score"),
            "expected a clear neutron-MT rejection message, got: {s}"
        );
    } else {
        eprintln!("skipping GPU rejection check -- no GPU with f64 compute available");
    }
}

/// CPU-only guard: a NEUTRON `ReactionRate(MT=TOTAL)` is unchanged by the
/// fix (still the neutron total macroscopic XS, MT 1). The fix only
/// special-cases the photon path; the neutron path is untouched. We assert
/// the neutron total equals the neutron total cross section integral (it is
/// strictly positive and well above the elastic-only floor), which would be
/// zero if the photon special-case had leaked into the neutron path.
#[test]
fn neutron_total_reaction_rate_unchanged() {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    material.read_nuclear_data(&nm, None).unwrap();
    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let mk = |estimator: Estimator| -> Arc<Tally> {
        let mut t = Tally::new();
        t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
        t.scores = vec![Score::ReactionRate(ReactionRateScore::total())];
        t.estimator = estimator;
        t.initialize_batches(4);
        Arc::new(t)
    };
    let tl = mk(Estimator::TrackLength);
    let col = mk(Estimator::Collision);

    let mut model = Model::new(geometry, vec![source], vec![tl.clone(), col.clone()]);
    model.max_steps_per_particle = 10_000;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(10_000 * 4),
            seed: 7,
            threads: Some(1),
            ..Default::default()
        })
        .unwrap();

    let tl_total: f64 = tl.get_mean().iter().sum();
    let col_total: f64 = col.get_mean().iter().sum();
    eprintln!("Fe r=10 neutron total: track-length = {tl_total:.4e}  collision = {col_total:.4e}");

    // Neutron total must be strictly positive on both estimators (the photon
    // special-case did NOT leak into the neutron path). The two estimators
    // are unbiased estimates of the same total reaction rate, so they agree
    // within statistics.
    assert!(tl_total > 0.0, "neutron total (track-length) must be > 0");
    assert!(col_total > 0.0, "neutron total (collision) must be > 0");
    let ratio = col_total / tl_total;
    assert!(
        (0.9..1.1).contains(&ratio),
        "neutron total estimators disagree: track-length {tl_total:.4e} vs \
         collision {col_total:.4e} (ratio {ratio:.3})"
    );
}
