//! A vacuum surface passing through a cell it does not bound, on the GPU.
//!
//! A flight can only leave its cell through one of the cell's own surfaces,
//! and the CPU (`Cell::closest_surface`) solves only those. The kernels solve
//! the current cell's surfaces too: a scan over every surface in the model
//! stops the particle at surfaces that do not bound its cell, and when such a
//! surface is a vacuum boundary it ends the history inside the cell.
//!
//! The model is a near-void sphere of radius `R_INNER` around an isotropic
//! point source, inside a shell out to `R_OUTER` that the vacuum plane
//! x = `X_PLANE` splits in two. The plane bounds only the two shell cells,
//! but it cuts through the inner sphere. Every particle crosses the inner
//! sphere uncollided (almost), so its flux there is close to `R_INNER` per
//! source particle and must match the CPU.
//!
//! Self-skips if the Be9 / Be fixtures or an f64 GPU are absent. Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_vacuum_surface_outside_its_cell -- --nocapture --test-threads=1
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
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const TOTAL: usize = 20_000;
const SEED: u64 = 7;
const R_INNER: f64 = 10.0;
const R_OUTER: f64 = 20.0;
const X_PLANE: f64 = 5.0;

/// Be9 at 1e-6 g/cm3: close to a void, so the inner cell's flux is nearly
/// the uncollided track length `R_INNER`.
fn thin_beryllium(id: u32, nuclide_dir: &str, element_dir: &str) -> Material {
    let mut material = Material::new(
        HashMap::from([("Be9".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1e-6),
    )
    .unwrap();
    material.set_material_id(id);
    material.set_temperature("294");
    let nuclides = HashMap::from([("Be9".to_string(), nuclide_dir.to_string())]);
    let elements = HashMap::from([("Be".to_string(), element_dir.to_string())]);
    material
        .read_nuclear_data(&nuclides, Some(&elements))
        .unwrap();
    material.init_photon_data(&elements).unwrap();
    material
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

fn above(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Above(s.clone()))
}

fn below(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Below(s.clone()))
}

/// Inner sphere (cell 1), and the shell around it split by the vacuum plane
/// into x < `X_PLANE` (cell 2) and x > `X_PLANE` (cell 3).
fn geometry(nuclide_dir: &str, element_dir: &str) -> Geometry {
    let inner = sphere(1, R_INNER, BoundaryType::Transmission);
    let outer = sphere(2, R_OUTER, BoundaryType::Vacuum);
    let plane = Arc::new(Surface::new_plane(
        1.0,
        0.0,
        0.0,
        X_PLANE,
        Some(3),
        Some(BoundaryType::Vacuum),
    ));
    let shell = above(&inner).intersection(&below(&outer));
    let cells = vec![
        Cell::new(Some(1), below(&inner), Some("inner".into()), Some(0)),
        Cell::new(
            Some(2),
            shell.intersection(&below(&plane)),
            Some("shell_low_x".into()),
            Some(0),
        ),
        Cell::new(
            Some(3),
            shell.intersection(&above(&plane)),
            Some("shell_high_x".into()),
            Some(0),
        ),
    ];
    let materials = vec![Arc::new(thin_beryllium(1, nuclide_dir, element_dir))];
    Geometry::new(cells, materials).unwrap()
}

fn point_source(photon: bool) -> ParticleSource {
    let energy = if photon { 1.0e6 } else { 14.06e6 };
    let source = Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![energy], vec![1.0]).unwrap()),
        strength: 1.0,
    };
    if photon {
        ParticleSource::Photon(source)
    } else {
        ParticleSource::Neutron(source)
    }
}

fn build(photon: bool, nuclide_dir: &str, element_dir: &str) -> (Model, Arc<Tally>) {
    let mut tally = Tally::new();
    tally
        .filters
        .push(Filter::Cell(CellFilter::from_ids(vec![1, 2, 3])));
    tally.scores = vec!["flux".parse().unwrap()];
    tally.estimator = Estimator::TrackLength;
    tally.reset_accumulation();
    let tally = Arc::new(tally);
    let mut model = Model::new(
        geometry(nuclide_dir, element_dir),
        vec![point_source(photon)],
        vec![tally.clone()],
    );
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    (model, tally)
}

fn settings() -> TransportSettings {
    TransportSettings {
        total_particles: Some(TOTAL),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    }
}

fn check(photon: bool) {
    let (Some(nuclide_dir), Some(element_dir)) = (
        yamc_test_cache::nuclide("Be9"),
        yamc_test_cache::element("Be"),
    ) else {
        eprintln!("skipping: Be9 / Be fixtures absent");
        return;
    };
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no GPU with f64 compute");
        return;
    }
    let kind = if photon { "photon" } else { "neutron" };

    let (mut cpu_model, cpu_tally) = build(photon, &nuclide_dir, &element_dir);
    cpu_model.simulate_transport(&settings()).unwrap();
    let cpu = cpu_tally.get_mean();

    let (mut gpu_model, gpu_tally) = build(photon, &nuclide_dir, &element_dir);
    yamc::gpu::run_on_gpu(&mut gpu_model, &settings()).expect("GPU dispatch");
    let gpu = gpu_tally.get_mean();

    eprintln!("{kind}: cell flux CPU {cpu:.3?}, GPU {gpu:.3?}");

    // Almost every particle crosses the inner sphere uncollided, so its
    // flux is close to the radius. Stopping at the plane would cut the
    // flights heading to x > X_PLANE short, removing about 8% of it.
    let inner = gpu[0] / R_INNER;
    assert!(
        (0.99..1.01).contains(&inner),
        "{kind}: GPU inner-sphere flux is {inner:.4} of the radius"
    );
    for (cell, (&g, &c)) in gpu.iter().zip(&cpu).enumerate() {
        let ratio = g / c;
        assert!(
            (0.98..1.02).contains(&ratio),
            "{kind}: GPU/CPU flux in cell {} is {ratio:.4} (GPU {g:.3}, CPU {c:.3})",
            cell + 1
        );
    }
}

#[test]
fn neutrons_cross_a_vacuum_plane_that_does_not_bound_their_cell() {
    check(false);
}

#[test]
fn photons_cross_a_vacuum_plane_that_does_not_bound_their_cell() {
    check(true);
}
