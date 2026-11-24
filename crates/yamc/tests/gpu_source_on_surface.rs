//! A source exactly on a CSG surface, on the GPU.
//!
//! A point on a plane is in neither half-space, so a kernel that locates a
//! source particle at its birth point alone loses every one of them. The CPU
//! locates a particle with no cell yet a birth step along its flight
//! (`GeometryKind::find_cell_index_along`), so it starts in the cell it is
//! heading into; the neutron and photon kernels now do the same.
//!
//! The model is a near-void slab split by the plane z = `zf` into two cells,
//! with an isotropic point source on that plane. Half the particles head into
//! each cell, so neither cell's flux may be zero, the two must agree, and the
//! GPU must match the CPU. `zf = 600` puts the plane far enough from the
//! origin that the coarser spacing of doubles there matters.
//!
//! Self-skips if the Be9 / Be fixtures or an f64 GPU are absent. Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_source_on_surface -- --nocapture --test-threads=1
#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface};
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
const SEED: u64 = 1;

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

/// Be9 at 1e-6 g/cm3: close to a void, so each cell's flux is mostly the
/// uncollided track length of the half of the source heading into it.
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

fn plane(id: usize, normal: [f64; 3], d: f64, boundary: BoundaryType) -> Arc<Surface> {
    Arc::new(Surface::new_plane(
        normal[0],
        normal[1],
        normal[2],
        d,
        Some(id),
        Some(boundary),
    ))
}

fn above(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Above(s.clone()))
}

fn below(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Below(s.clone()))
}

/// A 1000 x 1000 x 1200 cm box split at z = `zf` into a lower cell (id 1)
/// and an upper cell (id 2), each with its own material.
fn split_box(zf: f64, nuclide_dir: &str, element_dir: &str) -> Geometry {
    let v = BoundaryType::Vacuum;
    let x0 = plane(1, [1.0, 0.0, 0.0], 0.0, v.clone());
    let x1 = plane(2, [1.0, 0.0, 0.0], 1000.0, v.clone());
    let y0 = plane(3, [0.0, 1.0, 0.0], 0.0, v.clone());
    let y1 = plane(4, [0.0, 1.0, 0.0], 1000.0, v.clone());
    let bottom = plane(5, [0.0, 0.0, 1.0], zf - 600.0, v.clone());
    let top = plane(6, [0.0, 0.0, 1.0], zf + 600.0, v.clone());
    let face = plane(7, [0.0, 0.0, 1.0], zf, BoundaryType::Transmission);
    let side = above(&x0)
        .intersection(&below(&x1))
        .intersection(&above(&y0))
        .intersection(&below(&y1));
    let lower = side
        .intersection(&above(&bottom))
        .intersection(&below(&face));
    let upper = side.intersection(&above(&face)).intersection(&below(&top));
    let cells = vec![
        Cell::new(Some(1), lower, Some("lower".into()), Some(0)),
        Cell::new(Some(2), upper, Some("upper".into()), Some(1)),
    ];
    let materials = vec![
        Arc::new(thin_beryllium(1, nuclide_dir, element_dir)),
        Arc::new(thin_beryllium(2, nuclide_dir, element_dir)),
    ];
    Geometry::new(cells, materials).unwrap()
}

fn point_source(zf: f64, photon: bool) -> ParticleSource {
    let energy = if photon { 1.0e6 } else { 14.06e6 };
    let source = Source {
        space: SourceSpatialDistribution::Point(Point::new([500.0, 500.0, zf])),
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

fn cell_flux_tally() -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters
        .push(Filter::Cell(CellFilter::from_ids(vec![1, 2])));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.reset_accumulation();
    Arc::new(t)
}

fn build(zf: f64, photon: bool, nuclide_dir: &str, element_dir: &str) -> (Model, Arc<Tally>) {
    let tally = cell_flux_tally();
    let mut model = Model::new(
        split_box(zf, nuclide_dir, element_dir),
        vec![point_source(zf, photon)],
        vec![tally.clone()],
    );
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    // A loss must show up in the assertions below, not abort the run.
    model.max_lost_particles = usize::MAX;
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

/// Runs the model with the source on the plane on both backends and checks
/// the GPU loses nothing and fills both cells as the CPU does.
fn check(zf: f64, photon: bool) {
    let (Some(nuclide_dir), Some(element_dir)) = (
        yamc_test_cache::nuclide("Be9"),
        yamc_test_cache::element("Be"),
    ) else {
        eprintln!("skipping: Be9 / Be fixtures absent");
        return;
    };
    if !gpu_available() {
        eprintln!("skipping: no GPU with f64 compute");
        return;
    }
    let kind = if photon { "photon" } else { "neutron" };

    let (mut cpu_model, cpu_tally) = build(zf, photon, &nuclide_dir, &element_dir);
    cpu_model.simulate_transport(&settings()).unwrap();
    let cpu = cpu_tally.get_mean();

    let (mut gpu_model, gpu_tally) = build(zf, photon, &nuclide_dir, &element_dir);
    let run = yamc::gpu::run_on_gpu(&mut gpu_model, &settings()).expect("GPU dispatch");
    let gpu = gpu_tally.get_mean();

    eprintln!(
        "{kind} source on z = {zf}: CPU lost {}, flux {:.1} / {:.1}; GPU lost {}, flux {:.1} / {:.1}",
        cpu_model.lost_particles.len(),
        cpu[0],
        cpu[1],
        run.lost_count,
        gpu[0],
        gpu[1]
    );

    assert!(
        cpu_model.lost_particles.is_empty(),
        "{kind}: CPU lost particles"
    );
    assert_eq!(
        run.lost_count, 0,
        "{kind} source on the plane z = {zf}: the GPU lost {} particles",
        run.lost_count
    );
    for (cell, (&g, &c)) in gpu.iter().zip(&cpu).enumerate() {
        assert!(g > 0.0, "{kind}: GPU flux in cell {} is zero", cell + 1);
        let ratio = g / c;
        assert!(
            (0.95..1.05).contains(&ratio),
            "{kind}: GPU/CPU flux in cell {} is {ratio:.3} (GPU {g:.2}, CPU {c:.2})",
            cell + 1
        );
    }
    // Isotropic source on the dividing plane: each half gets half the
    // particles, so the two cells' fluxes agree.
    let balance = gpu[1] / gpu[0];
    assert!(
        (0.95..1.05).contains(&balance),
        "{kind}: GPU upper/lower flux is {balance:.3}"
    );
}

#[test]
fn neutron_source_on_a_plane_at_the_origin_is_not_lost_on_gpu() {
    check(0.0, false);
}

#[test]
fn neutron_source_on_a_plane_far_from_the_origin_is_not_lost_on_gpu() {
    check(600.0, false);
}

#[test]
fn photon_source_on_a_plane_at_the_origin_is_not_lost_on_gpu() {
    check(0.0, true);
}

#[test]
fn photon_source_on_a_plane_far_from_the_origin_is_not_lost_on_gpu() {
    check(600.0, true);
}
