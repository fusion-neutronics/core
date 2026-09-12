//! Issue #272: a ring/frame-shaped MATERIAL cell must run on the GPU.
//!
//! `Region::bounding_box()` used to return a non-finite box for a region built
//! with a union of opposing half-spaces (a rectangular ring, e.g. the tokamak
//! bioshield wall), so `translate_cells` rejected the material-filled cell with
//! `CellRegionUnsupported`. The recursive bounding-box fix returns the finite
//! ENCLOSING box, so the cell now translates and runs. This test builds exactly
//! that shape -- an outer box with a central (x,y) void column, leaving a C12
//! "wall" ring -- and checks the GPU (a) accepts it and (b) reproduces the CPU
//! cell-flux within Monte-Carlo tolerance.
//!
//! Self-skips without the C12 data cache or an f64 GPU. Run:
//!   cargo test -p yamc --features gpu --release --test gpu_ring_cell -- --nocapture --test-threads=1
#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 90210;
const H: f64 = 10.0; // outer box half-width
const HOLE: f64 = 5.0; // central column half-width (x,y)
const SOURCE_E: f64 = 14_000_000.0;

fn cache_dir(nuclide: &str) -> String {
    yamc_test_cache::nuclide_path(nuclide)
}
fn data_present(nuclide: &str) -> bool {
    std::path::Path::new(&cache_dir(nuclide)).is_dir()
}
fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

/// Axis-aligned plane `a*x+b*y+c*z = d`. `Above` => coord >= d, `Below` => <= d.
fn plane(a: f64, b: f64, c: f64, d: f64, id: u32, boundary: &BoundaryType) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: Some(id as usize),
        kind: SurfaceKind::Plane { a, b, c, d },
        boundary: boundary.clone(),
        name: None,
    })
}

/// Outer box [-H,H]^3 (vacuum faces) split into a C12 ring (everything outside
/// the central [-HOLE,HOLE] x/y column) and a void hole (the column). The ring
/// region uses a UNION of opposing half-spaces -- the shape the old bbox
/// mishandled.
fn ring_geometry() -> Geometry {
    let v = BoundaryType::Vacuum;
    let t = BoundaryType::default();
    // Outer box faces (vacuum).
    let px_lo = plane(1.0, 0.0, 0.0, -H, 1, &v);
    let px_hi = plane(1.0, 0.0, 0.0, H, 2, &v);
    let py_lo = plane(0.0, 1.0, 0.0, -H, 3, &v);
    let py_hi = plane(0.0, 1.0, 0.0, H, 4, &v);
    let pz_lo = plane(0.0, 0.0, 1.0, -H, 5, &v);
    let pz_hi = plane(0.0, 0.0, 1.0, H, 6, &v);
    // Inner column faces (transmission).
    let ix_lo = plane(1.0, 0.0, 0.0, -HOLE, 7, &t);
    let ix_hi = plane(1.0, 0.0, 0.0, HOLE, 8, &t);
    let iy_lo = plane(0.0, 1.0, 0.0, -HOLE, 9, &t);
    let iy_hi = plane(0.0, 1.0, 0.0, HOLE, 10, &t);

    let hs = |ht: HalfspaceType| Region::new_from_halfspace(ht);
    let outer_box = hs(HalfspaceType::Above(px_lo.clone()))
        .intersection(&hs(HalfspaceType::Below(px_hi.clone())))
        .intersection(&hs(HalfspaceType::Above(py_lo.clone())))
        .intersection(&hs(HalfspaceType::Below(py_hi.clone())))
        .intersection(&hs(HalfspaceType::Above(pz_lo.clone())))
        .intersection(&hs(HalfspaceType::Below(pz_hi.clone())));
    // Outside the central column: x<-HOLE | x>HOLE | y<-HOLE | y>HOLE.
    let outside_col = hs(HalfspaceType::Below(ix_lo.clone()))
        .union(&hs(HalfspaceType::Above(ix_hi.clone())))
        .union(&hs(HalfspaceType::Below(iy_lo.clone())))
        .union(&hs(HalfspaceType::Above(iy_hi.clone())));
    let ring_region = outer_box.intersection(&outside_col);
    // The central column (void).
    let hole_region = hs(HalfspaceType::Above(px_lo))
        .intersection(&hs(HalfspaceType::Below(px_hi)))
        .intersection(&hs(HalfspaceType::Above(pz_lo)))
        .intersection(&hs(HalfspaceType::Below(pz_hi)))
        .intersection(&hs(HalfspaceType::Above(ix_lo)))
        .intersection(&hs(HalfspaceType::Below(ix_hi)))
        .intersection(&hs(HalfspaceType::Above(iy_lo)))
        .intersection(&hs(HalfspaceType::Below(iy_hi)));

    let mut material = Material::new(
        HashMap::from([("C12".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(2.0),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material
        .read_nuclear_data(
            &HashMap::from([("C12".to_string(), cache_dir("C12"))]),
            None,
        )
        .unwrap();

    let ring = Cell::new(Some(1), ring_region, Some("c12".into()), Some(0));
    let hole = Cell::new(Some(2), hole_region, None, None);
    Geometry::new(vec![ring, hole], vec![Arc::new(material)]).unwrap()
}

fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        // Inside the ring wall (x = 7 > HOLE).
        space: SourceSpatialDistribution::Point(Point::new([7.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn ring_flux_tally() -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        ParticleType::Neutron,
    )));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(10);
    Arc::new(t)
}

fn build(total: usize) -> (Model, TransportSettings) {
    let mut model = Model::new(
        ring_geometry(),
        vec![neutron_source()],
        vec![ring_flux_tally()],
    );
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = 10_000;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(total),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (model, settings)
}

#[test]
fn gpu_accepts_ring_cell_and_matches_cpu() {
    if !data_present("C12") || !gpu_available() {
        eprintln!("skipping gpu_accepts_ring_cell_and_matches_cpu: data/GPU absent");
        return;
    }
    let total = 100_000;

    // CPU.
    let (mut cpu_model, cs) = build(total);
    let cpu_tally = Arc::clone(&cpu_model.tallies[0]);
    cpu_model.simulate_transport(&cs).expect("CPU run failed");
    let cpu_mean = cpu_tally.get_mean()[0];
    let cpu_sd = cpu_tally.get_std_dev()[0];

    // GPU: previously errored with CellRegionUnsupported on the ring cell.
    let (mut gpu_model, gs) = build(total);
    let gpu_tally = Arc::clone(&gpu_model.tallies[0]);
    let mut last = String::new();
    let mut ok = false;
    for _ in 0..5 {
        match yamc::gpu::run_on_gpu(&mut gpu_model, &gs) {
            Ok(_) => {
                ok = true;
                break;
            }
            Err(e) => {
                last = e.to_string();
                if !last.contains("BufferAsync") {
                    break;
                }
            }
        }
    }
    assert!(ok, "GPU rejected the ring geometry: {last}");
    let gpu_mean = gpu_tally.get_mean()[0];
    let gpu_sd = gpu_tally.get_std_dev()[0];

    assert!(
        cpu_mean > 0.0 && gpu_mean > 0.0,
        "no flux scored in the ring"
    );
    let ratio = gpu_mean / cpu_mean;
    eprintln!(
        "ring cell flux: CPU={cpu_mean:.5e}+/-{cpu_sd:.1e}  GPU={gpu_mean:.5e}+/-{gpu_sd:.1e}  ratio={ratio:.4}"
    );
    assert!(
        (0.98..=1.02).contains(&ratio),
        "GPU/CPU ring-cell flux ratio {ratio:.4} outside [0.98, 1.02]"
    );
}
