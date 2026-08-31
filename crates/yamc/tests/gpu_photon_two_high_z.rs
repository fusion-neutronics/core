//! Task #72 verification: per-collision ELEMENT selection in the GPU photon
//! kernel for a material with two comparable-Z elements.
//!
//! Before #72 the GPU carried only ONE dominant element's form-factor /
//! relaxation slab per material (#79), so a Pb+W (or Pb+Bi, ...) mixture --
//! two elements with comparable Z and comparable macroscopic contribution --
//! modelled only ONE element's coherent / incoherent / photoelectric /
//! relaxation physics per collision. The secondary / fluorescence / scattered
//! spectrum was therefore wrong. With true per-collision element selection the
//! kernel samples WHICH element the photon struck (proportional to that
//! element's macroscopic-total contribution, mirroring CPU
//! `Material::sample_element`) and runs THAT element's secondary physics.
//!
//! This test compares the GPU photon flux spectrum against the CPU reference
//! for a 50/50 Pb/W sphere at an ~8 MeV photon source. The full-spectrum flux
//! and the low-energy scatter/fluorescence band (the channel #72 fixes) should
//! both land near 1.0. It also pins that a SINGLE-element W sphere stays
//! correct (the count==1 fast path skips the selection draw).
//!
//! The GPU photon kernel is not bit-reproducible run-to-run (secondary cascade
//! ordering under non-deterministic GPU execution -- see the transport module
//! docs), so the comparison uses statistical tolerances, not equality.
//!
//! Run it (single-threaded -- the shared cubecl client serialises launches):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_photon_two_high_z -- --nocapture --test-threads=1
//! Self-skips when the W/Pb photon Arrow data or an f64 GPU adapter is absent.

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
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::kinds::{PhotonComponent, PhotonXSScore};
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 20256;
const RADIUS: f64 = 3.0;
const N_PER_BATCH: usize = 50_000;
const N_BATCHES: usize = 4;
const MAX_STEPS: u32 = 5_000;
const SOURCE_E: f64 = 8.0e6;

// W and Pb photon Arrow data in the local yamc cache (ENDF/B-8.1).
fn w_path() -> String {
    yamc_test_cache::nuclide_path("W")
}
fn pb_path() -> String {
    yamc_test_cache::nuclide_path("Pb")
}

fn data_present() -> bool {
    ["W", "Pb", "W184", "Pb208"]
        .iter()
        .all(|n| yamc_test_cache::have(n))
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

/// Build a sphere of the given material composition (nuclide -> atom fraction)
/// with W/Pb photon data attached. `density` g/cm3.
fn sphere_model(
    nuclides: HashMap<String, f64>,
    photon_paths: HashMap<String, String>,
    neutron_paths: HashMap<String, String>,
    density: f64,
) -> (Geometry, u32) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));

    let mut material = Material::new(nuclides, "atom", "g/cm3", Some(density)).unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material
        .read_nuclear_data(&neutron_paths, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();

    let cell = Cell::new(Some(1), region, Some("hz".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    (geometry, 1)
}

fn photon_source() -> ParticleSource {
    ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

/// A coarse-binned photon flux tally over [1 keV, 10 MeV], plus optional
/// PhotonXS component tallies, all track-length.
fn flux_tally(cell_id: u32) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    // Log-spaced edges; the lowest band captures the fluorescence / multiple-
    // scatter tail that the two-element mix governs, the top band the source.
    let edges = vec![1.0e3, 1.0e5, 1.0e6, 4.0e6, 1.0e7];
    t.filters.push(Filter::Energy(EnergyFilter::new(edges)));
    t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

fn component_tally(cell_id: u32, component: PhotonComponent) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.scores = vec![Score::PhotonXS(PhotonXSScore { component })];
    t.estimator = Estimator::Collision;
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

fn build_model(geometry: Geometry, tallies: Vec<Arc<Tally>>) -> (Model, TransportSettings) {
    let mut model = Model::new(geometry, vec![photon_source()], tallies);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    // TTB electron treatment -> the full relaxation / form-factor / pair
    // machinery (the per-element packs task #72 touches) is active.
    model.transport_secondary_photons = true;
    model.photon_cutoff_energy = 1000.0;
    let settings = TransportSettings {
        total_particles: Some(N_PER_BATCH * N_BATCHES),
        seed: SEED,
        ..Default::default()
    };
    (model, settings)
}

fn bins(t: &Arc<Tally>) -> Vec<f64> {
    t.get_mean().to_vec()
}

fn run_cpu(make_tallies: impl Fn(u32) -> Vec<Arc<Tally>>, geo: (Geometry, u32)) -> Vec<Arc<Tally>> {
    let (geometry, cell_id) = geo;
    let tallies = make_tallies(cell_id);
    let (mut model, settings) = build_model(geometry, tallies.clone());
    model
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .expect("CPU run failed");
    tallies
}

fn run_gpu(make_tallies: impl Fn(u32) -> Vec<Arc<Tally>>, geo: (Geometry, u32)) -> Vec<Arc<Tally>> {
    let (geometry, cell_id) = geo;
    let tallies = make_tallies(cell_id);
    let (mut model, settings) = build_model(geometry, tallies.clone());
    yamc::gpu::run_on_gpu(&mut model, &settings).expect("GPU run failed");
    tallies
}

fn w184_n() -> String {
    yamc_test_cache::nuclide_path("W184")
}
fn pb208_n() -> String {
    yamc_test_cache::nuclide_path("Pb208")
}

fn pb_w_geo() -> (Geometry, u32) {
    sphere_model(
        HashMap::from([("W184".into(), 0.5), ("Pb208".into(), 0.5)]),
        HashMap::from([("W".to_string(), w_path()), ("Pb".to_string(), pb_path())]),
        HashMap::from([
            ("W184".to_string(), w184_n()),
            ("Pb208".to_string(), pb208_n()),
        ]),
        17.0,
    )
}

fn w_geo() -> (Geometry, u32) {
    sphere_model(
        HashMap::from([("W184".into(), 1.0)]),
        HashMap::from([("W".to_string(), w_path())]),
        HashMap::from([("W184".to_string(), w184_n())]),
        19.3,
    )
}

/// Pb+W (two comparable-Z): GPU vs CPU photon flux spectrum + photoelectric /
/// coherent component scores all recover toward 1.0.
#[test]
fn pb_w_two_high_z_recovers() {
    if !data_present() {
        println!("W/Pb photon data absent -- skipping");
        return;
    }
    if !gpu_available() {
        println!("no f64 GPU adapter -- skipping");
        return;
    }

    let make = |cell: u32| {
        vec![
            flux_tally(cell),
            component_tally(cell, PhotonComponent::Coherent),
            component_tally(cell, PhotonComponent::Photoelectric),
            component_tally(cell, PhotonComponent::Incoherent),
        ]
    };

    let cpu = run_cpu(make, pb_w_geo());
    let gpu = run_gpu(make, pb_w_geo());

    let cpu_flux = bins(&cpu[0]);
    let gpu_flux = bins(&gpu[0]);
    let cpu_coh: f64 = bins(&cpu[1]).iter().sum();
    let gpu_coh: f64 = bins(&gpu[1]).iter().sum();
    let cpu_pe: f64 = bins(&cpu[2]).iter().sum();
    let gpu_pe: f64 = bins(&gpu[2]).iter().sum();
    let cpu_inc: f64 = bins(&cpu[3]).iter().sum();
    let gpu_inc: f64 = bins(&gpu[3]).iter().sum();

    let cpu_total: f64 = cpu_flux.iter().sum();
    let gpu_total: f64 = gpu_flux.iter().sum();
    let cpu_low: f64 = cpu_flux.iter().take(2).sum(); // < 1 MeV: scatter+fluor tail
    let gpu_low: f64 = gpu_flux.iter().take(2).sum();

    println!("\n== Pb+W 50/50 sphere, {SOURCE_E:.0e} eV photon source ==");
    println!(
        "flux total     CPU {cpu_total:.4e}  GPU {gpu_total:.4e}  ratio {:.3}",
        gpu_total / cpu_total
    );
    println!(
        "flux <1 MeV    CPU {cpu_low:.4e}  GPU {gpu_low:.4e}  ratio {:.3}",
        gpu_low / cpu_low
    );
    println!(
        "coherent       CPU {cpu_coh:.4e}  GPU {gpu_coh:.4e}  ratio {:.3}",
        gpu_coh / cpu_coh
    );
    println!(
        "photoelectric  CPU {cpu_pe:.4e}  GPU {gpu_pe:.4e}  ratio {:.3}",
        gpu_pe / cpu_pe
    );
    println!(
        "incoherent     CPU {cpu_inc:.4e}  GPU {gpu_inc:.4e}  ratio {:.3}",
        gpu_inc / cpu_inc
    );

    // Component rates ride on the per-material density-summed macro XS, so they
    // are tight. The flux spectrum (esp. the low-E scatter/fluor tail) is the
    // channel per-collision element selection fixes; allow looser MC tolerance.
    let tol = |g: f64, c: f64| (g / c - 1.0).abs();
    assert!(
        tol(gpu_coh, cpu_coh) < 0.05,
        "coherent off: {gpu_coh}/{cpu_coh}"
    );
    assert!(
        tol(gpu_pe, cpu_pe) < 0.05,
        "photoelectric off: {gpu_pe}/{cpu_pe}"
    );
    assert!(
        tol(gpu_inc, cpu_inc) < 0.05,
        "incoherent off: {gpu_inc}/{cpu_inc}"
    );
    assert!(
        tol(gpu_total, cpu_total) < 0.10,
        "total flux off: {gpu_total}/{cpu_total}"
    );
    assert!(
        tol(gpu_low, cpu_low) < 0.15,
        "low-E flux off: {gpu_low}/{cpu_low}"
    );
}

/// Single-element W: the count==1 fast path skips selection; GPU vs CPU stays
/// correct (and unchanged from the #79 behaviour).
#[test]
fn single_element_w_unchanged() {
    if !data_present() {
        println!("W photon data absent -- skipping");
        return;
    }
    if !gpu_available() {
        println!("no f64 GPU adapter -- skipping");
        return;
    }

    let make = |cell: u32| {
        vec![
            flux_tally(cell),
            component_tally(cell, PhotonComponent::Coherent),
            component_tally(cell, PhotonComponent::Photoelectric),
        ]
    };

    let cpu = run_cpu(make, w_geo());
    let gpu = run_gpu(make, w_geo());

    let cpu_total: f64 = bins(&cpu[0]).iter().sum();
    let gpu_total: f64 = bins(&gpu[0]).iter().sum();
    let cpu_coh: f64 = bins(&cpu[1]).iter().sum();
    let gpu_coh: f64 = bins(&gpu[1]).iter().sum();
    let cpu_pe: f64 = bins(&cpu[2]).iter().sum();
    let gpu_pe: f64 = bins(&gpu[2]).iter().sum();

    println!("\n== single-element W sphere ==");
    println!(
        "flux total  CPU {cpu_total:.4e}  GPU {gpu_total:.4e}  ratio {:.3}",
        gpu_total / cpu_total
    );
    println!(
        "coherent    CPU {cpu_coh:.4e}  GPU {gpu_coh:.4e}  ratio {:.3}",
        gpu_coh / cpu_coh
    );
    println!(
        "photoelec   CPU {cpu_pe:.4e}  GPU {gpu_pe:.4e}  ratio {:.3}",
        gpu_pe / cpu_pe
    );

    let tol = |g: f64, c: f64| (g / c - 1.0).abs();
    assert!(tol(gpu_coh, cpu_coh) < 0.05, "W coherent off");
    assert!(tol(gpu_pe, cpu_pe) < 0.05, "W photoelectric off");
    assert!(tol(gpu_total, cpu_total) < 0.10, "W total flux off");
}
