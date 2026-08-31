//! GPU Evaporation E_out distribution regression test.
//!
//! Pre-fix: C12's continuum (MT 91) and (n,n'p) (MT 28) channels
//! use the Evaporation energy distribution
//! (`p(E_out) ~ E_out · exp(-E_out/θ(E_in))`, `0 < E_out < E_in - u`).
//! The GPU kernel had no Evaporation sampler so those slots fell
//! through to closed-form Q-value energy -- a single fixed E_out
//! per E_in instead of the actual continuum spectrum. C12 is a
//! common moderator (graphite-moderated reactor problems), so the
//! gap matters in real workloads.
//!
//! Post-fix: the kernel samples E_out via the standard rejection
//! algorithm (`E = -ln((1 - v·xi1)·(1 - v·xi2))`, accept when
//! `E ≤ y = (E_in - u) / θ`, `E_out = E · θ`) using the slot's
//! tabulated `θ(E_in)` and restriction energy `u`. mu_sampled
//! continues to come from the slice-B angular table (UAE) or the
//! isotropic fallback (top-level Evaporation).
//!
//! This test runs CPU and GPU on a 14 MeV C12 sphere and asserts
//! the integrated flux ratio is close to 1. Pre-fix it would be
//! biased by the closed-form fallback at every MT 28 / MT 91
//! collision; post-fix the rejection sampler should bring it
//! within ~10% of CPU.

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
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

fn c12_sphere(
    seed: u64,
    radius: f64,
    source_energy: f64,
) -> (Model, Arc<Tally>, TransportSettings) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([("C12".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(2.26),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert("C12".to_string(), "tests/C12.arrow".to_string());
    material.read_nuclear_data(&nm, None).unwrap();
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![source_energy], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let n_particles = 5_000;
    let n_batches = 4;
    let mut tally = Tally::new();
    tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap())));
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.initialize_batches(n_batches);
    let tally = Arc::new(tally);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.max_steps_per_particle = 10_000;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        threads: Some(1),
        ..Default::default()
    };
    (model, tally, settings)
}

#[test]
fn gpu_evaporation_c12_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // C12 sphere, 14 MeV source. MT 91 (continuum) and MT 28
    // ((n,n'p)) both use Evaporation. Pre-fix the GPU used closed-
    // form Q-value energy for these and biased the spectrum;
    // post-fix the rejection sampler matches CPU within statistical
    // noise.
    let (mut cpu_m, cpu_t, settings) = c12_sphere(42, 10.0, 14.06e6);
    cpu_m.simulate_transport(&settings).unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();
    let (mut gpu_m, gpu_t, settings) = c12_sphere(42, 10.0, 14.06e6);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    let ratio = gpu / cpu;
    eprintln!("C12 r=10 (14.06 MeV): CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {ratio:.3}");
    assert!(
        (0.85..1.15).contains(&ratio),
        "C12 r=10 GPU/CPU flux ratio {ratio:.3} outside [0.85, 1.15] -- \
         Evaporation E_out sampler may have regressed (pre-fix used \
         closed-form Q-value fallback for MT 28 / MT 91)"
    );
}
