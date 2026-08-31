//! GPU neutron-emitting-MT coverage regression test.
//!
//! Several neutron-producing reaction MTs (5 (n,misc), 23 (n,n'3α),
//! 24 (n,2nα), 25 (n,3nα), 37 (n,4n), 41 (n,2np), 44 (n,n'2p),
//! 45 (n,n'pα)) were missing from the GPU's per-MT `MT_SLOTS` table.
//! Because absorption is derived as `σ_t − σ_e − σ_inelastic − σ_f`
//! and these channels were not in the inelastic sum, their neutrons
//! were silently counted as absorption -- the kernel never sampled
//! them, so they were lost. That over-absorption depressed the GPU
//! flux relative to the CPU (Li6 was ~0.86 of CPU).
//!
//! These fixtures exercise newly-added slots: the Li6 test data carries
//! a neutron-emitting MT 24 channel (the (n,xnα) channel, ~0.078 b at
//! 14 MeV), and Fe56 carries MT 5 (~0.076 b at 14 MeV) plus MT 16.
//! With the slot-table fix the GPU samples those channels (and the
//! kernel walks all `MT_INELASTIC_COUNT` slots, not a stale literal),
//! so the integrated GPU/CPU flux ratio is close to 1.

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

/// Build a single-nuclide sphere with a 14.06 MeV isotropic point
/// source at the origin.
fn nuclide_sphere(
    nuclide: &str,
    arrow_path: &str,
    density_g_cc: f64,
    seed: u64,
    radius: f64,
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
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density_g_cc),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert(nuclide.to_string(), arrow_path.to_string());
    material.read_nuclear_data(&nm, None).unwrap();
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let n_particles = 20_000;
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

fn cpu_gpu_flux_ratio(nuclide: &str, arrow_path: &str, density_g_cc: f64, radius: f64) -> f64 {
    let (mut cpu_m, cpu_t, settings) = nuclide_sphere(nuclide, arrow_path, density_g_cc, 7, radius);
    cpu_m.simulate_transport(&settings).unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();
    let (mut gpu_m, gpu_t, settings) = nuclide_sphere(nuclide, arrow_path, density_g_cc, 7, radius);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    gpu / cpu
}

/// Li6 sphere: the (n,xnα) neutron-emitting channel was missing from
/// the slot table. With it added (and the kernel walking the full slot
/// count) the GPU/CPU flux ratio is close to 1; without the fix the GPU
/// over-absorbs and the ratio drops well below 1.
#[test]
fn gpu_mt_coverage_li6_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    let ratio = cpu_gpu_flux_ratio("Li6", "tests/Li6.arrow", 1.0, 35.0);
    eprintln!("Li6 r=35 (14.06 MeV): GPU/CPU flux ratio = {ratio:.4}");
    assert!(
        (0.98..1.02).contains(&ratio),
        "Li6 GPU/CPU flux ratio {ratio:.4} outside [0.98, 1.02] -- a \
         neutron-emitting MT may have dropped out of MT_SLOTS (its \
         neutrons would be silently counted as absorption)"
    );
}

/// Fe56 sphere: carries MT 5 (n,misc, ~0.076 b at 14 MeV) plus MT 16.
/// MT 5 was previously folded into derived absorption; adding it to the
/// slot table samples its neutrons. The ratio must stay near 1 (and not
/// regress the already-good Fe56 agreement).
#[test]
fn gpu_mt_coverage_fe56_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    let ratio = cpu_gpu_flux_ratio("Fe56", "tests/Fe56.arrow", 1.0, 35.0);
    eprintln!("Fe56 r=35 (14.06 MeV): GPU/CPU flux ratio = {ratio:.4}");
    assert!(
        (0.97..1.03).contains(&ratio),
        "Fe56 GPU/CPU flux ratio {ratio:.4} outside [0.97, 1.03] -- the \
         MT 5 slot-table addition may have regressed"
    );
}
