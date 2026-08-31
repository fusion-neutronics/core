//! GPU vs CPU integral-flux regression for a strong, near-pure scatterer.
//!
//! The CPU transport loop runs every history to completion (it ignores
//! `max_steps_per_particle` entirely). The GPU kernel cannot loop
//! unbounded, so it caps each history at `max_steps_per_particle` steps.
//! When that cap was 1000, a 14.06 MeV neutron in a 35 cm pure-H2 sphere
//! -- which undergoes many hundreds of elastic collisions before leaking
//! (H2 barely absorbs) -- got truncated mid-history on the GPU. Its
//! remaining track length was never scored, so the GPU integral flux came
//! out ~10% LOW (ratio ~0.90) while the CPU was unaffected.
//!
//! The fix raises the default cap to 100000 (the loop still exits the
//! instant a particle leaks or is absorbed, so it is free for fast cases).
//! This test pins the behaviour: with the default cap the GPU/CPU flux
//! ratio must be within statistical noise; the historical 1000-step cap
//! is also exercised to document the divergence it caused.

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

fn sphere(
    nuclide: &str,
    arrow: &str,
    seed: u64,
    max_steps: u32,
) -> (Model, Arc<Tally>, TransportSettings) {
    let surface = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 35.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(surface)));
    let mut material = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.0),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert(nuclide.to_string(), arrow.to_string());
    material.read_nuclear_data(&nm, None).unwrap();
    let cell = Cell::new(Some(1), region, Some("s".into()), Some(0));
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
    model.max_steps_per_particle = max_steps;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        threads: Some(1),
        ..Default::default()
    };
    (model, tally, settings)
}

fn cpu_flux(max_steps: u32) -> f64 {
    let (mut m, t, settings) = sphere("H2", "tests/H2.arrow", 7, max_steps);
    m.simulate_transport(&settings).unwrap();
    t.get_mean().iter().sum::<f64>()
}

fn gpu_flux(max_steps: u32) -> f64 {
    let (mut m, t, settings) = sphere("H2", "tests/H2.arrow", 7, max_steps);
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("GPU dispatch");
    t.get_mean().iter().sum::<f64>()
}

#[test]
fn gpu_h2_flux_matches_cpu_with_default_step_cap() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // Default cap (100000 in Model::new) -- must agree with CPU.
    let cpu = cpu_flux(100_000);
    let gpu = gpu_flux(100_000);
    let ratio = gpu / cpu;
    eprintln!("H2 r=35 default cap: CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {ratio:.4}");
    assert!(
        (0.98..1.02).contains(&ratio),
        "H2 GPU/CPU flux ratio {ratio:.4} outside [0.98, 1.02] with the \
         default step cap -- the GPU step-cap truncation regression may be back"
    );
}

#[test]
fn gpu_h2_low_step_cap_truncates_flux() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // The CPU ignores the cap (runs to completion), so its flux is the
    // same at 1000 and 100000 steps. The GPU honours the cap; at 1000 it
    // truncates H2 histories and reports ~10% LOW. This documents WHY the
    // default was raised -- if the GPU ever stops truncating at 1000
    // (e.g. cap semantics change) this test should be revisited.
    let cpu = cpu_flux(1_000);
    let gpu_capped = gpu_flux(1_000);
    let ratio = gpu_capped / cpu;
    eprintln!("H2 r=35 cap=1000: CPU = {cpu:.4e}  GPU = {gpu_capped:.4e}  ratio = {ratio:.4}");
    assert!(
        ratio < 0.95,
        "expected the 1000-step cap to truncate H2 GPU flux well below CPU \
         (historical ~0.90), got ratio {ratio:.4}"
    );
}
