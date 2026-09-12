//! GPU discrete-level inelastic Q-value regression test.
//!
//! Light nuclides with strong discrete two-body inelastic scattering
//! (C12, Al27, ...) carry their outgoing energy as a closed-form
//! `LevelInelastic` distribution: the kernel computes
//! `E_cm = (A/(A+1))^2 * (E - (A+1)/A*|Q|)` from the per-MT Q-value.
//!
//! Pre-fix bug: `extract_material_xs` density-weighted each MT slot's
//! Q-value by the cross section at the TOP of the energy grid. Discrete-
//! level inelastic cross sections (MT 51..=90) peak near threshold and
//! fall to zero well below the grid maximum, so that weight was zero and
//! the slot's Q collapsed to 0. With Q = 0 the closed form loses its
//! threshold term, `E_cm = (A/(A+1))^2 * E`, and the neutron keeps almost
//! all its energy on every discrete-level inelastic collision. The flux
//! spectrum hardened and leaked out of the sphere -> integral flux ~13%
//! low on C12 (and similar on N14 / O16 / Al27 / Si28).
//!
//! Post-fix: the Q weight is the PEAK cross section over the grid, which
//! is nonzero for every channel, so each slot keeps its correct Q.
//!
//! This test runs CPU and GPU on a 35 cm C12 and Al27 sphere with a
//! 14.06 MeV source and asserts the integrated flux ratio is within 3%.
//! Pre-fix C12 was ~0.87; post-fix it is ~1.00.

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
    radius: f64,
) -> (Model, Arc<Tally>, TransportSettings) {
    let surface = Surface {
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
    model.gpu_max_steps_per_particle = 10_000;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        threads: Some(1),
        ..Default::default()
    };
    (model, tally, settings)
}

fn cpu_gpu_flux_ratio(nuclide: &str, arrow: &str) -> f64 {
    let (mut cpu_m, cpu_t, settings) = sphere(nuclide, arrow, 7, 35.0);
    cpu_m.simulate_transport(&settings).unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();
    let (mut gpu_m, gpu_t, settings) = sphere(nuclide, arrow, 7, 35.0);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    eprintln!(
        "{nuclide} r=35 (14.06 MeV): CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {:.4}",
        gpu / cpu
    );
    gpu / cpu
}

#[test]
fn gpu_discrete_level_inelastic_c12_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    let ratio = cpu_gpu_flux_ratio("C12", "tests/C12.arrow");
    assert!(
        (0.97..1.03).contains(&ratio),
        "C12 r=35 GPU/CPU flux ratio {ratio:.4} outside [0.97, 1.03] -- \
         per-MT inelastic Q-value weighting may have regressed (pre-fix the \
         discrete-level Q collapsed to 0 and the spectrum hardened to ~0.87)"
    );
}

#[test]
fn gpu_discrete_level_inelastic_al27_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    let ratio = cpu_gpu_flux_ratio("Al27", "tests/Al27.arrow");
    assert!(
        (0.97..1.03).contains(&ratio),
        "Al27 r=35 GPU/CPU flux ratio {ratio:.4} outside [0.97, 1.03] -- \
         per-MT inelastic Q-value weighting may have regressed"
    );
}
