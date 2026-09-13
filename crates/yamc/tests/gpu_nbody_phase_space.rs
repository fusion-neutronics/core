//! GPU N-body phase-space regression test.
//!
//! Pre-fix: H2's (n,2n) reaction (MT 16) carries an
//! `NBodyPhaseSpace` distribution with `n_bodies = 3`. The GPU
//! kernel had no NBPS sampler so MT 16 fell through to closed-form
//! Q-value energy + isotropic mu. Post-fix the kernel samples
//! E_out from the standard NBPS algorithm
//! (`E_out = E_max · x/(x+y)` with x, y from Maxwellian-product
//! distributions depending on n_bodies) and mu isotropic in CM.
//!
//! The integrated flux ratio is dominated by the elastic / capture
//! channels at deuterium densities, but with a 14 MeV source the
//! (n,2n) channel does fire. The pre-fix closed-form fallback
//! biases the (n,2n) outgoing-neutron spectrum and the integrated
//! flux. Post-fix the sampler matches CPU within statistical noise.
//!
//! NB: H2 elastic at MeV is roughly forward-peaked but with very
//! light target (AWR ~ 2) the elastic energy loss is large and
//! dominates the slowing-down. The NBPS fix is a smaller relative
//! correction than the KalbachMann or Evaporation fixes were on
//! their respective fixtures.

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

fn h2_sphere(seed: u64, radius: f64, source_energy: f64) -> (Model, Arc<Tally>, TransportSettings) {
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
    // Liquid deuterium ~ 0.169 g/cm³.
    let mut material = Material::new(
        HashMap::from([("H2".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.169),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert("H2".to_string(), "tests/H2.arrow".to_string());
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
    model.gpu_max_steps_per_particle = 10_000;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        threads: Some(1),
        ..Default::default()
    };
    (model, tally, settings)
}

#[test]
fn gpu_nbps_h2_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // H2 sphere, 14 MeV source. MT 16 (n,2n) uses NBPS with
    // n_bodies = 3. With the fix in place, GPU/CPU integrated
    // flux ratio should be close to 1.
    // Small radius: limited compounding of any per-collision
    // bias, so the integrated flux ratio is a clean measure of
    // whether NBPS is sampling correctly on the (n,2n) channel.
    // (Larger radii drift because elastic kinematics on light
    // targets like H2 amplifies any per-collision bias over many
    // collisions; that's a separate issue from NBPS.)
    let (mut cpu_m, cpu_t, settings) = h2_sphere(99, 20.0, 14.06e6);
    cpu_m.simulate_transport(&settings).unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();
    let (mut gpu_m, gpu_t, settings) = h2_sphere(99, 20.0, 14.06e6);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    let ratio = gpu / cpu;
    eprintln!("H2 r=20 (14.06 MeV): CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {ratio:.3}");
    assert!(
        (0.9..1.1).contains(&ratio),
        "H2 r=20 GPU/CPU flux ratio {ratio:.3} outside [0.9, 1.1] -- \
         NBodyPhaseSpace E_out sampler may have regressed (pre-fix used \
         closed-form Q-value fallback for MT 16 (n,2n))"
    );
}
