//! GPU Kalbach-Mann inelastic continuum regression test.
//!
//! Pre-fix: Pb208's continuum (MT 91) and multi-neutron-out
//! channels (MT 16, 17, 28, 32, 33, 37) all use Kalbach-Mann
//! correlated angle-energy distributions; the GPU kernel had no
//! KM sampler, so it fell back to closed-form Q-value energy +
//! isotropic mu. Both are wrong: closed-form gives a single
//! E_out per E_in (instead of the actual continuum spectrum) and
//! isotropic mu replaces the forward-peaked compound + precompound
//! mixture KM systematics produces. The bias compounds over
//! ~9 inelastic collisions per particle on a 20 cm Pb208 sphere
//! and pushed the integrated GPU/CPU flux ratio to 8.6×, even
//! after the elastic-angular fix.
//!
//! Post-fix: the kernel samples (E_out_cm, mu_cm) from the
//! tabulated KM data (PDF + CDF for E_out, plus per-(E_in, E_out)
//! `r` and `a` parameters for the mu sampler), then runs the
//! existing CM→lab kinematics. The integrated GPU/CPU ratio on
//! the same model drops from 8.6× to ~1.0.
//!
//! Asserts a moderately tight bound on the Pb208 r=20 cm sphere
//! to catch any regression that disables the KM path.

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

fn pb208_sphere(
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
        HashMap::from([("Pb208".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(11.34),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert("Pb208".to_string(), "tests/Pb208.arrow".to_string());
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
fn gpu_kalbach_mann_pb208_large_sphere_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // Pb208 r=20 cm with 14 MeV source. Pre-fix this hit GPU/CPU
    // = 8.6× because every inelastic collision used closed-form
    // Q-value energy and isotropic mu instead of KM data.
    // Post-fix the ratio is ~1.0.
    let (mut cpu_m, cpu_t, settings) = pb208_sphere(42, 20.0, 14.06e6);
    cpu_m.simulate_transport(&settings).unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();
    let (mut gpu_m, gpu_t, settings) = pb208_sphere(42, 20.0, 14.06e6);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    let ratio = gpu / cpu;
    eprintln!("Pb208 r=20 (14.06 MeV): CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {ratio:.3}");
    assert!(
        (0.85..1.15).contains(&ratio),
        "Pb208 r=20 14 MeV GPU/CPU flux ratio {ratio:.3} outside [0.85, 1.15] -- \
         Kalbach-Mann inelastic continuum sampling may have regressed \
         (pre-fix was 8.6× from closed-form + isotropic fallback)"
    );
}
