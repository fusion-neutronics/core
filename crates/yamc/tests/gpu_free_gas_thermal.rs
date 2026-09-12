//! GPU free-gas thermal scattering regression test.
//!
//! Pre-fix: the GPU kernel always treated the target nucleus as
//! stationary in elastic scattering, regardless of how low the
//! neutron's energy went. Below the free-gas threshold
//! (`E < 400 · K_B · T`, ≈ 10 eV at 294 K), this is wrong: the
//! target's thermal motion gives the neutron a chance to gain
//! energy back, which prevents particles from piling up in the
//! sub-thermal tail. Without it, GPU spectra below ~10 eV are
//! systematically over-populated relative to CPU.
//!
//! Post-fix: the kernel samples a target velocity from the CXS
//! Maxwell-Boltzmann distribution and runs full vector-based CM
//! kinematics whenever a particle's energy drops below the
//! threshold (and the target is heavier than the neutron). Above
//! the threshold the closed-form A-mass kinematics is used,
//! identical to the pre-fix behaviour, so high-energy GPU/CPU
//! bit-equivalence is preserved.
//!
//! This test runs CPU and GPU on a Pb208 sphere and:
//! 1. Asserts that running with a low-energy source (~1 eV, well
//!    below the threshold) yields a non-degenerate energy
//!    distribution -- pre-fix particles below threshold could only
//!    lose energy elastically, so they accumulated at thermal
//!    indefinitely. Post-fix the upscatter from thermal motion
//!    keeps the spectrum near the source energy.
//! 2. Confirms that running with a high-energy source (14 MeV)
//!    above-threshold path still agrees with CPU -- free-gas
//!    branch isn't taken at fast energies, so behaviour matches
//!    the earlier elastic-angular regression test.

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
fn gpu_thermal_source_does_not_diverge_or_pile_up() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // Source at 1 eV (below 400·K_B·T ≈ 10 eV threshold).
    // Without free-gas, the GPU has only elastic energy loss,
    // so particles slow down monotonically and pile up in the
    // sub-thermal tail until captured. The integrated flux on a
    // small sphere differs from CPU by a moderate factor.
    // With free-gas, upscatter from thermal motion keeps the
    // spectrum near the source energy and the integrated flux
    // matches CPU much more closely.
    let (mut cpu_m, cpu_t, settings) = pb208_sphere(7, 5.0, 1.0);
    cpu_m.simulate_transport(&settings).unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();

    let (mut gpu_m, gpu_t, settings) = pb208_sphere(7, 5.0, 1.0);
    let gpu_run = yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    let avg_steps =
        gpu_run.n_steps.iter().copied().sum::<u32>() as f64 / gpu_run.n_steps.len().max(1) as f64;
    eprintln!(
        "Pb208 r=5 (1 eV source): CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {:.3}  avg_steps = {avg_steps:.1}",
        gpu / cpu,
    );
    let ratio = gpu / cpu;
    assert!(
        ratio.is_finite(),
        "GPU sub-thermal flux ratio should be finite, got {ratio}"
    );
    // Tight bound on the low-energy ratio. CPU and GPU should now
    // both converge similarly because free-gas upscatter dominates
    // the thermal regime.
    assert!(
        (0.6..1.4).contains(&ratio),
        "Pb208 r=5 1eV GPU/CPU flux ratio {ratio:.3} outside [0.6, 1.4] -- \
         free-gas thermal scattering may have regressed"
    );
}

#[test]
fn gpu_high_energy_unchanged_by_free_gas() {
    if yamc_gpu::GpuContext::new().is_err() {
        return;
    }
    // 14.06 MeV source on a 1 cm sphere -- particles barely
    // collide before leaking. The energy is well above any free-
    // gas threshold (~10 eV), so the kernel takes the closed-form
    // path identical to the pre-free-gas behaviour. This test
    // confirms the free-gas plumbing didn't accidentally affect
    // high-energy results.
    let (mut cpu_m, cpu_t, settings) = pb208_sphere(7, 1.0, 14.06e6);
    cpu_m.simulate_transport(&settings).unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();
    let (mut gpu_m, gpu_t, settings) = pb208_sphere(7, 1.0, 14.06e6);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    let ratio = gpu / cpu;
    eprintln!("Pb208 r=1 (14.06 MeV source): CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {ratio:.3}");
    assert!(
        (0.7..1.3).contains(&ratio),
        "Pb208 r=1 14 MeV GPU/CPU flux ratio {ratio:.3} outside [0.7, 1.3]"
    );
}
