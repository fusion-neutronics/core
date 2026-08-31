//! GPU photon transport regression test.
//!
//! Co60 emits two prompt gammas at 1.173 MeV and 1.332 MeV from
//! every β decay; a 1 MeV-equivalent point source is the textbook
//! shielding-and-dose calculation problem. Iron is the canonical
//! shielding material at these energies. The test runs both CPU
//! and GPU photon transport on a 1 cm Fe sphere with a Co60-style
//! source biased into the centre and asserts the integrated flux
//! is within statistical noise.
//!
//! Pre-fix: `run_on_gpu` rejected any model with a photon source.
//! Post-fix: photon sources route to the dedicated GPU photon
//! kernel (`multi_cell_photon_transport`).

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

fn fe_sphere(seed: u64, radius: f64, source_energy: f64) -> (Model, Arc<Tally>, TransportSettings) {
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
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert("Fe56".to_string(), "tests/Fe56.arrow".to_string());
    let mut photon_paths: HashMap<String, String> = HashMap::new();
    photon_paths.insert("Fe".to_string(), "tests/Fe.arrow".to_string());
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();
    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Photon(Source {
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
    model.max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        ..Default::default()
    };
    (model, tally, settings)
}

#[test]
fn gpu_photon_co60_fe_sphere_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // 1.25 MeV -- the mean of Co60's two gammas (1.173 + 1.332 MeV).
    // At this energy in Fe, Compton is the dominant interaction
    // (~95% of total XS); photoelectric and pair are minor.
    let source_e = 1.25e6_f64;
    let (mut cpu_m, cpu_t, settings) = fe_sphere(42, 1.0, source_e);
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();
    let (mut gpu_m, gpu_t, settings) = fe_sphere(42, 1.0, source_e);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    let ratio = gpu / cpu;
    eprintln!("Fe r=1 ({source_e:.2e} eV): CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {ratio:.3}");
    // Tolerance is wider than the neutron tests because:
    // - Compton uses Kahn's free-electron sampler on GPU vs the
    //   CPU's incoherent-form-factor-rejected variant
    // - No atomic relaxation cascade on GPU (photoelectric absorbs)
    // - No Compton-electron secondaries on GPU
    // Each of these is a small effect but they accumulate. ±25% is
    // a regression catcher, not a physics-fidelity claim.
    assert!(
        (0.75..1.25).contains(&ratio),
        "Fe r=1 GPU/CPU photon flux ratio {ratio:.3} outside [0.75, 1.25] -- \
         photon kernel may have regressed"
    );
}

/// 5 MeV gammas in Fe -- well above the 1.022 MeV pair-production
/// threshold so the kernel's pair branch fires. Sigma_pair at 5 MeV
/// in Fe is ~few % of total, enough to produce a tally-relevant
/// number of annihilation photons over a 4-batch × 5k-particle run.
/// This is the regression catcher for slice 4 (pair-production
/// sampler + 2 × 511 keV annihilation photon emission). e+/e- TTB
/// is deferred to slice 4b -- without it we expect some flux
/// under-shoot relative to CPU, but the pair branch firing at all
/// (instead of silently absorbing) closes most of the gap.
#[test]
fn gpu_photon_pair_production_fe_sphere_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    let source_e = 5.0e6_f64;
    let (mut cpu_m, cpu_t, settings) = fe_sphere(7, 1.0, source_e);
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();
    let (mut gpu_m, gpu_t, settings) = fe_sphere(7, 1.0, source_e);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    let ratio = gpu / cpu;
    eprintln!("Fe r=1 ({source_e:.2e} eV): CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {ratio:.3}");
    // Wider tolerance than the 1.25 MeV test: pair production
    // sampler accuracy depends on a 12-term Taylor atan polyfill,
    // and e+/e- TTB is deferred to slice 4b so brem photons from
    // the e+/e- kinetic energy don't yet contribute to GPU flux.
    // ±35 % is a regression catcher; the slice 4b PR should tighten
    // this once the TTB blocks are wired in.
    assert!(
        (0.65..1.35).contains(&ratio),
        "Fe r=1 @ 5 MeV GPU/CPU photon flux ratio {ratio:.3} outside [0.65, 1.35] -- \
         pair-production branch may have regressed"
    );
}
