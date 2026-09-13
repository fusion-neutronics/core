//! GPU URR (unresolved resonance region) probability-table
//! regression test.
//!
//! Pre-fix: GPU transport had no URR sampling -- actinides and heavy
//! structurals at keV–MeV used the smooth resolved-resonance XS
//! everywhere, missing the self-shielding effect that URR probability
//! tables represent.
//!
//! Post-fix: when the dominant URR-bearing nuclide's data is loaded
//! and the particle's energy falls inside the URR range, the kernel
//! draws one random number, samples the URR probability table for
//! (elastic, capture, fission) micro XS, and perturbs the material
//! macroscopic Σ_e / Σ_a / Σ_f used for distance sampling AND
//! reaction-type sampling. Bit-equivalence with the CPU is not
//! attempted (RNG draw orders differ); the assertion is that the
//! GPU-averaged flux lands close to the CPU result.
//!
//! Co58 (cobalt-58) is the URR-bearing nuclide in our test set. The
//! comparison runs both CPU and GPU on a Co58 sphere with a source
//! biased into the URR range and checks the integrated flux ratio.

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

fn co58_sphere(
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
        HashMap::from([("Co58".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(8.9),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert("Co58".to_string(), "tests/Co58.arrow".to_string());
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
fn gpu_urr_co58_matches_cpu_in_resonance_region() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // Source at 10 keV -- inside Co58's URR window. Pre-fix the GPU
    // used the smooth XS for every history; post-fix it samples the
    // URR probability tables. We expect CPU/GPU agreement within
    // statistical noise on a 4×5k batched run.
    let source_e = 1.0e4_f64;
    let (mut cpu_m, cpu_t, settings) = co58_sphere(42, 5.0, source_e);
    cpu_m.simulate_transport(&settings).unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();
    let (mut gpu_m, gpu_t, settings) = co58_sphere(42, 5.0, source_e);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    let ratio = gpu / cpu;
    eprintln!("Co58 r=5 ({source_e:.2e} eV): CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {ratio:.3}");
    assert!(
        (0.85..1.15).contains(&ratio),
        "Co58 r=5 GPU/CPU flux ratio {ratio:.3} outside [0.85, 1.15] -- \
         URR probability-table sampling may have regressed"
    );
}

/// Mn55 sphere whose URR record uses LogLog interpolation (and has
/// low-lying inelastic). Exercises the issue #105 fixes -- LogLog
/// interpolation of the URR micro cross-sections and the smooth-inelastic
/// exclusion -- on a real nuclide, checking the GPU tracks the
/// OpenMC-validated CPU URR treatment in the URR window. Uses the cached
/// ENDF/B-VIII.1 Mn55 data; self-skips when it (or a GPU) is absent, so it
/// is inert in CI (which has neither).
fn mn55_sphere(
    seed: u64,
    radius: f64,
    source_energy: f64,
) -> Option<(Model, Arc<Tally>, TransportSettings)> {
    let cache = yamc_test_cache::nuclide_path("Mn55");
    if !std::path::Path::new(&cache).exists() {
        return None;
    }
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
        HashMap::from([("Mn55".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.21),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert("Mn55".to_string(), cache);
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
    Some((model, tally, settings))
}

#[test]
fn gpu_urr_mn55_loglog_matches_cpu_in_resonance_region() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // 300 keV -- inside Mn55's LogLog URR window (125 keV - 1 MeV). The
    // GPU now LogLog-interpolates the URR micro XS and excludes smooth
    // inelastic when the record's flag is <= 0, matching CPU `urr.rs`.
    let source_e = 3.0e5_f64;
    let Some((mut cpu_m, cpu_t, settings)) = mn55_sphere(7, 5.0, source_e) else {
        eprintln!("skipping -- cached Mn55 data not present");
        return;
    };
    cpu_m.simulate_transport(&settings).unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();
    let (mut gpu_m, gpu_t, settings) = mn55_sphere(7, 5.0, source_e).unwrap();
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    let ratio = gpu / cpu;
    eprintln!("Mn55 r=5 ({source_e:.2e} eV, LogLog URR): CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {ratio:.3}");
    assert!(
        (0.92..1.08).contains(&ratio),
        "Mn55 LogLog URR GPU/CPU flux ratio {ratio:.3} outside [0.92, 1.08]"
    );
}

#[test]
fn gpu_urr_co58_above_range_unchanged() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // Source at 14 MeV -- well above Co58's URR range. The URR
    // branch should never fire, so the result is unchanged from
    // pre-URR-PR behaviour. Same shape as the
    // `gpu_high_energy_unchanged_by_free_gas` regression: a
    // sanity check that adding URR didn't perturb the fast-energy
    // path.
    let source_e = 1.4e7_f64;
    let (mut cpu_m, cpu_t, settings) = co58_sphere(99, 5.0, source_e);
    cpu_m.simulate_transport(&settings).unwrap();
    let cpu = cpu_t.get_mean().iter().sum::<f64>();
    let (mut gpu_m, gpu_t, settings) = co58_sphere(99, 5.0, source_e);
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean().iter().sum::<f64>();
    let ratio = gpu / cpu;
    eprintln!(
        "Co58 r=5 ({source_e:.2e} eV, above URR): CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {ratio:.3}"
    );
    assert!(
        (0.85..1.15).contains(&ratio),
        "Co58 14 MeV (above URR) GPU/CPU flux ratio {ratio:.3} outside [0.85, 1.15] -- \
         URR branch may be firing outside its energy range"
    );
}
