//! DIAGNOSTIC (not a CI test): compare the CPU vs GPU photon flux
//! spectrum, energy bin by energy bin, on the Fe sphere used by the
//! photoelectric matrix sweep. photoelectric ∝ E^-3 is dominated by the
//! low-energy tail; this shows WHERE the GPU under-populates so we can
//! attribute the photoelectric deficit (#415). Run with:
//!   cargo test --release --test gpu_photon_spectrum_diag -- --nocapture --test-threads=1
//!
//! Both tests here assert nothing: they print tables for a person to read.
//! They no longer carry `#[ignore]`, because the adapter guard each one opens
//! with is already the right gate: they compare CPU against GPU, so without a
//! GPU there is nothing to print, and with one there is a person looking. On a
//! CI runner the guard skips them for free. `#[ignore]` on top of that only
//! meant a developer on a GPU box had to know to pass `-- --ignored` to run
//! the one thing these exist for.
//!
//! The command above lost `--features gpu` too, since that feature is in the
//! default set now. It keeps `--test-threads=1`, which is not optional: the
//! GPU tests share a process-global cubecl client and parallel launches
//! corrupt results (see the `test-gpu` alias in `.cargo/config.toml`).

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
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

fn bins() -> Vec<f64> {
    // Log-spaced edges from 1 keV (the cutoff) to 1.3 MeV.
    let (lo, hi, n) = (1.0e3_f64, 1.3e6_f64, 24usize);
    let ln_lo = lo.ln();
    let ln_hi = hi.ln();
    (0..=n)
        .map(|i| (ln_lo + (ln_hi - ln_lo) * (i as f64) / (n as f64)).exp())
        .collect()
}

fn build() -> (Model, Arc<Tally>, TransportSettings) {
    build_r(1.0)
}

fn build_r(radius: f64) -> (Model, Arc<Tally>, TransportSettings) {
    let sphere = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();
    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.25e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        yamc_particle::particle::ParticleType::Photon,
    )));
    t.filters.push(Filter::Energy(EnergyFilter::new(bins())));
    t.scores = vec![Score::Flux(FluxScore)];
    t.initialize_batches(8);
    let t = Arc::new(t);
    let mut model = Model::new(geometry, vec![source], vec![t.clone()]);
    model.max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(2_500 * 8),
        seed: 42,
        ..Default::default()
    };
    (model, t, settings)
}

/// Vary the sphere radius. If the GPU deep tail (< ~49 keV, bins 0..=12)
/// fills in relative to CPU as the sphere grows (more chances to scatter),
/// the per-scatter energy loss is fine and escape was truncating the chain.
/// If it stays empty even in a large sphere, the per-scatter Compton energy
/// loss is too small. (#415)
#[test]
fn diag_radius_sweep_deep_tail() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    eprintln!("\n| radius cm | deep-tail CPU (<49keV) | deep-tail GPU | GPU/CPU | total CPU | total GPU |");
    eprintln!("|---|---|---|---|---|---|");
    for &r in &[1.0_f64, 5.0, 20.0] {
        let (mut cpu_m, cpu_t, settings) = build_r(r);
        cpu_m
            .simulate_transport(&TransportSettings {
                threads: Some(1),
                ..settings
            })
            .unwrap();
        let cpu = cpu_t.get_mean();
        let (mut gpu_m, gpu_t, settings) = build_r(r);
        yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
        let gpu = gpu_t.get_mean();
        // bins 0..=12 are < ~49 keV (see bins()).
        let cpu_tail: f64 = cpu[0..=12].iter().sum();
        let gpu_tail: f64 = gpu[0..=12].iter().sum();
        let cpu_tot: f64 = cpu.iter().sum();
        let gpu_tot: f64 = gpu.iter().sum();
        eprintln!(
            "| {r} | {cpu_tail:.4e} | {gpu_tail:.4e} | {:.3} | {cpu_tot:.4e} | {gpu_tot:.4e} |",
            gpu_tail / cpu_tail
        );
    }
}

#[test]
fn diag_photon_flux_spectrum_cpu_vs_gpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    let (mut cpu_m, cpu_t, settings) = build();
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .unwrap();
    let cpu = cpu_t.get_mean();

    // CPU with secondary photons (TTB) OFF -- isolates how much of the
    // low-E tail is bremsstrahlung vs primary Compton downscatter.
    let (mut cpu_no_ttb_m, cpu_no_ttb_t, settings) = build();
    cpu_no_ttb_m.transport_secondary_photons = false;
    cpu_no_ttb_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .unwrap();
    let cpu_no_ttb = cpu_no_ttb_t.get_mean();

    let (mut gpu_m, gpu_t, settings) = build();
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean();

    let edges = bins();
    eprintln!("\n| bin | E_lo (eV) | E_hi (eV) | CPU(+TTB) | CPU(-TTB) | GPU | GPU/CPU |");
    eprintln!("|---|---|---|---|---|---|---|");
    for i in 0..cpu.len() {
        let r = if cpu[i] > 0.0 {
            gpu[i] / cpu[i]
        } else {
            f64::NAN
        };
        eprintln!(
            "| {i} | {:.3e} | {:.3e} | {:.4e} | {:.4e} | {:.4e} | {:.3} |",
            edges[i],
            edges[i + 1],
            cpu[i],
            cpu_no_ttb[i],
            gpu[i],
            r
        );
    }
    let cpu_tot: f64 = cpu.iter().sum();
    let cpu_no_ttb_tot: f64 = cpu_no_ttb.iter().sum();
    let gpu_tot: f64 = gpu.iter().sum();
    eprintln!(
        "TOTAL flux CPU(+TTB)={cpu_tot:.4e} CPU(-TTB)={cpu_no_ttb_tot:.4e} GPU={gpu_tot:.4e}"
    );
}
