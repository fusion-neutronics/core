//! Regression for the #111 fission sub-step (CPU fission chi routed through the
//! shared `yamc_physics::gpu::flat` fission-spectrum samplers + the per-particle
//! PCG stream).
//!
//! A subcritical U235 sphere (1 MeV point source, 5 cm) so fission fires
//! heavily. Prints the per-bin flux spectrum + integrated flux. yamc-CPU is the
//! OpenMC-validated reference; this captures its spectrum so a pre-vs-post-Stage-B
//! run confirms the fission-chi migration preserves the spectrum (statistical),
//! and (with a GPU) checks GPU-vs-CPU agreement now that both sample the fission
//! chi through the flat path. Run:
//!   cargo test --release -p yamc --test fission_chi_unification -- --ignored --nocapture

// Per-bin spectrum loop indexes `flux` and `edges` in lockstep; a range loop
// reads more clearly than zipped iterators here.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

fn cache(n: &str) -> String {
    yamc_test_cache::nuclide_path(n)
}

fn build(nuclide: &str, density: f64) -> Option<(Model, Arc<Tally>, TransportSettings)> {
    if !std::path::Path::new(&cache(nuclide)).exists() {
        return None;
    }
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 5.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let composition = HashMap::from([(nuclide.to_string(), 1.0)]);
    let data = HashMap::from([(nuclide.to_string(), cache(nuclide))]);
    let mut material = Material::new(composition, "atom", "g/cm3", Some(density)).ok()?;
    material.set_material_id(1);
    material.set_temperature("294");
    material.read_nuclear_data(&data, None).ok()?;
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut edges = vec![1.0e-5];
    let mut e = 1.0e-1;
    while e < 2.0e7 {
        edges.push(e);
        e *= 10f64.powf(0.5);
    }
    edges.push(2.0e7);
    let mut tally = Tally::new();
    tally.filters.push(Filter::Cell(CellFilter::from_id(1)));
    tally.filters.push(Filter::Energy(EnergyFilter::new(edges)));
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.initialize_batches(1);
    let tally = Arc::new(tally);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = 20_000;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(200_000),
        seed: 20_240_622,
        threads: Some(1),
        ..Default::default()
    };
    Some((model, tally, settings))
}

/// Captures the CPU fission spectrum (run pre- and post-Stage-B to confirm the
/// fission-chi migration preserves it). Asserts the integrated flux stays within
/// 5% of the recorded pre-migration baseline.
#[test]
#[ignore = "heavy CPU fission transport; run with --release --ignored"]
fn cpu_u235_fission_spectrum_matches_baseline() {
    let Some((mut cpu, t, settings)) = build("U235", 18.95) else {
        eprintln!("U235: skip (data missing)");
        return;
    };
    cpu.simulate_transport(&settings).unwrap();
    let c = t.get_mean();
    let edges = if let Filter::Energy(ef) = t
        .filters
        .iter()
        .find(|f| matches!(f, Filter::Energy(_)))
        .unwrap()
    {
        ef.bins.clone()
    } else {
        vec![]
    };
    let total: f64 = c.iter().sum();
    eprintln!("=== U235 fission sphere (sum CPU = {total:.6e}) ===");
    for i in 0..c.len() {
        if c[i] <= 0.0 {
            continue;
        }
        eprintln!(
            "  {:>12.4e}  {:>14.6e}   {:>7.4}%",
            edges.get(i).copied().unwrap_or(0.0),
            c[i],
            c[i] / total * 100.0
        );
    }
    // Pre-Stage-B (OpenMC-validated) CPU integrated flux baseline, captured by
    // stashing the fission-chi migration and re-running (15.524; post-migration
    // 15.463, a 0.4% MC-noise shift). 5% band: well above MC noise at 200k
    // histories, below a real spectrum shift.
    let baseline = 1.5524e1;
    let ratio = total / baseline;
    assert!(
        (0.95..=1.05).contains(&ratio),
        "U235 fission CPU flux {total:.4e} vs baseline {baseline:.4e} (ratio {ratio:.4}) outside [0.95, 1.05]"
    );
}

/// GPU-vs-CPU on the U235 fission sphere: both now sample the fission chi
/// through the flat path, so the integrated flux must agree. Self-skips without
/// an f64 GPU.
#[test]
#[ignore = "needs GPU; run with --release --features gpu --ignored"]
#[cfg(feature = "gpu")]
fn gpu_u235_fission_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skip: no GPU");
        return;
    }
    let Some((mut cpu, ct, cpu_settings)) = build("U235", 18.95) else {
        eprintln!("U235: skip (data missing)");
        return;
    };
    cpu.simulate_transport(&cpu_settings).unwrap();
    let c: f64 = ct.get_mean().iter().sum();
    let (mut gpu, gt, gpu_settings) = build("U235", 18.95).unwrap();
    yamc::gpu::run_on_gpu(&mut gpu, &gpu_settings).unwrap();
    let g: f64 = gt.get_mean().iter().sum();
    let ratio = g / c;
    eprintln!("U235 fission: CPU = {c:.4e}  GPU = {g:.4e}  ratio = {ratio:.3}");
    assert!(
        (0.92..=1.08).contains(&ratio),
        "U235 fission GPU/CPU flux ratio {ratio:.3} outside [0.92, 1.08]"
    );
}
