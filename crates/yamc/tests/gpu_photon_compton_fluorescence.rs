//! GPU-vs-CPU regression test for Compton-ionization fluorescence.
//!
//! An iron sphere is irradiated by a 1.25 MeV photon point source. At that
//! energy incoherent (Compton) scattering dominates and photoelectric
//! absorption of the *primary* is negligible, so most Fe K-shell vacancies are
//! created when a Compton scatter ionizes a bound K electron. The atom relaxes
//! and emits the characteristic Fe K-fluorescence line (Kalpha ~6.40 keV,
//! Kbeta ~7.06 keV), which shows up as a sharp peak in an energy-binned flux
//! tally, well above the smooth Compton down-scatter continuum around it.
//!
//! The CPU photon transport banks this Compton-ionization fluorescence
//! (crates/yamc/src/transport/photon.rs). Before the matching GPU change the
//! GPU Compton branch sampled the recoil but never relaxed the ionized shell,
//! so the GPU K-line was ~30% low versus the CPU. This test pins the fix:
//! - the K-line is a clear peak above the local continuum on BOTH backends;
//! - the GPU and CPU K-line-band flux agree within a tight band. Without the
//!   GPU relaxation cascade the K-line-band ratio drops to ~0.70 (measured),
//!   which fails the assertion.
//!
//! Uses the checked-in `tests/Fe.arrow` photon data (Compton profile + atomic
//! relaxation tables), so it runs offline. Self-skips without an f64 GPU
//! adapter so CI stays green.

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

/// 1.25 MeV (a Co-60-like line): Compton dominates and primary photoelectric
/// is negligible, so Fe K-vacancies come predominantly from Compton ionization.
const SOURCE_E: f64 = 1.25e6;

/// Explicit bin edges (eV). Bin index 2 = [5.9, 7.3] keV brackets the Fe
/// K-fluorescence lines (Kalpha 6.40, Kbeta 7.06; K-edge 7.11). Bins 1 and 3
/// are the local Compton continuum immediately below and above the line.
const EDGES: [f64; 8] = [1.0e3, 4.0e3, 5.9e3, 7.3e3, 11.0e3, 30.0e3, 200.0e3, 1.3e6];
/// Index of the Fe K-line bin in `EDGES`.
const KLINE: usize = 2;

fn build() -> (Model, Arc<Tally>, TransportSettings) {
    let sphere = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 1.0,
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
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        yamc_particle::particle::ParticleType::Photon,
    )));
    t.filters
        .push(Filter::Energy(EnergyFilter::new(EDGES.to_vec())));
    t.scores = vec![Score::Flux(FluxScore)];
    t.initialize_batches(8);
    let t = Arc::new(t);

    let mut model = Model::new(geometry, vec![source], vec![t.clone()]);
    model.max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(50_000 * 8),
        seed: 13,
        ..Default::default()
    };
    (model, t, settings)
}

#[test]
fn gpu_photon_compton_fluorescence_matches_cpu() {
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
        .expect("CPU run");
    let cpu = cpu_t.get_mean();

    let (mut gpu_m, gpu_t, settings) = build();
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean();

    eprintln!("\nFe K-fluorescence spectrum (1.25 MeV source, CPU vs GPU):");
    eprintln!("| bin | E_lo (keV) | E_hi (keV) | CPU flux | GPU flux | ratio |");
    eprintln!("|---|---|---|---|---|---|");
    for i in 0..cpu.len() {
        let ratio = if cpu[i] > 0.0 {
            format!("{:.3}", gpu[i] / cpu[i])
        } else {
            "n/a".to_string()
        };
        eprintln!(
            "| {i} | {:.2} | {:.2} | {:.4e} | {:.4e} | {ratio} |",
            EDGES[i] / 1e3,
            EDGES[i + 1] / 1e3,
            cpu[i],
            gpu[i],
        );
    }

    // (a) The K-line is a clear peak above the local continuum on BOTH
    // backends: the K-line bin must exceed 3x the larger of its two
    // neighbouring continuum bins.
    for (label, spec) in [("CPU", &cpu), ("GPU", &gpu)] {
        let line = spec[KLINE];
        let below = spec[KLINE - 1];
        let above = spec[KLINE + 1];
        let continuum = below.max(above);
        eprintln!(
            "{label}: K-line bin = {line:.4e}, neighbour continuum = {continuum:.4e} \
             (below {below:.4e}, above {above:.4e})"
        );
        assert!(
            line > 3.0 * continuum,
            "{label}: Fe K-line bin flux {line:.4e} is not clearly above the \
             local continuum {continuum:.4e} -- fluorescence peak absent"
        );
    }

    // (b) GPU and CPU agree on the K-line-band flux within a tight band. The
    // Compton-ionization fluorescence is ~30% of this line; dropping the GPU
    // relaxation cascade pushes the ratio to ~0.70, failing this bound.
    let cpu_line = cpu[KLINE];
    let gpu_line = gpu[KLINE];
    let ratio = gpu_line / cpu_line;
    eprintln!(
        "Fe K-line-band flux: CPU = {cpu_line:.4e}, GPU = {gpu_line:.4e}, ratio = {ratio:.3}"
    );
    assert!(
        (0.85..=1.15).contains(&ratio),
        "Fe K-line GPU/CPU flux ratio {ratio:.3} outside [0.85, 1.15] -- \
         Compton-ionization fluorescence differs between backends"
    );

    // (c) Sanity: integrated flux agrees within the usual photon band.
    let cpu_tot: f64 = cpu.iter().sum();
    let gpu_tot: f64 = gpu.iter().sum();
    assert!(cpu_tot > 0.0 && gpu_tot > 0.0, "empty spectrum");
    let tot_ratio = gpu_tot / cpu_tot;
    eprintln!("integrated flux: CPU = {cpu_tot:.4e}, GPU = {gpu_tot:.4e}, ratio = {tot_ratio:.3}");
    assert!(
        (0.9..=1.1).contains(&tot_ratio),
        "integrated GPU/CPU photon flux ratio {tot_ratio:.3} outside [0.9, 1.1]"
    );
}
