//! Survival biasing on a fissile material: the GPU runs the CPU / OpenMC
//! scheme (fusion-neutronics/core#25).
//!
//! Under implicit capture the CPU banks fission progeny from the pre-discount
//! weight, scaled by `sigma_f / sigma_t`, discounts the survivor by absorption
//! INCLUDING fission and scatters it. The GPU used to remove only capture and
//! sample fission analog alongside scatter. Both are unbiased, so the flux mean
//! never said which was running; the per-history variance and the fission-rate
//! tally did, and so did the loss of per-history comparability between the
//! backends. With the kernel on the same scheme, flux and fission rate agree
//! in mean AND in the per-history standard deviation the tallies report.
//!
//! Two fixtures. U240 is the only fissionable nuclide in the fixture set
//! (its partial fission channels are why it is there, issue #425); at 14 MeV
//! its fission cross section is about 1.3 b against ~6 b total, so roughly
//! one collision in five banks progeny under this scheme. U235 runs when the
//! endf-b8.1 cache has it (it is 189 MB and not a fixture), and is the
//! actinide every other fissile GPU test uses.
//!
//! Run it (needs an f64 GPU and the U240 fixture; U235 from the cache):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_survival_fissile_scheme -- --nocapture

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc::variance_reduction::{SurvivalBiasing, VarianceReduction};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 20260913;
const SOURCE_E: f64 = 14.06e6;
const N_HISTORIES: usize = 100_000;

/// A fissile sphere case: nuclide, where its data is, density, and a radius
/// of about three mean free paths at 14 MeV so a history collides several
/// times and the discount / banking scheme matters. U235 is kept small enough
/// to stay well subcritical.
struct Case {
    nuclide: &'static str,
    data: String,
    density: f64,
    radius: f64,
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    if std::path::Path::new("tests/U240.arrow").exists() {
        v.push(Case {
            nuclide: "U240",
            data: "tests/U240.arrow".to_string(),
            density: 19.1,
            radius: 12.0,
        });
    }
    if let Some(path) = yamc_test_cache::nuclide("U235") {
        v.push(Case {
            nuclide: "U235",
            data: path,
            density: 18.7,
            radius: 5.0,
        });
    }
    v
}

fn tally(cell_id: u32, score: &str) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        ParticleType::Neutron,
    )));
    t.scores = vec![score.parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

fn build_model(case: &Case, survival: bool) -> (Model, TransportSettings) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: case.radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([(case.nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(case.density),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material
        .read_nuclear_data(
            &HashMap::from([(case.nuclide.to_string(), case.data.clone())]),
            None,
        )
        .unwrap();
    let cell = Cell::new(Some(1), region, Some("u".into()), Some(0));
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let tallies = vec![tally(cell_id, "flux"), tally(cell_id, "fission")];
    let mut model = Model::new(geometry, vec![source], tallies);
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model.gpu_max_steps_per_particle = 10_000;
    if survival {
        model.variance_reduction = vec![VarianceReduction::SurvivalBiasing(
            SurvivalBiasing::default(),
        )];
    }
    let settings = TransportSettings {
        total_particles: Some(N_HISTORIES),
        seed: SEED,
        threads: Some(0),
        ..Default::default()
    };
    (model, settings)
}

/// `(mean, std_dev)` per tally: flux first, then the fission rate.
type Stats = [(f64, f64); 2];

/// `(mean, std_dev)` of tally `index`'s single bin.
fn stats(model: &Model, index: usize) -> (f64, f64) {
    let t = &model.tallies[index];
    let mean = t.get_mean();
    let sd = t.get_std_dev();
    assert_eq!(mean.len(), 1);
    (mean[0], sd[0])
}

fn run_both(case: &Case, survival: bool) -> (Stats, Stats) {
    let (mut cpu, cpu_settings) = build_model(case, survival);
    cpu.simulate_transport(&cpu_settings).expect("CPU run");
    let (mut gpu, gpu_settings) = build_model(case, survival);
    yamc::gpu::run_on_gpu(&mut gpu, &gpu_settings).expect("GPU run");
    (
        [stats(&cpu, 0), stats(&cpu, 1)],
        [stats(&gpu, 0), stats(&gpu, 1)],
    )
}

fn check(label: &str, cpu: Stats, gpu: Stats) {
    for (name, c, g) in [("flux", cpu[0], gpu[0]), ("fission", cpu[1], gpu[1])] {
        let mean_ratio = g.0 / c.0;
        let sd_ratio = g.1 / c.1;
        eprintln!(
            "{label} {name:7}: mean cpu {:.6e} gpu {:.6e} ratio {mean_ratio:.5} | std_dev cpu \
             {:.4e} gpu {:.4e} ratio {sd_ratio:.4}",
            c.0, g.0, c.1, g.1
        );
        // The mean is unbiased under either scheme; 1% is far outside the
        // statistical spread at 100k histories.
        assert!(
            (mean_ratio - 1.0).abs() < 0.01,
            "{label} {name}: GPU/CPU mean ratio {mean_ratio:.5}"
        );
        // The per-history standard deviation is what the scheme decides. One
        // known residual remains after the scheme change: the host relaunches
        // a banked progeny of fractional weight w as round(w) unit-weight
        // copies (`fission_source_inputs`, issue #236), a Russian-roulette
        // step the CPU does not take since it transports the progeny at
        // weight w. Under survival biasing every banked weight is fractional,
        // so the GPU's per-history spread sits a little above the CPU's:
        // measured 1.04 on U240 and 1.10 on U235, against 1.00 to 1.02 on the
        // analog controls where the weights are 1. Relaunching at the banked
        // weight is fusion-neutronics/core#88 and would close it; until then
        // 15% holds the scheme (a mismatched scheme moves this by more, and in
        // the fission-rate std_dev most of all) without asserting the residual
        // away.
        assert!(
            (sd_ratio - 1.0).abs() < 0.15,
            "{label} {name}: GPU/CPU per-history std_dev ratio {sd_ratio:.4}; the two \
             backends are not running the same implicit-capture scheme"
        );
    }
}

/// Survival biasing on, fission bank on: same scheme, same mean, same
/// per-history spread on flux and on the fission rate.
#[test]
fn survival_biased_fissile_run_matches_cpu_in_mean_and_variance() {
    let cases = cases();
    if cases.is_empty() || yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no fissile data or no f64 GPU");
        return;
    }
    for case in &cases {
        let (cpu, gpu) = run_both(case, true);
        check(&format!("{} survival", case.nuclide), cpu, gpu);
    }
}

/// Control: analog (no survival biasing) on the same model, where the two
/// backends already agreed. Pins that the agreement above is not tolerance.
#[test]
fn analog_fissile_run_still_matches_cpu() {
    let cases = cases();
    if cases.is_empty() || yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no fissile data or no f64 GPU");
        return;
    }
    for case in &cases {
        let (cpu, gpu) = run_both(case, false);
        check(&format!("{} analog", case.nuclide), cpu, gpu);
    }
}
