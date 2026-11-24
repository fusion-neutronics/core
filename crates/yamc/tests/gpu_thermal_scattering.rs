//! Thermal scattering tables on the GPU and in its CPU twin.
//!
//! A bound nuclide below its table's energy_max takes the table's elastic
//! and total cross sections and scatters by the table. The GPU used to refuse
//! a material with `s_alpha_beta`; it now builds the bound cross sections into
//! its buffers and samples the table in the kernel with the CPU's draws.
//!
//! Neutrons born at 1 eV slow down and thermalise in a graphite sphere
//! (coherent elastic and inelastic) and a light-water sphere (incoherent
//! elastic and inelastic on H). The flux spectrum in four bins must agree
//! between the production CPU, the GPU and the GPU's CPU twin. The tables also
//! have to matter: graphite's thermal spectrum with its table differs from the
//! free-gas one.
//!
//! The thermal tables are not among the CI fixtures. Set
//! `YAMC_THERMAL_LIBRARY` to a library keyword whose release carries
//! `c_Graphite` and `c_H_in_H2O` (`endf-b8.1`); its tables and nuclides are
//! downloaded on first use. Without it the tests skip. Run them:
//!   YAMC_THERMAL_LIBRARY=endf-b8.1 cargo test -p yamc --features gpu,download-tls \
//!       --release --test gpu_thermal_scattering -- --nocapture --test-threads=1
#![cfg(all(feature = "gpu", feature = "download", not(target_os = "macos")))]

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use common::gpu_twin::run_twin;
use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_gpu::common::tallies::TalliesPack;
use yamc_gpu::neutron::transport::PendDrain;
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
use yamc_tallies::tally::Tally;

const HISTORIES: usize = 20_000;
const MAX_STEPS: u32 = 20_000;
const SEED: u64 = 11;
const RADIUS: f64 = 40.0;
/// Flux bins (eV): deep thermal, thermal peak, epithermal tail, source.
const BINS: [f64; 5] = [1e-5, 0.01, 0.1, 1.0, 2.0];

/// The library keyword the materials and their tables read from: one source,
/// as a material's thermal tables require.
fn library() -> Option<String> {
    std::env::var("YAMC_THERMAL_LIBRARY")
        .ok()
        .filter(|s| !s.is_empty())
}

/// A material of `composition` at `density` g/cm3 and `kelvin`, its nuclides
/// and `tables` read from `library`.
fn material(
    library: &str,
    composition: &[(&str, f64)],
    density: f64,
    kelvin: &str,
    tables: &[&str],
) -> Material {
    let mut m = Material::new(
        composition
            .iter()
            .map(|(n, f)| (n.to_string(), *f))
            .collect(),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature(kelvin);
    m.set_s_alpha_beta(tables.iter().map(|t| (t.to_string(), 1.0)).collect())
        .unwrap();
    let data: HashMap<String, String> = composition
        .iter()
        .map(|(n, _)| (n.to_string(), library.to_string()))
        .collect();
    m.read_nuclear_data(&data, None).unwrap();
    m
}

fn model(material: Material) -> (Model, Arc<Tally>) {
    let sphere = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let cell = Cell::new(
        Some(1),
        Region::new_from_halfspace(HalfspaceType::Below(sphere)),
        Some("moderator".into()),
        Some(0),
    );
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.0], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut tally = Tally::new();
    tally.filters.push(Filter::Cell(CellFilter::from_id(1)));
    tally
        .filters
        .push(Filter::Energy(EnergyFilter::new(BINS.to_vec())));
    tally.scores = vec!["flux".parse().unwrap()];
    tally.reset_accumulation();
    let tally = Arc::new(tally);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    (model, tally)
}

fn settings() -> TransportSettings {
    TransportSettings {
        total_particles: Some(HISTORIES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    }
}

/// The four-bin flux spectrum per source neutron from the CPU, the GPU and
/// the twin, and the CPU's standard deviations.
fn spectra(make: &dyn Fn() -> Material) -> ([f64; 4], [f64; 4], [f64; 4], [f64; 4]) {
    let (mut cpu_model, cpu_tally) = model(make());
    cpu_model.simulate_transport(&settings()).unwrap();
    let cpu: [f64; 4] = cpu_tally.get_mean().try_into().unwrap();
    let sd: [f64; 4] = cpu_tally.get_std_dev().try_into().unwrap();

    let (mut gpu_model, gpu_tally) = model(make());
    yamc::gpu::run_on_gpu(&mut gpu_model, &settings()).expect("GPU transport");
    let gpu: [f64; 4] = gpu_tally.get_mean().try_into().unwrap();

    let (twin_model, _) = model(make());
    let inputs =
        yamc::gpu::translate::translate_for_gpu(&twin_model, HISTORIES, SEED).expect("translate");
    let edges: Vec<f64> = BINS.iter().map(|e| e.ln()).collect();
    let pack = TalliesPack::flux_abs_pack((inputs.cell_aabbs.len() / 6) as u32, &edges);
    let (result, _) = run_twin(&inputs, &pack, MAX_STEPS, 400.0, false, PendDrain::Fifo);
    let twin: [f64; 4] = std::array::from_fn(|b| result.tally_outputs[0][b] / HISTORIES as f64);
    (cpu, sd, gpu, twin)
}

/// Each bin of `got` within 5 standard deviations (or 3%) of the CPU.
fn assert_agrees(label: &str, cpu: &[f64; 4], sd: &[f64; 4], got: &[f64; 4]) {
    for b in 0..4 {
        let tol = (5.0 * sd[b]).max(0.03 * cpu[b]);
        assert!(
            (got[b] - cpu[b]).abs() <= tol,
            "{label} flux in [{:e}, {:e}] eV is {:.5e} against the CPU's {:.5e} +- {:.1e}",
            BINS[b],
            BINS[b + 1],
            got[b],
            cpu[b],
            sd[b]
        );
    }
}

fn check(name: &str, make: &dyn Fn() -> Material) -> [f64; 4] {
    let (cpu, sd, gpu, twin) = spectra(make);
    eprintln!("{name}: CPU {cpu:?}\n{name}: GPU {gpu:?}\n{name}: twin {twin:?}");
    assert_agrees(&format!("{name} GPU"), &cpu, &sd, &gpu);
    assert_agrees(&format!("{name} twin"), &cpu, &sd, &twin);
    cpu
}

#[test]
fn graphite_with_its_thermal_table_matches_the_cpu() {
    let Some(lib) = library() else {
        eprintln!("skipping: YAMC_THERMAL_LIBRARY not set");
        return;
    };
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no GPU with f64 compute");
        return;
    }
    let comp = [("C12", 0.9893), ("C13", 0.0107)];
    // Graphite's tables start at 296 K; 600 K is a rung of the table and of
    // the carbon data both.
    let bound = check("graphite", &|| {
        material(&lib, &comp, 1.7, "600", &["c_Graphite"])
    });
    // Without the table the thermal spectrum moves well past the noise in at
    // least one bin: the table is doing something.
    let (free, sd, _, _) = spectra(&|| material(&lib, &comp, 1.7, "600", &[]));
    let shift = (0..4)
        .map(|b| (bound[b] - free[b]).abs() / sd[b].max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max);
    eprintln!("graphite bound {bound:?}\ngraphite free {free:?}\nlargest shift {shift:.1} sd");
    assert!(
        shift > 5.0,
        "the graphite table moves no flux bin by more than {shift:.1} standard deviations"
    );
}

#[test]
fn light_water_with_its_thermal_table_matches_the_cpu() {
    let Some(lib) = library() else {
        eprintln!("skipping: YAMC_THERMAL_LIBRARY not set");
        return;
    };
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no GPU with f64 compute");
        return;
    }
    check("light water", &|| {
        material(
            &lib,
            &[("H1", 2.0), ("O16", 1.0)],
            1.0,
            "294",
            &["c_H_in_H2O"],
        )
    });
}
