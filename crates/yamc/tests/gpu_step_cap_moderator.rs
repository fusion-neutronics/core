//! GPU vs CPU integral-flux regression for a strong, near-pure scatterer.
//!
//! The CPU transport loop runs every history to completion (it ignores
//! `gpu_max_steps_per_particle` entirely). The GPU kernel cannot loop
//! unbounded, so it caps each history at `gpu_max_steps_per_particle` steps.
//! When that cap was 1000, a 14.06 MeV neutron in a 35 cm pure-H2 sphere
//! -- which undergoes many hundreds of elastic collisions before leaking
//! (H2 barely absorbs) -- got truncated mid-history on the GPU. Its
//! remaining track length was never scored, so the GPU integral flux came
//! out ~10% LOW (ratio ~0.90) while the CPU was unaffected.
//!
//! The fix raises the default cap to 100000 (the loop still exits the
//! instant a particle leaks or is absorbed, so it is free for fast cases).
//! This test pins the behaviour: with the default cap the GPU/CPU flux
//! ratio must be within statistical noise. The historical 1000-step cap is
//! also exercised: a binding cap is now an error rather than a warning
//! (fusion-neutronics/core#23), because the under-counted flux it produces
//! is not a valid answer, so that case asserts the refusal instead of the
//! ~10% deficit it used to document.

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

fn sphere(
    nuclide: &str,
    arrow: &str,
    seed: u64,
    max_steps: u32,
) -> (Model, Arc<Tally>, TransportSettings) {
    let surface = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 35.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(surface)));
    let mut material = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.0),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert(nuclide.to_string(), arrow.to_string());
    material.read_nuclear_data(&nm, None).unwrap();
    let cell = Cell::new(Some(1), region, Some("s".into()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let n_particles = 20_000;
    let n_batches = 4;
    let mut tally = Tally::new();
    tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap())));
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.initialize_batches(n_batches);
    let tally = Arc::new(tally);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.gpu_max_steps_per_particle = max_steps;
    let settings = TransportSettings {
        total_particles: Some(n_particles * n_batches),
        seed,
        threads: Some(1),
        ..Default::default()
    };
    (model, tally, settings)
}

fn cpu_flux(max_steps: u32) -> f64 {
    let (mut m, t, settings) = sphere("H2", "tests/H2.arrow", 7, max_steps);
    m.simulate_transport(&settings).unwrap();
    t.get_mean().iter().sum::<f64>()
}

fn gpu_flux(max_steps: u32) -> f64 {
    let (mut m, t, settings) = sphere("H2", "tests/H2.arrow", 7, max_steps);
    yamc::gpu::run_on_gpu(&mut m, &settings).expect("GPU dispatch");
    t.get_mean().iter().sum::<f64>()
}

fn gpu_run(max_steps: u32) -> Result<yamc::gpu::GpuRunResult, yamc::gpu::GpuDispatchError> {
    let (mut m, _t, settings) = sphere("H2", "tests/H2.arrow", 7, max_steps);
    yamc::gpu::run_on_gpu(&mut m, &settings)
}

#[test]
fn gpu_h2_flux_matches_cpu_with_default_step_cap() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // Default cap (100000 in Model::new) -- must agree with CPU.
    let cpu = cpu_flux(100_000);
    let gpu = gpu_flux(100_000);
    let ratio = gpu / cpu;
    eprintln!("H2 r=35 default cap: CPU = {cpu:.4e}  GPU = {gpu:.4e}  ratio = {ratio:.4}");
    assert!(
        (0.98..1.02).contains(&ratio),
        "H2 GPU/CPU flux ratio {ratio:.4} outside [0.98, 1.02] with the \
         default step cap -- the GPU step-cap truncation regression may be back"
    );
}

#[test]
fn gpu_h2_low_step_cap_is_refused() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }
    // The CPU ignores the cap (runs to completion). The GPU honours it, and
    // at 1000 steps a 14 MeV neutron in 35 cm of H2 is still scattering, so
    // histories truncate and the flux would come out ~10% LOW. That used to
    // be a stderr warning gated on `verbose.summary`; a `verbose=[]` run
    // returned the deficit in silence. It is an error now, raised at the
    // first launch that truncates, so the under-counted flux is never
    // returned at all. If the GPU ever stops truncating H2 at 1000 (e.g. the
    // cap semantics change) this test should be revisited.
    match gpu_run(1_000) {
        Err(yamc::gpu::GpuDispatchError::HistoriesTruncated {
            truncated,
            launched,
            max_steps,
        }) => {
            eprintln!("H2 r=35 cap=1000: {truncated} of {launched} truncated at {max_steps}");
            assert_eq!(max_steps, 1_000);
            assert!(truncated > 0 && truncated <= launched);
            // The message has to carry the remedy, since it is all the user sees.
            let msg = yamc::gpu::GpuDispatchError::HistoriesTruncated {
                truncated,
                launched,
                max_steps,
            }
            .to_string();
            assert!(msg.contains("gpu_max_steps_per_particle=1000"), "{msg}");
            assert!(msg.contains("Raise gpu_max_steps_per_particle"), "{msg}");
        }
        Err(other) => panic!("expected HistoriesTruncated, got {other}"),
        Ok(_) => panic!(
            "expected the 1000-step cap to truncate H2 histories and fail the run \
             (historically ~10% of the flux was lost); it ran to completion instead"
        ),
    }
}
