//! GPU against CPU photon flux in the once-scattered band
//! (fusion-neutronics/core#34 entry 3).
//!
//! The entry recorded a GPU excess of 0.8% and 1.4% in the 0.5 to 1 MeV and
//! 1 to 1.5 MeV bands of a 2 MeV photon source in an iron sphere, with the
//! source band and the total agreeing to 0.2%, and named the Compton chain as
//! the suspect. The Compton Doppler sampler has since been rewritten on both
//! backends to OpenMC PR 4036 (fusion-neutronics/core#22), so this re-measures
//! the same three bands and holds them to the CPU in units of the combined
//! statistical error. Measured on the stack that carries that rewrite: at 1M
//! histories the bands are 0.9930 / 0.9995 / 1.0009 of the CPU (z -2.6, -0.1,
//! +1.0) and with a second seed at 4M histories 1.0014 / 0.9970 / 1.0002 (z
//! +1.1, -1.6, +0.4), so the excess is gone and what remains changes sign
//! between seeds.
//!
//! Run it (needs an f64 GPU and the Fe56 / Fe fixtures):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_photon_once_scattered_band -- --nocapture

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
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
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 20260913;
const SOURCE_E: f64 = 2.0e6;
const N_HISTORIES: usize = 1_000_000;
const RADIUS: f64 = 5.0;

/// The entry's three bands: once-scattered low, once-scattered high, source.
fn band_edges() -> Vec<f64> {
    vec![5.0e5, 1.0e6, 1.5e6, 2.1e6]
}

fn build_model() -> (Model, Arc<Tally>, TransportSettings) {
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
        .push(Filter::Energy(EnergyFilter::new(band_edges())));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    let t = Arc::new(t);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&t)]);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(N_HISTORIES),
        seed: SEED,
        threads: Some(0),
        ..Default::default()
    };
    (model, t, settings)
}

fn run_gpu_retry(model: &mut Model, settings: &TransportSettings) -> Result<(), String> {
    let mut last = String::new();
    for attempt in 0..5 {
        match yamc::gpu::run_on_gpu(model, settings) {
            Ok(_) => return Ok(()),
            Err(e) => {
                last = e.to_string();
                let transient = last.contains("BufferAsync") || last.contains("buffer async");
                if !transient {
                    return Err(last);
                }
                eprintln!("GPU transient error (attempt {attempt}): {last}; retrying");
            }
        }
    }
    Err(last)
}

#[test]
fn once_scattered_bands_on_the_gpu_match_the_cpu() {
    if !std::path::Path::new("tests/Fe56.arrow").exists()
        || !std::path::Path::new("tests/Fe.arrow").exists()
        || yamc_gpu::GpuContext::new().is_err()
    {
        eprintln!("skipping: no Fe fixtures or no f64 GPU");
        return;
    }
    let (mut cpu, cpu_t, cpu_settings) = build_model();
    cpu.simulate_transport(&cpu_settings).expect("CPU run");
    let (mut gpu, gpu_t, gpu_settings) = build_model();
    run_gpu_retry(&mut gpu, &gpu_settings).expect("GPU run");

    let edges = band_edges();
    let cm = cpu_t.get_mean().to_vec();
    let cs = cpu_t.get_std_dev().to_vec();
    let gm = gpu_t.get_mean().to_vec();
    let gs = gpu_t.get_std_dev().to_vec();
    assert_eq!(cm.len(), 3);
    let cpu_total: f64 = cm.iter().sum();
    let gpu_total: f64 = gm.iter().sum();
    eprintln!("total flux GPU/CPU {:.4}", gpu_total / cpu_total);
    let mut worst_z = 0.0f64;
    for i in 0..3 {
        let sigma = (cs[i] * cs[i] + gs[i] * gs[i]).sqrt();
        let z = (gm[i] - cm[i]) / sigma;
        eprintln!(
            "[{:>8.2e}, {:>8.2e}) eV  CPU {:.5} +- {:.5}  GPU {:.5} +- {:.5}  GPU/CPU {:.4}  z {:+.2}",
            edges[i],
            edges[i + 1],
            cm[i],
            cs[i],
            gm[i],
            gs[i],
            gm[i] / cm[i],
            z
        );
        worst_z = worst_z.max(z.abs());
    }
    // Independent estimates of the same three band fluxes. The entry's 1.4%
    // excess on the 1 to 1.5 MeV band would be about 18 sigma at this history
    // count.
    assert!(
        worst_z < 4.0,
        "worst band {worst_z:.2} sigma from the CPU (once-scattered photon band)"
    );
}
