//! GPU `photon_cutoff_energy` (issue #286), on-hardware.
//!
//! The photon kernel used to hardcode the 1 keV default cutoff, so
//! `Model::photon_cutoff_energy` was a no-op on `compute='gpu'`: the GPU
//! returned a bit-identical flux for every cutoff while the CPU correctly
//! killed photons below it (GPU/CPU 2.14 at a 1 MeV cutoff). The cutoff now
//! rides in the kernel's `run_params` buffer and gates the top-of-step kill and
//! every secondary-emission site, as it does on the CPU.
//!
//! Self-skips if the endf-b8.1 Fe cache or an f64 GPU is absent. Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_photon_cutoff -- --nocapture --test-threads=1
#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
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

const SEED: u64 = 17;
const SOURCE_E: f64 = 2.0e6;
const TOTAL: usize = 40_000;

fn cache_dir(name: &str) -> String {
    yamc_test_cache::nuclide_path(name)
}

fn data_present() -> bool {
    std::path::Path::new(&cache_dir("Fe56")).is_dir()
        && std::path::Path::new(&cache_dir("Fe")).is_dir()
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

fn iron_sphere() -> Geometry {
    let outer = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 15.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region::new_from_halfspace(HalfspaceType::Below(outer));

    let mut material = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material
        .read_nuclear_data(
            &HashMap::from([("Fe56".to_string(), cache_dir("Fe56"))]),
            Some(&HashMap::from([("Fe".to_string(), cache_dir("Fe"))])),
        )
        .unwrap();

    let cell = Cell::new(Some(1), region, Some("iron".into()), Some(0));
    Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap()
}

fn photon_source() -> ParticleSource {
    ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn photon_flux_tally() -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        ParticleType::Photon,
    )));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

fn build(cutoff: f64) -> (Model, Arc<Tally>, TransportSettings) {
    let tally = photon_flux_tally();
    let mut model = Model::new(
        iron_sphere(),
        vec![photon_source()],
        vec![Arc::clone(&tally)],
    );
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model.photon_cutoff_energy = cutoff;
    let settings = TransportSettings {
        total_particles: Some(TOTAL),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (model, tally, settings)
}

fn cpu_flux(cutoff: f64) -> f64 {
    let (mut model, tally, settings) = build(cutoff);
    model.simulate_transport(&settings).expect("CPU run failed");
    tally.get_mean().iter().sum()
}

fn gpu_flux(cutoff: f64) -> f64 {
    let (mut model, tally, settings) = build(cutoff);
    let mut last = String::new();
    for _ in 0..5 {
        match yamc::gpu::run_on_gpu(&mut model, &settings) {
            Ok(_) => return tally.get_mean().iter().sum(),
            Err(e) => {
                last = e.to_string();
                if !last.contains("BufferAsync") {
                    break;
                }
            }
        }
    }
    panic!("GPU run failed: {last}");
}

/// The setting must do something at all: a 1 MeV cutoff on a 2 MeV source kills
/// every once-scattered photon that drops below it, so the flux must fall well
/// below the default-cutoff value. Before the fix these were bit-identical.
#[test]
fn gpu_photon_cutoff_changes_the_flux() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping gpu_photon_cutoff_changes_the_flux: data/GPU absent");
        return;
    }
    let default_cut = gpu_flux(1.0e3);
    let high_cut = gpu_flux(1.0e6);
    assert!(
        default_cut > 0.0 && high_cut > 0.0,
        "rig broken (zero flux)"
    );
    assert!(
        high_cut < 0.75 * default_cut,
        "a 1 MeV cutoff must cut the GPU photon flux well below the 1 keV value \
         (got {high_cut:.6} vs {default_cut:.6}); equal values mean the kernel \
         is ignoring photon_cutoff_energy again"
    );
}

/// And it must cut the same population the CPU cuts. The tolerance is 5%: a
/// residual ~2% remains at a 1 MeV cutoff from a pre-existing GPU/CPU
/// difference in the once-scattered photon spectrum (visible at the DEFAULT
/// cutoff too, as a +1.4% excess in the 1.0-1.5 MeV band), which is a separate
/// concern from the cutoff. Before the fix this ratio was 2.14.
#[test]
fn gpu_photon_cutoff_tracks_cpu() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping gpu_photon_cutoff_tracks_cpu: data/GPU absent");
        return;
    }
    for cutoff in [1.0e3, 1.0e6] {
        let cpu = cpu_flux(cutoff);
        let gpu = gpu_flux(cutoff);
        assert!(cpu > 0.0, "CPU flux zero at cutoff {cutoff:e}");
        let ratio = gpu / cpu;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "cutoff {cutoff:e}: GPU/CPU photon flux {ratio:.4} \
             (gpu {gpu:.6}, cpu {cpu:.6}) outside 5%"
        );
    }
}
