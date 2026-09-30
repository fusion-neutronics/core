//! GPU against CPU total photon flux across Z ("photoelectric is Z-biased").
//!
//! An earlier measurement recorded, on 1 MeV photon spheres at 50M histories,
//! the GPU flux 0.25% low on W and 0.22% low on Pb (15 sigma) and 0.17% / 0.15% high
//! on C and O, crossing zero at mid Z, and asked for a re-measurement at HEAD
//! before any new theory, since the leading explanation (a truncated GPU
//! relaxation cascade) has been removed. This runs that sweep on the current
//! stack: one single-nuclide sphere per element carrying the element's photon
//! data, 1 MeV isotropic point source, default 1 keV cutoff, secondary
//! photons on, total flux GPU against CPU.
//!
//! Measured on the stack that carries the Compton Doppler rewrite, 10M
//! histories at seed 20260913 then 50M at seed 7: W -0.018% / -0.042%,
//! Pb -0.039% / -0.063%, C -0.009% / +0.011%, O +0.003% / +0.005%. The bias was
//! four to five times smaller than filed but the same shape, and at 50M
//! histories still real (Pb 6.6 sigma).
//!
//! The cause was the photoelectron's bremsstrahlung direction: the GPU gave the
//! brem the parent photon's direction, where the CPU and OpenMC both give it a
//! Sauter-sampled one. The photoelectron carries nearly the whole photon
//! energy, so in a high-Z medium that brem is a large share of the secondary
//! flux, and aiming it along the parent's flight out of a central point source
//! shortened its tracks. Photoelectric is ~2.5% of 1 MeV collisions in Pb and
//! ~2e-5 in C, which is the whole Z shape. With the GPU sampling the same
//! Sauter direction, at 50M / seed 7: W -0.015% (z -1.32), Pb -0.003%
//! (z -0.36), C +0.011% (z +1.02), O +0.004% (z +0.52). Pb's 6.6 sigma is
//! 0.36 sigma and nothing here is significant.
//!
//! The bound below is a relative one, tightened to 0.1% now that the residual
//! is consistent with zero. It stays well clear of the ~0.02% one-sigma noise
//! at the default 10M so the sweep cannot drift back unnoticed.
//!
//! `YAMC_Z_SWEEP_HISTORIES` and `YAMC_Z_SWEEP_SEED` override the history count
//! (default 10M) and the seed. Run
//! it (needs an f64 GPU and the W184 / Pb208 / C12 / O16 nuclide fixtures with
//! the W / Pb / C / O photon data in the cache):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_photon_z_sweep -- --nocapture

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
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 20260913;
const SOURCE_E: f64 = 1.0e6;

struct Case {
    element: &'static str,
    nuclide: &'static str,
    density: f64,
    radius: f64,
}

/// A few mean free paths at 1 MeV for each element.
const CASES: [Case; 4] = [
    Case {
        element: "W",
        nuclide: "W184",
        density: 19.3,
        radius: 2.0,
    },
    Case {
        element: "Pb",
        nuclide: "Pb208",
        density: 11.35,
        radius: 2.0,
    },
    Case {
        element: "C",
        nuclide: "C12",
        density: 2.26,
        radius: 10.0,
    },
    Case {
        element: "O",
        nuclide: "O16",
        density: 1.14,
        radius: 10.0,
    },
];

fn histories() -> usize {
    std::env::var("YAMC_Z_SWEEP_HISTORIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000)
}

fn seed() -> u64 {
    std::env::var("YAMC_Z_SWEEP_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(SEED)
}

fn photon_path(element: &str) -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let p = format!("{home}/.cache/yamc/endf-b8.1-{element}.arrow");
    std::path::Path::new(&p).is_dir().then_some(p)
}

fn build_model(case: &Case, photon_path: &str) -> (Model, Arc<Tally>, TransportSettings) {
    let sphere = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: case.radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
    let mut material = Material::new(
        HashMap::from([(case.nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(case.density),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([(
        case.nuclide.to_string(),
        format!("tests/{}.arrow", case.nuclide),
    )]);
    let photon_paths = HashMap::from([(case.element.to_string(), photon_path.to_string())]);
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();
    let cell = Cell::new(Some(1), region, Some(case.element.into()), Some(0));
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
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    let t = Arc::new(t);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&t)]);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(histories()),
        seed: seed(),
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
fn total_photon_flux_across_z_matches_the_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no f64 GPU");
        return;
    }
    let mut worst_z = 0.0f64;
    let mut worst_rel = 0.0f64;
    let mut ran = 0usize;
    for case in &CASES {
        let Some(photon_path) = photon_path(case.element) else {
            eprintln!("{}: photon data not in the cache, skipping", case.element);
            continue;
        };
        if !std::path::Path::new(&format!("tests/{}.arrow", case.nuclide)).exists() {
            eprintln!("{}: nuclide fixture absent, skipping", case.nuclide);
            continue;
        }
        let (mut cpu, cpu_t, cpu_settings) = build_model(case, &photon_path);
        cpu.simulate_transport(&cpu_settings).expect("CPU run");
        let (mut gpu, gpu_t, gpu_settings) = build_model(case, &photon_path);
        run_gpu_retry(&mut gpu, &gpu_settings).expect("GPU run");
        let (cm, cs) = (cpu_t.get_mean()[0], cpu_t.get_std_dev()[0]);
        let (gm, gs) = (gpu_t.get_mean()[0], gpu_t.get_std_dev()[0]);
        let sigma = (cs * cs + gs * gs).sqrt();
        let z = (gm - cm) / sigma;
        eprintln!(
            "{:>2} (r = {:>4.1} cm, {} histories): CPU {:.6} +- {:.6}  GPU {:.6} +- {:.6}  GPU/CPU {:.5} ({:+.3}%)  z {:+.2}",
            case.element,
            case.radius,
            histories(),
            cm,
            cs,
            gm,
            gs,
            gm / cm,
            100.0 * (gm / cm - 1.0),
            z
        );
        worst_z = worst_z.max(z.abs());
        worst_rel = worst_rel.max((gm / cm - 1.0).abs());
        ran += 1;
    }
    assert!(ran > 0, "no element had its data present");
    // The entry filed 0.25% on W and 0.22% on Pb. With the photoelectron brem
    // direction fixed the residual is consistent with zero (see the module
    // doc), so the bound is 0.1%: ~5 sigma of headroom on the noisiest element
    // at the default 10M, and tight enough that the old 0.04% to 0.06% heavy-
    // element bias would show up as a near-miss in the log line below.
    // `worst_z` is reported for the log.
    eprintln!(
        "worst |z| {worst_z:.2}, worst |GPU/CPU - 1| {:.4}%",
        100.0 * worst_rel
    );
    assert!(
        worst_rel < 0.001,
        "worst element {:.3}% from the CPU (photoelectric Z bias)",
        100.0 * worst_rel
    );
}
