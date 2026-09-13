//! GPU/CPU parity for the URR probability-table band-hold rule (issue #342).
//!
//! A neutron's URR band is drawn once per ENERGY, not once per transport step:
//! OpenMC advances its URR seed only when the energy changes
//! (`physics.cpp:164`) and the CPU does the same via `Particle.urr_random` /
//! `urr_energy` (PR #207, which is what moved a W shell from 0.919 to ~1.00
//! against OpenMC). The kernel used to redraw the base uniform on every step,
//! so an isotope spanning more than one material presented a fresh, independent
//! resonance realisation at each boundary crossing.
//!
//! The invariant this test pins is a physical one, so it needs no reference
//! data: subdividing one homogeneous sphere into more `Material` objects is the
//! same physics problem, so the GPU/CPU flux ratio must not depend on how many
//! subdivisions there are. Before the fix, W184 read 0.9994 at one material and
//! 1.2927 at three; a no-URR isotope was flat across all three.
//!
//! Self-skips without an f64 GPU adapter or the cached ENDF/B-VIII.1 W184 data.
#![cfg(all(feature = "gpu", not(target_os = "macos")))]

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
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

/// W184 carries URR over 1e4 .. 1e5 eV; the source sits inside that band so
/// every collision of the walk samples the probability table.
const ISOTOPE: &str = "W184";
const DENSITY: f64 = 19.3;
const SOURCE_ENERGY: f64 = 5.0e4;
const RADIUS: f64 = 30.0;
const HISTORIES: usize = 200_000;

fn cache(n: &str) -> String {
    yamc_test_cache::nuclide_path(n)
}

/// One solid sphere of `ISOTOPE`, cut into `n_mats` concentric shells that each
/// get their own `Material`. Flux is tallied in the outermost shell, the
/// quantity most sensitive to how the band correlates across crossings.
fn build(n_mats: usize) -> Option<(Model, Arc<Tally>, TransportSettings)> {
    if !std::path::Path::new(&cache(ISOTOPE)).exists() {
        return None;
    }
    let mk_mat = |id: u32| -> Option<Arc<Material>> {
        let composition: HashMap<String, f64> = [(ISOTOPE.to_string(), 1.0)].into_iter().collect();
        let data: HashMap<String, String> = [(ISOTOPE.to_string(), cache(ISOTOPE))]
            .into_iter()
            .collect();
        let mut m = Material::new(composition, "atom", "g/cm3", Some(DENSITY)).ok()?;
        m.set_material_id(id);
        m.set_temperature("294");
        m.read_nuclear_data(&data, None).ok()?;
        Some(Arc::new(m))
    };
    let mats: Vec<Arc<Material>> = (0..n_mats).map(|i| mk_mat(i as u32 + 1).unwrap()).collect();

    let mut cells = Vec::new();
    let mut prev: Option<Arc<Surface>> = None;
    for i in 0..n_mats {
        let surface = Arc::new(Surface {
            surface_id: Some(i + 1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: RADIUS * (i as f64 + 1.0) / n_mats as f64,
            },
            boundary: if i + 1 == n_mats {
                BoundaryType::Vacuum
            } else {
                BoundaryType::Transmission
            },
            name: None,
        });
        let inside = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&surface)));
        let region = match prev {
            None => inside,
            Some(ref p) => Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(p)))
                .intersection(&inside),
        };
        cells.push(Cell::new(
            Some(i as u32 + 1),
            region,
            Some(format!("shell{i}")),
            Some(i as u32),
        ));
        prev = Some(surface);
    }
    let geometry = Geometry::new(cells, mats).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_ENERGY], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let mut tally = Tally::new();
    tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(n_mats as u32)));
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.initialize_batches(1);
    let tally = Arc::new(tally);

    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = 20_000;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(HISTORIES),
        seed: 4242,
        threads: Some(8),
        ..Default::default()
    };
    Some((model, tally, settings))
}

/// GPU/CPU outer-shell flux ratio for an `n_mats`-way subdivision.
fn gpu_cpu_ratio(n_mats: usize) -> Option<f64> {
    let (mut cpu, ct, settings) = build(n_mats)?;
    cpu.simulate_transport(&settings).unwrap();
    let (mut gpu, gt, settings) = build(n_mats).unwrap();
    yamc::gpu::run_on_gpu(&mut gpu, &settings).unwrap();
    let c: f64 = ct.get_mean().iter().sum();
    let g: f64 = gt.get_mean().iter().sum();
    assert!(c > 0.0, "{n_mats}-material CPU flux must be positive");
    Some(g / c)
}

#[test]
fn urr_band_survives_material_boundaries() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping gpu_urr_band_hold -- no f64 GPU adapter");
        return;
    }
    let Some(one) = gpu_cpu_ratio(1) else {
        eprintln!("skipping gpu_urr_band_hold -- no cached {ISOTOPE} data");
        return;
    };
    let three = gpu_cpu_ratio(3).unwrap();
    eprintln!("GPU/CPU outer-shell flux: 1 material {one:.4}, 3 materials {three:.4}");

    // One material has no interior crossings, so it is the control: it agreed
    // even with the per-step redraw, and a failure here is some other defect.
    assert!(
        (0.96..=1.04).contains(&one),
        "single-material GPU/CPU flux {one:.4} outside [0.96, 1.04]"
    );
    // The headline: subdividing the same sphere must not move the answer. This
    // read 1.2927 while the kernel redrew the band every step.
    assert!(
        (0.96..=1.04).contains(&three),
        "three-material GPU/CPU flux {three:.4} outside [0.96, 1.04] -- \
         is the kernel redrawing the URR band per step again (#342)?"
    );
}
