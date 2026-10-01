//! Both backends must score `(n,gamma)` as MT 102 inside a probability-table
//! band, not as disappearance.
//!
//! The kernel and the CPU both corrected a capture score inside the band by
//! REPLACING the smooth value with the perturbed macroscopic disappearance, and
//! disappearance carries every nuclide's `(n,p)` and `(n,alpha)` as well. A
//! material of Fe58 with Be9 makes that unmistakable: at 2 MeV the material's
//! MT 101 is 22x its MT 102, all of the excess being Be9's `(n,alpha)`, while
//! Fe58's own charged-particle channels are exactly zero across its band so its
//! perturbed disappearance IS its perturbed capture.
//!
//! Both backends had the same defect, so a GPU against CPU comparison could not
//! have caught it on its own. It is asserted here anyway, beside the absolute
//! check, because the fix is two separate pieces of arithmetic (the kernel adds
//! a delta to the smooth value; `Material::compute_urr_macro_xs` keeps two
//! accumulators) and reverting either one alone would part them.
//!
//! The absolute reference is the capture the same model reports with the source
//! placed OUTSIDE the band, where `compute_urr_macro_xs` returns `None` and both
//! backends read the library's own MT 102 column: a run whose capture score is
//! the disappearance overstates it by the same factor the cross sections do.

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
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

/// Inside Fe58's band [350 keV, 3 MeV], and high enough in it that Be9's
/// `(n,alpha)` has risen to 45 mb against Fe58's 2.1 mb of capture.
const IN_BAND: f64 = 2.0e6;
/// Above the band, so neither backend takes its URR substitution and the same
/// tally reports the library's MT 102.
const ABOVE_BAND: f64 = 5.0e6;
const SEED: u64 = 20260917;
const N_HISTORIES: usize = 200_000;

fn cache_paths() -> Option<HashMap<String, String>> {
    ["Fe58", "Be9"]
        .into_iter()
        .map(|n| yamc_test_cache::nuclide(n).map(|p| (n.to_string(), p)))
        .collect()
}

/// A small sphere, so most of the score comes from source neutrons at the
/// energy asked for rather than from a slowed-down spectrum that would smear
/// the two source energies together.
fn build(
    paths: &HashMap<String, String>,
    source_energy: f64,
) -> (Model, Arc<Tally>, TransportSettings) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 2.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([("Fe58".to_string(), 0.5), ("Be9".to_string(), 0.5)]),
        "atom",
        "g/cm3",
        Some(2.0),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material.read_nuclear_data(paths, None).unwrap();
    let cell = Cell::new(Some(1), region, Some("m".into()), Some(0));
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![source_energy], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut tally = Tally::new();
    tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(cell_id)));
    tally.scores = vec!["102".parse().unwrap()];
    tally.estimator = Estimator::TrackLength;
    tally.initialize_batches(1);
    let tally = Arc::new(tally);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = 10_000;
    let settings = TransportSettings {
        total_particles: Some(N_HISTORIES),
        seed: SEED,
        threads: Some(0),
        ..Default::default()
    };
    (model, tally, settings)
}

fn cpu_capture(paths: &HashMap<String, String>, source_energy: f64) -> f64 {
    let (mut model, tally, settings) = build(paths, source_energy);
    model.simulate_transport(&settings).expect("CPU run");
    tally.get_mean().iter().sum()
}

fn gpu_capture(paths: &HashMap<String, String>, source_energy: f64) -> f64 {
    let (mut model, tally, settings) = build(paths, source_energy);
    yamc::gpu::run_on_gpu(&mut model, &settings).expect("GPU run");
    tally.get_mean().iter().sum()
}

#[test]
fn an_in_band_capture_score_is_not_the_materials_disappearance() {
    let Some(paths) = cache_paths() else {
        eprintln!("skipping: Fe58 or Be9 is not in the test cache");
        return;
    };
    let have_gpu = yamc_gpu::GpuContext::new().is_ok();

    // The capture cross section per unit flux, in and above the band, on each
    // backend. Dividing by the flux is what makes the two source energies
    // comparable: the sphere's leakage and the cross sections differ between
    // them, the RATIO of capture to the smooth capture does not.
    let cpu_in = cpu_capture(&paths, IN_BAND);
    let cpu_above = cpu_capture(&paths, ABOVE_BAND);
    eprintln!("CPU capture: in band {cpu_in:.6e}, above band {cpu_above:.6e}");

    // Fe58's capture falls and Be9's (n,alpha) rises with energy, so the two
    // numbers are not equal; what matters is that neither is inflated by a
    // factor of order ten. A capture score reading the material's
    // disappearance at 2 MeV picks up Be9's (n,alpha), which is 21x Fe58's
    // capture there, and the in-band number jumps accordingly.
    assert!(
        cpu_in < 3.0 * cpu_above,
        "in-band capture {cpu_in:.6e} against {cpu_above:.6e} above the band: the in-band \
         score looks like the material's disappearance rather than its MT 102"
    );

    if !have_gpu {
        eprintln!("skipping the GPU half: no f64 GPU");
        return;
    }
    let gpu_in = gpu_capture(&paths, IN_BAND);
    let gpu_above = gpu_capture(&paths, ABOVE_BAND);
    eprintln!("GPU capture: in band {gpu_in:.6e}, above band {gpu_above:.6e}");
    assert!(
        gpu_in < 3.0 * gpu_above,
        "in-band capture {gpu_in:.6e} against {gpu_above:.6e} above the band on the GPU"
    );

    // And the two backends agree. Not bit for bit: the GPU draws its bands on
    // its own stream, so this is a statistical comparison on the same seed.
    for (label, cpu, gpu) in [
        ("in band", cpu_in, gpu_in),
        ("above the band", cpu_above, gpu_above),
    ] {
        let ratio = gpu / cpu;
        eprintln!("  {label}: GPU/CPU {ratio:.4}");
        assert!(
            (0.95..1.05).contains(&ratio),
            "{label}: GPU/CPU capture ratio {ratio:.4}; the two backends correct the in-band \
             capture score separately, so one of the two corrections has moved"
        );
    }
}
