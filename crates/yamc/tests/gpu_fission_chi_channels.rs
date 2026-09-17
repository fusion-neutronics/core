//! Fission chi per channel on the GPU (fusion-neutronics/core#34 entry 1).
//!
//! U240 is the one ENDF/B-VIII.1 fissionable whose evaluation carries the
//! partial fission channels MT 19 / 20 / 21 / 38, each with its own prompt
//! spectrum (MT 21 is markedly softer: mean E_out 1.66 MeV against 2.22 MeV
//! for MT 19 at 14.06 MeV incident). The CPU samples the channel that
//! fissioned and takes that channel's chi; the GPU used to hold one prompt
//! spectrum per material, filled from MT 19, so every U240 fission on the
//! device was first-chance fission regardless of energy. The kernel now selects
//! the channel from the struck nuclide's per-channel cross sections and samples
//! that channel's row.
//!
//! This holds the GPU neutron spectrum on a U240 sphere to the CPU's, bin by
//! bin in the fission-neutron range, in units of the combined statistical
//! error. The two runs share the seed and the history-keyed streams, so they
//! are largely in lockstep and the comparison is far more sensitive than two
//! independent estimates would be: before the per-channel chi the 4 to 6 MeV
//! bin sat 10% (13.6 sigma) above the CPU's and the reduced chi-square was 22;
//! after it every bin is within 0.15%. Run it (needs an f64 GPU and the U240
//! fixture):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_fission_chi_channels -- --nocapture

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
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 20260913;
const SOURCE_E: f64 = 14.06e6;
const N_HISTORIES: usize = 400_000;
const DENSITY: f64 = 19.1;
const RADIUS: f64 = 12.0;

/// Bin edges (eV) across the fission-neutron range up to the source line.
fn spectrum_edges() -> Vec<f64> {
    vec![
        1.0e3, 1.0e5, 3.0e5, 6.0e5, 1.0e6, 1.5e6, 2.0e6, 3.0e6, 4.0e6, 6.0e6, 9.0e6, 1.3e7, 1.5e7,
    ]
}

fn spectrum_tally(cell_id: u32) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.filters
        .push(Filter::Energy(EnergyFilter::new(spectrum_edges())));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

fn build_model() -> (Model, Arc<Tally>, TransportSettings) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([("U240".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(DENSITY),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material
        .read_nuclear_data(
            &HashMap::from([("U240".to_string(), "tests/U240.arrow".to_string())]),
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
    let t = spectrum_tally(cell_id);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&t)]);
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model.gpu_max_steps_per_particle = 10_000;
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
fn u240_spectrum_on_the_gpu_matches_the_cpu_bin_by_bin() {
    if !std::path::Path::new("tests/U240.arrow").exists() || yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no U240 fixture or no f64 GPU");
        return;
    }
    let (mut cpu, cpu_t, cpu_settings) = build_model();
    cpu.simulate_transport(&cpu_settings).expect("CPU run");
    let (mut gpu, gpu_t, gpu_settings) = build_model();
    run_gpu_retry(&mut gpu, &gpu_settings).expect("GPU run");

    let edges = spectrum_edges();
    let cm = cpu_t.get_mean().to_vec();
    let cs = cpu_t.get_std_dev().to_vec();
    let gm = gpu_t.get_mean().to_vec();
    let gs = gpu_t.get_std_dev().to_vec();
    assert_eq!(cm.len(), edges.len() - 1);
    assert_eq!(gm.len(), cm.len());
    let cpu_total: f64 = cm.iter().sum();
    let gpu_total: f64 = gm.iter().sum();
    eprintln!("total flux GPU/CPU {:.4}", gpu_total / cpu_total);
    let mut worst_z = 0.0f64;
    let mut sum_z2 = 0.0f64;
    let mut n_bins = 0usize;
    for i in 0..cm.len() {
        let sigma = (cs[i] * cs[i] + gs[i] * gs[i]).sqrt();
        if cm[i] <= 0.0 || sigma <= 0.0 {
            continue;
        }
        let z = (gm[i] - cm[i]) / sigma;
        eprintln!(
            "[{:>8.2e}, {:>8.2e}) eV  CPU {:.5e}  GPU {:.5e}  GPU/CPU {:.4}  z {:+.2}",
            edges[i],
            edges[i + 1],
            cm[i],
            gm[i],
            gm[i] / cm[i],
            z
        );
        worst_z = worst_z.max(z.abs());
        sum_z2 += z * z;
        n_bins += 1;
    }
    assert!(n_bins >= 10, "only {n_bins} populated bins");
    let chi2_dof = sum_z2 / n_bins as f64;
    eprintln!("worst |z| {worst_z:.2}, chi2/dof {chi2_dof:.2} over {n_bins} bins");
    // Two estimates of the same spectrum (largely lockstep, see the module
    // doc): |z| stays below 4 in every bin and the reduced chi-square below 2.
    // Before the per-channel chi, the GPU's unrestricted MT 19 spectrum for
    // every fission put the 4 to 6 MeV bin 13.6 sigma high at this history
    // count.
    assert!(
        worst_z < 4.0,
        "worst bin {worst_z:.2} sigma from the CPU (per-channel fission chi is off)"
    );
    assert!(
        chi2_dof < 2.0,
        "reduced chi-square {chi2_dof:.2} across the spectrum bins"
    );
}
