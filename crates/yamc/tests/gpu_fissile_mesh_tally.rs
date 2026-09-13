//! Mesh tallies on a fissile model with the GPU fission bank on
//! (fusion-neutronics/core#30).
//!
//! The dispatch used to refuse this combination because the fissile loop
//! accumulates per SOURCE neutron across fission generations while a mesh
//! tally scores directly into the per-source accumulator, and nothing had
//! shown the two together kept the variance right. Now the fissile loop runs
//! every launch (source and generation) in the direct per-source mode when a
//! mesh is present, so a descendant's voxel crossings fold into its source's
//! sample before squaring, exactly as its cell-bin contributions do.
//!
//! A mean-only check would pass even with the variance wrong, which is the
//! failure the refusal protected against, so this holds the per-voxel
//! standard deviation to the CPU as well as the mean, on the same fixtures
//! and with the same bounds as the non-fissile `gpu_mesh_tally`. It also
//! asserts the fission bank actually ran (progeny were re-launched), so the
//! test cannot pass by quietly taking the non-fissile loop.
//!
//! Run it (needs an f64 GPU and the U240 fixture; U235 from the cache):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_fissile_mesh_tally -- --nocapture --test-threads=1

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::gpu::GpuRunResult;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::mesh::MeshFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 20260913;
const SOURCE_E: f64 = 14.06e6;
const N_HISTORIES: usize = 200_000;
const SHAPE: usize = 4;

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

/// `SHAPE^3` voxels over the sphere's bounding box, row-major.
fn mesh_flux_tally(radius: f64) -> Arc<Tally> {
    let mesh = RegularRectangularMesh::new(
        [-radius, -radius, -radius],
        [radius, radius, radius],
        [SHAPE, SHAPE, SHAPE],
    );
    let mut t = Tally::new();
    t.filters.push(Filter::Mesh(MeshFilter::new(mesh)));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

/// A cell flux tally beside the mesh, so the run also checks that the cell
/// path is untouched by the mesh switch.
fn cell_flux_tally(cell_id: u32) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

fn build_model(case: &Case) -> (Model, Arc<Tally>, Arc<Tally>, TransportSettings) {
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
    let mesh_t = mesh_flux_tally(case.radius);
    let cell_t = cell_flux_tally(cell_id);
    let mut model = Model::new(
        geometry,
        vec![source],
        vec![Arc::clone(&mesh_t), Arc::clone(&cell_t)],
    );
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model.gpu_max_steps_per_particle = 10_000;
    assert!(model.gpu_fission_bank, "the bank is on by default");
    let settings = TransportSettings {
        total_particles: Some(N_HISTORIES),
        seed: SEED,
        threads: Some(0),
        ..Default::default()
    };
    (model, mesh_t, cell_t, settings)
}

fn run_gpu_retry(model: &mut Model, settings: &TransportSettings) -> Result<GpuRunResult, String> {
    let mut last = String::new();
    for attempt in 0..5 {
        match yamc::gpu::run_on_gpu(model, settings) {
            Ok(r) => return Ok(r),
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

fn compare(label: &str, cpu: &Tally, gpu: &Tally, well_populated: f64) {
    let cpu_mean = cpu.get_mean().to_vec();
    let cpu_sd = cpu.get_std_dev().to_vec();
    let gpu_mean = gpu.get_mean().to_vec();
    let gpu_sd = gpu.get_std_dev().to_vec();
    assert_eq!(cpu_mean.len(), gpu_mean.len(), "{label}: bin counts differ");
    let cpu_total: f64 = cpu_mean.iter().sum();
    let gpu_total: f64 = gpu_mean.iter().sum();
    assert!(
        cpu_total > 0.0 && gpu_total > 0.0,
        "{label}: nothing scored"
    );
    let total_ratio = gpu_total / cpu_total;
    let max_mean = cpu_mean.iter().cloned().fold(0.0f64, f64::max);
    let mut worst_mean_dev = 0.0f64;
    let mut sd_ratios = Vec::new();
    for i in 0..cpu_mean.len() {
        if cpu_mean[i] < well_populated * max_mean || cpu_sd[i] <= 0.0 || gpu_sd[i] <= 0.0 {
            continue;
        }
        worst_mean_dev = worst_mean_dev.max((gpu_mean[i] - cpu_mean[i]).abs() / cpu_mean[i]);
        sd_ratios.push(gpu_sd[i] / cpu_sd[i]);
    }
    assert!(
        !sd_ratios.is_empty(),
        "{label}: no well-populated bins to judge"
    );
    sd_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_sd = sd_ratios[sd_ratios.len() / 2];
    eprintln!(
        "{label}: total GPU/CPU {total_ratio:.4}; {} well-populated bins, worst mean dev {:.1}%, \
         median GPU/CPU std_dev ratio {median_sd:.3} (min {:.3}, max {:.3})",
        sd_ratios.len(),
        100.0 * worst_mean_dev,
        sd_ratios[0],
        sd_ratios[sd_ratios.len() - 1],
    );
    // Same bounds as the non-fissile `gpu_mesh_tally`: independent MC estimates
    // of the same per-voxel flux a few percent apart, and a per-source fold
    // that mis-attributed a descendant's contributions would move the median
    // std_dev ratio well outside this band while leaving the means alone.
    assert!(
        (0.98..=1.02).contains(&total_ratio),
        "{label}: GPU/CPU total ratio {total_ratio:.4} outside [0.98, 1.02]"
    );
    assert!(
        worst_mean_dev < 0.06,
        "{label}: worst well-populated-bin mean deviation {:.1}% exceeds 6%",
        100.0 * worst_mean_dev
    );
    assert!(
        (0.85..=1.18).contains(&median_sd),
        "{label}: median GPU/CPU std_dev ratio {median_sd:.3} not near 1 (per-source mesh \
         variance across fission generations is biased)"
    );
}

#[test]
fn fissile_mesh_tally_with_the_fission_bank_matches_cpu_mean_and_variance() {
    let cases = cases();
    if cases.is_empty() || yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no fissile data or no f64 GPU");
        return;
    }
    for case in &cases {
        let (mut cpu, cpu_mesh, cpu_cell, cpu_settings) = build_model(case);
        cpu.simulate_transport(&cpu_settings).expect("CPU run");
        let (mut gpu, gpu_mesh, gpu_cell, gpu_settings) = build_model(case);
        let result = run_gpu_retry(&mut gpu, &gpu_settings).expect("GPU run");
        assert!(
            result.n_bank_relaunched > 0,
            "{}: no fission progeny was re-launched from the device bank, so this run did not \
             exercise the fissile loop the mesh switch is for",
            case.nuclide
        );
        eprintln!(
            "{}: {} progeny re-launched from the bank",
            case.nuclide, result.n_bank_relaunched
        );
        compare(
            &format!("{} mesh", case.nuclide),
            &cpu_mesh,
            &gpu_mesh,
            0.05,
        );
        compare(&format!("{} cell", case.nuclide), &cpu_cell, &gpu_cell, 0.0);
    }
}
