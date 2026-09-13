//! Issue #237: batch-free per-history GPU variance throughput vs tally size.
//!
//! The batch-free path (#233) accumulates a history's per-bin totals in a
//! `PERHIST_K = 32` register touched-list, spilling the rest to a per-history
//! global list, and is far slower on a large tally than on a moderate one. The
//! issue attributed that to the per-step linear scan of the spill list. It is
//! not that: with the two spill scans deleted outright (wrong answers, but a
//! clean upper bound on what removing them could buy) the large case measured
//! 1,756 particles/s against 1,703 with them -- 3%.
//!
//! What actually costs is OCCUPANCY. Per-history variance state is provisioned
//! per THREAD at the worst case (`per_history_spill_cap` words), so the launch
//! chunk is bounded by the spill memory budget, and a fine tally collapses it:
//! 1,330 histories per launch on the shape below, about one wavefront per CU,
//! with nothing in flight to hide the memory latency the kernel is bound by.
//! Throughput then tracks the chunk almost linearly (1,686 particles/s at 1,330,
//! 11,271 at 13,472), which is why the large tally looked ~4x slow.
//!
//! This is the measurement harness: a concentric-graphite-shell model with a
//! per-cell, energy-binned flux tally, run at a configurable `cells x energy
//! bins`, reporting particles/second. Ignored by default (it is a benchmark, not
//! an assertion) -- run it with:
//!
//! ```text
//! cargo test -p yamc --features gpu --release \
//!     --test gpu_large_tally_variance_perf -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `PERF_N` sets the history count, `PERF_STEPS` the step cap, and `PERF_SHAPE`
//! one shape (`101x500`) instead of the default pair.

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

const SEED: u64 = 20237;
const SOURCE_E: f64 = 14_000_000.0;
const MAX_STEPS: u32 = 10_000;
fn max_steps() -> u32 {
    std::env::var("PERF_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_STEPS)
}
/// Graphite: dense enough to moderate hard, so a history sweeps far more than
/// `PERHIST_K` distinct (cell, energy) bins and lands in the spill.
const DENSITY: f64 = 2.26;
/// Outer radius, held FIXED as the cell count varies so the two shapes differ
/// only in how finely the same sphere is binned (not in how much material a
/// history has to cross).
const OUTER_RADIUS: f64 = 10.0;

fn c12_path() -> String {
    yamc_test_cache::nuclide_path("C12")
}

fn data_present() -> bool {
    std::path::Path::new(&c12_path()).exists()
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

fn graphite() -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([("C12".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(DENSITY),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(&HashMap::from([("C12".to_string(), c12_path())]), None)
        .unwrap();
    Arc::new(m)
}

fn sphere(id: usize, r: f64, boundary: BoundaryType) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: Some(id),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: r,
        },
        boundary,
        name: None,
    })
}

/// `n_cells` concentric graphite shells: a core plus `n_cells - 1` shells, all
/// the same material, vacuum on the outside.
fn concentric_shells(n_cells: usize) -> (Geometry, Vec<u32>) {
    let surfaces: Vec<Arc<Surface>> = (1..=n_cells)
        .map(|i| {
            let boundary = if i == n_cells {
                BoundaryType::Vacuum
            } else {
                BoundaryType::Transmission
            };
            sphere(i, i as f64 * OUTER_RADIUS / n_cells as f64, boundary)
        })
        .collect();

    let mut cells = Vec::with_capacity(n_cells);
    let mut ids = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        let region = if i == 0 {
            Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&surfaces[0])))
        } else {
            Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(&surfaces[i - 1])))
                .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                    Arc::clone(&surfaces[i]),
                )))
        };
        let id = i as u32 + 1;
        ids.push(id);
        cells.push(Cell::new(Some(id), region, Some(format!("s{id}")), Some(0)));
    }
    // Outermost-first, as the other concentric-geometry tests build them.
    cells.reverse();
    (Geometry::new(cells, vec![graphite()]).unwrap(), ids)
}

fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

/// Log-spaced energy edges from 1e-5 eV to 20 MeV: the moderating spectrum
/// walks down them one collision at a time.
fn energy_bins(n: usize) -> Vec<f64> {
    let (lo, hi) = (1.0e-5_f64, 20.0e6_f64);
    (0..=n)
        .map(|i| lo * (hi / lo).powf(i as f64 / n as f64))
        .collect()
}

/// ONE flux tally binned by cell and by energy: `n_cells * n_energy` flat bins
/// in a single tally, which is how a per-cell spectrum is actually written (and
/// what the issue measured). One tally keeps the per-history spill bound at
/// `max_steps * n_tallies`, so the shape under test is the BIN count.
fn tallies(cell_ids: &[u32], n_energy: usize) -> Vec<Arc<Tally>> {
    let mut t = Tally::new();
    t.filters
        .push(Filter::Cell(CellFilter::from_ids(cell_ids.to_vec())));
    t.filters
        .push(Filter::Energy(EnergyFilter::new(energy_bins(n_energy))));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    vec![Arc::new(t)]
}

/// Transport `n_particles` and return particles/second (GPU wall clock).
fn run_case(n_cells: usize, n_energy: usize, n_particles: usize) -> f64 {
    let (geometry, cell_ids) = concentric_shells(n_cells);
    let tallies = tallies(&cell_ids, n_energy);
    let mut model = Model::new(geometry, vec![neutron_source()], tallies);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = max_steps();
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(n_particles),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };

    // Warm up: the first launch pays kernel compilation and the geometry/XS
    // upload, which at these particle counts would swamp the transport.
    let warm = TransportSettings {
        total_particles: Some(64),
        ..settings.clone()
    };
    yamc::gpu::run_on_gpu(&mut model, &warm).expect("gpu warmup");

    let start = std::time::Instant::now();
    let res = yamc::gpu::run_on_gpu(&mut model, &settings).expect("gpu run");
    let secs = start.elapsed().as_secs_f64();
    let steps = &res.n_steps;
    let mean: f64 = steps.iter().map(|&s| s as f64).sum::<f64>() / steps.len().max(1) as f64;
    let maxs = steps.iter().copied().max().unwrap_or(0);
    let capped = steps.iter().filter(|&&s| s >= max_steps()).count();
    println!(
        "      steps: mean={mean:.0} max={maxs} at_cap={capped}/{}",
        steps.len()
    );
    n_particles as f64 / secs
}

/// The issue's two shapes: a moderate tally that fits the register touched-list
/// and a large one that does not. On a RADV STRIX_HALO, 60k histories:
///
/// | shape             | bins   | before | after  |
/// |-------------------|--------|--------|--------|
/// | 11 x 50           |    550 | 470057 | 460425 |
/// | 101 x 50          |   5050 |  12462 |  12510 |
/// | 101 x 150         |  15150 |   4531 |   6969 |
/// | 101 x 500         |  50500 |   1703 |   7054 |
///
/// "before"/"after" straddle dropping the watchdog clamp from
/// `spill_bounded_mem_safe_max`, which had been holding the large tally's launch
/// chunk at 1,330 histories.
#[test]
#[ignore = "benchmark: needs a GPU and takes minutes"]
fn large_tally_throughput() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping: C12 data or f64 GPU absent");
        return;
    }
    let n: usize = std::env::var("PERF_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);

    let shapes: Vec<(usize, usize)> = match std::env::var("PERF_SHAPE").ok().as_deref() {
        Some("moderate") => vec![(11, 50)],
        Some("large") => vec![(101, 500)],
        Some(spec) if spec.contains('x') => {
            let (c, e) = spec.split_once('x').unwrap();
            vec![(c.trim().parse().unwrap(), e.trim().parse().unwrap())]
        }
        _ => vec![(11, 50), (101, 500)],
    };
    for (cells, energy) in shapes {
        let pps = run_case(cells, energy, n);
        println!(
            "{cells:4} cells x {energy:4} energy = {:6} bins: {pps:10.0} particles/s",
            cells * energy
        );
    }
}
