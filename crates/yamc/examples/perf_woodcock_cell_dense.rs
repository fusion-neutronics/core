//! Where Woodcock *wins*: a cell-dense, single-material geometry.
//!
//! N concentric shells of the SAME material fill one ball, swept over shell
//! count. Surface tracking pays one boundary step at every shell crossing, so
//! its rate falls as the shells multiply; Woodcock samples flights against the
//! (single-material) majorant and crosses all those shells with no boundary
//! computation and zero fictitious collisions (Σ_local = Σ_global here). This
//! is the geometry class -- many cells per mean free path -- where Woodcock
//! beats surface tracking.
//!
//! This is exactly the void-free, dense case `TrackingMode::Woodcock` (pure
//! delta tracking) is meant for, so that is what it runs. Hybrid would give
//! the identical result here: a same-material cell has
//! `(Σ_global - Σ_local)·chord = 0` expected fictitious collisions, so the
//! hybrid fallback never fires and it stays on the Woodcock path anyway.
//!
//! With `--features gpu` it also measures GPU surface tracking (the GPU has
//! no delta tracking). Its rate at 1 shell over its rate at N shells is the
//! most delta tracking could gain on the GPU for that model: a single-material
//! delta tracker never computes an internal boundary, so it can at best run
//! the N-shell ball as fast as the 1-shell one.
//!
//! Run via:
//!
//!     cargo run --release --example perf_woodcock_cell_dense
//!     cargo run --release --features gpu --example perf_woodcock_cell_dense
//!
//! Measured (AMD Ryzen AI Max+ 395 CPU, all cores; Radeon 8060S GPU via
//! RADV; best of 3):
//!
//!   Li6, 0.46 g/cm3, r = 100 cm
//!   shells   gpu surface   cpu surface   cpu woodcock   gpu 1/N   cpu wood/surf
//!      1       3_813_409     5_844_803      5_352_050      1.00x       0.92x
//!      4       3_014_571     5_654_955      5_274_146      1.26x       0.93x
//!     16       1_565_601     4_201_248      4_561_691      2.44x       1.09x
//!     64         385_693     1_299_797      2_884_560      9.89x       2.22x
//!    128         135_250       470_147      1_988_633     28.20x       4.23x
//!
//!   Fe56, 7.874 g/cm3, r = 30 cm
//!      1         204_883     1_226_362      1_075_295      1.00x       0.88x
//!      4         178_771     1_175_124      1_074_427      1.15x       0.91x
//!     16         127_422     1_016_104        988_570      1.61x       0.97x
//!     64          44_270       473_230        779_168      4.63x       1.65x
//!    128          15_292       200_125        589_776     13.40x       2.95x
//!
//!   H2O, 1.0 g/cm3, r = 30 cm
//!      1         207_456     1_120_800        920_107      1.00x       0.82x
//!      4         176_112     1_098_707        877_998      1.18x       0.80x
//!     16         131_830       989_162        715_168      1.57x       0.72x
//!     64          55_796       530_616        440_985      3.72x       0.83x
//!    128          20_638       242_849        282_113     10.05x       1.16x
//!
//! (the live numbers printed below are what matters; the header is a guide)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::CellFilter;

const SHELL_COUNTS: &[usize] = &[1, 4, 16, 64, 128];
const CPU_PARTICLES: usize = 100_000;
#[cfg(feature = "gpu")]
const GPU_PARTICLES: usize = 400_000;
const REPEATS: usize = 3;

/// One swept material: label, `(nuclide, atom fraction)` list, density in
/// g/cm3, and the ball radius in cm.
struct Case {
    label: &'static str,
    nuclides: &'static [(&'static str, f64)],
    density: f64,
    r_max: f64,
}

const CASES: &[Case] = &[
    Case {
        label: "Li6, 0.46 g/cm3, r = 100 cm",
        nuclides: &[("Li6", 1.0)],
        density: 0.46,
        r_max: 100.0,
    },
    Case {
        label: "Fe56, 7.874 g/cm3, r = 30 cm",
        nuclides: &[("Fe56", 1.0)],
        density: 7.874,
        r_max: 30.0,
    },
    Case {
        label: "H2O, 1.0 g/cm3, r = 30 cm",
        nuclides: &[("H1", 2.0), ("O16", 1.0)],
        density: 1.0,
        r_max: 30.0,
    },
];

fn material(case: &Case) -> Material {
    let mut m = Material::new(
        case.nuclides
            .iter()
            .map(|(n, f)| (n.to_string(), *f))
            .collect(),
        "atom",
        "g/cm3",
        Some(case.density),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    let data: HashMap<String, String> = case
        .nuclides
        .iter()
        .map(|(n, _)| (n.to_string(), yamc_test_cache::nuclide_path(n)))
        .collect();
    m.read_nuclear_data(&data, None).unwrap();
    m
}

/// N concentric shells of one material filling a ball, vacuum outside.
fn build_model(case: &Case, n_shells: usize, mode: TrackingMode) -> Model {
    let surfaces: Vec<Arc<Surface>> = (1..=n_shells)
        .map(|i| {
            Arc::new(Surface {
                surface_id: Some(i),
                kind: SurfaceKind::Sphere {
                    x0: 0.0,
                    y0: 0.0,
                    z0: 0.0,
                    radius: case.r_max * (i as f64) / (n_shells as f64),
                },
                boundary: if i == n_shells {
                    BoundaryType::Vacuum
                } else {
                    BoundaryType::Transmission
                },
                name: None,
            })
        })
        .collect();

    let mut cells = Vec::with_capacity(n_shells);
    cells.push(Cell::new(
        Some(1),
        Region {
            expr: RegionExpr::Halfspace(HalfspaceType::Below(surfaces[0].clone())),
        },
        Some("shell_0".to_string()),
        Some(0),
    ));
    for i in 1..n_shells {
        cells.push(Cell::new(
            Some((i + 1) as u32),
            Region {
                expr: RegionExpr::Intersection(
                    Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
                        surfaces[i - 1].clone(),
                    ))),
                    Box::new(RegionExpr::Halfspace(HalfspaceType::Below(
                        surfaces[i].clone(),
                    ))),
                ),
            },
            Some(format!("shell_{i}")),
            Some(0),
        ));
    }

    let geometry = Geometry::new(cells, vec![Arc::new(material(case))]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    // Flux in the innermost cell: a single-cell filter keeps the tally cheap
    // and identical across shell counts.
    let mut tally = Tally::new();
    tally.estimator = yamc_tallies::Estimator::Collision;
    tally.filters = vec![Filter::Cell(CellFilter::from_id(1))];
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.name = Some("flux".to_string());

    let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = mode;
    model.max_lost_particles = usize::MAX;
    // Every shell crossing is a GPU step: give 128 shells room.
    model.gpu_max_steps_per_particle = 50_000;
    model
}

/// Best of `REPEATS` runs, in particles per second.
fn rate(case: &Case, n_shells: usize, mode: TrackingMode, gpu: bool) -> f64 {
    let particles = if gpu { gpu_particles() } else { CPU_PARTICLES };
    let mut best = 0.0f64;
    for _ in 0..REPEATS {
        let mut model = build_model(case, n_shells, mode);
        let settings = TransportSettings {
            total_particles: Some(particles),
            seed: 42,
            ..Default::default()
        };
        let t0 = Instant::now();
        if gpu {
            run_gpu(&mut model, &settings);
        } else {
            model.simulate_transport(&settings).unwrap();
        }
        best = best.max(particles as f64 / t0.elapsed().as_secs_f64());
    }
    best
}

#[cfg(feature = "gpu")]
fn gpu_particles() -> usize {
    GPU_PARTICLES
}

#[cfg(not(feature = "gpu"))]
fn gpu_particles() -> usize {
    0
}

#[cfg(feature = "gpu")]
fn run_gpu(model: &mut Model, settings: &TransportSettings) {
    yamc::gpu::run_on_gpu(model, settings).unwrap();
}

#[cfg(not(feature = "gpu"))]
fn run_gpu(_: &mut Model, _: &TransportSettings) {
    unreachable!("GPU runs need --features gpu");
}

fn main() {
    let gpu = cfg!(feature = "gpu");
    println!(
        "Cell-dense win: N concentric shells (one material), 14.06 MeV point \
         source, {CPU_PARTICLES} CPU particles{}, best of {REPEATS}.\n\
         Surface pays a boundary step per shell crossing; Woodcock skips them \
         (zero fictitious collisions, single material).",
        if gpu {
            format!(", {} GPU particles", gpu_particles())
        } else {
            String::new()
        }
    );
    for case in CASES {
        println!("\n{}", case.label);
        if gpu {
            println!(
                "{:>7} | {:>12} | {:>12} | {:>12} | {:>8} | {:>9}",
                "shells", "gpu surface", "cpu surface", "cpu woodcock", "gpu 1/N", "wood/surf"
            );
        } else {
            println!(
                "{:>7} | {:>12} | {:>12} | {:>9}",
                "shells", "cpu surface", "cpu woodcock", "wood/surf"
            );
        }
        let mut gpu_one = 0.0;
        for &n in SHELL_COUNTS {
            let s = rate(case, n, TrackingMode::Surface, false);
            let w = rate(case, n, TrackingMode::Woodcock, false);
            if gpu {
                let g = rate(case, n, TrackingMode::Surface, true);
                if n == 1 {
                    gpu_one = g;
                }
                println!(
                    "{n:>7} | {g:>12.0} | {s:>12.0} | {w:>12.0} | {:>7.2}x | {:>8.2}x",
                    gpu_one / g,
                    w / s
                );
            } else {
                println!("{n:>7} | {s:>12.0} | {w:>12.0} | {:>8.2}x", w / s);
            }
        }
    }
}
