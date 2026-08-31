//! Where Woodcock *wins*: a cell-dense, single-material geometry.
//!
//! N concentric Li6 shells of the SAME material fill one ball, swept over
//! shell count. Surface tracking pays one `closest_boundary` at every
//! shell crossing, so its rate falls as the shells multiply; Woodcock
//! samples flights against the (single-material) majorant and crosses all
//! those shells with no boundary computation and zero fictitious
//! collisions (Σ_local = Σ_global here), so its rate barely moves. This
//! is the geometry class -- many cells per mean free path -- where
//! Woodcock beats surface tracking.
//!
//! This is exactly the void-free, dense case `TrackingMode::Woodcock`
//! (pure delta tracking) is meant for, so that is what it runs. Hybrid
//! would give the identical result here: a same-material cell has
//! `(Σ_global - Σ_local)·chord = 0` expected fictitious collisions, so the
//! hybrid fallback never fires and it stays on the Woodcock path anyway.
//!
//! Run via:
//!
//!     cargo run --release --example perf_woodcock_cell_dense
//!
//! Measured (50k particles, this machine) -- woodcock/surface grows with
//! shell count as surface's per-crossing boundary cost piles up:
//!
//!   shells   surface pps   woodcock pps   wood/surf
//!      1       5_730_687     5_106_703       0.89x
//!      4       5_934_122     5_238_790       0.88x
//!     16       4_090_556     4_003_951       0.98x
//!     64       1_264_286     3_136_139       2.48x
//!    128         419_241     2_245_380       5.36x

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

const R_MAX: f64 = 100.0;
const SHELL_COUNTS: &[usize] = &[1, 4, 16, 64, 128];
const PARTICLES: usize = 50_000;

fn li6() -> Material {
    let mut m = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.46),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Li6".to_string(), "tests/Li6.arrow".to_string())]),
        None,
    )
    .unwrap();
    m
}

/// N concentric Li6 shells filling a ball of radius R_MAX, vacuum outside.
fn build_model(n_shells: usize, mode: TrackingMode) -> Model {
    let surfaces: Vec<Arc<Surface>> = (1..=n_shells)
        .map(|i| {
            Arc::new(Surface {
                surface_id: Some(i),
                kind: SurfaceKind::Sphere {
                    x0: 0.0,
                    y0: 0.0,
                    z0: 0.0,
                    radius: R_MAX * (i as f64) / (n_shells as f64),
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
            expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
                surfaces[0].clone(),
            )))),
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
                    Box::new(RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                        HalfspaceType::Above(surfaces[i].clone()),
                    )))),
                ),
            },
            Some(format!("shell_{i}")),
            Some(0),
        ));
    }

    let geometry = Geometry::new(cells, vec![Arc::new(li6())]).unwrap();
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
    // Flux over the whole ball (cell 1 = innermost; a single-cell filter
    // keeps the tally cheap and identical across shell counts).
    let mut tally = Tally::new();
    tally.estimator = yamc_tallies::Estimator::Collision;
    tally.filters = vec![Filter::Cell(CellFilter::from_id(1))];
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.name = Some("flux".to_string());

    let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = mode;
    model.max_lost_particles = usize::MAX;
    model
}

fn run(n_shells: usize, mode: TrackingMode) -> f64 {
    let mut model = build_model(n_shells, mode);
    let t0 = Instant::now();
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(PARTICLES),
            seed: 42,
            ..Default::default()
        })
        .unwrap();
    PARTICLES as f64 / t0.elapsed().as_secs_f64()
}

fn main() {
    println!(
        "Cell-dense win: N concentric Li6 shells (one material) in a \
         {R_MAX} cm ball, 14.06 MeV point source, {PARTICLES} particles.\n\
         Surface pays a boundary find per shell crossing; Woodcock skips \
         them (zero fictitious collisions, single material).\n"
    );
    println!(
        "{:>8} | {:>14} | {:>14} | {:>10}",
        "shells", "surface pps", "woodcock pps", "wood/surf"
    );
    println!("{}", "-".repeat(56));
    for &n in SHELL_COUNTS {
        let s = run(n, TrackingMode::Surface);
        let w = run(n, TrackingMode::Woodcock);
        println!("{n:>8} | {s:>14.0} | {w:>14.0} | {:>9.2}x", w / s);
    }
}
