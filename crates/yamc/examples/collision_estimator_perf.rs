//! Track-length vs collision estimator mesh-tally benchmark.
//!
//! Uses a TBR-like torus shell + ring source,
//! sweeps across (128³, 256³) × (TrackLength, Collision). The
//! collision estimator scores at most one mesh bin per collision --
//! versus track-length, which scores at every cell crossing the
//! particle takes through the mesh.
//!
//! Reading the output:
//!
//! - `total_mean` differs between estimators because the **track-length
//!   tally integrates flux over void voxels** (the inner and outer
//!   torus voids, where particles travel but don't collide), while
//!   the **collision tally only sees material voxels** (the shell).
//!   On a per-voxel basis inside the shell, the two converge to the
//!   same flux.
//! - `speedup` is wall-time `t_TL / t_CO` -- collision wins here
//!   because each history touches fewer bins (no contributions in
//!   void voxels, one bin per collision rather than one per crossing).
//! - `fom_ratio` is `FOM_CO / FOM_TL`. In an optically *thin*
//!   geometry like this Be/Li shell at 14 MeV, each particle has
//!   far fewer collisions than crossings, so collision-estimator
//!   per-history variance is higher -- the FOM payoff can be < 1
//!   even when wall time improves. Optically *thick* shielding
//!   (large Σ_t × thickness) flips the trade-off.
//!
//! Run via:
//!
//!     cargo run --release --example collision_estimator_perf

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::{Discrete, Uniform};
use yamc_source::distribution::spatial::{CylindricalRing, Univariate};
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::mesh::MeshFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::Estimator;

const MAJOR_RADIUS: f64 = 200.0;
const INNER_MINOR: f64 = 40.0;
const OUTER_MINOR: f64 = 80.0;
const BOUNDING_RADIUS: f64 = 500.0;

const MESH_DIMS: &[usize] = &[128, 256];
const PARTICLES_PER_BATCH: usize = 100_000;
const BATCHES: usize = 4;

fn build_geometry() -> Geometry {
    let outer_torus = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::ZTorus {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            a: MAJOR_RADIUS,
            b: OUTER_MINOR,
            c: OUTER_MINOR,
        },
        boundary: BoundaryType::Transmission,
        name: None,
    });
    let inner_torus = Arc::new(Surface {
        surface_id: Some(2),
        kind: SurfaceKind::ZTorus {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            a: MAJOR_RADIUS,
            b: INNER_MINOR,
            c: INNER_MINOR,
        },
        boundary: BoundaryType::Transmission,
        name: None,
    });
    let bounding_sphere = Arc::new(Surface {
        surface_id: Some(3),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: BOUNDING_RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });

    let inner_void = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            inner_torus.clone(),
        )))),
    };
    let shell = Region {
        expr: RegionExpr::Intersection(
            Box::new(RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                HalfspaceType::Above(outer_torus.clone()),
            )))),
            Box::new(RegionExpr::Halfspace(HalfspaceType::Above(inner_torus))),
        ),
    };
    let outer_void = Region {
        expr: RegionExpr::Intersection(
            Box::new(RegionExpr::Halfspace(HalfspaceType::Above(outer_torus))),
            Box::new(RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                HalfspaceType::Above(bounding_sphere),
            )))),
        ),
    };

    let mut mat = Material::new(
        HashMap::from([
            ("Li6".into(), 0.07 / 2.0),
            ("Li7".into(), 0.93 / 2.0),
            ("Be9".into(), 0.5),
        ]),
        "atom",
        "g/cm3",
        Some(2.0),
    )
    .unwrap();
    mat.set_material_id(1);
    mat.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Be9".to_string(), "tests/Be9.arrow".to_string());
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    nuclide_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
    mat.read_nuclear_data(&nuclide_map, None).unwrap();
    let mat_arc = Arc::new(mat);

    let cell_void_in = Cell::new(Some(1), inner_void, Some("inner_void".into()), None);
    let cell_shell = Cell::new(Some(2), shell, Some("shell".into()), Some(0));
    let cell_void_out = Cell::new(Some(3), outer_void, Some("outer_void".into()), None);

    Geometry::new(vec![cell_void_in, cell_shell, cell_void_out], vec![mat_arc]).unwrap()
}

fn build_source() -> ParticleSource {
    let ring = CylindricalRing::new(
        Univariate::Discrete(Discrete::new(vec![MAJOR_RADIUS], vec![1.0]).unwrap()),
        Univariate::Uniform(Uniform::new(0.0, std::f64::consts::TAU).unwrap()),
        Univariate::Discrete(Discrete::new(vec![0.0], vec![1.0]).unwrap()),
        [0.0, 0.0, 0.0],
    );
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::CylindricalRing(Box::new(ring)),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn build_tally(mesh_dim: usize, estimator: Estimator) -> Arc<Tally> {
    let lower_left = [-300.0, -300.0, -100.0];
    let upper_right = [300.0, 300.0, 100.0];
    let shape = [mesh_dim, mesh_dim, mesh_dim];
    let mesh = RegularRectangularMesh::new(lower_left, upper_right, shape);
    let mesh_filter = MeshFilter::new(mesh);
    let mut tally = Tally::new();
    tally.filters.push(Filter::Mesh(mesh_filter));
    tally.set_scores_mixed(vec![Score::Flux(FluxScore)]);
    tally.estimator = estimator;
    tally.name = Some(format!("flux_{estimator}"));
    Arc::new(tally)
}

#[allow(dead_code)]
struct RunStats {
    estimator: Estimator,
    mesh_dim: usize,
    elapsed_secs: f64,
    particles: u64,
    total_mean: f64,
    aggregate_fom: f64,
    nonzero_bins: usize,
}

fn run_one(mesh_dim: usize, estimator: Estimator) -> RunStats {
    let geometry = build_geometry();
    let source = build_source();
    let tally = build_tally(mesh_dim, estimator);

    let mut model = Model::new(geometry, vec![source], vec![tally]);

    let start = Instant::now();
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(PARTICLES_PER_BATCH * BATCHES),
            seed: 0xC0FFEE,
            ..Default::default()
        })
        .unwrap();
    let elapsed = start.elapsed().as_secs_f64();

    let result = model.tallies[0].finalize().with_fom(elapsed);
    let mean = result.mean.clone();
    let rel_err = result.relative_error.clone();
    let total_mean: f64 = mean.iter().sum();
    let mut weight_total = 0.0_f64;
    let mut weighted_rel_var = 0.0_f64;
    let mut nonzero = 0_usize;
    for (m, r) in mean.iter().zip(rel_err.iter()) {
        if *m > 0.0 && *r > 0.0 && r.is_finite() {
            weight_total += *m;
            weighted_rel_var += m * (r * r);
            nonzero += 1;
        }
    }
    let agg_rel_err = if weight_total > 0.0 {
        (weighted_rel_var / weight_total).sqrt()
    } else {
        f64::INFINITY
    };
    let aggregate_fom = if elapsed > 0.0 && agg_rel_err.is_finite() && agg_rel_err > 0.0 {
        1.0 / (agg_rel_err * agg_rel_err * elapsed)
    } else {
        0.0
    };
    RunStats {
        estimator,
        mesh_dim,
        elapsed_secs: elapsed,
        particles: (PARTICLES_PER_BATCH * BATCHES) as u64,
        total_mean,
        aggregate_fom,
        nonzero_bins: nonzero,
    }
}

fn main() {
    println!("=== Track-length vs Collision estimator mesh-tally benchmark ===");
    println!("torus shell: a={MAJOR_RADIUS}, inner_minor={INNER_MINOR}, outer_minor={OUTER_MINOR}");
    println!(
        "particles: {} batches x {} per batch = {}",
        BATCHES,
        PARTICLES_PER_BATCH,
        BATCHES * PARTICLES_PER_BATCH,
    );
    println!();

    // Warm-up at smallest size with the default estimator.
    let _warm = run_one(MESH_DIMS[0], Estimator::TrackLength);

    println!(
        "{:>4}  {:>13}  {:>9}  {:>10}  {:>9}  {:>10}",
        "dim", "estimator", "elapsed_s", "particle/s", "agg_FOM", "fom_ratio",
    );

    for &mesh_dim in MESH_DIMS {
        // Baseline: track-length. Everything is reported relative to this
        // as the FOM-ratio reference.
        let baseline = run_one(mesh_dim, Estimator::TrackLength);
        let baseline_fom = baseline.aggregate_fom;

        let runs = [baseline, run_one(mesh_dim, Estimator::Collision)];

        for r in &runs {
            let pps = r.particles as f64 / r.elapsed_secs;
            let fom_ratio = if baseline_fom > 0.0 {
                r.aggregate_fom / baseline_fom
            } else {
                0.0
            };
            let est_str = format!("{}", r.estimator);
            println!(
                "{:>4}  {:>13}  {:>9.3}  {:>10.0}  {:>9.3e}  {:>9.3}x",
                mesh_dim, est_str, r.elapsed_secs, pps, r.aggregate_fom, fom_ratio,
            );
        }
    }
}
