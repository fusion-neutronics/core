//! Surface vs pure `Woodcock` vs `Hybrid` through a large void next to a
//! dense material -- the canonical delta-tracking pathology and the case
//! `TrackingMode::Hybrid` exists to fix.
//!
//! Geometry: a large vacuum cavity (radius `R_VOID`, no material) wrapped
//! in a dense Pb208 shell (lead standing in for tungsten -- there is no
//! tungsten fixture in `tests/`). A 14.06 MeV point source sits at the
//! centre of the cavity.
//!
//! Why PURE Woodcock is catastrophic here: the global majorant is the
//! Pb208 total cross section (mean free path ~ a few cm), so a neutron
//! streaming across the metres-wide vacuum samples a flight every few cm
//! and rejects it as a *fictitious* collision -- hundreds of useless
//! delta-steps to cross a region surface tracking crosses in ONE step.
//! So `TrackingMode::Woodcock` (pure) collapses as the void grows: this
//! is the geometry class where you must NOT use it.
//!
//! `TrackingMode::Hybrid` surface-steps the void cell (one step to the
//! boundary) and only delta-tracks where there is material, so its rate
//! stays flat in void radius -- the pathology is gone. (Hybrid is not
//! *faster* than surface on this 2-region geometry; the Woodcock win is
//! in geometrically complex, many-cells-per-mfp models -- see
//! perf_woodcock_cell_dense. The point here is that Hybrid is not broken
//! by voids while pure Woodcock is.)
//!
//! Run via:
//!
//!     cargo run --release --example perf_woodcock_void
//!
//! Measured (50k particles, this machine). The tell-tale is the trend:
//! surface and hybrid pps are flat in void radius; pure Woodcock collapses.
//!
//!   void r/cm   surface pps   woodcock(pure)   wood/surf   hybrid pps   hyb/surf
//!      100        389910          307213          0.79x      331715       0.85x
//!      300        332220          214574          0.65x      282683       0.85x
//!     1000        305560          120101          0.39x      262225       0.85x
//!     3000        289050           56549          0.20x      252631       0.86x
//!
//! Pure Woodcock's ratio collapses with void radius (the pathology);
//! hybrid's is constant. The residual ~0.85x for hybrid is Woodcock's
//! constant per-step overhead in the shell (a 2-region model gives no
//! boundary-skipping win; that needs many cells per mean free path).
//! Shell flux matches surface to <1% for both modes throughout.
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

// Sweep the vacuum cavity radius. Surface tracking crosses the void in
// ONE step regardless of size, so its rate is flat; pure Woodcock takes
// ~R_VOID / mfp_majorant fictitious delta-steps to cross, so its rate
// collapses as the void grows -- that divergence is the pathology.
const VOID_RADII: &[f64] = &[100.0, 300.0, 1000.0, 3000.0];
const SHELL_THICK: f64 = 40.0; // Pb208 shell thickness (cm)
const PARTICLES: usize = 50_000;

fn sphere(id: usize, radius: f64, boundary: BoundaryType) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: Some(id),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary,
        name: None,
    })
}

/// Build the cavity+shell model. The inner cell is vacuum (no material);
/// the outer shell is Pb208 with a vacuum outer boundary.
fn build_model(mode: TrackingMode, r_void: f64) -> Model {
    let inner = sphere(1, r_void, BoundaryType::Transmission);
    let outer = sphere(2, r_void + SHELL_THICK, BoundaryType::Vacuum);

    // Cavity: inside the inner sphere, no material.
    let cavity = Cell::new(
        Some(1),
        Region {
            expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
                inner.clone(),
            )))),
        },
        Some("cavity".to_string()),
        None, // void -- no material
    );

    // Shell: between the two spheres, Pb208.
    let mut pb = Material::new(
        HashMap::from([("Pb208".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(11.34),
    )
    .unwrap();
    pb.set_material_id(1);
    pb.set_temperature("294");
    pb.read_nuclear_data(
        &HashMap::from([("Pb208".to_string(), "tests/Pb208.arrow".to_string())]),
        None,
    )
    .unwrap();

    let shell = Cell::new(
        Some(2),
        Region {
            expr: RegionExpr::Intersection(
                Box::new(RegionExpr::Halfspace(HalfspaceType::Above(inner.clone()))),
                Box::new(RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                    HalfspaceType::Above(outer.clone()),
                )))),
            ),
        },
        Some("shell".to_string()),
        Some(0), // Pb208 is materials[0]
    );

    let geometry = Geometry::new(vec![cavity, shell], vec![Arc::new(pb)]).unwrap();

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

    // Collision-estimator flux in the shell (bias-free under Woodcock).
    let mut tally = Tally::new();
    tally.estimator = yamc_tallies::Estimator::Collision;
    tally.filters = vec![Filter::Cell(CellFilter::from_id(2))];
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.name = Some("shell_flux".to_string());

    let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = mode;
    model.max_lost_particles = usize::MAX;
    model
}

/// Returns (transport pps, shell flux mean).
fn run(mode: TrackingMode, r_void: f64) -> (f64, f64) {
    let mut model = build_model(mode, r_void);
    let t0 = Instant::now();
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(PARTICLES),
            seed: 42,
            ..Default::default()
        })
        .unwrap();
    let pps = PARTICLES as f64 / t0.elapsed().as_secs_f64();
    (pps, model.tallies[0].total_mean())
}

fn main() {
    println!(
        "Void+dense pathology: vacuum cavity (swept radius) + Pb208 shell \
         {SHELL_THICK} cm, 14.06 MeV point source, {PARTICLES} particles.\n\
         Surface and Hybrid cross the void in one step (flat rate); pure \
         Woodcock takes ~R/mfp fictitious delta-steps (rate collapses).\n"
    );
    println!(
        "{:>10} | {:>13} | {:>14} | {:>9} | {:>12} | {:>8} | {:>9}",
        "void r/cm",
        "surface pps",
        "woodcock(pure)",
        "wood/surf",
        "hybrid pps",
        "hyb/surf",
        "max dflux"
    );
    println!("{}", "-".repeat(92));
    for &r in VOID_RADII {
        let (s_pps, s_flux) = run(TrackingMode::Surface, r);
        let (w_pps, w_flux) = run(TrackingMode::Woodcock, r);
        let (h_pps, h_flux) = run(TrackingMode::Hybrid, r);
        let rel = (s_flux - w_flux).abs().max((s_flux - h_flux).abs()) / s_flux.max(1e-30);
        println!(
            "{:>10.0} | {:>13.0} | {:>14.0} | {:>8.2}x | {:>12.0} | {:>7.2}x | {:>8.2}%",
            r,
            s_pps,
            w_pps,
            w_pps / s_pps,
            h_pps,
            h_pps / s_pps,
            rel * 100.0
        );
    }
}
