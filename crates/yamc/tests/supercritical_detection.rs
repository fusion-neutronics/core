//! A supercritical geometry handed to a fixed-source solver.
//!
//! yamc is fixed-source only. Before this guard, a supercritical model was
//! never noticed: the fission chain multiplied per history, the per-history
//! bank (an unbounded `Vec`) grew with it, and the run simply never finished --
//! no error, no warning, no diagnostic. A bare 30 cm U235 sphere burned 5.5
//! CPU-hours over 42 minutes on what should have been seconds of work.
//!
//! Two ceilings turn that into a fast, named error: a live-bank ceiling checked
//! at fission events (`FISSION_BANK_LIMIT_PER_HISTORY`) for a chain that
//! explodes, and the per-history drain ceiling in `model.rs` -- which used to
//! truncate silently -- for one that merely fails to die out. Both have to fire
//! WITHOUT firing on a legitimately subcritical model, so both directions are
//! asserted here. Bare U235 spheres at 19.1 g/cm3 (critical radius ~8.4 cm)
//! split exactly where they should: 5, 7 and 8 cm run to completion, 8.5 cm and
//! up are refused within a second or two.
//!
//! Data: endf-b8.1 Arrow tables in the fixture cache. Self-skips when absent.
//!
//! Run: `cargo test --release --test supercritical_detection`

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
use yamc::variance_reduction::{VarianceReduction, WeightWindowBounds};
use yamc_materials::Material;
use yamc_particle::ParticleType;
use yamc_physics::util::bank::WeightWindowCounts;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::mesh::RegularRectangularMesh;

/// The issue's reproducer: bare U235 metal at 19.1 g/cm3, 14.06 MeV point
/// source at the centre, vacuum boundary. The bare-sphere critical radius at
/// this density is about 8.4 cm.
const DENSITY: f64 = 19.1;
const E_SOURCE: f64 = 14.06e6;

fn u235_path() -> String {
    yamc_test_cache::nuclide_path("U235")
}

/// Bare U235 sphere of radius `r` cm, vacuum boundary.
fn bare_sphere(r: f64) -> Geometry {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: r,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));

    let mut material = Material::new(
        HashMap::from([("U235".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(DENSITY),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_name("bare u235".to_string());
    material.set_temperature("294");
    material
        .read_nuclear_data(&HashMap::from([("U235".to_string(), u235_path())]), None)
        .unwrap();

    let cell = Cell::new(Some(1), region, Some("fissile".into()), Some(0));
    Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap()
}

fn source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![E_SOURCE], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

/// Transport `n` histories through a bare sphere of radius `r`, returning the
/// panic message if the run aborts.
fn run(r: f64, n: usize) -> Result<(), String> {
    run_windowed(r, n, None).map(|_| ())
}

/// One weight window over the whole sphere with lower bound `lower` (upper
/// five times that): far below the source weight, every history splits until
/// its copies reach the window.
fn windows(r: f64, lower: f64) -> VarianceReduction {
    VarianceReduction::WeightWindowBounds(Box::new(WeightWindowBounds {
        mesh: RegularRectangularMesh::new([-r; 3], [r; 3], [1, 1, 1]),
        particle: ParticleType::Neutron,
        energy_bins: None,
        lower_bounds: vec![lower],
        upper_bounds: vec![5.0 * lower],
        survival_factor: 3.0,
        max_split: 10,
        weight_floor: 1e-38,
    }))
}

/// [`run`], with the sphere under splitting windows when `lower` is given,
/// returning what the windows did.
fn run_windowed(
    r: f64,
    n: usize,
    lower: Option<f64>,
) -> Result<Option<WeightWindowCounts>, String> {
    let mut model = Model::new(bare_sphere(r), vec![source()], vec![]);
    model.verbose = Verbose::silent();
    model.variance_reduction = lower.map(|l| windows(r, l)).into_iter().collect();
    let settings = TransportSettings {
        total_particles: Some(n),
        seed: 1234,
        threads: Some(1),
        ..Default::default()
    };

    // The budget fires from inside the rayon transport loop; swallow the
    // default hook so an expected abort does not spray a backtrace over the
    // test output.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model
            .simulate_transport(&settings)
            .map_err(|e| e.to_string())
    }));
    std::panic::set_hook(hook);

    match outcome {
        Ok(result) => result.map(|_| model.last_weight_window_counts),
        Err(payload) => Err(payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "<non-string panic>".to_string())),
    }
}

fn data_present() -> bool {
    if std::path::Path::new(&u235_path()).exists() {
        return true;
    }
    eprintln!("skipping: {} absent", u235_path());
    false
}

/// A supercritical geometry is reported, quickly, and the message says what is
/// actually wrong: the chain multiplies and yamc is fixed-source only.
#[test]
fn a_supercritical_sphere_is_reported_not_transported_forever() {
    if !data_present() {
        return;
    }
    let start = std::time::Instant::now();
    // 30 cm: the radius from the issue, ~3.5x the bare critical radius. 100
    // histories there did not complete at all before the guard.
    let err = run(30.0, 100).expect_err("a 30 cm bare U235 sphere must be refused");
    let elapsed = start.elapsed();

    assert!(
        err.contains("appears to be supercritical"),
        "the diagnosis must name the actual user error, got: {err}"
    );
    assert!(
        err.contains("fixed-source"),
        "the message must say yamc is fixed-source only, got: {err}"
    );
    assert!(
        err.contains("bare u235"),
        "the message must name the material, got: {err}"
    );
    // The point of the budget is that it fires within one history rather than
    // after minutes; the pre-fix run of this model never terminated.
    assert!(
        elapsed.as_secs() < 60,
        "the runaway must be caught promptly, took {elapsed:?}"
    );
}

/// A barely-supercritical sphere grows its chain too slowly to reach the live-bank
/// ceiling, so it lands on the drain ceiling in `model.rs` instead. That used to
/// be a silent `break` that discarded every still-queued particle: at 10 cm it
/// truncated 42% of histories and biased the tallies low with no diagnostic at
/// all. It must abort, and say why.
#[test]
fn a_barely_supercritical_sphere_is_not_silently_truncated() {
    if !data_present() {
        return;
    }
    let err = run(10.0, 300).expect_err("a 10 cm bare U235 sphere must be refused");
    assert!(
        err.contains("did not drain"),
        "the truncation must be reported, got: {err}"
    );
    assert!(
        err.contains("at or above critical") && err.contains("fixed-source"),
        "the message must diagnose the multiplying chain, got: {err}"
    );
}

/// The guard must not fire on a legitimately subcritical multiplying model: a
/// 5 cm bare U235 sphere is well under the ~8.4 cm critical radius, and every
/// one of its chains dies out on its own.
#[test]
fn a_subcritical_sphere_runs_untouched() {
    if !data_present() {
        return;
    }
    run(5.0, 2_000).expect("a subcritical multiplying sphere must transport normally");
}

/// Near-critical margin check, and the reason the ceiling is on the LIVE bank
/// rather than on the chain's total progeny: at 8 cm (k ~ 0.99) a single history
/// already produces up to ~20,000 fission neutrons in 20,000 histories, but its
/// peak bank stays around 200. The model is subcritical, every chain dies out on
/// its own, and the run must not be refused.
#[test]
fn a_just_subcritical_sphere_runs_untouched() {
    if !data_present() {
        return;
    }
    run(8.0, 20_000).expect("a just-subcritical sphere must transport normally");
}

/// Splitting windows must not make a subcritical sphere look critical (#36).
/// Each split copy starts a fission chain of its own, so a history of the 8 cm
/// sphere under windows far below the source weight leads tens of thousands of
/// particles. A fixed ceiling of 50,000 per history refused it as "at or above
/// critical"; the ceiling is per lineage now.
#[test]
fn a_just_subcritical_sphere_under_splitting_windows_runs() {
    if !data_present() {
        return;
    }
    // About 7,000 split copies a history. One history is enough, as long as it
    // leads more particles than the fixed ceiling allowed.
    let c = run_windowed(8.0, 1, Some(6.7e-5))
        .expect("splitting windows must not refuse a subcritical sphere")
        .expect("a windowed run reports its counts");
    assert!(
        c.max_particles_per_history > 50_000,
        "the fixed ceiling would not have fired: {c:?}"
    );
    // Windows so low that the history exhausts its split budget.
    let c = run_windowed(8.0, 1, Some(1e-7))
        .expect("a spent split budget must not refuse a subcritical sphere")
        .expect("a windowed run reports its counts");
    assert!(
        c.histories_at_split_budget == 1,
        "the split budget should be spent: {c:?}"
    );
}

/// The per-lineage ceiling must not let a supercritical sphere through: under
/// the same splitting windows the 10 cm sphere is still refused, and promptly.
/// The fission-bank guard refuses it within the history whose chain runs
/// away, while the per-lineage ceiling would allow 50,000 particles per split
/// copy first, so the message must be the guard's.
#[test]
fn a_barely_supercritical_sphere_under_splitting_windows_is_refused() {
    if !data_present() {
        return;
    }
    let err = run_windowed(10.0, 300, Some(6.7e-5))
        .expect_err("a 10 cm bare U235 sphere must be refused under windows too");
    assert!(
        err.contains("Fission chain did not terminate") && err.contains("fixed-source"),
        "the fission-bank guard must refuse the runaway chain, got: {err}"
    );
}
