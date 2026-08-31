//! True track-length mesh tallies under delta tracking (issue #350).
//!
//! Under `TrackingMode::Woodcock` / `Hybrid`, a flux-score mesh tally
//! with `Estimator::TrackLength` is scored along each majorant flight
//! segment with the mesh DDA (the same iterator surface tracking uses),
//! instead of the collision-density equivalent at delta-collisions.
//!
//! Coverage:
//! - Woodcock mesh flux matches Surface per converged bin and in total,
//!   including leaking flights (mesh extends past the vacuum boundary,
//!   so the exit-clipped partial segment is exercised).
//! - Mesh bins wholly outside the geometry stay exactly zero (a naive
//!   unclipped implementation would deposit phantom track length there).
//! - The track-length estimator's variance beats the collision
//!   estimator's on the same mesh (the point of the feature).
//! - A mesh tally's presence leaves other tallies bit-identical
//!   (the DDA consumes no RNG and perturbs no particle state).
//! - Ineligible mesh tallies (XS-weighted score, or a cell filter)
//!   keep the collision-density path and still match Surface.
//! - Hybrid mode agrees too (surface steps and delta steps score the
//!   same mesh through different paths; no double counting).
//! - The photon flight path scores the same way (Fe sphere).

use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::mesh::MeshFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::tally::{FluxScore, Mt, ReactionRateScore, Score, Tally};
use yamc_tallies::CellFilter;

/// Inner Li6 sphere (r=5) + outer Be9 shell (r=10), vacuum beyond.
/// Two materials so Woodcock takes real rejection branches, and a
/// boundary at r=10 inside the mesh below so flights leak mid-mesh.
fn build_two_material_geometry() -> Geometry {
    let inner = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 5.0,
        },
        boundary: BoundaryType::Transmission,
        name: None,
    });
    let outer = Arc::new(Surface {
        surface_id: Some(2),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let inner_region = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            inner.clone(),
        )))),
    };
    let outer_region = Region {
        expr: RegionExpr::Intersection(
            Box::new(RegionExpr::Halfspace(HalfspaceType::Above(inner.clone()))),
            Box::new(RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                HalfspaceType::Above(outer.clone()),
            )))),
        ),
    };

    let mut mat_li6 = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.534),
    )
    .unwrap();
    mat_li6.set_material_id(1);
    mat_li6.set_temperature("294");
    let nuclide_map = HashMap::from([("Li6".to_string(), "tests/Li6.arrow".to_string())]);
    mat_li6.read_nuclear_data(&nuclide_map, None).unwrap();

    let mut mat_be9 = Material::new(
        HashMap::from([("Be9".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.85),
    )
    .unwrap();
    mat_be9.set_material_id(2);
    mat_be9.set_temperature("294");
    let nuclide_map = HashMap::from([("Be9".to_string(), "tests/Be9.arrow".to_string())]);
    mat_be9.read_nuclear_data(&nuclide_map, None).unwrap();

    let cell_inner = Cell::new(
        Some(1),
        inner_region,
        Some("inner_li6".to_string()),
        Some(0),
    );
    let cell_outer = Cell::new(
        Some(2),
        outer_region,
        Some("outer_be9".to_string()),
        Some(1),
    );
    Geometry::new(
        vec![cell_inner, cell_outer],
        vec![Arc::new(mat_li6), Arc::new(mat_be9)],
    )
    .unwrap()
}

fn make_source_14mev() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

/// 4x4x4 mesh over [-12,12]^3: extends 2 cm past the vacuum boundary,
/// so leaking flights must be clipped at the geometry exit. The eight
/// corner bins (|x|,|y|,|z| in [6,12]) lie wholly outside the r=10
/// sphere (closest point at radius sqrt(108) > 10).
fn make_mesh() -> RegularRectangularMesh {
    RegularRectangularMesh::new([-12.0, -12.0, -12.0], [12.0, 12.0, 12.0], [4, 4, 4])
}

fn make_mesh_flux_tally(estimator: yamc_tallies::Estimator, name: &str) -> Tally {
    let mut tally = Tally::new();
    tally.estimator = estimator;
    tally.filters = vec![Filter::Mesh(MeshFilter::new(make_mesh()))];
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.name = Some(name.to_string());
    tally
}

/// Bin indices of the eight corner voxels that are wholly outside the
/// r=10 sphere, resolved through the same lookup the tally uses.
fn outside_corner_bins() -> Vec<usize> {
    let filter = MeshFilter::new(make_mesh());
    let mut bins = Vec::new();
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                bins.push(
                    filter
                        .get_bin([9.0 * sx, 9.0 * sy, 9.0 * sz])
                        .expect("corner bin centre must be inside the mesh"),
                );
            }
        }
    }
    bins
}

fn run_mesh_flux(
    mode: TrackingMode,
    particles: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let geometry = build_two_material_geometry();
    let tally = make_mesh_flux_tally(yamc_tallies::Estimator::TrackLength, "mesh_flux");
    let mut model = Model::new(geometry, vec![make_source_14mev()], vec![Arc::new(tally)]);
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = mode;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles),
            seed,
            ..Default::default()
        })
        .unwrap();
    let t = &model.tallies[0];
    (t.get_mean(), t.get_std_dev(), t.total_mean(), t.total_std())
}

/// Surface and `mode` must agree on the mesh's TOTAL flux.
///
/// The uncertainty comes from the per-bin sigmas, NOT from `Tally::total_std()`.
/// `total_std` is `sqrt(sum of the per-bin variances)`, which assumes the bins
/// are independent; on a flux mesh tally one history lays track length down in
/// many bins at once, so they are strongly positively correlated and that form
/// under-estimates the total's spread. The rigorous bound on the standard
/// deviation of a sum of correlated terms is the SUM of their standard
/// deviations (Cauchy-Schwarz, attained at perfect correlation), which is what
/// this uses.
///
/// Recalibrated once, when issue #111 gave every in-history secondary its own
/// stream and so re-rolled which realisation this fixed seed lands on. The bound
/// was `3 * total_std_s`: the Surface run's quadrature sigma ALONE, for a
/// difference between two independent estimates whose variance is the sum of
/// both. Measured over 4 seeds at 20k and again at 200k histories, the
/// difference normalised by the combined quadrature sigma sits at 0.1 to 3.6
/// and does NOT shrink with history count, which is the signature of a
/// mis-specified sigma rather than of a fluctuation: 2 of those 4 seeds
/// exceeded the old bound. The means themselves are unmoved (Surface total
/// 17.4345 after vs 17.4280 before, averaged over the same 4 seeds at 200k, each
/// run carrying a per-history batch sigma of 0.026, so 0.4 sigma apart).
///
/// The bound this leaves is deliberately loose: it catches a gross
/// normalisation error, and the STRICT statement lives in the per-bin
/// comparison below, where the per-bin sigma is a valid per-history error and
/// the two estimators are compared at 5 sigma of their combined uncertainty.
fn assert_totals_agree(
    mode: TrackingMode,
    total_s: f64,
    total_w: f64,
    std_s: &[f64],
    std_w: &[f64],
) {
    let corr_sum = |s: &[f64]| -> f64 { s.iter().sum() };
    let sigma_diff = (corr_sum(std_s).powi(2) + corr_sum(std_w).powi(2)).sqrt();
    let total_diff = (total_s - total_w).abs();
    let total_tol = 3.0 * sigma_diff;
    assert!(
        total_diff < total_tol,
        "{mode:?} mesh flux total {total_w:.6e} differs from Surface {total_s:.6e} \
         by {total_diff:.2e}, exceeding 3 sigma {total_tol:.2e}",
    );
}

/// Per-bin and total agreement against surface tracking, plus the
/// exit-clip property: bins wholly outside the geometry stay zero.
fn assert_mesh_matches_surface(mode: TrackingMode) {
    let particles = 20_000;
    let (mean_s, std_s, total_s, _) = run_mesh_flux(TrackingMode::Surface, particles, 42);
    let (mean_w, std_w, total_w, _) = run_mesh_flux(mode, particles, 42);

    assert!(
        total_s > 0.0,
        "Surface mesh flux total was zero -- rig broken"
    );
    assert_totals_agree(mode, total_s, total_w, &std_s, &std_w);

    // Wholly-outside corner voxels: exactly zero under both modes. A
    // naive implementation scoring the full unclipped leak segment
    // would deposit phantom track length here.
    for bin in outside_corner_bins() {
        assert_eq!(
            mean_s[bin], 0.0,
            "Surface scored outside-geometry bin {bin}: rig assumption broken"
        );
        assert_eq!(
            mean_w[bin], 0.0,
            "{mode:?} deposited track length in bin {bin}, which is wholly \
             outside the geometry -- leak segments are not being clipped",
        );
    }

    // Strict per-bin agreement on well-converged bins. 5 sigma of the
    // combined uncertainty: per-bin sigma estimates are only trusted
    // where Surface converged below 10% relative error.
    let mut checked = 0;
    for i in 0..mean_s.len() {
        if mean_s[i] <= 0.0 || std_s[i] / mean_s[i] > 0.10 {
            continue;
        }
        checked += 1;
        let diff = (mean_s[i] - mean_w[i]).abs();
        let tol = 5.0 * (std_s[i].powi(2) + std_w[i].powi(2)).sqrt();
        assert!(
            diff < tol,
            "bin {i}: {mode:?} {wm:.6e} vs Surface {sm:.6e}, diff {diff:.2e} \
             exceeds 5 sigma {tol:.2e}",
            wm = mean_w[i],
            sm = mean_s[i],
        );
    }
    assert!(
        checked >= 20,
        "only {checked} bins converged below 10% -- statistical check is too weak",
    );
}

#[test]
fn woodcock_mesh_flux_matches_surface() {
    assert_mesh_matches_surface(TrackingMode::Woodcock);
}

#[test]
fn hybrid_mesh_flux_matches_surface() {
    assert_mesh_matches_surface(TrackingMode::Hybrid);
}

#[test]
fn woodcock_energy_binned_mesh_flux_matches_surface() {
    // Energy filters compose with the segment path (energy is constant
    // along a delta flight, so binning the whole segment by
    // particle.energy is exact). Two groups split at 1 MeV: the 14 MeV
    // source populates the fast group directly and the slow group via
    // downscatter, so both groups must independently match surface
    // tracking.
    use yamc_tallies::filter::energy::EnergyFilter;
    let run = |mode: TrackingMode| -> (Vec<f64>, Vec<f64>) {
        let geometry = build_two_material_geometry();
        let mut tally = make_mesh_flux_tally(yamc_tallies::Estimator::TrackLength, "mesh_flux_e");
        tally
            .filters
            .push(Filter::Energy(EnergyFilter::new(vec![0.0, 1.0e6, 20.0e6])));
        let mut model = Model::new(geometry, vec![make_source_14mev()], vec![Arc::new(tally)]);
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(20_000),
                seed: 42,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.get_mean(), t.get_std_dev())
    };
    let (mean_s, std_s) = run(TrackingMode::Surface);
    let (mean_w, std_w) = run(TrackingMode::Woodcock);
    assert_eq!(mean_s.len(), 2 * 64, "expected 2 energy groups x 64 voxels");
    assert_eq!(mean_w.len(), mean_s.len());

    // Both halves of the flat vector (one energy group each) must carry
    // scores: a bug collapsing everything into one group would zero the
    // other half in both runs and silently pass a totals-only check.
    let half = mean_s.len() / 2;
    assert!(mean_s[..half].iter().any(|&v| v > 0.0));
    assert!(mean_s[half..].iter().any(|&v| v > 0.0));

    // Per-bin agreement on converged bins (per-bin sigmas are valid;
    // group-sum sigmas would need the inter-bin covariance).
    let mut checked = 0;
    for i in 0..mean_s.len() {
        if mean_s[i] <= 0.0 || std_s[i] / mean_s[i] > 0.10 {
            continue;
        }
        checked += 1;
        let diff = (mean_s[i] - mean_w[i]).abs();
        let tol = 5.0 * (std_s[i].powi(2) + std_w[i].powi(2)).sqrt();
        assert!(
            diff < tol,
            "bin {i}: Woodcock {wm:.6e} vs Surface {sm:.6e}, diff {diff:.2e} \
             exceeds 5 sigma {tol:.2e}",
            wm = mean_w[i],
            sm = mean_s[i],
        );
    }
    assert!(
        checked >= 20,
        "only {checked} energy-binned voxels converged below 10%",
    );
}

#[test]
fn woodcock_cylindrical_mesh_flux_matches_surface() {
    // Eligibility covers both MeshFilter kinds; the cylindrical DDA
    // (ray-cylinder quadratics rather than the rectangular stepper)
    // must agree with surface tracking the same way. The mesh extends
    // radially past the r=10 vacuum boundary to exercise the exit clip.
    use yamc_tallies::mesh::CylindricalMesh;
    let run = |mode: TrackingMode| -> (Vec<f64>, Vec<f64>, f64) {
        let geometry = build_two_material_geometry();
        let mesh = CylindricalMesh::uniform(
            [0.0, 0.0, 0.0],
            (0.0, 12.0),
            (0.0, std::f64::consts::TAU),
            (-12.0, 12.0),
            [6, 4, 6],
        );
        let mut tally = Tally::new();
        tally.estimator = yamc_tallies::Estimator::TrackLength;
        tally.filters = vec![Filter::Mesh(MeshFilter::new_cylindrical(mesh))];
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.name = Some("cyl_mesh_flux".to_string());
        let mut model = Model::new(geometry, vec![make_source_14mev()], vec![Arc::new(tally)]);
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(20_000),
                seed: 42,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.get_mean(), t.get_std_dev(), t.total_mean())
    };
    let (mean_s, std_s, total_s) = run(TrackingMode::Surface);
    let (mean_w, std_w, total_w) = run(TrackingMode::Woodcock);
    assert!(total_s > 0.0, "Surface cylindrical mesh flux was zero");
    assert!(total_w > 0.0, "Woodcock cylindrical mesh flux was zero");
    assert_totals_agree(TrackingMode::Woodcock, total_s, total_w, &std_s, &std_w);

    // The strict statement, matching the rectangular case: per converged
    // voxel, at 5 sigma of the two estimators' combined uncertainty. Per bin
    // the Welford sigma IS a valid per-history error (the correlation that
    // spoils it for the TOTAL is between bins), so this is where a real
    // cylindrical-DDA disagreement would show.
    let mut checked = 0;
    for i in 0..mean_s.len() {
        if mean_s[i] <= 0.0 || std_s[i] / mean_s[i] > 0.10 {
            continue;
        }
        checked += 1;
        let diff = (mean_s[i] - mean_w[i]).abs();
        let tol = 5.0 * (std_s[i].powi(2) + std_w[i].powi(2)).sqrt();
        assert!(
            diff < tol,
            "cylindrical voxel {i}: Woodcock {wm:.6e} vs Surface {sm:.6e}, diff \
             {diff:.2e} exceeds 5 sigma {tol:.2e}",
            wm = mean_w[i],
            sm = mean_s[i],
        );
    }
    assert!(
        checked >= 20,
        "only {checked} cylindrical voxels converged below 10% -- statistical \
         check is too weak",
    );
}

#[test]
fn woodcock_track_length_variance_beats_collision_estimator() {
    // The point of the feature: on the same mesh under Woodcock, the
    // true track-length estimator (every flight segment contributes to
    // every crossed bin) accumulates less variance than the collision
    // estimator (contributions only at real collision sites).
    let geometry = build_two_material_geometry();
    let tl = make_mesh_flux_tally(yamc_tallies::Estimator::TrackLength, "tl");
    let coll = make_mesh_flux_tally(yamc_tallies::Estimator::Collision, "coll");
    let mut model = Model::new(
        geometry,
        vec![make_source_14mev()],
        vec![Arc::new(tl), Arc::new(coll)],
    );
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = TrackingMode::Woodcock;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(20_000),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    let var_sum = |t: &Tally| -> f64 { t.get_std_dev().iter().map(|s| s * s).sum() };
    let var_tl = var_sum(&model.tallies[0]);
    let var_coll = var_sum(&model.tallies[1]);
    assert!(
        var_tl > 0.0,
        "track-length mesh variance was zero -- rig broken"
    );
    assert!(
        var_tl < var_coll,
        "track-length mesh variance {var_tl:.6e} is not below the collision \
         estimator's {var_coll:.6e}",
    );

    // Free statistical check: the two estimators agree on the total.
    let t0 = &model.tallies[0];
    let t1 = &model.tallies[1];
    let diff = (t0.total_mean() - t1.total_mean()).abs();
    let tol = 4.0 * (t0.total_std().powi(2) + t1.total_std().powi(2)).sqrt();
    assert!(
        diff < tol,
        "TL total {a:.6e} vs collision total {b:.6e} disagree beyond 4 sigma",
        a = t0.total_mean(),
        b = t1.total_mean(),
    );
}

#[test]
fn mesh_tally_presence_leaves_other_tallies_bit_identical() {
    // The DDA consumes no RNG and perturbs no particle state, so adding
    // a mesh tally must leave every other tally's result bit-identical
    // (fixed seed, threads=1 for a deterministic fold order).
    let run = |with_mesh: bool| -> String {
        let geometry = build_two_material_geometry();
        let mut cell_tally = Tally::new();
        cell_tally.estimator = yamc_tallies::Estimator::TrackLength;
        cell_tally.filters = vec![Filter::Cell(CellFilter::from_id(1))];
        cell_tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
            105,
        )))];
        cell_tally.name = Some("cell_rxrate".to_string());
        let mut tallies = vec![Arc::new(cell_tally)];
        if with_mesh {
            tallies.push(Arc::new(make_mesh_flux_tally(
                yamc_tallies::Estimator::TrackLength,
                "mesh_flux",
            )));
        }
        let mut model = Model::new(geometry, vec![make_source_14mev()], tallies);
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = TrackingMode::Woodcock;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(2_000),
                seed: 42,
                threads: Some(1),
                ..Default::default()
            })
            .unwrap();
        format!("{:?}", model.tallies[0].get_mean())
    };
    let without = run(false);
    let with = run(true);
    assert_eq!(
        without, with,
        "adding a mesh tally changed another tally's bits",
    );
}

#[test]
fn ineligible_mesh_tallies_still_match_surface() {
    // Two mesh tallies that must NOT take the segment path: an
    // XS-weighted score (MT 105 reaction rate), and a flux tally with a
    // cell filter. Both keep the delta-collision collision-density
    // estimator (which knows the local material/cell) and must still
    // match surface tracking. A bug that wrongly marks them eligible
    // zeroes or halves them: the segment scorer carries no cell or
    // material, so the rate response and the cell filter cannot score.
    let run = |mode: TrackingMode| -> Vec<(f64, f64)> {
        let geometry = build_two_material_geometry();
        let mut rxrate = Tally::new();
        rxrate.estimator = yamc_tallies::Estimator::TrackLength;
        rxrate.filters = vec![Filter::Mesh(MeshFilter::new(make_mesh()))];
        rxrate.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
            105,
        )))];
        rxrate.name = Some("mesh_rxrate".to_string());

        let mut cell_filtered = make_mesh_flux_tally(
            yamc_tallies::Estimator::TrackLength,
            "mesh_flux_cell_filtered",
        );
        cell_filtered
            .filters
            .push(Filter::Cell(CellFilter::from_id(2)));

        let mut model = Model::new(
            geometry,
            vec![make_source_14mev()],
            vec![Arc::new(rxrate), Arc::new(cell_filtered)],
        );
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(20_000),
                seed: 42,
                ..Default::default()
            })
            .unwrap();
        model
            .tallies
            .iter()
            .map(|t| (t.total_mean(), t.total_std()))
            .collect()
    };
    let surf = run(TrackingMode::Surface);
    let wood = run(TrackingMode::Woodcock);
    for (name, (s, w)) in ["mesh_rxrate", "mesh_flux_cell_filtered"]
        .iter()
        .zip(surf.iter().zip(wood.iter()))
    {
        assert!(s.0 > 0.0, "{name}: Surface total was zero -- rig broken");
        let diff = (s.0 - w.0).abs();
        let tol = 4.0 * (s.1.powi(2) + w.1.powi(2)).sqrt();
        assert!(
            diff < tol,
            "{name}: Woodcock {w0:.6e} vs Surface {s0:.6e}, diff {diff:.2e} \
             exceeds 4 sigma {tol:.2e}",
            w0 = w.0,
            s0 = s.0,
        );
    }
}

#[test]
#[ignore = "manual A/B harness: prints analog-woodcock repr for bit-identity check"]
fn print_analog_woodcock_repr() {
    // No mesh tally anywhere: the issue-#350 gate must leave this run
    // bit-identical across the change (fixed seed, threads=1).
    let geometry = build_two_material_geometry();
    let mut cell_tally = Tally::new();
    cell_tally.estimator = yamc_tallies::Estimator::TrackLength;
    cell_tally.filters = vec![Filter::Cell(CellFilter::from_id(1))];
    cell_tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        105,
    )))];
    let mut model = Model::new(
        geometry,
        vec![make_source_14mev()],
        vec![Arc::new(cell_tally)],
    );
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = TrackingMode::Woodcock;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(5_000),
            seed: 42,
            threads: Some(1),
            ..Default::default()
        })
        .unwrap();
    println!(
        "repr {:?} {:?}",
        model.tallies[0].get_mean(),
        model.tallies[0].get_std_dev()
    );
}

#[test]
#[ignore = "manual A/B harness: times a no-mesh woodcock run (regression check)"]
fn print_analog_woodcock_time() {
    // No mesh tally: the only new work on this path is one hoisted bool
    // and a per-tally slice index at delta-collisions. Expect noise.
    let geometry = build_two_material_geometry();
    let mut cell_tally = Tally::new();
    cell_tally.estimator = yamc_tallies::Estimator::TrackLength;
    cell_tally.filters = vec![Filter::Cell(CellFilter::from_id(1))];
    cell_tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        105,
    )))];
    let mut model = Model::new(
        geometry,
        vec![make_source_14mev()],
        vec![Arc::new(cell_tally)],
    );
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = TrackingMode::Woodcock;
    let t0 = std::time::Instant::now();
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(2_000_000),
            seed: 42,
            ..Default::default()
        })
        .unwrap();
    println!("no_mesh_time_s {:.3}", t0.elapsed().as_secs_f64());
}

#[test]
#[ignore = "manual A/B measurement harness: prints variance/FOM numbers"]
fn print_mesh_variance_ab_numbers() {
    // Fine mesh (2 cm bins) so bins are optically thin relative to the
    // majorant: the regime where true track-length beats the
    // collision-density equivalent. Run on both sides of a git stash
    // of the src changes and compare.
    let geometry = build_two_material_geometry();
    let mesh = RegularRectangularMesh::new([-12.0, -12.0, -12.0], [12.0, 12.0, 12.0], [12, 12, 12]);
    let mut tally = Tally::new();
    tally.estimator = yamc_tallies::Estimator::TrackLength;
    tally.filters = vec![Filter::Mesh(MeshFilter::new(mesh))];
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.name = Some("mesh_flux".to_string());
    let mut model = Model::new(geometry, vec![make_source_14mev()], vec![Arc::new(tally)]);
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = TrackingMode::Woodcock;
    let t0 = std::time::Instant::now();
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(200_000),
            seed: 42,
            ..Default::default()
        })
        .unwrap();
    let secs = t0.elapsed().as_secs_f64();

    let t = &model.tallies[0];
    let mean = t.get_mean();
    let std = t.get_std_dev();
    let var_sum: f64 = std.iter().map(|s| s * s).sum();
    let mut rel_errs: Vec<f64> = mean
        .iter()
        .zip(&std)
        .filter(|(m, _)| **m > 0.0)
        .map(|(m, s)| s / m)
        .collect();
    rel_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_rel = rel_errs.iter().sum::<f64>() / rel_errs.len() as f64;
    let med_rel = rel_errs[rel_errs.len() / 2];
    println!("time_s          {secs:.3}");
    println!("nonzero_bins    {}", rel_errs.len());
    println!("var_sum         {var_sum:.6e}");
    println!("avg_rel_err     {avg_rel:.6e}");
    println!("median_rel_err  {med_rel:.6e}");
    println!("fom_avg         {:.6e}", 1.0 / (avg_rel * avg_rel * secs));
    println!("fom_median      {:.6e}", 1.0 / (med_rel * med_rel * secs));
}

/// Fe sphere for the photon flight path (only Be/Fe/Li carry photon
/// data among the bundled fixtures).
fn build_fe_photon_geometry() -> Geometry {
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            surf.clone(),
        )))),
    };
    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nuclide_map = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    material
        .read_nuclear_data(&nuclide_map, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();
    let cell = Cell::new(Some(1), region, Some("fe".to_string()), Some(0));
    Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap()
}

#[test]
fn woodcock_photon_mesh_flux_matches_surface() {
    // The photon Woodcock branch shares the segment scorer; same
    // total-agreement and exit-clip properties as the neutron tests.
    let run = |mode: TrackingMode| -> (Vec<f64>, f64, f64) {
        let geometry = build_fe_photon_geometry();
        let tally = make_mesh_flux_tally(yamc_tallies::Estimator::TrackLength, "photon_mesh_flux");
        let source = ParticleSource::Photon(Source {
            space: yamc_source::source::SourceSpatialDistribution::Point(
                yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
            ),
            angle: AngularDistribution::Isotropic,
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![1.0e6], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        });
        let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
        model.verbose = yamc::model::Verbose::silent();
        model.transport_secondary_photons = true;
        model.tracking_mode = mode;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(20_000),
                seed: 7,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.get_mean(), t.total_mean(), t.total_std())
    };
    let (mean_s, total_s, total_std_s) = run(TrackingMode::Surface);
    let (mean_w, total_w, _) = run(TrackingMode::Woodcock);
    assert!(
        total_s > 0.0,
        "Surface photon mesh flux was zero -- rig broken"
    );
    assert!(total_w > 0.0, "Woodcock photon mesh flux was zero");
    let diff = (total_s - total_w).abs();
    let tol = 3.0 * total_std_s;
    assert!(
        diff < tol,
        "Woodcock photon mesh flux {total_w:.6e} vs Surface {total_s:.6e}, \
         diff {diff:.2e} exceeds 3 sigma {tol:.2e}",
    );
    for bin in outside_corner_bins() {
        assert_eq!(mean_s[bin], 0.0, "Surface scored outside bin {bin}");
        assert_eq!(
            mean_w[bin], 0.0,
            "photon leak segment not clipped: track length in outside bin {bin}",
        );
    }
}
