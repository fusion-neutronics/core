//! Tests for the delta-tracking API surface and transport:
//! `TrackingMode::Woodcock` (pure) and `TrackingMode::Hybrid` (delta
//! tracking with a per-cell surface fallback).
//!
//! Coverage:
//! - Default `tracking_mode=Surface` still works (no regression).
//! - `TrackingMode` round-trips through its string names.
//! - Pure `Woodcock` matches `Surface` within statistics on dense,
//!   void-free geometries: collision and track-length estimators, one-
//!   and two-material, URR, and photon (source / coupled / track-length).
//! - `Hybrid` matches `Surface` where pure Woodcock would be inefficient
//!   and falls back to surface stepping: a large void cell, a low-density
//!   material cell, and a low-density photon cell.
//! - Pure `Woodcock` is also correct (just slower) inside a void.
//!
//! Note on the URR + photon + decay-photon combination: the literal
//! single-nuclide case (one nuclide that is URR-bearing AND has photon
//! data AND is in the transmutation chain) is not representable with the
//! bundled fixtures -- only Co58 carries URR data, and only Be/Fe/Li
//! have photon data, with no overlap. So those interactions are tested
//! separately (URR via Co58; photons/decay via Fe) rather than as one
//! model.

use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Mt, ReactionRateScore, Score, Tally};
use yamc_tallies::CellFilter;

fn build_li6_geometry() -> (Geometry, Arc<Cell>) {
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 200.0,
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
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.534),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    let mat_arc = Arc::new(material);
    let cell = Cell::new(Some(1), region, Some("test_cell".to_string()), Some(0));
    let cell_arc = Arc::new(cell.clone());
    let geometry = Geometry::new(vec![cell], vec![mat_arc]).unwrap();
    (geometry, cell_arc)
}

fn make_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    })
}

fn make_collision_rxrate_tally(cell_id: u32) -> Tally {
    let mut tally = Tally::new();
    tally.estimator = yamc_tallies::Estimator::Collision;
    tally.filters = vec![Filter::Cell(CellFilter::from_id(cell_id))];
    tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        105,
    )))];
    tally.name = Some("rxrate_collision".to_string());
    tally
}

fn make_track_length_rxrate_tally(cell_id: u32) -> Tally {
    let mut tally = Tally::new();
    tally.estimator = yamc_tallies::Estimator::TrackLength;
    tally.filters = vec![Filter::Cell(CellFilter::from_id(cell_id))];
    tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        105,
    )))];
    tally.name = Some("rxrate_track_length".to_string());
    tally
}

#[test]
fn default_tracking_mode_is_surface() {
    // Belt-and-braces: the default Model has TrackingMode::Surface.
    // Anyone migrating from main shouldn't notice a behaviour change
    // unless they explicitly opt into Woodcock.
    let (geometry, _cell) = build_li6_geometry();
    let model = Model::new(geometry, vec![make_source()], vec![]);
    assert_eq!(model.tracking_mode, TrackingMode::Surface);
}

#[test]
fn woodcock_collision_rxrate_runs_and_is_nonzero() {
    // End-to-end smoke test: Woodcock transport + collision-estimator
    // MT 105 tally on a Li6 sphere with 1 MeV neutrons should produce
    // a non-zero reaction rate. If this fails, the transport loop or
    // the rejection/scoring path is broken.
    let (geometry, cell) = build_li6_geometry();
    let tally = make_collision_rxrate_tally(cell.cell_id.unwrap());
    let mut model = Model::new(geometry, vec![make_source()], vec![Arc::new(tally)]);
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = TrackingMode::Woodcock;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(2000),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    let t = &model.tallies[0];
    let mean = t.total_mean();
    assert!(
        mean > 0.0,
        "Woodcock + collision MT 105 on Li6 sphere returned zero mean ({mean})"
    );
}

/// Two-material geometry: inner Li6 sphere + outer Be9 annular shell.
/// Used by the LocalMajorant structural test and the Phase 4
/// multi-cell statistical-match tests.
fn build_li6_be9_two_material_geometry() -> (Geometry, Arc<Cell>) {
    // Inner Li6 sphere, r=5 -- larger than the 1.0/2.0 shell used by
    // the absorption_leakage_filters test, to give numerical lost-
    // particle handling more headroom.
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
    // Outer vacuum boundary, r=10
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

    // Realistic densities (Li6 = 0.534 g/cm³, Be9 = 1.85 g/cm³). The
    // two-material contrast is what exercises LocalMajorant; absolute
    // density doesn't matter for correctness.
    let mut mat_li6 = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.534),
    )
    .unwrap();
    mat_li6.set_material_id(1);
    mat_li6.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
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
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Be9".to_string(), "tests/Be9.arrow".to_string());
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
    let cell_inner_arc = Arc::new(cell_inner.clone());
    let geometry = Geometry::new(
        vec![cell_inner, cell_outer],
        vec![Arc::new(mat_li6), Arc::new(mat_be9)],
    )
    .unwrap();
    (geometry, cell_inner_arc)
}

#[test]
fn local_majorant_built_from_two_materials_has_two_entries() {
    // Phase 2 structural check: when the runtime sees a two-material
    // geometry, the LocalMajorant it would build via
    // `yamc_materials::LocalMajorant::new(material_refs)` must contain
    // one per-material entry per unique material_id. Just verifies
    // the construction path; the per-material *value* correctness is
    // covered by the LocalMajorant unit tests in yamc-materials
    // (which use synthetic data so they don't need macroscopic-XS
    // preparation -- which the runtime does inside `run_internal`).
    let (geometry, _inner) = build_li6_be9_two_material_geometry();
    let material_refs: Vec<&yamc_materials::Material> =
        geometry.materials.iter().map(|m| m.as_ref()).collect();
    let local = yamc_materials::LocalMajorant::new(&material_refs);
    assert_eq!(
        local.num_materials(),
        2,
        "two-material geometry should produce a LocalMajorant with 2 entries"
    );
    use yamc_materials::Majorant;
    let vac = local.sigma_max(None, 1.0e6);
    assert_eq!(vac, 0.0, "vacuum (material_id=None) must return 0");
}

#[test]
fn woodcock_track_length_matches_surface_within_statistics() {
    // Phase 3: track-length tallies work under Woodcock. Same
    // seed for Surface vs Woodcock with a track-length tally:
    // results must agree on the MT 105 reaction rate within 3σ.
    // This is the strongest test that the per-segment track-
    // length scoring in `transport_particle_woodcock` is
    // statistically correct.
    let run = |mode: TrackingMode| -> (f64, f64) {
        let (geometry, cell) = build_li6_geometry();
        let tally = make_track_length_rxrate_tally(cell.cell_id.unwrap());
        let mut model = Model::new(geometry, vec![make_source()], vec![Arc::new(tally)]);
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(5000),
                seed: 42,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.total_mean(), t.total_std())
    };
    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_wood, _) = run(TrackingMode::Woodcock);
    assert!(
        mean_surf > 0.0,
        "Surface (TL) mean was zero -- test rig broken"
    );
    assert!(mean_wood > 0.0, "Woodcock (TL) mean was zero");
    let diff = (mean_surf - mean_wood).abs();
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Woodcock TL mean {mean_wood:.6e} differs from Surface TL mean \
         {mean_surf:.6e} by {diff:.2e}, exceeding 3σ tolerance {tol:.2e}",
    );
}

#[test]
fn woodcock_matches_surface_within_statistics() {
    // Statistical-match test: same seed, same geometry, same source.
    // Woodcock+collision and Surface+collision must agree on the
    // MT 105 reaction rate within 3σ (using surface's std as the
    // tolerance scale).
    let run = |mode: TrackingMode| -> (f64, f64) {
        let (geometry, cell) = build_li6_geometry();
        let tally = make_collision_rxrate_tally(cell.cell_id.unwrap());
        let mut model = Model::new(geometry, vec![make_source()], vec![Arc::new(tally)]);
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(5000),
                seed: 42,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.total_mean(), t.total_std())
    };

    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_wood, _std_wood) = run(TrackingMode::Woodcock);

    assert!(mean_surf > 0.0, "Surface mean was zero -- test rig broken");
    assert!(mean_wood > 0.0, "Woodcock mean was zero");
    let diff = (mean_surf - mean_wood).abs();
    // 3σ tolerance using surface's std-error-of-mean.
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Woodcock mean {mean_wood:.6e} differs from Surface mean \
         {mean_surf:.6e} by {diff:.2e}, exceeding 3σ tolerance {tol:.2e}",
    );
}

#[test]
fn woodcock_collision_matches_surface_two_material() {
    // Phase 4 multi-cell statistical-match (collision estimator).
    // Two-material Li6 + Be9 geometry: same seed, same tally, both
    // modes. The collision estimator scores at the collision site
    // (event position) so there's no segment-cell-attribution bias --
    // this test should hold straightforwardly.
    let run = |mode: TrackingMode| -> (f64, f64) {
        let (geometry, inner_cell) = build_li6_be9_two_material_geometry();
        let tally = make_collision_rxrate_tally(inner_cell.cell_id.unwrap());
        let mut model = Model::new(geometry, vec![make_source()], vec![Arc::new(tally)]);
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model.max_lost_particles = usize::MAX;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(10_000),
                seed: 42,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.total_mean(), t.total_std())
    };
    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_wood, _) = run(TrackingMode::Woodcock);
    assert!(mean_surf > 0.0, "Surface (collision) mean was zero");
    assert!(mean_wood > 0.0, "Woodcock (collision) mean was zero");
    let diff = (mean_surf - mean_wood).abs();
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Woodcock collision mean {mean_wood:.6e} differs from Surface \
         mean {mean_surf:.6e} by {diff:.2e}, exceeding 3σ tolerance \
         {tol:.2e} on the two-material Li6+Be9 geometry",
    );
}

/// Multi-cell statistical-match, **track-length** estimator (neutron).
///
/// This is the regression test for the Phase 4.0 segment-attribution
/// bias. Earlier, the loop scored the full flight to the pre-flight
/// cell, so a flight crossing inner Li6 -> outer Be9 over-attributed
/// the Be9 portion to inner Li6 (~9 % error, 3σ-failing). The
/// delta-tracking flux estimator scores `weight / Σ_maj` at each
/// delta-collision in the cell that actually contains it, removing the
/// bias. Inner-cell MT 105 must now match surface tracking within 3σ.
#[test]
fn woodcock_track_length_matches_surface_two_material() {
    let run = |mode: TrackingMode| -> (f64, f64) {
        let (geometry, inner_cell) = build_li6_be9_two_material_geometry();
        let tally = make_track_length_rxrate_tally(inner_cell.cell_id.unwrap());
        let mut model = Model::new(geometry, vec![make_source()], vec![Arc::new(tally)]);
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model.max_lost_particles = usize::MAX;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(20_000),
                seed: 42,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.total_mean(), t.total_std())
    };
    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_wood, _) = run(TrackingMode::Woodcock);
    assert!(mean_surf > 0.0, "Surface (TL) mean was zero");
    assert!(mean_wood > 0.0, "Woodcock (TL) mean was zero");
    let diff = (mean_surf - mean_wood).abs();
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Woodcock TL mean {mean_wood:.6e} differs from Surface TL \
         mean {mean_surf:.6e} by {diff:.2e}, exceeding 3σ tolerance \
         {tol:.2e} on the two-material Li6+Be9 geometry",
    );
}

/// Build a U235 sphere model. URR-bearing material. Skipped if the
/// local nuclear-data fixture isn't present.
fn build_u235_geometry() -> Option<(Geometry, Arc<Cell>)> {
    let u235_path = "/home/jon/nuclear_data/endf-b8.0-arrow/neutron/U235.arrow";
    if !std::path::Path::new(u235_path).exists() {
        eprintln!("Skipping: {u235_path} not found");
        return None;
    }
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 5.0,
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
        HashMap::from([("U235".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(18.7),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("U235".to_string(), u235_path.to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    let mat_arc = Arc::new(material);
    let cell = Cell::new(Some(1), region, Some("u235_cell".to_string()), Some(0));
    let cell_arc = Arc::new(cell.clone());
    let geometry = Geometry::new(vec![cell], vec![mat_arc]).unwrap();
    Some((geometry, cell_arc))
}

#[test]
fn li6_material_does_not_have_urr_data() {
    // Sanity: the Li6 sphere material used by every other Woodcock test
    // must not carry URR data, otherwise those tests would all start
    // panicking with the new URR-rejection validation.
    let (geometry, _cell) = build_li6_geometry();
    for material in &geometry.materials {
        assert!(
            !material.has_urr_data(),
            "Li6 material unexpectedly has URR data -- the rest of the \
             Woodcock test suite would now fail validation"
        );
    }
}

#[test]
fn u235_material_has_urr_data() {
    let Some((geometry, _cell)) = build_u235_geometry() else {
        return;
    };
    let mat = &geometry.materials[0];
    assert!(
        mat.has_urr_data(),
        "U235 material should carry URR probability-table data"
    );
}

#[test]
fn woodcock_with_urr_material_runs() {
    // Phase 2c: URR-bearing materials are now supported. The
    // URR-aware majorant in `Material::total_xs_majorant` bounds
    // the worst-case URR-sampled Σ_t, so the rejection loop stays
    // in [0, 1] for any URR draw. Smoke test: run Woodcock on a
    // U235 sphere and assert the simulation completes (i.e. no
    // panic from `p_real > 1`).
    let Some((geometry, cell)) = build_u235_geometry() else {
        return; // U235 fixture not available; skip silently
    };
    let tally = make_collision_rxrate_tally(cell.cell_id.unwrap());
    let mut model = Model::new(geometry, vec![make_source()], vec![Arc::new(tally)]);
    model.tracking_mode = TrackingMode::Woodcock;
    model.verbose = yamc::model::Verbose::silent();
    // Thick U235 at 1 MeV bounces a lot; numerical edges are common.
    model.max_lost_particles = usize::MAX;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(500),
            ..Default::default()
        })
        .unwrap();

    // No assertion on the tally mean -- MT 105 (n,t) on U235 is rare
    // at 1 MeV so the count may be small or zero. The contract this
    // test verifies is that the simulation *completes* without the
    // rejection-loop assertion failing.
    let _t = &model.tallies[0];
}

#[test]
fn urr_aware_majorant_bounds_urr_samples() {
    // The strongest correctness check for Phase 2c: for a URR
    // material at energies inside the URR range, the URR-aware
    // Σ_maj must dominate every single Σ_t sample the existing
    // `total_xs_with_urr` would produce. Spot-check at energies
    // inside U235's URR range with many random samples each.
    let Some((geometry, _cell)) = build_u235_geometry() else {
        return; // skip silently
    };
    let material = &geometry.materials[0];
    // U235's URR range is approximately 2.25e3 - 25e3 eV in ENDF-B8.0.
    let energies = [3.0e3, 5.0e3, 1.0e4, 2.0e4];
    let mut rng = yamc::util::fast_rng::FastRng::new(0xc0ffee);
    for &e in &energies {
        let bound = material.total_xs_majorant(e);
        assert!(bound > 0.0, "URR-aware Σ_maj should be > 0 at {e} eV");
        // Sample many URR draws; the bound must dominate every one.
        for _ in 0..1000 {
            let urr_random: f64 = rng.random();
            let mut dummy_rng = yamc::util::fast_rng::FastRng::new(0);
            let (sigma_t_sample, _) =
                material.total_xs_with_urr(e, Some(urr_random), &mut dummy_rng);
            assert!(
                sigma_t_sample <= bound + 1e-9,
                "URR sample {sigma_t_sample:.6e} at {e} eV exceeds \
                 majorant bound {bound:.6e} (urr_random={urr_random})"
            );
        }
    }
}

/// Co58 sphere -- a bundled fixture that carries URR probability-table
/// data, so the URR + Woodcock self-shielding behaviour can be tested
/// in-tree (unlike the skip-if-absent U235 rig above).
fn build_co58_geometry() -> (Geometry, Arc<Cell>) {
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 30.0,
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
        HashMap::from([("Co58".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(8.9),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nuclide_map = HashMap::from([("Co58".to_string(), "tests/Co58.arrow".to_string())]);
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    let cell = Cell::new(Some(1), region, Some("co58".to_string()), Some(0));
    let cell_arc = Arc::new(cell.clone());
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    (geometry, cell_arc)
}

#[test]
fn woodcock_urr_matches_surface() {
    // Regression test for the URR + Woodcock self-shielding bias. The
    // URR probability-table band must be held across delta-collisions at
    // a given energy so the free-flight distance tracks the *sampled*
    // band (preserving resonance self-shielding), not the mean cross
    // section. Re-sampling the band per delta-collision under-counted the
    // flux badly (Co58 collision flux ~0.69x surface at 50 keV, ~0.25x at
    // 5 keV). With the fix, Woodcock matches surface within 3σ.
    let run = |mode: TrackingMode| -> (f64, f64) {
        let (geometry, cell) = build_co58_geometry();
        let mut tally = Tally::new();
        tally.estimator = yamc_tallies::Estimator::Collision;
        tally.filters = vec![Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap()))];
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.name = Some("urr_flux".to_string());
        // 50 keV source -- inside Co58's unresolved resonance range.
        let source = ParticleSource::Neutron(Source {
            space: yamc_source::source::SourceSpatialDistribution::Point(
                yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
            ),
            angle: AngularDistribution::Isotropic,
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![5.0e4], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        });
        let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model.max_lost_particles = usize::MAX;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(40_000),
                seed: 5,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.total_mean(), t.total_std())
    };
    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_wood, _) = run(TrackingMode::Woodcock);
    assert!(mean_surf > 0.0, "Surface URR flux was zero -- rig broken");
    assert!(mean_wood > 0.0, "Woodcock URR flux was zero");
    let diff = (mean_surf - mean_wood).abs();
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Woodcock URR flux {mean_wood:.6e} differs from Surface {mean_surf:.6e} \
         by {diff:.2e}, exceeding 3σ tolerance {tol:.2e} -- URR self-shielding \
         is not being preserved under delta tracking",
    );
}

/// Regression test for issue #223: Woodcock transport on concentric-
/// sphere geometries used to lose ~0.1 % of particles because the
/// boundary-crossing branch double-moved the particle (once explicitly,
/// then again inside `handle_surface_crossing`). After one such crossing
/// the particle landed past the outer vacuum surface but `current_cell`
/// was reset to "find next", so it ended up in "no cell" territory and
/// was flagged as lost. The fix removes the explicit move; this test
/// asserts no particles are lost on a 3-shell Li6 geometry at 10k
/// particles -- well above the rate that previously triggered the abort.
#[test]
fn woodcock_concentric_shells_no_lost_particles() {
    let r_max = 50.0_f64;
    let n_shells = 3_usize;
    let surfaces: Vec<Arc<Surface>> = (1..=n_shells)
        .map(|i| {
            Arc::new(Surface {
                surface_id: Some(i),
                kind: SurfaceKind::Sphere {
                    x0: 0.0,
                    y0: 0.0,
                    z0: 0.0,
                    radius: r_max * (i as f64) / (n_shells as f64),
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
    let mut material = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.46),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    let geometry = Geometry::new(cells, vec![Arc::new(material)]).unwrap();

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

    let mut model = Model::new(geometry, vec![source], vec![]);
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = TrackingMode::Woodcock;
    // Set high enough that the panic doesn't fire before we can inspect.
    model.max_lost_particles = usize::MAX;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(10_000),
            seed: 42,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        model.lost_particles.len(),
        0,
        "Woodcock on 3 concentric Li6 shells should not lose particles \
         (was ~0.1 % before fix); got {} lost",
        model.lost_particles.len()
    );
}

/// Defensive twin of `woodcock_concentric_shells_no_lost_particles`.
/// Issue #223's double-move bug lived in `transport_particle_woodcock`,
/// and Surface tracking was empirically clean on the same geometry both
/// before and after the fix. This test pins that down: the same 3-shell
/// Li6 geometry under Surface tracking must keep returning 0 lost
/// particles, so a future symmetric regression in the Surface path
/// can't ship silently.
#[test]
fn surface_concentric_shells_no_lost_particles() {
    let r_max = 50.0_f64;
    let n_shells = 3_usize;
    let surfaces: Vec<Arc<Surface>> = (1..=n_shells)
        .map(|i| {
            Arc::new(Surface {
                surface_id: Some(i),
                kind: SurfaceKind::Sphere {
                    x0: 0.0,
                    y0: 0.0,
                    z0: 0.0,
                    radius: r_max * (i as f64) / (n_shells as f64),
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
    let mut material = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.46),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    let geometry = Geometry::new(cells, vec![Arc::new(material)]).unwrap();

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

    let mut model = Model::new(geometry, vec![source], vec![]);
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model.max_lost_particles = usize::MAX;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(10_000),
            seed: 42,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        model.lost_particles.len(),
        0,
        "Surface on 3 concentric Li6 shells should not lose particles; \
         got {} lost",
        model.lost_particles.len()
    );
}

#[test]
fn tracking_mode_round_trips_through_string() {
    use std::str::FromStr;
    assert_eq!(
        TrackingMode::from_str("surface").unwrap(),
        TrackingMode::Surface
    );
    assert_eq!(
        TrackingMode::from_str("woodcock").unwrap(),
        TrackingMode::Woodcock
    );
    assert_eq!(
        TrackingMode::from_str("hybrid").unwrap(),
        TrackingMode::Hybrid
    );
    // Display round-trips for every variant.
    for mode in [
        TrackingMode::Surface,
        TrackingMode::Woodcock,
        TrackingMode::Hybrid,
    ] {
        assert_eq!(TrackingMode::from_str(&mode.to_string()).unwrap(), mode);
    }
    // Only the three named values are accepted -- no aliases.
    assert!(TrackingMode::from_str("delta").is_err());
    assert!(TrackingMode::from_str("nope").is_err());
}

// ===========================================================================
// Photon transport under Woodcock tracking
// ===========================================================================
//
// Phase 4 added photons to the pure-Woodcock loop: photon sources,
// coupled neutron->photon production, and D1S decay photons (all three
// share the banked-secondary transport path). These tests verify the
// photon branch runs and is statistically equivalent to surface tracking.

/// Iron sphere with both neutron (Fe56) and photon (Fe) data loaded.
/// Vacuum outer boundary. Single cell so track-length scoring is exact.
fn build_fe_photon_geometry(radius: f64) -> (Geometry, Arc<Cell>) {
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
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
    let cell_arc = Arc::new(cell.clone());
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    (geometry, cell_arc)
}

fn make_photon_source(energy_ev: f64) -> ParticleSource {
    ParticleSource::Photon(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![energy_ev], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn make_neutron_source_14mev() -> ParticleSource {
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

/// Cell-filtered photon flux tally. When `photon_only` is set, a
/// ParticleType filter restricts scoring to photons (used by the
/// coupled neutron->photon test). Statistical-match tests use the
/// collision estimator: under Woodcock it scores weight/Σ_t at real
/// collision sites in the post-flight cell, so it is free of the
/// track-length flight-segment cell-attribution bias (Phase 4.0).
fn make_photon_flux_tally(
    cell_id: u32,
    photon_only: bool,
    estimator: yamc_tallies::Estimator,
) -> Tally {
    let mut tally = Tally::new();
    tally.estimator = estimator;
    tally.filters = vec![Filter::Cell(CellFilter::from_id(cell_id))];
    if photon_only {
        tally
            .filters
            .push(Filter::ParticleType(ParticleTypeFilter::new(
                ParticleType::Photon,
            )));
    }
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.name = Some("photon_flux".to_string());
    tally
}

#[test]
fn woodcock_photon_source_runs_and_is_nonzero() {
    // Smoke test: a 1 MeV photon source on an Fe sphere under Woodcock
    // produces a non-zero flux. Exercises the photon branch end to end:
    // photon majorant sampling, rejection against the photon total XS,
    // handle_photon_collision, and flux scoring.
    let (geometry, cell) = build_fe_photon_geometry(10.0);
    let tally = make_photon_flux_tally(
        cell.cell_id.unwrap(),
        false,
        yamc_tallies::Estimator::TrackLength,
    );
    let mut model = Model::new(
        geometry,
        vec![make_photon_source(1.0e6)],
        vec![Arc::new(tally)],
    );
    model.verbose = yamc::model::Verbose::silent();
    model.transport_secondary_photons = true;
    model.tracking_mode = TrackingMode::Woodcock;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(5000),
            seed: 42,
            ..Default::default()
        })
        .unwrap();
    let mean = model.tallies[0].total_mean();
    assert!(
        mean > 0.0,
        "Woodcock photon-source flux on Fe sphere returned zero mean ({mean})"
    );
}

#[test]
fn woodcock_photon_source_matches_surface() {
    // Same seed, same geometry: Woodcock and Surface must agree on the
    // photon flux within 3σ. Single-cell so track-length is unbiased.
    let run = |mode: TrackingMode| -> (f64, f64) {
        let (geometry, cell) = build_fe_photon_geometry(10.0);
        let tally = make_photon_flux_tally(
            cell.cell_id.unwrap(),
            false,
            yamc_tallies::Estimator::Collision,
        );
        let mut model = Model::new(
            geometry,
            vec![make_photon_source(1.0e6)],
            vec![Arc::new(tally)],
        );
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
        (t.total_mean(), t.total_std())
    };
    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_wood, _) = run(TrackingMode::Woodcock);
    assert!(
        mean_surf > 0.0,
        "Surface photon flux was zero -- rig broken"
    );
    assert!(mean_wood > 0.0, "Woodcock photon flux was zero");
    let diff = (mean_surf - mean_wood).abs();
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Woodcock photon flux {mean_wood:.6e} differs from Surface flux \
         {mean_surf:.6e} by {diff:.2e}, exceeding 3σ tolerance {tol:.2e}",
    );
}

#[test]
fn woodcock_coupled_neutron_photon_produces_photons() {
    // Neutron source + transport_secondary_photons: (n,γ) and inelastic photon
    // production bank secondary photons that transport through the same
    // Woodcock loop. A photon-filtered flux tally must be non-zero and
    // agree with surface tracking within 3σ.
    let run = |mode: TrackingMode| -> (f64, f64) {
        let (geometry, cell) = build_fe_photon_geometry(20.0);
        let tally = make_photon_flux_tally(
            cell.cell_id.unwrap(),
            true,
            yamc_tallies::Estimator::Collision,
        );
        let mut model = Model::new(
            geometry,
            vec![make_neutron_source_14mev()],
            vec![Arc::new(tally)],
        );
        model.verbose = yamc::model::Verbose::silent();
        model.transport_secondary_photons = true;
        model.tracking_mode = mode;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(20_000),
                seed: 123,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.total_mean(), t.total_std())
    };
    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_wood, _) = run(TrackingMode::Woodcock);
    assert!(
        mean_wood > 0.0,
        "Woodcock coupled n->photon flux was zero -- no secondary photons transported"
    );
    assert!(
        mean_surf > 0.0,
        "Surface coupled n->photon flux was zero -- rig broken"
    );
    let diff = (mean_surf - mean_wood).abs();
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Woodcock coupled photon flux {mean_wood:.6e} differs from Surface \
         {mean_surf:.6e} by {diff:.2e}, exceeding 3σ tolerance {tol:.2e}",
    );
}

#[test]
fn woodcock_photon_track_length_matches_surface() {
    // Photon **track-length** flux must match surface within 3σ. Before
    // the delta-tracking estimator this failed: a photon flight that
    // exited the sphere had its full sampled length attributed to the
    // cell, over-counting flux (the Phase 4.0 leakage bias, severe for
    // long-mean-free-path photons). The delta-tracking estimator scores
    // weight/Σ_maj at each delta-collision in the containing cell, which
    // removes the bias.
    let run = |mode: TrackingMode| -> (f64, f64) {
        let (geometry, cell) = build_fe_photon_geometry(10.0);
        let tally = make_photon_flux_tally(
            cell.cell_id.unwrap(),
            false,
            yamc_tallies::Estimator::TrackLength,
        );
        let mut model = Model::new(
            geometry,
            vec![make_photon_source(1.0e6)],
            vec![Arc::new(tally)],
        );
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
        (t.total_mean(), t.total_std())
    };
    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_wood, _) = run(TrackingMode::Woodcock);
    assert!(mean_surf > 0.0 && mean_wood > 0.0);
    let diff = (mean_surf - mean_wood).abs();
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Woodcock photon TL flux {mean_wood:.6e} differs from Surface \
         {mean_surf:.6e} by {diff:.2e}, exceeding 3σ tolerance {tol:.2e}",
    );
}

#[test]
fn photon_majorant_bounds_photon_xs() {
    // Correctness invariant for Woodcock photon rejection: the photon
    // majorant must be >= the macroscopic photon total XS at every
    // energy (otherwise p_real = Σ_t / Σ_maj > 1 and the rejection
    // loop is biased). Build the majorant from an Fe material and check
    // it bounds calculate_photon_xs(E).total across the spectrum.
    use yamc_materials::{GlobalPhotonMajorant, Majorant};

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
    // Mirror the run-time material prep order: macroscopic XS first
    // (populates atom-density caches), then photon data.
    material.calculate_macroscopic_xs(&vec![1], true);
    material.init_photon_data(&photon_paths).unwrap();

    let majorant = GlobalPhotonMajorant::new(&[&material]);
    assert!(majorant.num_energy_points() > 0, "empty photon grid");

    // Sample log-spaced energies from 10 keV to 20 MeV (covers the
    // photoelectric, Compton, and pair-production regimes).
    let mut e = 1.0e4_f64;
    while e <= 2.0e7 {
        let sigma_t = material.calculate_photon_xs(e).total;
        let sigma_maj = majorant.sigma_max(Some(1), e);
        // Allow a tiny relative slack for floating-point interpolation
        // mismatch between the linear-interp majorant and the log-log
        // XS evaluation; this is the same slack the rejection clamp
        // absorbs at runtime.
        assert!(
            sigma_maj >= sigma_t * (1.0 - 1e-9),
            "photon majorant {sigma_maj:.6e} < Σ_t {sigma_t:.6e} at {e:.3e} eV"
        );
        e *= 1.2;
    }
}

/// Regression test for the Hybrid void handling. A neutron source sits
/// in a large vacuum cavity wrapped in a Pb208 shell. Pure Woodcock takes
/// ~Σ_maj·R fictitious delta-steps to cross the void (collapsing
/// throughput as the cavity grows); `Hybrid` streams straight to the
/// boundary in void cells instead. This checks the *correctness* side of
/// that fallback: the track-length flux in the void cell must still match
/// surface tracking within 3σ (the streaming scores the same weight ×
/// path-length the surface estimator does, i.e. the expected value of the
/// collision-density estimator it replaces). Pure Woodcock correctness in
/// a void is covered separately by [`woodcock_pure_void_flux_matches_surface`].
#[test]
fn hybrid_void_flux_matches_surface() {
    let r_void = 200.0_f64;
    let inner = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: r_void,
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
            radius: r_void + 40.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });

    let run = |mode: TrackingMode| -> (f64, f64) {
        // Cavity: inside inner sphere, no material (void).
        let cavity = Cell::new(
            Some(1),
            Region {
                expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                    HalfspaceType::Above(inner.clone()),
                ))),
            },
            Some("cavity".to_string()),
            None,
        );
        // Shell: Pb208 between the spheres.
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
            Some(0),
        );
        let geometry = Geometry::new(vec![cavity, shell], vec![Arc::new(pb)]).unwrap();

        // Track-length flux in the VOID cell (cell 1). Collision estimator
        // would be zero there (no collisions), so track-length is the one
        // that exercises the void-streaming scoring.
        let mut tally = Tally::new();
        tally.estimator = yamc_tallies::Estimator::TrackLength;
        tally.filters = vec![Filter::Cell(CellFilter::from_id(1))];
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.name = Some("void_flux".to_string());

        let mut model = Model::new(
            geometry,
            vec![make_neutron_source_14mev()],
            vec![Arc::new(tally)],
        );
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model.max_lost_particles = usize::MAX;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(20_000),
                seed: 99,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.total_mean(), t.total_std())
    };

    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_hyb, _) = run(TrackingMode::Hybrid);
    assert!(mean_surf > 0.0, "Surface void flux was zero -- rig broken");
    assert!(mean_hyb > 0.0, "Hybrid void flux was zero");
    let diff = (mean_surf - mean_hyb).abs();
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Hybrid void flux {mean_hyb:.6e} differs from Surface {mean_surf:.6e} \
         by {diff:.2e}, exceeding 3σ tolerance {tol:.2e} -- void-streaming \
         track-length scoring is biased",
    );
}

/// Pure `Woodcock` (no hybrid fallback) is also *correct* inside a void,
/// just slower: it streams across the cavity via fictitious delta-steps,
/// scoring `weight/Σ_maj` at each, which is the same expectation as the
/// surface track-length estimator. A small cavity keeps the fictitious-
/// step count (and so the runtime) modest. Guards the pure-mode void path
/// against regressions now that it is user-selectable.
#[test]
fn woodcock_pure_void_flux_matches_surface() {
    let r_void = 20.0_f64;
    let inner = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: r_void,
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
            radius: r_void + 40.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });

    let run = |mode: TrackingMode| -> (f64, f64) {
        let cavity = Cell::new(
            Some(1),
            Region {
                expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                    HalfspaceType::Above(inner.clone()),
                ))),
            },
            Some("cavity".to_string()),
            None,
        );
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
            Some(0),
        );
        let geometry = Geometry::new(vec![cavity, shell], vec![Arc::new(pb)]).unwrap();

        let mut tally = Tally::new();
        tally.estimator = yamc_tallies::Estimator::TrackLength;
        tally.filters = vec![Filter::Cell(CellFilter::from_id(1))];
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.name = Some("void_flux".to_string());

        let mut model = Model::new(
            geometry,
            vec![make_neutron_source_14mev()],
            vec![Arc::new(tally)],
        );
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model.max_lost_particles = usize::MAX;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(20_000),
                seed: 99,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.total_mean(), t.total_std())
    };

    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_wood, _) = run(TrackingMode::Woodcock);
    assert!(mean_surf > 0.0, "Surface void flux was zero -- rig broken");
    assert!(mean_wood > 0.0, "Pure Woodcock void flux was zero");
    let diff = (mean_surf - mean_wood).abs();
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Pure Woodcock void flux {mean_wood:.6e} differs from Surface \
         {mean_surf:.6e} by {diff:.2e}, exceeding 3σ tolerance {tol:.2e} \
         -- pure-mode void streaming is biased",
    );
}

/// Regression test for the low-density-material hybrid. A low-density
/// Li6 ball (majorant-inefficient: its Σ_t is a small fraction of the
/// Pb208 shell that sets the global majorant) is surface-tracked rather
/// than delta-tracked, which exercises the surface-step *collision*
/// branch -- the part that distinguishes a low-density material cell
/// (collisions happen) from a void (none do). The MT 105 reaction rate
/// in the Li6 ball must match surface tracking within 3σ, confirming the
/// hybrid dispatch is unbiased where it switches a material cell to
/// surface stepping.
#[test]
fn hybrid_low_density_material_matches_surface() {
    let inner = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 100.0,
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
            radius: 140.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });

    let run = |mode: TrackingMode| -> (f64, f64) {
        // Inner ball: low-density Li6 (0.2 g/cc) -- Σ_t well below the
        // Pb208 majorant, so the hybrid surface-tracks it.
        let mut li6 = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(0.2),
        )
        .unwrap();
        li6.set_material_id(1);
        li6.set_temperature("294");
        li6.read_nuclear_data(
            &HashMap::from([("Li6".to_string(), "tests/Li6.arrow".to_string())]),
            None,
        )
        .unwrap();
        // Dense Pb208 shell sets the global majorant.
        let mut pb = Material::new(
            HashMap::from([("Pb208".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(11.34),
        )
        .unwrap();
        pb.set_material_id(2);
        pb.set_temperature("294");
        pb.read_nuclear_data(
            &HashMap::from([("Pb208".to_string(), "tests/Pb208.arrow".to_string())]),
            None,
        )
        .unwrap();

        let ball = Cell::new(
            Some(1),
            Region {
                expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                    HalfspaceType::Above(inner.clone()),
                ))),
            },
            Some("li6_ball".to_string()),
            Some(0),
        );
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
            Some("pb_shell".to_string()),
            Some(1),
        );
        let geometry = Geometry::new(vec![ball, shell], vec![Arc::new(li6), Arc::new(pb)]).unwrap();

        // MT 105 (Li6 (n,t)) reaction rate in the low-density ball.
        let mut tally = Tally::new();
        tally.estimator = yamc_tallies::Estimator::Collision;
        tally.filters = vec![Filter::Cell(CellFilter::from_id(1))];
        tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
            105,
        )))];
        tally.name = Some("li6_nt".to_string());

        let mut model = Model::new(
            geometry,
            vec![make_neutron_source_14mev()],
            vec![Arc::new(tally)],
        );
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model.max_lost_particles = usize::MAX;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(20_000),
                seed: 7,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.total_mean(), t.total_std())
    };

    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_hyb, _) = run(TrackingMode::Hybrid);
    assert!(mean_surf > 0.0, "Surface MT 105 was zero -- rig broken");
    assert!(mean_hyb > 0.0, "Hybrid MT 105 was zero");
    let diff = (mean_surf - mean_hyb).abs();
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Hybrid low-density MT 105 {mean_hyb:.6e} differs from Surface \
         {mean_surf:.6e} by {diff:.2e}, exceeding 3σ tolerance {tol:.2e} -- \
         the low-density-material hybrid surface step is biased",
    );
}

/// Regression test for the **photon** low-density hybrid. A large
/// low-density Fe cavity (its photon Σ_t is a small fraction of the
/// dense Fe shell that sets the photon majorant, and its chord is large)
/// is surface-tracked, exercising the photon branch of the hybrid
/// surface step (cross + `do_photon_collision`). Photon flux in the
/// cavity must match surface tracking within 3σ.
#[test]
fn hybrid_photon_low_density_matches_surface() {
    let inner = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 50.0,
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
            radius: 70.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });

    let fe = |density: f64, id: u32| -> Material {
        let mut m = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(density),
        )
        .unwrap();
        m.set_material_id(id);
        m.set_temperature("294");
        let nuclide_map = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
        let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
        m.read_nuclear_data(&nuclide_map, Some(&photon_paths))
            .unwrap();
        m
    };

    let run = |mode: TrackingMode| -> (f64, f64) {
        // Low-density Fe cavity (0.5 g/cc) + dense Fe shell (7.874 g/cc).
        // The cavity's photon Σ_t is ~1/16 of the shell's, and its chord
        // is large, so the hybrid surface-tracks it.
        let cavity = Cell::new(
            Some(1),
            Region {
                expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                    HalfspaceType::Above(inner.clone()),
                ))),
            },
            Some("low_density".to_string()),
            Some(0),
        );
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
            Some("dense".to_string()),
            Some(1),
        );
        let geometry = Geometry::new(
            vec![cavity, shell],
            vec![Arc::new(fe(0.5, 1)), Arc::new(fe(7.874, 2))],
        )
        .unwrap();

        // Photon flux (track-length) in the surface-tracked cavity.
        let mut tally = Tally::new();
        tally.estimator = yamc_tallies::Estimator::TrackLength;
        tally.filters = vec![Filter::Cell(CellFilter::from_id(1))];
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.name = Some("photon_flux".to_string());

        let mut model = Model::new(
            geometry,
            vec![make_photon_source(1.0e6)],
            vec![Arc::new(tally)],
        );
        model.verbose = yamc::model::Verbose::silent();
        model.transport_secondary_photons = true;
        model.tracking_mode = mode;
        model.max_lost_particles = usize::MAX;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(20_000),
                seed: 11,
                ..Default::default()
            })
            .unwrap();
        let t = &model.tallies[0];
        (t.total_mean(), t.total_std())
    };

    let (mean_surf, std_surf) = run(TrackingMode::Surface);
    let (mean_hyb, _) = run(TrackingMode::Hybrid);
    assert!(
        mean_surf > 0.0,
        "Surface photon flux was zero -- rig broken"
    );
    assert!(mean_hyb > 0.0, "Hybrid photon flux was zero");
    let diff = (mean_surf - mean_hyb).abs();
    let tol = 3.0 * std_surf;
    assert!(
        diff < tol,
        "Hybrid low-density photon flux {mean_hyb:.6e} differs from Surface \
         {mean_surf:.6e} by {diff:.2e}, exceeding 3σ tolerance {tol:.2e} -- the \
         photon hybrid surface step is biased",
    );
}
