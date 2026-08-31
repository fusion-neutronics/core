//! Collision-estimator flux tally tests.
//!
//! The collision estimator scores `weight / Σ_t` at every collision
//! site. Summed over all collisions of all histories and divided by
//! the number of histories, it converges to the same physical quantity
//! the track-length estimator estimates (volume-averaged flux). So
//! independent runs at the same seed must agree to within statistical
//! noise -- and the validator must reject `Estimator::Collision` for
//! scores that haven't been ported yet (reaction-rate / production /
//! damage-energy / photon-XS).

use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::mesh::MeshFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::tally::{
    DamageEnergyScore, FluxScore, HeatingLocalScore, HeatingScore, Mt, PhotonComponent,
    PhotonXSScore, ProductionScore, ReactionRateScore, Score, Tally,
};
use yamc_tallies::{CellFilter, Estimator};

/// Build a fixed Li6 sphere with a 14.06 MeV point source. The tally
/// scores `score` under the requested estimator. Same seed across
/// callers so estimator-vs-estimator comparisons are deterministic at a
/// few-thousand-history budget.
fn build_sphere_model_with_score(estimator: Estimator, score: Score) -> Model {
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
    let mat_arc = Arc::new(material);
    let cell = Cell::new(Some(1), region, Some("cell".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![mat_arc]).unwrap();

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

    let cell_filter = Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap()));
    let mut tally = Tally::new();
    tally.filters.push(cell_filter);
    let score_name = score.name();
    tally.set_scores_mixed(vec![score]);
    tally.name = Some(format!("{score_name}_{estimator:?}"));
    tally.estimator = estimator;

    Model::new(geometry, vec![source], vec![Arc::new(tally)])
}

/// Backwards-compatible shim -- most tests just want flux.
fn build_sphere_model(estimator: Estimator) -> Model {
    build_sphere_model_with_score(estimator, Score::Flux(FluxScore))
}

/// Shared run settings for the neutron sphere: 5000 histories at a fixed seed
/// so estimator-vs-estimator comparisons are deterministic.
fn sphere_settings() -> TransportSettings {
    TransportSettings {
        total_particles: Some(5_000),
        seed: 0xABCD,
        ..Default::default()
    }
}

#[test]
fn collision_flux_matches_track_length_to_statistical_noise() {
    // Same geometry, source, seed -- only the estimator changes.
    // Both estimators target the same physical flux, so they must
    // agree to within a few statistical sigmas at a few-thousand
    // particle history budget.
    let mut tl = build_sphere_model(Estimator::TrackLength);
    let mut co = build_sphere_model(Estimator::Collision);
    tl.simulate_transport(&sphere_settings()).unwrap();
    co.simulate_transport(&sphere_settings()).unwrap();

    let tl_mean = tl.tallies[0].get_mean();
    let co_mean = co.tallies[0].get_mean();
    assert_eq!(tl_mean.len(), 1);
    assert_eq!(co_mean.len(), 1);
    let tl_v = tl_mean[0];
    let co_v = co_mean[0];

    // Cell-volume averaged flux (units: cm/source = cm here since
    // weight is 1.0 per history). For an absorbing sphere with a
    // central isotropic 14.06 MeV source, the two estimators
    // should agree to within ~1–2% at 5000 particles.
    let rel = (tl_v - co_v).abs() / tl_v.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel < 0.05,
        "track-length {tl_v:.4e} vs collision {co_v:.4e} disagree (rel diff {rel:.3e})"
    );
    assert!(tl_v > 0.0, "expected non-zero track-length flux");
    assert!(co_v > 0.0, "expected non-zero collision flux");
}

#[test]
fn collision_mesh_flux_produces_nonzero_bins() {
    // Confirms the collision estimator at least lights up the mesh --
    // a smoke test that the scoring hook is actually firing.
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
    let mat_arc = Arc::new(material);
    let cell = Cell::new(Some(1), region, Some("cell".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![mat_arc]).unwrap();

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

    let mesh = RegularRectangularMesh::new([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [8, 8, 8]);
    let mesh_filter = MeshFilter::new(mesh);
    let mut tally = Tally::new();
    tally.filters.push(Filter::Mesh(mesh_filter));
    tally.set_scores_mixed(vec![Score::Flux(FluxScore)]);
    tally.estimator = Estimator::Collision;
    tally.name = Some("flux_collision_mesh".into());

    let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(1_000),
            ..Default::default()
        })
        .unwrap();

    let mean = model.tallies[0].get_mean();
    assert_eq!(mean.len(), 8 * 8 * 8);
    let nonzero = mean.iter().filter(|&&v| v > 0.0).count();
    assert!(
        nonzero > 0,
        "collision estimator should score in at least one mesh bin"
    );
}

// (The validator-rejects-collision test was removed once every score
// type gained collision support -- `Score::required_estimator` now
// returns `None` for every variant, so there is no live mismatch to
// exercise. If a future score type is added that's TL-only, restore a
// `validator_rejects_collision_with_<that_score>` test here.)

#[test]
fn collision_neutron_heating_matches_track_length() {
    // Neutron heating under TrackLength uses KERMA: `heating_xs · weight · Δs`
    // integrated over each segment. Under Collision it uses
    // `(heating_xs / Σ_t) · weight` summed at every collision site.
    // At equilibrium <Σ_t · Δs> = <N_collisions> per history, so the
    // two estimators converge to the same expected eV deposited per
    // source neutron. Same model, same seed → must agree to within
    // statistical noise.
    let mut tl =
        build_sphere_model_with_score(Estimator::TrackLength, Score::Heating(HeatingScore));
    let mut co = build_sphere_model_with_score(Estimator::Collision, Score::Heating(HeatingScore));
    tl.simulate_transport(&sphere_settings()).unwrap();
    co.simulate_transport(&sphere_settings()).unwrap();

    let tl_v = tl.tallies[0].get_mean()[0];
    let co_v = co.tallies[0].get_mean()[0];

    assert!(tl_v > 0.0, "expected non-zero track-length neutron heating");
    assert!(co_v > 0.0, "expected non-zero collision neutron heating");
    let rel = (tl_v - co_v).abs() / tl_v.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel < 0.10,
        "track-length {tl_v:.4e} vs collision {co_v:.4e} disagree (rel diff {rel:.3e})"
    );
}

#[test]
fn collision_neutron_heating_local_matches_track_length() {
    // Same equivalence for `heating-local` (MT 901) -- uses
    // `lookup_heating_local_xs` instead of `lookup_heating_xs`.
    let mut tl = build_sphere_model_with_score(
        Estimator::TrackLength,
        Score::HeatingLocal(HeatingLocalScore),
    );
    let mut co =
        build_sphere_model_with_score(Estimator::Collision, Score::HeatingLocal(HeatingLocalScore));
    tl.simulate_transport(&sphere_settings()).unwrap();
    co.simulate_transport(&sphere_settings()).unwrap();

    let tl_v = tl.tallies[0].get_mean()[0];
    let co_v = co.tallies[0].get_mean()[0];

    assert!(tl_v > 0.0, "expected non-zero track-length heating-local");
    assert!(co_v > 0.0, "expected non-zero collision heating-local");
    let rel = (tl_v - co_v).abs() / tl_v.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel < 0.10,
        "track-length {tl_v:.4e} vs collision {co_v:.4e} disagree (rel diff {rel:.3e})"
    );
}

#[test]
fn collision_reaction_rate_matches_track_length() {
    // Per-collision contribution is `(σ_r / Σ_t) · weight`; summed
    // across all collisions of all histories it converges to the
    // expected number of type-r reactions per source neutron -- the
    // same quantity track-length integrates as `Σ_r · weight · Δs`.
    // Same model, same seed → must agree to within statistical noise.
    let rr = Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(2))); // elastic
    let mut tl = build_sphere_model_with_score(Estimator::TrackLength, rr.clone());
    let mut co = build_sphere_model_with_score(Estimator::Collision, rr);
    tl.simulate_transport(&sphere_settings()).unwrap();
    co.simulate_transport(&sphere_settings()).unwrap();

    let tl_v = tl.tallies[0].get_mean()[0];
    let co_v = co.tallies[0].get_mean()[0];

    assert!(tl_v > 0.0, "expected non-zero track-length elastic RR");
    assert!(co_v > 0.0, "expected non-zero collision elastic RR");
    let rel = (tl_v - co_v).abs() / tl_v.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel < 0.05,
        "track-length {tl_v:.4e} vs collision {co_v:.4e} disagree (rel diff {rel:.3e})"
    );
}

#[test]
fn collision_reaction_rate_matches_track_length_total() {
    // Sanity: MT 1 (total) -- under URR this hits the URR shortcut
    // in score_collision and `(σ_total / Σ_t) · weight = weight / N`,
    // i.e. essentially the per-nuclide collision rate. Outside URR
    // it's a smooth-XS lookup. Both estimators must agree.
    let rr = Score::ReactionRate(ReactionRateScore::total());
    let mut tl = build_sphere_model_with_score(Estimator::TrackLength, rr.clone());
    let mut co = build_sphere_model_with_score(Estimator::Collision, rr);
    tl.simulate_transport(&sphere_settings()).unwrap();
    co.simulate_transport(&sphere_settings()).unwrap();

    let tl_v = tl.tallies[0].get_mean()[0];
    let co_v = co.tallies[0].get_mean()[0];

    assert!(tl_v > 0.0, "expected non-zero track-length total RR");
    assert!(co_v > 0.0, "expected non-zero collision total RR");
    let rel = (tl_v - co_v).abs() / tl_v.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel < 0.05,
        "track-length {tl_v:.4e} vs collision {co_v:.4e} disagree (rel diff {rel:.3e})"
    );
}

#[test]
fn collision_reaction_rate_matches_track_length_absorption() {
    // Absorption (MT 27) is the textbook collision-estimator use case:
    // optically-thick / absorbing regions where the variance gain is
    // largest. Li6 has (n,t) absorption at 14 MeV so the score is
    // non-trivial.
    let rr = Score::ReactionRate(ReactionRateScore::absorption());
    let mut tl = build_sphere_model_with_score(Estimator::TrackLength, rr.clone());
    let mut co = build_sphere_model_with_score(Estimator::Collision, rr);
    tl.simulate_transport(&sphere_settings()).unwrap();
    co.simulate_transport(&sphere_settings()).unwrap();

    let tl_v = tl.tallies[0].get_mean()[0];
    let co_v = co.tallies[0].get_mean()[0];

    assert!(tl_v > 0.0, "expected non-zero track-length absorption RR");
    assert!(co_v > 0.0, "expected non-zero collision absorption RR");
    let rel = (tl_v - co_v).abs() / tl_v.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel < 0.10,
        "track-length {tl_v:.4e} vs collision {co_v:.4e} disagree (rel diff {rel:.3e})"
    );
}

#[test]
fn collision_production_h3_matches_track_length() {
    // Per-collision contribution is `(σ_production_mt / Σ_t) · weight`;
    // summed across all neutron collisions of all histories it converges
    // to the expected number of produced particles per source neutron --
    // the same quantity track-length integrates as `Σ_production · w · Δs`.
    // Li6 (n,t) at 14 MeV gives a strong H3-production signal.
    let p = Score::Production(ProductionScore::H3);
    let mut tl = build_sphere_model_with_score(Estimator::TrackLength, p.clone());
    let mut co = build_sphere_model_with_score(Estimator::Collision, p);
    tl.simulate_transport(&sphere_settings()).unwrap();
    co.simulate_transport(&sphere_settings()).unwrap();

    let tl_v = tl.tallies[0].get_mean()[0];
    let co_v = co.tallies[0].get_mean()[0];

    assert!(tl_v > 0.0, "expected non-zero track-length H3-production");
    assert!(co_v > 0.0, "expected non-zero collision H3-production");
    let rel = (tl_v - co_v).abs() / tl_v.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel < 0.10,
        "track-length {tl_v:.4e} vs collision {co_v:.4e} disagree (rel diff {rel:.3e})"
    );
}

#[test]
fn collision_production_he4_matches_track_length() {
    // Li6 (n,α) at 14 MeV → He4-production; same equivalence check
    // exercises a different MT (MT 207) and confirms the `p.mt` plumbing
    // isn't hard-coded to one production channel.
    let p = Score::Production(ProductionScore::HE4);
    let mut tl = build_sphere_model_with_score(Estimator::TrackLength, p.clone());
    let mut co = build_sphere_model_with_score(Estimator::Collision, p);
    tl.simulate_transport(&sphere_settings()).unwrap();
    co.simulate_transport(&sphere_settings()).unwrap();

    let tl_v = tl.tallies[0].get_mean()[0];
    let co_v = co.tallies[0].get_mean()[0];

    assert!(tl_v > 0.0, "expected non-zero track-length He4-production");
    assert!(co_v > 0.0, "expected non-zero collision He4-production");
    let rel = (tl_v - co_v).abs() / tl_v.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel < 0.10,
        "track-length {tl_v:.4e} vs collision {co_v:.4e} disagree (rel diff {rel:.3e})"
    );
}

#[test]
fn collision_damage_energy_matches_track_length() {
    // Per-collision contribution is `(damage_energy_xs / Σ_t) · weight`
    // (MT 444). Damage-energy is meaningful for radiation-damage / DPA
    // workflows. Li6 has a small but non-zero damage XS at 14 MeV; both
    // estimators must agree on it within statistical noise.
    let d = Score::DamageEnergy(DamageEnergyScore);
    let mut tl = build_sphere_model_with_score(Estimator::TrackLength, d.clone());
    let mut co = build_sphere_model_with_score(Estimator::Collision, d);
    tl.simulate_transport(&sphere_settings()).unwrap();
    co.simulate_transport(&sphere_settings()).unwrap();

    let tl_v = tl.tallies[0].get_mean()[0];
    let co_v = co.tallies[0].get_mean()[0];

    assert!(tl_v > 0.0, "expected non-zero track-length damage-energy");
    assert!(co_v > 0.0, "expected non-zero collision damage-energy");
    let rel = (tl_v - co_v).abs() / tl_v.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel < 0.10,
        "track-length {tl_v:.4e} vs collision {co_v:.4e} disagree (rel diff {rel:.3e})"
    );
}

// ---------------------------------------------------------------------
// Photon-side helpers + tests
// ---------------------------------------------------------------------

/// Fe (natural)-photon sphere with a single-component PhotonXS or Flux
/// tally under the requested estimator. Photon source energy is chosen
/// per test so all four PhotonXS components have non-trivial signal
/// (coherent + incoherent + photoelectric below pair-production
/// threshold; ≥1.022 MeV opens pair-production).
fn build_photon_sphere_model_with_score(
    estimator: Estimator,
    score: Score,
    source_energy_ev: f64,
) -> Model {
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
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Fe56".to_string(), "tests/Fe56.arrow".to_string());
    let mut photon_paths: HashMap<String, String> = HashMap::new();
    photon_paths.insert("Fe".to_string(), "tests/Fe.arrow".to_string());
    material
        .read_nuclear_data(&nuclide_map, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();
    let mat_arc = Arc::new(material);
    let cell = Cell::new(Some(1), region, Some("cell".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![mat_arc]).unwrap();

    let source = ParticleSource::Photon(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![source_energy_ev], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let cell_filter = Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap()));
    let mut tally = Tally::new();
    tally.filters.push(cell_filter);
    // Gate this tally to photons so neutron sources can't accidentally
    // contribute to a photon-XS score in mixed simulations.
    tally
        .filters
        .push(Filter::ParticleType(ParticleTypeFilter::new(
            yamc_particle::particle::ParticleType::Photon,
        )));
    let score_name = score.name();
    tally.set_scores_mixed(vec![score]);
    tally.name = Some(format!("{score_name}_{estimator:?}"));
    tally.estimator = estimator;

    // 200k particles per estimator run gives a noise floor of ~3 %
    // even on the weakest channel (photoelectric in Fe at 1 MeV),
    // comfortably inside the 10 % parity tolerance below. 10k was
    // enough locally but tripped CI on platforms whose float-ordering
    // landed in the unlucky tail (#208).
    let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
    model.transport_secondary_photons = true;
    model
}

/// Shared run settings for the photon sphere: 200k histories at a fixed seed
/// (noise floor ~3 %, see #208).
fn photon_sphere_settings() -> TransportSettings {
    TransportSettings {
        total_particles: Some(200_000),
        seed: 0xABCD,
        ..Default::default()
    }
}

fn check_photon_score_parity(score: Score, source_energy_ev: f64, score_label: &str) {
    let mut tl = build_photon_sphere_model_with_score(
        Estimator::TrackLength,
        score.clone(),
        source_energy_ev,
    );
    let mut co =
        build_photon_sphere_model_with_score(Estimator::Collision, score, source_energy_ev);
    tl.simulate_transport(&photon_sphere_settings()).unwrap();
    co.simulate_transport(&photon_sphere_settings()).unwrap();

    let tl_v = tl.tallies[0].get_mean()[0];
    let co_v = co.tallies[0].get_mean()[0];

    assert!(
        tl_v > 0.0,
        "{score_label}: expected non-zero track-length value"
    );
    assert!(
        co_v > 0.0,
        "{score_label}: expected non-zero collision value"
    );
    let rel = (tl_v - co_v).abs() / tl_v.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel < 0.10,
        "{score_label}: track-length {tl_v:.4e} vs collision {co_v:.4e} disagree (rel diff {rel:.3e})"
    );
}

#[test]
fn collision_photon_xs_coherent_matches_track_length() {
    check_photon_score_parity(
        Score::PhotonXS(PhotonXSScore {
            component: PhotonComponent::Coherent,
        }),
        1.0e6, // 1 MeV -- coherent + incoherent + photoelectric all open
        "PhotonXS::Coherent",
    );
}

#[test]
fn collision_photon_xs_incoherent_matches_track_length() {
    check_photon_score_parity(
        Score::PhotonXS(PhotonXSScore {
            component: PhotonComponent::Incoherent,
        }),
        1.0e6,
        "PhotonXS::Incoherent",
    );
}

#[test]
fn collision_photon_xs_photoelectric_matches_track_length() {
    check_photon_score_parity(
        Score::PhotonXS(PhotonXSScore {
            component: PhotonComponent::Photoelectric,
        }),
        1.0e6,
        "PhotonXS::Photoelectric",
    );
}

#[test]
fn collision_photon_xs_pair_production_matches_track_length() {
    // Pair production has a 2·m_e c² ≈ 1.022 MeV threshold; pick 5 MeV
    // so the channel is open and non-trivial.
    check_photon_score_parity(
        Score::PhotonXS(PhotonXSScore {
            component: PhotonComponent::PairProduction,
        }),
        5.0e6,
        "PhotonXS::PairProduction",
    );
}

#[test]
fn collision_photon_flux_matches_track_length() {
    // Photon flux under Collision estimator was silently zero before
    // PhotonXS support landed (the pre-collision dispatch fell back on
    // the neutron MT-1 lookup, which returns 0 for photons). With the
    // photon-aware Σ_t in score_collision_event, photon flux now scores
    // under both estimators and must agree.
    check_photon_score_parity(Score::Flux(FluxScore), 1.0e6, "Flux (photon source)");
}
