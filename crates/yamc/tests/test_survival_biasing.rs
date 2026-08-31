//! Survival biasing (implicit capture) + weight-cutoff Russian roulette.
//!
//! With `survival_biasing` on, absorption never terminates a history:
//! every collision multiplies the particle weight by sigma_s / sigma_t
//! and the particle always scatters, with low-weight histories culled by
//! weight-cutoff Russian roulette. The biased game must estimate the
//! same physical means as the analog game (flux AND the analytically
//! scored MT-101 absorption rate), must not worsen the variance of a
//! deep tally at equal history count, and must stay bit-reproducible at
//! a fixed seed.

use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc::variance_reduction::{SurvivalBiasing, VarianceReduction};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Mt, ReactionRateScore, Score, Tally};
use yamc_tallies::CellFilter;

/// The standard opt-in: one default-parameter SurvivalBiasing entry.
fn survival_entry() -> Vec<VarianceReduction> {
    vec![VarianceReduction::SurvivalBiasing(
        SurvivalBiasing::default(),
    )]
}

fn li6_material(density: f64, id: u32) -> Arc<Material> {
    let mut material = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    material.set_material_id(id);
    material.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    Arc::new(material)
}

fn point_source(energy_ev: f64) -> ParticleSource {
    ParticleSource::Neutron(Source {
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

fn sphere_surface(id: usize, radius: f64, boundary: BoundaryType) -> Arc<Surface> {
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

/// Single Li6 sphere (radius 10 cm, 2.0 g/cm3) with a central 1 MeV
/// point source and two cell tallies: track-length flux and the MT-101
/// absorption rate. Absorption-dominated enough that survival biasing
/// changes the game materially.
fn build_sphere_model(
    survival_biasing: bool,
    particles: usize,
    seed: u64,
) -> (Model, TransportSettings) {
    let surf = sphere_surface(1, 10.0, BoundaryType::Vacuum);
    let region = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            surf.clone(),
        )))),
    };
    let mat = li6_material(2.0, 1);
    let cell = Cell::new(Some(1), region, Some("sphere".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![mat]).unwrap();

    let cell_filter = Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap()));
    let mut flux = Tally::new();
    flux.filters.push(cell_filter.clone());
    flux.scores = vec![Score::Flux(FluxScore)];
    flux.name = Some("flux".to_string());

    let mut absorption = Tally::new();
    absorption.filters.push(cell_filter);
    absorption.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    absorption.name = Some("absorption".to_string());

    let mut model = Model::new(
        geometry,
        vec![point_source(1.0e6)],
        vec![Arc::new(flux), Arc::new(absorption)],
    );
    if survival_biasing {
        model.variance_reduction = survival_entry();
    }
    let settings = TransportSettings {
        total_particles: Some(particles),
        seed,
        ..Default::default()
    };
    (model, settings)
}

/// Two concentric Li6 spheres: a dense absorbing inner sphere (r = 8 cm,
/// 2.0 g/cm3) and a thin outer shell (8..10 cm, 0.46 g/cm3) whose flux
/// is the deep tally. Few analog histories survive into the shell;
/// survival biasing carries weight through instead.
fn build_deep_model(
    survival_biasing: bool,
    particles: usize,
    seed: u64,
) -> (Model, TransportSettings) {
    let inner = sphere_surface(1, 8.0, BoundaryType::Transmission);
    let outer = sphere_surface(2, 10.0, BoundaryType::Vacuum);
    let region1 = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            inner.clone(),
        )))),
    };
    let region2 = Region {
        expr: RegionExpr::Intersection(
            Box::new(RegionExpr::Halfspace(HalfspaceType::Above(inner.clone()))),
            Box::new(RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                HalfspaceType::Above(outer.clone()),
            )))),
        ),
    };
    let mat1 = li6_material(2.0, 1);
    let mat2 = li6_material(0.46, 2);
    let cell1 = Cell::new(Some(1), region1, Some("inner".to_string()), Some(0));
    let cell2 = Cell::new(Some(2), region2, Some("shell".to_string()), Some(1));
    let geometry = Geometry::new(vec![cell1, cell2.clone()], vec![mat1, mat2]).unwrap();

    let mut flux = Tally::new();
    flux.filters
        .push(Filter::Cell(CellFilter::from_id(cell2.cell_id.unwrap())));
    flux.scores = vec![Score::Flux(FluxScore)];
    flux.name = Some("shell flux".to_string());

    let mut model = Model::new(geometry, vec![point_source(1.0e6)], vec![Arc::new(flux)]);
    if survival_biasing {
        model.variance_reduction = survival_entry();
    }
    let settings = TransportSettings {
        total_particles: Some(particles),
        seed,
        ..Default::default()
    };
    (model, settings)
}

#[test]
fn survival_matches_analog_within_statistics() {
    // Same problem, same seed, only the capture treatment differs. Both
    // games estimate the same flux and the same MT-101 absorption rate,
    // so the means must agree to a few percent at this history budget.
    // The absorption-rate agreement is the key check: under implicit
    // capture no history ever terminates by sampling MT 101, yet the
    // track-length estimator must still reproduce the analog rate.
    let (mut analog, analog_settings) = build_sphere_model(false, 20_000, 0xABCD);
    let (mut survival, survival_settings) = build_sphere_model(true, 20_000, 0xABCD);
    analog.simulate_transport(&analog_settings).unwrap();
    survival.simulate_transport(&survival_settings).unwrap();

    for t in 0..2 {
        let name = analog.tallies[t].name.clone().unwrap();
        let a = analog.tallies[t].get_mean()[0];
        let s = survival.tallies[t].get_mean()[0];
        assert!(a > 0.0, "{name}: analog mean must be positive (got {a})");
        assert!(s > 0.0, "{name}: survival mean must be positive (got {s})");
        let rel = (a - s).abs() / a;
        assert!(
            rel < 0.05,
            "{name}: analog {a:.4e} vs survival {s:.4e} disagree (rel {rel:.3e})"
        );
    }
}

#[test]
fn survival_does_not_worsen_deep_tally_variance() {
    // At equal history count the shell flux behind 8 cm of dense Li6
    // must come out at least as precise with survival biasing: histories
    // are never cut short by capture, so every history can contribute
    // track length in the shell. Allow 5% slack so the assertion tests
    // the mechanism rather than RNG luck; runs are seeded so the result
    // is deterministic.
    let (mut analog, analog_settings) = build_deep_model(false, 20_000, 0xBEEF);
    let (mut survival, survival_settings) = build_deep_model(true, 20_000, 0xBEEF);
    analog.simulate_transport(&analog_settings).unwrap();
    survival.simulate_transport(&survival_settings).unwrap();

    let a_mean = analog.tallies[0].get_mean()[0];
    let s_mean = survival.tallies[0].get_mean()[0];
    assert!(a_mean > 0.0 && s_mean > 0.0);
    let rel = (a_mean - s_mean).abs() / a_mean;
    assert!(
        rel < 0.10,
        "shell flux: analog {a_mean:.4e} vs survival {s_mean:.4e} disagree (rel {rel:.3e})"
    );

    let a_err = analog.tallies[0].get_rel_error()[0];
    let s_err = survival.tallies[0].get_rel_error()[0];
    assert!(
        s_err <= a_err * 1.05,
        "survival rel error {s_err:.4e} worse than analog {a_err:.4e}"
    );
}

#[test]
fn survival_reproducible_at_fixed_seed() {
    // Roulette and the survival reweighting must be deterministic: two
    // identical seeded runs give the same tallies. ULP-level relative
    // tolerance because multi-threaded f64 accumulation order is not
    // fixed (same convention as tests/reproducibility.rs); a roulette or
    // RNG-stream nondeterminism would show up as a gross difference, not
    // a last-ulp one.
    let (mut a, a_settings) = build_sphere_model(true, 5_000, 7);
    let (mut b, b_settings) = build_sphere_model(true, 5_000, 7);
    a.simulate_transport(&a_settings).unwrap();
    b.simulate_transport(&b_settings).unwrap();
    for t in 0..2 {
        let av = a.tallies[t].get_mean()[0];
        let bv = b.tallies[t].get_mean()[0];
        let rel = (av - bv).abs() / av.abs().max(f64::MIN_POSITIVE);
        assert!(
            rel < 1e-12,
            "same-seed survival runs disagree: {av} vs {bv} (rel {rel:.3e})"
        );
    }
}

#[test]
fn survival_rejects_invalid_configuration() {
    // weight_cutoff must be positive.
    let (mut m, settings) = build_sphere_model(false, 10, 1);
    m.variance_reduction = vec![VarianceReduction::SurvivalBiasing(SurvivalBiasing {
        weight_cutoff: 0.0,
        weight_survive: 1.0,
    })];
    let err = m.simulate_transport(&settings).unwrap_err();
    assert!(
        err.contains("weight_cutoff > 0"),
        "unexpected error message: {err}"
    );

    // A survivor weight below the cutoff would re-trigger roulette
    // forever; rejected up front.
    let (mut m, settings) = build_sphere_model(false, 10, 1);
    m.variance_reduction = vec![VarianceReduction::SurvivalBiasing(SurvivalBiasing {
        weight_cutoff: 0.5,
        weight_survive: 0.25,
    })];
    let err = m.simulate_transport(&settings).unwrap_err();
    assert!(
        err.contains("weight_survive >= weight_cutoff"),
        "unexpected error message: {err}"
    );

    // Duplicate SurvivalBiasing entries are meaningless.
    let (mut m, settings) = build_sphere_model(false, 10, 1);
    m.variance_reduction = vec![
        VarianceReduction::SurvivalBiasing(SurvivalBiasing::default()),
        VarianceReduction::SurvivalBiasing(SurvivalBiasing::default()),
    ];
    let err = m.simulate_transport(&settings).unwrap_err();
    assert!(
        err.contains("at most one SurvivalBiasing"),
        "unexpected error message: {err}"
    );
}
