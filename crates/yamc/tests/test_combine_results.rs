//! End-to-end tests for combining results of independent runs.
//!
//! These run real Li6 transport and exercise the full chain: provenance
//! construction (fingerprint + data libraries), the exact pooled Welford
//! merge, the seed/fingerprint guards, the tallies-do-not-perturb
//! property behind pass-through semantics, and the lossless Arrow round
//! trip. Statistics-level unit tests live in yamc-tallies::combine.

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
use yamc_tallies::filter::Filter;
use yamc_tallies::simulation_results::{RunProvenance, SimulationResults};
use yamc_tallies::tally::{FluxScore, Mt, ReactionRateScore, Score, Tally};
use yamc_tallies::CellFilter;

fn li6_material(density: f64) -> Arc<Material> {
    let mut material = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    Arc::new(material)
}

/// Li6 sphere with a 1 MeV point source. `with_absorption` adds a second
/// (MT-101 reaction rate) tally so tally-set differences can be tested.
fn build_model(
    density: f64,
    particles: usize,
    seed: u64,
    with_absorption: bool,
) -> (Model, TransportSettings) {
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
    let cell = Cell::new(Some(1), region, Some("sphere".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![li6_material(density)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });

    let cell_filter = Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap()));
    let mut flux = Tally::new();
    flux.filters.push(cell_filter.clone());
    flux.scores = vec![Score::Flux(FluxScore)];
    flux.name = Some("flux".to_string());
    let mut tallies = vec![Arc::new(flux)];
    if with_absorption {
        let mut absorption = Tally::new();
        absorption.filters.push(cell_filter);
        absorption.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
            101,
        )))];
        absorption.name = Some("absorption".to_string());
        tallies.push(Arc::new(absorption));
    }

    let model = Model::new(geometry, vec![point_clone(&source)], tallies);
    let settings = TransportSettings {
        total_particles: Some(particles),
        seed,
        threads: Some(1), // single-threaded (deterministic), matching run_to_results
        ..Default::default()
    };
    (model, settings)
}

fn point_clone(s: &ParticleSource) -> ParticleSource {
    s.clone()
}

/// Run single-threaded (deterministic) and wrap the finalized tallies in
/// a provenance-carrying `SimulationResults`, the same way the Python
/// `simulate_transport` path does.
fn run_to_results(built: (Model, TransportSettings)) -> SimulationResults {
    let (mut model, settings) = built;
    model.simulate_transport(&settings).unwrap();
    let run = RunProvenance {
        seed: settings.seed,
        n_histories: model
            .tallies
            .first()
            .map(|t| t.get_n_histories())
            .unwrap_or(0),
        elapsed_secs: 1.0, // fixed so FOM comparisons are deterministic
        fingerprint: model.fingerprint().unwrap(),
        data_libraries: model.data_libraries(),
        compute: "cpu".into(),
        yamc_version: "test".into(),
        mpi_size: 1,
        mpi_rank: 0,
    };
    SimulationResults::from_tallies_with_run(&model.tallies, 1.0, run).unwrap()
}

#[test]
fn merged_runs_agree_with_single_long_run() {
    let a = run_to_results(build_model(2.0, 8_000, 11, true));
    let b = run_to_results(build_model(2.0, 12_000, 22, true));
    let single = run_to_results(build_model(2.0, 20_000, 33, true));

    let (merged, warnings) = yamc_tallies::combine::combine_results(&[&a, &b]).unwrap();
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");

    for name in ["flux", "absorption"] {
        let m = merged.get_by_name(name).unwrap();
        let s = single.get_by_name(name).unwrap();
        assert_eq!(m.n_histories, 20_000, "{name}: pooled history count");

        // The pooled estimate and an independent 20k run estimate the
        // same mean; they must agree within their combined statistics.
        let combined_sigma =
            (m.standard_deviation[0].powi(2) + s.standard_deviation[0].powi(2)).sqrt();
        let diff = (m.mean[0] - s.mean[0]).abs();
        assert!(
            diff <= 4.0 * combined_sigma.max(1e-12),
            "{name}: merged {} vs single {} differ by {diff:.3e} (> 4 sigma = {:.3e})",
            m.mean[0],
            s.mean[0],
            4.0 * combined_sigma
        );

        // Pooling must tighten the error bars relative to each part.
        let a_std = a.get_by_name(name).unwrap().standard_deviation[0];
        let b_std = b.get_by_name(name).unwrap().standard_deviation[0];
        assert!(
            m.standard_deviation[0] < a_std && m.standard_deviation[0] < b_std,
            "{name}: merged std {} not below parts ({a_std}, {b_std})",
            m.standard_deviation[0]
        );
    }
}

/// The property behind pass-through semantics: tallies are observation
/// only. Same seed + same particle count with different tally sets must
/// give bit-identical statistics for the shared tally (single-threaded).
#[test]
fn tally_set_does_not_perturb_transport() {
    let flux_only = run_to_results(build_model(2.0, 5_000, 7, false));
    let flux_and_abs = run_to_results(build_model(2.0, 5_000, 7, true));

    let f1 = flux_only.get_by_name("flux").unwrap();
    let f2 = flux_and_abs.get_by_name("flux").unwrap();
    assert_eq!(f1.mean, f2.mean, "flux means must be bit-identical");
    assert_eq!(f1.m2, f2.m2, "flux m2 must be bit-identical");
    assert_eq!(f1.n_histories, f2.n_histories);
}

#[test]
fn pass_through_and_seed_guard_end_to_end() {
    // Different tally sets, different seeds: combinable with a warning.
    let a = run_to_results(build_model(2.0, 4_000, 7, true));
    let b = run_to_results(build_model(2.0, 4_000, 8, false));
    let (merged, warnings) = yamc_tallies::combine::combine_results(&[&a, &b]).unwrap();
    assert!(
        warnings.iter().any(|w| w.contains("'absorption'")),
        "expected pass-through warning, got {warnings:?}"
    );
    let flux = merged.get_by_name("flux").unwrap();
    assert_eq!(flux.n_histories, 8_000);
    let abs = merged.get_by_name("absorption").unwrap();
    assert_eq!(abs.n_histories, 4_000, "absorption is pass-through");
    assert_eq!(
        abs.mean,
        a.get_by_name("absorption").unwrap().mean,
        "pass-through statistics must be unchanged"
    );

    // Same seed: refused regardless of everything else matching.
    let c = run_to_results(build_model(2.0, 4_000, 7, true));
    let err = yamc_tallies::combine::combine_results(&[&a, &c]).unwrap_err();
    assert!(err.contains("share base seed 7"), "got: {err}");
}

#[test]
fn fingerprint_separates_physics_from_observation() {
    // Different density -> different physics -> refused.
    let a = run_to_results(build_model(2.0, 2_000, 1, true));
    let b = run_to_results(build_model(2.5, 2_000, 2, true));
    let err = yamc_tallies::combine::combine_results(&[&a, &b]).unwrap_err();
    assert!(err.contains("different models"), "got: {err}");

    // Same physics with different tallies, seed and particle count ->
    // identical fingerprint (observation-only knobs are excluded).
    let (m1, _) = build_model(2.0, 1_000, 1, true);
    let (m2, _) = build_model(2.0, 9_999, 42, false);
    assert_eq!(m1.fingerprint().unwrap(), m2.fingerprint().unwrap());

    // The data-library map names the Li6 fixture.
    let libs = m1.data_libraries();
    assert_eq!(
        libs.get("n:Li6").map(String::as_str),
        Some("tests/Li6.arrow")
    );
}

#[test]
fn arrow_roundtrip_is_lossless_and_combinable() {
    let a = run_to_results(build_model(2.0, 3_000, 5, true));
    let b = run_to_results(build_model(2.0, 3_000, 6, true));

    let dir = std::env::temp_dir();
    let path_a = dir.join(format!("yamc_combine_a_{}.arrow", std::process::id()));
    a.to_arrow(&path_a).unwrap();
    let loaded_a = SimulationResults::from_arrow(&path_a).unwrap();
    let _ = std::fs::remove_file(&path_a);

    // Lossless: numeric state, exact merge state, provenance and full
    // tally configs all survive bit-for-bit.
    assert_eq!(loaded_a.runs.len(), 1);
    assert_eq!(loaded_a.runs[0].seed, 5);
    assert_eq!(loaded_a.runs[0].fingerprint, a.runs[0].fingerprint);
    assert_eq!(
        loaded_a.runs[0].data_libraries.get("n:Li6"),
        a.runs[0].data_libraries.get("n:Li6")
    );
    for name in ["flux", "absorption"] {
        let orig = a.get_by_name(name).unwrap();
        let back = loaded_a.get_by_name(name).unwrap();
        assert_eq!(back.mean, orig.mean);
        assert_eq!(back.m2, orig.m2);
        assert_eq!(back.standard_deviation, orig.standard_deviation);
        assert_eq!(back.n_histories, orig.n_histories);
        assert_eq!(back.run_indices, orig.run_indices);
        assert_eq!(back.tally.scores.len(), orig.tally.scores.len());
        assert_eq!(back.tally.filters.len(), orig.tally.filters.len());
    }

    // A loaded result combines exactly like the in-memory one.
    let (merged_mem, _) = yamc_tallies::combine::combine_results(&[&a, &b]).unwrap();
    let (merged_load, _) = yamc_tallies::combine::combine_results(&[&loaded_a, &b]).unwrap();
    for name in ["flux", "absorption"] {
        assert_eq!(
            merged_mem.get_by_name(name).unwrap().mean,
            merged_load.get_by_name(name).unwrap().mean
        );
        assert_eq!(
            merged_mem.get_by_name(name).unwrap().m2,
            merged_load.get_by_name(name).unwrap().m2
        );
    }
}
