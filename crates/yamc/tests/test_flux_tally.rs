// Test flux tallying functionality
use std::collections::HashMap;
use std::str::FromStr;
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
use yamc_tallies::tally::{FluxScore, Mt, ReactionRateScore, Score, Tally};

#[test]
fn test_score_enum() {
    // Test Score enum conversions
    let _flux_score = Score::Flux(FluxScore);

    let mt_score = Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(101)));
    assert_eq!(mt_score.to_i32(), 101);

    // Test from_str
    let _parsed_flux = Score::from_str("flux").unwrap();
    let invalid_result = Score::from_str("invalid");
    assert!(invalid_result.is_err());
}

#[test]
fn test_flux_tally_simulation() {
    // Create a simple sphere geometry
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

    // Create material with Li6
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

    // Create cell
    let cell = Cell::new(Some(1), region, Some("test_cell".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![mat_arc]).unwrap();

    // Create source: point source at origin with 14.1 MeV
    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });

    // Create flux tally using Score enum
    let mut flux_tally = Tally::new();
    flux_tally.set_scores_mixed(vec![Score::Flux(FluxScore)]);
    flux_tally.name = Some("Flux Tally".to_string());

    // Create model and run
    let mut model = Model::new(geometry, vec![source], vec![Arc::new(flux_tally)]);
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(1000 * 10),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    // Check results
    let tally_result = &model.tallies[0];

    // Flux should be positive
    let mean_flux = tally_result.get_mean()[0];
    assert!(
        mean_flux > 0.0,
        "Flux should be positive, got {}",
        mean_flux
    );

    // Should have proper batch count
    use std::sync::atomic::Ordering;
    assert_eq!(tally_result.n_batches.load(Ordering::Relaxed), 10);
    assert_eq!(
        tally_result.particles_per_chunk.load(Ordering::Relaxed),
        1000
    );
}

#[test]
fn test_flux_and_reaction_mixed_tally() {
    // Create a simple sphere geometry
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

    // Create material with Li6
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

    // Create cell
    let cell = Cell::new(Some(1), region, Some("test_cell".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![mat_arc]).unwrap();

    // Create source
    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });

    // Create tally with both flux and absorption
    let mut tally = Tally::new();
    tally.set_scores_mixed(vec![
        Score::Flux(FluxScore),
        Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(101))),
    ]);
    tally.name = Some("Mixed Tally".to_string());

    // Create model and run
    let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(1000 * 10),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    // Check results
    let tally_result = &model.tallies[0];

    let means = tally_result.get_mean();
    assert_eq!(means.len(), 2);

    // Flux (index 0) should be positive
    assert!(means[0] > 0.0, "Flux should be positive, got {}", means[0]);

    // Absorption (index 1) should be non-negative
    assert!(
        means[1] >= 0.0,
        "Absorption should be non-negative, got {}",
        means[1]
    );
}

#[test]
fn test_set_scores_mixed_reinitializes_storage() {
    // Create tally and set scores
    let mut tally = Tally::new();

    // Initial set
    tally.set_scores_mixed(vec![Score::ReactionRate(ReactionRateScore::from_mt(
        Mt::new(101),
    ))]);
    assert_eq!(tally.scores.len(), 1);
    assert_eq!(tally.num_bins(), 1);

    // Change scores - should reinitialize all storage
    tally.set_scores_mixed(vec![
        Score::Flux(FluxScore),
        Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(101))),
        Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(2))),
    ]);
    assert_eq!(tally.scores.len(), 3);
    assert_eq!(tally.num_bins(), 3);

    // Verify actual score values
    assert_eq!(
        tally.scores,
        vec![
            Score::Flux(FluxScore),
            Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(101))),
            Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(2))),
        ]
    );
}
