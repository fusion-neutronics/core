// Integration test for reproducibility - verifies that simulations with the same seed produce identical results

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
use yamc_tallies::tally::{Mt, ReactionRateScore, Score, Tally};

#[test]
fn test_separate_vs_multi_score_tallies_equivalence() {
    // Setup: geometry, material, source, settings
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 2.0,
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
        Some(10.0),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nuclide_map = std::collections::HashMap::new();
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    let mat_arc = Arc::new(material);
    let cell = Cell::new(Some(1), region, Some("test_cell".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![mat_arc]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let particles = 100;
    let batches = 10;
    let num_batches = batches;
    let settings = TransportSettings {
        total_particles: Some(particles * batches),
        seed: 12345,
        ..Default::default()
    };

    // --- Two tallies, one score each ---
    let mut tally_a = Tally::new();
    tally_a.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally_a.name = Some("absorption_101".to_string());
    tally_a.initialize_batches(num_batches);

    let mut tally_b = Tally::new();
    tally_b.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        102,
    )))];
    tally_b.name = Some("absorption_102".to_string());
    tally_b.initialize_batches(num_batches);

    let mut model_sep = Model::new(
        geometry.clone(),
        vec![source.clone()],
        vec![Arc::new(tally_a), Arc::new(tally_b)],
    );
    model_sep.simulate_transport(&settings).unwrap();
    let mean_a = model_sep.tallies[0].get_mean()[0];
    let mean_b = model_sep.tallies[1].get_mean()[0];

    // --- One tally, two scores ---
    let mut tally_multi = Tally::new();
    tally_multi.scores = vec![
        Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(101))),
        Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(102))),
    ];
    tally_multi.name = Some("absorption_101_102".to_string());
    tally_multi.initialize_batches(num_batches);

    let mut model_multi = Model::new(
        geometry.clone(),
        vec![source.clone()],
        vec![Arc::new(tally_multi)],
    );
    model_multi.simulate_transport(&settings).unwrap();
    let means_multi = &model_multi.tallies[0].get_mean();

    assert!(
        (mean_a - means_multi[0]).abs() < 1e-12,
        "Separate tally 101 and multi-score tally[0] should match"
    );
    assert!(
        (mean_b - means_multi[1]).abs() < 1e-12,
        "Separate tally 102 and multi-score tally[1] should match"
    );

    // --- Three tallies, one score each ---

    // Create fresh tallies for three-tally test
    let mut tally_a3 = Tally::new();
    tally_a3.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally_a3.name = Some("absorption_101".to_string());
    tally_a3.initialize_batches(num_batches);

    let mut tally_b3 = Tally::new();
    tally_b3.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        102,
    )))];
    tally_b3.name = Some("absorption_102".to_string());
    tally_b3.initialize_batches(num_batches);

    let mut tally_c = Tally::new();
    tally_c.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        103,
    )))];
    tally_c.name = Some("absorption_103".to_string());
    tally_c.initialize_batches(num_batches);

    let mut model_sep3 = Model::new(
        geometry.clone(),
        vec![source.clone()],
        vec![Arc::new(tally_a3), Arc::new(tally_b3), Arc::new(tally_c)],
    );
    model_sep3.simulate_transport(&settings).unwrap();
    let mean_a3 = model_sep3.tallies[0].get_mean()[0];
    let mean_b3 = model_sep3.tallies[1].get_mean()[0];
    let mean_c3 = model_sep3.tallies[2].get_mean()[0];

    // --- One tally, three scores ---
    let mut tally_multi3 = Tally::new();
    tally_multi3.scores = vec![
        Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(101))),
        Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(102))),
        Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(103))),
    ];
    tally_multi3.name = Some("absorption_101_102_103".to_string());
    tally_multi3.initialize_batches(num_batches);

    let mut model_multi3 = Model::new(
        geometry.clone(),
        vec![source.clone()],
        vec![Arc::new(tally_multi3)],
    );
    model_multi3.simulate_transport(&settings).unwrap();
    let means_multi3 = &model_multi3.tallies[0].get_mean();

    assert!(
        (mean_a3 - means_multi3[0]).abs() < 1e-12,
        "Separate tally 101 and multi-score tally[0] should match"
    );
    assert!(
        (mean_b3 - means_multi3[1]).abs() < 1e-12,
        "Separate tally 102 and multi-score tally[1] should match"
    );
    assert!(
        (mean_c3 - means_multi3[2]).abs() < 1e-12,
        "Separate tally 103 and multi-score tally[2] should match"
    );

    println!("✓ Separate vs multi-score tally equivalence test passed!");
}

#[test]
fn test_reproducibility_with_same_seed() {
    // Create geometry
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 2.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            surf.clone(),
        )))),
    };

    // Create material
    let mut material = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(10.0),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();

    let mat_arc = Arc::new(material);
    let cell = Cell::new(Some(1), region, Some("test_cell".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![mat_arc]).unwrap();

    // Source
    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });

    let particles = 100;
    let batches = 10;
    let num_batches = batches;
    let settings = TransportSettings {
        total_particles: Some(particles * batches),
        seed: 42,
        ..Default::default()
    };

    // Create tallies
    let mut tally1 = Tally::new();
    tally1.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally1.name = Some("test_absorption_1".to_string());
    let mut tally2 = Tally::new();
    tally2.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally2.name = Some("test_absorption_2".to_string());
    let mut tally3 = Tally::new();
    tally3.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally3.name = Some("test_absorption_3".to_string());
    // Initialize batch data for all tallies before wrapping in Arc
    tally1.initialize_batches(num_batches);
    tally2.initialize_batches(num_batches);
    tally3.initialize_batches(num_batches);

    // Run simulation 1
    let mut model1 = Model::new(
        geometry.clone(),
        vec![source.clone()],
        vec![Arc::new(tally1)],
    );
    model1.simulate_transport(&settings).unwrap();

    // Run simulation 2 with same seed
    let mut model2 = Model::new(
        geometry.clone(),
        vec![source.clone()],
        vec![Arc::new(tally2)],
    );
    model2.simulate_transport(&settings).unwrap();

    // Run simulation 3 with same seed
    let mut model3 = Model::new(
        geometry.clone(),
        vec![source.clone()],
        vec![Arc::new(tally3)],
    );
    model3.simulate_transport(&settings).unwrap();

    // Verify all three runs produced identical results
    assert_eq!(
        model1.tallies.len(),
        model2.tallies.len(),
        "Should have same number of tallies"
    );
    assert_eq!(
        model1.tallies.len(),
        model3.tallies.len(),
        "Should have same number of tallies"
    );

    // Check absorption tally (index 0) - no leakage tally anymore
    // Track-length estimation uses f64 arithmetic, so we allow ULP-level tolerance
    let mean1 = model1.tallies[0].get_mean()[0];
    let mean2 = model2.tallies[0].get_mean()[0];
    let mean3 = model3.tallies[0].get_mean()[0];
    assert!(
        (mean1 - mean2).abs() < 1e-12,
        "Absorption should be nearly identical with same seed: {mean1} vs {mean2}"
    );
    assert!(
        (mean1 - mean3).abs() < 1e-12,
        "Absorption should be nearly identical with same seed: {mean1} vs {mean3}"
    );

    // Compare per-bin means across all bins (same-seed determinism)
    let bins1 = model1.tallies[0].get_mean();
    let bins2 = model2.tallies[0].get_mean();
    let bins3 = model3.tallies[0].get_mean();
    for i in 0..bins1.len() {
        assert!(
            (bins1[i] - bins2[i]).abs() < 1e-12,
            "Mean should be identical with same seed"
        );
        assert!(
            (bins1[i] - bins3[i]).abs() < 1e-12,
            "Mean should be identical with same seed"
        );
    }

    println!("✓ Reproducibility test passed!");
    println!(
        "  Run 1 - Absorption: {:?}",
        model1.tallies[0].get_mean()[0]
    );
    println!(
        "  Run 2 - Absorption: {:?}",
        model2.tallies[0].get_mean()[0]
    );
    println!(
        "  Run 3 - Absorption: {:?}",
        model3.tallies[0].get_mean()[0]
    );
}

#[test]
fn test_different_seeds_produce_different_results() {
    // Create geometry
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 2.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            surf.clone(),
        )))),
    };

    // Create material
    let mut material = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(10.0),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();

    let mat_arc = Arc::new(material);
    let cell = Cell::new(Some(1), region, Some("test_cell".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![mat_arc]).unwrap();

    // Source
    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });

    let particles = 100;
    let batches = 10;
    let num_batches = batches;

    // Create tallies
    let mut tally1 = Tally::new();
    tally1.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))]; // absorption
    tally1.name = Some("test_absorption_1".to_string());
    let mut tally2 = Tally::new();
    tally2.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))]; // absorption
    tally2.name = Some("test_absorption_2".to_string());
    // Initialize batch data for all tallies before wrapping in Arc
    tally1.initialize_batches(num_batches);
    tally2.initialize_batches(num_batches);

    // Run simulation with seed 42
    let mut model1 = Model::new(
        geometry.clone(),
        vec![source.clone()],
        vec![Arc::new(tally1)],
    );
    model1
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles * batches),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    // Run simulation with seed 123
    let mut model2 = Model::new(
        geometry.clone(),
        vec![source.clone()],
        vec![Arc::new(tally2)],
    );
    model2
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles * batches),
            seed: 123,
            ..Default::default()
        })
        .unwrap();

    // Verify different seeds produce different results (with high probability)
    // Note: In principle they could be equal by chance, but with 100 particles this is extremely unlikely
    let different_absorption = model1.tallies[0].get_mean()[0] != model2.tallies[0].get_mean()[0];

    // Check per-bin means across all bins for any difference
    let mut different_mean_data = false;
    {
        let mean1 = model1.tallies[0].get_mean();
        let mean2 = model2.tallies[0].get_mean();
        for i in 0..mean1.len() {
            println!("Mean 1[{}]: {:?}", i, mean1[i]);
            println!("Mean 2[{}]: {:?}", i, mean2[i]);
            if (mean1[i] - mean2[i]).abs() > 1e-12 {
                different_mean_data = true;
                break;
            }
        }
    }

    // For this test, it's possible (though unlikely) that different seeds produce the same mean
    // We'll accept the test passing if any bin's mean differs.
    // However, if all are identical, that indicates a problem with seeding
    println!(
        "Different absorption: {}, Different mean data: {}",
        different_absorption, different_mean_data
    );

    assert!(
        different_absorption || different_mean_data,
        "Different seeds should produce different results (absorption: {:?} vs {:?})",
        model1.tallies[0].get_mean()[0],
        model2.tallies[0].get_mean()[0]
    );

    println!("✓ Different seeds test passed!");
    println!(
        "  Seed 42  - Absorption: {:?}",
        model1.tallies[0].get_mean()[0]
    );
    println!(
        "  Seed 123 - Absorption: {:?}",
        model2.tallies[0].get_mean()[0]
    );
}
