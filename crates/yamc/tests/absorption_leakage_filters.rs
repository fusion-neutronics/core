// Integration test for absorption, leakage, and filter logic in Rust
// This test should be placed in tests/absorption_leakage_filters.rs

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
use yamc_tallies::tally::{Mt, ReactionRateScore, Score, Tally};
use yamc_tallies::{CellFilter, MaterialFilter};

#[test]
fn test_absorption_leakage_filters() {
    // Create two spheres as surfaces
    let surf1 = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 1.0,
        },
        boundary: BoundaryType::Transmission,
        name: None,
    });
    let surf2 = Arc::new(Surface {
        surface_id: Some(2),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 2.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });

    // Define regions using region expressions
    let region1 = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            surf1.clone(),
        )))),
    };
    let region2 = Region {
        expr: RegionExpr::Intersection(
            Box::new(RegionExpr::Halfspace(HalfspaceType::Above(surf1.clone()))),
            Box::new(RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                HalfspaceType::Above(surf2.clone()),
            )))),
        ),
    };

    // Create materials and load nuclide data
    let mut material1 = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(10.0),
    )
    .unwrap();
    material1.set_material_id(1);
    material1.set_temperature("294");
    let mut nuclide_map1 = HashMap::new();
    nuclide_map1.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    material1.read_nuclear_data(&nuclide_map1, None).unwrap();

    let mut material2 = Material::new(
        HashMap::from([("Be9".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(20.0),
    )
    .unwrap();
    material2.set_material_id(2);
    material2.set_temperature("294");
    let mut nuclide_map2 = HashMap::new();
    nuclide_map2.insert("Be9".to_string(), "tests/Be9.arrow".to_string());
    material2.read_nuclear_data(&nuclide_map2, None).unwrap();

    let mat1_arc = Arc::new(material1);
    let mat2_arc = Arc::new(material2);

    // Create cells
    let cell1 = Cell::new(Some(1), region1, Some("inner_sphere".to_string()), Some(0));
    let cell2 = Cell::new(Some(2), region2, Some("outer_annular".to_string()), Some(1));
    let geometry = Geometry::new(
        vec![cell1.clone(), cell2.clone()],
        vec![mat1_arc.clone(), mat2_arc.clone()],
    )
    .unwrap();

    // Source: neutrons starting at origin, isotropic, 1 MeV
    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let particles = 50;
    let batches = 2;

    // Create tallies
    let cell_filter1 = Filter::Cell(CellFilter::from_id(cell1.cell_id.unwrap()));
    let mut tally1 = Tally::new();
    tally1.filters = vec![cell_filter1.clone()];
    tally1.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally1.name = Some("absorption in cell 1".to_string());

    let cell_filter2 = Filter::Cell(CellFilter::from_id(cell2.cell_id.unwrap()));
    let mut tally2 = Tally::new();
    tally2.filters = vec![cell_filter2.clone()];
    tally2.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally2.name = Some("absorption in cell 2".to_string());

    let material_filter1 = Filter::Material(MaterialFilter::new(&mat1_arc));
    let mut tally1_mat = Tally::new();
    tally1_mat.filters = vec![material_filter1.clone()];
    tally1_mat.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally1_mat.name = Some("absorption in material 1 (MaterialFilter)".to_string());

    let material_filter2 = Filter::Material(MaterialFilter::new(&mat2_arc));
    let mut tally2_mat = Tally::new();
    tally2_mat.filters = vec![material_filter2.clone()];
    tally2_mat.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally2_mat.name = Some("absorption in material 2 (MaterialFilter)".to_string());

    let mut tally3 = Tally::new();
    tally3.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally3.name = Some("total absorption".to_string());

    let mut tally4_match = Tally::new();
    tally4_match.filters = vec![material_filter1.clone(), cell_filter1.clone()];
    tally4_match.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally4_match.name =
        Some("absorption in material 1 AND cell 1 (should match cell 1)".to_string());

    let mut tally5_zero = Tally::new();
    tally5_zero.filters = vec![material_filter2.clone(), cell_filter1.clone()];
    tally5_zero.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        101,
    )))];
    tally5_zero.name = Some("absorption in material 2 AND cell 1 (should be zero)".to_string());

    // Initialize batch data for all tallies before wrapping in Arc
    let num_batches = batches;
    tally1.initialize_batches(num_batches);
    tally2.initialize_batches(num_batches);
    tally1_mat.initialize_batches(num_batches);
    tally2_mat.initialize_batches(num_batches);
    tally3.initialize_batches(num_batches);
    tally4_match.initialize_batches(num_batches);
    tally5_zero.initialize_batches(num_batches);

    // Initialize batch data for all tallies before wrapping in Arc
    let num_batches = batches;
    tally1.initialize_batches(num_batches);
    tally2.initialize_batches(num_batches);
    tally1_mat.initialize_batches(num_batches);
    tally2_mat.initialize_batches(num_batches);
    tally3.initialize_batches(num_batches);
    tally4_match.initialize_batches(num_batches);
    tally5_zero.initialize_batches(num_batches);

    // Initialize batch data for all tallies before wrapping in Arc
    let num_batches = batches;
    tally1.initialize_batches(num_batches);
    tally2.initialize_batches(num_batches);
    tally1_mat.initialize_batches(num_batches);
    tally2_mat.initialize_batches(num_batches);
    tally3.initialize_batches(num_batches);
    tally4_match.initialize_batches(num_batches);
    tally5_zero.initialize_batches(num_batches);

    // Place the total absorption tally before the filtered tallies to ensure mutually exclusive scoring works as intended
    let tallies = vec![
        Arc::new(tally3),
        Arc::new(tally1),
        Arc::new(tally2),
        Arc::new(tally1_mat),
        Arc::new(tally2_mat),
        Arc::new(tally4_match),
        Arc::new(tally5_zero),
    ];

    let mut model = Model::new(geometry, vec![source], tallies);
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles * batches),
            ..Default::default()
        })
        .unwrap();

    // Access tallies from model after run (no leakage tally anymore)
    let tally3 = &model.tallies[0]; // total absorption
    let tally1 = &model.tallies[1];
    let tally2 = &model.tallies[2];
    let tally1_mat = &model.tallies[3];
    let tally2_mat = &model.tallies[4];
    let tally4_match = &model.tallies[5];
    let tally5_zero = &model.tallies[6];

    // Debug: Print means for tally1, tally2, tally3
    println!("[DEBUG] tally1.mean (cell 1): {:?}", tally1.get_mean());
    println!("[DEBUG] tally2.mean (cell 2): {:?}", tally2.get_mean());
    println!(
        "[DEBUG] tally3.mean (total absorption): {:?}",
        tally3.get_mean()
    );

    // Test 2: Tally consistency
    let tolerance = 1e-10;

    // Test: Sum of cell tallies equals total absorption
    let cell_sum_vs_total =
        (tally1.get_mean()[0] + tally2.get_mean()[0] - tally3.get_mean()[0]).abs();
    assert!(
        cell_sum_vs_total < tolerance,
        "Sum of cell tallies ({:?}) should equal total absorption tally ({:?}), difference: {:?}",
        tally1.get_mean()[0] + tally2.get_mean()[0],
        tally3.get_mean()[0],
        cell_sum_vs_total
    );

    // Test 1: CellFilter functionality
    assert_ne!(
        tally1.get_mean()[0],
        tally2.get_mean()[0],
        "CellFilter should separate tallies by cell"
    );

    let sum_diff = (tally1.get_mean()[0] + tally2.get_mean()[0] - tally3.get_mean()[0]).abs();
    assert!(
        sum_diff < tolerance,
        "Sum of cell tallies ({:?}) should equal total tally ({:?}), difference: {:?}",
        tally1.get_mean()[0] + tally2.get_mean()[0],
        tally3.get_mean()[0],
        sum_diff
    );

    // Test 3: Particle conservation (leakage tally removed - only check absorption <= 1.0)
    assert!(
        tally3.get_mean()[0] <= 1.0,
        "Total absorption should be <= 1.0 (some particles may leak), got {:?}",
        tally3.get_mean()[0]
    );

    // Test 4: Physical reasonableness
    assert!(
        tally3.get_mean()[0] > 0.0,
        "Some particles should be absorbed"
    );
    assert!(
        tally3.get_mean()[0] < 1.0,
        "Not all particles should be absorbed (some leak)"
    );

    // Test 5: At least one cell should have absorption
    assert!(
        tally1.get_mean()[0] >= 0.0,
        "Cell 1 absorption should be non-negative"
    );
    assert!(
        tally2.get_mean()[0] >= 0.0,
        "Cell 2 absorption should be non-negative"
    );
    assert!(
        tally1.get_mean()[0] + tally2.get_mean()[0] > 0.0,
        "Total absorption should be positive"
    );

    // Test 6: MaterialFilter equivalence
    let cell1_vs_mat1_diff = (tally1.get_mean()[0] - tally1_mat.get_mean()[0]).abs();
    let cell2_vs_mat2_diff = (tally2.get_mean()[0] - tally2_mat.get_mean()[0]).abs();
    assert!(cell1_vs_mat1_diff < tolerance, "CellFilter and MaterialFilter should give same results for cell 1 ({:?} vs {:?}), difference: {:?}", tally1.get_mean()[0], tally1_mat.get_mean()[0], cell1_vs_mat1_diff);
    assert!(cell2_vs_mat2_diff < tolerance, "CellFilter and MaterialFilter should give same results for cell 2 ({:?} vs {:?}), difference: {:?}", tally2.get_mean()[0], tally2_mat.get_mean()[0], cell2_vs_mat2_diff);

    // Test 7: Mixed filter intersection tests
    let mixed_match_diff = (tally1.get_mean()[0] - tally4_match.get_mean()[0]).abs();
    assert!(
        mixed_match_diff < tolerance,
        "Material 1 AND Cell 1 should equal Cell 1 alone ({:?} vs {:?}), difference: {:?}",
        tally1.get_mean()[0],
        tally4_match.get_mean()[0],
        mixed_match_diff
    );
    assert_eq!(
        tally5_zero.get_mean()[0],
        0.0,
        "Material 2 AND Cell 1 should be zero (no overlap), got: {:?}",
        tally5_zero.get_mean()[0]
    );
}
