// Test mesh tally functionality
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
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::mesh::MeshFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::tally::{FluxScore, HeatingScore, Score, Tally};

/// Shared transport settings for the mesh-tally simulations: 100 histories at
/// a fixed seed.
fn mesh_settings() -> TransportSettings {
    TransportSettings {
        total_particles: Some(100),
        seed: 42,
        ..Default::default()
    }
}

#[test]
fn test_mesh_tally_bin_count_matches_dimensions() {
    // Create a simple sphere geometry
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

    // Create source: point source at origin
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

    // Create mesh with specific dimensions: 3 x 4 x 5 = 60 voxels
    let mesh = RegularRectangularMesh::new([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [3, 4, 5]);

    // Verify mesh properties
    assert_eq!(mesh.num_voxels(), 60, "Mesh should have 60 voxels (3*4*5)");

    let mesh_filter = MeshFilter::new(mesh);
    assert_eq!(
        mesh_filter.num_bins(),
        60,
        "MeshFilter should report 60 bins"
    );

    // Create flux tally with mesh filter
    let mut flux_tally = Tally::new();
    flux_tally.filters.push(Filter::Mesh(mesh_filter));
    flux_tally.set_scores_mixed(vec![Score::Flux(FluxScore)]);
    flux_tally.name = Some("Mesh Flux Tally".to_string());

    // Verify tally reports correct number of bins before simulation
    assert_eq!(
        flux_tally.num_bins(),
        60,
        "Tally should report 60 bins (1 score * 1 energy bin * 60 mesh bins)"
    );

    // Create model and run
    let mut model = Model::new(geometry, vec![source], vec![Arc::new(flux_tally)]);
    model.simulate_transport(&mesh_settings()).unwrap();

    // Check results - THIS IS THE CRITICAL TEST
    let tally_result = &model.tallies[0];

    let mean_flux = tally_result.get_mean();
    let std_dev = tally_result.get_std_dev();

    // CRITICAL: All result vectors must have exactly 60 elements
    assert_eq!(
        mean_flux.len(),
        60,
        "Mean flux vector should have 60 elements (3*4*5), got {}",
        mean_flux.len()
    );
    assert_eq!(
        std_dev.len(),
        60,
        "Std dev vector should have 60 elements (3*4*5), got {}",
        std_dev.len()
    );

    // All values should be reasonable (not garbage)
    for (i, &val) in mean_flux.iter().enumerate() {
        assert!(val >= 0.0, "Bin {} has negative flux: {}", i, val);
        assert!(val < 1e10, "Bin {} has garbage value: {}", i, val);
    }

    // At least some bins should have non-zero flux
    let non_zero_count = mean_flux.iter().filter(|&&v| v > 0.0).count();
    assert!(
        non_zero_count > 0,
        "At least some mesh bins should have non-zero flux"
    );
}

#[test]
fn test_mesh_tally_with_energy_filter() {
    // Create a simple sphere geometry
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

    // Create material
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
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    // Create mesh: 2x2x2 = 8 voxels
    let mesh = RegularRectangularMesh::new([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [2, 2, 2]);

    let mesh_filter = MeshFilter::new(mesh);
    assert_eq!(mesh_filter.num_bins(), 8);

    // Create energy filter: 3 energy bins
    let energy_filter = EnergyFilter::new(vec![0.0, 1e6, 10e6, 20e6]);
    assert_eq!(energy_filter.num_bins(), 3);

    // Create flux tally with both mesh and energy filters
    let mut flux_tally = Tally::new();
    flux_tally.filters.push(Filter::Mesh(mesh_filter));
    flux_tally.filters.push(Filter::Energy(energy_filter));
    flux_tally.set_scores_mixed(vec![Score::Flux(FluxScore)]);
    flux_tally.name = Some("Mesh+Energy Flux Tally".to_string());

    // Expected bins: 1 score * 3 energy bins * 8 mesh bins = 24
    assert_eq!(
        flux_tally.num_bins(),
        24,
        "Tally should report 24 bins (1 score * 3 energy * 8 mesh)"
    );

    // Create model and run
    let mut model = Model::new(geometry, vec![source], vec![Arc::new(flux_tally)]);
    model.simulate_transport(&mesh_settings()).unwrap();

    // Check results
    let tally_result = &model.tallies[0];

    let mean_flux = tally_result.get_mean();
    let std_dev = tally_result.get_std_dev();

    // CRITICAL: All result vectors must have exactly 24 elements
    assert_eq!(
        mean_flux.len(),
        24,
        "Mean flux vector should have 24 elements (3 energy * 8 mesh), got {}",
        mean_flux.len()
    );
    assert_eq!(
        std_dev.len(),
        24,
        "Std dev vector should have 24 elements (3 energy * 8 mesh), got {}",
        std_dev.len()
    );

    // All values should be reasonable
    for (i, &val) in mean_flux.iter().enumerate() {
        assert!(val >= 0.0, "Bin {} has negative flux: {}", i, val);
        assert!(val < 1e10, "Bin {} has garbage value: {}", i, val);
    }
}

#[test]
fn test_mesh_tally_multiple_scores() {
    // Create a simple sphere geometry
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

    // Create material
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
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    // Create mesh: 2x2x2 = 8 voxels
    let mesh = RegularRectangularMesh::new([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [2, 2, 2]);

    let mesh_filter = MeshFilter::new(mesh);

    // Create tally with 2 scores (flux and heating)
    let mut tally = Tally::new();
    tally.filters.push(Filter::Mesh(mesh_filter));
    tally.set_scores_mixed(vec![Score::Flux(FluxScore), Score::Heating(HeatingScore)]);
    tally.name = Some("Multi-Score Mesh Tally".to_string());

    // Expected bins: 2 scores * 1 energy bin * 8 mesh bins = 16
    assert_eq!(
        tally.num_bins(),
        16,
        "Tally should report 16 bins (2 scores * 1 energy * 8 mesh)"
    );

    // Create model and run
    let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
    model.simulate_transport(&mesh_settings()).unwrap();

    // Check results
    let tally_result = &model.tallies[0];

    let mean = tally_result.get_mean();

    // CRITICAL: Result vector must have exactly 16 elements
    assert_eq!(
        mean.len(),
        16,
        "Mean vector should have 16 elements (2 scores * 8 mesh), got {}",
        mean.len()
    );

    // All values should be reasonable
    for (i, &val) in mean.iter().enumerate() {
        assert!(val >= 0.0, "Bin {} has negative value: {}", i, val);
        assert!(val < 1e10, "Bin {} has garbage value: {}", i, val);
    }
}

#[test]
fn test_mesh_tally_non_cubic_dimensions() {
    // Test with non-cubic mesh dimensions (common source of indexing bugs)
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

    let cell = Cell::new(Some(1), region, Some("test_cell".to_string()), Some(0));
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

    // Create mesh with DIFFERENT dimensions in each direction: 2 x 3 x 7 = 42 voxels
    // This is a good test for proper indexing (Z-major ordering)
    let mesh = RegularRectangularMesh::new([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [2, 3, 7]);

    assert_eq!(mesh.num_voxels(), 42, "Mesh should have 42 voxels (2*3*7)");

    let mesh_filter = MeshFilter::new(mesh);

    let mut flux_tally = Tally::new();
    flux_tally.filters.push(Filter::Mesh(mesh_filter));
    flux_tally.set_scores_mixed(vec![Score::Flux(FluxScore)]);
    flux_tally.name = Some("Non-cubic Mesh Tally".to_string());

    assert_eq!(flux_tally.num_bins(), 42);

    let mut model = Model::new(geometry, vec![source], vec![Arc::new(flux_tally)]);
    model.simulate_transport(&mesh_settings()).unwrap();

    let tally_result = &model.tallies[0];
    let mean_flux = tally_result.get_mean();

    // CRITICAL: Must have exactly 42 elements
    assert_eq!(
        mean_flux.len(),
        42,
        "Mean flux vector should have 42 elements (2*3*7), got {}",
        mean_flux.len()
    );

    // All values should be reasonable
    for (i, &val) in mean_flux.iter().enumerate() {
        assert!(val >= 0.0, "Bin {} has negative flux: {}", i, val);
        assert!(val < 1e10, "Bin {} has garbage value: {}", i, val);
    }
}
