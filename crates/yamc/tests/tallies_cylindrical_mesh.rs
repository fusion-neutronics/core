// End-to-end tests for cylindrical mesh tallies: build a tally with a
// CylindricalMesh filter, run a small simulation, and verify the scoring path
// produces a correctly sized, sane result with per-ring volume normalisation.
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
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::CylindricalMesh;
use yamc_tallies::tally::{FluxScore, Score, Tally};

fn lithium_sphere() -> Geometry {
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 12.0,
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
    let cell = Cell::new(Some(1), region, Some("test_cell".to_string()), Some(0));
    Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap()
}

fn point_source() -> ParticleSource {
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

#[test]
fn cylindrical_mesh_tally_bin_count_and_sanity() {
    let geometry = lithium_sphere();

    // 5 radial rings x 4 azimuthal sectors x 5 z layers = 100 cells.
    let mesh = CylindricalMesh::uniform(
        [0.0, 0.0, 0.0],
        (0.0, 10.0),
        (0.0, std::f64::consts::TAU),
        (-10.0, 10.0),
        [5, 4, 5],
    );
    assert_eq!(mesh.num_bins(), 100);

    let mesh_filter = MeshFilter::new_cylindrical(mesh);
    assert_eq!(mesh_filter.num_bins(), 100);

    let mut flux_tally = Tally::new();
    flux_tally.filters.push(Filter::Mesh(mesh_filter));
    flux_tally.set_scores_mixed(vec![Score::Flux(FluxScore)]);
    flux_tally.name = Some("Cylindrical Flux Tally".to_string());
    assert_eq!(flux_tally.num_bins(), 100);

    let mut model = Model::new(geometry, vec![point_source()], vec![Arc::new(flux_tally)]);
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(2000),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    let result = &model.tallies[0];
    let mean = result.get_mean();
    assert_eq!(mean.len(), 100, "result must have one entry per cell");

    for (i, &v) in mean.iter().enumerate() {
        assert!((0.0..1e10).contains(&v), "bin {i} has bad flux {v}");
    }
    assert!(
        mean.iter().filter(|&&v| v > 0.0).count() > 0,
        "some cells should record flux"
    );
}

#[test]
fn cylindrical_per_ring_volume_normalisation_is_radially_flat() {
    // With a uniform medium and an isotropic point source at the centre, the
    // raw track-length flux scales with the (larger) volume of outer rings.
    // Dividing by the per-ring volume must remove that geometric bias, leaving
    // a volume-normalised flux that DECREASES with radius (1/r² attenuation) --
    // crucially, it must NOT increase with radius, which is the symptom of the
    // equal-volume bug this mesh is meant to avoid.
    let geometry = lithium_sphere();

    // Single z layer, single azimuthal bin: isolate the radial behaviour.
    let nr = 5;
    let mesh = CylindricalMesh::uniform(
        [0.0, 0.0, 0.0],
        (0.0, 10.0),
        (0.0, std::f64::consts::TAU),
        (-10.0, 10.0),
        [nr, 1, 1],
    );
    let mesh_filter = MeshFilter::new_cylindrical(mesh);

    let mut flux_tally = Tally::new();
    flux_tally.filters.push(Filter::Mesh(mesh_filter.clone()));
    flux_tally.set_scores_mixed(vec![Score::Flux(FluxScore)]);

    let mut model = Model::new(geometry, vec![point_source()], vec![Arc::new(flux_tally)]);
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(20000),
            seed: 7,
            ..Default::default()
        })
        .unwrap();

    let result = &model.tallies[0];
    let raw = result.get_mean();
    assert_eq!(raw.len(), nr);

    // Volume-normalised flux per ring.
    let norm: Vec<f64> = (0..nr)
        .map(|b| raw[b] / mesh_filter.get_element_volume(b))
        .collect();

    // Outer rings have larger volume, so raw flux should trend UP with radius
    // while the volume-normalised flux trends DOWN. Check the inner ring's
    // normalised flux exceeds the outer ring's (the volume bug would invert
    // this because the equal-volume assumption inflates outer rings).
    assert!(
        norm[0] > norm[nr - 1],
        "volume-normalised flux should fall with radius: inner {} vs outer {}",
        norm[0],
        norm[nr - 1]
    );
    // And the per-ring volumes really are increasing (sanity on the mesh).
    assert!(
        mesh_filter.get_element_volume(nr - 1) > mesh_filter.get_element_volume(0),
        "outer ring must have larger volume than inner ring"
    );
}
