use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
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
use yamc_tallies::CellFilter;

fn main() {
    // Create two-cell geometry: inner sphere (Li6) and outer annular region (Be9)
    let sphere1 = Arc::new(Surface {
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

    let sphere2 = Arc::new(Surface {
        surface_id: Some(2),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 200.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });

    // Create regions
    // region1: -sphere1 (inside sphere1)
    let region1 = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            sphere1.clone(),
        )))),
    };

    // region2: +sphere1 & -sphere2 (between sphere1 and sphere2)
    let region2 = Region {
        expr: RegionExpr::Intersection(
            Box::new(RegionExpr::Halfspace(HalfspaceType::Above(sphere1.clone()))),
            Box::new(RegionExpr::Complement(Box::new(RegionExpr::Halfspace(
                HalfspaceType::Above(sphere2.clone()),
            )))),
        ),
    };

    // Create materials with different absorption characteristics
    let mut material1 = Material::new(
        HashMap::from([
            ("Li6".into(), 0.07 / 2.0), // Li4SiO4
            ("Li7".into(), 0.93 / 2.0),
            ("Be9".into(), 0.5),
        ]),
        "atom",
        "g/cm3",
        Some(2.0),
    )
    .unwrap();
    material1.set_material_id(1); // Set material_id for MaterialFilter testing
    material1.set_temperature("294");

    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Be9".to_string(), "tests/Be9.arrow".to_string());
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    nuclide_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
    material1.read_nuclear_data(&nuclide_map, None).unwrap();

    let mat_arc = Arc::new(material1);

    // Create cells
    let cell1 = Cell::new(
        Some(1),
        region1,
        Some("inner_sphere".to_string()),
        None, // No material fill
    );

    let cell2 = Cell::new(Some(2), region2, Some("outer_annular".to_string()), Some(0));

    let geometry = Geometry::new(vec![cell1.clone(), cell2.clone()], vec![mat_arc]).unwrap();

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

    let particles = 50000;
    let batches = 2;

    // Create tallies with CellFilters
    let cell_filter2 = Filter::Cell(CellFilter::from_id(cell2.cell_id.unwrap()));
    let mut tally1 = Tally::new();
    tally1.filters = vec![cell_filter2];
    tally1.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
        105,
    )))]; // n,t (tritium production)
    tally1.name = Some("tbr".to_string());

    // Initialize batch data
    tally1.initialize_batches(batches);

    let tallies = vec![Arc::new(tally1)];

    let mut model = Model::new(geometry, vec![source], tallies);

    let start = Instant::now();
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles * batches),
            ..Default::default()
        })
        .unwrap();
    let elapsed = start.elapsed();

    // Check if we're the root rank (works with or without MPI)
    let is_root = yamc::mpi_context::mpi_rank() == 0;

    // Only root rank prints the results
    if is_root {
        println!(
            "Simulation completed in {:.6} seconds.",
            elapsed.as_secs_f64()
        );

        // Tallies are updated in place!
        let mean = model.tallies[0].get_mean();
        println!("TBR (tritium breeding ratio): {}", mean[0]);
    }

    // Properly finalize MPI to avoid spurious error messages
    yamc::mpi_context::mpi_finalize();
}
