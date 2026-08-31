/// Test Co58 reaction sampling at 3500 eV
use std::sync::Arc;
use yamc::model::{Model, TransportSettings};
use yamc::*;
use yamc_materials::Material;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score};
use yamc_tallies::{CellFilter, Tally};

fn main() {
    // Create material with Co58
    let mut material = Material::new(
        std::collections::HashMap::from([("Co58".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.0),
    )
    .expect("Failed to create material");
    material.material_id = Some(1);
    material.set_temperature("294");

    // Load the nuclear data
    material
        .load_nuclide_from_file("Co58", "tests/Co58.arrow")
        .expect("Failed to load Co58");

    // Create geometry - a sphere of Co58
    let sphere = Arc::new(Surface::sphere(
        0.0,
        0.0,
        0.0,
        100.0,
        Some(1),
        Some(BoundaryType::Vacuum),
    ));

    let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
    let cell = Cell::new(Some(1), region, Some("sphere".to_string()), Some(0));
    let geometry =
        Geometry::new(vec![cell], vec![Arc::new(material)]).expect("Failed to create geometry");

    // Create source at 3500 eV (monoenergetic, in URR range)
    let source_energy = 3500.0;
    let mut source = Source::new();
    source.energy = SourceEnergyDistribution::Discrete(
        Discrete::new(vec![source_energy], vec![1.0]).expect("Failed to create discrete dist"),
    );

    // Settings
    let n_particles = 100000;

    // Create tallies
    let mut tally = Tally::new();
    tally.name = Some("flux".to_string());
    tally.scores.push(Score::Flux(FluxScore));
    tally
        .filters
        .push(Filter::Cell(CellFilter { cell_ids: vec![1] }));
    let flux_tally = Arc::new(tally);

    // Create model
    let mut model = Model::new(
        geometry,
        vec![ParticleSource::Neutron(source)],
        vec![flux_tally.clone()],
    );

    println!("=== Running Co58 test at 3500 eV (in URR range 3000-25000 eV) ===");
    println!("Number of particles: {}", n_particles);

    // Run simulation
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(n_particles),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    // Get flux result
    let flux_mean = flux_tally.get_mean();
    let total_flux: f64 = flux_mean.iter().sum();
    println!("\nTotal flux: {:.4e}", total_flux);
    println!("Flux per particle: {:.4e}", total_flux / n_particles as f64);

    // Expected values from nuclear data at 3500 eV:
    // elastic: 32.7 b, n,gamma: 1.2 b, n,p: 4.4 b
    // P(absorption) should be ~14.8% but URR with absorption_flag=0 gives ~3.6%
    println!("\nExpected P(absorption):");
    println!("  Full (elastic + full_abs): 5.68 / 38.4 = 14.8%");
    println!("  URR (elastic + n_gamma): 1.23 / 33.9 = 3.6%");
}
