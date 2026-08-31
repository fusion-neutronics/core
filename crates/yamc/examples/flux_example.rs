use std::sync::Arc;
use std::time::Instant;
use yamc::model::{Model, TransportSettings};
use yamc::*;
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::{CellFilter, EnergyFilter};

fn main() {
    // Create a simple spherical geometry with Cr52 (like Python example)
    let sphere1 = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 1.0,
        },
        boundary: BoundaryType::Transmission,
        name: None,
    };

    let sphere2 = Surface {
        surface_id: Some(2),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 200.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };

    // Share a single Arc<Surface> for sphere1 between the two regions
    // -- region1 is "inside sphere1", region2 is "outside sphere1, inside
    // sphere2", and they meet on the same surface. Geometry::new()
    // dedups by Arc identity, so this avoids a duplicate-surface_id error.
    let sphere1_arc = Arc::new(sphere1);
    let region1 = Region::new_from_halfspace(HalfspaceType::Below(sphere1_arc.clone()));
    let region2_inner = Region::new_from_halfspace(HalfspaceType::Above(sphere1_arc));
    let region2_outer = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere2)));
    let region2 = region2_inner.intersection(&region2_outer);

    // Create Cr52 material (same as Python flux.py example)
    let mut material = Material::new(
        std::collections::HashMap::from([("Cr52".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(2.0),
    )
    .unwrap();
    material.set_temperature("294");
    let mut nuclide_json_map = std::collections::HashMap::new();
    nuclide_json_map.insert("Cr52".to_string(), "tests/Cr52.arrow".to_string());
    material.read_nuclear_data(&nuclide_json_map, None).unwrap();
    material.calculate_macroscopic_xs(&vec![1], true);
    let material_arc = Arc::new(material);

    let cell1 = Cell::new(Some(1), region1, Some("inner".to_string()), None);
    let cell2 = Cell::new(Some(2), region2.clone(), Some("outer".to_string()), Some(0));

    let geometry = Geometry::new(vec![cell1, cell2], vec![material_arc]).unwrap();

    // Create source - 14.06 MeV (same as Python example)
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

    // Create tallies: one for flux, one for tritium production
    let cell_for_filter = Cell::new(Some(2), region2, Some("outer".to_string()), None);
    let cell_filter = Filter::Cell(CellFilter::from_id(cell_for_filter.cell_id.unwrap()));

    let mut flux_tally = Tally::new();
    flux_tally.filters = vec![cell_filter.clone()];
    flux_tally.scores = vec![Score::Flux(FluxScore)];
    flux_tally.name = Some("flux".to_string());
    flux_tally.initialize_batches(batches);

    // Create energy-binned flux tally - use same bins as Python example
    // Energy bins: logarithmically spaced from 0.01 eV to 20 MeV (20 bins)
    let n_bins = 20;
    let e_min = 0.01_f64;
    let e_max = 20e6_f64;
    let energy_bins: Vec<f64> = (0..n_bins)
        .map(|i| {
            10_f64.powf(
                (i as f64) / (n_bins as f64 - 1.0) * (e_max.log10() - e_min.log10())
                    + e_min.log10(),
            )
        })
        .collect();
    let energy_filter = EnergyFilter::new(energy_bins.clone());

    let mut flux_energy_tally = Tally::new();
    flux_energy_tally.filters = vec![cell_filter, Filter::Energy(energy_filter)];
    flux_energy_tally.scores = vec![Score::Flux(FluxScore)];
    flux_energy_tally.name = Some("flux_energy_binned".to_string());
    flux_energy_tally.initialize_batches(batches);

    let tallies = vec![Arc::new(flux_tally), Arc::new(flux_energy_tally)];

    let mut model = Model::new(geometry, vec![source], tallies);

    let start = Instant::now();
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles * batches),
            ..Default::default()
        })
        .unwrap();
    let elapsed = start.elapsed();

    println!(
        "\nSimulation completed in {:.3} seconds.",
        elapsed.as_secs_f64()
    );

    // Print results
    let flux_mean = model.tallies[0].get_mean();
    let flux_energy_mean = model.tallies[1].get_mean();

    println!("Total Flux: {:.6e}", flux_mean[0]);
    println!("Energy-binned flux has {} bins", flux_energy_mean.len());
    println!(
        "Sum of energy bins: {:.6e}",
        flux_energy_mean.iter().sum::<f64>()
    );

    // Print energy bin table
    println!("\n{}", "=".repeat(80));
    println!("FLUX BY ENERGY BIN (Cr52, 14.06 MeV source)");
    println!("{}", "=".repeat(80));
    println!(
        "{:<4} {:<14} {:<14} {:<16}",
        "Bin", "E_low (eV)", "E_high (eV)", "YAMC Flux"
    );
    println!("{}", "-".repeat(80));

    for i in 0..flux_energy_mean.len() {
        let e_low = energy_bins[i];
        let e_high = energy_bins[i + 1];
        let flux = flux_energy_mean[i];
        println!("{i:<4} {e_low:<14.4e} {e_high:<14.4e} {flux:<16.6e}");
    }
    println!("{}", "-".repeat(80));
    println!(
        "{:<4} {:<14} {:<14} {:<16.6e}",
        "TOT",
        "",
        "",
        flux_energy_mean.iter().sum::<f64>()
    );
    println!("{}", "=".repeat(80));
}
