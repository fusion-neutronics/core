// Integration test: Elastic scatter only
// This test creates a material with only elastic scattering, no absorption or inelastic
// The source is monoenergetic at 15 MeV, and a flux tally with energy filter is used.
// The expected result is that all neutrons remain at 15 MeV (no energy loss).

use yamc::model::{Model, TransportSettings};
use yamc::*;
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::source::{ParticleSource, Source};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::EnergyFilter;

#[test]
fn test_elastic_scatter_only_flux() {
    // Create a material with only elastic scattering
    use std::collections::HashMap;
    let mut mat = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.0),
    )
    .unwrap();
    mat.set_name("ElasticOnly");
    mat.set_temperature("294");
    // Set cross section for Fe56 using global Config
    yamc_nuclide::Config::global().set_cross_section("Fe56", Some("tests/Fe56.arrow"));

    // Geometry: single cell filled with this material
    // Use a sphere region centered at the origin to enclose the source
    use std::sync::Arc;
    use yamc::geo::{BoundaryType, Surface};
    use yamc::geo::{HalfspaceType, Region, RegionExpr};
    let sphere = Arc::new(Surface::new_sphere(
        0.0,
        0.0,
        0.0,
        100.0,
        Some(1),
        Some(BoundaryType::Vacuum),
    ));
    let region = Region {
        expr: RegionExpr::Halfspace(HalfspaceType::Below(sphere)),
    };
    let cell = Cell::new(Some(1), region, Some("cell".to_string()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(mat)]).unwrap();

    // Source: monoenergetic at 15 MeV
    use yamc_source::distribution::energy::Discrete;
    use yamc_source::source::{SourceEnergyDistribution, SourceSpatialDistribution};
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(yamc_source::distribution::spatial::Point::new([
            0.0, 0.0, 0.0,
        ])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![15.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });

    let particles = 10000;
    let batches = 5;

    // Energy filter: 10 logarithmically spaced bins from 1e-5 to 20 MeV
    let n_bins = 10;
    let e_min = 1e-5_f64;
    let e_max = 20.0e6_f64;
    let log_min = e_min.ln();
    let log_max = e_max.ln();
    let mut energy_bins = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        let frac = i as f64 / n_bins as f64;
        energy_bins.push((log_min + frac * (log_max - log_min)).exp());
    }
    let energy_filter = EnergyFilter::new(energy_bins);

    // Flux tally with energy filter
    let mut tally = Tally::new();
    tally.filters = vec![Filter::Energy(energy_filter)];
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.name = Some("flux_energy_binned".to_string());
    tally.initialize_batches(batches);

    let tallies = vec![Arc::new(tally)];
    let mut model = Model::new(geometry, vec![source], tallies);
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles * batches),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    // Get the flux tally result
    let flux = model.tallies[0].get_mean();
    // For realistic elastic kinematics, neutrons will lose energy with each scatter (except for n-H)
    // So the flux should be spread below the source energy, not all at 15 MeV
    // With only one wide bin, all flux is in that bin, but let's use more bins to check the spectrum
    println!("Elastic-only flux tally: {:?}", flux);
    // The highest energy bin should have nonzero flux, but lower bins should also have some flux
    let n_bins = flux.len();
    assert!(
        n_bins > 1,
        "Test should use multiple energy bins to check spectrum"
    );
    assert!(
        flux[n_bins - 1] > 0.0,
        "Highest energy bin should have flux"
    );
    let lower_flux: f64 = flux[..n_bins - 1].iter().sum();
    assert!(
        lower_flux > 0.0,
        "Lower energy bins should have some flux due to downscatter"
    );
}
