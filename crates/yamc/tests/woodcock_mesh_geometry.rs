//! Delta tracking on mesh geometry: Woodcock and hybrid agree with surface
//! tracking on a mesh with more than one volume.
//!
//! Delta tracking finds the cell at the end of every flight by point
//! location, and stops a flight where it leaves through a vacuum boundary.
//! The fixture is a fuel sphere imprinted inside a moderator cube whose
//! outside is a vacuum boundary, with a 14 MeV point source at the centre,
//! so histories cross from one volume into the other and leak: a lookup that
//! keeps particles in the source volume, or a flight that is not stopped at
//! the vacuum boundary, shows up as a flux in the wrong place.
#![cfg(feature = "mesh")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geometry::mesh::MeshGeometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::MaterialFilter;

const HISTORIES: usize = 40_000;

fn material(nuclide: &str, density: f64, id: u32) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    m.set_material_id(id);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([(nuclide.to_string(), format!("tests/{nuclide}.arrow"))]),
        None,
    )
    .unwrap();
    Arc::new(m)
}

/// Flux (mean, standard deviation) in the fuel and the moderator.
fn fluxes(
    mode: TrackingMode,
    fuel: &Arc<Material>,
    moderator: &Arc<Material>,
    seed: u64,
) -> [(f64, f64); 2] {
    let materials = HashMap::from([
        ("fuel".to_string(), fuel.clone()),
        ("moderator".to_string(), moderator.clone()),
    ]);
    let mesh = MeshGeometry::from_arrow(
        std::path::Path::new("../yamt/tests/data/sphere_in_cube.arrow"),
        &materials,
    )
    .expect("load sphere_in_cube.arrow");
    let tallies: Vec<Arc<Tally>> = [fuel, moderator]
        .into_iter()
        .map(|m| {
            let mut t = Tally::new();
            t.filters = vec![Filter::Material(MaterialFilter::new(m))];
            t.scores = vec![Score::Flux(FluxScore)];
            t.reset_accumulation();
            Arc::new(t)
        })
        .collect();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new_with_mesh(mesh, vec![source], tallies.clone());
    model.verbose = Verbose::silent();
    model.tracking_mode = mode;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(HISTORIES),
            seed,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("{mode:?} run failed: {e}"));
    [0, 1].map(|i| (tallies[i].get_mean()[0], tallies[i].get_std_dev()[0]))
}

#[test]
fn delta_tracking_on_a_mesh_matches_surface_tracking() {
    let fuel = material("Fe56", 7.874, 1);
    let moderator = material("Be9", 1.85, 2);
    let surface = fluxes(TrackingMode::Surface, &fuel, &moderator, 1);
    for (mode, seed) in [(TrackingMode::Woodcock, 2), (TrackingMode::Hybrid, 3)] {
        let delta = fluxes(mode, &fuel, &moderator, seed);
        for (name, (mean, sd), (reference, reference_sd)) in [
            ("fuel", delta[0], surface[0]),
            ("moderator", delta[1], surface[1]),
        ] {
            let z = (mean - reference) / sd.hypot(reference_sd);
            assert!(
                z.abs() < 5.0,
                "{mode:?} {name} flux {mean:.5e} +- {sd:.1e}, surface tracking \
                 {reference:.5e} +- {reference_sd:.1e} (z = {z:.1})"
            );
        }
    }
}
