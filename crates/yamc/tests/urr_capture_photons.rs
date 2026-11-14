//! Capture photons inside a probability-table band carry the band's capture
//! factor, not its total factor.
//!
//! A collision sampled from a band's total `σ_t(ξ)` carried photons at the
//! smooth `photon_prod / σ_t`, so the production per unit flux came out as the
//! smooth capture scaled by the band's TOTAL factor rather than its capture
//! factor. In W186 the two differ by about a factor of two across the band,
//! and with the flux shielded (roughly `1/σ_t`) a source inside the band made
//! about 30% less photon energy per capture than one below it.
//!
//! Photon heating in a large sphere with local electrons is all the photon
//! energy born less a small leakage, so photon heating over the track-length
//! capture is the photon energy produced per capture. It must not depend on
//! whether the neutrons are born inside the band. Needs the W186 neutron and
//! W photon fixtures, and skips without them.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{ElectronTreatment, Model, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

fn tungsten() -> Option<Material> {
    let neutron = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/W186.arrow");
    let photon = yamc_test_cache::element("W")?;
    if !neutron.join("urr.arrow").is_file() {
        return None;
    }
    let mut m = Material::new(
        HashMap::from([("W186".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(19.3),
    )
    .ok()?;
    m.set_material_id(1);
    m.set_temperature("294");
    let photon_paths = HashMap::from([("W".to_string(), photon)]);
    m.read_nuclear_data(
        &HashMap::from([("W186".to_string(), neutron.to_string_lossy().into_owned())]),
        Some(&photon_paths),
    )
    .ok()?;
    m.init_photon_data(&photon_paths).ok()?;
    Some(m)
}

/// Photon energy per capture, and its relative standard error, for neutrons
/// born at `energy` at the centre of a tungsten sphere.
fn photon_energy_per_capture(material: Material, energy: f64, particles: usize) -> (f64, f64) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 20.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("w".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).expect("geometry");

    let mut capture = Tally::new();
    capture.scores = vec!["(n,gamma)".parse::<Score>().unwrap()];
    capture.estimator = Estimator::TrackLength;
    let mut heating = Tally::new();
    heating.scores = vec!["heating".parse::<Score>().unwrap()];
    heating.estimator = Estimator::Collision;
    heating
        .filters
        .push(Filter::ParticleType(ParticleTypeFilter::new(
            ParticleType::Photon,
        )));
    let tallies = [Arc::new(capture), Arc::new(heating)];

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![energy], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(
        geometry,
        vec![source],
        tallies.iter().map(Arc::clone).collect(),
    );
    model.verbose = Verbose::silent();
    model.transport_secondary_photons = true;
    model.electron_treatment = ElectronTreatment::Local;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles),
            seed: 5,
            ..Default::default()
        })
        .expect("transport");
    let (c, sc) = (tallies[0].total_mean(), tallies[0].get_std_dev()[0]);
    let (h, sh) = (tallies[1].total_mean(), tallies[1].get_std_dev()[0]);
    let ratio = h / c;
    (ratio, ((sc / c).powi(2) + (sh / h).powi(2)).sqrt())
}

#[test]
fn photon_energy_per_capture_does_not_depend_on_the_band() {
    let Some(m) = tungsten() else {
        eprintln!("skipping: W186 or W photon fixtures missing");
        return;
    };
    let particles = 20_000;
    // W186's probability tables start above 5 keV and cover 50 keV.
    let (below, s_below) = photon_energy_per_capture(m.clone(), 5.0e3, particles);
    let (inside, s_inside) = photon_energy_per_capture(m, 5.0e4, particles);
    let sigma = (s_below.powi(2) + s_inside.powi(2)).sqrt();
    let difference = inside / below - 1.0;
    eprintln!(
        "photon energy per capture: below the band {below:.4e} eV, inside {inside:.4e} eV \
         ({:+.1}% +- {:.1}%)",
        100.0 * difference,
        100.0 * sigma
    );
    assert!(
        difference.abs() < 0.05_f64.max(4.0 * sigma),
        "photon energy per capture inside the band is {inside:.4e} eV against {below:.4e} eV \
         below it ({:+.1}%)",
        100.0 * difference
    );
}
