//! The track-length photon heating coefficient deposits what the collisions
//! deposit.
//!
//! With local electrons, no bremsstrahlung banked and nothing leaking, the
//! collision estimator of photon heating returns the source energy. The
//! track-length estimator reads the KERMA coefficient instead, whose Compton
//! term paired the bound incoherent cross section with the free-electron
//! Klein-Nishina transfer fraction: the incoherent scattering function removes
//! the forward scatters, which transfer least, so that under-counted (1.4% in
//! beryllium at 60 keV, 0.6% in iron at 150 keV). What is left, about 0.2%
//! in beryllium, is the Doppler broadening the coefficient does not carry.
//! Needs the Fe and Be fixtures, and skips without them.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{ElectronTreatment, Model, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

/// Track-length over collision photon heating, and its standard error, for a
/// photon point source of `energy` at the centre of a sphere of `element`.
fn ratio(
    nuclide: &str,
    element: &str,
    density: f64,
    radius: f64,
    energy: f64,
    particles: usize,
) -> Option<(f64, f64)> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let (n, p) = (
        dir.join(format!("{nuclide}.arrow")),
        dir.join(format!("{element}.arrow")),
    );
    if !n.is_dir() || !p.is_dir() {
        return None;
    }
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    let photon = HashMap::from([(element.to_string(), p.to_string_lossy().into_owned())]);
    m.read_nuclear_data(
        &HashMap::from([(nuclide.to_string(), n.to_string_lossy().into_owned())]),
        Some(&photon),
    )
    .unwrap();
    m.init_photon_data(&photon).unwrap();
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("s".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(m)]).unwrap();
    let tally = |estimator| {
        let mut t = Tally::new();
        t.scores = vec!["heating".parse::<Score>().unwrap()];
        t.estimator = estimator;
        Arc::new(t)
    };
    let tallies = [tally(Estimator::TrackLength), tally(Estimator::Collision)];
    let source = ParticleSource::Photon(Source {
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
    model.electron_treatment = ElectronTreatment::Local;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles),
            seed: 9,
            ..Default::default()
        })
        .unwrap();
    let (t, st) = (tallies[0].total_mean(), tallies[0].get_std_dev()[0]);
    let (c, sc) = (tallies[1].total_mean(), tallies[1].get_std_dev()[0]);
    let r = t / c;
    Some((r, r * ((st / t).powi(2) + (sc / c).powi(2)).sqrt()))
}

#[test]
fn track_length_photon_heating_matches_the_collisions() {
    let cases = [
        ("Be9", "Be", 1.85, 150.0, 6.0e4),
        ("Fe56", "Fe", 7.874, 30.0, 1.5e5),
    ];
    for (nuclide, element, density, radius, energy) in cases {
        let Some((r, s)) = ratio(nuclide, element, density, radius, energy, 100_000) else {
            eprintln!("skipping {element}: fixtures missing");
            continue;
        };
        eprintln!("{element} at {energy:e} eV: track length / collision {r:.5} +- {s:.5}");
        assert!(
            (r - 1.0).abs() < 4.0 * s + 1e-3,
            "{element} at {energy:e} eV: track length / collision {r:.5} +- {s:.5}"
        );
    }
}
