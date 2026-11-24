//! Transmutation rates read the probability-table band the flight drew.
//!
//! The reaction rates `Model::transmute` scores were `sigma(E) * track length`
//! with the smooth cross section, while the transport and the ordinary
//! tallies use the probability tables. Inside a band the flux and the drawn
//! cross section are anti-correlated, so in a self-shielding material the
//! smooth rate over-counts: W186 capture was 37% high in a 10 cm sphere with
//! a 30 keV source.
//!
//! One independent-mode run scores both the transmutation rates and an
//! ordinary track-length `(n,gamma)` tally in the same transport, so the W187
//! made per source neutron must equal the tally to well within its noise.
//! Needs the W186 fixture with its probability tables, and skips without it.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
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
use yani::{BranchTable, ChainNuclide, ChainReaction};
use yani_transmute::{Schedule, ScheduleStep};

const RADIUS: f64 = 10.0;

fn nuclide(name: &str, reactions: Vec<ChainReaction>) -> ChainNuclide {
    ChainNuclide {
        name: name.to_string(),
        half_life: None,
        decay_energy: 0.0,
        reactions: reactions.into(),
        decays: vec![],
        fission_yields: None,
        sources: Default::default(),
        half_life_uncertainty: None,
        decay_energy_uncertainty: None,
        decay_evaluated: true,
        excitation_energy: None,
        level_index: None,
        decay_energy_components: Default::default(),
    }
}

/// W186 captures to a stable W187, and nothing else happens.
fn chain() -> Arc<HashMap<String, ChainNuclide>> {
    let capture = ChainReaction {
        kind: "(n,gamma)".to_string(),
        target: Some("W187".to_string()),
        branching: 1.0,
        q_value: None,
        branching_uncertainty: None,
        evaluated_branching: None,
        multiplicity: None,
    };
    Arc::new(HashMap::from([
        ("W186".to_string(), nuclide("W186", vec![capture])),
        ("W187".to_string(), nuclide("W187", vec![])),
    ]))
}

#[test]
fn capture_rate_inside_the_band_matches_the_capture_tally() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/W186.arrow");
    if !path.join("urr.arrow").is_file() {
        eprintln!("skipping: W186 fixture with probability tables missing");
        return;
    }
    let volume = 4.0 / 3.0 * std::f64::consts::PI * RADIUS.powi(3);
    let mut material = Material::new(
        HashMap::from([("W186".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(19.3),
    )
    .unwrap();
    material.set_temperature("294");
    material.set_material_id(1);
    material.transmutable = true;
    material.volume = Some(volume);
    material
        .read_nuclear_data(
            &HashMap::from([("W186".to_string(), path.to_string_lossy().into_owned())]),
            None,
        )
        .unwrap();
    let w186 = material.get_atoms_per_barn_cm().unwrap()["W186"];

    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("w".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![3.0e4], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut capture = Tally::new();
    capture.scores = vec!["(n,gamma)".parse::<Score>().unwrap()];
    capture.estimator = Estimator::TrackLength;
    let capture = Arc::new(capture);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&capture)]);
    model.verbose = Verbose::silent();

    // A rate low enough that W186 is not depleted: W187 is then exactly the
    // captures.
    let (rate, dt) = (1.0e10, 1.0);
    let schedule = Schedule::new(vec![ScheduleStep {
        rate,
        dt,
        is_pulse: true,
    }])
    .unwrap();
    let (results, tallies) = model
        .transmute_with_tallies(
            "independent",
            &schedule,
            chain(),
            Arc::new(BranchTable::new()),
            Default::default(),
            &TransportSettings {
                total_particles: Some(20_000),
                seed: 3,
                ..Default::default()
            },
            None,
            0.0,
            None,
        )
        .unwrap();
    let after = results.get_final_material(1).unwrap();
    let w187 = after.get_atoms_per_barn_cm().unwrap()["W187"];
    let made = w187 * 1.0e24 * volume / (rate * dt);
    let (mean, sd) = tallies.evolution(&capture).unwrap();
    // The step tallies come back at the step's rate; per source neutron the
    // tally is N sigma track length.
    let (tallied, sigma) = (mean[0][0] / rate, sd[0][0] / rate);
    eprintln!(
        "W187 per source neutron {made:.5}, (n,gamma) tally {tallied:.5} +- {sigma:.5} \
         (W186 {w186:.4e} /b-cm)"
    );
    assert!(
        (made / tallied - 1.0).abs() < 0.005,
        "the transmutation capture rate {made:.5} per source neutron disagrees with the \
         capture tally {tallied:.5} scored in the same transport"
    );
}
