//! A material built without a temperature transports at the 294 K default.
//!
//! An unset temperature used to reach `resolve_temperature` with the full
//! temperature ladder loaded and no rule to pick from it, so the run panicked
//! with "Temperature not specified or not available" even though the data had
//! a 294 K entry. Now the default is applied when every nuclide can serve it,
//! and when one cannot, transport setup returns an error rather than panicking.
//!
//! Data: endf-b8.1 Fe56 in the fixture cache. Self-skips when absent.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::tally::{FluxScore, Score, Tally};

/// Fe56 at 7.8 g/cm3 with no temperature set, its data read from `path`.
fn unlabelled_fe56(path: &str) -> Material {
    let mut material = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.8),
    )
    .unwrap();
    material
        .read_nuclear_data(
            &HashMap::from([("Fe56".to_string(), path.to_string())]),
            None,
        )
        .unwrap();
    assert!(
        material.temperature().is_empty(),
        "the material must start unlabelled for this test to mean anything"
    );
    material
}

/// The issue's model: a 10 cm vacuum-bounded sphere of `material`, a
/// 14.06 MeV point source at the centre and a flux tally.
fn sphere_model(material: Material) -> Model {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let mut tally = Tally::new();
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.name = Some("flux".to_string());

    Model::new(geometry, vec![source], vec![Arc::new(tally)])
}

fn settings() -> TransportSettings {
    TransportSettings {
        total_particles: Some(500),
        seed: 1,
        ..Default::default()
    }
}

#[test]
fn an_unset_temperature_transports_at_294_k() {
    let Some(path) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("an_unset_temperature_transports_at_294_k: skip (Fe56 fixture missing)");
        return;
    };
    let material = unlabelled_fe56(&path);
    assert!(
        material.nuclide_data["Fe56"].available_temperatures.len() > 1,
        "needs a multi-temperature fixture, or the single-temperature \
         auto-detect answers instead of the default"
    );

    let mut model = sphere_model(material);
    model
        .simulate_transport(&settings())
        .expect("an unset temperature must fall back to 294 K and run");

    let resolved = model.geometry.materials()[0].temperature().to_string();
    assert_eq!(resolved, "294", "the run must have used the 294 K default");
    let flux = model.tallies[0].get_mean();
    assert!(
        flux.iter().sum::<f64>() > 0.0,
        "the flux tally scored nothing"
    );
}

#[test]
fn an_unset_temperature_the_data_cannot_serve_is_an_error_not_a_panic() {
    let Some(path) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!(
            "an_unset_temperature_the_data_cannot_serve_is_an_error_not_a_panic: \
             skip (Fe56 fixture missing)"
        );
        return;
    };
    let mut material = unlabelled_fe56(&path);

    // No published library lacks 294 K, so narrow the advertised ladder to
    // one that sits wholly above it: 294 K is then neither listed nor
    // bracketed, and there is more than one temperature to choose from.
    let mut nuclide = (*material.nuclide_data["Fe56"]).clone();
    nuclide.available_temperatures = vec!["600".to_string(), "900".to_string()];
    material
        .nuclide_data
        .insert("Fe56".to_string(), Arc::new(nuclide));

    let mut model = sphere_model(material);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.simulate_transport(&settings())
    }));
    let err = match result {
        Ok(Ok(())) => panic!("an unservable default temperature must not run"),
        Ok(Err(e)) => e,
        Err(_) => panic!("an unservable default temperature must be an error, not a panic"),
    };
    assert!(
        err.contains("294"),
        "the error must name the default: {err}"
    );
    assert!(
        err.contains("Fe56"),
        "the error must name the nuclide: {err}"
    );
    assert!(
        err.contains("600") && err.contains("900"),
        "the error must list what is available: {err}"
    );
}
