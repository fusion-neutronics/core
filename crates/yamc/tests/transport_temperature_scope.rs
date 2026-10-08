//! Transport loads a nuclide at its material's temperature, not at every
//! temperature the file carries.
//!
//! A material whose nuclides reach transport unloaded (the usual case from
//! Python, where only the global cross-section setting is given) had them
//! loaded with every temperature in the file: seven in the published
//! libraries, each with its own transport lookup, when transport reads one.
//! On a 22-nuclide PbLi blanket that was 3.3 GB of a 6.6 GB heap.
//!
//! A temperature between two of the file's is still served, by loading both
//! neighbours and blending them, and must give exactly what the blend built
//! from a load of every temperature gives.
//!
//! Each test uses its own nuclide: the nuclide cache is process-wide, and a
//! wider entry left by one test would satisfy another's narrower request.

use std::collections::{BTreeSet, HashMap};
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
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;

/// Point the global configuration at the fixture for `nuclide`, or `None` to skip.
fn configure(nuclide: &str) -> Option<()> {
    let path = yamc_test_cache::transport_nuclide(nuclide)?;
    yamc_nuclide::Config::global().set_cross_section(nuclide, Some(&path));
    Some(())
}

/// A material of `nuclide` with no nuclear data read, so transport loads it.
fn material(nuclide: &str, temperature: Option<&str>) -> Material {
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(2.0),
    )
    .unwrap();
    m.set_material_id(1);
    if let Some(t) = temperature {
        m.set_temperature(t);
    }
    m
}

/// Run a 14 MeV point source in a 10 cm sphere of `material`, returning the
/// flux tally and the material as transport left it.
fn run(material: Material) -> (Vec<f64>, Arc<Material>) {
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
    let cell = Cell::new(Some(1), region, Some("sphere".into()), Some(0));
    let cell_id = cell.cell_id.unwrap();
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
    tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(cell_id)));
    tally.scores = vec!["flux".parse::<Score>().unwrap()];
    let tally = Arc::new(tally);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    let settings = TransportSettings {
        total_particles: Some(2_000),
        seed: 7,
        threads: Some(1),
        ..Default::default()
    };
    model.simulate_transport(&settings).expect("transport run");
    let material = Arc::clone(&model.geometry.materials()[0]);
    (tally.get_mean(), material)
}

fn loaded(material: &Material, nuclide: &str) -> BTreeSet<String> {
    material.nuclide_data[nuclide]
        .loaded_temperatures
        .iter()
        .cloned()
        .collect()
}

#[test]
fn a_listed_temperature_loads_only_that_temperature() {
    let nuclide = "Li7";
    if configure(nuclide).is_none() {
        eprintln!("skip: no {nuclide} fixture");
        return;
    }
    let (flux, material) = run(material(nuclide, Some("294")));
    assert!(flux[0] > 0.0, "no flux scored");
    let data = &material.nuclide_data[nuclide];
    assert!(
        data.available_temperatures.len() >= 3,
        "expected a multi-temperature fixture, got {:?}",
        data.available_temperatures
    );
    assert_eq!(
        loaded(&material, nuclide),
        BTreeSet::from(["294".to_string()]),
        "transport loaded more than the material's own temperature"
    );
}

#[test]
fn a_temperature_between_two_blends_exactly_as_a_full_load_does() {
    let nuclide = "Pb208";
    if configure(nuclide).is_none() {
        eprintln!("skip: no {nuclide} fixture");
        return;
    }

    // 450 K is published by no library: it falls between 294 K and 600 K.
    let (scoped_flux, scoped) = run(material(nuclide, Some("450")));
    assert_eq!(
        loaded(&scoped, nuclide),
        BTreeSet::from(["294".to_string(), "450".to_string(), "600".to_string()]),
        "a 450 K material should load its two neighbours and the blend of them"
    );

    // The reference: every temperature loaded first, as transport used to,
    // and only then the material labelled 450 K.
    let mut full = material(nuclide, None);
    full.ensure_nuclides_loaded()
        .expect("load every temperature");
    assert!(
        loaded(&full, nuclide).len() >= 3,
        "the reference was meant to hold every temperature, got {:?}",
        loaded(&full, nuclide)
    );
    full.set_temperature("450");
    let (full_flux, _) = run(full);

    assert!(scoped_flux[0] > 0.0, "no flux scored");
    assert_eq!(
        scoped_flux, full_flux,
        "the blend built from the two neighbours alone differs from the blend \
         built after loading every temperature"
    );
}

#[test]
fn a_temperature_outside_the_data_is_reported() {
    let nuclide = "Li6";
    if configure(nuclide).is_none() {
        eprintln!("skip: no {nuclide} fixture");
        return;
    }
    let outcome = std::panic::catch_unwind(|| run(material(nuclide, Some("5000"))));
    let message = match outcome {
        Ok(_) => panic!("a 5000 K material ran, but no library tabulates above 2500 K"),
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    };
    assert!(
        message.contains("5000") && message.contains("outside the range"),
        "the error should name the temperature and the range, got: {message}"
    );
}
