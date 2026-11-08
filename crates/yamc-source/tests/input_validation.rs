//! Every number a source is built from is refused where it is given when it
//! is NaN or infinite (and a probability or a strength when it is negative):
//! in the constructor, and the same way as a model file is read. Each message
//! names the parameter and the value.
//!
//! Before, all of these were accepted and failed later, if at all: a NaN
//! energy or coordinate as lost particles or a NaN tally mid-run, a negative
//! strength by picking sources wrongly without a word.

use serde_json::json;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::{Discrete, Histogram, Normal, Uniform};
use yamc_source::distribution::spatial::{check_position, CylindricalRing, Point, Univariate};
use yamc_source::source::{
    check_strength, ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

#[test]
fn a_discrete_entry_that_is_not_finite_or_is_negative_is_refused_naming_it() {
    for (energies, probabilities, says) in [
        (
            vec![1e6, NAN],
            vec![1.0, 1.0],
            "Discrete energies[1] must be a finite number (got NaN)",
        ),
        (
            vec![INF],
            vec![1.0],
            "Discrete energies[0] must be a finite number (got inf)",
        ),
        (
            vec![1e6, 2e6],
            vec![1.0, NAN],
            "Discrete probabilities[1] must be a finite number (got NaN)",
        ),
        (
            vec![1e6],
            vec![INF],
            "Discrete probabilities[0] must be a finite number (got inf)",
        ),
        (
            vec![1e6, 2e6],
            vec![1.0, -0.5],
            "Discrete probabilities[1] is -0.5: probabilities cannot be negative",
        ),
    ] {
        assert_eq!(
            Discrete::new(energies.clone(), probabilities.clone()).unwrap_err(),
            says,
            "{energies:?} {probabilities:?}"
        );
    }
    // Negative values themselves are fine: Discrete also holds ring
    // coordinates, where z and phi can be negative.
    assert!(Discrete::new(vec![-3.0, 0.0], vec![1.0, 0.0]).is_ok());
}

#[test]
fn uniform_and_normal_refuse_parameters_that_are_not_finite() {
    assert_eq!(
        Uniform::new(NAN, 1.0).unwrap_err(),
        "Uniform low must be a finite number (got NaN)"
    );
    assert_eq!(
        Uniform::new(0.0, INF).unwrap_err(),
        "Uniform high must be a finite number (got inf)"
    );
    assert_eq!(
        Normal::new(NAN, 1.0).unwrap_err(),
        "Normal mean must be a finite number (got NaN)"
    );
    assert_eq!(
        Normal::new(1.0, INF).unwrap_err(),
        "Normal std_dev must be a finite number (got inf)"
    );
    // A NaN standard deviation used to pass the `std_dev <= 0.0` check.
    assert_eq!(
        Normal::new(1.0, NAN).unwrap_err(),
        "Normal std_dev must be a finite number (got NaN)"
    );
}

#[test]
fn a_position_or_ring_origin_that_is_not_finite_is_refused_naming_the_coordinate() {
    assert_eq!(
        check_position("position", [NAN, 0.0, 0.0]).unwrap_err(),
        "position [NaN, 0.0, 0.0] has a NaN coordinate (x = NaN). Every coordinate must be a \
         finite number of cm."
    );
    let err = check_position("position", [0.0, 0.0, -INF]).unwrap_err();
    assert!(
        err.contains("has an infinite coordinate (z = -inf)"),
        "{err}"
    );
    assert!(check_position("position", [0.0, -1e300, 5.0]).is_ok());

    let ring = |origin| {
        CylindricalRing::new(
            Univariate::Discrete(Discrete::new(vec![300.0], vec![1.0]).unwrap()),
            Univariate::Uniform(Uniform::new(0.0, std::f64::consts::TAU).unwrap()),
            Univariate::Discrete(Discrete::new(vec![0.0], vec![1.0]).unwrap()),
            origin,
        )
    };
    let err = ring([0.0, INF, 0.0]).unwrap_err();
    assert!(
        err.starts_with(
            "CylindricalRing origin [0.0, inf, 0.0] has an infinite coordinate (y = inf)"
        ),
        "{err}"
    );
    assert!(ring([1.0, 2.0, 3.0]).is_ok());
}

#[test]
fn a_strength_that_is_nan_infinite_or_negative_is_refused() {
    assert_eq!(
        check_strength(NAN).unwrap_err(),
        "strength must be a finite number (got NaN)"
    );
    assert_eq!(
        check_strength(INF).unwrap_err(),
        "strength must be a finite number (got inf)"
    );
    let err = check_strength(-1.0).unwrap_err();
    assert!(
        err.starts_with("strength must not be negative (got -1)"),
        "{err}"
    );
    // A source of strength 0 is allowed: it never emits.
    assert!(check_strength(0.0).is_ok() && check_strength(2.5).is_ok());
}

/// A neutron source with every distribution a model file can hold.
fn source(space: SourceSpatialDistribution, energy: SourceEnergyDistribution) -> ParticleSource {
    ParticleSource::Neutron(Source {
        space,
        angle: AngularDistribution::Isotropic,
        energy,
        strength: 2.0,
    })
}

fn load(value: serde_json::Value) -> Result<ParticleSource, String> {
    serde_json::from_value(value).map_err(|e| e.to_string())
}

/// What a model file holds is read back exactly, alias tables included.
#[test]
fn a_saved_source_loads_back_exactly() {
    let ring = CylindricalRing::new(
        Univariate::Normal(Normal::new(300.0, 2.0).unwrap()),
        Univariate::Uniform(Uniform::new(0.0, 1.5).unwrap()),
        Univariate::Discrete(Discrete::new(vec![-1.0, 1.0], vec![0.3, 0.7]).unwrap()),
        [1.0, -2.0, 3.0],
    )
    .unwrap();
    for saved in [
        source(
            SourceSpatialDistribution::Point(Point::new([1.0, 2.0, 3.0])),
            SourceEnergyDistribution::Discrete(
                Discrete::new(vec![1e6, 2e6, 3e6], vec![0.2, 0.5, 0.3]).unwrap(),
            ),
        ),
        source(
            SourceSpatialDistribution::CylindricalRing(Box::new(ring)),
            SourceEnergyDistribution::Histogram(
                Histogram::new(vec![0.0, 1e6, 2e6], vec![1.0, 3.0]).unwrap(),
            ),
        ),
    ] {
        let json = serde_json::to_value(&saved).unwrap();
        let loaded = load(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(&loaded).unwrap(), json);
        assert_eq!(loaded.source().energy, saved.source().energy);
        assert_eq!(loaded.source().space, saved.source().space);
    }
}

/// A model file edited by hand is refused for what a constructor refuses,
/// with the same message.
#[test]
fn a_model_file_is_checked_as_a_constructed_source_is() {
    let saved = serde_json::to_value(source(
        SourceSpatialDistribution::Point(Point::new([1.0, 2.0, 3.0])),
        SourceEnergyDistribution::Discrete(
            Discrete::new(vec![1e6, 2e6], vec![0.25, 0.75]).unwrap(),
        ),
    ))
    .unwrap();
    assert!(saved["Neutron"]["strength"].is_number(), "{saved}");
    assert!(
        saved["Neutron"]["energy"]["Discrete"]["probabilities"].is_array(),
        "{saved}"
    );

    let edit = |path: &[&str], value: serde_json::Value| {
        let mut json = saved.clone();
        let mut node = &mut json;
        for key in path {
            node = &mut node[*key];
        }
        *node = value;
        load(json).unwrap_err()
    };
    for (path, value, says) in [
        (
            &["Neutron", "strength"][..],
            json!(-1.0),
            "strength must not be negative (got -1)",
        ),
        (
            &["Neutron", "energy", "Discrete", "probabilities"][..],
            json!([0.25, -0.75]),
            "Discrete probabilities[1] is -0.75: probabilities cannot be negative",
        ),
        (
            &["Neutron", "energy"][..],
            json!({"Uniform": {"a": 2e6, "b": 1e6}}),
            "Lower bound must be less than upper bound",
        ),
        (
            &["Neutron", "energy"][..],
            json!({"Normal": {"mean_val": 14e6, "std_dev": 0.0}}),
            "Standard deviation must be positive",
        ),
        (
            &["Neutron", "energy"][..],
            json!({"Histogram": {"boundaries": [0.0, 1e6], "probabilities": [-1.0]}}),
            "Histogram probabilities cannot be negative",
        ),
    ] {
        let err = edit(path, value);
        assert!(err.contains(says), "{path:?}: {err}");
    }

    // JSON has no NaN or infinity, and a number past the range of f64 does
    // not parse, so a model file cannot carry a coordinate that is not finite
    // into a run either.
    let text = saved
        .to_string()
        .replace("[1.0,2.0,3.0]", "[1.0,2.0,1e999]");
    assert_ne!(text, saved.to_string());
    assert!(serde_json::from_str::<ParticleSource>(&text).is_err());
}
