//! A correlated flux error reaches the inventory with its correlations
//! (issue #140, item 6).
//!
//! Fe56(n,p)Mn56 under a spectrum of two fast groups, each carrying a 10%
//! error. If the two groups move together, the rate moves by the full 10%. If
//! they move independently it moves by `sqrt(w1^2 + w2^2) x 10%`, where `w` is
//! each group's share of the rate. A per-bin standard deviation can only say
//! the second; the covariance form says whichever is true.
//!
//! Self-skips when the Fe56 fixture is missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::flux_uncertainty::{FluxError, RelativeFluxCovariance};
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const RELATIVE: f64 = 0.10;
const FLUX: [f64; 2] = [1.0, 1.0];

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

fn iron() -> Option<Material> {
    let path = yamc_test_cache::nuclide("Fe56")?;
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.nuclides.insert("Fe56".to_string(), 8.5e-2);
    m.volume = Some(1.0);
    m.set_temperature("294");
    m.read_nuclear_data(&HashMap::from([("Fe56".to_string(), path)]), None)
        .expect("read Fe56");
    Some(m)
}

/// Mn56's relative sigma after a short irradiation, with a given flux error.
fn mn56_relative_sigma(error: FluxError) -> Option<f64> {
    let mut material = iron()?;
    let spectrum = MultigroupSpectrum {
        boundaries: vec![5.0e6, 1.0e7, 2.0e7],
        masses: vec![0.5, 0.5],
        flux_error: Some(error),
    };
    // Short, so Mn56 is far from saturation and its density is proportional
    // to the rate.
    let steps = vec![TransmuteStep {
        dt: 60.0,
        irradiation: Some((0, 1.0e14)),
    }];
    let request = DataUncertainty {
        seed: 2,
        samples: Some(1024),
        sources: vec![Source::FluxSpectrum],
    };
    let results = transmute_material(
        &mut material,
        &[spectrum],
        &steps,
        chain(),
        &Default::default(),
        Default::default(),
        Some(&request),
    )
    .expect("transmute");
    let nominal = results.get_material(0, 1).unwrap().nuclides["Mn56"];
    Some(results.uncertainty[&0].std_dev_at(0)["Mn56"] / nominal)
}

fn covariance(correlation: f64) -> FluxError {
    let v = RELATIVE * RELATIVE;
    let cov = vec![vec![v, correlation * v], vec![correlation * v, v]];
    FluxError::RelativeCovariance(RelativeFluxCovariance::from_absolute(&FLUX, &cov).unwrap())
}

#[test]
fn correlated_bins_move_a_rate_by_their_full_error() {
    let Some(correlated) = mn56_relative_sigma(covariance(1.0)) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    let independent = mn56_relative_sigma(covariance(0.0)).unwrap();
    let per_bin = mn56_relative_sigma(FluxError::RelativeStdDev(vec![RELATIVE; 2])).unwrap();

    // Sampling error on a sigma from 1024 replicas is about 2%.
    assert!(
        (correlated / RELATIVE - 1.0).abs() < 0.06,
        "fully correlated bins move the rate by their whole error: {correlated:.4}"
    );
    // Uncorrelated, the covariance and the per-bin form are one statement and
    // must agree; both sit below the correlated case by the rate's split
    // between the groups, which here leaves them well under 10%.
    assert!(
        (independent / per_bin - 1.0).abs() < 1e-9,
        "a diagonal covariance is the per-bin sigma: {independent} vs {per_bin}"
    );
    assert!(
        independent < 0.9 * correlated,
        "independent bins partly average out: {independent:.4} vs {correlated:.4}"
    );
}
