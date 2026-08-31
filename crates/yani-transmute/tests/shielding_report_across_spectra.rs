//! A schedule with two spectra must report on both of them.
//!
//! The driver assigned each spectrum's [`ShieldingInfo`] over the last, so with
//! two spectra only the second one's nuclides were ever named. #564's whole
//! point is that report -- what a dilute run did not correct for -- and it was
//! silently halved by any schedule that switched spectrum. Issue #576.
//!
//! Built so the two spectra genuinely disagree: one puts all its flux in the
//! resonance region, where Fe56 has structure and the indicator fires, and the
//! other puts it above 1 MeV, where it does not. Whichever way round the
//! driver walked them, the merged report has to name the nuclide.
//!
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const DAY: f64 = 86400.0;
const RATE: f64 = 1.0e14;

/// Resonance region only: Fe56's own structure is here, so the dilute run's
/// self-shielding indicator has something to report.
const RESONANT: [f64; 2] = [1.0e2, 1.0e5];
/// Fast only, above the resolved resonances.
const FAST: [f64; 2] = [1.0e6, 2.0e7];

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
    // A real density, so the indicator's `1 / (1 + N sigma)` weight is not 1.
    m.nuclides.insert("Fe56".to_string(), 8.5e-2);
    m.density = Some(7.87);
    m.volume = Some(1.0);
    m.set_temperature("294");
    m.read_nuclear_data(&HashMap::from([("Fe56".to_string(), path)]), None)
        .expect("read Fe56");
    Some(m)
}

fn one_group(edges: [f64; 2]) -> MultigroupSpectrum {
    MultigroupSpectrum {
        boundaries: edges.to_vec(),
        masses: vec![1.0],
        relative_std_dev: None,
    }
}

/// The dilute run's self-shielding indicator, per nuclide.
fn would_shield(spectra: &[MultigroupSpectrum]) -> Vec<(String, f64)> {
    let steps: Vec<TransmuteStep> = (0..spectra.len())
        .map(|idx| TransmuteStep {
            dt: DAY,
            irradiation: Some((idx, RATE)),
        })
        .collect();
    let mut material = iron().expect("checked by the caller");
    let results = transmute_material(
        &mut material,
        spectra,
        &steps,
        chain(),
        &Default::default(),
        Default::default(),
        None,
    )
    .expect("transmute");
    let info = results.shielding_info.expect("a report either way");
    let mut out: Vec<(String, f64)> = info.would_shield.into_iter().collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// What merging two reports has to produce: every nuclide either names, at the
/// strongest suppression either saw.
fn strongest_of(a: &[(String, f64)], b: &[(String, f64)]) -> Vec<(String, f64)> {
    let mut merged: std::collections::BTreeMap<String, f64> = a.iter().cloned().collect();
    for (name, bound) in b {
        let entry = merged.entry(name.clone()).or_insert(*bound);
        if bound < entry {
            *entry = *bound;
        }
    }
    merged.into_iter().collect()
}

#[test]
fn a_two_spectrum_schedule_reports_on_both() {
    if iron().is_none() {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    }
    let resonant = would_shield(&[one_group(RESONANT)]);
    let fast = would_shield(&[one_group(FAST)]);
    assert!(
        !resonant.is_empty(),
        "the resonance-region spectrum must trigger the indicator for this test \
         to mean anything"
    );
    assert_ne!(
        resonant, fast,
        "the two spectra must disagree for this test to mean anything"
    );

    // Whichever order the driver walks them in, the merged report is what both
    // spectra found -- not whichever one happened to be assigned last.
    let expected = strongest_of(&resonant, &fast);
    assert_eq!(
        would_shield(&[one_group(RESONANT), one_group(FAST)]),
        expected,
        "the merged report is not what the two spectra found between them"
    );
    assert_eq!(
        would_shield(&[one_group(FAST), one_group(RESONANT)]),
        expected,
        "the merged report depends on the order the spectra are listed in"
    );
}
