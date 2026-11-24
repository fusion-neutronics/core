//! Memory probe for an ensemble's replica inventories.
//!
//! Run as:
//! `cargo run --release -p yani-transmute --example ensemble_memory -- NEUTRON_DIR MATERIALS SAMPLES STEPS`
//!
//! `NEUTRON_DIR` is a library's `neutron/` folder (a cached TENDL-2025
//! release, say). Each material is an SS316-like steel irradiated for a year
//! and then cooled over `STEPS - 1` steps. Only cross sections are sampled, on
//! nuclides with no resonance parameters, so a replica is cheap and the ensemble's stored inventories are what the
//! memory holds. Prints the wall time, the peak resident set size (Linux)
//! and, per material, a checksum of every replica's inventory at every step,
//! so two builds can be compared for identical output.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_materials, MultigroupSpectrum, TransmuteCase, TransmuteStep};

/// Atom fractions of an iron-molybdenum alloy over nuclides whose
/// evaluations carry MF=33 covariance and no resonance parameters, so a
/// cross-section replica is a cheap solve.
const STEEL: &[(&str, f64)] = &[
    ("Fe54", 0.0560),
    ("Fe56", 0.8800),
    ("Fe57", 0.0200),
    ("Fe58", 0.0030),
    ("Ni61", 0.0030),
    ("Ni62", 0.0090),
    ("Ni64", 0.0020),
    ("Mo92", 0.0050),
    ("Mo94", 0.0030),
    ("Mo95", 0.0050),
    ("Mo96", 0.0050),
    ("Mo98", 0.0090),
];

fn status(field: &str) -> String {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with(field))
        .map(|l| l.split_whitespace().skip(1).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|| "n/a".into())
}

/// 80 logarithmic groups from 1e-5 eV to 20 MeV, `1/E` tilted by `tilt`.
fn spectrum(tilt: f64) -> MultigroupSpectrum {
    let n = 80;
    let (lo, hi): (f64, f64) = (1.0e-5, 2.0e7);
    let boundaries: Vec<f64> = (0..=n)
        .map(|i| lo * (hi / lo).powf(i as f64 / n as f64))
        .collect();
    let raw: Vec<f64> = (0..n).map(|i| (1.0 + i as f64).powf(tilt)).collect();
    let total: f64 = raw.iter().sum();
    MultigroupSpectrum {
        masses: raw.iter().map(|m| m / total).collect(),
        boundaries,
        flux_error: None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(&args[1]);
    let n_materials: u32 = args[2].parse().expect("MATERIALS");
    let samples: usize = args[3].parse().expect("SAMPLES");
    let n_steps: usize = args[4].parse().expect("STEPS");
    let chain_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let chain = Arc::new(yani::parse_chain_arrow(&chain_path).expect("parse chain"));
    let paths: HashMap<String, String> = STEEL
        .iter()
        .map(|(name, _)| {
            let path = dir.join(format!("{name}.arrow"));
            (name.to_string(), path.to_string_lossy().into_owned())
        })
        .collect();

    let mut materials: Vec<Material> = (0..n_materials)
        .map(|k| {
            let fractions: HashMap<String, f64> =
                STEEL.iter().map(|(n, f)| (n.to_string(), *f)).collect();
            let mut m = Material::new(fractions, "atom", "sum", None).expect("steel");
            m.density = Some(8.0);
            m.set_temperature("294");
            m.set_material_id(k + 1);
            m.read_nuclear_data(&paths, None).expect("read steel data");
            m
        })
        .collect();
    let request = DataUncertainty {
        seed: 42,
        samples: Some(samples),
        sources: vec![Source::CrossSections],
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let cases = materials
        .iter_mut()
        .enumerate()
        .map(|(k, m)| TransmuteCase {
            material: m,
            spectra: vec![spectrum(0.25 * k as f64)],
            steps: std::iter::once(TransmuteStep {
                dt: 3.15e7,
                irradiation: Some((0, 1.0e14)),
            })
            .chain((1..n_steps).map(|i| TransmuteStep {
                dt: 60.0 * 4.0_f64.powi(i as i32),
                irradiation: None,
            }))
            .collect(),
            shielding: None,
            product_threshold: yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
        })
        .collect();
    let results = transmute_materials(
        cases,
        chain,
        &Default::default(),
        Default::default(),
        Some(&request),
        None,
        None,
    )
    .expect("transmute");
    let elapsed = start.elapsed();
    println!(
        "{n_materials} materials x {samples} replicas: {elapsed:.1?}, peak RSS {}",
        status("VmHWM")
    );
    for id in 1..=n_materials {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for step in 1..=n_steps {
            let inventories = results.uncertainty_inventories(id, step).expect("ensemble");
            for inventory in inventories.iter() {
                let mut rows: Vec<(&String, &f64)> = inventory.iter().collect();
                rows.sort_by(|a, b| a.0.cmp(b.0));
                for (name, atoms) in rows {
                    name.hash(&mut hasher);
                    atoms.to_bits().hash(&mut hasher);
                }
            }
        }
        println!("material {id}: replica checksum {:016x}", hasher.finish());
    }
}
