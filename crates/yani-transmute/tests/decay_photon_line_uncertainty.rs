//! Decay photon line uncertainty on the photon spectrum and contact dose.
//!
//! Co60 cooling on its own, with sigmas placed on its one photon spectrum: a
//! normalisation sigma common to every line, and a dRI and dER of their own on
//! the 1173 and 1332 keV lines. No photon enters the Bateman matrix, so the
//! inventory and the activity must not move at all, while each line moves by
//! the normalisation's and its own sigma in quadrature, its energy by its dER,
//! and a contact dose fed by one spectrum by the normalisation alone.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, TransmutationResults, TransmuteStep};

const LINE_1173: f64 = 1_173_228.0;
const LINE_1332: f64 = 1_332_492.0;
const LINE_826: f64 = 826_100.0;

/// The fixture chain with Co60's photon spectrum given a normalisation sigma
/// `fd` (relative), and when `own` is set, a dRI and dER on its two strong
/// lines: 2% and 30 eV on 1173 keV, 3% and 20 eV on 1332 keV.
fn chain(fd: f64, own: bool) -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let mut chain = yani::parse_chain_arrow(&path).expect("parse chain");
    let co60 = chain.get_mut("Co60").expect("Co60");
    let source = co60
        .sources
        .iter_mut()
        .find(|s| s.particle == "photon")
        .expect("Co60 photon lines");
    let yani::DecaySourceDistribution::Discrete {
        energies,
        intensities,
    } = &source.distribution
    else {
        panic!("Co60's photon source is a line spectrum");
    };
    let mut intensity_sigmas = vec![0.0; energies.len()];
    let mut energy_sigmas = vec![0.0; energies.len()];
    if own {
        for (i, energy) in energies.iter().enumerate() {
            if *energy == LINE_1173 {
                intensity_sigmas[i] = 0.02 * intensities[i];
                energy_sigmas[i] = 30.0;
            } else if *energy == LINE_1332 {
                intensity_sigmas[i] = 0.03 * intensities[i];
                energy_sigmas[i] = 20.0;
            }
        }
    }
    source.uncertainty = Some(Arc::new(yani::DecaySourceUncertainty {
        normalization: Some(1.0),
        normalization_uncertainty: Some(fd),
        intensity_uncertainties: Some(intensity_sigmas),
        energy_uncertainties: Some(energy_sigmas),
        covariance: None,
    }));
    Arc::new(chain)
}

fn cobalt() -> Material {
    let mut m = Material::new(
        HashMap::from([("Co60".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Co60 material");
    m.nuclides.insert("Co60".to_string(), 1.0e-6);
    m.volume = Some(1.0);
    m
}

fn run(
    chain: &Arc<HashMap<String, yani::ChainNuclide>>,
    sources: Vec<Source>,
    samples: usize,
) -> TransmutationResults {
    let mut material = cobalt();
    let steps = vec![TransmuteStep {
        dt: 86400.0,
        irradiation: None,
    }];
    let request = DataUncertainty {
        seed: 11,
        samples: Some(samples),
        sources,
        attribution: false,
        ..Default::default()
    };
    transmute_material(
        &mut material,
        &[],
        &steps,
        Arc::clone(chain),
        &Default::default(),
        Default::default(),
        Some(&request),
    )
    .expect("transmute")
}

fn assert_close(got: f64, want: f64, tolerance: f64, what: &str) {
    assert!(
        (got / want - 1.0).abs() < tolerance,
        "{what}: {got:.5e} against {want:.5e}"
    );
}

#[test]
fn lines_move_by_their_stated_sigmas_and_the_inventory_does_not() {
    let fd = 0.01;
    let chain = chain(fd, true);
    let results = run(&chain, vec![Source::DecayPhotonLines], 4096);

    let info = &results.uncertainty_info[&0];
    assert!(
        info.decay_photon_lines_perturbed.contains("Co60"),
        "{info:?}"
    );
    assert!(!info
        .not_perturbed
        .iter()
        .any(|s| s.starts_with("decay photon line energy and intensity")));
    assert_eq!(info.sources, vec!["decay_photon_lines".to_string()]);

    let lines = results
        .photon_spectrum_uncertainty(0, 1, &chain)
        .unwrap()
        .expect("an ensemble");
    let line = |energy: f64| {
        *lines
            .iter()
            .find(|l| l.energy == energy)
            .unwrap_or_else(|| panic!("no line at {energy} eV"))
    };
    for (energy, own, de) in [(LINE_1173, 0.02, 30.0), (LINE_1332, 0.03, 20.0)] {
        let l = line(energy);
        assert_eq!(l.emitting, 4096, "every replica emits it");
        assert_close(
            l.estimate.relative_std_dev().unwrap(),
            (fd * fd + own * own).sqrt(),
            0.05,
            "rate sigma",
        );
        assert_eq!(l.energy_estimate.nominal, energy);
        assert_close(l.energy_estimate.mean.unwrap(), energy, 1e-5, "energy mean");
        assert_close(l.energy_estimate.std_dev.unwrap(), de, 0.05, "energy sigma");
    }
    // A line with no sigma of its own moves with the spectrum's normalisation.
    let weak = line(LINE_826);
    assert_close(
        weak.estimate.relative_std_dev().unwrap(),
        fd,
        0.05,
        "FD alone",
    );
    assert_eq!(weak.energy_estimate.std_dev, Some(0.0));

    // Matched on the nominal energy, so the spectrum has exactly the nominal
    // lines, and the nominal rates are Material.decay_photon_spectrum's to the
    // last bit.
    let material = results.get_material(0, 1).unwrap();
    let nominal = yani_decay::decay_photon_lines(
        &material.get_atoms_per_barn_cm().unwrap(),
        material.volume.unwrap(),
        &chain,
    );
    assert_eq!(lines.len(), nominal.len());
    for (l, (energy, rate)) in lines.iter().zip(&nominal) {
        assert_eq!(l.energy, *energy);
        assert_eq!(l.estimate.nominal, *rate);
    }

    // The inventory never saw a photon.
    let activity = results.activity_uncertainty(0, 1, &chain).unwrap().unwrap();
    assert_eq!(activity.std_dev, Some(0.0));
}

/// A contact dose is linear in the line intensities at a fixed inventory, so a
/// normalisation common to every line moves it by exactly that much.
#[test]
fn a_contact_dose_moves_by_the_common_normalisation() {
    let fd = 0.05;
    let chain = chain(fd, false);
    let results = run(&chain, vec![Source::DecayPhotonLines], 4096);
    let dose = results
        .contact_dose_uncertainty(0, 1, &chain, yani_decay::DoseQuantity::AbsorbedAir, 2.0)
        .unwrap()
        .expect("an ensemble");
    assert_close(dose.relative_std_dev().unwrap(), fd, 0.05, "dose sigma");
}

/// With the source off, nothing about the photons moves and the report says
/// they were held.
#[test]
fn switched_off_the_lines_are_held_and_reported() {
    let chain = chain(0.05, true);
    let results = run(&chain, vec![Source::HalfLife], 64);
    let info = &results.uncertainty_info[&0];
    assert!(info.decay_photon_lines_perturbed.is_empty());
    assert!(info
        .not_perturbed
        .iter()
        .any(|s| s == "decay photon line energy and intensity, and continuum normalisation"));
    let lines = results
        .photon_spectrum_uncertainty(0, 1, &chain)
        .unwrap()
        .expect("an ensemble");
    for l in lines {
        assert_eq!(l.energy_estimate.std_dev.unwrap_or(0.0), 0.0);
    }
}

/// A nuclide whose photon data states no sigma is listed as such, so a zero
/// spread on its lines does not read as an exact evaluation.
#[test]
fn photons_without_a_sigma_are_listed() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let chain = Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"));
    let results = run(&chain, vec![Source::DecayPhotonLines], 8);
    let info = &results.uncertainty_info[&0];
    assert!(
        info.no_decay_photon_line_uncertainty.contains("Co60"),
        "{info:?}"
    );
    assert!(info.decay_photon_lines_perturbed.is_empty());
    assert!(info.has_gaps());
}
