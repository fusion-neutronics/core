//! Every photon-line uncertainty MT=457 gives, from the tape to yani's reader.
//!
//! Issue #163: the sigmas were parsed and then dropped where
//! `decay/sources.arrow` was written, and the gamma and x-ray spectra were
//! merged into one row, which lost which normalisation each line shares. These
//! tests read three real evaluations that between them state all of it: a
//! JENDL-5.0 normalisation sigma kept apart from the lines (Sn111), a JEFF-4.0
//! continuum normalisation sigma (Cf252), and an ENDF/B-VIII.1 nuclide that
//! emits both gammas and x-rays (In116m1).

use endf::chain::Chain;
use endf::{Decay, Material};
use yani::{ChainNuclide, DecaySource, DecaySourceDistribution};

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("../../endf/fixtures/", $name))
    };
}

fn material(compressed: &[u8]) -> Material {
    let mut text = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut text).expect("fixture decompresses");
    Material::from_str(&String::from_utf8(text).expect("fixture is UTF-8")).expect("parses")
}

/// The tape's decay data, and the nuclide as yani reads it back after a
/// conversion. `tag` keeps concurrent tests' directories apart.
fn converted(compressed: &[u8], name: &str, tag: &str) -> (Decay, ChainNuclide) {
    let decay = [material(compressed)];
    let chain = Chain::from_endf(
        &decay,
        &[],
        &endf::chain::q_values(&[]),
        &endf::chain::DEFAULT_REACTIONS,
    )
    .expect("chain builds");
    let sources = yani_convert::decay_sources(&decay).expect("sources read");
    let dir = std::env::temp_dir().join(format!("yani-convert-lines-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    yani_convert::write_decay(&chain, &sources, &dir.join("decay")).expect("decay written");

    // The file has the declared columns, the new ones last.
    let file = std::fs::File::open(dir.join("decay/sources.arrow")).expect("written");
    let reader = arrow_ipc::reader::FileReader::try_new(file, None).expect("readable");
    let written: Vec<String> = reader
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let declared: Vec<String> = nuclear_data_schema::section("decay/sources.arrow")
        .expect("declared")
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(written, declared);

    let (back, _branch) =
        yani::parse_chain_parts(&dir.join("decay"), None, None, None).expect("yani reads it");
    let _ = std::fs::remove_dir_all(&dir);
    (
        Decay::from_material(&decay[0]).expect("decay reads"),
        back[name].clone(),
    )
}

fn photon_rows<'a>(nuclide: &'a ChainNuclide, radiation: &str) -> Vec<&'a DecaySource> {
    nuclide
        .sources
        .iter()
        .filter(|s| s.particle == "photon" && s.radiation.as_deref() == Some(radiation))
        .collect()
}

/// JENDL-5.0 Sn111 states its gamma normalisation sigma in dFD. It comes back
/// as written, and every line keeps its own dRI and dER in the order the tape
/// lists them, each intensity sigma scaled exactly as its intensity is.
#[test]
fn a_spectrum_normalisation_sigma_and_each_line_sigma_survive() {
    let (decay, sn111) = converted(fixture!("dec-050-Sn-111.jendl5.endf.xz"), "Sn111", "sn111");
    let tape = &decay.spectra["gamma"];
    assert_eq!(tape.discrete_normalization, (1.0, 0.1509434));
    let (lambda, _) = decay.decay_constant().expect("unstable");

    let rows = photon_rows(&sn111, "gamma");
    assert_eq!(rows.len(), 1, "one row for the one gamma spectrum");
    let DecaySourceDistribution::Discrete {
        energies,
        intensities,
    } = &rows[0].distribution
    else {
        panic!("the gamma spectrum is lines");
    };
    let stated = rows[0]
        .uncertainty
        .as_deref()
        .expect("the tape states sigmas");
    assert_eq!(stated.normalization, Some(1.0));
    assert_eq!(stated.normalization_uncertainty, Some(0.1509434));
    assert_eq!(stated.covariance, None, "LCOV is 0");

    let sigmas = stated
        .intensity_uncertainties
        .as_ref()
        .expect("lines have sigmas");
    let energy_sigmas = stated
        .energy_uncertainties
        .as_ref()
        .expect("and energy sigmas");
    assert_eq!(energies.len(), tape.discrete.len());
    for (i, line) in tape.discrete.iter().enumerate() {
        let norm = tape.discrete_normalization.0;
        assert_eq!(energies[i], line.energy.0, "tape order, line {i}");
        assert_eq!(intensities[i], lambda * norm * line.intensity.0);
        assert_eq!(sigmas[i], lambda * norm * line.intensity.1);
        assert_eq!(energy_sigmas[i], line.energy.1);
    }
    // The first line, read off the tape by hand: 265.2 +- 0.6 keV at
    // RI = 8.745e-4 +- 7.95e-5.
    assert_eq!((energies[0], energy_sigmas[0]), (2.652e5, 600.0));
    assert_eq!(sigmas[0] / intensities[0], 7.95e-5 / 8.745e-4);
}

/// A sigma the evaluation wrote as 0.0 stays 0.0 rather than becoming null,
/// so the file says what the tape says. Readers take either as "not stated".
#[test]
fn a_zero_sigma_is_stored_as_the_tape_writes_it() {
    let (decay, sn111) = converted(fixture!("dec-050-Sn-111.jendl5.endf.xz"), "Sn111", "zeros");
    let tape = &decay.spectra["xray"];
    let rows = photon_rows(&sn111, "xray");
    assert_eq!(rows.len(), 1);
    let stated = rows[0].uncertainty.as_deref().expect("stated");
    assert_eq!(stated.normalization, Some(tape.discrete_normalization.0));
    assert_eq!(
        stated.normalization_uncertainty,
        Some(tape.discrete_normalization.1)
    );
    let zeros = tape
        .discrete
        .iter()
        .zip(stated.energy_uncertainties.as_ref().unwrap())
        .filter(|(line, stored)| line.energy.1 == 0.0 && **stored == 0.0)
        .count();
    assert!(
        zeros > 0,
        "the fixture has x-ray lines with dER written as 0.0"
    );
}

/// JEFF-4.0 Cf252's continuum carries FC with a sigma; it arrives on the
/// tabular row, and the per-line lists stay null there.
#[test]
fn a_continuum_keeps_its_normalisation_sigma_and_no_per_point_sigma() {
    let (decay, cf252) = converted(fixture!("dec-098_Cf_252.jeff40.endf.xz"), "Cf252", "cf252");
    let tape = &decay.spectra["gamma"];
    assert!(
        tape.continuous_normalization.1 > 0.0,
        "the fixture states dFC"
    );
    let continuum: Vec<&DecaySource> = photon_rows(&cf252, "gamma")
        .into_iter()
        .filter(|s| matches!(s.distribution, DecaySourceDistribution::Tabular { .. }))
        .collect();
    assert_eq!(continuum.len(), 1);
    let stated = continuum[0].uncertainty.as_deref().expect("stated");
    assert_eq!(stated.normalization, Some(tape.continuous_normalization.0));
    assert_eq!(
        stated.normalization_uncertainty,
        Some(tape.continuous_normalization.1)
    );
    assert_eq!(stated.intensity_uncertainties, None);
    assert_eq!(stated.energy_uncertainties, None);
}

/// In116m1 emits gammas and x-rays. They are two rows now, each with its own
/// spectrum's normalisation, and together they are exactly the one merged
/// photon distribution `Decay::sources` gives, to the last bit.
#[test]
fn gamma_and_xray_spectra_are_separate_rows_that_merge_to_the_old_one() {
    let (decay, in116m1) = converted(fixture!("dec-049_In_116m1.endf.xz"), "In116_m1", "in116");
    for radiation in ["gamma", "xray"] {
        let rows = photon_rows(&in116m1, radiation);
        assert_eq!(rows.len(), 1, "{radiation}");
        let stated = rows[0].uncertainty.as_deref().expect("stated");
        assert_eq!(
            stated.normalization,
            Some(decay.spectra[radiation].discrete_normalization.0)
        );
    }

    let endf::univariate::Univariate::Discrete(merged) = &decay.sources().unwrap()["photon"] else {
        panic!("both photon spectra are lines, so they merge into one");
    };
    let (energies, intensities): (Vec<f64>, Vec<f64>) = in116m1.photon_lines().into_iter().unzip();
    assert_eq!(energies, merged.x);
    assert_eq!(intensities, merged.p);
}

/// No library ships a photon covariance, so one is set by hand on the JEFF-4.0
/// Cf252 gamma spectrum's rows (LCOV=3: its lines and its continuum). What
/// `write_decay` writes, yani reads back as written, with LS on the lines
/// only. The two lists differ in length so a swap of the columns shows.
#[test]
fn a_stated_covariance_round_trips_through_the_file() {
    let decay = [material(fixture!("dec-098_Cf_252.jeff40.endf.xz"))];
    let chain = Chain::from_endf(
        &decay,
        &[],
        &endf::chain::q_values(&[]),
        &endf::chain::DEFAULT_REACTIONS,
    )
    .expect("chain builds");
    let mut sources = yani_convert::decay_sources(&decay).expect("sources read");
    let lines = endf::SpectrumCovariance {
        ls: Some(1),
        lb: 5,
        energies: vec![1.0e5, 2.0e6],
        values: vec![1.0e-4, 2.0e-5, 3.0e-4],
    };
    let continuum = endf::SpectrumCovariance {
        ls: None,
        lb: 2,
        energies: vec![0.0, 5.0e6, 1.0e7],
        values: vec![0.01, 0.02, 0.0],
    };
    let mut set = 0;
    for row in sources.get_mut("Cf252").expect("Cf252 has sources") {
        if row.radiation == "gamma" {
            row.covariance = Some(if row.kind == "discrete" {
                lines.clone()
            } else {
                continuum.clone()
            });
            set += 1;
        }
    }
    assert_eq!(set, 2, "the gamma spectrum has lines and a continuum");

    let dir = std::env::temp_dir().join(format!("yani-convert-covariance-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    yani_convert::write_decay(&chain, &sources, &dir.join("decay")).expect("decay written");
    let (back, _branch) =
        yani::parse_chain_parts(&dir.join("decay"), None, None, None).expect("yani reads it");
    let _ = std::fs::remove_dir_all(&dir);

    let covariance = |tabular: bool| {
        let rows: Vec<&DecaySource> = photon_rows(&back["Cf252"], "gamma")
            .into_iter()
            .filter(|s| {
                matches!(s.distribution, DecaySourceDistribution::Tabular { .. }) == tabular
            })
            .collect();
        assert_eq!(rows.len(), 1);
        rows[0]
            .uncertainty
            .as_deref()
            .expect("stated")
            .covariance
            .clone()
    };
    assert_eq!(
        covariance(false),
        Some(yani::SourceCovariance {
            ls: Some(1),
            lb: 5,
            energies: lines.energies.clone(),
            values: lines.values.clone(),
        })
    );
    assert_eq!(
        covariance(true),
        Some(yani::SourceCovariance {
            ls: None,
            lb: 2,
            energies: continuum.energies.clone(),
            values: continuum.values.clone(),
        })
    );
    for row in photon_rows(&back["Cf252"], "xray") {
        assert_eq!(row.uncertainty.as_deref().expect("stated").covariance, None);
    }
}
