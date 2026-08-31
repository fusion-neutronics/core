//! Tritium production in a breeder blanket, with its nuclear-data uncertainty.
//!
//! The tracking benchmark for the uncertainty work. It exists to be re-run as
//! each source is added (issue #520), so the inventory sigma can be watched
//! growing rather than asserted about in the abstract, and it prints a row for
//! that purpose as well as asserting.
//!
//! # The case
//!
//! Li4SiO4 breeder, Be9 multiplier, steel structure and water coolant, under a
//! fusion-weighted spectrum. Tritium comes overwhelmingly from Li6(n,t)He4,
//! which is MT=105 -- and MT=105 is precisely the channel the committed Li6
//! evaluation keeps covariance for. So the benchmark's headline number is
//! driven by real MF=33 data rather than by a synthetic matrix.
//!
//! # Why the fixture is built rather than downloaded
//!
//! Nothing published carries `covariance.arrow` yet, so the Li6 directory is
//! copied out of the cache and the real converter writes the section into it
//! from the committed ENDF evaluation. Every other nuclide is used as cached,
//! which means they have no covariance and contribute nothing to the spread.
//! That is not a defect of the benchmark: it is the coverage gap the report
//! exists to make visible, and it is what the next source will close.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const LI6_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-003_Li_006_trimmed.endf.xz");

/// Li6 enrichment, in atom percent of the lithium.
///
/// Natural lithium is 7.6% Li6. Breeder designs enrich it because Li6(n,t) is
/// the reaction that matters, and 60% is a representative choice.
const LI6_ENRICHMENT: f64 = 0.60;

/// A fusion-weighted spectrum: most of the flux at 14 MeV, a moderated tail.
///
/// Four groups rather than a named 709-group structure so the case can be read
/// on the page. The shape is what drives the Li6(n,t) rate, and Li6(n,t) is
/// large at thermal energies and again at 14 MeV, so both ends matter.
const GROUPS: [f64; 5] = [1.0e-5, 1.0e0, 1.0e5, 1.0e6, 2.0e7];
const FLUX: [f64; 4] = [2.0e13, 1.0e13, 3.0e13, 1.4e14];

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

/// Every nuclide the blanket needs, mapped to the data directory to read it
/// from. Li6 gets a writable copy carrying `covariance.arrow`.
fn data_paths(tmp: &Path) -> Option<HashMap<String, String>> {
    let names = [
        "Li6", "Li7", "Si28", "Si29", "Si30", "O16", "Be9", "Fe54", "Fe56", "Fe57", "Fe58", "Cr52",
        "H1",
    ];
    let mut paths = HashMap::new();
    for name in names {
        paths.insert(name.to_string(), yamc_test_cache::nuclide(name)?);
    }

    // Li6 is copied so the converter can write the covariance section beside
    // the cross sections, without touching the shared cache.
    let li6 = tmp.join("Li6.arrow");
    std::fs::create_dir_all(&li6).expect("mkdir");
    let cached = PathBuf::from(&paths["Li6"]);
    for entry in std::fs::read_dir(&cached).expect("read cached Li6") {
        let entry = entry.expect("dir entry");
        if entry.path().is_file() {
            std::fs::copy(entry.path(), li6.join(entry.file_name())).expect("copy section");
        }
    }

    let evaluation = tmp.join("li6.endf");
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &LI6_ENDF[..], &mut raw).expect("fixture decompresses");
    std::fs::write(&evaluation, raw).expect("write evaluation");
    let material = endf::Material::from_file(&evaluation).expect("Li6 parses");
    assert!(
        yamc_convert::covariance::write_covariance(&material, &li6).expect("covariance writes"),
        "the Li6 fixture must carry MF=33 for this benchmark to mean anything"
    );

    paths.insert("Li6".to_string(), li6.to_string_lossy().into_owned());
    Some(paths)
}

/// Li4SiO4 breeder with a Be9 multiplier, steel structure and water coolant.
///
/// Atom fractions, roughly a 60/20/10/10 volume split flattened into one
/// homogenised material. A real blanket is layered; homogenising it keeps the
/// benchmark about the uncertainty rather than about geometry.
fn blanket(paths: &HashMap<String, String>) -> Material {
    let li = 4.0;
    let composition = HashMap::from([
        ("Li6".to_string(), li * LI6_ENRICHMENT),
        ("Li7".to_string(), li * (1.0 - LI6_ENRICHMENT)),
        // Silicon at natural abundance.
        ("Si28".to_string(), 1.0 * 0.9223),
        ("Si29".to_string(), 1.0 * 0.0467),
        ("Si30".to_string(), 1.0 * 0.0310),
        // Oxygen from the ceramic and from the coolant.
        ("O16".to_string(), 4.0 + 0.6),
        ("Be9".to_string(), 2.5),
        // Steel, as its major isotopes.
        ("Fe54".to_string(), 0.9 * 0.0585),
        ("Fe56".to_string(), 0.9 * 0.9175),
        ("Fe57".to_string(), 0.9 * 0.0212),
        ("Fe58".to_string(), 0.9 * 0.0028),
        ("Cr52".to_string(), 0.25),
        // Water.
        ("H1".to_string(), 1.2),
    ]);

    let mut m = Material::new(composition, "atom", "sum", None).expect("blanket material");
    m.density = Some(2.4);
    m.set_temperature("294");
    m.read_nuclear_data(paths, None).expect("read blanket data");
    m
}

fn spectra() -> Vec<MultigroupSpectrum> {
    let total: f64 = FLUX.iter().sum();
    vec![MultigroupSpectrum {
        boundaries: GROUPS.to_vec(),
        masses: FLUX.iter().map(|f| f / total).collect(),
        relative_std_dev: None,
    }]
}

/// One year of irradiation, then a day of cooling.
fn schedule() -> Vec<TransmuteStep> {
    vec![
        TransmuteStep {
            dt: 365.0 * 24.0 * 3600.0,
            irradiation: Some((0, FLUX.iter().sum())),
        },
        TransmuteStep {
            dt: 24.0 * 3600.0,
            irradiation: None,
        },
    ]
}

fn run(
    material: &mut Material,
    uncertainty: Option<&DataUncertainty>,
) -> yani_transmute::TransmutationResults {
    transmute_material(
        material,
        &spectra(),
        &schedule(),
        chain(),
        &Default::default(),
        Default::default(),
        uncertainty,
    )
    .expect("transmute")
}

/// Tritium comes back with an uncertainty, and it comes from Li6(n,t).
///
/// Prints the row this benchmark exists to produce. Re-run it as each source
/// lands and the sigma column is the thing to watch.
#[test]
fn tritium_production_carries_a_nuclear_data_uncertainty() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(paths) = data_paths(tmp.path()) else {
        eprintln!("skipping tritium_breeder (nuclear-data fixtures missing)");
        return;
    };
    let mut material = blanket(&paths);
    let id = material.material_id.unwrap_or(0);

    let results = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 20260825,
            samples: Some(256),
            sources: vec![Source::CrossSections],
        }),
    );

    let mean = results
        .get_nuclide_density(id, "H3", 1)
        .expect("the blanket breeds tritium");
    let sigma = results
        .get_nuclide_uncertainty(id, "H3", 1)
        .expect("uncertainty was requested");
    let info = results.uncertainty_info.clone().expect("info is reported");

    println!(
        "\n| sources | H3 [atom/b-cm] | sigma | rel |\n\
         |---|---|---|---|\n\
         | {} | {mean:.4e} | {sigma:.4e} | {:.2}% |\n\
         perturbed: {:?}   no covariance: {} nuclides   samples: {}",
        info.sources.join(" + "),
        100.0 * sigma / mean,
        info.perturbed,
        info.no_covariance_data.len(),
        info.samples,
    );

    assert!(mean > 0.0, "Li6(n,t) must breed tritium");
    assert!(
        sigma > 0.0,
        "Li6 carries MF=33 for MT=105, which IS the tritium channel, so \
         perturbing it must move H3"
    );
    assert!(
        sigma < mean,
        "a relative uncertainty above 100% on the dominant breeding channel \
         would mean the fold is wrong, not the evaluation: {mean:e} +/- {sigma:e}"
    );
    assert!(
        info.perturbed.contains("Li6"),
        "Li6 is the nuclide carrying covariance here: {:?}",
        info.perturbed
    );
}

/// The other blanket nuclides have no covariance, and that is reported.
///
/// The benchmark's headline sigma is Li6-only today. Anyone reading it has to
/// be able to tell that from a claim that the rest are well known.
#[test]
fn the_uncovered_blanket_nuclides_are_named() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(paths) = data_paths(tmp.path()) else {
        eprintln!("skipping tritium_breeder (nuclear-data fixtures missing)");
        return;
    };
    let mut material = blanket(&paths);

    let results = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 1,
            samples: Some(32),
            sources: vec![Source::CrossSections],
        }),
    );
    let info = results.uncertainty_info.expect("info");

    assert!(
        info.has_gaps(),
        "every nuclide but Li6 is used as cached and has no covariance, so the \
         report must say the coverage is partial"
    );
    assert!(
        !info.no_covariance_data.is_empty(),
        "the uncovered nuclides must be named, not merely counted"
    );
    assert!(
        info.perturbed
            .intersection(&info.no_covariance_data)
            .next()
            .is_none(),
        "a nuclide cannot be both perturbed and lacking data"
    );
}

/// Switching every source off is the zero row of the tracking table.
#[test]
fn the_baseline_row_has_no_uncertainty_and_says_why() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(paths) = data_paths(tmp.path()) else {
        eprintln!("skipping tritium_breeder (nuclear-data fixtures missing)");
        return;
    };
    let mut material = blanket(&paths);
    let id = material.material_id.unwrap_or(0);

    let plain = run(&mut material, None);
    assert_eq!(
        plain.get_nuclide_uncertainty(id, "H3", 1),
        None,
        "not requested reads as None, which is not the same as zero"
    );

    // And the means must not move when uncertainty is switched on.
    let with = run(
        &mut material,
        Some(&DataUncertainty {
            seed: 2,
            samples: Some(32),
            sources: vec![Source::CrossSections],
        }),
    );
    assert_eq!(
        plain.get_nuclide_density(id, "H3", 1),
        with.get_nuclide_density(id, "H3", 1),
        "the tritium mean is bit-identical with and without uncertainty"
    );
}

/// Adding the flux error on top of the cross sections grows the spread.
///
/// The row this benchmark exists to produce, one source at a time. Sources are
/// independent, so the combined sigma must exceed either alone and must not
/// exceed their quadrature sum by more than sampling noise.
#[test]
fn flux_and_cross_sections_combine() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(paths) = data_paths(tmp.path()) else {
        eprintln!("skipping tritium_breeder (nuclear-data fixtures missing)");
        return;
    };
    let mut material = blanket(&paths);
    let id = material.material_id.unwrap_or(0);

    // 5% on every bin, the sort of thing a converged transport tally gives.
    let total: f64 = FLUX.iter().sum();
    let with_sigma = vec![MultigroupSpectrum {
        boundaries: GROUPS.to_vec(),
        masses: FLUX.iter().map(|f| f / total).collect(),
        relative_std_dev: Some(vec![0.05; FLUX.len()]),
    }];

    let mut sigma_of = |sources: Vec<Source>| {
        let r = transmute_material(
            &mut material,
            &with_sigma,
            &schedule(),
            chain(),
            &Default::default(),
            Default::default(),
            Some(&DataUncertainty {
                seed: 20260826,
                samples: Some(256),
                sources,
            }),
        )
        .expect("transmute");
        let mean = r.get_nuclide_density(id, "H3", 1).expect("H3");
        let sd = r.get_nuclide_uncertainty(id, "H3", 1).expect("requested");
        (mean, sd)
    };

    let (mean, xs_only) = sigma_of(vec![Source::CrossSections]);
    let (_, flux_only) = sigma_of(vec![Source::FluxSpectrum]);
    let (_, both) = sigma_of(vec![Source::CrossSections, Source::FluxSpectrum]);

    println!(
        "\n| sources | H3 [atom/b-cm] | sigma | rel |\n\
         |---|---|---|---|\n\
         | cross_sections | {mean:.4e} | {xs_only:.4e} | {:.2}% |\n\
         | flux_spectrum | {mean:.4e} | {flux_only:.4e} | {:.2}% |\n\
         | both | {mean:.4e} | {both:.4e} | {:.2}% |",
        100.0 * xs_only / mean,
        100.0 * flux_only / mean,
        100.0 * both / mean,
    );

    assert!(flux_only > 0.0, "a 5% flux error must move tritium");
    assert!(
        both > xs_only && both > flux_only,
        "two independent sources must exceed either alone: \
         xs {xs_only:e}, flux {flux_only:e}, both {both:e}"
    );

    // Independent sources add in quadrature. Sampling noise on 256 replicas is
    // a few percent of sigma, so the band is loose deliberately.
    let quadrature = (xs_only * xs_only + flux_only * flux_only).sqrt();
    let ratio = both / quadrature;
    assert!(
        (0.85..1.15).contains(&ratio),
        "the combined sigma should be the quadrature sum to within sampling \
         noise, got {both:e} against {quadrature:e} (ratio {ratio:.3})"
    );
}
