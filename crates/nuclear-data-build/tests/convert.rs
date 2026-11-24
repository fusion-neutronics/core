//! The conversion stage on real evaluations: what it writes, what a rerun
//! keeps, and what it does with an evaluation that fails.
//!
//! The photon tests need no NJOY. The neutron test runs NJOY when one the
//! builder accepts (2016.80 or later) is on PATH, and skips otherwise, as
//! the converter's own NJOY tests do.

mod common;

use std::path::{Path, PathBuf};

use common::Scratch;
use nuclear_data_build::convert::{self, Options, Particle};
use nuclear_data_build::identify::identify;
use nuclear_data_build::recipe::Sublibrary;
use nuclear_data_build::Material;
use yamc_convert::entry::PhotonTabulations;

const PHOTOAT_H: &[u8] = include_bytes!("../../endf/fixtures/photoat-001_H_000.endf.xz");
const ATOM_H: &[u8] = include_bytes!("../../endf/fixtures/atom-001_H_000.endf.xz");
const LI6: &[u8] = include_bytes!("../../endf/fixtures/n-003_Li_006_trimmed.endf.xz");

const TEMPERATURES: &[f64] = &[293.6];

fn unxz(compressed: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut lzma_rust2::XzReader::new(compressed, true), &mut out).unwrap();
    out
}

/// The materials of a sources folder holding the given files, one folder
/// per sublibrary, as the extract stage leaves them.
fn materials(scratch: &Scratch, files: &[(Sublibrary, &str, &[u8])]) -> Vec<Material> {
    let mut folders: Vec<(Sublibrary, PathBuf)> = Vec::new();
    for (sub, name, xz) in files {
        let dir = scratch.dir(&format!("sources/{}", sub.dir()));
        std::fs::write(dir.join(name), unxz(xz)).unwrap();
        if !folders.iter().any(|(s, _)| s == sub) {
            folders.push((*sub, dir));
        }
    }
    identify(&folders).unwrap()
}

fn photon_h(scratch: &Scratch) -> Vec<Material> {
    materials(
        scratch,
        &[
            (Sublibrary::Photoatomic, "photoat-001_H_000.endf", PHOTOAT_H),
            (Sublibrary::AtomicRelaxation, "atom-001_H_000.endf", ATOM_H),
        ],
    )
}

fn tabulation_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc-convert/data")
}

fn options<'a>(
    particles: &'a [Particle],
    tabulations: &'a Path,
    njoy_exec: &'a str,
) -> Options<'a> {
    Options {
        library: "endf-b8.1",
        release: "2026-11-06",
        converter: "yani-python 0.0.0.dev0 (core 0123abc)",
        njoy_exec,
        temperatures: TEMPERATURES,
        particles,
        nuclides: None,
        tabulations: Some(tabulations_in(tabulations)),
        memory_budget: None,
        workers: Some(2),
        stop: None,
        progress: None,
    }
}

fn tabulations_in(dir: &Path) -> PhotonTabulations<'static> {
    let leak = |name: &str| -> &'static Path { Box::leak(dir.join(name).into_boxed_path()) };
    PhotonTabulations {
        compton_profiles: leak("compton_profiles_biggs1975.txt"),
        density_effect: leak("density_effect_sternheimer1982.txt"),
        bremsstrahlung: leak("bremsstrahlung_seltzer_berger1986.txt"),
    }
}

fn keys(names: &[&str]) -> std::collections::BTreeSet<String> {
    names.iter().map(|s| s.to_string()).collect()
}

#[test]
fn a_photon_element_is_converted_with_its_relaxation_and_tabulations() {
    let scratch = Scratch::new("convert-photon");
    let found = photon_h(&scratch);
    let root = scratch.dir("lib");
    let tabs = tabulation_dir();
    let got =
        convert::convert(&found, &root, &options(&[Particle::Photon], &tabs, "njoy")).unwrap();

    assert_eq!(got.converted, keys(&["photon/H"]));
    assert!(got.failed.is_empty() && got.reused.is_empty());
    let dir = root.join("photon/H.arrow");
    for table in [
        "element.arrow",
        "subshells.arrow",
        "compton.arrow",
        "bremsstrahlung.arrow",
    ] {
        assert!(dir.join(table).is_file(), "{table} is missing");
    }
    let marker: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("version.json")).unwrap()).unwrap();
    assert_eq!(marker["library"], "endf-b8.1");
    assert_eq!(marker["data_version"], "2026-11-06");
    assert_eq!(marker["converter"], "yani-python 0.0.0.dev0 (core 0123abc)");
}

#[test]
fn a_rerun_keeps_finished_folders_and_reconverts_any_other() {
    let scratch = Scratch::new("convert-resume");
    let found = photon_h(&scratch);
    let root = scratch.dir("lib");
    let tabs = tabulation_dir();
    let particles = [Particle::Photon];
    convert::convert(&found, &root, &options(&particles, &tabs, "njoy")).unwrap();

    let again = convert::convert(&found, &root, &options(&particles, &tabs, "njoy")).unwrap();
    assert_eq!(again.reused, keys(&["photon/H"]));
    assert!(again.converted.is_empty());

    // Another release's folder is not this release's, so it is redone.
    let mut other = options(&particles, &tabs, "njoy");
    other.release = "2026-12-01";
    let redone = convert::convert(&found, &root, &other).unwrap();
    assert_eq!(redone.converted, keys(&["photon/H"]));

    // So is a folder an interrupted run left: a table and no version.json.
    let dir = root.join("photon/H.arrow");
    std::fs::remove_file(dir.join("version.json")).unwrap();
    std::fs::write(dir.join("left-behind"), b"").unwrap();
    let resumed = convert::convert(&found, &root, &other).unwrap();
    assert_eq!(resumed.converted, keys(&["photon/H"]));
    assert!(
        !dir.join("left-behind").exists(),
        "the unfinished folder was not cleared"
    );
}

#[test]
fn a_failed_evaluation_is_recorded_and_the_rest_still_convert() {
    let scratch = Scratch::new("convert-failure");
    let mut found = photon_h(&scratch);
    found.extend(materials(
        &scratch,
        &[(Sublibrary::Neutron, "n-003_Li_006.endf", LI6)],
    ));
    let root = scratch.dir("lib");
    let tabs = tabulation_dir();
    let particles = [Particle::Neutron, Particle::Photon];
    let got = convert::convert(
        &found,
        &root,
        &options(&particles, &tabs, "no-such-njoy-anywhere"),
    )
    .unwrap();

    assert_eq!(got.converted, keys(&["photon/H"]));
    assert_eq!(
        got.failed.keys().cloned().collect::<Vec<_>>(),
        ["neutron/Li6"]
    );
    let reason = &got.failed["neutron/Li6"];
    assert!(reason.contains("no-such-njoy-anywhere"), "{reason}");
    let record = std::fs::read_to_string(root.join(".build/failed/neutron-Li6.txt")).unwrap();
    assert!(record.contains("n-003_Li_006.endf") && record.contains(reason.as_str()));
    assert!(!root.join("neutron/Li6.arrow").exists());
}

#[test]
fn a_nuclide_list_converts_only_what_it_names_and_refuses_a_name_it_cannot_find() {
    let scratch = Scratch::new("convert-select");
    let mut found = photon_h(&scratch);
    found.extend(materials(
        &scratch,
        &[(Sublibrary::Neutron, "n-003_Li_006.endf", LI6)],
    ));
    let root = scratch.dir("lib");
    let tabs = tabulation_dir();
    let particles = [Particle::Neutron, Particle::Photon];

    // "H" is an element, so it selects the photon H and no neutron.
    let wanted = vec!["H".to_string()];
    let mut only_h = options(&particles, &tabs, "no-such-njoy-anywhere");
    only_h.nuclides = Some(&wanted);
    let got = convert::convert(&found, &root, &only_h).unwrap();
    assert_eq!(got.converted, keys(&["photon/H"]));
    assert!(got.failed.is_empty());

    let typo = vec!["Li66".to_string(), "H".to_string()];
    let mut wrong = options(&particles, &tabs, "njoy");
    wrong.nuclides = Some(&typo);
    let err = convert::convert(&found, &root, &wrong).unwrap_err();
    assert!(err.contains("Li66") && !err.contains(", H"), "{err}");
}

#[test]
fn photons_without_tabulations_given_use_the_bundled_ones() {
    let scratch = Scratch::new("convert-no-tabs");
    let found = photon_h(&scratch);
    let root = scratch.dir("lib");
    let tabs = tabulation_dir();
    let mut opts = options(&[Particle::Photon], &tabs, "njoy");
    opts.tabulations = None;
    let got = convert::convert(&found, &root, &opts).unwrap();
    assert_eq!(got.converted, keys(&["photon/H"]));
    for table in ["compton.arrow", "bremsstrahlung.arrow"] {
        assert!(root.join("photon/H.arrow").join(table).is_file(), "{table}");
    }
}

#[test]
fn a_stopped_run_starts_nothing() {
    let scratch = Scratch::new("convert-stop");
    let found = photon_h(&scratch);
    let root = scratch.dir("lib");
    let tabs = tabulation_dir();
    let stop = std::sync::atomic::AtomicBool::new(true);
    let mut opts = options(&[Particle::Photon], &tabs, "njoy");
    opts.stop = Some(&stop);
    let got = convert::convert(&found, &root, &opts).unwrap();
    assert_eq!(got.not_started, keys(&["photon/H"]));
    assert!(got.converted.is_empty() && got.failed.is_empty());
}

/// The whole neutron route, through NJOY. Skips without an NJOY the builder
/// accepts, so CI (which has none) passes it by.
#[test]
fn a_neutron_evaluation_is_converted_through_njoy_with_its_range_indexes() {
    if let Err(e) = endf::njoy::require("njoy", endf::njoy::Flavour::Upstream) {
        eprintln!("skip: {e}");
        return;
    }
    let scratch = Scratch::new("convert-neutron");
    let found = materials(&scratch, &[(Sublibrary::Neutron, "n-003_Li_006.endf", LI6)]);
    let root = scratch.dir("lib");
    let tabs = tabulation_dir();
    let got =
        convert::convert(&found, &root, &options(&[Particle::Neutron], &tabs, "njoy")).unwrap();
    assert_eq!(got.converted, keys(&["neutron/Li6"]), "{:?}", got.failed);
    let marker: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("neutron/Li6.arrow/version.json")).unwrap(),
    )
    .unwrap();
    assert!(!marker["reaction_ranges"].is_null() && !marker["energy_ranges"].is_null());
}
