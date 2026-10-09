//! `resonance_parameters.arrow`: MF=2 and MF=32 written by the converter as
//! ENDF-6 text, read back by the loader, and parsed again.
//!
//! The section's whole claim is that it is lossless: the parameters and their
//! covariance a consumer gets from the stored text are the ones the converter
//! read off the tape. So the check is on what the next step computes from
//! them, [`endf::resonance_covariance::range_covariances`], as well as on the
//! parsed sections and the text itself.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use endf::Material;
use yamc_nuclide::arrow::covariance_arrow::read_resonance_parameters;
use yamc_nuclide::arrow::nuclide_arrow::read_nuclide_from_arrow;
use yamc_nuclide::LoadScope;

/// Every fixture with MF=2 and MF=32, which between them cover Breit-Wigner
/// (LCOMP 0, 1 and 2), Reich-Moore, R-matrix limited and unresolved ranges.
const WITH_MF32: [(&str, &[u8]); 10] = [
    (
        "Na23",
        include_bytes!("../../endf/fixtures/n-011_Na_023_mf2_mf32.endf.xz"),
    ),
    (
        "Cl35",
        include_bytes!("../../endf/fixtures/n-017_Cl_035_mf2_mf32.endf.xz"),
    ),
    (
        "Cu63",
        include_bytes!("../../endf/fixtures/n-029_Cu_063_mf2_mf32.endf.xz"),
    ),
    (
        "Cu65",
        include_bytes!("../../endf/fixtures/n-029_Cu_065_mf2_mf32.endf.xz"),
    ),
    (
        "Rh103",
        include_bytes!("../../endf/fixtures/n-045_Rh_103_mf2_mf32.endf.xz"),
    ),
    (
        "Dy158",
        include_bytes!("../../endf/fixtures/n-066_Dy_158_mf2_mf32.endf.xz"),
    ),
    (
        "W183",
        include_bytes!("../../endf/fixtures/n-074_W_183_mf2_mf32.endf.xz"),
    ),
    (
        "W186",
        include_bytes!("../../endf/fixtures/n-074_W_186_mf2_mf32.endf.xz"),
    ),
    (
        "Th232",
        include_bytes!("../../endf/fixtures/n-090_Th_232_mf2_mf32.endf.xz"),
    ),
    (
        "Pu244",
        include_bytes!("../../endf/fixtures/n-094_Pu_244_mf2_mf32.endf.xz"),
    ),
];
/// Ca40 has MF=2 and no MF=32, which is what "absent" is tested against.
const CA40_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-020_Ca_040_mf2.endf.xz");

fn material(compressed: &[u8], dir: &Path, name: &str) -> Material {
    let path = dir.join(format!("{name}.endf"));
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut raw).expect("fixture decompresses");
    std::fs::write(&path, raw).expect("fixture writes");
    Material::from_file(&path).expect("evaluation parses")
}

#[test]
fn resonance_parameters_round_trip_through_the_section() {
    for (name, compressed) in WITH_MF32 {
        let tmp = tempfile::tempdir().expect("temp dir");
        let material = material(compressed, tmp.path(), name);

        let written =
            yamc_convert::resonance_parameters::write_resonance_parameters(&material, tmp.path())
                .expect("resonance parameters write");
        assert!(written, "{name} carries MF=32, so a file must be written");
        let stored = read_resonance_parameters(tmp.path(), name)
            .expect("reads")
            .expect("the file is there");

        // The text is the tape's, byte for byte.
        assert_eq!(stored.mf2_text, material.section_text[&(2, 151)], "{name}");
        assert_eq!(
            stored.mf32_text,
            material.section_text[&(32, 151)],
            "{name}"
        );

        // Parsed again, it is what the converter parsed.
        let (mf2, mf32) = stored.parse().expect("the stored text parses");
        assert_eq!(Some(&mf2), material.mf2(), "{name} MF=2");
        assert_eq!(Some(&mf32), material.mf32(), "{name} MF=32");

        // And the covariance the next step builds from it is the tape's: the
        // same parameters, matched to the same MF=2 resonances, with the same
        // matrix.
        let from_tape = endf::resonance_covariance::range_covariances(
            material.mf2().expect("MF=2"),
            material.mf32().expect("MF=32"),
        )
        .map_err(|e| e.to_string());
        let from_section = stored.range_covariances().map_err(|e| e.to_string());
        assert_eq!(from_section, from_tape, "{name} range covariances");
    }
}

#[test]
fn an_evaluation_without_mf32_writes_no_file_and_reads_as_none() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let material = material(CA40_ENDF, tmp.path(), "Ca40");
    assert!(material.mf2().is_some(), "Ca40 has MF=2, just no MF=32");
    assert!(
        !yamc_convert::resonance_parameters::write_resonance_parameters(&material, tmp.path())
            .expect("writes nothing without error")
    );
    assert!(!tmp.path().join("resonance_parameters.arrow").exists());
    assert_eq!(
        read_resonance_parameters(tmp.path(), "Ca40").expect("reads"),
        None
    );
}

/// A cached Fe56, if this machine has the fixtures.
fn fe56() -> Option<PathBuf> {
    yamc_test_cache::nuclide("Fe56").map(PathBuf::from)
}

/// The loader reads the section only when the scope asks for it: a default
/// load of a directory that has the file carries none of it, and a load that
/// asks gets exactly what was written.
#[test]
fn the_loader_reads_the_section_only_when_asked() {
    let Some(published) = fe56() else {
        eprintln!("skipping: the_loader_reads_the_section_only_when_asked (fixtures missing)");
        return;
    };
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().join("Fe56.arrow");
    std::fs::create_dir(&dir).expect("dir");
    for entry in std::fs::read_dir(&published).expect("published dir lists") {
        let entry = entry.expect("entry");
        if entry.file_type().expect("file type").is_file() {
            std::fs::copy(entry.path(), dir.join(entry.file_name())).expect("copies");
        }
    }
    // Any evaluation's MF=2 and MF=32 will do: the loader does not parse the
    // text, so this checks only that the section is gated and carried.
    let (name, compressed) = WITH_MF32[2];
    let material = material(compressed, tmp.path(), name);
    assert!(
        yamc_convert::resonance_parameters::write_resonance_parameters(&material, &dir)
            .expect("writes")
    );

    let activation = LoadScope::activation(HashSet::from([102]));
    let not_asked = read_nuclide_from_arrow(&dir, &activation).expect("loads");
    assert!(not_asked.resonance_parameters.is_none());
    assert!(!not_asked.load_scope.resonance_parameters);
    assert!(!LoadScope::full().resonance_parameters);

    let scope = activation.clone().with_resonance_parameters(true);
    let asked = read_nuclide_from_arrow(&dir, &scope).expect("loads");
    let stored = asked
        .resonance_parameters
        .as_ref()
        .expect("asked for and present");
    assert_eq!(stored.mf2_text, material.section_text[&(2, 151)]);
    assert_eq!(stored.mf32_text, material.section_text[&(32, 151)]);
    assert!(
        !activation.covers(&scope),
        "a load without MF=2 and MF=32 must not stand in for one that asked for them"
    );
}
