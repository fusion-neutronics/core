//! A thermal evaluation whose later temperatures are copies of its first
//! (JEFF-4.0's T+1 K records) is handed to NJOY cut to its first temperature,
//! so THERMR, which takes any coherent elastic LIST within `temp/1000 + 5` K
//! of the temperature asked for, cannot use the copy in its place.
//!
//! A stand-in `njoy` saves the deck and the thermal tape it was given and
//! fails, so this checks what NJOY would have read on a machine without it.

#![cfg(unix)]

use std::io::Write;
use std::path::{Path, PathBuf};

use yamc_convert::entry::Provenance;
use yamc_convert::thermal::{convert_thermal, ThermalSource};

/// JEFF-4.0's Zr90 in Zircaloy-4 at 350 K, cut to three betas, with a 351 K
/// copy after every TAB1.
const ZY4_350K: &[u8] = include_bytes!("../../endf/fixtures/tsl-Zr90inZy4-350K.jeff40.endf.xz");
/// Any neutron evaluation: the stand-in never reads it.
const NEUTRON: &[u8] = include_bytes!("../../endf/fixtures/n-003_Li_006_trimmed.endf.xz");

fn unpack(compressed: &[u8], to: &Path) -> PathBuf {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    std::fs::write(to, out).expect("fixture is written");
    to.to_path_buf()
}

/// An `njoy` that copies its deck and thermal tape (unit 21) into `dir` and
/// fails. Written by `sh`, as `njoy_scratch.rs` explains.
fn njoy(dir: &Path) -> PathBuf {
    let script = dir.join("njoy");
    let text = format!(
        "#!/bin/sh\ncat > '{}'\ncp tape21 '{}'\nexit 1\n",
        dir.join("deck").display(),
        dir.join("thermal").display()
    );
    let mut sh = std::process::Command::new("sh")
        .args(["-c", "cat > \"$1\" && chmod 755 \"$1\"", "sh"])
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .expect("sh starts");
    sh.stdin
        .take()
        .expect("a pipe to sh")
        .write_all(text.as_bytes())
        .expect("the stand-in is written");
    assert!(sh.wait().expect("sh finishes").success());
    script
}

#[test]
fn njoy_is_handed_the_first_temperature_only() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thermal = unpack(ZY4_350K, &tmp.path().join("tsl_Zr90_Zy4_350K.jeff"));
    let neutron = unpack(NEUTRON, &tmp.path().join("n.endf"));
    let njoy = njoy(tmp.path());
    let source = ThermalSource {
        name: "c_Zr90_in_Zy4",
        thermal: std::slice::from_ref(&thermal),
        neutron: &neutron,
        bound_nuclides: &["Zr90".to_string()],
        zaids: &[40090],
        ace_name: "zr90z",
        nmix: 1,
        njoy_exec: njoy.to_str().expect("a UTF-8 path"),
    };

    let err = convert_thermal(&source, &tmp.path().join("out"), &Provenance::default())
        .expect_err("the stand-in fails")
        .to_string();
    assert!(err.contains("NJOY failed on"), "{err}");

    let deck = std::fs::read_to_string(tmp.path().join("deck")).expect("njoy was run");
    assert!(deck.contains("350.0"), "{deck}");
    assert!(
        !deck.contains("351.0"),
        "NJOY was asked for the copy:\n{deck}"
    );

    let given = endf::Material::from_file(tmp.path().join("thermal")).expect("a valid tape");
    let law = endf::ThermalScattering::from_endf(&given).expect("the cut reads");
    assert_eq!(law.temperatures, vec![350.0]);
    let original = endf::Material::from_file(&thermal).expect("the fixture reads");
    assert_eq!(
        given.mf7_mt2().unwrap().coherent.as_ref().unwrap().s,
        original.mf7_mt2().unwrap().coherent.as_ref().unwrap().s,
        "the 350 K Bragg edges and factors are handed over as written"
    );
    // The evaluation file itself is untouched.
    assert_eq!(
        original
            .mf7_mt2()
            .unwrap()
            .coherent
            .as_ref()
            .unwrap()
            .others
            .len(),
        1
    );
}
