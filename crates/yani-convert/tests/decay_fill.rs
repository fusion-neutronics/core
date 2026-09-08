//! Placeholder mean decay energies: labelled in the provenance, replaced from a
//! second library on request, and never touched otherwise.
//!
//! ENDF/B-VIII.1's Sn111 is a Nuclear Wallet Cards conversion with Q/3 in each
//! of the light and electromagnetic slots, 1.63 MeV per decay against the
//! 0.69 MeV JENDL-5.0 evaluates from the decay scheme. Fixtures are pulled in
//! with `include_bytes!`, so a missing one is a compile error.

use endf::Material;

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("../../endf/fixtures/", $name))
    };
}

fn material(compressed: &[u8]) -> Material {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    Material::from_str(&String::from_utf8(out).expect("fixture is UTF-8")).expect("fixture parses")
}

fn provenance() -> yani_convert::Provenance {
    yani_convert::Provenance {
        library: "endf-b8.1".to_string(),
        decay_library: String::new(),
        data_version: "test".to_string(),
        created_utc: "2026-09-06T00:00:00+00:00".to_string(),
    }
}

fn convert(name: &str, fill: &[Material]) -> (std::path::PathBuf, serde_json::Value) {
    let decay = vec![
        material(fixture!("dec-050_Sn_111.endf.xz")),
        material(fixture!("dec-049_In_116m1.endf.xz")),
    ];
    let dir = std::env::temp_dir().join(format!("yani-convert-fill-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    yani_convert::convert_transmutation(
        &yani_convert::Inputs {
            decay: &decay,
            fpy: &[],
            q_values: &endf::chain::QValues::new(),
            decay_fill: fill,
            decay_fill_library: if fill.is_empty() { "" } else { "jendl-5.0" },
        },
        &[],
        None,
        &["decay"],
        &dir,
        &provenance(),
    )
    .expect("conversion runs");
    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("decay").join("provenance.json"))
            .expect("provenance written"),
    )
    .expect("provenance is JSON");
    (dir, record)
}

fn decay_energy(dir: &std::path::Path, nuclide: &str) -> f64 {
    let (chain, _) =
        yani::parse_chain_parts(&dir.join("decay"), None, None, None).expect("yani reads it back");
    chain[nuclide].decay_energy
}

#[test]
fn placeholders_are_listed_and_left_alone_without_a_fill() {
    let (dir, record) = convert("plain", &[]);
    let placeholders = &record["decay_energy_placeholders"];
    assert_eq!(placeholders["count"], 1);
    assert_eq!(placeholders["nuclides"], serde_json::json!(["Sn111"]));
    assert!(record.get("decay_energy_fill").is_none());
    // The number written is the library's own, placeholder or not.
    assert!((decay_energy(&dir, "Sn111") - 1_634_549.4).abs() < 1.0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_fill_replaces_the_placeholder_and_records_it() {
    let fill = vec![material(fixture!("dec-050-Sn-111.jendl5.endf.xz"))];
    let (dir, record) = convert("filled", &fill);
    assert_eq!(record["decay_energy_placeholders"]["count"], 0);
    let fill_record = &record["decay_energy_fill"];
    assert_eq!(fill_record["library"], "jendl-5.0");
    assert_eq!(fill_record["replaced"].as_array().map(Vec::len), Some(1));
    assert_eq!(fill_record["replaced"][0]["nuclide"], "Sn111");
    assert!((fill_record["replaced"][0]["after_eV"].as_f64().unwrap() - 693_315.8).abs() < 1.0);
    assert_eq!(fill_record["half_life_mismatch"], serde_json::json!([]));
    assert_eq!(fill_record["unfilled"], serde_json::json!([]));
    assert!((decay_energy(&dir, "Sn111") - 693_315.8).abs() < 1.0);
    // An evaluated record is not a candidate, whatever the fill holds.
    assert!((decay_energy(&dir, "In116_m1") - 2_804_000.0).abs() < 50_000.0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_fill_needs_its_library_named() {
    let err = yani_convert::convert_transmutation_files(
        &[],
        &[],
        &[],
        &["anything".to_string()],
        "",
        None,
        None,
        None,
        &std::env::temp_dir().join("yani-convert-fill-unnamed"),
        &provenance(),
    )
    .expect_err("refused");
    assert!(err.to_string().contains("decay_fill_library"), "{err}");
}
