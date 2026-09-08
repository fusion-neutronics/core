//! The decay subsection's provenance lists what its records say that cannot
//! be right, one list per kind, so a heat carrier can be looked up before its
//! number is believed. The records themselves are written as the library has
//! them.

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

/// Xe136 is flagged unstable with a half-life of zero, Hf177_m1's isomeric
/// transition pays out 1.52 MeV from 1.32 MeV, and In116_m1 is an evaluated
/// beta- record with nothing wrong with it.
#[test]
fn inconsistent_records_are_listed_by_kind() {
    let decay = vec![
        material(fixture!("dec-054_Xe_136.endf.xz")),
        material(fixture!("dec-072_Hf_177m1.endf.xz")),
        material(fixture!("dec-049_In_116m1.endf.xz")),
    ];
    let dir = std::env::temp_dir().join(format!(
        "yani-convert-inconsistencies-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    yani_convert::convert_transmutation(
        &yani_convert::Inputs {
            decay: &decay,
            fpy: &[],
            q_values: &endf::chain::QValues::new(),
            decay_fill: &[],
            decay_fill_library: "",
        },
        &[],
        None,
        &["decay"],
        &dir,
        &yani_convert::Provenance {
            library: "endf-b8.1".to_string(),
            decay_library: String::new(),
            data_version: "test".to_string(),
            created_utc: "2026-09-07T00:00:00+00:00".to_string(),
        },
    )
    .expect("conversion runs");
    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("decay").join("provenance.json"))
            .expect("provenance written"),
    )
    .expect("provenance is JSON");
    let _ = std::fs::remove_dir_all(&dir);

    let found = &record["decay_inconsistencies"];
    assert_eq!(found["count"], 2);
    assert_eq!(
        found["zero_half_life"],
        serde_json::json!([{"nuclide": "Xe136"}])
    );
    assert_eq!(found["branching_ratio_sum"], serde_json::json!([]));
    let transitions = found["isomeric_transition_energy"]
        .as_array()
        .expect("a list");
    assert_eq!(transitions.len(), 1, "{transitions:?}");
    assert_eq!(transitions[0]["nuclide"], "Hf177_m1");
    assert!((transitions[0]["q_eV"].as_f64().unwrap() - 1_315_450.0).abs() < 1.0);
    assert!((transitions[0]["recoverable_eV"].as_f64().unwrap() - 1_518_972.4).abs() < 1.0);
}
