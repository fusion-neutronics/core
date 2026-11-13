//! Round-trip the arrow chain writer: parse a chain, export it, re-parse, and
//! verify the re-parsed chain matches the original.

use std::collections::HashMap;
use std::path::PathBuf;

use yani::{export_chain_arrow, parse_chain_arrow, ChainNuclide, DecaySourceDistribution};

fn arrow_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/transmutation-endf-b8.1-sfr.arrow")
}

fn sort_chain_fields(chain: &mut HashMap<String, ChainNuclide>) {
    for nuc in chain.values_mut() {
        nuc.reactions.sort_by(|a, b| {
            a.kind
                .cmp(&b.kind)
                .then_with(|| a.target.cmp(&b.target))
                .then_with(|| a.branching.partial_cmp(&b.branching).unwrap())
        });
        nuc.decays.sort_by(|a, b| {
            a.kind
                .cmp(&b.kind)
                .then_with(|| a.target.cmp(&b.target))
                .then_with(|| a.branching.partial_cmp(&b.branching).unwrap())
        });
    }
}

fn approx_eq(a: f64, b: f64, rel: f64) -> bool {
    if a == b {
        return true;
    }
    let denom = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() / denom < rel
}

#[test]
fn arrow_writer_roundtrip() {
    let src = arrow_path();
    if !src.exists() {
        eprintln!("skipping: missing fixture {}", src.display());
        return;
    }

    let mut original = parse_chain_arrow(&src).expect("parse source arrow");
    // The exact source comparison below proves that each continuum's law, or
    // its absence, survives the round trip. Only the count is pinned here, so
    // that a reader that drops continua cannot pass by comparing empty sets.
    let continua = original
        .values()
        .flat_map(|n| n.sources.iter())
        .filter(|s| matches!(s.distribution, DecaySourceDistribution::Tabular { .. }))
        .count();
    assert_eq!(
        continua,
        276 + 276 + 287,
        "photon, electron and neutron continua"
    );

    let tmp =
        std::env::temp_dir().join(format!("yani_chain_roundtrip_{}.arrow", std::process::id()));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp).unwrap();
    }
    let converter = "yani-python 0.22.1 (core 0123abc)";
    export_chain_arrow(&original, &tmp, Some("endf-b8.1"), converter).expect("export");

    // version.json records the real library (not "unknown") and the build
    // that wrote it.
    let version_json =
        std::fs::read_to_string(tmp.join("version.json")).expect("read version.json");
    let version: serde_json::Value = serde_json::from_str(&version_json).expect("JSON");
    assert_eq!(version["library"], "endf-b8.1", "{version_json}");
    assert_eq!(version["converter"], converter, "{version_json}");

    let mut reparsed = parse_chain_arrow(&tmp).expect("reparse");
    std::fs::remove_dir_all(&tmp).ok();

    assert_eq!(original.len(), reparsed.len(), "nuclide count");

    sort_chain_fields(&mut original);
    sort_chain_fields(&mut reparsed);

    let mut failures = Vec::new();
    for (name, a) in &original {
        let b = reparsed.get(name).expect("nuclide in reparsed");

        match (a.half_life, b.half_life) {
            (Some(x), Some(y)) if approx_eq(x, y, 1e-12) => {}
            (None, None) => {}
            (x, y) => failures.push(format!("{name}: half_life {x:?} != {y:?}")),
        }

        assert_eq!(a.decays.len(), b.decays.len(), "{name}: decay count");
        for (x, y) in a.decays.iter().zip(&b.decays) {
            if x.kind != y.kind
                || x.target != y.target
                || !approx_eq(x.branching, y.branching, 1e-12)
            {
                failures.push(format!("{name}: decay mismatch"));
                break;
            }
        }

        assert_eq!(
            a.reactions.len(),
            b.reactions.len(),
            "{name}: reaction count"
        );
        for (x, y) in a.reactions.iter().zip(&b.reactions) {
            if x.kind != y.kind
                || x.target != y.target
                || !approx_eq(x.branching, y.branching, 1e-12)
            {
                failures.push(format!("{name}: reaction mismatch"));
                break;
            }
        }

        // Exactly, the kind and the law included: a continuum that came back
        // as lines would be read in the wrong units.
        assert_eq!(a.sources, b.sources, "{name}: sources");

        match (&a.fission_yields, &b.fission_yields) {
            (Some(x), Some(y)) => {
                assert_eq!(x.yields.len(), y.yields.len(), "{name}: fy count");
                for (fx, fy) in x.yields.iter().zip(&y.yields) {
                    assert!(approx_eq(fx.energy, fy.energy, 1e-12));
                    assert_eq!(fx.products.len(), fy.products.len());
                }
            }
            (None, None) => {}
            _ => failures.push(format!("{name}: fission_yields presence mismatch")),
        }
    }

    assert!(
        failures.is_empty(),
        "{} mismatches (first 10):\n{}",
        failures.len(),
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
