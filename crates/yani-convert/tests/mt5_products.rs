//! MT=5's products read from MF=6 into `(n,X)` rows.
//!
//! The fixtures are three evaluations trimmed to MF=1 MT=451, MF=3 MT=5 and
//! MF=6 MT=5, with each MF=6 product's energy-angle distribution replaced by
//! LAW=0 (none) to keep them small: the yields, which are all the reader
//! takes, are the tapes' own. ENDF/B-VIII.1 Fe54 lumps its (n,np) and most of
//! its (n,alpha) into MT=5 with every residual given; ENDF/B-VIII.1 Zr96 gives
//! MT=5's light particles and no residual; TENDL-2025 Ni58 gives its residuals'
//! isomers by LIP (Co58 and Co58_m1).

use std::collections::{BTreeMap, BTreeSet};

use endf::Material;
use yani_convert::anything::{self, DecayIndex, Evaluation, Row};

macro_rules! local_fixture {
    ($name:literal) => {
        include_bytes!(concat!("fixtures/", $name))
    };
}

fn material(compressed: &[u8]) -> Material {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    Material::from_str(&String::from_utf8(out).expect("UTF-8")).expect("fixture parses")
}

/// Every ground state from H to Zn as a nuclide with decay data, and the
/// isomers named, so each product is named as itself unless left out here.
fn decay(isomers: &[&str]) -> DecayIndex {
    let mut names: Vec<(String, bool, f64)> = Vec::new();
    for z in 1..=30u32 {
        for a in z..=(3 * z + 10) {
            names.push((endf::gnds_name(z, a, 0), false, 1.0));
        }
    }
    names.extend(isomers.iter().map(|n| (n.to_string(), false, 1.0)));
    DecayIndex::from_library(names)
}

fn named(
    evaluation: Evaluation,
    decay: &DecayIndex,
) -> (Vec<Row>, serde_json::Map<String, serde_json::Value>) {
    let parent = evaluation.parent.clone();
    let parents = BTreeSet::from([parent.clone()]);
    let out = anything::name_all(
        &BTreeMap::from([(parent.clone(), evaluation)]),
        decay,
        &parents,
    );
    (
        out.rows
            .get(&parent)
            .map(|r| r.rows.clone())
            .unwrap_or_default(),
        out.record,
    )
}

fn at(row: &Row, e: f64) -> f64 {
    let i = row.energy.partition_point(|&x| x <= e);
    let (x0, x1) = (row.energy[i - 1], row.energy[i]);
    let (y0, y1) = (row.multiplicity[i - 1], row.multiplicity[i]);
    y0 + (e - x0) / (x1 - x0) * (y1 - y0)
}

fn find<'a>(rows: &'a [Row], target: &str) -> &'a Row {
    rows.iter()
        .find(|r| r.target.as_deref() == Some(target))
        .unwrap_or_else(|| panic!("no row for {target}"))
}

/// Fe54's residuals are shares and its light particles multiplicities, each
/// the tape's own value: at 14 MeV Mn53 0.8449, Cr51 0.1546, H1 0.8385 and
/// He4 0.1550 per MT=5 reaction. Charge and mass number balance to 0.4%
/// wherever MT=5 is a few percent of its peak or more, so nothing is listed
/// as not conserving at those energies, and no residual is missing.
#[test]
fn fe54_residuals_are_shares_and_its_particles_multiplicities() {
    let evaluation = anything::read(&material(local_fixture!(
        "n-026_Fe_054_endfb81_mt5.endf.xz"
    )))
    .expect("Fe54 has MT=5");
    assert_eq!(evaluation.parent, "Fe54");
    let (rows, record) = named(evaluation, &decay(&[]));

    for (target, share, value) in [
        ("Mn53", true, 0.844948),
        ("Cr51", true, 0.154576),
        ("H1", false, 0.83852),
        ("He4", false, 0.155042),
    ] {
        let row = find(&rows, target);
        assert_eq!(row.share, Some(share), "{target}");
        assert_eq!(row.lip, Some(0), "{target}");
        assert!((at(row, 1.4e7) - value).abs() < 1e-5, "{target}");
    }
    assert!(
        rows.iter().all(|r| r.target.is_some()),
        "no residual is missing"
    );
    let residuals: f64 = rows
        .iter()
        .filter(|r| r.share == Some(true))
        .map(|r| at(r, 1.4e7))
        .sum();
    assert!((residuals - 1.0).abs() < 1e-4, "{residuals}");

    let balance = &record["conservation"]["Fe54"];
    for key in ["charge_weighted", "mass_number_weighted"] {
        assert!(balance[key].as_f64().unwrap() < 1.0e-2, "{balance}");
    }
    assert!(record["residual_not_given"].as_array().unwrap().is_empty());
}

/// Conservation is recorded for the parent whether or not it holds, with
/// where it is worst: Fe54's products carry no charge at all where its MT=5
/// opens at 5.5 MeV (1e-7 b with no yields yet), which is said, while the
/// MT=5-weighted imbalance the list is decided on stays under the tolerance.
#[test]
fn conservation_is_recorded_for_every_parent() {
    let evaluation = anything::read(&material(local_fixture!(
        "n-026_Fe_054_endfb81_mt5.endf.xz"
    )))
    .expect("Fe54 has MT=5");
    let (_, record) = named(evaluation, &decay(&[]));
    assert_eq!(record["parents"], 1);
    let balance = &record["conservation"]["Fe54"];
    assert_eq!(balance["charge"].as_f64().unwrap(), 1.0, "{balance}");
    assert_eq!(balance["charge_at_eV"].as_f64().unwrap(), 5.5e6);
    assert!(record["not_conserving"].as_array().unwrap().is_empty());
}

/// Zr96 lists MT=5's light particles and no residual, and they carry a few
/// percent of zirconium's charge, so the residual is not given: a row with no
/// target says so beside the particles, and the record names it.
#[test]
fn a_residual_not_given_is_written_as_such() {
    let evaluation = anything::read(&material(local_fixture!(
        "n-040_Zr_096_endfb81_mt5.endf.xz"
    )))
    .expect("Zr96 has MT=5");
    let (rows, record) = named(evaluation, &decay(&[]));
    let marker: Vec<&Row> = rows.iter().filter(|r| r.target.is_none()).collect();
    assert_eq!(marker.len(), 1);
    assert_eq!((marker[0].lip, marker[0].share), (None, None));
    assert!(rows
        .iter()
        .filter(|r| r.target.is_some())
        .all(|r| r.share == Some(false)));
    let entry = &record["residual_not_given"][0];
    assert_eq!(entry["nuclide"], "Zr96");
    assert!(
        entry["charge_share_carried"].as_f64().unwrap() < 0.1,
        "{entry}"
    );
}

/// TENDL-2025's Ni58 gives Co58 at LIP 0 and 1. With Co58_m1 in the decay
/// data each is its own row; without it the isomer goes where every other
/// reaction's product with no decay data goes, the ground state, and the row
/// is their sum.
#[test]
fn an_isomer_comes_from_lip() {
    let fixture = || {
        anything::read(&material(local_fixture!(
            "n-028_Ni_058_tendl2025_mt5.endf.xz"
        )))
        .expect("Ni58 has MT=5")
    };
    let (rows, _) = named(fixture(), &decay(&["Co58_m1"]));
    let (ground, isomer) = (find(&rows, "Co58"), find(&rows, "Co58_m1"));
    assert_eq!(ground.lip, Some(0));
    assert_eq!(isomer.lip, Some(1));
    let (g, m) = (at(ground, 4.0e7), at(isomer, 4.0e7));
    assert!(g > 0.0 && m > 0.0, "{g} {m}");

    let (merged, _) = named(fixture(), &decay(&[]));
    assert!(merged
        .iter()
        .all(|r| r.target.as_deref() != Some("Co58_m1")));
    let sum = at(find(&merged, "Co58"), 4.0e7);
    assert!((sum - (g + m)).abs() < 1e-12 * sum, "{sum} vs {}", g + m);
}

/// The rows go through the writer and come back through yani's reader with
/// their curves: the residuals as shares, the particles as multiplicities,
/// the residual not given as a row with no target, each `(n,X)` row's
/// branching NaN until a fold sets it.
#[test]
fn the_rows_reach_yanis_reader() {
    let parents = BTreeSet::from(["Fe54".to_string(), "Zr96".to_string()]);
    let evaluations = BTreeMap::from([
        (
            "Fe54".to_string(),
            anything::read(&material(local_fixture!(
                "n-026_Fe_054_endfb81_mt5.endf.xz"
            )))
            .unwrap(),
        ),
        (
            "Zr96".to_string(),
            anything::read(&material(local_fixture!(
                "n-040_Zr_096_endfb81_mt5.endf.xz"
            )))
            .unwrap(),
        ),
    ]);
    let named = anything::name_all(&evaluations, &decay(&[]), &parents);
    let chain = endf::Chain {
        nuclides: ["Fe54", "Zr96"]
            .iter()
            .map(|n| endf::chain::Nuclide::new(n))
            .collect(),
    };
    let dir = std::env::temp_dir().join(format!("yani-convert-mt5-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    yani_convert::write_reactions(&chain, &named.rows, &dir.join("reactions")).expect("written");
    // The reader needs a decay index; one nuclide is enough to make a chain.
    let decay_dir = dir.join("decay");
    std::fs::create_dir_all(&decay_dir).unwrap();
    yani::export_chain_parts(
        &std::collections::HashMap::from([(
            "Fe54".to_string(),
            yani::ChainNuclide {
                name: "Fe54".to_string(),
                half_life: None,
                half_life_uncertainty: None,
                decay_energy: 0.0,
                decay_energy_uncertainty: None,
                decay_energy_components: Default::default(),
                reactions: Vec::new(),
                decays: Vec::new(),
                fission_yields: None,
                sources: Vec::new(),
            },
        )]),
        dir.join("stub"),
        None,
    )
    .unwrap();
    std::fs::copy(
        dir.join("stub/decay/nuclides.arrow"),
        decay_dir.join("nuclides.arrow"),
    )
    .unwrap();
    let (chain, branch) =
        yani::parse_chain_parts(&decay_dir, Some(&dir.join("reactions")), None, None)
            .expect("yani reads the rows");
    let _ = std::fs::remove_dir_all(&dir);

    let fe54 = &branch.curves()["Fe54"][yani::reactions::ANYTHING];
    let quantity = |t: &str| fe54.iter().find(|c| c.target == t).unwrap().quantity;
    assert_eq!(quantity("Mn53"), yani::BranchQuantity::Yield);
    assert_eq!(quantity("He4"), yani::BranchQuantity::Multiplicity);
    assert!(chain["Fe54"]
        .reactions
        .iter()
        .all(|r| r.kind == yani::reactions::ANYTHING && r.branching.is_nan()));
    assert!(chain["Zr96"]
        .reactions
        .iter()
        .any(|r| r.target.is_none() && r.branching == 0.0));
}
