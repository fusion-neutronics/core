//! The crate used the way a consumer uses it, through its root re-exports.
//!
//! `golden.rs` reaches into module paths freely, because it has to name every
//! type the dumpers touch. That makes it a poor guide to whether the crate is
//! pleasant (or even possible) to depend on from outside. This walks the
//! path a transport code takes, from a file to the distributions it samples,
//! and imports everything from `endf::` rather than `endf::mf::mf6::`.
//!
//! It is a shape test, not a value test: the goldens check the numbers. What
//! fails here is a type that stopped being public, a re-export that went
//! missing, or an accessor that a consumer cannot reach without knowing the
//! crate's internal layout.

use std::path::{Path, PathBuf};

use endf::{
    tables_from_str, AngleEnergy, Chain, Decay, FissionProductYields, IncidentNeutron,
    IncidentPhoton, Interpretation, Material, MetastableScheme, ProbabilityTables,
    RadionuclideProduction, Tabulated1D,
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn read_text(name: &str) -> String {
    let path = fixture(name);
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut raw.as_slice(), &mut out)
        .unwrap_or_else(|e| panic!("decompressing {}: {e}", path.display()));
    String::from_utf8(out).expect("a fixture is not UTF-8")
}

/// A file to a nuclide with cross sections on it, the first thing any
/// consumer does.
#[test]
fn from_a_file_to_a_cross_section() {
    let material = Material::from_str(&read_text("n-095_Am_244.endf.xz")).unwrap();
    assert_eq!(material.mat, 9552);
    assert!(material.contains(3, 1));

    // The sublibrary decides the class, without the caller reading NSUB.
    let neutron = match material.interpret().unwrap() {
        Interpretation::IncidentNeutron(n) => n,
        other => panic!("an incident-neutron evaluation gave {other:?}"),
    };
    assert_eq!(neutron.name(), "Am244");
    assert_eq!(neutron.atomic_number, 95);

    // A reaction, its cross section, and the cross section evaluated.
    let capture = &neutron.reactions[&102];
    let xs: &Tabulated1D = capture.xs.values().next().expect("a capture cross section");
    assert!(xs.eval(0.0253) > 0.0);
    // No unionised energy grid from ENDF: each cross section carries its own.
    // A consumer that wants one gets it from ACE, or builds it itself; see
    // the ACE test below.
    assert!(neutron.energy.is_empty());

    // The same nuclide built directly, which is the other entry point.
    let direct = IncidentNeutron::from_endf(&material).unwrap();
    assert_eq!(direct.name(), neutron.name());
}

/// Every product distribution shape reachable without naming a private type.
///
/// This is the part an Arrow projection consumes, so a consumer has to be able
/// to match on `AngleEnergy` from the crate root.
#[test]
fn product_distributions_are_matchable() {
    let material = Material::from_str(&read_text("n-003_Li_006_trimmed.endf.xz")).unwrap();
    let neutron = IncidentNeutron::from_endf(&material).unwrap();

    let mut seen = 0;
    for reaction in neutron.reactions.values() {
        for product in &reaction.products {
            for distribution in &product.distribution {
                // The four shapes the Arrow schema has columns for. A new
                // variant makes this stop compiling, which is the point.
                match distribution {
                    AngleEnergy::Uncorrelated(_)
                    | AngleEnergy::Correlated(_)
                    | AngleEnergy::KalbachMann(_)
                    | AngleEnergy::NBodyPhaseSpace(_) => seen += 1,
                }
            }
        }
    }
    assert!(seen > 0, "no product carried a distribution");
}

/// The ACE side, which is what a transport code actually reads.
#[test]
fn from_an_ace_table_to_a_nuclide() {
    let tables = tables_from_str(&read_text("Li6.ace.xz"), None).unwrap();
    let table = &tables[0];
    assert!(table.name.starts_with("3006"));
    assert!(!table.xss.is_empty());

    let neutron = IncidentNeutron::from_ace(table, MetastableScheme::default()).unwrap();
    assert_eq!(neutron.name(), "Li6");
    assert!(!neutron.reactions.is_empty());
    assert_eq!(neutron.temperatures().len(), 1);
    // The unionised grid an ENDF evaluation does not have, keyed by the same
    // temperature string the cross sections are.
    let temperature = &neutron.temperatures()[0];
    assert!(!neutron.energy[temperature].is_empty());

    // Probability tables come back keyed by temperature, empty when the table
    // has no URR block.
    let urr: &std::collections::BTreeMap<String, ProbabilityTables> = &neutron.urr;
    assert!(urr.is_empty(), "Li6 has no unresolved resonance block");
}

/// Decay data, fission yields and radionuclide production, the depletion
/// inputs, each reachable from the crate root.
#[test]
fn the_depletion_inputs_are_reachable() {
    let decay_material = Material::from_str(&read_text("dec-049_In_116m1.endf.xz")).unwrap();
    let decay = Decay::from_material(&decay_material).unwrap();
    assert_eq!(decay.nuclide.name, "In116_m1");
    assert!(decay.half_life.is_some());
    assert!(!decay.modes.is_empty());

    let yields_material = Material::from_str(&read_text("synthetic-nfy.endf.xz")).unwrap();
    let yields = FissionProductYields::from_material(&yields_material).unwrap();
    assert_eq!(yields.nuclide.name, "U235");
    assert_eq!(yields.energies.len(), 2);

    let production_material =
        Material::from_str(&read_text("n-049_In-115_trimmed.endf.xz")).unwrap();
    let production = endf::radionuclide_production(&production_material);
    assert!(!production.is_empty());
    let states: &Vec<RadionuclideProduction> = production.values().next().unwrap();
    assert!(states[0].excitation_energy() >= 0.0);
}

/// A depletion chain, built from evaluations the caller supplies.
#[test]
fn a_chain_can_be_built_from_materials() {
    let decay: Vec<Material> = ["dec-049_In_116m1.endf.xz", "dec-049_In_116m2.endf.xz"]
        .iter()
        .map(|name| Material::from_str(&read_text(name)).unwrap())
        .collect();
    let neutron = vec![Material::from_str(&read_text("n-049_In-115_trimmed.endf.xz")).unwrap()];

    let chain = Chain::from_endf(&decay, &[], &neutron, &["(n,gamma)"]).unwrap();
    assert!(!chain.nuclides.is_empty());
}

/// Photon data, the other sublibrary with a high-level class.
#[test]
fn photoatomic_data_is_reachable() {
    let material = Material::from_str(&read_text("photoat-001_H_000.endf.xz")).unwrap();
    let photon = IncidentPhoton::from_endf(&material, None).unwrap();
    assert_eq!(photon.atomic_number, 1);
    assert!(!photon.reactions.is_empty());
}
