//! Each branching row carries what the evaluation states about the production
//! states behind it, and none of it changes the row.

use std::sync::Arc;

use endf::radionuclide_production::{LevelRoute, ISOMER_ENERGY_TOLERANCE};
use endf::Material;
use yani_convert::branching::{extract_branching, BranchingRow, DEFAULT_LINEARIZE_TOL};

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("../../endf/fixtures/", $name))
    };
}

/// Fixtures kept beside these tests rather than in the `endf` crate's set, so
/// they need no golden dump. The Nb93 file is TENDL-2025's with MF=1, the
/// MF=3 sections its production lists belong to (and MT=1, MT=3), and MF=8,
/// 9 and 10 kept; MT=5 and every other file are dropped.
macro_rules! local_fixture {
    ($name:literal) => {
        include_bytes!(concat!("fixtures/", $name))
    };
}

fn material(compressed: &[u8]) -> Material {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    Material::from_str(&String::from_utf8(out).expect("fixture is UTF-8")).expect("fixture parses")
}

fn nb93() -> Material {
    material(local_fixture!("n-041_Nb_093_tendl-2025_trimmed.endf.xz"))
}

fn nb93_rows() -> (Vec<BranchingRow>, yani_convert::branching::BranchingStats) {
    let decay = vec![
        material(local_fixture!("dec-041_Nb_092m1.endf.xz")),
        material(local_fixture!("dec-041_Nb_093m1.endf.xz")),
        material(local_fixture!("dec-041_Nb_094m1.endf.xz")),
    ];
    extract_branching(
        &[nb93()],
        &decay,
        ISOMER_ENERGY_TOLERANCE,
        DEFAULT_LINEARIZE_TOL,
    )
    .expect("branching extracts")
}

fn find<'a>(rows: &'a [BranchingRow], reaction: &str, target: &str) -> &'a BranchingRow {
    rows.iter()
        .find(|r| r.reaction == reaction && r.target == target)
        .unwrap_or_else(|| panic!("no {reaction} -> {target} row"))
}

/// TENDL-2025 normalised Nb93's (n,2n) isomer partial to IRDFF-II and says so
/// in MF=1; every row keeps that block as written.
const NB93_NORMALISATION: &str = "projectile n\n\
element Nb\n\
mass 093\n\
library irdff2.0\n\
norm mt=4 isom=1 width=0.05 emin=0. emax=20. ebeg=0. eend=16.\n\
norm mt=16 isom=1 width=0.05 emin=0. emax=20. ebeg=0. eend=16.\n\
norm mt=102 width=0.05 emin=0. emax=20. ebeg=0. eend=16.";

#[test]
fn each_row_records_the_states_it_was_summed_from() {
    let (rows, stats) = nb93_rows();
    assert!(rows
        .iter()
        .all(|r| r.normalisation.as_deref() == Some(NB93_NORMALISATION)));

    // (n,2n): TENDL lists both states of Nb92 in MF=10, so the list is
    // complete, and the isomer's 135.5 keV level is the decay data's.
    let m1 = find(&rows, "(n,2n)", "Nb92_m1");
    assert_eq!(m1.states.len(), 1);
    let state = &m1.states[0];
    assert_eq!((state.mt, state.lfs, state.lmf), (16, 1, Some(10)));
    assert!(state.list_complete);
    assert_eq!(state.level_route, LevelRoute::Energy);
    assert_eq!(state.level_energy, 135_500.0);
    assert_eq!(state.level_energy_difference, Some(0.0));
    let mf3 = state.mf3.as_ref().expect("Nb93 has MF=3 MT=16");
    assert_eq!(**mf3, nb93().mf3(16).expect("MT=16").sigma);

    let ground = &find(&rows, "(n,2n)", "Nb92").states[0];
    assert_eq!((ground.lfs, ground.level_route), (0, LevelRoute::Ground));
    assert_eq!(ground.level_energy_difference, Some(0.0));
    assert!(
        Arc::ptr_eq(mf3, ground.mf3.as_ref().expect("MF=3")),
        "the states of one MT share its MF=3"
    );

    // Nb93_m1 sits 1 eV below the level TENDL gives it; the difference is kept
    // to the eV, signed as level less isomer.
    let nn = &find(&rows, "(n,n')", "Nb93_m1").states[0];
    assert_eq!(nn.level_energy_difference, Some(30_769.0 - 30_770.0));

    // Zr90 has no isomer, so (n,2nd)'s ground and its 2.319 MeV level both
    // land on it, and the row keeps both, in the order they were summed.
    let zr = find(&rows, "(n,2nd)", "Zr90");
    let levels: Vec<(i32, i64, LevelRoute)> = zr
        .states
        .iter()
        .map(|s| (s.mt, s.lfs, s.level_route))
        .collect();
    assert_eq!(
        levels,
        [
            (11, 0, LevelRoute::NoIsomers),
            (11, 3, LevelRoute::NoIsomers)
        ]
    );
    assert_eq!(zr.states[1].level_energy_difference, Some(2_319_001.0));

    // Capture is an MF=9 yield list.
    let capture = &find(&rows, "(n,gamma)", "Nb94_m1").states[0];
    assert_eq!((capture.mt, capture.lmf), (102, Some(9)));

    let line = stats
        .list_facts
        .iter()
        .find(|l| l.starts_with("Nb93 MT16 "))
        .expect("a line for (n,2n)");
    assert_eq!(
        line,
        "Nb93 MT16 (n,2n) MF=10: ground listed, MF=3 given; \
         LFS 0 -> Nb92 (LMF 10, ground, +0.000 keV); \
         LFS 1 -> Nb92_m1 (LMF 10, energy, +0.000 keV); \
         normalised: library irdff2.0 | \
         norm mt=16 isom=1 width=0.05 emin=0. emax=20. ebeg=0. eend=16."
    );
    assert_eq!(stats.list_facts.len(), 15);
    assert_eq!(stats.list_counts.get("MF=10 complete"), Some(&14));
    assert_eq!(stats.list_counts.get("MF=9 complete"), Some(&1));
    // MT=4, 16 and 102 are named in a norm line; the rest are not.
    assert_eq!(stats.list_counts.get("normalised"), Some(&3));
}

/// ENDF/B-VIII.1's In115 lists only the isomer, for all three reactions, and
/// has no normalisation block.
#[test]
fn an_isomer_only_list_is_marked_incomplete() {
    let neutron = vec![material(fixture!("n-049_In-115_trimmed.endf.xz"))];
    let decay = vec![
        material(fixture!("dec-049_In_116m1.endf.xz")),
        material(fixture!("dec-049_In_116m2.endf.xz")),
    ];
    let (rows, stats) = extract_branching(
        &neutron,
        &decay,
        ISOMER_ENERGY_TOLERANCE,
        DEFAULT_LINEARIZE_TOL,
    )
    .expect("branching extracts");
    assert!(!rows.is_empty());
    for row in &rows {
        assert!(row.normalisation.is_none());
        assert!(
            row.states.iter().all(|s| !s.list_complete),
            "{} {}: {:?}",
            row.reaction,
            row.target,
            row.states
        );
    }
    assert_eq!(stats.list_counts.get("MF=10 isomers only"), Some(&2));
    assert_eq!(stats.list_counts.get("MF=9 isomers only"), Some(&1));
    assert_eq!(stats.list_counts.get("MF=10 complete"), None);
}
