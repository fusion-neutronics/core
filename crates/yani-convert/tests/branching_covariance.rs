//! MF=40, the covariance of the isomeric branching partials, written beside
//! `branching.arrow` and read back by the reader an uncertainty run uses.
//!
//! TENDL-2017 Nb93 carries MF=40 on (n,n') and (n,2n), ground and isomer each,
//! which is the common TENDL shape. The In115 fixture carries none. The two
//! synthetic cases are edits of the Nb93 tape, so the evaluation around them
//! stays a real one.

use std::path::PathBuf;

use endf::Material;
use yamc_nuclide::arrow::covariance_arrow::read_branching_covariance;
use yamc_nuclide::covariance::CovarianceData;
use yani_convert::branching::{
    write_branching, write_branching_covariance, BranchingExtractor, Extracted,
    DEFAULT_LINEARIZE_TOL,
};

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("../../endf/fixtures/", $name))
    };
}

const NB93: &[u8] = fixture!("n-041_Nb_093_tendl2017_trimmed.endf.xz");
const NB_DECAY: &[&[u8]] = &[
    fixture!("dec-041_Nb_092.endf.xz"),
    fixture!("dec-041_Nb_092m1.endf.xz"),
    fixture!("dec-041_Nb_093m1.endf.xz"),
];

fn text(compressed: &[u8]) -> String {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    String::from_utf8(out).expect("fixture is UTF-8")
}

fn material(text: &str) -> Material {
    Material::from_str(text).expect("tape parses")
}

fn extract(neutron: &[Material], decay: &[Material]) -> Extracted {
    let mut extractor = BranchingExtractor::new(
        decay,
        endf::radionuclide_production::ISOMER_ENERGY_TOLERANCE,
        DEFAULT_LINEARIZE_TOL,
    );
    for m in neutron {
        extractor.add(m);
    }
    extractor.finish()
}

fn nb_decay() -> Vec<Material> {
    NB_DECAY.iter().map(|b| material(&text(b))).collect()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("yani-convert-mf40-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

#[test]
fn mf40_is_written_beside_branching_for_nb93() {
    let neutron = vec![material(&text(NB93))];
    let out = extract(&neutron, &nb_decay());
    let dir = scratch("nb93");
    write_branching(&out.rows, &dir).expect("branching writes");
    assert!(write_branching_covariance(&out.covariance, &dir).expect("covariance writes"));

    let blocks =
        read_branching_covariance(&dir.join("branching_covariance.arrow")).expect("reads back");
    for (reaction, target) in [
        ("(n,2n)", "Nb92"),
        ("(n,2n)", "Nb92_m1"),
        ("(n,n')", "Nb93_m1"),
    ] {
        let of: Vec<_> = blocks
            .iter()
            .filter(|b| b.nuclide == "Nb93" && b.reaction == reaction && b.target == target)
            .collect();
        assert!(!of.is_empty(), "no block for Nb93 {reaction} {target}");
        for b in of {
            let CovarianceData::Ni(ni) = &b.block.data else {
                panic!("Nb93 {reaction} {target} has an NC block");
            };
            assert_eq!((ni.lb, ni.ls), (5, 1), "Nb93 {reaction} {target}");
            assert_eq!(b.block.xmf1, 10.0, "the partner is an MF=10 partial");
            assert_eq!(b.block.mat1, 0);
            // TENDL states no cross-state term: every block is the state's
            // covariance with itself.
            assert_eq!(b.target1.as_deref(), Some(target));
            assert_eq!(b.lfs1, b.lfs);
            // One state per target, so the branching.arrow curve is its own.
            assert!(b.energy.is_none() && b.values.is_none());
        }
    }
    let stats = &out.stats;
    assert_eq!(stats.mf40_sections, 2, "{stats:?}");
    assert!(stats.mf40_blocks > 0);
    assert_eq!(stats.mf40_blocks_by_lb.get(&5), Some(&stats.mf40_blocks));
    assert_eq!(stats.mf40_nc_blocks, 0);
    assert!(
        stats.mf40_unmatched_states.is_empty(),
        "{:?}",
        stats.mf40_unmatched_states
    );
    assert_eq!(stats.mf40_on_yield_channels, 0);
    assert_eq!(stats.mf40_own_mat_normalised, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every block reads back as the parser produced it: the file is a faithful
/// dump, so a fold that trusts it is folding the evaluation.
#[test]
fn the_blocks_round_trip_exactly() {
    let nb93 = material(&text(NB93));
    let out = extract(std::slice::from_ref(&nb93), &nb_decay());
    let dir = scratch("round-trip");
    write_branching_covariance(&out.covariance, &dir).expect("covariance writes");
    let blocks =
        read_branching_covariance(&dir.join("branching_covariance.arrow")).expect("reads back");

    let mut expected = Vec::new();
    for mt in [4, 16] {
        let mf40 = nb93.mf40(mt).expect("MF=40 on the tape");
        for state in &mf40.subsections {
            for (j, sub) in state.subsubsections.iter().enumerate() {
                for (k, ni) in sub.ni_subsections.iter().enumerate() {
                    expected.push((mt, state.lfs as i32, j as i32, k as i32, sub.xlfs1, ni));
                }
            }
        }
    }
    assert_eq!(blocks.len(), expected.len());
    for (mt, lfs, j, k, xlfs1, ni) in expected {
        let b = blocks
            .iter()
            .find(|b| {
                (b.block.mt, b.lfs, b.block.subsection_idx, b.block.block_idx) == (mt, lfs, j, k)
            })
            .unwrap_or_else(|| panic!("MT{mt} LFS {lfs} block {j}/{k} did not come back"));
        assert_eq!(b.block.data, CovarianceData::Ni(ni.clone()));
        assert_eq!(b.block.xlfs1, xlfs1);
        assert_eq!(b.lfs1 as f64, xlfs1);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_file_without_mf40() {
    let neutron = vec![material(&text(fixture!("n-049_In-115_trimmed.endf.xz")))];
    let decay = vec![material(&text(fixture!("dec-049_In_116m1.endf.xz")))];
    let out = extract(&neutron, &decay);
    assert!(!out.rows.is_empty());
    assert!(out.covariance.is_empty());
    assert_eq!(out.stats.mf40_sections, 0);

    let dir = scratch("in115");
    write_branching(&out.rows, &dir).expect("branching writes");
    assert!(!write_branching_covariance(&out.covariance, &dir).expect("nothing to write"));
    assert!(dir.join("branching.arrow").is_file());
    assert!(!dir.join("branching_covariance.arrow").exists());
    assert!(!dir.join("branching_covariance.arrow.absent").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A rebuild into the directory an MF=40 library was written to, from one
/// without, must not leave the old covariance beside the new curves.
#[test]
fn a_rebuild_without_mf40_removes_a_stale_file() {
    let dir = scratch("stale");
    let nb93 = extract(&[material(&text(NB93))], &nb_decay());
    assert!(write_branching_covariance(&nb93.covariance, &dir).expect("writes"));
    assert!(dir.join("branching_covariance.arrow").is_file());

    let in115 = extract(
        &[material(&text(fixture!("n-049_In-115_trimmed.endf.xz")))],
        &[],
    );
    write_branching(&in115.rows, &dir).expect("branching writes");
    assert!(!write_branching_covariance(&in115.covariance, &dir).expect("removes"));
    assert!(!dir.join("branching_covariance.arrow").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two MF=10 states resolving to one target are summed in `branching.arrow`,
/// so each state's block carries that state's own partial; a target one state
/// makes carries none.
///
/// With no decay file for Nb92's isomer both (n,2n) levels land on Nb92, the
/// isomer being unknown to the table, while Nb93_m1 stays a single state.
#[test]
fn merged_targets_carry_their_own_curve() {
    let nb93 = material(&text(NB93));
    let decay = vec![material(&text(fixture!("dec-041_Nb_093m1.endf.xz")))];
    let out = extract(std::slice::from_ref(&nb93), &decay);

    let n2n: Vec<_> = out
        .covariance
        .iter()
        .filter(|r| r.reaction == "(n,2n)")
        .collect();
    assert_eq!(n2n.len(), 2, "one sub-subsection per state");
    let production = endf::radionuclide_production::radionuclide_production(&nb93);
    for row in n2n {
        assert_eq!(row.target, "Nb92");
        let state = production[&16]
            .iter()
            .find(|s| s.lfs == row.lfs as i64)
            .expect("the MF=10 state");
        let (energy, values) = yani_convert::branching::linearize(
            state.cross_section.as_ref().expect("an MF=10 partial"),
            DEFAULT_LINEARIZE_TOL,
        );
        assert_eq!(row.energy.as_ref(), Some(&energy), "LFS {}", row.lfs);
        assert_eq!(row.values.as_ref(), Some(&values), "LFS {}", row.lfs);
    }
    let isomer = out
        .covariance
        .iter()
        .find(|r| r.reaction == "(n,n')" && r.target == "Nb93_m1")
        .expect("the (n,n') isomer");
    assert!(isomer.energy.is_none() && isomer.values.is_none());

    // And the reader keeps the null a null.
    let dir = scratch("merged");
    write_branching_covariance(&out.covariance, &dir).expect("writes");
    let blocks = read_branching_covariance(&dir.join("branching_covariance.arrow")).expect("reads");
    assert!(blocks
        .iter()
        .filter(|b| b.reaction == "(n,2n)")
        .all(|b| b.energy.is_some() && b.values.is_some()));
    assert!(blocks
        .iter()
        .filter(|b| b.target == "Nb93_m1")
        .all(|b| b.energy.is_none() && b.values.is_none()));
    let _ = std::fs::remove_dir_all(&dir);
}

/// MF=10 and MF=40 need not number a level alike. ENDF/B-VIII.1 Pb204 MT4 has
/// its isomer at LFS=21 in MF=10 and LFS=1 in MF=40, with one QI; here the
/// Nb92_m1 state of (n,2n) is renumbered 7 in MF=40 alone, its QI kept.
#[test]
fn an_mf40_state_matches_mf10_by_excitation_when_lfs_differs() {
    let tape = text(NB93);
    let renumber = |tape: String, from: &str, to: &str| {
        assert_eq!(tape.matches(from).count(), 1, "{from}");
        tape.replace(from, to)
    };
    // The state's CONT (LFS, in the fourth field) and its self block's XLFS1.
    let tape = renumber(
        tape,
        "-8.830870+6-8.966370+6      41092          1          0          1412540 16   10",
        "-8.830870+6-8.966370+6      41092          7          0          1412540 16   10",
    );
    let tape = renumber(
        tape,
        " 1.000000+1 1.000000+0          0         16          0          1412540 16   11",
        " 1.000000+1 7.000000+0          0         16          0          1412540 16   11",
    );
    let out = extract(&[material(&tape)], &nb_decay());
    assert!(
        out.stats.mf40_unmatched_states.is_empty(),
        "{:?}",
        out.stats.mf40_unmatched_states
    );
    let row = out
        .covariance
        .iter()
        .find(|r| r.reaction == "(n,2n)" && r.lfs == 7)
        .expect("the renumbered state is written");
    assert_eq!(row.target, "Nb92_m1", "matched by its 135.5 keV excitation");
    assert_eq!(row.target1.as_deref(), Some("Nb92_m1"));
    assert!(row.energy.is_none(), "still one state per target");
}
