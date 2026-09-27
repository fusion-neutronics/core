//! MF=40, the covariance of the isomeric branching partials, written beside
//! `branching.arrow` as the tape gives it and read back by the reader a
//! consumer uses.
//!
//! TENDL-2017 Nb93 carries MF=40 on (n,n') and (n,2n), ground and isomer each,
//! which is the common TENDL shape. The In115 fixture carries none. The
//! synthetic cases are edits of the Nb93 tape, so the evaluation around them
//! stays a real one.

use std::path::{Path, PathBuf};

use endf::Material;
use yamc_nuclide::arrow::covariance_arrow::branching_covariance_blocks;
use yamc_nuclide::covariance::{BranchingCovarianceBlock, CovarianceData};
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

/// Nb93's MAT in TENDL-2017.
const NB93_MAT: i32 = 4125;

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

/// Replace one whole tape line, which must occur exactly once.
fn edit(tape: String, from: &str, to: &str) -> String {
    assert_eq!(tape.matches(from).count(), 1, "{from}");
    tape.replace(from, to)
}

/// The blocks as a consumer gets them: the chain loader reads the file into
/// the branch table, and the covariance reader turns its rows into blocks.
fn read_back(dir: &Path) -> Vec<BranchingCovarianceBlock> {
    let chain_dir = dir.join("chain");
    let nuclide = yani::ChainNuclide {
        name: "Nb93".to_string(),
        half_life: None,
        half_life_uncertainty: None,
        decay_energy: 0.0,
        decay_energy_uncertainty: None,
        decay_energy_components: Default::default(),
        reactions: Vec::new(),
        decays: Vec::new(),
        fission_yields: None,
        sources: Vec::new(),
    };
    let chain = std::collections::HashMap::from([("Nb93".to_string(), nuclide)]);
    yani::export_chain_parts(&chain, &chain_dir, Some("test")).expect("chain exports");
    let (_, branch) = yani::parse_chain_parts(&chain_dir.join("decay"), None, None, Some(dir))
        .expect("chain loads");
    branching_covariance_blocks(branch.covariance().expect("the covariance is loaded"))
        .expect("blocks read")
}

#[test]
fn mf40_is_written_beside_branching_for_nb93() {
    let neutron = vec![material(&text(NB93))];
    let out = extract(&neutron, &nb_decay());
    let dir = scratch("nb93");
    write_branching(&out.rows, &dir).expect("branching writes");
    assert!(write_branching_covariance(&out.covariance, &dir).expect("covariance writes"));

    let blocks = read_back(&dir);
    for (mt, reaction, target) in [
        (4, "(n,n')", "Nb93"),
        (4, "(n,n')", "Nb93_m1"),
        (16, "(n,2n)", "Nb92"),
        (16, "(n,2n)", "Nb92_m1"),
    ] {
        let of: Vec<_> = blocks
            .iter()
            .filter(|b| {
                b.reaction.as_deref() == Some(reaction) && b.target.as_deref() == Some(target)
            })
            .collect();
        assert!(!of.is_empty(), "no block for Nb93 {reaction} {target}");
        // Every target is a branching.arrow row of the same parent and kind.
        assert!(out
            .rows
            .iter()
            .any(|r| r.nuclide == "Nb93" && r.reaction == reaction && r.target == target));
        for b in of {
            assert_eq!(b.nuclide, "Nb93");
            assert_eq!(b.block.mt, mt);
            assert_eq!((b.mat, b.za, b.lis), (NB93_MAT, 41093, 0));
            let CovarianceData::Ni(ni) = &b.block.data else {
                panic!("Nb93 {reaction} {target} has an NC block");
            };
            assert_eq!((ni.lb, ni.ls), (5, 1), "Nb93 {reaction} {target}");
            assert_eq!(b.block.xmf1, 10.0, "the partner is an MF=10 partial");
            assert_eq!(b.block.mat1, 0);
            assert_eq!(b.block.mt1, mt);
            // MF=40 has no MTL; the column is null, which reads as 0.
            assert_eq!(b.block.mtl, 0);
            // TENDL states no cross-state term: every block is the state's
            // covariance with itself.
            assert_eq!(b.target1.as_deref(), Some(target));
            assert_eq!(b.block.xlfs1, b.lfs as f64);
            assert!(b.is_self_block() && !b.is_cross_material());
            // One state per target, so the branching.arrow curve is its own.
            assert!(b.energy.is_none() && b.values.is_none());
        }
    }
    let stats = &out.stats;
    assert_eq!(stats.mf40_sections, 2, "{stats:?}");
    assert_eq!(stats.mf40_blocks, 4);
    assert_eq!(stats.mf40_blocks_by_lb.get(&5), Some(&4));
    assert_eq!(stats.mf40_nc_blocks, 0);
    assert!(
        stats.mf40_unmatched_states.is_empty(),
        "{:?}",
        stats.mf40_unmatched_states
    );
    assert_eq!(stats.mf40_states_without_chain_kind, 0);
    assert_eq!(stats.mf40_on_yield_channels, 0);
    assert_eq!(stats.mf40_mat1_naming_itself, 0);
    assert_eq!(stats.mf40_cross_state_blocks, 0);
    assert!(stats.mf40_without_blocks.is_empty());
    // One product per level in each MT, so no partner rests on the own-IZAP
    // reading.
    assert!(stats.mf40_partner_by_own_izap.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every sub-subsection of every section reads back as the parser produced
/// it, with the section HEAD and the state CONT beside it: the file is the
/// tape's MF=40, so a consumer that trusts it is reading the evaluation.
#[test]
fn every_block_round_trips_exactly() {
    let nb93 = material(&text(NB93));
    let out = extract(std::slice::from_ref(&nb93), &nb_decay());
    let dir = scratch("round-trip");
    write_branching_covariance(&out.covariance, &dir).expect("covariance writes");
    let blocks = read_back(&dir);

    let mut expected = 0;
    for mt in [4, 16] {
        let mf40 = nb93.mf40(mt).expect("MF=40 on the tape");
        for (i, state) in mf40.subsections.iter().enumerate() {
            for (j, sub) in state.subsubsections.iter().enumerate() {
                let blocks_of_sub: Vec<_> = sub
                    .nc_subsections
                    .iter()
                    .cloned()
                    .map(CovarianceData::Nc)
                    .chain(sub.ni_subsections.iter().cloned().map(CovarianceData::Ni))
                    .collect();
                for (k, data) in blocks_of_sub.into_iter().enumerate() {
                    expected += 1;
                    let b = blocks
                        .iter()
                        .find(|b| {
                            (
                                b.block.mt,
                                b.state_idx,
                                b.block.subsection_idx,
                                b.block.block_idx,
                            ) == (mt, i as i32, j as i32, k as i32)
                        })
                        .unwrap_or_else(|| panic!("MT{mt} state {i} block {j}/{k} is missing"));
                    assert_eq!(b.block.data, data);
                    assert_eq!(
                        (
                            b.block.xmf1,
                            b.block.xlfs1,
                            b.block.mat1 as i64,
                            b.block.mt1 as i64
                        ),
                        (sub.xmf1, sub.xlfs1, sub.mat1, sub.mt1)
                    );
                    assert_eq!(
                        (b.qm, b.qi, b.izap as i64, b.lfs as i64),
                        (state.qm, state.qi, state.izap, state.lfs)
                    );
                    assert_eq!(
                        (b.za as i64, b.awr, b.lis as i64),
                        (mf40.za, mf40.awr, mf40.lis)
                    );
                }
            }
        }
    }
    assert_eq!(blocks.len(), expected);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A block between two states, in the LB=6 layout, as JEFF-4.0 U235 MT 4
/// correlates its ground with its 77 eV isomer. Here Nb93's (n,n') ground
/// gains a second sub-subsection pairing it with the isomer (XLFS1 1). The
/// rectangular block reads back exactly, and its partner resolves to the
/// isomer while the row's own state stays the ground.
#[test]
fn a_cross_state_lb6_block_round_trips() {
    let tape = text(NB93);
    // The ground state's CONT, now with two sub-subsections.
    let tape = edit(
        tape,
        " 0.000000+0 0.000000+0      41093          0          0          1412540  4    2",
        " 0.000000+0 0.000000+0      41093          0          0          2412540  4    2",
    );
    // After the ground's self block: XMF1 10, XLFS1 1, MT1 4, one NI block,
    // then LB=6 with NER 3 and NT 10, so NEC 3 and a 2 x 2 matrix.
    let last = " 6.391090-2 6.091560-2 5.715220-2 5.814100-2 5.464040-2 5.147370-2412540  4   24";
    let lines = [
        last,
        " 1.000000+1 1.000000+0          0          4          0          1",
        " 0.000000+0 0.000000+0          0          6         10          3",
        " 1.000000-5 1.000000+6 2.000000+7 1.000000-5 5.000000+6 2.000000+7",
        " 1.000000-2-2.000000-3 3.000000-3 4.000000-3",
    ];
    let inserted: Vec<String> = std::iter::once(last.to_string())
        .chain(lines[1..].iter().map(|l| format!("{l:<66}412540  4   24")))
        .collect();
    let tape = edit(tape, last, &inserted.join("\n"));
    let nb93 = material(&tape);
    let out = extract(std::slice::from_ref(&nb93), &nb_decay());
    assert_eq!(out.stats.mf40_cross_state_blocks, 1);
    assert_eq!(out.stats.mf40_blocks_by_lb.get(&6), Some(&1));
    assert_eq!(out.stats.mf40_blocks, 5);

    let dir = scratch("cross-state");
    write_branching_covariance(&out.covariance, &dir).expect("writes");
    let blocks = read_back(&dir);
    let b = blocks
        .iter()
        .find(|b| b.block.mt == 4 && b.lfs == 0 && b.block.subsection_idx == 1)
        .expect("the cross-state block");
    let tape_block = &nb93.mf40(4).expect("MF=40 MT 4").subsections[0].subsubsections[1];
    assert_eq!(
        b.block.data,
        CovarianceData::Ni(tape_block.ni_subsections[0].clone())
    );
    let CovarianceData::Ni(ni) = &b.block.data else {
        panic!("an NI block");
    };
    assert_eq!((ni.lb, ni.nt, ni.ner, ni.nec), (6, 10, 3, 3));
    assert_eq!(ni.er, [1.0e-5, 1.0e6, 2.0e7]);
    assert_eq!(ni.ec, [1.0e-5, 5.0e6, 2.0e7]);
    assert_eq!(ni.fkl, [1.0e-2, -2.0e-3, 3.0e-3, 4.0e-3]);
    assert_eq!((b.block.mt1, b.block.xmf1, b.block.xlfs1), (4, 10.0, 1.0));
    assert_ne!(b.block.xlfs1, b.lfs as f64);
    assert_eq!(b.target.as_deref(), Some("Nb93"));
    assert_eq!(b.target1.as_deref(), Some("Nb93_m1"));
    // MF=33's helper takes it for a self block, never having compared XLFS1
    // with LFS; MF=40's does not.
    assert!(b.block.is_diagonal());
    assert!(!b.is_self_block());
    // The ground's self block is untouched beside it.
    let own = blocks
        .iter()
        .find(|b| b.block.mt == 4 && b.lfs == 0 && b.block.subsection_idx == 0)
        .expect("the self block");
    assert_eq!(own.target1.as_deref(), Some("Nb93"));
    assert!(own.is_self_block());
    let _ = std::fs::remove_dir_all(&dir);
}

/// An NC block, which none of the published libraries has in MF=40, round
/// trips beside the NI block it precedes. Here the (n,n') ground's self
/// sub-subsection gains an LTY=0 block ahead of its LB=5 one: the two share
/// the sub-subsection's key, `block_idx` runs over both, NC first, as the
/// tape orders them, and `mtl` is null on both.
#[test]
fn an_nc_block_round_trips_ahead_of_its_ni_block() {
    let tape = text(NB93);
    let own = " 1.000000+1 0.000000+0          0          4          0          1412540  4    3";
    let with_nc = [
        " 1.000000+1 0.000000+0          0          4          1          1",
        " 0.000000+0 0.000000+0          0          0          0          0",
        " 1.000000+6 2.000000+7          0          0          4          2",
        " 1.000000+0 1.600000+1-1.000000+0 2.200000+1",
    ]
    .map(|l| format!("{l:<66}412540  4    3"));
    let tape = edit(tape, own, &with_nc.join("\n"));
    let nb93 = material(&tape);
    let out = extract(std::slice::from_ref(&nb93), &nb_decay());
    assert_eq!(out.stats.mf40_nc_blocks, 1);
    assert_eq!(out.stats.mf40_blocks, 5);

    let dir = scratch("nc");
    write_branching_covariance(&out.covariance, &dir).expect("writes");
    let blocks = read_back(&dir);
    assert_eq!(blocks.len(), 5);
    let of: Vec<_> = blocks
        .iter()
        .filter(|b| b.block.mt == 4 && b.lfs == 0 && b.block.subsection_idx == 0)
        .collect();
    assert_eq!(of.len(), 2);
    let tape_sub = &nb93.mf40(4).expect("MF=40 MT 4").subsections[0].subsubsections[0];
    let (nc, ni) = (of[0], of[1]);
    assert_eq!((nc.block.block_idx, ni.block.block_idx), (0, 1));
    assert_eq!(
        nc.block.data,
        CovarianceData::Nc(tape_sub.nc_subsections[0].clone())
    );
    assert_eq!(
        ni.block.data,
        CovarianceData::Ni(tape_sub.ni_subsections[0].clone())
    );
    let CovarianceData::Nc(data) = &nc.block.data else {
        panic!("an NC block");
    };
    assert_eq!((data.lty, data.e1, data.e2, data.nci), (0, 1.0e6, 2.0e7, 2));
    assert_eq!(data.ci, [1.0, -1.0]);
    assert_eq!(data.xmti, [16.0, 22.0]);
    for b in [nc, ni] {
        assert_eq!(
            (
                b.state_idx,
                b.block.xmf1,
                b.block.xlfs1,
                b.block.mat1,
                b.block.mt1
            ),
            (0, 10.0, 0.0, 0, 4)
        );
        assert_eq!(
            (b.target.as_deref(), b.target1.as_deref()),
            (Some("Nb93"), Some("Nb93"))
        );
        assert!(b.is_self_block());
    }

    // The kinds and the null MTL as they sit in the file.
    let file = std::fs::File::open(dir.join("branching_covariance.arrow")).expect("the file");
    let reader = arrow_ipc::reader::FileReader::try_new(file, None).expect("arrow");
    let (mut kinds, mut rows, mut null_mtl) = (Vec::new(), 0, 0);
    for batch in reader {
        let batch = batch.expect("a batch");
        let kind = batch
            .column_by_name("kind")
            .expect("kind")
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .expect("utf8");
        kinds.extend(kind.iter().map(|k| k.expect("kind is set").to_string()));
        rows += batch.num_rows();
        null_mtl += batch.column_by_name("mtl").expect("mtl").null_count();
    }
    assert_eq!(kinds.iter().filter(|k| *k == "nc").count(), 1);
    assert_eq!(null_mtl, rows, "MF=40 has no MTL");
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
        .filter(|r| r.reaction.as_deref() == Some("(n,2n)"))
        .collect();
    assert_eq!(n2n.len(), 2, "one sub-subsection per state");
    let production = endf::radionuclide_production::radionuclide_production(&nb93);
    for row in n2n {
        assert_eq!(row.target.as_deref(), Some("Nb92"));
        let state = production[&16]
            .iter()
            .find(|s| s.lfs == row.lfs)
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
        .find(|r| r.target.as_deref() == Some("Nb93_m1"))
        .expect("the (n,n') isomer");
    assert!(isomer.energy.is_none() && isomer.values.is_none());

    // And the reader keeps the null a null.
    let dir = scratch("merged");
    write_branching_covariance(&out.covariance, &dir).expect("writes");
    let blocks = read_back(&dir);
    assert!(blocks
        .iter()
        .filter(|b| b.reaction.as_deref() == Some("(n,2n)"))
        .all(|b| b.energy.is_some() && b.values.is_some()));
    assert!(blocks
        .iter()
        .filter(|b| b.target.as_deref() == Some("Nb93_m1"))
        .all(|b| b.energy.is_none() && b.values.is_none()));
    let _ = std::fs::remove_dir_all(&dir);
}

/// MF=10 and MF=40 need not number a level alike. ENDF/B-VIII.1 Pb204 MT4 has
/// its isomer at LFS=21 in MF=10 and LFS=1 in MF=40, with one QI; here the
/// Nb92_m1 state of (n,2n) is renumbered 7 in MF=40 alone, its QI kept. The
/// row keeps the tape's 7.
#[test]
fn an_mf40_state_matches_mf10_by_excitation_when_lfs_differs() {
    let tape = text(NB93);
    // The state's CONT (LFS, in the fourth field) and its self block's XLFS1.
    let tape = edit(
        tape,
        "-8.830870+6-8.966370+6      41092          1          0          1412540 16   10",
        "-8.830870+6-8.966370+6      41092          7          0          1412540 16   10",
    );
    let tape = edit(
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
        .find(|r| r.reaction.as_deref() == Some("(n,2n)") && r.lfs == 7)
        .expect("the renumbered state is written");
    assert_eq!(
        row.target.as_deref(),
        Some("Nb92_m1"),
        "matched by its 135.5 keV excitation"
    );
    assert_eq!(row.target1.as_deref(), Some("Nb92_m1"));
    assert!(row.energy.is_none(), "still one state per target");
}

/// The excitation fallback does not reach past `tol_ev`. Here the (n,2n)
/// Nb92_m1 state is renumbered 7 and moved to 2 MeV in MF=40 alone, a level
/// MF=10 does not give. Nb92 has one isomer, so the level resolves to Nb92_m1
/// without regard to energy; the 135.5 keV MF=10 partial it lands on is
/// another level's, so the state is left unmatched and counted.
#[test]
fn an_mf40_state_at_no_mf10_excitation_is_unmatched() {
    let tape = text(NB93);
    let tape = edit(
        tape,
        "-8.830870+6-8.966370+6      41092          1          0          1412540 16   10",
        "-8.830870+6-1.083087+7      41092          7          0          1412540 16   10",
    );
    let tape = edit(
        tape,
        " 1.000000+1 1.000000+0          0         16          0          1412540 16   11",
        " 1.000000+1 7.000000+0          0         16          0          1412540 16   11",
    );
    let out = extract(&[material(&tape)], &nb_decay());
    assert_eq!(
        out.stats.mf40_unmatched_states.len(),
        1,
        "{:?}",
        out.stats.mf40_unmatched_states
    );
    assert!(out.stats.mf40_unmatched_states[0].contains("IZAP 41092 LFS 7"));
    let row = out
        .covariance
        .iter()
        .find(|r| r.reaction.as_deref() == Some("(n,2n)") && r.lfs == 7)
        .expect("the state is still written");
    assert_eq!(row.target, None);
    assert_eq!(row.target1, None, "its partner is itself, also unmatched");
    assert!(row.energy.is_none() && row.values.is_none());
    assert_eq!(row.qi, -1.083087e7, "QI is the tape's");
}

/// An MT1 of 0 has no meaning the manual gives in MF=40, so its partner is not
/// read as this MT: `target1` is null and the tape's 0 is kept.
#[test]
fn an_mt1_of_zero_leaves_the_partner_unresolved() {
    let tape = text(NB93);
    let tape = edit(
        tape,
        " 1.000000+1 1.000000+0          0         16          0          1412540 16   11",
        " 1.000000+1 1.000000+0          0          0          0          1412540 16   11",
    );
    let out = extract(&[material(&tape)], &nb_decay());
    let row = out
        .covariance
        .iter()
        .find(|r| r.reaction.as_deref() == Some("(n,2n)") && r.lfs == 1)
        .expect("the (n,2n) isomer");
    assert_eq!(row.subsection.mt1, 0);
    assert_eq!(row.target.as_deref(), Some("Nb92_m1"));
    assert_eq!(row.target1, None);
}

/// JEFF-4.0 U235 MT 4 writes IZAP 0 for the target itself and its own MAT as
/// MAT1. Both are matched the way they are meant, and both are written as the
/// tape gives them: the file says what the evaluator wrote, and `mat` beside
/// `mat1` is what lets a reader see the MAT1 is this evaluation's.
#[test]
fn tape_values_are_written_literally() {
    let tape = text(NB93);
    let tape = edit(
        tape,
        " 0.000000+0-3.077000+4      41093          1          0          1412540  4   25",
        " 0.000000+0-3.077000+4          0          1          0          1412540  4   25",
    );
    let tape = edit(
        tape,
        " 1.000000+1 1.000000+0          0          4          0          1412540  4   26",
        " 1.000000+1 1.000000+0       4125          4          0          1412540  4   26",
    );
    let out = extract(&[material(&tape)], &nb_decay());
    assert!(
        out.stats.mf40_unmatched_states.is_empty(),
        "{:?}",
        out.stats.mf40_unmatched_states
    );
    assert_eq!(out.stats.mf40_mat1_naming_itself, 1);
    let dir = scratch("literal");
    write_branching_covariance(&out.covariance, &dir).expect("writes");
    let blocks = read_back(&dir);
    let b = blocks
        .iter()
        .find(|b| b.block.mt == 4 && b.lfs == 1)
        .expect("the (n,n') isomer");
    assert_eq!(b.izap, 0, "IZAP is the tape's");
    assert_eq!(b.block.mat1, NB93_MAT, "MAT1 is the tape's");
    assert_eq!(b.mat, NB93_MAT);
    // A MAT1 naming the evaluation itself is not another material, whatever
    // MF=33's helper, which compares with zero, says.
    assert!(b.block.is_cross_material());
    assert!(!b.is_cross_material() && b.is_self_block());
    assert_eq!(b.target.as_deref(), Some("Nb93_m1"));
    assert_eq!(b.target1.as_deref(), Some("Nb93_m1"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A state that matches nothing, and a section whose MT has no chain kind, are
/// still written, with the key they could not be given left null. A partner
/// in an MT with MF=10 but no MF=40 section is still resolved.
#[test]
fn states_with_no_chain_row_are_still_written() {
    let tape = text(NB93);
    // The (n,2n) isomer's IZAP moved to Nb91, which MF=10 does not make.
    let tape = edit(
        tape,
        "-8.830870+6-8.966370+6      41092          1          0          1412540 16   10",
        "-8.830870+6-8.966370+6      41091          1          0          1412540 16   10",
    );
    // MF=40 MT 4 moved to MT 18, which has no chain kind. Its MT1 still says
    // 4, a reaction with no MF=40 section left to find the partner in, so the
    // partner is found among MT 4's MF=10 states instead.
    let tape: String = tape
        .lines()
        .map(|line| {
            if line.len() >= 75 && &line[66..75] == "412540  4" {
                format!("{}412540 18{}\n", &line[..66], &line[75..])
            } else {
                format!("{line}\n")
            }
        })
        .collect();
    let out = extract(&[material(&tape)], &nb_decay());
    assert_eq!(out.stats.mf40_sections, 2);
    assert_eq!(out.stats.mf40_states_without_chain_kind, 2);
    assert_eq!(
        out.stats.mf40_unmatched_states.len(),
        1,
        "{:?}",
        out.stats.mf40_unmatched_states
    );
    assert!(out.stats.mf40_unmatched_states[0].contains("IZAP 41091 LFS 1"));

    let dir = scratch("unkeyed");
    write_branching_covariance(&out.covariance, &dir).expect("writes");
    let blocks = read_back(&dir);
    assert_eq!(blocks.len(), 4, "nothing is dropped");
    let fission: Vec<_> = blocks.iter().filter(|b| b.block.mt == 18).collect();
    assert_eq!(fission.len(), 2);
    assert!(fission
        .iter()
        .all(|b| b.reaction.is_none() && b.target.is_none()));
    for b in fission {
        let partner = if b.block.xlfs1 == 0.0 {
            "Nb93"
        } else {
            "Nb93_m1"
        };
        assert_eq!(
            b.target1.as_deref(),
            Some(partner),
            "XLFS1 {}",
            b.block.xlfs1
        );
    }
    let moved = blocks
        .iter()
        .find(|b| b.izap == 41091)
        .expect("the unmatched state");
    assert_eq!(moved.reaction.as_deref(), Some("(n,2n)"));
    assert!(moved.target.is_none() && moved.target1.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}
