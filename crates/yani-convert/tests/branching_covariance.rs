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
/// Read both from disk and as bytes, the browser's path, which must agree.
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
    let blocks =
        branching_covariance_blocks(branch.covariance().expect("the covariance is loaded"))
            .expect("blocks read");

    let mut sections = yani::ChainSections::default();
    for (subsection, from) in [
        ("decay", chain_dir.join("decay")),
        ("branching", dir.to_path_buf()),
    ] {
        for entry in std::fs::read_dir(&from).expect("listed") {
            let path = entry.expect("entry").path();
            if path.extension().is_some_and(|ext| ext == "arrow") {
                let file = path.file_name().expect("named").to_str().expect("UTF-8");
                sections
                    .insert(subsection, file, std::fs::read(&path).expect("read"))
                    .expect("a known subsection");
            }
        }
    }
    let (_, from_bytes) = yani::parse_chain_parts_from_bytes(&sections).expect("bytes load");
    let bytes_blocks = branching_covariance_blocks(
        from_bytes
            .covariance()
            .expect("the covariance is loaded from bytes"),
    )
    .expect("blocks read from bytes");
    assert_eq!(
        bytes_blocks, blocks,
        "the bytes loader reads what the disk one does"
    );
    blocks
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
    assert_eq!(stats.mf40_blocks_outside_mf10, 0);
    assert!(stats.mf40_without_blocks.is_empty());
    // One product per level in each MT, so every partner is pinned, and
    // MF=10 numbers each level as MF=40 does.
    assert!(stats.mf40_partner_unresolved.is_empty());
    assert!(stats.mf40_states_placed_by_excitation.is_empty());
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
    // MF=33's helper takes no MF=40 partner for a cross section, so it is no
    // self block there; MF=40's compares XLFS1 with LFS and is not one either.
    assert!(!b.block.is_diagonal());
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
        assert_eq!(row.quantity.as_deref(), Some("cross_section"));
    }
    let isomer = out
        .covariance
        .iter()
        .find(|r| r.target.as_deref() == Some("Nb93_m1"))
        .expect("the (n,n') isomer");
    assert!(isomer.energy.is_none() && isomer.values.is_none());
    // Its curve is not merged, but which of the target's curves it is still
    // is written.
    assert_eq!(isomer.quantity.as_deref(), Some("cross_section"));

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
    // The self block names the state by its own subsection's label, so MF=10
    // having no level 7 leaves nothing to choose between; that the label is
    // not MF=10's number for the level is on record with the placement.
    assert!(
        out.stats.mf40_partner_unresolved.is_empty(),
        "{:?}",
        out.stats.mf40_partner_unresolved
    );
    assert_eq!(
        out.stats.mf40_states_placed_by_excitation,
        [
            "Nb93 MT16: IZAP 41092 LFS 7 at 135.5 keV is placed on Nb92_m1 by its \
          excitation, MF=9 and MF=10 giving that IZAP and LFS no state"
        ]
    );
}

/// A partner level that MT1's MF=40 section gives no state at has no
/// excitation to confirm an MF=10 level by, so it is null and listed with that
/// reason, not as a level several states share. Here the (n,n') isomer's
/// block is pointed at level 2 of (n,2n), whose MF=40 has only LFS 0 and 1.
#[test]
fn a_partner_level_with_no_mf40_state_is_unresolved() {
    let tape = edit(
        text(NB93),
        " 1.000000+1 1.000000+0          0          4          0          1412540  4   26",
        " 1.000000+1 2.000000+0          0         16          0          1412540  4   26",
    );
    let out = extract(&[material(&tape)], &nb_decay());
    assert_eq!(
        out.stats.mf40_partner_unresolved,
        [
            "Nb93 MT4: IZAP 41093 LFS 1 sub-subsection 0, partner MT16 level 2: \
          that MT's MF=40 section has no state at that level"
        ]
    );
    let row = out
        .covariance
        .iter()
        .find(|r| r.mt == 4 && r.lfs == 1)
        .expect("the (n,n') isomer");
    assert_eq!(row.target.as_deref(), Some("Nb93_m1"));
    assert_eq!(row.target1, None);
}

/// The manual numbers XLFS1 as MF=10 does, and MF=40 need not, so a partner
/// in another MT is pinned only where both numberings put the same state at
/// XLFS1. Here the (n,n') isomer's block is pointed at level 7 of (n,2n),
/// which MF=40 alone gives (the renumbered Nb92_m1), and then at level 1 of
/// (n,2n) with MF=40's ground renumbered 1 and its isomer 7, so that MF=40's
/// level 1 is the ground and MF=10's the isomer. Either way the partner is
/// null and listed.
#[test]
fn a_partner_level_the_two_numberings_disagree_on_is_unresolved() {
    let to_mt16 = |tape: String, xlfs1: &str| {
        edit(
            tape,
            " 1.000000+1 1.000000+0          0          4          0          1412540  4   26",
            &format!(
                " 1.000000+1 {xlfs1}          0         16          0          1412540  4   26"
            ),
        )
    };
    let isomer_row = |out: &yani_convert::branching::Extracted| {
        out.covariance
            .iter()
            .find(|r| r.mt == 4 && r.lfs == 1)
            .expect("the (n,n') isomer")
            .clone()
    };

    let tape = text(NB93);
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
    let out = extract(&[material(&to_mt16(tape, "7.000000+0"))], &nb_decay());
    assert_eq!(
        out.stats.mf40_partner_unresolved,
        [
            "Nb93 MT4: IZAP 41093 LFS 1 sub-subsection 0, partner MT16 level 7: \
          MF=9 and MF=10 give IZAP 41092 no state at that LFS, MF=40 gives it one \
          at 135.5 keV"
        ]
    );
    let row = isomer_row(&out);
    assert_eq!(row.target.as_deref(), Some("Nb93_m1"));
    assert_eq!(row.target1, None);

    let tape = text(NB93);
    let tape = edit(
        tape,
        "-8.830870+6-8.830870+6      41092          0          0          1412540 16    2",
        "-8.830870+6-8.830870+6      41092          1          0          1412540 16    2",
    );
    let tape = edit(
        tape,
        " 1.000000+1 0.000000+0          0         16          0          1412540 16    3",
        " 1.000000+1 1.000000+0          0         16          0          1412540 16    3",
    );
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
    let out = extract(&[material(&to_mt16(tape, "1.000000+0"))], &nb_decay());
    // MF=40's ground, now LFS 1 at QM - QI of 0, states no excitation, so
    // nothing confirms which level it is and it is not placed.
    assert_eq!(
        out.stats.mf40_unmatched_states,
        [
            "Nb93 MT16: IZAP 41092 LFS 1 is excited but QM - QI is 0.0 keV, which \
          states no excitation to confirm an MF=9 or MF=10 state by, so is not placed"
        ]
    );
    assert_eq!(
        out.stats.mf40_partner_unresolved,
        [
            "Nb93 MT4: IZAP 41093 LFS 1 sub-subsection 0, partner MT16 level 1: \
          MF=9 or MF=10 gives IZAP 41092 that LFS at 135.5 keV, MF=40 at 0.0 keV"
        ]
    );
    assert_eq!(isomer_row(&out).target1, None);
    // The isomer, now LFS 7, is placed by its excitation and listed as such,
    // and its self block names it by that label.
    let mt16 = |lfs: i64| {
        out.covariance
            .iter()
            .find(|r| r.mt == 16 && r.lfs == lfs)
            .expect("written")
    };
    assert_eq!(mt16(1).target, None);
    assert_eq!(mt16(1).target1, None);
    assert_eq!(mt16(7).target.as_deref(), Some("Nb92_m1"));
    assert_eq!(mt16(7).target1.as_deref(), Some("Nb92_m1"));
    assert_eq!(
        out.stats.mf40_states_placed_by_excitation,
        [
            "Nb93 MT16: IZAP 41092 LFS 7 at 135.5 keV is placed on Nb92_m1 by its \
          excitation, MF=9 and MF=10 giving that IZAP and LFS no state"
        ]
    );
}

/// The excitation fallback places a state only when one level of the chain
/// nuclide it resolves to sits within `tol_ev`. Here MF=10's (n,2n) ground is
/// moved to a second level 1 keV above the 135.5 keV isomer, so both MF=10
/// states resolve to Nb92_m1, and the MF=40 isomer, renumbered 7, lies within
/// 3 keV of both: nearest would be a guess at which partial weights it, so it
/// is left unplaced and listed.
#[test]
fn an_mf40_state_near_several_mf10_levels_is_not_placed() {
    let tape = text(NB93);
    let tape = edit(
        tape,
        "-8.830870+6-8.830870+6      41092          0          1         31412510 16    2",
        "-8.830870+6-8.967370+6      41092          2          1         31412510 16    2",
    );
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
    let ambiguous: Vec<&String> = out
        .stats
        .mf40_unmatched_states
        .iter()
        .filter(|line| line.contains("IZAP 41092 LFS 7"))
        .collect();
    assert_eq!(
        ambiguous,
        [
            "Nb93 MT16: IZAP 41092 LFS 7 at 135.5 keV is within tolerance of 2 \
          MF=9 or MF=10 states of one target, so is not placed"
        ],
        "{:?}",
        out.stats.mf40_unmatched_states
    );
    let row = out
        .covariance
        .iter()
        .find(|r| r.reaction.as_deref() == Some("(n,2n)") && r.lfs == 7)
        .expect("the state is still written");
    assert_eq!(row.target, None);
    assert_eq!(row.target1, None);
    assert!(row.energy.is_none() && row.values.is_none());
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

/// An (IZAP, LFS) that MF=10 also gives is not taken on the key alone: the
/// two files can number levels differently, so the level has to agree in
/// excitation too. Here the (n,2n) isomer keeps LFS 1 in MF=40 but moves to
/// 2 MeV, where MF=10's LFS 1 is at 135.5 keV. It is not matched to it, nor,
/// the fallback being held to the same tolerance, to anything else.
#[test]
fn an_mf40_state_matching_an_mf10_key_at_another_level_is_unmatched() {
    let tape = text(NB93);
    let tape = edit(
        tape,
        "-8.830870+6-8.966370+6      41092          1          0          1412540 16   10",
        "-8.830870+6-1.083087+7      41092          1          0          1412540 16   10",
    );
    let out = extract(&[material(&tape)], &nb_decay());
    assert_eq!(
        out.stats.mf40_unmatched_states.len(),
        1,
        "{:?}",
        out.stats.mf40_unmatched_states
    );
    assert!(out.stats.mf40_unmatched_states[0].contains("IZAP 41092 LFS 1"));
    let row = out
        .covariance
        .iter()
        .find(|r| r.reaction.as_deref() == Some("(n,2n)") && r.lfs == 1)
        .expect("the state is still written");
    assert_eq!(row.target, None);
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
    // A MAT1 naming the evaluation itself is not another material, for
    // MF=33's helper too, since the block carries the key's MAT.
    assert_eq!(b.block.mat, NB93_MAT);
    assert!(!b.block.is_cross_material());
    assert!(!b.is_cross_material() && b.is_self_block());
    assert_eq!(b.target.as_deref(), Some("Nb93_m1"));
    assert_eq!(b.target1.as_deref(), Some("Nb93_m1"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A state that matches nothing, and a section whose MT has no chain kind, are
/// still written, with the key they could not be given left null. So is a
/// partner in an MT with MF=10 but no MF=40 section: XLFS1 is MF=40's level
/// number, which MF=10's need not be, and there is no MF=40 state to check
/// its excitation against.
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
    // 4, a reaction with no MF=40 section left to find the partner in.
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
        .all(|b| b.reaction.is_none() && b.target.is_none() && b.target1.is_none()));
    let moved = blocks
        .iter()
        .find(|b| b.izap == 41091)
        .expect("the unmatched state");
    assert_eq!(moved.reaction.as_deref(), Some("(n,2n)"));
    assert!(moved.target.is_none() && moved.target1.is_none());
    // The two MT 18 partners have no MF=40 section to be found in. The moved
    // state's block with itself names it by its own label, so its null
    // `target1` is its unmatched `target`, listed once as such.
    let unresolved = &out.stats.mf40_partner_unresolved;
    assert_eq!(unresolved.len(), 2, "{unresolved:?}");
    assert_eq!(
        unresolved
            .iter()
            .filter(|l| l.starts_with("Nb93 MT18") && l.ends_with("that MT has no MF=40 section"))
            .count(),
        2,
        "{unresolved:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Drop the lines of one MF=40 section with the given sequence numbers.
fn drop_lines(tape: String, mt: &str, seqs: std::ops::RangeInclusive<u32>) -> String {
    let control = format!("412540{mt:>3}");
    tape.lines()
        .filter(|line| {
            !(line.len() >= 80
                && line[66..75] == control
                && line[75..80]
                    .trim()
                    .parse::<u32>()
                    .is_ok_and(|n| seqs.contains(&n)))
        })
        .map(|line| format!("{line}\n"))
        .collect()
}

/// MF=40 names a block's partner by MT1 and XLFS1, with no IZAP, so two states
/// of one MT at one level leave it open which of them is meant. Here the
/// (n,n') isomer is renumbered to level 0 in MF=40, beside the ground: neither
/// self block's partner is taken for its own state, both are null and listed.
#[test]
fn two_states_at_the_partner_level_leave_it_unresolved() {
    let tape = text(NB93);
    let tape = edit(
        tape,
        " 0.000000+0-3.077000+4      41093          1          0          1412540  4   25",
        " 0.000000+0-3.077000+4      41093          0          0          1412540  4   25",
    );
    let tape = edit(
        tape,
        " 1.000000+1 1.000000+0          0          4          0          1412540  4   26",
        " 1.000000+1 0.000000+0          0          4          0          1412540  4   26",
    );
    let out = extract(&[material(&tape)], &nb_decay());
    let unresolved = &out.stats.mf40_partner_unresolved;
    assert_eq!(unresolved.len(), 2, "{unresolved:?}");
    assert!(unresolved
        .iter()
        .all(|l| l.starts_with("Nb93 MT4") && l.ends_with("several states sit at that level")));
    let mt4: Vec<_> = out.covariance.iter().filter(|r| r.mt == 4).collect();
    assert_eq!(mt4.len(), 2);
    assert!(mt4.iter().all(|r| r.lfs == 0 && r.target1.is_none()));
    // The (n,2n) states, one per level, are pinned as before.
    assert!(out
        .covariance
        .iter()
        .filter(|r| r.mt == 16)
        .all(|r| r.target1.is_some() && r.target1 == r.target));
}

/// A partner in another material, or one whose XMF1 is not 10, is not an MF=10
/// partial of this evaluation, so `target1` is null. The tape's values are
/// kept, and both count as blocks that are not a state's covariance with
/// itself.
#[test]
fn a_partner_outside_this_mf10_has_no_target() {
    let tape = text(NB93);
    // The (n,n') isomer's partner in another MAT, the (n,2n) isomer's in MF=3.
    let tape = edit(
        tape,
        " 1.000000+1 1.000000+0          0          4          0          1412540  4   26",
        " 1.000000+1 1.000000+0       2625          4          0          1412540  4   26",
    );
    let tape = edit(
        tape,
        " 1.000000+1 1.000000+0          0         16          0          1412540 16   11",
        " 3.000000+0 1.000000+0          0         16          0          1412540 16   11",
    );
    let out = extract(&[material(&tape)], &nb_decay());
    assert_eq!(out.stats.mf40_blocks_outside_mf10, 2);
    assert_eq!(out.stats.mf40_cross_state_blocks, 2);
    assert!(out.stats.mf40_partner_unresolved.is_empty());

    let dir = scratch("outside-mf10");
    write_branching_covariance(&out.covariance, &dir).expect("writes");
    let blocks = read_back(&dir);
    let foreign = blocks
        .iter()
        .find(|b| b.block.mt == 4 && b.lfs == 1)
        .expect("the (n,n') isomer");
    assert_eq!(foreign.block.mat1, 2625);
    assert!(foreign.is_cross_material() && !foreign.is_self_block());
    assert_eq!(foreign.target.as_deref(), Some("Nb93_m1"));
    assert_eq!(foreign.target1, None);
    let mf3 = blocks
        .iter()
        .find(|b| b.block.mt == 16 && b.lfs == 1)
        .expect("the (n,2n) isomer");
    assert_eq!(mf3.block.xmf1, 3.0);
    assert!(!mf3.is_cross_material() && !mf3.is_self_block());
    assert_eq!(mf3.target.as_deref(), Some("Nb92_m1"));
    assert_eq!(mf3.target1, None);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A product state with no sub-subsection (NL 0), and a sub-subsection with no
/// block (NC and NI 0), hold no covariance and so have no row. Each is listed
/// with every tape value it holds, and that list is what the conversion
/// writes to `provenance.json` beside the file.
#[test]
fn parts_without_a_block_are_listed_with_their_tape_values() {
    let tape = text(NB93);
    // The (n,2n) ground keeps its sub-subsection but loses its block.
    let tape = edit(
        tape,
        " 1.000000+1 0.000000+0          0         16          0          1412540 16    3",
        " 1.000000+1 0.000000+0          0         16          0          0412540 16    3",
    );
    // The (n,2n) isomer loses its sub-subsection.
    let tape = edit(
        tape,
        "-8.830870+6-8.966370+6      41092          1          0          1412540 16   10",
        "-8.830870+6-8.966370+6      41092          1          0          0412540 16   10",
    );
    let tape = drop_lines(tape, "16", 4..=9);
    let tape = drop_lines(tape, "16", 11..=17);
    let nb93 = material(&tape);
    let mf40 = nb93.mf40(16).expect("MF=40 MT 16");
    assert!(mf40.subsections[0].subsubsections[0]
        .ni_subsections
        .is_empty());
    assert!(mf40.subsections[1].subsubsections.is_empty());

    let out = extract(std::slice::from_ref(&nb93), &nb_decay());
    assert_eq!(
        out.stats.mf40_without_blocks,
        [
            "Nb93 MT16 state 0 sub-subsection 0: MAT1 0 MT1 16 XMF1 10 XLFS1 0 has no block",
            "Nb93 MT16 state 1: QM -8830870 QI -8966370 IZAP 41092 LFS 1 has no sub-subsection",
        ]
    );
    assert_eq!(out.stats.mf40_blocks, 2);
    let dir = scratch("without-blocks");
    write_branching_covariance(&out.covariance, &dir).expect("writes");
    let blocks = read_back(&dir);
    assert_eq!(blocks.len(), 2);
    assert!(blocks.iter().all(|b| b.block.mt == 4));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A section with no product state (NS 0) has no row either, so it is listed
/// with its HEAD. Here MF=40 MT 4 keeps its HEAD and loses both states.
#[test]
fn a_section_without_a_product_state_is_listed_with_its_head() {
    let tape = text(NB93);
    let tape = edit(
        tape,
        " 4.109300+4 9.210827+1          0          0          2          0412540  4    1",
        " 4.109300+4 9.210827+1          0          0          0          0412540  4    1",
    );
    let tape = drop_lines(tape, "4", 2..=47);
    let nb93 = material(&tape);
    assert!(nb93.mf40(4).expect("MF=40 MT 4").subsections.is_empty());

    let out = extract(std::slice::from_ref(&nb93), &nb_decay());
    assert_eq!(
        out.stats.mf40_without_blocks,
        ["Nb93 MT4: ZA 41093 AWR 92.10827 LIS 0 has no product state"]
    );
    assert_eq!(out.stats.mf40_sections, 2);
    assert_eq!(out.stats.mf40_blocks, 2);
    assert!(out.covariance.iter().all(|r| r.mt == 16));
}

/// A state MF=9 gives as a yield has no MF=10 partial, and is counted. Here
/// MF=40 MT 16 becomes MT 102, its ground state IZAP 41094 at the Q value of
/// Nb93's MF=9 capture yield to Nb94. The decay fixtures have no Nb94 isomer,
/// so MF=9's 40.9 keV level lands on Nb94 too and `branching.arrow` carries
/// the two yields summed: the row then carries the state's own yield, since
/// a relative covariance of one state is weighted by its own partial.
#[test]
fn a_state_on_a_merged_mf9_yield_carries_its_own_yield() {
    let tape = text(NB93);
    let tape = edit(
        tape,
        "-8.830870+6-8.830870+6      41092          0          0          1412540 16    2",
        " 7.227550+6 7.227550+6      41094          0          0          1412540 16    2",
    );
    let tape = edit(
        tape,
        " 1.000000+1 0.000000+0          0         16          0          1412540 16    3",
        " 1.000000+1 0.000000+0          0        102          0          1412540 16    3",
    );
    let tape: String = tape
        .lines()
        .map(|line| {
            if line.len() >= 75 && &line[66..75] == "412540 16" {
                format!("{}412540102{}\n", &line[..66], &line[75..])
            } else {
                format!("{line}\n")
            }
        })
        .collect();
    let out = extract(&[material(&tape)], &nb_decay());
    assert_eq!(out.stats.mf40_on_yield_channels, 1, "{:?}", out.stats);
    let row = out
        .covariance
        .iter()
        .find(|r| r.mt == 102 && r.lfs == 0)
        .expect("the capture ground state");
    assert_eq!(row.reaction.as_deref(), Some("(n,gamma)"));
    assert_eq!(row.target.as_deref(), Some("Nb94"));
    assert_eq!(row.target1.as_deref(), Some("Nb94"));
    let nb94 = material(&tape);
    let production = endf::radionuclide_production::radionuclide_production(&nb94);
    let state = production[&102]
        .iter()
        .find(|s| s.lfs == 0)
        .expect("the MF=9 ground state");
    let (energy, values) = yani_convert::branching::linearize(
        state.yields.as_ref().expect("an MF=9 yield"),
        DEFAULT_LINEARIZE_TOL,
    );
    assert_eq!(row.energy.as_ref(), Some(&energy));
    assert_eq!(row.values.as_ref(), Some(&values));
    assert_eq!(row.quantity.as_deref(), Some("yield"));
    let merged = out
        .rows
        .iter()
        .find(|r| r.reaction == "(n,gamma)" && r.target == "Nb94")
        .expect("the merged capture row");
    assert_eq!(merged.quantity, "yield");
    assert_ne!(merged.values, values, "branching.arrow has the sum");

    let dir = scratch("merged-yield");
    write_branching_covariance(&out.covariance, &dir).expect("writes");
    let blocks = read_back(&dir);
    let block = blocks
        .iter()
        .find(|b| b.block.mt == 102 && b.lfs == 0)
        .expect("read back");
    assert_eq!(block.quantity.as_deref(), Some("yield"));
    assert_eq!(block.values.as_ref(), Some(&values));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The excitation fallback never puts an excited state on a ground partial.
/// With no decay file for Nb92's isomer, the (n,2n) isomer, renumbered 7 in
/// MF=40 and moved to 1 keV, resolves to Nb92, as MF=10's ground does, and
/// lies within 3 keV of it; but it is an excited level, and MF=10 gives none
/// there, so it is left unmatched rather than keyed to the ground partial.
#[test]
fn an_excited_mf40_state_is_never_placed_on_a_ground_partial() {
    let tape = text(NB93);
    let tape = edit(
        tape,
        "-8.830870+6-8.966370+6      41092          1          0          1412540 16   10",
        "-8.830870+6-8.831870+6      41092          7          0          1412540 16   10",
    );
    let tape = edit(
        tape,
        " 1.000000+1 1.000000+0          0         16          0          1412540 16   11",
        " 1.000000+1 7.000000+0          0         16          0          1412540 16   11",
    );
    let decay = vec![material(&text(fixture!("dec-041_Nb_093m1.endf.xz")))];
    let out = extract(&[material(&tape)], &decay);
    assert_eq!(
        out.stats.mf40_unmatched_states,
        ["Nb93 MT16: IZAP 41092 LFS 7 at 1.0 keV matches no MF=9 or MF=10 state"]
    );
    let row = |lfs: i64| {
        out.covariance
            .iter()
            .find(|r| r.mt == 16 && r.lfs == lfs)
            .expect("written")
    };
    assert_eq!(row(7).target, None);
    assert_eq!(row(7).target1, None);
    assert_eq!(
        row(0).target.as_deref(),
        Some("Nb92"),
        "the ground is its own"
    );
    assert!(out.stats.mf40_states_placed_by_excitation.is_empty());
}

/// An excited MF=40 state whose QM - QI is not positive states no energy, so
/// nothing on the tape confirms which MF=9 or MF=10 level its LFS names: not
/// the (IZAP, LFS) join, since the two files need not number levels alike,
/// and not the excitation fallback. Here the (n,2n) isomer is given QI equal
/// to QM, first at its own LFS 1, which MF=10 also gives, then renumbered 7.
#[test]
fn an_excited_mf40_state_with_no_energy_is_not_placed() {
    for lfs in ["1", "7"] {
        let tape = text(NB93);
        let tape = edit(
            tape,
            "-8.830870+6-8.966370+6      41092          1          0          1412540 16   10",
            &format!(
                "-8.830870+6-8.830870+6      41092          {lfs}          0          1412540 16   10"
            ),
        );
        let tape = edit(
            tape,
            " 1.000000+1 1.000000+0          0         16          0          1412540 16   11",
            &format!(
                " 1.000000+1 {lfs}.000000+0          0         16          0          1412540 16   11"
            ),
        );
        let out = extract(&[material(&tape)], &nb_decay());
        assert_eq!(
            out.stats.mf40_unmatched_states,
            [format!(
                "Nb93 MT16: IZAP 41092 LFS {lfs} is excited but QM - QI is 0.0 keV, which \
                 states no excitation to confirm an MF=9 or MF=10 state by, so is not placed"
            )]
        );
        let row = out
            .covariance
            .iter()
            .find(|r| r.mt == 16 && r.lfs.to_string() == lfs)
            .expect("the state is still written");
        assert_eq!(row.target, None, "LFS {lfs}");
        assert_eq!(row.target1, None, "LFS {lfs}");
        assert_eq!(row.qi, row.qm, "QI is the tape's");
        assert!(out.stats.mf40_partner_unresolved.is_empty());
    }
}

/// LFS 0 names the ground whatever QM - QI says, as MF=9 and MF=10 are read,
/// so a ground whose QI sits 1 MeV below its QM is still placed on the ground
/// partial, and the tape's QI is kept.
#[test]
fn an_mf40_ground_is_placed_whatever_its_qm_minus_qi() {
    let tape = text(NB93);
    let tape = edit(
        tape,
        "-8.830870+6-8.830870+6      41092          0          0          1412540 16    2",
        "-8.830870+6-9.830870+6      41092          0          0          1412540 16    2",
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
        .find(|r| r.mt == 16 && r.lfs == 0)
        .expect("the ground is written");
    assert_eq!(row.target.as_deref(), Some("Nb92"));
    assert_eq!(row.qi, -9.83087e6, "QI is the tape's");
    assert!(out.stats.mf40_states_placed_by_excitation.is_empty());
}
