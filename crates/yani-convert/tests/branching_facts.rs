//! Each branching row carries what the evaluation states about the production
//! states behind it, and none of it changes the row.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};
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

/// The chain the reader tests put the branching beside: the downloaded
/// ENDF/B-VIII.1 split fixture, whose decay data names Nb93 and its products.
/// `python3 scripts/fetch_test_fixtures.py` populates it; skip rather than
/// fail when it is absent, as the rest of the suite does.
fn chain_fixture() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../yamc/tests/transmutation-endf-b8.1-sfr.arrow");
    dir.join("decay/nuclides.arrow").exists().then_some(dir)
}

fn read_back(chain: &Path, branching: &Path) -> yani::BranchTable {
    yani::parse_chain_parts(
        &chain.join("decay"),
        Some(&chain.join("reactions")),
        Some(&chain.join("fission_yields")),
        Some(branching),
    )
    .expect("yani reads the branching subsection")
    .1
}

/// The six columns a branching file had before the facts were stored.
fn write_without_facts(rows: &[BranchingRow], dir: &Path) {
    std::fs::create_dir_all(dir).expect("mkdir");
    let strings = |f: fn(&BranchingRow) -> &str| -> ArrayRef {
        let mut b = StringBuilder::new();
        for r in rows {
            b.append_value(f(r));
        }
        Arc::new(b.finish())
    };
    let lists = |f: fn(&BranchingRow) -> &[f64]| -> ArrayRef {
        let mut b = ListBuilder::new(Float64Builder::new());
        for r in rows {
            b.values().append_slice(f(r));
            b.append(true);
        }
        Arc::new(b.finish())
    };
    let list = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
    let schema = Arc::new(Schema::new(vec![
        Field::new("nuclide", DataType::Utf8, false),
        Field::new("reaction", DataType::Utf8, false),
        Field::new("target", DataType::Utf8, false),
        Field::new("quantity", DataType::Utf8, false),
        Field::new("energy", list.clone(), false),
        Field::new("values", list, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            strings(|r| &r.nuclide),
            strings(|r| &r.reaction),
            strings(|r| &r.target),
            strings(|r| &r.quantity),
            lists(|r| &r.energy),
            lists(|r| &r.values),
        ],
    )
    .expect("batch");
    let file = std::fs::File::create(dir.join("branching.arrow")).expect("create");
    let mut writer = FileWriter::try_new(file, &schema).expect("writer");
    writer.write(&batch).expect("write");
    writer.finish().expect("finish");
}

/// The curves yani folds are the same with and without the facts, and the
/// facts come back as written: MF=3 sampled on each curve's own nodes, null
/// where MF=3 is not tabulated.
#[test]
fn the_facts_round_trip_and_leave_the_curves_alone() {
    let Some(chain) = chain_fixture() else {
        eprintln!("skipping: the transmutation-endf-b8.1-sfr.arrow fixture is not downloaded");
        return;
    };
    let (rows, _) = nb93_rows();
    let dir = std::env::temp_dir().join(format!("yani-branching-facts-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    yani_convert::branching::write_branching(&rows, &dir.join("with")).expect("write");
    write_without_facts(&rows, &dir.join("without"));
    let with = read_back(&chain, &dir.join("with"));
    let without = read_back(&chain, &dir.join("without"));

    let with_nb = &with["Nb93"];
    let without_nb = &without["Nb93"];
    assert_eq!(with_nb.len(), without_nb.len());
    for (kind, curves) in with_nb {
        let old = &without_nb[kind];
        assert_eq!(curves.len(), old.len(), "{kind}");
        for (new, old) in curves.iter().zip(old) {
            assert_eq!(new.target, old.target);
            assert_eq!(new.quantity, old.quantity);
            // Bit for bit, not within a tolerance.
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&new.energy),
                bits(&old.energy),
                "{kind} {}",
                new.target
            );
            assert_eq!(
                bits(&new.values),
                bits(&old.values),
                "{kind} {}",
                new.target
            );
            assert!(old.states.is_empty() && old.normalisation.is_none());
            assert!(!new.states.is_empty());
            assert_eq!(new.normalisation.as_deref(), Some(NB93_NORMALISATION));
        }
    }

    let mf3 = nb93().mf3(16).expect("MT=16").sigma.clone();
    let curve = with_nb["(n,2n)"]
        .iter()
        .find(|c| c.target == "Nb92_m1")
        .expect("(n,2n) -> Nb92_m1");
    let state = &curve.states[0];
    assert_eq!((state.mt, state.lfs, state.lmf), (16, 1, Some(10)));
    assert!(state.list_complete);
    assert_eq!(state.level_route, "energy");
    assert_eq!(state.level_energy_difference, Some(0.0));
    let sampled = state.mf3_cross_section.as_ref().expect("MF=3 is given");
    assert_eq!(sampled.len(), curve.energy.len());
    let (lo, hi) = (mf3.x[0], *mf3.x.last().expect("points"));
    let mut shared_nodes = 0;
    for (&e, &value) in curve.energy.iter().zip(sampled) {
        if e < lo || e > hi {
            assert_eq!(value, None, "MF=3 states nothing at {e} eV");
            continue;
        }
        assert_eq!(value, Some(mf3.eval(e)), "at {e} eV");
        // Where the partial and MF=3 share a node, the value is the tape's.
        if let Ok(k) = mf3.x.binary_search_by(|x| x.total_cmp(&e)) {
            assert_eq!(value, Some(mf3.y[k]));
            shared_nodes += 1;
        }
    }
    assert!(
        shared_nodes > 0,
        "no node in common, so the literal check checked nothing"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
