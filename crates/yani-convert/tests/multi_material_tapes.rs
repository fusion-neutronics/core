//! A tape of many materials converts to what its materials, split into one
//! file each, convert to.
//!
//! JEFF-4.0 ships its decay data and fission yields as one tape each, the
//! yields with no TPID record, and IRDFF-II its neutron library as one tape of
//! 70. Each test reads the same evaluations twice, one file per material and
//! as tapes made of those files, and compares what comes out byte for byte.
//! Every tape has materials after its first that the output needs, so a
//! reader that stopped at the first material cannot pass.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("../../endf/fixtures/", $name))
    };
}

/// Cs137 opens the tape and Nb93m1 closes it. U235 is the parent the yields
/// belong to, and the In116 and Nb isomers are what branching resolves to.
const DECAY: &[&[u8]] = &[
    fixture!("dec-055_Cs_137.endf.xz"),
    fixture!("dec-054_Xe_136.endf.xz"),
    fixture!("dec-054_Xe_137.endf.xz"),
    fixture!("dec-049_In_115.endf.xz"),
    fixture!("dec-049_In_116.endf.xz"),
    fixture!("dec-049_In_116m1.endf.xz"),
    fixture!("dec-049_In_116m2.endf.xz"),
    fixture!("dec-050_Sn_115.endf.xz"),
    fixture!("dec-050_Sn_116.endf.xz"),
    fixture!("dec-048_Cd_116.endf.xz"),
    fixture!("dec-092_U_235.endf.xz"),
    fixture!("dec-041_Nb_092.endf.xz"),
    fixture!("dec-041_Nb_092m1.endf.xz"),
    fixture!("dec-041_Nb_093m1.endf.xz"),
];

/// In115 and Nb93 carry MF=9/10, Nb93 MF=40 too, and U235 fission.
const NEUTRON: &[&[u8]] = &[
    fixture!("n-049_In-115_trimmed.endf.xz"),
    fixture!("n-054_Xe_136_trimmed.endf.xz"),
    fixture!("n-092_U_235_trimmed.endf.xz"),
    fixture!("n-041_Nb_093_tendl2017_trimmed.endf.xz"),
];

const U235_YIELDS: &[u8] = fixture!("synthetic-nfy.endf.xz");

fn provenance() -> yani_convert::Provenance {
    yani_convert::Provenance {
        library: "jeff-4.0".to_string(),
        decay_library: String::new(),
        data_version: "2026-10-23".to_string(),
        created_utc: "2026-10-23T00:00:00+00:00".to_string(),
        converter: String::new(),
    }
}

fn text(compressed: &[u8]) -> String {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    String::from_utf8(out).expect("fixture is UTF-8")
}

/// The synthetic U235 yields restated as Th232's (ZA 90232, MAT 9040), so
/// the yield tape opens, as JEFF-4.0's does, with a material ahead of the one
/// the chain reads.
fn as_th232(u235: &str) -> String {
    let mut out = String::new();
    for line in u235.lines() {
        let mut line = line.to_string();
        if line.get(66..70) == Some("9999") {
            line.replace_range(66..70, "9040");
        }
        if line.get(70..80) == Some(" 1451    1") {
            line.replace_range(0..11, " 9.023200+4");
        }
        out.push_str(&line);
        out.push('\n');
    }
    assert!(
        out.contains(" 9.023200+4"),
        "the HEAD record was not restated"
    );
    out
}

/// One-material files joined into a tape: the first file's TPID if `tpid`,
/// every material through its MEND, then one TEND.
fn tape(files: &[String], tpid: bool) -> String {
    let mut out = String::new();
    for (i, file) in files.iter().enumerate() {
        let mut lines = file.lines();
        let header = lines.next().expect("a file opens with its TPID");
        assert_ne!(header.get(70..75), Some(" 1451"), "a fixture has no TPID");
        if tpid && i == 0 {
            out.push_str(header);
            out.push('\n');
        }
        for line in lines.take_while(|line| line.get(66..70).map(str::trim) != Some("-1")) {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str(&format!("{:66}{:>4}{:>2}{:>3}{:>5}\n", "", -1, 0, 0, 0));
    out
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("yani-convert-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

fn write(dir: &Path, stem: &str, texts: &[String]) -> Vec<String> {
    texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let path = dir.join(format!("{stem}{i}.endf"));
            std::fs::write(&path, text).expect("write input");
            path.to_string_lossy().into_owned()
        })
        .collect()
}

/// Every file under `root`, by its path below `root`, with its bytes.
fn tree(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read output directory") {
            let path = entry.expect("directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let bytes = std::fs::read(&path).expect("read output file");
                let below = path.strip_prefix(root).expect("under root").to_path_buf();
                out.insert(below, bytes);
            }
        }
    }
    out
}

/// The chain and its branching from tapes, against the same evaluations one
/// per file: the decay and neutron tapes with a TPID, the yield tape without
/// one, as JEFF-4.0's.
#[test]
fn tapes_convert_to_what_their_split_files_convert_to() {
    let dir = scratch("tapes");
    let decay: Vec<String> = DECAY.iter().map(|b| text(b)).collect();
    let neutron: Vec<String> = NEUTRON.iter().map(|b| text(b)).collect();
    let u235 = text(U235_YIELDS);
    let fpy = vec![as_th232(&u235), u235];

    let split = [
        write(&dir, "d", &decay),
        write(&dir, "f", &fpy),
        write(&dir, "n", &neutron),
    ];
    let tapes = [
        write(&dir, "decay-tape", &[tape(&decay, true)]),
        write(&dir, "fpy-tape", &[tape(&fpy, false)]),
        write(&dir, "neutron-tape", &[tape(&neutron, true)]),
    ];

    let convert = |[decay, fpy, neutron]: &[Vec<String>; 3], out: &Path| {
        yani_convert::convert_transmutation_files(
            decay,
            fpy,
            neutron,
            &[],
            "",
            None,
            None,
            None,
            out,
            &provenance(),
        )
        .expect("chain conversion succeeds");
        yani_convert::convert_branching_files(
            neutron,
            decay,
            out,
            &provenance(),
            3000.0,
            yani_convert::branching::DEFAULT_LINEARIZE_TOL,
        )
        .expect("branching conversion succeeds")
    };
    let (from_split, from_tapes) = (dir.join("split"), dir.join("tapes"));
    let split_stats = convert(&split, &from_split);
    let tape_stats = convert(&tapes, &from_tapes);

    // Each tape's later materials reach the output, so the comparison below
    // is not between two trees that both stopped at a first material.
    let (chain, _) = yani::parse_chain_parts(
        &from_tapes.join("decay"),
        Some(&from_tapes.join("reactions")),
        Some(&from_tapes.join("fission_yields")),
        &[],
    )
    .expect("yani reads the chain converted from tapes");
    let evaluated = chain.values().filter(|n| n.decay_evaluated).count();
    assert_eq!(evaluated, DECAY.len(), "decay materials were lost");
    assert!(
        chain.contains_key("Nb93_m1"),
        "the last decay material was lost"
    );
    assert!(
        chain["U235"].fission_yields.is_some(),
        "U235's yields, second on a tape with no TPID, were lost"
    );
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(from_tapes.join("manifest.json")).expect("manifest"),
    )
    .expect("json");
    let parents = |subsection: &str| manifest["subsections"][subsection]["parents"].clone();
    // Every neutron evaluation is a reactions parent, with or without a decay
    // record of its own (the fixtures' decay data has Nb93_m1 and not Nb93).
    assert_eq!(
        parents("reactions"),
        serde_json::json!(["In115", "Nb93", "U235", "Xe136"])
    );
    assert_eq!(
        parents("branching"),
        serde_json::json!(["In115", "Nb93", "U235"])
    );

    let (split_tree, tape_tree) = (tree(&from_split), tree(&from_tapes));
    assert_eq!(
        tape_tree.keys().collect::<Vec<_>>(),
        split_tree.keys().collect::<Vec<_>>(),
        "tapes and split files wrote different sets of files"
    );
    for (path, bytes) in &split_tree {
        assert!(
            tape_tree[path] == *bytes,
            "{} differs between tapes and split files",
            path.display()
        );
    }
    assert_eq!(tape_stats, split_stats);

    let _ = std::fs::remove_dir_all(&dir);
}

/// The radionuclide production survey reads every material of a tape.
#[test]
fn production_reads_every_material_of_a_tape() {
    let dir = scratch("production-tape");
    let neutron: Vec<String> = NEUTRON.iter().map(|b| text(b)).collect();
    let split = write(&dir, "n", &neutron);
    let one = write(&dir, "tape", &[tape(&neutron, true)]);

    let from_split =
        yani_convert::production::production_from_files(&split).expect("split files read");
    let from_tape = yani_convert::production::production_from_files(&one).expect("tape reads");
    assert!(
        from_split.iter().any(|c| c.parent == "Nb93"),
        "the last material gave no channels, so this proves nothing"
    );
    assert_eq!(from_tape, from_split);

    let _ = std::fs::remove_dir_all(&dir);
}

/// A file holding no material is refused by name rather than read as nothing.
#[test]
fn a_file_with_no_material_is_refused() {
    let dir = scratch("no-material");
    let record = |text: &str, mat: i32| format!("{text:66}{mat:>4}{:>2}{:>3}{:>5}\n", 0, 0, 0);
    let empty = write(
        &dir,
        "empty",
        &[record(" a TPID and no material", 1) + &record("", -1)],
    );

    let err = yani_convert::convert_transmutation_files(
        &empty,
        &[],
        &[],
        &[],
        "",
        None,
        None,
        Some(&["decay".to_string()]),
        &dir.join("out"),
        &provenance(),
    )
    .expect_err("a decay file with no material converted");
    let message = err.to_string();
    assert!(
        message.contains(&empty[0]) && message.contains("no ENDF material"),
        "{message}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
