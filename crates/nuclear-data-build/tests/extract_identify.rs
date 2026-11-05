//! Unpacking the sources per sublibrary, in recipe order, and naming every
//! material from its header, on small archives built here and on the ENDF
//! parser's own fixtures.

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};

use common::*;
use nuclear_data_build::extract::{extract, sources_dir, MARKER};
use nuclear_data_build::identify::identify;
use nuclear_data_build::recipe::{Source, Sublibrary};
use nuclear_data_build::sources::Fetched;

const FE56: &[u8] = include_bytes!("../../endf/fixtures/n-026_Fe_056_trimmed.endf.xz");
const LI6: &[u8] = include_bytes!("../../endf/fixtures/n-003_Li_006_trimmed.endf.xz");
const CS137: &[u8] = include_bytes!("../../endf/fixtures/dec-055_Cs_137.endf.xz");
const U235: &[u8] = include_bytes!("../../endf/fixtures/dec-092_U_235.endf.xz");

fn unxz(compressed: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut lzma_rust2::XzReader::new(compressed, true), &mut out).unwrap();
    out
}

fn tar_gz(path: &Path, members: &[(&str, &[u8])]) {
    let file = std::fs::File::create(path).unwrap();
    let gz = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
    let mut tar = tar::Builder::new(gz);
    for (name, body) in members {
        let mut header = tar::Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        // Written raw, since the builder refuses names with `..` in them,
        // which is exactly what a hostile archive would carry.
        let raw = &mut header.as_gnu_mut().unwrap().name;
        raw[..name.len()].copy_from_slice(name.as_bytes());
        header.set_cksum();
        tar.append(&header, *body).unwrap();
    }
    tar.into_inner().unwrap().finish().unwrap();
}

fn zip_file(path: &Path, members: &[(&str, &[u8])]) {
    let file = std::fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, body) in members {
        zip.start_file(*name, options).unwrap();
        zip.write_all(body).unwrap();
    }
    zip.finish().unwrap();
}

fn fetched(path: &Path, name: &str, sub: Sublibrary, keep: fn(&str) -> bool) -> Fetched {
    let body = read(path);
    let mut source: Source = source(name, "https://example.invalid/", &body, sub);
    source.keep = keep;
    Fetched {
        source,
        path: path.to_path_buf(),
    }
}

fn evaluations(n: &str) -> bool {
    n.ends_with(".endf")
}

#[test]
fn updates_replace_earlier_files_and_only_kept_members_are_written() {
    let scratch = Scratch::new("updates");
    let base = scratch.0.join("jendl5-n.tar.gz");
    let upd = scratch.0.join("jendl5-n_upd7.tar.gz");
    tar_gz(
        &base,
        &[
            ("jendl5-n/n_001-H-002.endf", b"base H2"),
            ("jendl5-n/n_026-Fe-056.endf", b"base Fe56"),
            ("jendl5-n/README.md", b"not data"),
        ],
    );
    tar_gz(
        &upd,
        &[
            ("n_001-H-002.endf", b"upd7 H2"),
            ("../../escape.endf", b"stays in the folder"),
        ],
    );
    let got = vec![
        fetched(&base, "jendl5-n.tar.gz", Sublibrary::Neutron, evaluations),
        fetched(
            &upd,
            "jendl5-n_upd7.tar.gz",
            Sublibrary::Neutron,
            evaluations,
        ),
    ];
    let r = recipe("jendl-5.0", got.iter().map(|f| f.source).collect());
    let root = scratch.dir("lib");
    let folders = extract(&r, &got, &root).unwrap();
    let dir = sources_dir(&root).join("neutron");
    assert_eq!(folders, vec![(Sublibrary::Neutron, dir.clone())]);
    assert_eq!(read(&dir.join("n_001-H-002.endf")), b"upd7 H2");
    assert_eq!(read(&dir.join("n_026-Fe-056.endf")), b"base Fe56");
    assert_eq!(read(&dir.join("escape.endf")), b"stays in the folder");
    assert!(!dir.join("README.md").exists());
    assert!(!root.join("escape.endf").exists());
    let marker = std::fs::read_to_string(dir.join(MARKER)).unwrap();
    assert!(marker.ends_with("jendl5-n_upd7.tar.gz\n"), "{marker}");
}

#[test]
fn a_rerun_with_the_same_sources_leaves_the_folder_and_other_sources_redo_it() {
    let scratch = Scratch::new("rerun");
    let a = scratch.0.join("a.zip");
    zip_file(&a, &[("neutron/endf/n-Fe056.endf", b"Fe56")]);
    let f = fetched(&a, "a.zip", Sublibrary::Neutron, evaluations);
    let r = recipe("fendl-3.2d", vec![f.source]);
    let root = scratch.dir("lib");
    extract(&r, std::slice::from_ref(&f), &root).unwrap();
    let dir = sources_dir(&root).join("neutron");
    std::fs::write(dir.join("sentinel"), b"").unwrap();
    extract(&r, std::slice::from_ref(&f), &root).unwrap();
    assert!(dir.join("sentinel").exists(), "unpacked again");

    zip_file(&a, &[("neutron/endf/n-Fe056.endf", b"Fe56 replaced")]);
    let f = fetched(&a, "a.zip", Sublibrary::Neutron, evaluations);
    let r = recipe("fendl-3.2d", vec![f.source]);
    extract(&r, std::slice::from_ref(&f), &root).unwrap();
    assert!(!dir.join("sentinel").exists(), "not unpacked again");
    assert_eq!(read(&dir.join("n-Fe056.endf")), b"Fe56 replaced");
}

#[test]
fn an_interrupted_unpack_is_redone() {
    let scratch = Scratch::new("interrupted");
    let a = scratch.0.join("a.tar.gz");
    tar_gz(&a, &[("n-Fe056.endf", b"Fe56")]);
    let f = fetched(&a, "a.tar.gz", Sublibrary::Neutron, evaluations);
    let r = recipe("endf-b8.1", vec![f.source]);
    let root = scratch.dir("lib");
    let dir = sources_dir(&root).join("neutron");
    // What a run stopped before writing its marker leaves.
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("n-Fe056.endf"), b"Fe5").unwrap();
    extract(&r, &[f], &root).unwrap();
    assert_eq!(read(&dir.join("n-Fe056.endf")), b"Fe56");
}

#[test]
fn two_members_with_one_name_in_an_archive_are_refused() {
    let scratch = Scratch::new("duplicate");
    let a = scratch.0.join("a.tar.gz");
    tar_gz(
        &a,
        &[("x/n-Fe056.endf", b"one"), ("y/n-Fe056.endf", b"two")],
    );
    let f = fetched(&a, "a.tar.gz", Sublibrary::Neutron, evaluations);
    let r = recipe("endf-b8.1", vec![f.source]);
    let err = extract(&r, &[f], &scratch.dir("lib")).unwrap_err();
    assert!(err.contains("holds two files named n-Fe056.endf"), "{err}");
}

#[test]
fn an_archive_with_no_evaluations_is_refused() {
    let scratch = Scratch::new("empty");
    let a = scratch.0.join("a.tar.gz");
    tar_gz(&a, &[("README.md", b"x")]);
    let f = fetched(&a, "a.tar.gz", Sublibrary::Decay, evaluations);
    let r = recipe("endf-b8.1", vec![f.source]);
    let err = extract(&r, &[f], &scratch.dir("lib")).unwrap_err();
    assert!(err.contains("holds no decay evaluations"), "{err}");
}

#[test]
fn a_file_that_is_not_the_archive_it_is_named_as_is_refused() {
    let scratch = Scratch::new("not-archive");
    let a = scratch.0.join("a.zip");
    std::fs::write(&a, b"<html>a login page</html>").unwrap();
    let f = fetched(&a, "a.zip", Sublibrary::Neutron, evaluations);
    let r = recipe("fendl-3.2d", vec![f.source]);
    let err = extract(&r, &[f], &scratch.dir("lib")).unwrap_err();
    assert!(err.contains("not a zip archive"), "{err}");
}

/// Builds a decay tape of two materials and a neutron archive from the
/// parser's fixtures, as JEFF ships them, and names every material.
fn decay_tape_and_neutron_archive(scratch: &Scratch) -> (PathBuf, Vec<(Sublibrary, PathBuf)>) {
    let tape = scratch.0.join("Radioactive_Decay_Data_JEFF-40.txt");
    let cs = String::from_utf8(unxz(CS137)).unwrap();
    let u = String::from_utf8(unxz(U235)).unwrap();
    // One TPID, both materials, one TEND, as a tape of many materials has.
    let inner = |text: &str| {
        let lines: Vec<&str> = text.lines().collect();
        let tend = lines
            .iter()
            .rposition(|l| l.len() >= 70 && l[66..70].trim() == "-1")
            .unwrap();
        lines[1..tend]
            .iter()
            .map(|l| format!("{l}\n"))
            .collect::<String>()
    };
    let mut joined = cs.lines().next().unwrap().to_string() + "\n";
    joined += &inner(&cs);
    joined += &inner(&u);
    joined += &format!("{:<66}{:>4}{:>2}{:>3}\n", "", -1, 0, 0);
    std::fs::write(&tape, joined).unwrap();

    let zip = scratch.0.join("JEFF40-Evaluations-Neutron-593.zip");
    zip_file(
        &zip,
        &[
            ("n_26-Fe-056g.jeff", &unxz(FE56)),
            ("n_3-Li-006g.jeff", &unxz(LI6)),
        ],
    );
    let jeff = |n: &str| n.ends_with(".jeff");
    let got = vec![
        fetched(
            &zip,
            "JEFF40-Evaluations-Neutron-593.zip",
            Sublibrary::Neutron,
            jeff,
        ),
        fetched(
            &tape,
            "Radioactive_Decay_Data_JEFF-40.txt",
            Sublibrary::Decay,
            |_| true,
        ),
    ];
    let r = recipe("jeff-4.0", got.iter().map(|f| f.source).collect());
    let root = scratch.dir("lib");
    let folders = extract(&r, &got, &root).unwrap();
    (root, folders)
}

#[test]
fn every_material_is_named_from_its_header() {
    let scratch = Scratch::new("identify");
    let (_, folders) = decay_tape_and_neutron_archive(&scratch);
    let got: Vec<(Sublibrary, String)> = identify(&folders)
        .unwrap()
        .into_iter()
        .map(|m| (m.sublibrary, m.name))
        .collect();
    assert_eq!(
        got,
        vec![
            (Sublibrary::Neutron, "Fe56".to_string()),
            (Sublibrary::Neutron, "Li6".to_string()),
            (Sublibrary::Decay, "Cs137".to_string()),
            (Sublibrary::Decay, "U235".to_string()),
        ]
    );
}

#[test]
fn a_material_of_another_sublibrary_or_twice_is_reported_with_the_rest() {
    let scratch = Scratch::new("identify-wrong");
    let (_, folders) = decay_tape_and_neutron_archive(&scratch);
    let neutron = &folders
        .iter()
        .find(|(s, _)| *s == Sublibrary::Neutron)
        .unwrap()
        .1;
    std::fs::write(neutron.join("dec-055_Cs_137.endf"), unxz(CS137)).unwrap();
    std::fs::write(neutron.join("n-026_Fe_056.copy"), unxz(FE56)).unwrap();
    std::fs::write(neutron.join("broken.endf"), b"not an ENDF file\n").unwrap();
    let err = identify(&folders).unwrap_err();
    assert!(
        err.starts_with("3 problems in the unpacked sources"),
        "{err}"
    );
    assert!(
        err.contains("is NSUB 4, not the neutron sublibrary (NSUB 10)"),
        "{err}"
    );
    assert!(err.contains("Fe56 (neutron) is in both"), "{err}");
    assert!(err.contains("broken.endf"), "{err}");
}
