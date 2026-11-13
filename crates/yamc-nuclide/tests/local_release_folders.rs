//! A release folder on local disk serves every data setting from one path.
//!
//! The folder is laid out as a published release is, with its own
//! `manifest.json` at the top: cross sections under `neutron/` and `photon/`,
//! and the transmutation subsections under `transmutation/<subsection>.arrow/`.
//! These pin what the readers make of it: where each setting resolves, what is
//! refused and how, and what the run's provenance records.

use std::path::{Path, PathBuf};

use yamc_nuclide::storage::release::sha256_hex;
use yamc_nuclide::url_cache::{data_releases, resolve_path_or_url, resolve_subsection, DataKind};
use yamc_nuclide::LoadScope;

/// The files of a release folder that carries every part, one byte string
/// each: the resolvers check sizes and presence, and parse none of them.
const EVERY_PART: &[(&str, &[u8])] = &[
    ("neutron/Fe56.arrow/version.json", b"{}"),
    ("neutron/Fe56.arrow/nuclide.arrow", b"nuclide"),
    ("photon/Fe.arrow/version.json", b"{}"),
    ("transmutation/decay.arrow/nuclides.arrow", b"nuclides"),
    ("transmutation/decay.arrow/decay_modes.arrow", b"modes"),
    (
        "transmutation/reactions.arrow/reactions.arrow",
        b"reactions",
    ),
    (
        "transmutation/fission_yields.arrow/fission_yields.arrow",
        b"yields",
    ),
    (
        "transmutation/branching.arrow/branching.arrow",
        b"branching",
    ),
];

/// The transmutation files of a release that, as TENDL's, has no decay or
/// fission-yield data.
const REACTIONS_AND_BRANCHING: &[(&str, &[u8])] = &[
    ("neutron/Fe56.arrow/version.json", b"{}"),
    (
        "transmutation/reactions.arrow/reactions.arrow",
        b"reactions",
    ),
    (
        "transmutation/branching.arrow/branching.arrow",
        b"branching",
    ),
];

/// A fresh, empty folder for one test.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("yamc-local-release-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write `files` into `dir` and a manifest listing them, for release
/// `release` of `keyword` in `format_version`. Returns the manifest's bytes.
fn write_release(
    dir: &Path,
    keyword: &str,
    release: &str,
    format_version: u32,
    files: &[(&str, &[u8])],
) -> Vec<u8> {
    let entries: Vec<serde_json::Value> = files
        .iter()
        .map(|(path, bytes)| {
            let to = dir.join(path);
            std::fs::create_dir_all(to.parent().unwrap()).unwrap();
            std::fs::write(&to, bytes).unwrap();
            serde_json::json!({"path": path, "bytes": bytes.len(), "sha256": sha256_hex(bytes)})
        })
        .collect();
    let manifest = serde_json::to_vec(&serde_json::json!({
        "schema": "yamc-data-manifest",
        "schema_version": 1,
        "keyword": keyword,
        "release": release,
        "format_version": format_version,
        "converters": ["yamc-convert 0.1.0", "yani-convert 0.1.0"],
        "files": entries,
    }))
    .unwrap();
    std::fs::write(dir.join("manifest.json"), &manifest).unwrap();
    manifest
}

fn source(dir: &Path) -> String {
    dir.to_string_lossy().into_owned()
}

#[test]
fn every_transmutation_part_resolves_under_transmutation() {
    let dir = scratch("every-part");
    write_release(&dir, "endf-b8.1", "2026-10-23", 2, EVERY_PART);
    for subsection in ["decay", "reactions", "fission_yields", "branching"] {
        let resolved = resolve_subsection(&source(&dir), subsection).unwrap();
        assert_eq!(
            resolved,
            dir.join("transmutation")
                .join(format!("{subsection}.arrow")),
            "{subsection}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_cross_sections_resolve_under_neutron_and_photon() {
    let dir = scratch("particles");
    write_release(&dir, "endf-b8.1", "2026-10-23", 2, EVERY_PART);
    let scope = LoadScope::full();
    let neutron = resolve_path_or_url(&source(&dir), "Fe56", DataKind::Neutron, &scope).unwrap();
    assert_eq!(neutron, dir.join("neutron").join("Fe56.arrow"));
    let photon = resolve_path_or_url(&source(&dir), "Fe", DataKind::Photon, &scope).unwrap();
    assert_eq!(photon, dir.join("photon").join("Fe.arrow"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A part the release does not carry is refused, naming the part, the folder
/// and the parts it does carry, rather than read from the folder's top.
#[test]
fn a_part_the_folder_lacks_is_refused_by_name() {
    let dir = scratch("tendl");
    write_release(&dir, "tendl-2025", "2026-10-18", 2, REACTIONS_AND_BRANCHING);
    for subsection in ["decay", "fission_yields"] {
        let err = resolve_subsection(&source(&dir), subsection)
            .expect_err("TENDL publishes no such part")
            .to_string();
        assert!(
            err.contains(&format!(
                "does not provide a '{subsection}' transmutation subsection"
            )),
            "{err}"
        );
        assert!(err.contains("it provides branching, reactions"), "{err}");
        assert!(err.contains("'tendl-2025' release 2026-10-18"), "{err}");
        assert!(err.contains(&dir.display().to_string()), "{err}");
    }
    assert!(resolve_subsection(&source(&dir), "reactions").is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A file the manifest lists that the folder does not hold is an incomplete
/// copy, which would load as a subsection with less in it than the release.
#[test]
fn a_listed_file_the_folder_lacks_is_refused() {
    let dir = scratch("incomplete");
    write_release(&dir, "endf-b8.1", "2026-10-23", 2, EVERY_PART);
    std::fs::remove_file(dir.join("transmutation/decay.arrow/decay_modes.arrow")).unwrap();
    let err = resolve_subsection(&source(&dir), "decay")
        .expect_err("an incomplete copy")
        .to_string();
    assert!(
        err.contains("does not hold transmutation/decay.arrow/decay_modes.arrow"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_truncated_subsection_file_is_refused() {
    let dir = scratch("truncated");
    write_release(&dir, "endf-b8.1", "2026-10-23", 2, EVERY_PART);
    std::fs::write(
        dir.join("transmutation/reactions.arrow/reactions.arrow"),
        b"re",
    )
    .unwrap();
    let err = resolve_subsection(&source(&dir), "reactions")
        .expect_err("a truncated copy")
        .to_string();
    assert!(err.contains("size mismatch"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A folder in a format this build does not read is refused before anything
/// in it is read, whichever setting names it.
#[test]
fn a_folder_in_a_format_this_build_does_not_read_is_refused() {
    let dir = scratch("format");
    write_release(&dir, "endf-b8.1", "2027-01-01", 99, EVERY_PART);
    let err = resolve_subsection(&source(&dir), "decay")
        .expect_err("format 99")
        .to_string();
    assert!(err.contains("format_version 99"), "{err}");
    assert!(err.contains("manifest.json"), "{err}");
    let scope = LoadScope::full();
    let err = resolve_path_or_url(&source(&dir), "Fe56", DataKind::Neutron, &scope)
        .expect_err("format 99")
        .to_string();
    assert!(err.contains("format_version 99"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Each folder read is recorded in the provenance, keyed by its absolute
/// path, with the release its manifest names and that manifest's hash.
#[test]
fn a_folder_read_is_recorded_by_its_path() {
    let dir = scratch("provenance");
    let manifest = write_release(&dir, "jeff-4.0", "2026-10-20", 2, EVERY_PART);
    let key = source(&std::path::absolute(&dir).unwrap());
    assert!(!data_releases().contains_key(&key));

    resolve_subsection(&source(&dir), "decay").unwrap();
    let record = &data_releases()[&key];
    assert_eq!(record.release, "2026-10-20");
    assert_eq!(record.manifest_sha256, sha256_hex(&manifest));
    assert_eq!(record.format_version, 2);
    assert!(!record.offline);
    assert_eq!(
        record.converters,
        ["yamc-convert 0.1.0", "yani-convert 0.1.0"]
    );

    // A manifest rewritten in place (the folder rebuilt) is read again, and
    // the new release is the one recorded.
    let rebuilt = write_release(&dir, "jeff-4.0", "2026-10-21.1", 2, EVERY_PART);
    let scope = LoadScope::full();
    resolve_path_or_url(&source(&dir), "Fe56", DataKind::Neutron, &scope).unwrap();
    let record = &data_releases()[&key];
    assert_eq!(record.release, "2026-10-21.1");
    assert_eq!(record.manifest_sha256, sha256_hex(&rebuilt));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The `manifest.json` a converter writes at the top of a chain directory is
/// not a release manifest: the directory reads as it always has, and is not
/// recorded as a release.
#[test]
fn a_chain_directory_with_its_own_manifest_reads_as_before() {
    let dir = scratch("chain");
    std::fs::create_dir_all(dir.join("decay")).unwrap();
    std::fs::write(
        dir.join("manifest.json"),
        br#"{"format_version": 2, "library": "endf-b8.1", "subsections": {"decay": {"path": "decay"}}}"#,
    )
    .unwrap();
    assert_eq!(
        resolve_subsection(&source(&dir), "decay").unwrap(),
        dir.join("decay")
    );
    // A subsection the chain has no folder for is the path itself, as before.
    assert_eq!(resolve_subsection(&source(&dir), "reactions").unwrap(), dir);
    let key = source(&std::path::absolute(&dir).unwrap());
    assert!(!data_releases().contains_key(&key));
    let _ = std::fs::remove_dir_all(&dir);
}
