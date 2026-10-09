//! The published release layout of a nuclear-data library, and what a reader
//! checks about it.
//!
//! Every library keyword is published as immutable release folders plus one
//! mutable pointer:
//!
//! ```text
//! <origin>/<keyword>/latest.json                      the pointer (short TTL)
//! <origin>/<keyword>/<release>/manifest.json          every file, bytes and sha256
//! <origin>/<keyword>/<release>/neutron/<Name>.arrow/...
//! <origin>/<keyword>/<release>/photon/<Name>.arrow/...
//! <origin>/<keyword>/<release>/transmutation/<subsection>.arrow/...
//! ```
//!
//! A republish uploads a new release folder, then its manifest (so the
//! manifest's presence marks a complete release), then moves `latest.json`.
//! Nothing under a release prefix is ever rewritten, so a cached file is
//! current for as long as its release is the one in use, and no stamp has to
//! be compiled into a build to tell a stale cache from a current one.
//!
//! This module is the one place that knows the shape of `latest.json` and
//! `manifest.json`, and which `format_version`s this build reads. It does no
//! I/O, so the native downloader in [`super::url_cache`] and the browser
//! fetcher in `yamc::wasm` share it.

use std::collections::HashMap;
use std::ops::RangeInclusive;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The `format_version`s of published data this build reads.
///
/// A release whose `format_version` falls outside this range is refused before
/// any of its files are fetched. Widen it when the reader learns a new format;
/// a data fix that keeps the format needs no change here, and no yamc release.
///
/// Format 2 is the current one: the union energy grids live in `energy.arrow`
/// and `reactions.arrow` holds one record batch per (MT, temperature).
pub const SUPPORTED_FORMAT_VERSIONS: RangeInclusive<u32> = 2..=2;

/// The pointer object under `<origin>/<keyword>/`.
pub const LATEST: &str = "latest.json";

/// The manifest object under `<origin>/<keyword>/<release>/`, and its name in
/// a release folder on disk.
pub const MANIFEST: &str = "manifest.json";

/// `<origin>/<keyword>/latest.json`: which release of a library is current.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatestPointer {
    /// The release identifier, which is also the release folder's name.
    pub release: String,
    /// The data format of every file in the release.
    pub format_version: u32,
    /// Path of the release's manifest, relative to `<origin>/<keyword>/`.
    pub manifest: String,
    /// Hex sha256 of the manifest's bytes.
    pub manifest_sha256: String,
    /// Length of the manifest in bytes.
    pub manifest_bytes: u64,
}

/// One file of a release, as its manifest lists it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFile {
    /// Path relative to the release folder, `/`-separated.
    pub path: String,
    /// Length in bytes.
    pub bytes: u64,
    /// Hex sha256 of the contents.
    pub sha256: String,
}

/// `<origin>/<keyword>/<release>/manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Identifier of the manifest schema.
    #[serde(default)]
    pub schema: Option<String>,
    pub keyword: String,
    pub release: String,
    pub format_version: u32,
    /// Version of the converter that wrote the release.
    #[serde(default)]
    pub converter_version: Option<String>,
    /// When the release was built.
    #[serde(default)]
    pub created: Option<String>,
    pub files: Vec<ManifestFile>,
}

/// The release of one library a run used, as its results record it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataRelease {
    /// The release identifier `latest.json` named (or the cached release used
    /// when the origin was unreachable).
    pub release: String,
    /// Hex sha256 of that release's manifest, which pins every file in it.
    pub manifest_sha256: String,
    /// The data format of the release.
    pub format_version: u32,
    /// Whether the origin was unreachable, so the release is the newest
    /// complete one in the cache rather than the one the origin names now.
    pub offline: bool,
}

/// A parsed manifest with its files indexed by path.
#[derive(Debug, Clone)]
pub struct Release {
    pub manifest: Manifest,
    /// Hex sha256 of the manifest bytes this was parsed from, recorded as the
    /// release's identity in results.
    pub manifest_sha256: String,
    by_path: HashMap<String, usize>,
}

impl Release {
    /// The manifest entry for `path` (relative to the release folder).
    pub fn file(&self, path: &str) -> Option<&ManifestFile> {
        self.by_path.get(path).map(|&i| &self.manifest.files[i])
    }

    /// The names published directly under `<dir>/` as `<Name>.arrow/`
    /// folders, sorted: the nuclides under `neutron`, the elements under
    /// `photon`, the subsections under `transmutation`. This is the runtime
    /// index of what the release carries.
    pub fn names_under(&self, dir: &str) -> Vec<String> {
        let prefix = format!("{}/", dir.trim_end_matches('/'));
        let mut names: Vec<String> = self
            .manifest
            .files
            .iter()
            .filter_map(|f| f.path.strip_prefix(&prefix))
            .filter_map(|rest| rest.split_once('/').map(|(first, _)| first))
            .filter_map(|first| first.strip_suffix(".arrow"))
            .map(str::to_string)
            .collect();
        names.sort_unstable();
        names.dedup();
        names
    }

    /// Whether any file is published under the folder `dir`.
    pub fn has_dir(&self, dir: &str) -> bool {
        let prefix = format!("{}/", dir.trim_end_matches('/'));
        self.manifest
            .files
            .iter()
            .any(|f| f.path.starts_with(&prefix))
    }
}

/// Hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// A sha256 computed over a stream, for files too large to hold whole.
#[derive(Default)]
pub struct StreamHash {
    hasher: Sha256,
    bytes: u64,
}

impl StreamHash {
    pub fn update(&mut self, chunk: &[u8]) {
        self.hasher.update(chunk);
        self.bytes += chunk.len() as u64;
    }

    /// `(bytes seen, hex sha256)`.
    pub fn finish(self) -> (u64, String) {
        (self.bytes, hex(&self.hasher.finalize()))
    }
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Check what was received against what was published. The error names the
/// file, and what was expected and got, so a corrupted or truncated object is
/// reported as exactly that rather than as a parse failure further on.
pub fn verify(what: &str, expected: &ManifestFile, bytes: u64, sha256: &str) -> Result<(), String> {
    if bytes != expected.bytes {
        return Err(format!(
            "{what}: size mismatch, the release manifest says {} bytes and {bytes} were received",
            expected.bytes
        ));
    }
    if !sha256.eq_ignore_ascii_case(&expected.sha256) {
        return Err(format!(
            "{what}: sha256 mismatch, the release manifest says {} and the received bytes hash \
             to {sha256}",
            expected.sha256
        ));
    }
    Ok(())
}

/// A path segment that is safe to use as a directory name: a release
/// identifier names a cache folder, so `..` or a separator in it would write
/// somewhere else.
fn safe_segment(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && !s.contains(['/', '\\', ':'])
        && !s.chars().any(char::is_control)
}

/// A manifest path that stays inside the release folder.
fn safe_relative_path(p: &str) -> bool {
    !p.starts_with('/') && p.split('/').all(safe_segment)
}

/// Refuse a release whose format this build cannot read, saying what to do.
pub fn check_format_version(
    keyword: &str,
    release: &str,
    format_version: u32,
) -> Result<(), String> {
    if SUPPORTED_FORMAT_VERSIONS.contains(&format_version) {
        return Ok(());
    }
    let (lo, hi) = (
        SUPPORTED_FORMAT_VERSIONS.start(),
        SUPPORTED_FORMAT_VERSIONS.end(),
    );
    let reads = if lo == hi {
        format!("{lo}")
    } else {
        format!("{lo} to {hi}")
    };
    let advice = if format_version > *hi {
        "Upgrade yamc-core / yani-core to a release that reads it"
    } else {
        "This build no longer reads that format; use an older yamc release"
    };
    Err(format!(
        "'{keyword}' release {release} is published in data format_version {format_version}, \
         and this build of yamc reads format_version {reads}. {advice}, or point the data \
         source at a local directory holding a library this build reads."
    ))
}

/// Parse `latest.json` and check its format and fields.
pub fn parse_latest(keyword: &str, bytes: &[u8]) -> Result<LatestPointer, String> {
    let latest: LatestPointer = serde_json::from_slice(bytes)
        .map_err(|e| format!("'{keyword}' {LATEST} is not a valid release pointer: {e}"))?;
    if !safe_segment(&latest.release) {
        return Err(format!(
            "'{keyword}' {LATEST} names release {:?}, which is not a valid release identifier",
            latest.release
        ));
    }
    if !safe_relative_path(&latest.manifest) {
        return Err(format!(
            "'{keyword}' {LATEST} names manifest path {:?}, which is not a relative path",
            latest.manifest
        ));
    }
    check_format_version(keyword, &latest.release, latest.format_version)?;
    Ok(latest)
}

/// Parse a manifest, checking it against the pointer that named it when there
/// is one (size, sha256, keyword, release and format all have to agree), and
/// that every path it lists stays inside the release folder.
///
/// `pointer` is `None` for a manifest read back from a cached release folder
/// with no pointer to compare to (the origin is unreachable); its hash is then
/// computed and recorded rather than compared.
pub fn parse_manifest(
    keyword: &str,
    bytes: &[u8],
    pointer: Option<&LatestPointer>,
) -> Result<Release, String> {
    let manifest_sha256 = sha256_hex(bytes);
    if let Some(p) = pointer {
        let expected = ManifestFile {
            path: p.manifest.clone(),
            bytes: p.manifest_bytes,
            sha256: p.manifest_sha256.clone(),
        };
        verify(
            &format!("'{keyword}' manifest {}", p.manifest),
            &expected,
            bytes.len() as u64,
            &manifest_sha256,
        )?;
    }
    let manifest: Manifest = serde_json::from_slice(bytes)
        .map_err(|e| format!("'{keyword}' {MANIFEST} is not a valid release manifest: {e}"))?;
    if manifest.keyword != keyword {
        return Err(format!(
            "the {MANIFEST} fetched for '{keyword}' describes '{}'",
            manifest.keyword
        ));
    }
    if !safe_segment(&manifest.release) {
        return Err(format!(
            "'{keyword}' {MANIFEST} names release {:?}, which is not a valid release identifier",
            manifest.release
        ));
    }
    if let Some(p) = pointer {
        if manifest.release != p.release || manifest.format_version != p.format_version {
            return Err(format!(
                "'{keyword}' {LATEST} names release {} (format_version {}) but its manifest \
                 describes release {} (format_version {})",
                p.release, p.format_version, manifest.release, manifest.format_version
            ));
        }
    }
    check_format_version(keyword, &manifest.release, manifest.format_version)?;
    let mut by_path = HashMap::with_capacity(manifest.files.len());
    for (i, file) in manifest.files.iter().enumerate() {
        if !safe_relative_path(&file.path) {
            return Err(format!(
                "'{keyword}' release {} lists {:?}, which is not a path inside the release",
                manifest.release, file.path
            ));
        }
        by_path.insert(file.path.clone(), i);
    }
    Ok(Release {
        manifest,
        manifest_sha256,
        by_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_json(keyword: &str, release: &str, format_version: u32) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema": "yamc-release-manifest/1",
            "keyword": keyword,
            "release": release,
            "format_version": format_version,
            "converter_version": "0.15.0",
            "created": "2026-10-01T00:00:00Z",
            "files": [
                {"path": "neutron/Fe56.arrow/version.json", "bytes": 3, "sha256": sha256_hex(b"abc")},
                {"path": "neutron/Li6.arrow/version.json", "bytes": 3, "sha256": sha256_hex(b"abc")},
                {"path": "photon/Fe.arrow/element.arrow", "bytes": 3, "sha256": sha256_hex(b"abc")},
                {"path": "transmutation/decay.arrow/nuclides.arrow", "bytes": 3, "sha256": sha256_hex(b"abc")},
            ],
        }))
        .unwrap()
    }

    fn pointer_for(release: &str, bytes: &[u8]) -> LatestPointer {
        LatestPointer {
            release: release.into(),
            format_version: 2,
            manifest: format!("{release}/manifest.json"),
            manifest_sha256: sha256_hex(bytes),
            manifest_bytes: bytes.len() as u64,
        }
    }

    #[test]
    fn sha256_of_a_known_string() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut s = StreamHash::default();
        s.update(b"a");
        s.update(b"bc");
        assert_eq!(s.finish(), (3, sha256_hex(b"abc")));
    }

    #[test]
    fn a_manifest_matching_its_pointer_parses_and_indexes() {
        let bytes = manifest_json("endf-b8.1", "2026-10-01", 2);
        let release = parse_manifest(
            "endf-b8.1",
            &bytes,
            Some(&pointer_for("2026-10-01", &bytes)),
        )
        .unwrap();
        assert_eq!(release.manifest_sha256, sha256_hex(&bytes));
        assert!(release.file("neutron/Fe56.arrow/version.json").is_some());
        assert!(release.file("neutron/Fe57.arrow/version.json").is_none());
        assert_eq!(release.names_under("neutron"), ["Fe56", "Li6"]);
        assert_eq!(release.names_under("photon"), ["Fe"]);
        assert_eq!(release.names_under("transmutation"), ["decay"]);
        assert!(release.has_dir("transmutation/decay.arrow"));
        assert!(!release.has_dir("transmutation/branching.arrow"));
    }

    #[test]
    fn a_manifest_whose_hash_differs_from_the_pointer_is_refused() {
        let bytes = manifest_json("endf-b8.1", "2026-10-01", 2);
        let mut pointer = pointer_for("2026-10-01", &bytes);
        pointer.manifest_sha256 = sha256_hex(b"something else");
        let err = parse_manifest("endf-b8.1", &bytes, Some(&pointer)).unwrap_err();
        assert!(err.contains("sha256 mismatch"), "{err}");
        assert!(err.contains("manifest"), "{err}");
    }

    #[test]
    fn a_manifest_for_another_release_is_refused() {
        let bytes = manifest_json("endf-b8.1", "2026-09-18", 2);
        let err = parse_manifest(
            "endf-b8.1",
            &bytes,
            Some(&pointer_for("2026-10-01", &bytes)),
        )
        .unwrap_err();
        assert!(err.contains("2026-09-18"), "{err}");
    }

    #[test]
    fn a_format_outside_the_range_is_refused_with_advice() {
        let err = check_format_version("endf-b8.1", "2027-01-01", 99).unwrap_err();
        assert!(err.contains("format_version 99"), "{err}");
        assert!(err.contains("Upgrade"), "{err}");
        assert!(err.contains("local directory"), "{err}");
        let pointer = serde_json::json!({
            "release": "2026-10-01", "format_version": 1,
            "manifest": "2026-10-01/manifest.json", "manifest_sha256": "00", "manifest_bytes": 1
        });
        let err = parse_latest("endf-b8.1", pointer.to_string().as_bytes()).unwrap_err();
        assert!(err.contains("no longer reads"), "{err}");
    }

    #[test]
    fn paths_that_escape_the_release_are_refused() {
        for bad in ["../x", "/etc", "a/../b", "a\\b"] {
            assert!(!safe_relative_path(bad), "{bad}");
        }
        assert!(safe_relative_path("neutron/Fe56.arrow/version.json"));
        let pointer = serde_json::json!({
            "release": "../../home", "format_version": 2,
            "manifest": "x/manifest.json", "manifest_sha256": "00", "manifest_bytes": 1
        });
        assert!(parse_latest("endf-b8.1", pointer.to_string().as_bytes()).is_err());
    }

    #[test]
    fn verify_names_the_file_and_both_values() {
        let expected = ManifestFile {
            path: "p".into(),
            bytes: 3,
            sha256: sha256_hex(b"abc"),
        };
        assert!(verify("p", &expected, 3, &sha256_hex(b"abc")).is_ok());
        let short = verify("neutron/Fe56.arrow/reactions.arrow", &expected, 2, "x").unwrap_err();
        assert!(
            short.contains("reactions.arrow")
                && short.contains("3 bytes")
                && short.contains("2 were")
        );
        let bad = verify("p", &expected, 3, &sha256_hex(b"abd")).unwrap_err();
        assert!(bad.contains(&expected.sha256) && bad.contains(&sha256_hex(b"abd")));
    }
}
