//! The published release layout of a nuclear-data library, and what a reader
//! checks about it.
//!
//! Every library keyword is published as immutable release folders plus one
//! mutable pointer:
//!
//! ```text
//! <origin>/<keyword>/latest.json                      the pointer
//! <origin>/<keyword>/<release>/manifest.json          every file, bytes and sha256
//! <origin>/<keyword>/<release>/neutron/<Name>.arrow/...
//! <origin>/<keyword>/<release>/photon/<Name>.arrow/...
//! <origin>/<keyword>/<release>/photonuclear/<Name>.arrow/...
//! <origin>/<keyword>/<release>/<particle>/index.json
//! <origin>/<keyword>/<release>/transmutation/<subsection>.arrow/...
//! ```
//!
//! A republish uploads a new release folder, then its manifest (so the
//! manifest's presence marks a complete release), then moves `latest.json`.
//! Nothing under a release prefix is ever rewritten, so a cached file is
//! current for as long as its release is the one in use, and no stamp has to
//! be compiled into a build to tell a stale cache from a current one. The
//! pointer is the one object that changes, so every reader asks for it with a
//! cache-busting query ([`pointer_url`]) and everything else plainly.
//!
//! This module is the one place that knows the shape of `latest.json` and
//! `manifest.json`, and which `format_version`s this build reads. Parsing and
//! checking them does no I/O, so the native downloader in [`super::url_cache`]
//! and the browser fetcher in `yamc::wasm` share it. The rest works on a
//! release folder on disk: [`write_manifest`] describes one, so a library
//! built locally is described exactly as a published release is, and
//! [`verify_folder`] checks one against its manifest.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{ErrorKind, Read, Write};
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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
///
/// The range ends at [`nuclear_data_schema::FORMAT_VERSION`], the format the
/// converters write, and the assertion below holds it there: a build reads
/// what it writes and nothing newer. How far back it reaches is a separate
/// decision. The bounds stay literals because the data-format check
/// (`scripts/check_format_version_bumps.py`) reads them from this source.
pub const SUPPORTED_FORMAT_VERSIONS: RangeInclusive<u32> = 2..=2;

const _: () = assert!(
    *SUPPORTED_FORMAT_VERSIONS.start() <= nuclear_data_schema::FORMAT_VERSION
        && *SUPPORTED_FORMAT_VERSIONS.end() == nuclear_data_schema::FORMAT_VERSION,
    "SUPPORTED_FORMAT_VERSIONS must end at nuclear_data_schema::FORMAT_VERSION"
);

/// The pointer object under `<origin>/<keyword>/`.
pub const LATEST: &str = "latest.json";

/// The manifest object under `<origin>/<keyword>/<release>/`, and its name in
/// a release folder on disk.
pub const MANIFEST: &str = "manifest.json";

/// The `schema` of the manifest shape [`write_manifest`] writes, which the
/// publishing scripts write too (`release_manifest.py` in
/// nuclear_data_generation_scripts).
pub const MANIFEST_SCHEMA: &str = "yamc-data-manifest";

/// The `schema_version` of that shape.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// The particle folders a release can carry, in the order a manifest lists
/// them under `particles`.
pub const PARTICLES: [&str; 3] = ["neutron", "photon", "photonuclear"];

/// The folder of a release that holds the transmutation chain's subsections.
const TRANSMUTATION: &str = "transmutation";

/// The query parameter every request for `latest.json` carries, with a random
/// value, so the CDN in front of the origin cannot answer it from its cache.
///
/// `latest.json` is the one object of a library that changes, and the zone's
/// cache settings, not the `Cache-Control` it is uploaded with, decide how long
/// the edge keeps a copy. The edge once served a pointer 5.8 days old, naming a
/// release that had since been deleted, so every download from an empty cache
/// failed with a 404. The edge keys its cache on the query string as well as
/// the path, so a value it has not seen goes through to the origin, which
/// answers with the pointer as it is now. The manifest and the release files
/// are asked for without one: they never change, and the edge caching them is
/// what the CDN is for.
pub const CACHE_BUST_PARAM: &str = "release-check";

/// The URL of the `latest.json` of the library at `library_url`
/// (`<origin>/<keyword>`), with a [`CACHE_BUST_PARAM`] value of 64 random
/// bits, so in practice no two requests, from one process or from many, carry
/// the same one.
pub fn pointer_url(library_url: &str) -> String {
    format!(
        "{}/{LATEST}?{CACHE_BUST_PARAM}={:016x}",
        library_url.trim_end_matches('/'),
        rand::random::<u64>()
    )
}

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

/// `<origin>/<keyword>/<release>/manifest.json`, the [`MANIFEST_SCHEMA`] v1
/// shape. Every field but `keyword`, `release`, `format_version` and `files`
/// may be absent when reading.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Identifier of the manifest schema, [`MANIFEST_SCHEMA`].
    #[serde(default)]
    pub schema: Option<String>,
    /// Version of that schema, [`MANIFEST_SCHEMA_VERSION`].
    #[serde(default)]
    pub schema_version: Option<u32>,
    pub keyword: String,
    pub release: String,
    pub format_version: u32,
    /// Every converter that wrote the release's files, with its version
    /// (`yamc-convert 0.1.0`, `yani-convert 0.1.0`): the publishing scripts
    /// list each distinct `converter_version` stamp in the tree, and the cross
    /// sections and the transmutation chain are written by different
    /// converters. Empty when the manifest lists none.
    #[serde(default)]
    pub converter_versions: Vec<String>,
    /// When the release was built.
    #[serde(default)]
    pub created: Option<String>,
    /// What built the release, free-form: the publishing scripts record their
    /// commit as `scripts_commit`.
    #[serde(default)]
    pub provenance: serde_json::Map<String, serde_json::Value>,
    /// The particle folders the release carries, in [`PARTICLES`] order.
    #[serde(default)]
    pub particles: Vec<String>,
    /// How many files `files` lists.
    #[serde(default)]
    pub file_count: Option<u64>,
    /// The sum of their sizes in bytes.
    #[serde(default)]
    pub total_bytes: Option<u64>,
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
    /// Every converter that wrote the release's files, with its version, as
    /// the release's manifest lists them (`yamc-convert 0.1.0`,
    /// `yani-convert 0.1.0`). Empty when the manifest lists none, and when
    /// reading results written before releases recorded them.
    #[serde(default)]
    pub converter_versions: Vec<String>,
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

/// Refuse a release identifier that cannot name a release. It is the release
/// folder's name under `<origin>/<keyword>/`, beside `latest.json`, so it is
/// one URL path segment (letters, digits, `.`, `_` and `-`) and starts with a
/// digit, which keeps it apart from any other name there. Releases are dated,
/// `2026-10-12.1`. The same rule as the publishing scripts' `RELEASE_RE`.
pub fn check_release_id(release: &str) -> Result<(), String> {
    let mut chars = release.chars();
    let valid = chars.next().is_some_and(|c| c.is_ascii_digit())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if valid {
        Ok(())
    } else {
        Err(format!(
            "release {release:?} is not a valid release identifier: it must start with a digit \
             and contain only letters, digits, '.', '_' and '-', such as 2026-10-12 or 2026-10-12.1"
        ))
    }
}

/// Check the files of `entry` (a folder of the release, such as
/// `neutron/Fe56.arrow`) that are present under `dir` against the sizes
/// `manifest` lists. A listed file that is absent is not an error here: a
/// release folder may hold only what was downloaded.
pub fn check_sizes(dir: &std::path::Path, manifest: &Manifest, entry: &str) -> Result<(), String> {
    let prefix = format!("{entry}/");
    for file in manifest
        .files
        .iter()
        .filter(|f| f.path.starts_with(&prefix))
    {
        if !safe_relative_path(&file.path) {
            continue;
        }
        let path = dir.join(&file.path);
        if let Ok(meta) = std::fs::metadata(&path)
            && meta.len() != file.bytes
        {
            return Err(format!(
                "{}: size mismatch, the release manifest says {} bytes and the file has {}. \
                     The local copy is incomplete or modified; download or build it again.",
                path.display(),
                file.bytes,
                meta.len()
            ));
        }
    }
    Ok(())
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
    let advice = if format_version > *SUPPORTED_FORMAT_VERSIONS.end() {
        "Upgrade yamc-core / yani-core to a release that reads it"
    } else {
        "This build no longer reads that format; use an older yamc release"
    };
    Err(format!(
        "'{keyword}' release {release} is published in data format_version {format_version}, \
         and this build of yamc reads format_version {}. {advice}, or point the data \
         source at a local directory holding a library this build reads.",
        readable_formats()
    ))
}

/// The [`SUPPORTED_FORMAT_VERSIONS`] as a message says them: `2`, or `2 to 3`.
fn readable_formats() -> String {
    let (lo, hi) = (
        SUPPORTED_FORMAT_VERSIONS.start(),
        SUPPORTED_FORMAT_VERSIONS.end(),
    );
    if lo == hi {
        format!("{lo}")
    } else {
        format!("{lo} to {hi}")
    }
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
    checked(manifest, manifest_sha256, pointer)
}

/// What every parsed manifest is checked for (a valid release identifier,
/// agreement with the pointer that named it, a format this build reads, paths
/// that stay inside the release), with its files indexed by path.
fn checked(
    manifest: Manifest,
    manifest_sha256: String,
    pointer: Option<&LatestPointer>,
) -> Result<Release, String> {
    let keyword = &manifest.keyword;
    if !safe_segment(&manifest.release) {
        return Err(format!(
            "'{keyword}' {MANIFEST} names release {:?}, which is not a valid release identifier",
            manifest.release
        ));
    }
    if let Some(p) = pointer
        && (manifest.release != p.release || manifest.format_version != p.format_version)
    {
        return Err(format!(
            "'{keyword}' {LATEST} names release {} (format_version {}) but its manifest \
                 describes release {} (format_version {})",
            p.release, p.format_version, manifest.release, manifest.format_version
        ));
    }
    index(manifest, manifest_sha256)
}

/// Parse the `manifest.json` at `path`, the top of a release folder on local
/// disk (a library downloaded or built whole), or `None` when it is not a
/// release manifest.
///
/// A release manifest lists the release's `files`. The `manifest.json` the
/// converters write at the top of a chain directory lists its `subsections`
/// instead, and a directory holding one is not a release folder. With no
/// pointer to check against, the keyword is the one the manifest names and its
/// hash is computed rather than compared. Its format must be one this build
/// reads, and its paths must stay inside the folder, as for a published
/// release.
pub fn parse_local_manifest(
    path: &std::path::Path,
    bytes: &[u8],
) -> Result<Option<Release>, String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| format!("{}: not valid JSON: {e}", path.display()))?;
    if value.get("files").is_none() {
        return Ok(None);
    }
    let manifest: Manifest = serde_json::from_value(value)
        .map_err(|e| format!("{}: not a valid release manifest: {e}", path.display()))?;
    if !safe_segment(&manifest.release) {
        return Err(format!(
            "{}: names release {:?}, which is not a valid release identifier",
            path.display(),
            manifest.release
        ));
    }
    index(manifest, sha256_hex(bytes))
        .map(Some)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Check that this build reads `manifest`'s format and that every path it
/// lists stays inside the release, and index its files by path.
fn index(manifest: Manifest, manifest_sha256: String) -> Result<Release, String> {
    let keyword = &manifest.keyword;
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

/// Write `<dir>/manifest.json` for the release folder `dir`, and return the
/// release it describes.
///
/// It lists every file the release publishes, with its size and sha256: each
/// file in the `<Name>.arrow/` folders under the [`PARTICLES`] folders and
/// `transmutation/`, and a particle's `index.json`. Nothing else in `dir` is
/// part of the release, so the manifest itself, and the `.build/` folder a
/// local build keeps its sources and scratch in, are not listed.
///
/// The bytes are those `build_manifest` and `write_manifest` in the publishing
/// scripts' `release_manifest.py` write for the same files at the same
/// `created` with the same `provenance`: files sorted by path, keys sorted, no
/// whitespace, one trailing newline. `particles` names the particle folders
/// listed, and `converter_versions` every distinct `converter_version` the
/// `version.json` and `provenance.json` files carry. `format_version` is the
/// caller's to state, since a chain subsection carries no stamp of its own
/// (the chain records its format in its own manifest, which is not part of
/// the release), and every `format_version` a file carries has to agree with
/// it: a release holds one format.
///
/// Written to a temporary file and renamed into place, so a reader sees the
/// old manifest or the new one, never part of either. Refused with nothing
/// written: a keyword or release identifier that cannot name a release, a
/// format this build does not read, a folder with nothing to publish, a file
/// stamped with another format, or a path the reader would refuse.
pub fn write_manifest(
    dir: &Path,
    keyword: &str,
    release: &str,
    format_version: u32,
    created: SystemTime,
    provenance: serde_json::Map<String, serde_json::Value>,
) -> Result<Release, String> {
    if !safe_segment(keyword) {
        return Err(format!(
            "keyword {keyword:?} cannot name a library: it is the library's folder name, \
             such as endf-b8.1"
        ));
    }
    check_release_id(release)?;
    if !SUPPORTED_FORMAT_VERSIONS.contains(&format_version) {
        return Err(format!(
            "format_version {format_version} is not one this build reads (it reads {}), \
             so it writes no release in it",
            readable_formats()
        ));
    }
    let created = utc_timestamp(created)?;
    let files = release_files(dir)?;
    if files.is_empty() {
        return Err(format!(
            "{}: nothing to publish, no <Name>.arrow/ folder under {}/ or {TRANSMUTATION}/",
            dir.display(),
            PARTICLES.join("/, ")
        ));
    }
    if let Some(path) = files.keys().find(|path| !safe_relative_path(path)) {
        return Err(format!(
            "{}: a release cannot publish this path, since a folder or file name in it \
             holds '\\', ':' or a control character",
            dir.join(path).display()
        ));
    }
    let converter_versions = converter_versions(&files, format_version)?;
    let locals: Vec<&PathBuf> = files.values().collect();
    let mut entries = Vec::with_capacity(files.len());
    for ((path, local), hashed) in files.iter().zip(hash_files(&locals)) {
        let (bytes, sha256) =
            hashed.map_err(|e| format!("could not read {}: {e}", local.display()))?;
        entries.push(ManifestFile {
            path: path.clone(),
            bytes,
            sha256,
        });
    }
    let present: BTreeSet<&str> = files
        .keys()
        .filter_map(|path| path.split_once('/').map(|(top, _)| top))
        .collect();
    let manifest = Manifest {
        schema: Some(MANIFEST_SCHEMA.to_string()),
        schema_version: Some(MANIFEST_SCHEMA_VERSION),
        keyword: keyword.to_string(),
        release: release.to_string(),
        format_version,
        converter_versions,
        created: Some(created),
        provenance,
        particles: PARTICLES
            .into_iter()
            .filter(|p| present.contains(p))
            .map(str::to_string)
            .collect(),
        file_count: Some(entries.len() as u64),
        total_bytes: Some(entries.iter().map(|f| f.bytes).sum()),
        files: entries,
    };
    let bytes = manifest_bytes(&manifest);
    let described = checked(manifest, sha256_hex(&bytes), None)?;
    let path = dir.join(MANIFEST);
    write_atomically(&path, &bytes)
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(described)
}

/// What [`verify_folder`] found.
#[derive(Debug, Clone)]
pub struct Verification {
    /// The folder's manifest.
    pub release: Release,
    /// Files the manifest lists that the folder does not hold.
    pub missing: Vec<String>,
    /// Files the manifest lists whose size or sha256 differs from it.
    pub changed: Vec<String>,
    /// Files the folder publishes (see [`write_manifest`]) that the manifest
    /// does not list.
    pub unlisted: Vec<String>,
}

impl Verification {
    /// Whether the folder holds exactly the files its manifest lists, each as
    /// listed.
    pub fn ok(&self) -> bool {
        self.missing.is_empty() && self.changed.is_empty() && self.unlisted.is_empty()
    }
}

/// Check the release folder `dir` against its `manifest.json`: re-hash every
/// file the manifest lists, and look for files the folder publishes that it
/// does not list. All paths are relative to `dir`, sorted.
///
/// Loading data from a folder checks only the sizes of the files it reads
/// ([`check_sizes`]); this is the full check. Missing, changed and unlisted
/// files are reported rather than raised, so a caller sees all of them at
/// once. An unlisted file fails the check like the other two: results record
/// the manifest's sha256 as the identity of the library a run read, and a
/// file the manifest does not list is data that identity does not cover.
/// What the folder does not publish, `.build/` among it, is not looked at.
///
/// An error when there is nothing to check against: `dir` is not a folder,
/// holds no manifest, or holds one that is not the [`MANIFEST_SCHEMA`] v1
/// shape or is in a format this build does not read. A listed file that
/// cannot be read is an error too.
pub fn verify_folder(dir: &Path) -> Result<Verification, String> {
    let release = read_manifest(dir)?;
    let listed = &release.manifest.files;
    let locals: Vec<PathBuf> = listed.iter().map(|f| dir.join(&f.path)).collect();
    let (mut missing, mut changed) = (Vec::new(), Vec::new());
    for ((file, local), hashed) in listed.iter().zip(&locals).zip(hash_files(&locals)) {
        match hashed {
            Ok((bytes, sha256)) => {
                if bytes != file.bytes || !sha256.eq_ignore_ascii_case(&file.sha256) {
                    changed.push(file.path.clone());
                }
            }
            // Absent, or a folder where the file should be (which Unix opens
            // and fails to read, and Windows refuses to open).
            Err(e)
                if matches!(e.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory)
                    || local.is_dir() =>
            {
                missing.push(file.path.clone());
            }
            Err(e) => return Err(format!("could not read {}: {e}", local.display())),
        }
    }
    missing.sort_unstable();
    changed.sort_unstable();
    let unlisted = release_files(dir)?
        .into_keys()
        .filter(|path| release.file(path).is_none())
        .collect();
    Ok(Verification {
        release,
        missing,
        changed,
        unlisted,
    })
}

/// The manifest of the release folder `dir`, whatever library it describes,
/// checked as [`parse_manifest`] checks one with no pointer to compare it
/// with, and refused unless it is the shape [`write_manifest`] writes.
fn read_manifest(dir: &Path) -> Result<Release, String> {
    if !dir.is_dir() {
        return Err(format!("{} is not a folder", dir.display()));
    }
    let path = dir.join(MANIFEST);
    let bytes = std::fs::read(&path).map_err(|e| match e.kind() {
        ErrorKind::NotFound => format!(
            "{} holds no {MANIFEST}, so there is nothing to check it against: a library \
             folder keeps the manifest it was built or published with",
            dir.display()
        ),
        _ => format!("could not read {}: {e}", path.display()),
    })?;
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| format!("{}: not a release manifest: {e}", path.display()))?;
    if manifest.schema.as_deref() != Some(MANIFEST_SCHEMA)
        || manifest.schema_version != Some(MANIFEST_SCHEMA_VERSION)
    {
        return Err(format!(
            "{}: not a {MANIFEST_SCHEMA} v{MANIFEST_SCHEMA_VERSION} manifest (schema {:?}, \
             schema_version {:?})",
            path.display(),
            manifest.schema,
            manifest.schema_version
        ));
    }
    checked(manifest, sha256_hex(&bytes), None).map_err(|e| format!("{}: {e}", path.display()))
}

/// The files the release folder `dir` publishes, by path relative to it: the
/// rule `release_manifest.py` publishes by, applied to the release layout.
fn release_files(dir: &Path) -> Result<BTreeMap<String, PathBuf>, String> {
    let mut files = BTreeMap::new();
    for top in PARTICLES.into_iter().chain([TRANSMUTATION]) {
        let parent = dir.join(top);
        if !parent.is_dir() {
            continue;
        }
        let mut any = false;
        for (name, path) in entries(&parent)? {
            if name.ends_with(".arrow") && path.is_dir() {
                any = true;
                collect_files(&path, &format!("{top}/{name}"), &mut files)?;
            }
        }
        let index = parent.join("index.json");
        if any && top != TRANSMUTATION && index.is_file() {
            files.insert(format!("{top}/index.json"), index);
        }
    }
    Ok(files)
}

/// Every file under the folder `dir`, at depth, keyed `<rel>/<path below dir>`.
fn collect_files(
    dir: &Path,
    rel: &str,
    files: &mut BTreeMap<String, PathBuf>,
) -> Result<(), String> {
    for (name, path) in entries(dir)? {
        let rel = format!("{rel}/{name}");
        if path.is_dir() {
            collect_files(&path, &rel, files)?;
        } else if path.is_file() {
            files.insert(rel, path);
        } else {
            return Err(format!("{} is neither a file nor a folder", path.display()));
        }
    }
    Ok(())
}

/// `(name, path)` of each entry of the folder `dir`.
fn entries(dir: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let listing = |e: std::io::Error| format!("could not list {}: {e}", dir.display());
    std::fs::read_dir(dir)
        .map_err(listing)?
        .map(|entry| {
            let path = entry.map_err(listing)?.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| format!("{}: a release's file names are UTF-8", path.display()))?
                .to_string();
            Ok((name, path))
        })
        .collect()
}

/// The distinct `converter_version`s of the `version.json` and
/// `provenance.json` among `files`, sorted, refusing a `format_version` stamp
/// other than `format_version`.
fn converter_versions(
    files: &BTreeMap<String, PathBuf>,
    format_version: u32,
) -> Result<Vec<String>, String> {
    let mut converters = BTreeSet::new();
    for (path, local) in files {
        if !(path.ends_with("/version.json") || path.ends_with("/provenance.json")) {
            continue;
        }
        let bytes =
            std::fs::read(local).map_err(|e| format!("could not read {}: {e}", local.display()))?;
        let stamp: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("{}: not valid JSON: {e}", local.display()))?;
        if let Some(stamped) = stamp.get("format_version")
            && stamped.as_u64() != Some(u64::from(format_version))
        {
            return Err(format!(
                "{} says format_version {stamped}, and the release is format_version \
                 {format_version}: a release holds one format, so convert that folder again",
                local.display()
            ));
        }
        if let Some(converter) = stamp.get("converter_version").and_then(|v| v.as_str())
            && !converter.is_empty()
        {
            converters.insert(converter.to_string());
        }
    }
    Ok(converters.into_iter().collect())
}

/// `(bytes, hex sha256)` of each of `paths`, in order, several files at a
/// time: a library is thousands of files and gigabytes, and one reader leaves
/// most of a disk's bandwidth unused. At most 16 readers, as
/// `release_manifest.py` uses, and one in a wasm build, which has no threads.
fn hash_files<P: AsRef<Path> + Sync>(paths: &[P]) -> Vec<std::io::Result<(u64, String)>> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let readers = if cfg!(target_family = "wasm") {
        1
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get().min(16))
    };
    let next = AtomicUsize::new(0);
    let read = || {
        let mut buffer = vec![0u8; 1 << 20];
        let mut done = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(path) = paths.get(i) else {
                return done;
            };
            done.push((i, hash_file(path.as_ref(), &mut buffer)));
        }
    };
    let mut done: Vec<_> = if readers.min(paths.len()) > 1 {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..readers.min(paths.len()))
                .map(|_| scope.spawn(read))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| {
                    h.join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                })
                .collect()
        })
    } else {
        read()
    };
    done.sort_unstable_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, hashed)| hashed).collect()
}

/// `(bytes, hex sha256)` of the file at `path`, read through `buffer`.
fn hash_file(path: &Path, buffer: &mut [u8]) -> std::io::Result<(u64, String)> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = StreamHash::default();
    loop {
        match file.read(buffer) {
            Ok(0) => return Ok(hash.finish()),
            Ok(n) => hash.update(&buffer[..n]),
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

/// Write `bytes` to `path` through a temporary file and a rename.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().ok_or(ErrorKind::InvalidInput)?;
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
        std::process::id()
    ));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

/// `time` as a manifest's `created` holds it, `2026-10-12T23:24:22Z`: UTC to
/// the second, as the publishing scripts write it.
fn utc_timestamp(time: SystemTime) -> Result<String, String> {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "the release's creation time is before 1970, so the clock is wrong")?
        .as_secs();
    let (days, second) = (seconds / 86_400, seconds % 86_400);
    // The civil date of a day count (Howard Hinnant's `civil_from_days`),
    // counting from 0000-03-01 so that a leap day ends its year.
    let shifted = days + 719_468;
    let era = shifted / 146_097;
    let day_of_era = shifted % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_from_march + 2) / 5 + 1;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    };
    let year = era * 400 + year_of_era + u64::from(month <= 2);
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        second / 3600,
        second % 3600 / 60,
        second % 60
    ))
}

/// `manifest` as `manifest.json` holds it.
fn manifest_bytes(manifest: &Manifest) -> Vec<u8> {
    let value = serde_json::to_value(manifest).expect("a manifest is strings, numbers and lists");
    let mut out = String::new();
    write_canonical(&value, &mut out);
    out.push('\n');
    out.into_bytes()
}

/// `value` as Python's `json.dumps(value, sort_keys=True, separators=(",",
/// ":"))` writes it, which is how the publishing scripts serialise a manifest:
/// no whitespace, the keys of every object sorted, every character outside
/// printable ASCII escaped, and floats as Python's `repr` writes them.
fn write_canonical(value: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        // serde_json writes a float with ryu: the shortest digits that read
        // back as it, and of two equally close, the even one, which is
        // Python's choice too. Only the layout differs.
        Value::Number(n) if n.is_f64() => out.push_str(&python_float(&n.to_string())),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut fields: Vec<_> = map.iter().collect();
            fields.sort_unstable_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            for (i, (key, item)) in fields.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write_canonical(item, out);
            }
            out.push('}');
        }
    }
}

/// `s` as a JSON string the way Python's `json.dumps` writes one by default
/// (`ensure_ascii=True`): printable ASCII as itself, the usual short escapes,
/// and anything else as `\uXXXX` in lower-case hex, in UTF-16 surrogate pairs
/// above U+FFFF.
fn write_string(s: &str, out: &mut String) {
    use std::fmt::Write as _;
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                for unit in c.encode_utf16(&mut [0u16; 2]) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// The float `shortest` (its shortest round-trip digits, laid out as ryu lays
/// them out: `250.0`, `0.0001`, `1e16`, `1.5e-7`) as Python's `repr` writes
/// it: positional for a decimal exponent from -4 up to 15, and otherwise with
/// a signed exponent of at least two digits, as in `1e+16` and `1.5e-07`.
fn python_float(shortest: &str) -> String {
    let (sign, unsigned) = match shortest.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", shortest),
    };
    let (mantissa, exponent) = unsigned.split_once('e').unwrap_or((unsigned, "0"));
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all = format!("{whole}{fraction}");
    let significant = all.trim_start_matches('0');
    // The value is 0.<digits> x 10^point.
    let point = whole.len() as i32 + exponent.parse::<i32>().unwrap_or(0)
        - (all.len() - significant.len()) as i32;
    let digits = significant.trim_end_matches('0');
    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    let len = digits.len() as i32;
    let body = if (-3..=16).contains(&point) {
        if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point >= len {
            format!("{digits}{}.0", "0".repeat((point - len) as usize))
        } else {
            let (whole, fraction) = digits.split_at(point as usize);
            format!("{whole}.{fraction}")
        }
    } else {
        // d.ddd x 10^exponent
        let exponent = point - 1;
        let (first, rest) = digits.split_at(1);
        let dot = if rest.is_empty() { "" } else { "." };
        let towards = if exponent < 0 { '-' } else { '+' };
        format!("{first}{dot}{rest}e{towards}{:02}", exponent.unsigned_abs())
    };
    format!("{sign}{body}")
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
            "converter_versions": ["yamc-convert 0.15.0", "yani-convert 0.15.0"],
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
    fn every_pointer_url_carries_a_cache_busting_value_of_its_own() {
        let prefix = "https://yamc-data.xsplot.com/endf-b8.1/latest.json?release-check=";
        let urls: Vec<String> = ["endf-b8.1", "endf-b8.1/"]
            .iter()
            .map(|library| pointer_url(&format!("https://yamc-data.xsplot.com/{library}")))
            .collect();
        for url in &urls {
            let value = url.strip_prefix(prefix).unwrap_or_else(|| panic!("{url}"));
            assert!(
                value.len() == 16 && value.chars().all(|c| c.is_ascii_hexdigit()),
                "{url}"
            );
        }
        assert_ne!(urls[0], urls[1], "two requests, two values");
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

    /// The shape the publishing scripts write (nuclear_data_generation_scripts
    /// `release_manifest.py`), with the values of the published endf-b8.1
    /// release 2026-10-12.1 and two of its 5008 files: compact, sorted keys.
    const PUBLISHED_MANIFEST: &str = concat!(
        r#"{"converter_versions":["yamc-convert 0.1.0","yani-convert 0.1.0"],"#,
        r#""created":"2026-10-12T23:24:22Z","file_count":5008,"files":["#,
        r#"{"bytes":17932,"path":"neutron/Li6.arrow/version.json","#,
        r#""sha256":"1245da86f7b99438ba860630a210b18277f3e0c6a81d96fc9457f12caa1711a9"},"#,
        r#"{"bytes":21052,"path":"transmutation/decay.arrow/provenance.json","#,
        r#""sha256":"43184e3209f2c49cb8ef41133701b83802d5044385cfb57ad5cdb3d4d28f7988"}],"#,
        r#""format_version":2,"keyword":"endf-b8.1","particles":["neutron","photon"],"#,
        r#""provenance":{"scripts_commit":"4fe9688493aefeb75c1a3a5d714c9f5b0a4c8439"},"#,
        r#""release":"2026-10-12.1","schema":"yamc-data-manifest","schema_version":1,"#,
        r#""total_bytes":5994239038}"#,
    );

    #[test]
    fn a_published_manifest_records_every_converter_that_wrote_it() {
        let bytes = PUBLISHED_MANIFEST.as_bytes();
        let release = parse_manifest(
            "endf-b8.1",
            bytes,
            Some(&pointer_for("2026-10-12.1", bytes)),
        )
        .unwrap();
        assert_eq!(
            release.manifest.converter_versions,
            ["yamc-convert 0.1.0", "yani-convert 0.1.0"]
        );
        assert_eq!(release.manifest.release, "2026-10-12.1");
        assert!(release.file("neutron/Li6.arrow/version.json").is_some());
    }

    /// Results saved before releases recorded their converters read back
    /// with an empty list.
    #[test]
    fn a_data_release_recorded_before_converter_versions_still_loads() {
        let saved =
            r#"{"release":"2026-10-01","manifest_sha256":"ab","format_version":2,"offline":false}"#;
        let record: DataRelease = serde_json::from_str(saved).unwrap();
        assert_eq!(record.release, "2026-10-01");
        assert!(record.converter_versions.is_empty());

        let record = DataRelease {
            converter_versions: vec!["yamc-convert 0.1.0".into(), "yani-convert 0.1.0".into()],
            ..record
        };
        let json = serde_json::to_string(&record).unwrap();
        assert_eq!(serde_json::from_str::<DataRelease>(&json).unwrap(), record);
    }

    #[test]
    fn a_manifest_that_lists_no_converters_still_parses() {
        let bytes = serde_json::to_vec(&serde_json::json!({
            "keyword": "endf-b8.1",
            "release": "2026-10-01",
            "format_version": 2,
            "files": [],
        }))
        .unwrap();
        let release = parse_manifest("endf-b8.1", &bytes, None).unwrap();
        assert!(release.manifest.converter_versions.is_empty());
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

    /// A release in the format the converters write is read, and the next
    /// format is refused with the advice to upgrade.
    #[test]
    fn the_format_the_converters_write_is_the_newest_this_build_reads() {
        let written = nuclear_data_schema::FORMAT_VERSION;
        assert!(check_format_version("endf-b8.1", "2026-10-01", written).is_ok());
        let err = check_format_version("endf-b8.1", "2027-01-01", written + 1).unwrap_err();
        assert!(err.contains("Upgrade"), "{err}");
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
    fn a_local_release_folder_has_its_sizes_checked() {
        let dir = std::env::temp_dir().join(format!("yamc-release-sizes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("neutron/Fe56.arrow")).unwrap();
        std::fs::write(dir.join("neutron/Fe56.arrow/version.json"), b"abc").unwrap();
        let bytes = manifest_json("endf-b8.1", "2026-10-01", 2);
        let manifest = parse_manifest("endf-b8.1", &bytes, None).unwrap().manifest;
        assert!(check_sizes(&dir, &manifest, "neutron/Fe56.arrow").is_ok());
        // Absent files are fine: Li6 was never downloaded.
        assert!(check_sizes(&dir, &manifest, "neutron/Li6.arrow").is_ok());
        std::fs::write(dir.join("neutron/Fe56.arrow/version.json"), b"ab").unwrap();
        let err = check_sizes(&dir, &manifest, "neutron/Fe56.arrow").unwrap_err();
        assert!(
            err.contains("size mismatch") && err.contains("version.json"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A local manifest names its own keyword, and its hash is computed: there
    /// is no pointer to compare either with.
    #[test]
    fn a_local_release_manifest_parses_with_its_own_keyword() {
        let path = std::path::Path::new("/data/jeff-4.0/manifest.json");
        let bytes = manifest_json("jeff-4.0", "2026-10-01", 2);
        let release = parse_local_manifest(path, &bytes)
            .unwrap()
            .expect("a release manifest");
        assert_eq!(release.manifest.keyword, "jeff-4.0");
        assert_eq!(release.manifest_sha256, sha256_hex(&bytes));
        assert_eq!(release.names_under("transmutation"), ["decay"]);
    }

    /// The `manifest.json` a converter writes at the top of a chain directory
    /// lists subsections rather than files: that directory is no release
    /// folder, and its manifest is no error.
    #[test]
    fn a_chain_manifest_is_not_a_release_manifest() {
        let path = std::path::Path::new("/data/chain/manifest.json");
        let chain = br#"{"format_version": 2, "library": "endf-b8.1", "subsections": {"decay": {"path": "decay"}}}"#;
        assert!(parse_local_manifest(path, chain).unwrap().is_none());
        let err = parse_local_manifest(path, b"{\"files\": [").unwrap_err();
        assert!(
            err.starts_with("/data/chain/manifest.json: not valid JSON"),
            "{err}"
        );
        let err = parse_local_manifest(path, br#"{"files": []}"#).unwrap_err();
        assert!(err.contains("not a valid release manifest"), "{err}");
    }

    /// A local release folder in a format this build cannot read is refused as
    /// a published one is, naming the manifest, and so is one whose paths
    /// leave the folder.
    #[test]
    fn a_local_release_manifest_is_checked_as_a_published_one_is() {
        let path = std::path::Path::new("/data/endf-b8.1/manifest.json");
        let err =
            parse_local_manifest(path, &manifest_json("endf-b8.1", "2027-01-01", 99)).unwrap_err();
        assert!(err.starts_with("/data/endf-b8.1/manifest.json: "), "{err}");
        assert!(
            err.contains("format_version 99") && err.contains("Upgrade"),
            "{err}"
        );
        let escaping = serde_json::to_vec(&serde_json::json!({
            "keyword": "endf-b8.1", "release": "2026-10-01", "format_version": 2,
            "files": [{"path": "../outside", "bytes": 1, "sha256": "00"}],
        }))
        .unwrap();
        let err = parse_local_manifest(path, &escaping).unwrap_err();
        assert!(err.contains("not a path inside the release"), "{err}");
        let unsafe_release = manifest_json("endf-b8.1", "..", 2);
        assert!(parse_local_manifest(path, &unsafe_release).is_err());
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

    const VERSION: &[u8] = br#"{"format_version": 2, "data_version": "2026-10-01", "converter_version": "yamc-convert 0.1.0", "library": "endf-b8.1"}"#;
    const PROVENANCE: &[u8] =
        br#"{"converter_version": "yani-convert 0.1.0", "data_version": "2026-10-01"}"#;

    /// A release folder in the shape of the tree `release_manifest.py`'s own
    /// tests publish (`make_tree` in its `tests/test_release_publishing.py`):
    /// an isomer beside its ground state (`.` sorts before `_`), raw bytes and
    /// an empty file, all three particles, two chain subsections, and
    /// [`UNPUBLISHED`].
    const TREE: &[(&str, &[u8])] = &[
        ("neutron/Am242.arrow/reactions.arrow", b"Am242-reactions"),
        ("neutron/Am242.arrow/version.json", VERSION),
        ("neutron/Am242_m1.arrow/reactions.arrow", b"Am242_m1-reactions"),
        ("neutron/Am242_m1.arrow/version.json", VERSION),
        (
            "neutron/Fe56.arrow/energy.arrow",
            b"ARROW1\x00\x00\xff\xfe\x80\x01",
        ),
        ("neutron/Fe56.arrow/urr.arrow", b""),
        ("neutron/Fe56.arrow/version.json", VERSION),
        ("neutron/index.json", b"{\"format\":1}\n"),
        ("neutron/index.txt", b"Am242\nAm242_m1\nFe56\n"),
        ("neutron/scratch/Fe57.arrow", b"not a nuclide folder"),
        ("photon/Fe.arrow/element.arrow", b"Fe-element"),
        ("photon/Fe.arrow/version.json", VERSION),
        (
            "photon/index.json",
            b"{\"format\":1,\"particle\":\"photon\"}\n",
        ),
        ("photonuclear/Be9.arrow/reactions.arrow", b"Be9-photonuclear"),
        ("photonuclear/Be9.arrow/version.json", VERSION),
        (
            "transmutation/branching.arrow/branching.arrow",
            b"branching-table",
        ),
        ("transmutation/branching.arrow/provenance.json", PROVENANCE),
        ("transmutation/decay.arrow/nuclides.arrow", b"decay-table"),
        ("transmutation/decay.arrow/provenance.json", PROVENANCE),
        ("index.txt", b"Am242\nAm242_m1\nFe56\n"),
        (
            ".build/chain/manifest.json",
            br#"{"format_version": 2, "library": "endf-b8.1", "converter_version": "yani-convert 0.1.0", "subsections": {}}"#,
        ),
        (".build/state.json", b"{}"),
    ];

    /// The files of [`TREE`] that are not part of the release: lists beside
    /// the folders, a folder not named `<Name>.arrow`, and what a local build
    /// keeps in `.build/`, the chain's own manifest among it.
    const UNPUBLISHED: &[&str] = &[
        "neutron/index.txt",
        "neutron/scratch/Fe57.arrow",
        "index.txt",
        ".build/chain/manifest.json",
        ".build/state.json",
    ];

    /// What `release_manifest.py build --keyword endf-b8.1 --particles "neutron
    /// photon photonuclear" --release 2026-10-01` writes for [`TREE`] laid out
    /// as its build tree (`endf-b8.1-arrow/<particle>/...`, the chain in
    /// `transmutation-endf-b8.1.arrow/<subsection>/` with its manifest beside
    /// the subsections), run from nuclear_data_generation_scripts b813787 with
    /// `SOURCE_DATE_EPOCH=1790000000`, outside a git checkout so that
    /// `scripts_commit` is empty.
    const RELEASE_MANIFEST_PY: &str = concat!(
        r#"{"converter_versions":["yamc-convert 0.1.0","yani-convert 0.1.0"],"#,
        r#""created":"2026-09-21T14:13:20Z","file_count":17,"files":["#,
        r#"{"bytes":15,"path":"neutron/Am242.arrow/reactions.arrow","#,
        r#""sha256":"498dbbbb6a01f00bda9b016664aeaaa630dcafebf6f261811c596d0fd4fd3150"},"#,
        r#"{"bytes":118,"path":"neutron/Am242.arrow/version.json","#,
        r#""sha256":"cbd21517fd692fb1eda29939cd99d3e7dd433c2303671ff627bff5ffd1445eec"},"#,
        r#"{"bytes":18,"path":"neutron/Am242_m1.arrow/reactions.arrow","#,
        r#""sha256":"3c1103f0d6ce0a5ebc677384a6367de35c491470f47cdec4eada8ac50392cfab"},"#,
        r#"{"bytes":118,"path":"neutron/Am242_m1.arrow/version.json","#,
        r#""sha256":"cbd21517fd692fb1eda29939cd99d3e7dd433c2303671ff627bff5ffd1445eec"},"#,
        r#"{"bytes":12,"path":"neutron/Fe56.arrow/energy.arrow","#,
        r#""sha256":"4ead9a897a3066f53bcecc233c79d964f4734ccaa6e13f015761aa22c913a4d2"},"#,
        r#"{"bytes":0,"path":"neutron/Fe56.arrow/urr.arrow","#,
        r#""sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},"#,
        r#"{"bytes":118,"path":"neutron/Fe56.arrow/version.json","#,
        r#""sha256":"cbd21517fd692fb1eda29939cd99d3e7dd433c2303671ff627bff5ffd1445eec"},"#,
        r#"{"bytes":13,"path":"neutron/index.json","#,
        r#""sha256":"2f0fd78a91263ee7352df687964d366f9e7323d54cb0412cb1c1d73afeb5bcf1"},"#,
        r#"{"bytes":10,"path":"photon/Fe.arrow/element.arrow","#,
        r#""sha256":"c3c5c3b22b99b16f258814f977bc9aa985dc26770fa73ae2ff520ba240b908bf"},"#,
        r#"{"bytes":118,"path":"photon/Fe.arrow/version.json","#,
        r#""sha256":"cbd21517fd692fb1eda29939cd99d3e7dd433c2303671ff627bff5ffd1445eec"},"#,
        r#"{"bytes":33,"path":"photon/index.json","#,
        r#""sha256":"7d2e7c53f5f2e7d2c17e0285b8423bbb8b41ab7e6b0c98c3423a9f44088418f5"},"#,
        r#"{"bytes":16,"path":"photonuclear/Be9.arrow/reactions.arrow","#,
        r#""sha256":"35802f6feab4bdd7d73224fea1b0f6e8bb8fec66b50f3ee5a0fa0e4523a3e1ce"},"#,
        r#"{"bytes":118,"path":"photonuclear/Be9.arrow/version.json","#,
        r#""sha256":"cbd21517fd692fb1eda29939cd99d3e7dd433c2303671ff627bff5ffd1445eec"},"#,
        r#"{"bytes":15,"path":"transmutation/branching.arrow/branching.arrow","#,
        r#""sha256":"3244041a18ca4dc04452ec06d60231ca305dbc62c5ca629f119af811e751085c"},"#,
        r#"{"bytes":73,"path":"transmutation/branching.arrow/provenance.json","#,
        r#""sha256":"e946d95abf6bd6280a465a84bfeff700bcaa57204c97f1b3821e8c22c92e930b"},"#,
        r#"{"bytes":11,"path":"transmutation/decay.arrow/nuclides.arrow","#,
        r#""sha256":"eba3b096d5a1eb49549f9e6d217754eb5d3347e03beb4437dc2c179e5c04428f"},"#,
        r#"{"bytes":73,"path":"transmutation/decay.arrow/provenance.json","#,
        r#""sha256":"e946d95abf6bd6280a465a84bfeff700bcaa57204c97f1b3821e8c22c92e930b"}],"#,
        r#""format_version":2,"keyword":"endf-b8.1","#,
        r#""particles":["neutron","photon","photonuclear"],"#,
        r#""provenance":{"scripts_commit":""},"release":"2026-10-01","#,
        r#""schema":"yamc-data-manifest","schema_version":1,"total_bytes":879}"#,
        "\n",
    );

    /// `SOURCE_DATE_EPOCH=1790000000`, as `release_manifest.py` was run.
    fn created() -> SystemTime {
        UNIX_EPOCH + std::time::Duration::from_secs(1_790_000_000)
    }

    /// `files` written into a fresh folder named for `test`.
    fn folder(test: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("yamc-release-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for (path, data) in files {
            let file = dir.join(path);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, data).unwrap();
        }
        dir
    }

    /// The manifest of [`TREE`], with what `release_manifest.py` was run with.
    fn write_tree_manifest(dir: &Path) -> Result<Release, String> {
        let provenance = serde_json::json!({"scripts_commit": ""});
        write_manifest(
            dir,
            "endf-b8.1",
            "2026-10-01",
            2,
            created(),
            provenance.as_object().unwrap().clone(),
        )
    }

    #[test]
    fn a_manifest_is_byte_for_byte_what_release_manifest_py_writes_for_the_same_files() {
        let dir = folder("ndgs-bytes", TREE);
        write_tree_manifest(&dir).unwrap();
        let written = std::fs::read_to_string(dir.join(MANIFEST)).unwrap();
        assert_eq!(written, RELEASE_MANIFEST_PY);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_written_manifest_reads_back_as_written() {
        let dir = folder("round-trip", TREE);
        let written = write_tree_manifest(&dir).unwrap();
        let bytes = std::fs::read(dir.join(MANIFEST)).unwrap();
        let read = parse_manifest("endf-b8.1", &bytes, None).unwrap();
        assert_eq!(read.manifest, written.manifest);
        assert_eq!(read.manifest_sha256, written.manifest_sha256);
        assert_eq!(read.manifest_sha256, sha256_hex(&bytes));

        let manifest = &read.manifest;
        let listed: Vec<&str> = manifest.files.iter().map(|f| f.path.as_str()).collect();
        let published: BTreeSet<&str> = TREE
            .iter()
            .map(|(path, _)| *path)
            .filter(|path| !UNPUBLISHED.contains(path))
            .collect();
        assert_eq!(
            listed,
            published.into_iter().collect::<Vec<_>>(),
            "every published file, sorted, and nothing else"
        );
        for file in &manifest.files {
            let on_disk = std::fs::read(dir.join(&file.path)).unwrap();
            assert_eq!(file.bytes, on_disk.len() as u64, "{}", file.path);
            assert_eq!(file.sha256, sha256_hex(&on_disk), "{}", file.path);
        }
        assert_eq!(manifest.schema.as_deref(), Some(MANIFEST_SCHEMA));
        assert_eq!(manifest.schema_version, Some(MANIFEST_SCHEMA_VERSION));
        assert_eq!(manifest.created.as_deref(), Some("2026-09-21T14:13:20Z"));
        assert_eq!(manifest.particles, PARTICLES);
        assert_eq!(
            manifest.converter_versions,
            ["yamc-convert 0.1.0", "yani-convert 0.1.0"]
        );
        assert_eq!(manifest.file_count, Some(17));
        assert_eq!(
            manifest.total_bytes,
            Some(manifest.files.iter().map(|f| f.bytes).sum())
        );
        assert_eq!(read.names_under("neutron"), ["Am242", "Am242_m1", "Fe56"]);
        assert_eq!(read.names_under("transmutation"), ["branching", "decay"]);

        // The manifest in the folder is not part of what it describes, so
        // describing the folder again gives the same bytes.
        write_tree_manifest(&dir).unwrap();
        assert_eq!(std::fs::read(dir.join(MANIFEST)).unwrap(), bytes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// irdff-ii publishes branching ratios and nothing else, and no file of a
    /// chain subsection carries the format, so the caller's word is the only
    /// source of it.
    #[test]
    fn a_chain_only_library_is_described_in_the_format_the_caller_states() {
        let dir = folder(
            "chain-only",
            &[
                (
                    "transmutation/branching.arrow/branching.arrow",
                    b"branching-table",
                ),
                ("transmutation/branching.arrow/provenance.json", PROVENANCE),
            ],
        );
        let release = write_manifest(
            &dir,
            "irdff-ii",
            "2026-10-12.1",
            2,
            created(),
            Default::default(),
        )
        .unwrap();
        let manifest = &release.manifest;
        assert_eq!(manifest.format_version, 2);
        assert!(manifest.particles.is_empty());
        assert!(manifest.provenance.is_empty());
        assert_eq!(manifest.converter_versions, ["yani-convert 0.1.0"]);
        assert_eq!(release.names_under("transmutation"), ["branching"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_published_manifest_serialises_back_to_its_bytes() {
        let release = parse_manifest("endf-b8.1", PUBLISHED_MANIFEST.as_bytes(), None).unwrap();
        assert_eq!(
            String::from_utf8(manifest_bytes(&release.manifest)).unwrap(),
            format!("{PUBLISHED_MANIFEST}\n")
        );
    }

    /// The expected string is what Python 3.13 prints for the same document
    /// with `json.dumps(doc, sort_keys=True, separators=(",", ":"))`.
    #[test]
    fn the_canonical_form_is_what_pythons_json_dumps_writes() {
        // Exactly 242972808493158.625, halfway between two 17-digit strings
        // that both read back as it: Python writes the even one, and Rust's own
        // shortest formatting (`{:e}`) the other.
        let tie = f64::from_bits(0x42eb_9f70_7e17_0cd4);
        let doc = serde_json::json!({
            "text": "Fe\u{e9} \u{2622} \u{1d509} \u{7f} \u{1} \" \\ / \n\t\r\u{8}\u{c}",
            "floats": [
                0.0, -0.0, 293.6, 250.0, 1e16, 9999999999999998.0, 1e-5, 0.0001, 0.00015,
                5e-324, f64::MAX, 12345.678, -1.5e-7, 0.1, tie, 1.2345e-5, 1e22
            ],
            "ints": [u64::MAX, i64::MIN, 0],
            "nested": {"z": 1, "y": null, "x": true, "w": false, "\u{e9}": []},
        });
        let mut out = String::new();
        write_canonical(&doc, &mut out);
        assert_eq!(
            out,
            concat!(
                r#"{"floats":[0.0,-0.0,293.6,250.0,1e+16,9999999999999998.0,1e-05,0.0001,"#,
                r#"0.00015,5e-324,1.7976931348623157e+308,12345.678,-1.5e-07,0.1,"#,
                r#"242972808493158.62,1.2345e-05,1e+22],"#,
                r#""ints":[18446744073709551615,-9223372036854775808,0],"#,
                r#""nested":{"w":false,"x":true,"y":null,"z":1,"\u00e9":[]},"#,
                r#""text":"Fe\u00e9 \u2622 \ud835\udd09 \u007f \u0001 \" \\ / \n\t\r\b\f"}"#,
            )
        );
    }

    /// The expected strings are what Python 3.13's
    /// `datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")`
    /// prints, which is how `release_manifest.py` stamps `created`.
    #[test]
    fn the_creation_time_is_written_as_release_manifest_py_writes_it() {
        let at = |seconds: u64| utc_timestamp(UNIX_EPOCH + std::time::Duration::from_secs(seconds));
        for (seconds, expected) in [
            (0, "1970-01-01T00:00:00Z"),
            (951_782_400, "2000-02-29T00:00:00Z"),
            (1_709_251_199, "2024-02-29T23:59:59Z"),
            (1_790_000_000, "2026-09-21T14:13:20Z"),
            (4_102_444_800, "2100-01-01T00:00:00Z"),
            (253_402_300_799, "9999-12-31T23:59:59Z"),
        ] {
            assert_eq!(at(seconds).unwrap(), expected, "{seconds}");
        }
        // To the second, as strftime has it.
        let late = created() + std::time::Duration::from_millis(999);
        assert_eq!(utc_timestamp(late).unwrap(), "2026-09-21T14:13:20Z");
        let before = UNIX_EPOCH - std::time::Duration::from_secs(1);
        assert!(utc_timestamp(before).unwrap_err().contains("before 1970"));
    }

    #[test]
    fn a_manifest_is_refused_for_a_folder_a_release_cannot_be() {
        let dir = folder("refused", TREE);
        let write = |keyword: &str, release: &str, format_version: u32| {
            write_manifest(
                &dir,
                keyword,
                release,
                format_version,
                created(),
                Default::default(),
            )
        };
        for good in ["2026-10-12", "2026-10-12.1", "1_a-B.c"] {
            assert!(check_release_id(good).is_ok(), "{good}");
        }
        for bad in [
            "",
            "latest",
            "../2026",
            "2026/10",
            "2026 10",
            "2026-10-01\n",
        ] {
            let err = write("endf-b8.1", bad, 2).unwrap_err();
            assert!(
                err.contains("not a valid release identifier"),
                "{bad:?}: {err}"
            );
        }
        for bad in ["", "..", "endf/b8.1"] {
            let err = write(bad, "2026-10-01", 2).unwrap_err();
            assert!(err.contains("cannot name a library"), "{bad:?}: {err}");
        }
        let err = write("endf-b8.1", "2026-10-01", 99).unwrap_err();
        assert!(
            err.contains("format_version 99") && err.contains("it reads 2"),
            "{err}"
        );

        // One folder in a format other than the release's.
        let fe56 = dir.join("neutron/Fe56.arrow/version.json");
        std::fs::write(&fe56, r#"{"format_version": 1}"#).unwrap();
        let err = write("endf-b8.1", "2026-10-01", 2).unwrap_err();
        assert!(
            err.contains("Fe56.arrow") && err.contains("format_version 1"),
            "{err}"
        );
        std::fs::write(&fe56, VERSION).unwrap();

        // A name the reader refuses would make the manifest unreadable. Only
        // Unix can make one: Windows refuses these names itself (and takes
        // `a:b` for the stream `b` of the file `a`).
        #[cfg(unix)]
        {
            std::fs::write(dir.join("photon/Fe.arrow/a:b"), "x").unwrap();
            let err = write("endf-b8.1", "2026-10-01", 2).unwrap_err();
            assert!(err.contains("a:b"), "{err}");
            std::fs::remove_file(dir.join("photon/Fe.arrow/a:b")).unwrap();
        }
        assert!(
            !dir.join(MANIFEST).exists(),
            "a refused folder gets no manifest"
        );

        // Only what a release does not publish.
        let err = write_manifest(
            &dir.join(".build"),
            "endf-b8.1",
            "2026-10-01",
            2,
            created(),
            Default::default(),
        )
        .unwrap_err();
        assert!(err.contains("nothing to publish"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_reports_changed_missing_and_unlisted_files() {
        let dir = folder("verify", TREE);
        let written = write_tree_manifest(&dir).unwrap();
        let clean = verify_folder(&dir).unwrap();
        assert!(clean.ok(), "{clean:?}");
        assert_eq!(clean.release.manifest_sha256, written.manifest_sha256);

        // The same size with other bytes, which only the hash catches, and a
        // shorter file.
        std::fs::write(
            dir.join("neutron/Am242.arrow/reactions.arrow"),
            "AM242-reactions",
        )
        .unwrap();
        std::fs::write(dir.join("photon/index.json"), "{}").unwrap();
        // Gone, and a folder where a file was.
        std::fs::remove_file(dir.join("transmutation/decay.arrow/nuclides.arrow")).unwrap();
        std::fs::remove_file(dir.join("neutron/Fe56.arrow/urr.arrow")).unwrap();
        std::fs::create_dir(dir.join("neutron/Fe56.arrow/urr.arrow")).unwrap();
        // Data the manifest does not cover, which a load would read: a
        // nuclide, and a section copy ranged by temperature, as the cache
        // keeps one until it holds the whole section.
        std::fs::create_dir_all(dir.join("neutron/Fe57.arrow")).unwrap();
        std::fs::write(dir.join("neutron/Fe57.arrow/version.json"), VERSION).unwrap();
        std::fs::create_dir_all(dir.join("neutron/Fe56.arrow/temperatures")).unwrap();
        std::fs::write(
            dir.join("neutron/Fe56.arrow/temperatures/reactions.arrow"),
            "x",
        )
        .unwrap();
        // Not part of the release, so not looked at.
        std::fs::write(dir.join(".build/state.json"), "{\"stage\":2}").unwrap();
        std::fs::write(dir.join("notes.txt"), "mine").unwrap();
        std::fs::write(dir.join("transmutation/manifest.json"), "{}").unwrap();

        let found = verify_folder(&dir).unwrap();
        assert_eq!(
            found.changed,
            ["neutron/Am242.arrow/reactions.arrow", "photon/index.json"]
        );
        assert_eq!(
            found.missing,
            [
                "neutron/Fe56.arrow/urr.arrow",
                "transmutation/decay.arrow/nuclides.arrow"
            ]
        );
        assert_eq!(
            found.unlisted,
            [
                "neutron/Fe56.arrow/temperatures/reactions.arrow",
                "neutron/Fe57.arrow/version.json"
            ]
        );
        assert!(!found.ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_refuses_a_folder_it_has_nothing_to_check_against() {
        let dir = folder("verify-refused", TREE);
        let err = verify_folder(&dir).unwrap_err();
        assert!(err.contains("holds no manifest.json"), "{err}");
        let err = verify_folder(&dir.join("index.txt")).unwrap_err();
        assert!(err.contains("is not a folder"), "{err}");

        // A converted chain's own manifest is not a release's.
        let chain = r#"{"format_version": 2, "library": "endf-b8.1", "subsections": {}}"#;
        std::fs::write(dir.join(MANIFEST), chain).unwrap();
        let err = verify_folder(&dir).unwrap_err();
        assert!(err.contains("not a release manifest"), "{err}");

        std::fs::write(
            dir.join(MANIFEST),
            manifest_json("endf-b8.1", "2026-10-01", 2),
        )
        .unwrap();
        let err = verify_folder(&dir).unwrap_err();
        assert!(
            err.contains("not a yamc-data-manifest v1 manifest"),
            "{err}"
        );

        let mut newer: serde_json::Value = serde_json::from_str(RELEASE_MANIFEST_PY).unwrap();
        newer["format_version"] = 99.into();
        std::fs::write(dir.join(MANIFEST), newer.to_string()).unwrap();
        let err = verify_folder(&dir).unwrap_err();
        assert!(
            err.contains("manifest.json") && err.contains("format_version 99"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
