//! Library keywords resolved against the published release layout, with every
//! downloaded file verified against its release manifest.
//!
//! Per library keyword, once per process:
//!
//! 1. `<origin>/<keyword>/latest.json` names the current release. Its
//!    `format_version` must be one this build reads
//!    ([`super::release::SUPPORTED_FORMAT_VERSIONS`]).
//! 2. The release's `manifest.json` is fetched (or read back from the cache),
//!    checked against the pointer's size and sha256, and kept in memory and in
//!    the cache beside the release's files.
//! 3. Every file the run needs is served from
//!    `<cache root>/<keyword>/<release>/<path>`, downloaded first when it is
//!    not there: streamed to a temporary file, counted and hashed while
//!    streaming, compared with the manifest entry, fsynced, and renamed into
//!    place. A mismatch is a hard error naming the file.
//!
//! One release per library per process. When the pointer names a newer
//! release than the one the cache holds, every file of that library this run
//! reads comes from the new release (downloaded as needed), and the switch is
//! logged once; files of two releases are never combined.
//!
//! An unreachable origin is not an error when the data is already local. If
//! the pointer or the manifest cannot be fetched for a transport reason (DNS,
//! connect, timeout, TLS, a 5xx), the newest cached release that holds every
//! file the run asks for is used, the fact is logged once, and that release is
//! what the run's provenance records. Only a file missing from the cache is an
//! error then, and the error says no connection was available.
//!
//! The cache layout:
//!
//! ```text
//! <cache root>/<keyword>/<release>/manifest.json
//! <cache root>/<keyword>/<release>/neutron/<Nuclide>.arrow/<section>
//! <cache root>/<keyword>/<release>/photon/<Element>.arrow/<section>
//! <cache root>/<keyword>/<release>/transmutation/<subsection>.arrow/<section>
//! ```
//!
//! A file is only ever renamed into a release folder after it verified, so a
//! file that is present there is a verified one.
//!
//! `reactions.arrow` and `energy.arrow` are always downloaded whole on this
//! path. A byte-range fetch of a few MTs or temperatures cannot be checked
//! against a whole-file hash, so the ranged fetches are kept for raw URL
//! sources only (which have no manifest to check against); a keyword load that
//! names its MTs or temperatures downloads the whole tables once, verified,
//! and every later load reads them from the cache.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use once_cell::sync::{Lazy, OnceCell};

use super::release::{
    parse_latest, parse_manifest, verify, DataRelease, LatestPointer, ManifestFile, Release,
    StreamHash, LATEST, MANIFEST,
};
use super::url_cache::{cache_root, get_path_lock, sections_for, transmutation_sections, DataKind};

/// How long each phase of a request may take.
///
/// The first contact with the origin is `latest.json`, a few hundred bytes, so
/// its budget is short: an offline machine whose DNS fails is told so at once,
/// and one whose packets vanish waits at most
/// `pointer_attempts * (connect or pointer timeout)` before falling back to the
/// cache, 2 x 3 s with the defaults. A data file is bounded per read rather
/// than in total, since a 190 MB actinide on a slow link legitimately takes
/// minutes, but a stalled one must not hang forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Timeouts {
    /// TCP and TLS connection setup, every request.
    pub connect: Duration,
    /// The whole `latest.json` request, per attempt.
    pub pointer: Duration,
    /// Attempts at `latest.json` before the origin is taken as unreachable.
    pub pointer_attempts: usize,
    /// Waiting for response headers, and for each read of a body.
    pub read: Duration,
}

impl Timeouts {
    /// The values every download uses. See the type's docs for why each is
    /// what it is; docs/developer_info.md quotes them.
    pub(crate) const DEFAULT: Timeouts = Timeouts {
        connect: Duration::from_secs(3),
        pointer: Duration::from_secs(3),
        pointer_attempts: 2,
        read: Duration::from_secs(30),
    };
}

/// Build an HTTP client with `timeouts` applied.
pub(crate) fn build_client(timeouts: Timeouts) -> reqwest::Result<reqwest::blocking::Client> {
    #[cfg(feature = "download-tls")]
    super::url_cache::ensure_tls_provider();
    reqwest::blocking::Client::builder()
        .connect_timeout(timeouts.connect)
        // On the blocking client this bounds the wait for headers and each
        // read of the body, not the transfer as a whole.
        .timeout(timeouts.read)
        .build()
}

/// A release folder in the cache, with its parsed manifest.
#[derive(Debug)]
struct CachedRelease {
    release: Release,
    /// `<cache root>/<keyword>/<release>`.
    dir: PathBuf,
}

impl CachedRelease {
    fn id(&self) -> &str {
        &self.release.manifest.release
    }

    fn record(&self, offline: bool) -> DataRelease {
        DataRelease {
            release: self.id().to_string(),
            manifest_sha256: self.release.manifest_sha256.clone(),
            format_version: self.release.manifest.format_version,
            offline,
        }
    }
}

/// How a library resolved for this process.
#[derive(Debug)]
enum Library {
    /// The origin answered: files come from this release, downloaded as needed.
    Online {
        release: Box<CachedRelease>,
        /// `<origin>/<keyword>/<release>/`.
        base_url: String,
    },
    /// The origin could not be reached. The release is chosen on the first
    /// request (the newest cached one holding every file it needs) and then
    /// kept for the rest of the process.
    Offline {
        reason: String,
        candidates: Vec<Arc<CachedRelease>>,
        chosen: Mutex<Option<Arc<CachedRelease>>>,
    },
}

/// One library's resolution, made once: `None` until the first request.
type ResolutionSlot = Mutex<Option<Result<Arc<Library>, String>>>;

/// The per-process state: which release each library resolved to.
pub(crate) struct Registry {
    /// Scheme and host, with a trailing `/`.
    origin: String,
    /// Cache root; `None` resolves [`cache_root`] at each use.
    root: Option<PathBuf>,
    timeouts: Timeouts,
    client: OnceCell<reqwest::Result<reqwest::blocking::Client>>,
    /// Keyword -> its resolution, success or failure, made once.
    libraries: Mutex<HashMap<String, Arc<ResolutionSlot>>>,
    /// Keyword -> the release the run used, for provenance.
    used: Mutex<BTreeMap<String, DataRelease>>,
}

/// The registry every keyword load goes through.
pub(crate) static REGISTRY: Lazy<Registry> =
    Lazy::new(|| Registry::new(super::url_cache::ORIGIN, None, Timeouts::DEFAULT));

/// The release each library keyword resolved to in this process, keyed by
/// keyword. Only libraries a file was actually served from appear.
pub fn data_releases() -> BTreeMap<String, DataRelease> {
    REGISTRY.data_releases()
}

impl Registry {
    pub(crate) fn new(origin: &str, root: Option<PathBuf>, timeouts: Timeouts) -> Self {
        let origin = format!("{}/", origin.trim_end_matches('/'));
        Registry {
            origin,
            root,
            timeouts,
            client: OnceCell::new(),
            libraries: Mutex::new(HashMap::new()),
            used: Mutex::new(BTreeMap::new()),
        }
    }

    fn root(&self) -> Result<PathBuf, String> {
        self.root.clone().or_else(cache_root).ok_or_else(|| {
            "Could not find a cache directory: the platform cache directory did not resolve"
                .to_string()
        })
    }

    fn client(&self) -> Result<&reqwest::blocking::Client, String> {
        self.client
            .get_or_init(|| build_client(self.timeouts))
            .as_ref()
            .map_err(|e| format!("could not build an HTTP client: {e}"))
    }

    pub(crate) fn data_releases(&self) -> BTreeMap<String, DataRelease> {
        self.used.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn record_use(&self, keyword: &str, release: DataRelease) {
        self.used
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(keyword.to_string(), release);
    }

    /// The library's resolution, made on first use and kept for the process.
    fn library(&self, keyword: &str) -> Result<Arc<Library>, String> {
        let slot = self
            .libraries
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(keyword.to_string())
            .or_default()
            .clone();
        let mut slot = slot.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some(self.resolve(keyword).map(Arc::new));
        }
        slot.clone().expect("resolved above")
    }

    /// Ask the origin which release is current, falling back to the cache when
    /// it cannot be reached.
    fn resolve(&self, keyword: &str) -> Result<Library, String> {
        let root = self.root()?;
        let library_dir = root.join(keyword);
        let latest_url = format!("{}{keyword}/{LATEST}", self.origin);
        let pointer = match self.get_small(&latest_url, true) {
            Ok(Some(bytes)) => parse_latest(keyword, &bytes)?,
            Ok(None) => return Err(not_published_error(keyword, &latest_url)),
            Err(Unreachable(reason)) => return Ok(self.offline(keyword, &library_dir, reason)),
        };
        let release_dir = library_dir.join(&pointer.release);
        // Whether this process is the first to use the release, which is when
        // a switch from an older one is worth saying.
        let first_use = !release_dir.join(MANIFEST).is_file();
        let release = match self.manifest(keyword, &pointer, &release_dir) {
            Ok(release) => release,
            Err(ManifestError::Unreachable(reason)) => {
                return Ok(self.offline(keyword, &library_dir, reason))
            }
            Err(ManifestError::Invalid(e)) => return Err(e),
        };
        let previous = cached_releases(keyword, &library_dir)
            .into_iter()
            .find(|cached| cached.id() != pointer.release);
        if let (true, Some(previous)) = (first_use, previous) {
            println!(
                "yamc: updated {keyword} from {} to {}: every {keyword} file this run reads \
                 comes from {}, downloaded and verified where the cache does not hold it yet.",
                previous.id(),
                pointer.release,
                pointer.release
            );
        }
        Ok(Library::Online {
            base_url: format!("{}{keyword}/{}/", self.origin, pointer.release),
            release: Box::new(CachedRelease {
                release,
                dir: release_dir,
            }),
        })
    }

    /// The release's manifest: the cached copy when it matches the pointer,
    /// otherwise fetched, verified, and written to the cache.
    fn manifest(
        &self,
        keyword: &str,
        pointer: &LatestPointer,
        release_dir: &Path,
    ) -> Result<Release, ManifestError> {
        let cached = release_dir.join(MANIFEST);
        if let Ok(bytes) = fs::read(&cached) {
            if let Ok(release) = parse_manifest(keyword, &bytes, Some(pointer)) {
                return Ok(release);
            }
        }
        let url = format!("{}{keyword}/{}", self.origin, pointer.manifest);
        let bytes = match self.get_small(&url, false) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                return Err(ManifestError::Invalid(format!(
                    "'{keyword}' {LATEST} names release {} but its manifest {url} answers 404. \
                     The release is not completely published; try again shortly.",
                    pointer.release
                )))
            }
            Err(Unreachable(reason)) => return Err(ManifestError::Unreachable(reason)),
        };
        let release =
            parse_manifest(keyword, &bytes, Some(pointer)).map_err(ManifestError::Invalid)?;
        write_atomically(&cached, &bytes).map_err(|e| {
            ManifestError::Invalid(format!("could not cache {}: {e}", cached.display()))
        })?;
        Ok(release)
    }

    /// The offline resolution: every cached release of the library, newest
    /// first, to choose from on the first request.
    fn offline(&self, keyword: &str, library_dir: &Path, reason: String) -> Library {
        Library::Offline {
            reason,
            candidates: cached_releases(keyword, library_dir)
                .into_iter()
                .map(Arc::new)
                .collect(),
            chosen: Mutex::new(None),
        }
    }

    /// GET a small object whole. `Ok(None)` is a 404. A transport failure or
    /// a 5xx that outlasts the retries is `Unreachable`.
    fn get_small(&self, url: &str, pointer: bool) -> Result<Option<Vec<u8>>, Unreachable> {
        let client = self.client().map_err(Unreachable)?;
        let attempts = if pointer {
            self.timeouts.pointer_attempts.max(1)
        } else {
            RETRY_DELAYS_MS.len() + 1
        };
        let mut last = String::new();
        for attempt in 0..attempts {
            if attempt > 0 {
                std::thread::sleep(Duration::from_millis(
                    RETRY_DELAYS_MS[(attempt - 1).min(RETRY_DELAYS_MS.len() - 1)],
                ));
            }
            let mut request = client.get(url);
            if pointer {
                request = request.timeout(self.timeouts.pointer);
            }
            let response = match request.send() {
                Ok(r) => r,
                Err(e) => {
                    last = describe(&e);
                    continue;
                }
            };
            let status = response.status();
            if status == reqwest::StatusCode::NOT_FOUND {
                return Ok(None);
            }
            if !status.is_success() {
                last = format!("{url}: HTTP {status}");
                continue;
            }
            match response.bytes() {
                Ok(bytes) => return Ok(Some(bytes.to_vec())),
                Err(e) => last = format!("{url}: reading the body: {}", describe(&e)),
            }
        }
        Err(Unreachable(last))
    }

    /// The local folder holding `sections` of `dir` (`neutron/Fe56.arrow`,
    /// `transmutation/decay.arrow`, ...) of `keyword`'s release, downloading
    /// and verifying whatever the cache lacks. `what` names the data in
    /// messages.
    pub(crate) fn fetch_dir(
        &self,
        keyword: &str,
        dir: &str,
        sections: &[(&str, bool)],
        what: &str,
    ) -> Result<PathBuf, String> {
        let library = self.library(keyword)?;
        match &*library {
            Library::Online { release, base_url } => {
                let needed = needed_files(keyword, release, dir, sections, what)?;
                let local = release.dir.join(dir);
                self.record_use(keyword, release.record(false));
                let missing: Vec<&ManifestFile> = needed
                    .iter()
                    .copied()
                    .filter(|f| !release.dir.join(&f.path).is_file())
                    .collect();
                if missing.is_empty() {
                    log_from_cache(keyword, release.id(), what, &local);
                    return Ok(local);
                }
                let lock = get_path_lock(&local);
                let _guard = lock.lock().unwrap_or_else(|p| p.into_inner());
                let missing: Vec<&ManifestFile> = missing
                    .into_iter()
                    .filter(|f| !release.dir.join(&f.path).is_file())
                    .collect();
                if !missing.is_empty() {
                    self.download(keyword, release, base_url, &missing, what, &local)?;
                }
                Ok(local)
            }
            Library::Offline {
                reason,
                candidates,
                chosen,
            } => {
                let mut chosen = chosen.lock().unwrap_or_else(|p| p.into_inner());
                if chosen.is_none() {
                    let pick = candidates.iter().find(|c| {
                        needed_files(keyword, c, dir, sections, what)
                            .is_ok_and(|files| files.iter().all(|f| c.dir.join(&f.path).is_file()))
                    });
                    let Some(pick) = pick else {
                        return Err(offline_missing_error(
                            keyword,
                            reason,
                            candidates.first().map(|c| &**c),
                            dir,
                            sections,
                            what,
                            &self.root()?,
                        ));
                    };
                    println!(
                        "yamc: could not reach the nuclear-data origin for {keyword} ({reason}); \
                         using cached release {} (manifest sha256 {}), verified when it was \
                         downloaded.",
                        pick.id(),
                        pick.release.manifest_sha256
                    );
                    self.record_use(keyword, pick.record(true));
                    *chosen = Some(Arc::clone(pick));
                }
                let release = chosen.as_ref().expect("chosen above");
                let needed = needed_files(keyword, release, dir, sections, what)?;
                if needed.iter().any(|f| !release.dir.join(&f.path).is_file()) {
                    return Err(offline_missing_error(
                        keyword,
                        reason,
                        Some(release),
                        dir,
                        sections,
                        what,
                        &self.root()?,
                    ));
                }
                let local = release.dir.join(dir);
                log_from_cache(keyword, release.id(), what, &local);
                Ok(local)
            }
        }
    }

    /// Download `files` into the release folder, each verified, then publish
    /// them by rename.
    fn download(
        &self,
        keyword: &str,
        release: &CachedRelease,
        base_url: &str,
        files: &[&ManifestFile],
        what: &str,
        local: &Path,
    ) -> Result<(), String> {
        let client = self.client()?;
        fs::create_dir_all(&release.dir)
            .map_err(|e| format!("could not create {}: {e}", release.dir.display()))?;
        let staging = tempdir_in(&release.dir)?;
        let result = (|| -> Result<u64, String> {
            let mut total = 0;
            for (i, file) in files.iter().enumerate() {
                let tmp = staging.join(i.to_string());
                fetch_verified(client, &format!("{base_url}{}", file.path), file, &tmp)?;
                total += file.bytes;
            }
            for (i, file) in files.iter().enumerate() {
                let to = release.dir.join(&file.path);
                if let Some(parent) = to.parent() {
                    fs::create_dir_all(parent)
                        .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
                }
                fs::rename(staging.join(i.to_string()), &to)
                    .map_err(|e| format!("could not publish {}: {e}", to.display()))?;
            }
            Ok(total)
        })();
        fs::remove_dir_all(&staging).ok();
        let total = result?;
        println!(
            "{what}: {keyword} release {}, downloaded and verified {} file(s), {} bytes -> {}",
            release.id(),
            files.len(),
            total,
            local.display()
        );
        Ok(())
    }
}

/// Delays before each retry of a transport failure or a 5xx.
const RETRY_DELAYS_MS: &[u64] = &[200, 500, 1000];

/// The origin could not be reached, and why.
struct Unreachable(String);

enum ManifestError {
    Unreachable(String),
    Invalid(String),
}

/// A reqwest error with its causes, which is where "dns error" or "timed out"
/// actually is.
fn describe(e: &reqwest::Error) -> String {
    let mut text = e.to_string();
    let mut source = std::error::Error::source(e);
    while let Some(s) = source {
        text.push_str(": ");
        text.push_str(&s.to_string());
        source = s.source();
    }
    text
}

/// The manifest entries `sections` of `dir` resolve to. An optional section
/// the manifest does not list is absent, with nothing to fetch; a required one
/// it does not list means the nuclide (or element, or subsection) is not in
/// the release at all, and the error lists what is.
fn needed_files<'a>(
    keyword: &str,
    release: &'a CachedRelease,
    dir: &str,
    sections: &[(&str, bool)],
    what: &str,
) -> Result<Vec<&'a ManifestFile>, String> {
    let mut needed = Vec::with_capacity(sections.len());
    for (name, required) in sections {
        let path = format!("{dir}/{name}");
        match release.release.file(&path) {
            Some(file) => needed.push(file),
            None if *required => {
                return Err(not_in_release_error(keyword, release, dir, what, &path))
            }
            None => {}
        }
    }
    Ok(needed)
}

fn not_in_release_error(
    keyword: &str,
    release: &CachedRelease,
    dir: &str,
    what: &str,
    path: &str,
) -> String {
    let parent = dir.rsplit_once('/').map_or("", |(p, _)| p);
    if release.release.has_dir(dir) {
        return format!(
            "'{keyword}' release {} publishes {dir} without its required file {path}",
            release.id()
        );
    }
    let available = release.release.names_under(parent);
    let listing = if available.is_empty() {
        format!("It publishes no {parent} data at all.")
    } else {
        format!("Available under {parent}/:\n{}", available.join(", "))
    };
    format!(
        "{what} is not available in '{keyword}' (release {}).\n\n{listing}",
        release.id()
    )
}

fn offline_missing_error(
    keyword: &str,
    reason: &str,
    release: Option<&CachedRelease>,
    dir: &str,
    sections: &[(&str, bool)],
    what: &str,
    root: &Path,
) -> String {
    let Some(release) = release else {
        return format!(
            "{what} from '{keyword}' is needed, the nuclear-data origin could not be reached \
             ({reason}), and the cache holds no release of '{keyword}' (looked in {}). Connect \
             to the network once to download it, or point the data source at a local directory.",
            root.join(keyword).display()
        );
    };
    let missing = match needed_files(keyword, release, dir, sections, what) {
        Ok(files) => files
            .into_iter()
            .filter(|f| !release.dir.join(&f.path).is_file())
            .map(|f| f.path.clone())
            .collect::<Vec<_>>()
            .join(", "),
        Err(e) => e,
    };
    format!(
        "{what} from '{keyword}' is needed and no connection to the nuclear-data origin was \
         available ({reason}). The cached release {} does not hold {missing} (in {}). Connect \
         to the network once to download it, or point the data source at a local directory.",
        release.id(),
        release.dir.display()
    )
}

fn not_published_error(keyword: &str, url: &str) -> String {
    format!(
        "'{keyword}' has not been published in the release layout yet: {url} answers 404. This \
         build of yamc reads only versioned releases (<keyword>/<release>/ with a manifest), and \
         the origin still serves only the older unversioned layout for this library. Use a yamc \
         release that reads that layout, or point the data source at a local directory holding \
         the library (e.g. yamc.cross_section_data = \"/path/to/library\")."
    )
}

/// Every cached release folder of a library that carries a readable manifest
/// of a format this build reads, newest first.
///
/// Newest by release identifier, which the publishing scripts write as an ISO
/// date (`2026-10-01`), so it sorts lexically. A folder without a manifest is
/// not a release this module wrote and is ignored.
fn cached_releases(keyword: &str, library_dir: &Path) -> Vec<CachedRelease> {
    let Ok(entries) = fs::read_dir(library_dir) else {
        return Vec::new();
    };
    let mut found: Vec<CachedRelease> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let dir = entry.path();
            let bytes = fs::read(dir.join(MANIFEST)).ok()?;
            let release = parse_manifest(keyword, &bytes, None).ok()?;
            (dir.file_name()?.to_str()? == release.manifest.release)
                .then_some(CachedRelease { release, dir })
        })
        .collect();
    found.sort_by(|a, b| b.id().cmp(a.id()));
    found
}

fn log_from_cache(keyword: &str, release: &str, what: &str, local: &Path) {
    if crate::load_logging_enabled() {
        println!(
            "{what}: {keyword} release {release}, from cache {}",
            local.display()
        );
    }
}

/// A fresh private directory under `parent`, on the same filesystem so a
/// rename out of it is atomic.
fn tempdir_in(parent: &Path) -> Result<PathBuf, String> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = parent.join(format!(".staging-{}-{n}", std::process::id()));
    if dir.exists() {
        fs::remove_dir_all(&dir).ok();
    }
    fs::create_dir_all(&dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    Ok(dir)
}

/// Write `bytes` to `path` through a temporary file and a rename.
fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().ok_or(std::io::ErrorKind::InvalidInput)?;
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
        std::process::id()
    ));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)
}

/// Stream `url` into `dest`, counting and hashing as it arrives, and keep it
/// only if it is the file the manifest describes.
///
/// Transport failures and 5xx answers are retried, since a fresh cache issues
/// hundreds of requests and the occasional dropped connection succeeds on a
/// second attempt. A body that arrives in full and does not match is not: the
/// origin is serving the wrong bytes, and asking again would only make the
/// error slower.
fn fetch_verified(
    client: &reqwest::blocking::Client,
    url: &str,
    expected: &ManifestFile,
    dest: &Path,
) -> Result<(), String> {
    let mut last = String::new();
    for &delay_ms in std::iter::once(&0u64).chain(RETRY_DELAYS_MS.iter()) {
        if delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(delay_ms));
        }
        let mut response = match client.get(url).send() {
            Ok(r) => r,
            Err(e) => {
                last = describe(&e);
                continue;
            }
        };
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(format!(
                "{url}: 404, but the release manifest lists {} ({} bytes)",
                expected.path, expected.bytes
            ));
        }
        if !status.is_success() {
            last = format!("HTTP {status}");
            continue;
        }
        match stream_to_file(&mut response, dest) {
            Ok((bytes, sha256)) => {
                return verify(
                    &format!("{url} ({})", expected.path),
                    expected,
                    bytes,
                    &sha256,
                )
                .inspect_err(|_| {
                    fs::remove_file(dest).ok();
                })
            }
            Err(e) => last = format!("reading the response body: {e}"),
        }
    }
    Err(format!(
        "Failed to download {url} after {} attempts: {last}",
        RETRY_DELAYS_MS.len() + 1
    ))
}

/// Copy a body to `dest` in chunks, hashing as it goes, and fsync it.
fn stream_to_file(body: &mut impl Read, dest: &Path) -> std::io::Result<(u64, String)> {
    let mut file = fs::File::create(dest)?;
    let mut hash = StreamHash::default();
    let mut buffer = vec![0u8; 256 * 1024];
    loop {
        let n = body.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        file.write_all(&buffer[..n])?;
    }
    file.sync_all()?;
    Ok(hash.finish())
}

/// The release-relative folder of a nuclide's or element's sections.
pub(crate) fn particle_dir(kind: DataKind, name: &str) -> String {
    match kind {
        DataKind::Neutron => format!("neutron/{name}.arrow"),
        DataKind::Photon => format!("photon/{name}.arrow"),
    }
}

/// Resolve `name` of `keyword` to a local folder holding the sections `scope`
/// needs.
pub(crate) fn fetch_particle(
    registry: &Registry,
    keyword: &str,
    name: &str,
    kind: DataKind,
    scope: &crate::load_scope::LoadScope,
) -> Result<PathBuf, String> {
    let what = match kind {
        DataKind::Neutron => format!("Nuclide '{name}'"),
        DataKind::Photon => format!("Photon data for '{name}'"),
    };
    registry.fetch_dir(
        keyword,
        &particle_dir(kind, name),
        &sections_for(kind, scope),
        &what,
    )
}

/// Resolve one transmutation subsection of `keyword` to a local folder.
pub(crate) fn fetch_subsection(
    registry: &Registry,
    keyword: &str,
    subsection: &str,
) -> Result<PathBuf, String> {
    let sections = transmutation_sections(subsection);
    if sections.is_empty() {
        return Err(format!("Unknown transmutation subsection '{subsection}'"));
    }
    let what = format!("The '{subsection}' transmutation subsection");
    registry
        .fetch_dir(
            keyword,
            &format!("transmutation/{subsection}.arrow"),
            sections,
            &what,
        )
        .map_err(|e| {
            if e.starts_with(&what) && e.contains("is not available") {
                format!(
                    "Library '{keyword}' does not provide a '{subsection}' transmutation \
                     subsection. Point this transmutation part at a library that provides \
                     '{subsection}' (endf-b8.1 provides all parts) or at a local path.\n\n{e}"
                )
            } else {
                e
            }
        })
}

#[cfg(test)]
mod tests;
