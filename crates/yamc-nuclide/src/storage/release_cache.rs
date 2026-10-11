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
//! # Byte-range fetches
//!
//! A load that names its MTs (activation) or its temperatures (transport at a
//! material temperature) reads a few batches of `reactions.arrow` and
//! `energy.arrow`, which are most of a nuclide's bytes. Those are fetched as
//! byte ranges when they can be verified, which is when the nuclide's
//! `version.json` gives a sha256 for every index entry the load needs (see
//! `nuclear_data_schema::reaction_ranges`). `version.json` is downloaded and
//! verified against the manifest first, so its hashes are as trustworthy as
//! the manifest's own. Each entry received is checked against its hash before
//! anything is cached, and a mismatch is a hard error naming the nuclide, the
//! file, the MT or temperature, and both hashes, not retried.
//!
//! The ranged copy is spliced and cached exactly as the raw-URL path caches
//! one (`url_cache::fetch_ranged`): under `temperatures/` or `subset/` in the
//! nuclide's folder, with a record of the labels or MTs it holds. Those paths
//! are not manifest paths, so a ranged copy is never taken for the whole
//! verified file, and the whole file, once downloaded for a load that needs
//! it, supersedes the copy.
//!
//! A release whose index carries no hash for some entry a load needs (a
//! library indexed before hashes existed) is fetched whole for that section,
//! verified against the manifest, as before. So is an origin that ignores the
//! `Range` header: the whole object it sends is checked against the manifest.
//!
//! # Prefetched replies
//!
//! A load of many nuclides calls `url_cache::prefetch` first, which asks
//! [`Registry::pending`] what each nuclide's download would fetch, fetches all
//! of it a batch of nuclides at a time with many requests in flight, and holds
//! the replies. The download here then takes a held reply in place of each
//! request it would have sent, and checks it exactly as it checks a reply from
//! the network: a whole file against the manifest as it is written, a range
//! against the index. A request with no held reply goes to the network.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use once_cell::sync::Lazy;
#[cfg(not(target_os = "emscripten"))]
use once_cell::sync::OnceCell;

use super::release::{
    parse_latest, parse_manifest, verify, DataRelease, LatestPointer, ManifestFile, Release,
    StreamHash, LATEST, MANIFEST,
};
use super::url_cache::{
    cache_root, drop_superseded, fetch_ranged, get_path_lock, ranged, ranges_section,
    section_covered, sections_for, transmutation_sections, DataKind, Ranged,
};

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
#[cfg(not(target_os = "emscripten"))]
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

/// What a download goes through: an HTTP client, or in a build with no
/// sockets of its own, the fetcher the host installed
/// ([`super::url_cache::set_host_fetcher`]).
#[cfg(not(target_os = "emscripten"))]
type Client = reqwest::blocking::Client;
#[cfg(target_os = "emscripten")]
struct Client;

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
    #[cfg(not(target_os = "emscripten"))]
    client: OnceCell<reqwest::Result<Client>>,
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
            #[cfg(not(target_os = "emscripten"))]
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

    #[cfg(not(target_os = "emscripten"))]
    fn client(&self) -> Result<&Client, String> {
        self.client
            .get_or_init(|| build_client(self.timeouts))
            .as_ref()
            .map_err(|e| format!("could not build an HTTP client: {e}"))
    }

    #[cfg(target_os = "emscripten")]
    fn client(&self) -> Result<&Client, String> {
        if super::url_cache::host_fetcher_installed() {
            Ok(&Client)
        } else {
            Err("no data fetcher is installed for this build; \
                 the host must call set_host_fetcher before downloading"
                .to_string())
        }
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
            let timeout = pointer.then_some(self.timeouts.pointer);
            match get_small_once(client, url, timeout) {
                Ok((404, _)) => return Ok(None),
                Ok((200..=299, body)) => return Ok(Some(body)),
                Ok((status, _)) => last = format!("{url}: HTTP {status}"),
                Err(e) => last = e,
            }
        }
        Err(Unreachable(last))
    }

    /// What a load of `name` at `scope` would download from `keyword` now, or
    /// `None` when it would download nothing: every file is cached, the
    /// release does not publish the nuclide, or the origin is unreachable.
    ///
    /// Resolves the library if nothing has yet, as the download would first.
    /// For `url_cache::prefetch`, which fetches the files ahead of the
    /// download.
    pub(crate) fn pending(
        &self,
        keyword: &str,
        name: &str,
        kind: DataKind,
        scope: &crate::load_scope::LoadScope,
    ) -> Option<Pending> {
        let library = self.library(keyword).ok()?;
        let Library::Online { release, base_url } = &*library else {
            return None;
        };
        let dir = particle_dir(kind, name);
        let ranging = ranged(kind, scope);
        let needed = needed_files(keyword, release, &dir, &sections_for(kind, scope), name).ok()?;
        let sections: Vec<String> = needed
            .into_iter()
            .filter(|f| !held(release, f, ranging))
            .map(|f| section_name(f).to_string())
            .collect();
        (!sections.is_empty()).then(|| Pending {
            url: format!("{base_url}{dir}"),
            local: release.dir.join(&dir),
            sections,
        })
    }

    /// The local folder holding `sections` of `dir` (`neutron/Fe56.arrow`,
    /// `transmutation/decay.arrow`, ...) of `keyword`'s release, downloading
    /// and verifying whatever the cache lacks. `ranging` is which MTs or
    /// temperatures of the indexed sections the load reads, if only some.
    /// `what` names the data in messages.
    pub(crate) fn fetch_dir(
        &self,
        keyword: &str,
        dir: &str,
        sections: &[(&str, bool)],
        ranging: Ranged<'_>,
        what: &str,
    ) -> Result<PathBuf, String> {
        let library = self.library(keyword)?;
        match &*library {
            Library::Online { release, base_url } => {
                let needed = needed_files(keyword, release, dir, sections, what)?;
                let local = release.dir.join(dir);
                self.record_use(keyword, release.record(false));
                if needed.iter().all(|f| held(release, f, ranging)) {
                    log_from_cache(keyword, release.id(), what, &local);
                    return Ok(local);
                }
                let lock = get_path_lock(&local);
                let _guard = lock.lock().unwrap_or_else(|p| p.into_inner());
                let missing: Vec<&ManifestFile> = needed
                    .into_iter()
                    .filter(|f| !held(release, f, ranging))
                    .collect();
                if !missing.is_empty() {
                    self.download(keyword, release, base_url, dir, &missing, ranging, what)?;
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
                            .is_ok_and(|files| files.iter().all(|f| held(c, f, ranging)))
                    });
                    let Some(pick) = pick else {
                        return Err(offline_missing_error(
                            keyword,
                            reason,
                            candidates.first().map(|c| &**c),
                            dir,
                            sections,
                            ranging,
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
                if needed.iter().any(|f| !held(release, f, ranging)) {
                    return Err(offline_missing_error(
                        keyword,
                        reason,
                        Some(release),
                        dir,
                        sections,
                        ranging,
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

    /// Download `files` of `dir` into the release folder, every byte verified.
    ///
    /// The sections `ranging` reads only some of go last, as byte ranges where
    /// the index lets them be verified: that needs the nuclide's
    /// `version.json`, which is among the others when it is not cached yet and
    /// so is in place, verified, by then. Whatever cannot be ranged is
    /// downloaded whole.
    #[allow(clippy::too_many_arguments)]
    fn download(
        &self,
        keyword: &str,
        release: &CachedRelease,
        base_url: &str,
        dir: &str,
        files: &[&ManifestFile],
        ranging: Ranged<'_>,
        what: &str,
    ) -> Result<(), String> {
        let local = release.dir.join(dir);
        let (ranged_files, mut whole): (Vec<&ManifestFile>, Vec<&ManifestFile>) = files
            .iter()
            .copied()
            .partition(|f| ranges_section(section_name(f), ranging));
        self.download_whole(keyword, release, base_url, &whole, what, &local)?;

        whole.clear();
        for file in ranged_files {
            if !self.download_ranged(keyword, release, base_url, dir, file, ranging, what)? {
                whole.push(file);
            }
        }
        self.download_whole(keyword, release, base_url, &whole, what, &local)
    }

    /// Fetch the byte ranges of `file` that `ranging` reads, verify each
    /// against the index in the cached `version.json`, and publish the spliced
    /// copy into the nuclide's folder.
    ///
    /// `Ok(false)` when the file is to be downloaded whole instead (see
    /// [`fetch_ranged`]); nothing has been written then.
    #[allow(clippy::too_many_arguments)]
    fn download_ranged(
        &self,
        keyword: &str,
        release: &CachedRelease,
        base_url: &str,
        dir: &str,
        file: &ManifestFile,
        ranging: Ranged<'_>,
        what: &str,
    ) -> Result<bool, String> {
        self.client()?;
        let local = release.dir.join(dir);
        let name = dir
            .rsplit('/')
            .next()
            .and_then(|n| n.strip_suffix(".arrow"))
            .unwrap_or(dir);
        let section = section_name(file);
        let staging = tempdir_in(&release.dir)?;
        let result = (|| -> Result<bool, String> {
            let Some(fetched) = fetch_ranged(
                &format!("{base_url}{dir}"),
                &staging,
                &local,
                section,
                ranging,
                name,
                Some(file),
            )
            .map_err(|e| e.to_string())?
            else {
                return Ok(false);
            };
            for path in &fetched.staged {
                let to = local.join(path);
                if let Some(parent) = to.parent() {
                    fs::create_dir_all(parent)
                        .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
                }
                fs::rename(staging.join(path), &to)
                    .map_err(|e| format!("could not publish {}: {e}", to.display()))?;
            }
            drop_superseded(&local, &fetched.staged);
            // A prefetch counts nuclides instead.
            if !super::url_cache::batch::quiet() {
                if fetched.is_whole() {
                    println!(
                        "{what}: {keyword} release {}, the origin answered a range request for \
                         {section} with the whole file; downloaded and verified it, {} bytes -> {}",
                        release.id(),
                        fetched.bytes,
                        local.display()
                    );
                } else {
                    println!(
                        "{what}: {keyword} release {}, downloaded and verified {} byte range(s) of \
                         {section}, {} of its {} bytes -> {}",
                        release.id(),
                        fetched.spans,
                        fetched.bytes,
                        file.bytes,
                        local.display()
                    );
                }
            }
            Ok(true)
        })();
        fs::remove_dir_all(&staging).ok();
        result
    }

    /// Download `files` whole into the release folder, each verified, then
    /// publish them by rename.
    fn download_whole(
        &self,
        keyword: &str,
        release: &CachedRelease,
        base_url: &str,
        files: &[&ManifestFile],
        what: &str,
        local: &Path,
    ) -> Result<(), String> {
        if files.is_empty() {
            return Ok(());
        }
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
        // A section now held whole supersedes any ranged copy of it.
        let names: Vec<String> = files.iter().map(|f| section_name(f).to_string()).collect();
        drop_superseded(local, &names);
        // A prefetch counts nuclides instead.
        if !super::url_cache::batch::quiet() {
            println!(
                "{what}: {keyword} release {}, downloaded and verified {} file(s), {} bytes -> {}",
                release.id(),
                files.len(),
                total,
                local.display()
            );
        }
        Ok(())
    }
}

/// The files a keyword load of one nuclide or element would download: see
/// [`Registry::pending`].
pub(crate) struct Pending {
    /// `<origin>/<keyword>/<release>/<dir>`: each file is `<url>/<section>`.
    pub url: String,
    /// The release folder's `<dir>` in the cache, which the files go into.
    pub local: PathBuf,
    /// The file names (`version.json`, `reactions.arrow`, ...) the cache lacks.
    /// An indexed one is fetched as byte ranges when the load reads only some
    /// of it and the index allows.
    pub sections: Vec<String>,
}

/// The file name of a manifest entry: `reactions.arrow` for
/// `neutron/Fe56.arrow/reactions.arrow`.
fn section_name(file: &ManifestFile) -> &str {
    file.path.rsplit('/').next().unwrap_or(&file.path)
}

/// Whether `release`'s cache folder already holds `file` as far as a load
/// reading `ranging` of it needs: the whole verified file, or for a section
/// `ranging` reads only some of, a ranged copy covering the load.
fn held(release: &CachedRelease, file: &ManifestFile, ranging: Ranged<'_>) -> bool {
    let path = release.dir.join(&file.path);
    if path.is_file() {
        return true;
    }
    let section = section_name(file);
    ranges_section(section, ranging)
        && path
            .parent()
            .is_some_and(|dir| section_covered(dir, section, ranging))
}

/// Delays before each retry of a transport failure or a 5xx.
const RETRY_DELAYS_MS: &[u64] = &[200, 500, 1000];

/// The origin could not be reached, and why.
struct Unreachable(String);

enum ManifestError {
    Unreachable(String),
    Invalid(String),
}

/// One GET of a small object: the status, and the body of a success. A
/// transport failure, or a body cut short, is the error.
#[cfg(not(target_os = "emscripten"))]
fn get_small_once(
    client: &Client,
    url: &str,
    timeout: Option<Duration>,
) -> Result<(u16, Vec<u8>), String> {
    if let Some(reply) = super::url_cache::take_prefetched(url) {
        let (status, body) = reply?;
        return Ok((status, if is_success(status) { body } else { Vec::new() }));
    }
    let mut request = client.get(url);
    if let Some(timeout) = timeout {
        request = request.timeout(timeout);
    }
    let response = request.send().map_err(|e| describe(&e))?;
    let status = response.status();
    if !status.is_success() {
        return Ok((status.as_u16(), Vec::new()));
    }
    let body = response
        .bytes()
        .map_err(|e| format!("{url}: reading the body: {}", describe(&e)))?;
    Ok((status.as_u16(), body.to_vec()))
}

/// One GET of a small object through the host's fetcher. The host owns the
/// request, so `timeout` is not applied here.
#[cfg(target_os = "emscripten")]
fn get_small_once(
    _client: &Client,
    url: &str,
    _timeout: Option<Duration>,
) -> Result<(u16, Vec<u8>), String> {
    let (status, body) = super::url_cache::host_get(url)?;
    Ok((status, if is_success(status) { body } else { Vec::new() }))
}

fn is_success(status: u16) -> bool {
    (200..300).contains(&status)
}

/// One GET of `url` into `dest`, hashed as it is written: the status, and for
/// a success the byte count and sha256 of what was written.
///
/// A reply a prefetch holds for `url` is written through the same hashing
/// path, and checked by the caller the same way.
#[cfg(not(target_os = "emscripten"))]
fn get_to_file(
    client: &Client,
    url: &str,
    dest: &Path,
) -> Result<(u16, Option<(u64, String)>), String> {
    if let Some(reply) = super::url_cache::take_prefetched(url) {
        let (status, body) = reply?;
        return body_to_file(status, &body, dest);
    }
    let mut response = client.get(url).send().map_err(|e| describe(&e))?;
    let status = response.status();
    if !status.is_success() {
        return Ok((status.as_u16(), None));
    }
    let written = stream_to_file(&mut response, dest)
        .map_err(|e| format!("reading the response body: {e}"))?;
    Ok((status.as_u16(), Some(written)))
}

/// One GET of `url` into `dest` through the host's fetcher, which answers
/// with the whole body.
#[cfg(target_os = "emscripten")]
fn get_to_file(
    _client: &Client,
    url: &str,
    dest: &Path,
) -> Result<(u16, Option<(u64, String)>), String> {
    let (status, body) = super::url_cache::host_get(url)?;
    body_to_file(status, &body, dest)
}

/// Write a body already in memory into `dest` as [`get_to_file`] writes one
/// it streams: the status, and for a success the byte count and sha256.
fn body_to_file(
    status: u16,
    body: &[u8],
    dest: &Path,
) -> Result<(u16, Option<(u64, String)>), String> {
    if !is_success(status) {
        return Ok((status, None));
    }
    let written = stream_to_file(&mut &body[..], dest)
        .map_err(|e| format!("writing {}: {e}", dest.display()))?;
    Ok((status, Some(written)))
}

/// A reqwest error with its causes, which is where "dns error" or "timed out"
/// actually is.
#[cfg(not(target_os = "emscripten"))]
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

#[allow(clippy::too_many_arguments)]
fn offline_missing_error(
    keyword: &str,
    reason: &str,
    release: Option<&CachedRelease>,
    dir: &str,
    sections: &[(&str, bool)],
    ranging: Ranged<'_>,
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
            .filter(|f| !held(release, f, ranging))
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
    client: &Client,
    url: &str,
    expected: &ManifestFile,
    dest: &Path,
) -> Result<(), String> {
    let mut last = String::new();
    for &delay_ms in std::iter::once(&0u64).chain(RETRY_DELAYS_MS.iter()) {
        if delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(delay_ms));
        }
        match get_to_file(client, url, dest) {
            Ok((404, _)) => {
                return Err(format!(
                    "{url}: 404, but the release manifest lists {} ({} bytes)",
                    expected.path, expected.bytes
                ))
            }
            Ok((_, Some((bytes, sha256)))) => {
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
            Ok((status, None)) => last = format!("HTTP {status}"),
            Err(e) => last = e,
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
        ranged(kind, scope),
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
            Ranged::Whole,
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
