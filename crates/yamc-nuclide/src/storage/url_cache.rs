#[cfg(feature = "download")]
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;

#[cfg(feature = "download")]
use std::fs;

#[cfg(feature = "download")]
pub(super) mod batch;

#[cfg(feature = "download")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "download")]
use once_cell::sync::Lazy;

#[cfg(feature = "download")]
use nuclear_data_schema::energy_ranges::EnergyRanges;
#[cfg(feature = "download")]
use nuclear_data_schema::reaction_ranges::{splice_spans, Range, ReactionRanges};

/// Per-cache-path mutexes to serialize concurrent downloads of the same nuclide.
/// Without this, parallel rayon threads loading the same nuclide would each issue
/// an HTTP request, wasting bandwidth and triggering transient 404s from GitHub's
/// release CDN under burst concurrency.
#[cfg(feature = "download")]
static DOWNLOAD_LOCKS: Lazy<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "download")]
pub(crate) fn get_path_lock(path: &std::path::Path) -> Arc<Mutex<()>> {
    let mut locks = DOWNLOAD_LOCKS.lock().unwrap_or_else(|p| p.into_inner());
    locks
        .entry(path.to_path_buf())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Scheme + host of the hosted nuclear-data CDN.
///
/// HTTPS when a TLS provider is compiled in (`download-tls`, the default in
/// wheels and dev builds). Plain HTTP in the crypto-free `download`-only build,
/// which has no TLS backend at all and is the pure-Rust fallback for
/// architectures graviola does not support and for minimal C-toolchain-free
/// containers. Kept as a compile-time literal so the URLs stay `&'static str`.
#[cfg(any(feature = "download-tls", target_os = "emscripten"))]
macro_rules! data_origin {
    () => {
        "https://yamc-data.xsplot.com/"
    };
}
#[cfg(all(
    feature = "download",
    not(feature = "download-tls"),
    not(target_os = "emscripten")
))]
macro_rules! data_origin {
    () => {
        "http://yamc-data.xsplot.com/"
    };
}

/// The origin every library keyword resolves against (see [`data_origin`]).
#[cfg(feature = "download")]
pub(crate) const ORIGIN: &str = data_origin!();

/// Install graviola as the process-wide rustls crypto provider exactly once,
/// before the first reqwest client is built. reqwest is compiled with
/// `rustls-tls-webpki-roots-no-provider`, so it uses the process default
/// provider and panics at client construction if none is installed. graviola is
/// pure Rust, so this keeps HTTPS working with no C toolchain (unlike ring).
///
/// graviola supports x86_64 + aarch64 only, which covers every published wheel
/// and mainstream target. On other architectures (POWER9, RISC-V, 32-bit ARM,
/// ...) build without `download-tls` for the crypto-free HTTP-only path, or use
/// local data files.
///
/// FUTURE: to get pure-Rust HTTPS on those arches too, swap the provider here
/// for `rustls-rustcrypto` once it matures past its current 0.0.x-alpha and
/// drops the `rsa` crate (RUSTSEC-2023-0071 "Marvin" advisory), or once graviola
/// grows more target arches. Until one of those lands, full-arch HTTPS (incl.
/// POWER9 / RISC-V) requires ring, which is why the ring-based build is kept.
#[cfg(feature = "download-tls")]
pub(crate) fn ensure_tls_provider() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // Err only if a provider was already installed by the host application
        // (e.g. a Rust app embedding yamc); either way a usable default is
        // present afterwards, so ignore the result.
        let _ = rustls_graviola::default_provider().install_default();
    });
}

/// One client for the process, built on first use.
///
/// `reqwest::blocking::get` constructs and drops a `Client` per call, so every
/// section paid a fresh TCP and TLS handshake. That was amortized over a 12 MB
/// object and is not amortized over a 40 kB byte range: measured against the
/// CDN, twenty small ranges cost 20.4 ms each over fresh connections and 8.3 ms
/// each over one. The free function also exposes no builder, so a `Range`
/// header could not be attached to it at all.
///
/// A failure to build is kept rather than retried: it means the TLS stack could
/// not be initialized, which the next call will not fix, and reporting it per
/// request keeps the error at the fetch that needed it.
///
/// Connect and read timeouts are the ones every download uses
/// ([`super::release_cache::Timeouts::DEFAULT`]), so a stalled origin fails a
/// fetch rather than hanging it.
#[cfg(all(feature = "download", not(target_os = "emscripten")))]
static CLIENT: Lazy<reqwest::Result<reqwest::blocking::Client>> =
    Lazy::new(|| super::release_cache::build_client(super::release_cache::Timeouts::DEFAULT));

/// The shared client, or the error it failed to build with.
#[cfg(all(feature = "download", not(target_os = "emscripten")))]
fn client() -> Result<&'static reqwest::blocking::Client, Box<dyn std::error::Error>> {
    CLIENT
        .as_ref()
        .map_err(|e| format!("could not build an HTTP client: {e}").into())
}

/// One GET as the transport sees it: the URL, and the `Range` header value to
/// send with it (`bytes=0-99,200-299`), or `None` for the whole object.
#[cfg(feature = "download")]
type Get = (String, Option<String>);

/// Issue a blocking GET, optionally for byte ranges.
///
/// `spans` are `(offset, length)` pairs, all named in one `Range` header; each
/// is inclusive of both ends, so its last byte is `offset + length - 1`. Empty
/// asks for the whole object. More than one span is answered with a
/// `multipart/byteranges` body, which [`parse_byteranges`] reads.
///
/// A reply [`prefetch`] already fetched for exactly this request is handed
/// over instead of asking again, once: a retry after it goes to the network.
#[cfg(feature = "download")]
fn blocking_get(url: &str, spans: &[Range]) -> Result<Reply, Box<dyn std::error::Error>> {
    let range = (!spans.is_empty()).then(|| range_header(spans));
    let get = (url.to_string(), range);
    if let Some(reply) = batch::take(&get) {
        return Ok(reply?);
    }
    let reply = send_many(std::slice::from_ref(&get)).pop();
    Ok(reply.ok_or("the fetcher answered no requests")??)
}

/// Send one GET over the shared client.
#[cfg(all(feature = "download", not(target_os = "emscripten")))]
fn send(url: &str, range: Option<&str>) -> Result<Reply, String> {
    let mut request = client().map_err(|e| e.to_string())?.get(url);
    if let Some(range) = range {
        request = request.header(reqwest::header::RANGE, range);
    }
    let response = request.send().map_err(|e| e.to_string())?;
    let status = response.status().as_u16();
    // Only a success carries a body worth reading.
    let body = if response.status().is_success() {
        response
            .bytes()
            .map(|b| b.to_vec())
            .map_err(|e| format!("reading the response body: {e}"))
    } else {
        Ok(Vec::new())
    };
    Ok(Reply { status, body })
}

/// The most GETs a native batch has in flight at once.
#[cfg(all(feature = "download", not(target_os = "emscripten")))]
const NATIVE_CONCURRENCY: usize = 8;

/// Send every one of `gets`, [`NATIVE_CONCURRENCY`] at a time, and answer them
/// in the order asked. One failing does not fail the others.
#[cfg(all(feature = "download", not(target_os = "emscripten")))]
fn send_many(gets: &[Get]) -> Vec<Result<Reply, String>> {
    if let [(url, range)] = gets {
        return vec![send(url, range.as_deref())];
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let replies: Vec<Mutex<Option<Result<Reply, String>>>> =
        gets.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|threads| {
        for _ in 0..NATIVE_CONCURRENCY.min(gets.len()) {
            threads.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some((url, range)) = gets.get(i) else {
                    break;
                };
                let reply = send(url, range.as_deref());
                *replies[i].lock().unwrap_or_else(|p| p.into_inner()) = Some(reply);
            });
        }
    });
    replies
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .unwrap_or_else(|p| p.into_inner())
                .unwrap_or_else(|| Err("the request was never sent".to_string()))
        })
        .collect()
}

/// What one GET came back with, before retry policy is applied.
#[cfg(feature = "download")]
struct Reply {
    status: u16,
    body: Result<Vec<u8>, String>,
}

/// The GETs supplied by the host, for builds with no socket of their own.
///
/// Called with a list of `(url, range)` requests, `range` being the `Range`
/// header value to send (`bytes=0-99,200-299`) or `None` for the whole object,
/// and answers one `(status, body)` per request, in the same order. Status 0
/// means the request did not complete (offline, refused), which is retried
/// like a 5xx. The requests in one call are independent and the host should
/// have them in flight together: that is what [`prefetch`] batches them for.
/// In the browser the fetcher hands them to a helper worker, which has them in
/// flight together with async `fetch()`, and blocks on `Atomics.wait` until
/// all are answered, so the Range planning, temperature
/// selection and caching above it are the same code a native build runs.
#[cfg(all(feature = "download", target_os = "emscripten"))]
pub type HostFetcher =
    dyn Fn(&[(String, Option<String>)]) -> Result<Vec<(u16, Vec<u8>)>, String> + Send + Sync;

#[cfg(all(feature = "download", target_os = "emscripten"))]
static HOST_FETCHER: std::sync::RwLock<Option<Box<HostFetcher>>> = std::sync::RwLock::new(None);

/// Install the fetcher every download goes through on this target.
#[cfg(all(feature = "download", target_os = "emscripten"))]
pub fn set_host_fetcher(fetcher: Box<HostFetcher>) {
    *HOST_FETCHER.write().unwrap_or_else(|e| e.into_inner()) = Some(fetcher);
}

/// Whether the host has installed a fetcher yet.
#[cfg(all(feature = "download", target_os = "emscripten"))]
pub(super) fn host_fetcher_installed() -> bool {
    HOST_FETCHER
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .is_some()
}

/// A whole-object GET through the host's fetcher, as a batch of one: the
/// status and the body. A reply [`prefetch`] holds for it is used instead.
#[cfg(all(feature = "download", target_os = "emscripten"))]
pub(super) fn host_get(url: &str) -> Result<(u16, Vec<u8>), String> {
    let reply = blocking_get(url, &[]).map_err(|e| e.to_string())?;
    Ok((reply.status, reply.body?))
}

/// The reply [`prefetch`] holds for a whole-object GET of `url`, if it holds
/// one: the status and the body. Taken, so a retry goes to the network.
///
/// For the release downloads, which make their own requests over the
/// registry's client on this target rather than through [`blocking_get`].
#[cfg(all(feature = "download", not(target_os = "emscripten")))]
pub(super) fn take_prefetched(url: &str) -> Option<Result<(u16, Vec<u8>), String>> {
    let reply = batch::take(&(url.to_string(), None))?;
    Some(reply.and_then(|r| Ok((r.status, r.body?))))
}

#[cfg(all(feature = "download", target_os = "emscripten"))]
fn client() -> Result<(), Box<dyn std::error::Error>> {
    match HOST_FETCHER
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
    {
        Some(_) => Ok(()),
        None => Err("no data fetcher is installed for this build; \
                     the host must call set_host_fetcher before downloading"
            .into()),
    }
}

/// Hand every one of `gets` to the host fetcher in one call.
#[cfg(all(feature = "download", target_os = "emscripten"))]
fn send_many(gets: &[Get]) -> Vec<Result<Reply, String>> {
    let answered = {
        let guard = HOST_FETCHER.read().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(fetcher) => fetcher(gets),
            None => Err("no data fetcher is installed".to_string()),
        }
    };
    match answered {
        Ok(replies) if replies.len() == gets.len() => replies
            .into_iter()
            .zip(gets)
            .map(|((status, bytes), (url, _))| match status {
                0 => Err(format!("the request for {url} did not complete")),
                status => Ok(Reply {
                    status,
                    body: Ok(bytes),
                }),
            })
            .collect(),
        Ok(replies) => {
            let e = format!(
                "the data fetcher answered {} requests of {}",
                replies.len(),
                gets.len()
            );
            gets.iter().map(|_| Err(e.clone())).collect()
        }
        Err(e) => gets.iter().map(|_| Err(e.clone())).collect(),
    }
}

/// One nuclide's (or element's) data a caller is about to load: the source as
/// the config names it (a library keyword, a URL, a directory or a path), the
/// name, which particle's data, and the scope it will be loaded at.
#[derive(Debug, Clone, Copy)]
pub struct DataRequest<'a> {
    pub source: &'a str,
    pub name: &'a str,
    pub kind: DataKind,
    pub scope: &'a crate::load_scope::LoadScope,
}

/// Download, into the cache, whatever `requests` will need that it does not
/// already hold, with many requests in flight at once.
///
/// A load downloads its nuclides one at a time and each nuclide's files one at
/// a time, which is a round trip per request. A transmutation closure of a few
/// hundred nuclides is thousands of those, and in the browser, where each is a
/// synchronous request made in turn, that took minutes. Called before such a
/// loop, this fetches the same data in two rounds per batch of nuclides: every
/// request that needs nothing first (`version.json` and the files fetched
/// whole), then every byte-range request the indexes in those `version.json`
/// files plan. Each round goes out concurrently, and the nuclides are then
/// cached by the ordinary download path, which finds its requests already
/// answered.
///
/// A library keyword is resolved to its current release first, once (its
/// `latest.json` and manifest), and planned from that release's manifest: the
/// files the cache lacks, and the byte ranges of the indexed sections where
/// the index has a hash for every entry the load needs, exactly as the release
/// download decides. That download verifies every prefetched byte as it
/// verifies bytes from the network, against the manifest or the index.
///
/// Purely an optimisation: nothing is cached here that the loop would not have
/// cached itself, by the same code. A request this plans that the download does
/// not make is dropped, and one the download makes that this did not plan goes
/// to the network as before, as does every retry. Errors are not reported
/// here, because the load that follows meets them again and reports them where
/// it always has. Sources that are not keywords or URLs are skipped.
pub fn prefetch(requests: &[DataRequest<'_>]) {
    #[cfg(feature = "download")]
    batch::prefetch(&super::release_cache::REGISTRY, requests);
    #[cfg(not(feature = "download"))]
    let _ = requests;
}

/// The `Range` header value naming `spans`: `bytes=0-99,200-299`.
#[cfg(feature = "download")]
fn range_header(spans: &[Range]) -> String {
    let ranges: Vec<String> = spans
        .iter()
        .map(|(offset, len)| format!("{}-{}", offset, offset + len - 1))
        .collect();
    format!("bytes={}", ranges.join(","))
}

/// Every recognized library keyword. Each is published on the data origin as
/// immutable release folders under `<keyword>/<release>/` plus a
/// `<keyword>/latest.json` pointer (see [`super::release`]); what a release
/// carries (which nuclides, whether it has photon data, which transmutation
/// subsections) is read from its manifest at run time rather than listed here.
///
/// - `tendl-2025`, `tendl-2017`: neutron cross sections, plus the branching and
///   reactions transmutation subsections. tendl-2017 is library-matched to the
///   TENDL-2017 activation chains, the apples-to-apples transport library for
///   FISPACT-II comparisons.
/// - `fendl-3.2d`: neutron and photon cross sections, plus the branching and
///   reactions transmutation subsections (its MF=8/9/10 isomer production and
///   MF=40 covariances). Like TENDL it has no decay or fission-yield data.
/// - `endf-b8.1`: neutron and photon (with atomic relaxation) cross sections
///   and all four transmutation subsections.
/// - `jeff-4.0`: neutron cross sections and all four transmutation subsections.
///   Its photon sublibrary is photonuclear data, not the photoatomic and
///   relaxation pair the photon loader reads, so it publishes none.
/// - `jendl-5.0`: neutron data (most evaluations to 200 MeV), the EPICS2017
///   photon pair, and all four transmutation subsections.
/// - `irdff-ii`: isomeric branching only (see [`BRANCHING_ONLY_KEYWORDS`]).
pub(crate) const KEYWORDS: &[&str] = &[
    "tendl-2025",
    "tendl-2017",
    "fendl-3.2d",
    "endf-b8.1",
    "jeff-4.0",
    "jendl-5.0",
    "irdff-ii",
];

/// The keywords of libraries that publish isomeric branching and nothing
/// else: no cross sections, and no decay, reactions or fission-yield
/// subsection. Such a library can only be a branching source
/// (`transmutation_branch_ratios`), and naming it as anything else is refused
/// before anything is downloaded.
///
/// - `irdff-ii`: the IAEA International Reactor Dosimetry and Fusion File
///   (IRDFF-II, 2020). Its `branching` subsection is the isomer production it
///   evaluates for dosimetry (MF=10, with the MF=40 covariances), among them
///   In115 `(n,n')` In115_m1, In115 `(n,2n)` In114_m1 and Nb93 `(n,2n)`
///   Nb92_m1, with product levels resolved against ENDF/B-VIII.1 decay data.
///   It is never a default: a run uses it only where it is listed.
pub const BRANCHING_ONLY_KEYWORDS: &[&str] = &["irdff-ii"];

/// Whether `source` is the keyword of a library that publishes isomeric
/// branching only (see [`BRANCHING_ONLY_KEYWORDS`]).
pub fn is_branching_only(source: &str) -> bool {
    BRANCHING_ONLY_KEYWORDS.contains(&source)
}

/// The keywords a cross-section source can name: every keyword but the
/// branching-only ones.
pub fn cross_section_keywords() -> Vec<&'static str> {
    KEYWORDS
        .iter()
        .copied()
        .filter(|k| !is_branching_only(k))
        .collect()
}

/// Why branching-only `keyword` cannot be `role` (e.g. "the neutron
/// cross-section library"), saying what it can be.
pub fn branching_only_error(keyword: &str, role: &str) -> String {
    format!(
        "'{keyword}' supplies isomeric branching only: that library publishes no cross \
         sections and no decay, reactions or fission-yield data, so it cannot be {role}. \
         Name it as a branching source with transmutation_branch_ratios, on its own or in \
         an ordered list such as [\"{keyword}\", \"endf-b8.1\"], where each reaction takes \
         its isomeric split from the first source that evaluates it"
    )
}

/// Which particle's data file to resolve under a library keyword. Different
/// libraries lay out their files differently; the loader uses this to pick
/// the right per-particle subdirectory under `url_stem`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataKind {
    Neutron,
    Photon,
}

/// What a library keyword is asked to be when it is named for `kind`'s data,
/// as a refusal names it.
fn data_role(kind: DataKind) -> &'static str {
    match kind {
        DataKind::Neutron => "the neutron cross-section library",
        DataKind::Photon => "the photon cross-section library",
    }
}

/// Check if a string is a known keyword
pub fn is_keyword(input: &str) -> bool {
    KEYWORDS.contains(&input)
}

/// The library keyword for a library name as a data folder records it. The
/// converter stamps ENDF/B-VIII.1 as `endfb-8.1` in `version.json`, where the
/// keyword is `endf-b8.1`; every other library is stamped as its keyword.
pub fn library_keyword(library: &str) -> &str {
    match library {
        "endfb-8.1" => "endf-b8.1",
        other => other,
    }
}

/// Section files published in each transmutation subsection dir (option-D),
/// `(filename, required)`. The primary section is required; auxiliary sections
/// and `provenance.json` are optional, and absent wherever the release manifest
/// does not list them. Lists the files each
/// subsection can publish; the yani chain loader reads the ones it
/// understands. Returns an empty slice for an unknown subsection (callers gate
/// on the release manifest listing the subsection first).
///
/// `branching_covariance.arrow` is the optional MF=40 covariance of the
/// isomeric branching. A library without one does not list it, so a missing
/// file means "no covariance" rather than an incomplete download.
///
/// `evaluated_yields.arrow` is the tape-literal fission yield evaluation, both
/// MT=454 and MT=459 with DY. A library published without it loads with no
/// evaluated yields; the nominal yields are unaffected.
pub(crate) fn transmutation_sections(subsection: &str) -> &'static [(&'static str, bool)] {
    match subsection {
        "decay" => &[
            ("nuclides.arrow", true),
            ("decay_modes.arrow", false),
            ("sources.arrow", false),
            ("provenance.json", false),
        ],
        "reactions" => &[("reactions.arrow", true), ("provenance.json", false)],
        "fission_yields" => &[
            ("fission_yields.arrow", true),
            ("aliases.arrow", false),
            ("evaluated_yields.arrow", false),
            ("provenance.json", false),
        ],
        "branching" => &[
            ("branching.arrow", true),
            ("branching_covariance.arrow", false),
            ("provenance.json", false),
        ],
        _ => &[],
    }
}

/// Resolve a transmutation subsection source (library keyword or path) to a
/// local directory containing that subsection's arrow files.
///
/// - Keyword (e.g. `"endf-b8.1"`): the subsection folder of the library's
///   current release, `transmutation/{subsection}.arrow/`, downloaded and
///   verified into the cache (see [`super::release_cache`]). A library whose
///   release does not publish the subsection is refused with a message listing
///   the subsections it does publish.
/// - Anything else is a local path: a release folder's
///   `transmutation/{subsection}.arrow/`, a converter root's `{subsection}/`,
///   or the subsection directory itself (see `resolve_local_subsection`).
///
/// A bare name that is both a keyword and a directory is refused.
#[cfg(feature = "download")]
pub fn resolve_subsection(
    source: &str,
    subsection: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    refuse_shadowed_keyword(source)?;
    if is_branching_only(source) && subsection != "branching" {
        return Err(branching_only_error(
            source,
            &format!("the source of the '{subsection}' transmutation subsection"),
        )
        .into());
    }
    if is_keyword(source) {
        Ok(super::release_cache::fetch_subsection(
            &super::release_cache::REGISTRY,
            source,
            subsection,
        )?)
    } else {
        resolve_local_subsection(source, subsection)
    }
}

/// Resolve a transmutation subsection under a local path.
///
/// - A release folder (it holds a release `manifest.json`, see
///   [`super::local_release`]): its `transmutation/{subsection}.arrow/`. A
///   subsection the release does not carry is refused, naming the ones it does.
/// - A converter root (it holds a `{subsection}/` subdir): that subdir.
/// - Anything else is taken to be the subsection directory itself.
fn resolve_local_subsection(
    source: &str,
    subsection: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let base = PathBuf::from(source);
    if let Some(release) = super::local_release::open(&base)? {
        return Ok(release.subsection(subsection)?);
    }
    let nested = base.join(subsection);
    Ok(if nested.is_dir() { nested } else { base })
}

/// Whether `source` is a library keyword that also names a directory in the
/// working directory.
fn is_shadowed_keyword(source: &str) -> bool {
    is_keyword(source) && std::path::Path::new(source).is_dir()
}

/// Refuse `source` when it is both a library keyword and a directory in the
/// working directory: reading either one would be a guess at which was meant.
/// `./<name>` (or an absolute path) names the directory unambiguously.
fn refuse_shadowed_keyword(source: &str) -> Result<(), String> {
    if !is_shadowed_keyword(source) {
        return Ok(());
    }
    let dir = std::path::absolute(source).unwrap_or_else(|_| PathBuf::from(source));
    Err(format!(
        "'{source}' is both a library keyword and the directory {}, so which one is meant is \
         not clear. To read the directory, give it as a path (\"./{source}\" or \"{}\"); to \
         use the published '{source}' library, rename or move the directory.",
        dir.display(),
        dir.display()
    ))
}

/// Download the sections of one nuclide or element from a raw base URL (one
/// that is not a library keyword) into the cache, and return the local path.
///
/// A raw URL has no release manifest, so nothing here is verified against
/// one: this is the path for a self-hosted unversioned tree. A library keyword
/// goes through [`super::release_cache`] instead, which verifies every file.
/// The cache folder is `<cache root>/<Name>.arrow`.
#[cfg(feature = "download")]
pub fn download_and_cache(
    url: &str,
    source: &str,
    nuclide_name: &str,
    kind: DataKind,
    scope: &crate::load_scope::LoadScope,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = get_cache_dir()?;
    let local_path = cache_dir.join(format!("{nuclide_name}.arrow"));
    let sections = sections_for(kind, scope);
    let sections = sections.as_slice();
    // Which MTs or temperatures, if this load reads only some of them. Decides
    // whether reactions.arrow (and, by temperature, energy.arrow) is fetched
    // whole or as the byte ranges the load reads.
    let ranging = ranged(kind, scope);

    // Fast path: cache hit without any locking. The gate is per-section rather
    // than per-directory, because a cache dir populated by an earlier
    // transmutation load holds only some of what transport now wants. Each
    // section is published by rename, so a section that is present
    // is complete.
    if have_all_sections(&local_path, sections, ranging) {
        return Ok(local_path);
    }

    // Serialize concurrent downloads of the same nuclide. Re-check after
    // acquiring the lock in case another thread finished the download while
    // we were waiting.
    let path_lock = get_path_lock(&local_path);
    let _guard = path_lock.lock().unwrap_or_else(|p| p.into_inner());
    if have_all_sections(&local_path, sections, ranging) {
        return Ok(local_path);
    }

    download_sections(url, &local_path, sections, source, nuclide_name, ranging)?;
    Ok(local_path)
}

/// The error for a required section a raw URL answered 404 for, which usually
/// means the nuclide itself is not published there.
#[cfg(feature = "download")]
fn download_error(
    url: &str,
    status: &str,
    source: &str,
    nuclide_name: &str,
) -> Box<dyn std::error::Error> {
    format!(
        "Failed to download {url}: {status}. Nuclide '{nuclide_name}' is probably not \
         published under '{source}'."
    )
    .into()
}

/// Per-nuclide neutron section objects (option-D). `(filename, required)`:
/// optional sections are absent for many nuclides (urr only in resonance
/// nuclides, total_nu and fission_photon only in fissionables). On a library
/// keyword an optional section is absent when the release manifest does not
/// list it; on a raw URL a 404 settles it.
///
/// Public and not gated on `download`: the browser build has no reqwest and
/// fetches these same objects through the JS `fetch` API, and the one list is
/// what keeps the two downloaders asking for the same files.
pub const NEUTRON_SECTIONS: &[(&str, bool)] = &[
    ("version.json", true),
    ("nuclide.arrow", true),
    // Required from format_version 2: the union energy grids, which every
    // neutron scope interpolates against.
    ("energy.arrow", true),
    ("reactions.arrow", true),
    ("products.arrow", true),
    ("distributions.arrow", true),
    ("urr.arrow", false),
    ("total_nu.arrow", false),
    ("fission_photon.arrow", false),
];

/// [`NEUTRON_SECTIONS`] plus the optional covariance table.
///
/// A separate list rather than a runtime push, because the fetcher takes a
/// `&'static` slice and covariance is orthogonal to the transport/activation
/// split: both scopes can be asked for it and both can be asked without it.
#[cfg(feature = "download")]
const NEUTRON_SECTIONS_WITH_COVARIANCE: &[(&str, bool)] = &[
    ("version.json", true),
    ("nuclide.arrow", true),
    // Required from format_version 2: the union energy grids, which every
    // neutron scope interpolates against.
    ("energy.arrow", true),
    ("reactions.arrow", true),
    ("products.arrow", true),
    ("distributions.arrow", true),
    ("urr.arrow", false),
    ("total_nu.arrow", false),
    ("fission_photon.arrow", false),
    // Optional: most evaluations carry no MF=33, so a 404 here is expected and
    // is recorded as settled rather than retried on every later load.
    ("covariance.arrow", false),
];

/// Per-element photon section objects (option-D). Public for the same reason
/// [`NEUTRON_SECTIONS`] is.
pub const PHOTON_SECTIONS: &[(&str, bool)] = &[
    ("version.json", true),
    ("element.arrow", true),
    ("subshells.arrow", false),
    ("compton.arrow", false),
    ("bremsstrahlung.arrow", false),
];

/// The neutron sections a transport-free reaction-rate collapse reads: the
/// union energy grid and the per-MT cross sections, and nothing else. All three
/// are required, so an activation fetch has no 404 to reason about.
#[cfg(feature = "download")]
const NEUTRON_XS_ONLY_SECTIONS: &[(&str, bool)] = &[
    ("version.json", true),
    ("nuclide.arrow", true),
    // Required from format_version 2: the union energy grids, which every
    // neutron scope interpolates against.
    ("energy.arrow", true),
    ("reactions.arrow", true),
];

/// [`NEUTRON_XS_ONLY_SECTIONS`] plus the optional covariance table.
///
/// The combination an uncertainty-aware `Material.transmute` asks for: the
/// cross sections it folds, and the covariance of those cross sections.
#[cfg(feature = "download")]
const NEUTRON_XS_ONLY_SECTIONS_WITH_COVARIANCE: &[(&str, bool)] = &[
    ("version.json", true),
    ("nuclide.arrow", true),
    // Required from format_version 2: the union energy grids, which every
    // neutron scope interpolates against.
    ("energy.arrow", true),
    ("reactions.arrow", true),
    ("covariance.arrow", false),
];

/// Which section objects a scope needs.
///
/// Photon data has no transmutation path, so it is always fetched whole. MF=34
/// (`angular_covariance.arrow`), MF=31 and MF=35 (`nubar_covariance.arrow`,
/// `spectrum_covariance.arrow`), and MF=2 with MF=32
/// (`resonance_parameters.arrow`) are added only when the scope asks for
/// them, and optional like `covariance.arrow`: most evaluations have none,
/// and a 404 is recorded rather than retried.
#[cfg(feature = "download")]
pub(crate) fn sections_for(
    kind: DataKind,
    scope: &crate::load_scope::LoadScope,
) -> Vec<(&'static str, bool)> {
    let base: &'static [(&'static str, bool)] = match kind {
        DataKind::Neutron if !scope.wants_transport_sections() => {
            if scope.covariance {
                NEUTRON_XS_ONLY_SECTIONS_WITH_COVARIANCE
            } else {
                NEUTRON_XS_ONLY_SECTIONS
            }
        }
        DataKind::Neutron if scope.covariance => NEUTRON_SECTIONS_WITH_COVARIANCE,
        DataKind::Neutron => NEUTRON_SECTIONS,
        DataKind::Photon => PHOTON_SECTIONS,
    };
    let mut sections = base.to_vec();
    if kind == DataKind::Neutron && scope.angular_covariance {
        sections.push(("angular_covariance.arrow", false));
    }
    if kind == DataKind::Neutron && scope.fission_covariance {
        sections.push(("nubar_covariance.arrow", false));
        sections.push(("spectrum_covariance.arrow", false));
    }
    if kind == DataKind::Neutron && scope.resonance_parameters {
        sections.push(("resonance_parameters.arrow", false));
    }
    sections
}

/// The section whose MTs can be fetched a few byte ranges at a time.
#[cfg(feature = "download")]
const REACTIONS: &str = "reactions.arrow";

/// Where a partially-fetched `reactions.arrow` is cached, relative to the
/// nuclide directory, and the record of which MTs it holds.
///
/// A subdirectory, and NOT `reactions.arrow` beside the whole file, for one
/// reason: nothing else in this module or in the reader can tell a partial file
/// from a complete one. `section_is_resolved` is a bare existence test, so a
/// partial written under the canonical name would satisfy every completeness
/// gate here forever, and a later transport load would top up the other
/// sections and never refetch it. Downstream that is silent rather than loud:
/// the transport lookup is built from whatever reactions the table holds, so a
/// nuclide can come back with no elastic scattering, no `fissionable` flag and
/// no MT 101 absorption, and nothing anywhere says so.
///
/// Under `subset/` the canonical name stays absent until the whole object is
/// fetched, so that gate keeps working untouched. The file inside is still
/// called `reactions.arrow` so `flat_section_for_path` resolves it and the
/// declared-schema check applies to a spliced stream exactly as it does to a
/// published file.
#[cfg(feature = "download")]
const SUBSET_DIR: &str = "subset";
#[cfg(feature = "download")]
const SUBSET_MTS: &str = "subset/mts.json";
#[cfg(feature = "download")]
const SUBSET_REACTIONS: &str = "subset/reactions.arrow";

/// The section of union energy grids, one batch per temperature, which a
/// temperature-ranged load fetches a few byte ranges at a time.
#[cfg(feature = "download")]
const ENERGY: &str = "energy.arrow";

/// Where a temperature-ranged `reactions.arrow` or `energy.arrow` is cached,
/// relative to the nuclide directory.
///
/// Its own subdirectory for the reason `subset/` is one: under the canonical
/// name a partial file would satisfy every existence gate here, and a later
/// load at another temperature would read a table with nothing at that
/// temperature. Beside each file is a `<section>.json` naming the temperature
/// labels it holds (see [`ranged_section`] and [`ranged_labels_file`]).
#[cfg(feature = "download")]
const RANGED_DIR: &str = "temperatures";

/// A temperature-ranged section's path, relative to the nuclide directory.
#[cfg(feature = "download")]
fn ranged_section(section: &str) -> String {
    format!("{RANGED_DIR}/{section}")
}

/// The record of which temperature labels a ranged section holds.
#[cfg(feature = "download")]
fn ranged_labels_file(section: &str) -> String {
    format!("{RANGED_DIR}/{}.json", section.trim_end_matches(".arrow"))
}

/// What a load wants out of the two sections that carry a byte-range index.
#[cfg(feature = "download")]
#[derive(Debug, Clone, Copy)]
pub(super) enum Ranged<'a> {
    /// Every section whole.
    Whole,
    /// These MTs of `reactions.arrow`, at every temperature. An activation
    /// load, cached under `subset/`.
    Mts(&'a HashSet<i32>),
    /// Every MT of `reactions.arrow`, and the energy grids, at these
    /// temperatures. A transport load for materials with a temperature,
    /// cached under `temperatures/`.
    Temperatures(&'a HashSet<String>),
}

/// How a scope's load ranges, if it does.
///
/// [`Ranged::Whole`] for photon data, for transport with no temperature (the
/// loader then reads every temperature), and for an activation load that
/// asked for every MT.
#[cfg(feature = "download")]
pub(super) fn ranged(kind: DataKind, scope: &crate::load_scope::LoadScope) -> Ranged<'_> {
    match kind {
        DataKind::Neutron if scope.ranges_temperatures() => scope
            .temperatures
            .as_ref()
            .map_or(Ranged::Whole, Ranged::Temperatures),
        DataKind::Neutron if !scope.wants_transport_sections() => {
            scope.mts.as_ref().map_or(Ranged::Whole, Ranged::Mts)
        }
        _ => Ranged::Whole,
    }
}

/// A nuclide directory's parsed `version.json`, if it has a readable one.
#[cfg(feature = "download")]
fn cached_version(dir: &std::path::Path) -> Option<serde_json::Value> {
    serde_json::from_str(&fs::read_to_string(dir.join("version.json")).ok()?).ok()
}

/// The byte-range index in a nuclide directory's `version.json`.
///
/// `None` whenever ranging is not possible: no marker yet, unreadable JSON, or
/// data published before the index existed. Every one of those means the same
/// thing to the caller, which is to fetch the section whole.
#[cfg(feature = "download")]
fn cached_index(dir: &std::path::Path) -> Option<ReactionRanges> {
    ReactionRanges::from_version_json(&cached_version(dir)?)
}

/// Every temperature label the reactions index publishes a batch at, spelled
/// as the files spell them (`"294K"`). The nuclide's temperature ladder.
#[cfg(feature = "download")]
fn published_labels(index: &ReactionRanges) -> BTreeSet<String> {
    index
        .mts
        .values()
        .flat_map(|by_temperature| by_temperature.keys().cloned())
        .collect()
}

/// The published labels the reader will parse for `requested`.
///
/// Resolved by `temperature::resolve`, the function the reader applies to the
/// same request, so what is fetched is what is read: a temperature the data
/// carries resolves to itself, and one it brackets to both neighbours, which
/// the reader blends. A temperature outside the ladder is the reader's error,
/// raised here before any bytes are fetched.
#[cfg(feature = "download")]
fn labels_for(
    index: &ReactionRanges,
    requested: &HashSet<String>,
) -> Result<BTreeSet<String>, crate::temperature::TemperatureError> {
    use crate::temperature::{resolve, TemperatureSource};
    let available: Vec<String> = published_labels(index).into_iter().collect();
    let mut labels = BTreeSet::new();
    for label in requested {
        match resolve(label, &available)? {
            TemperatureSource::Exact { idx } => {
                labels.insert(available[idx].clone());
            }
            TemperatureSource::Blend { lo_idx, hi_idx, .. } => {
                labels.insert(available[lo_idx].clone());
                labels.insert(available[hi_idx].clone());
            }
        }
    }
    Ok(labels)
}

/// The temperature labels a cached ranged section holds, as recorded beside it.
#[cfg(feature = "download")]
fn cached_labels(dir: &std::path::Path, section: &str) -> Option<BTreeSet<String>> {
    let text = fs::read_to_string(dir.join(ranged_labels_file(section))).ok()?;
    serde_json::from_str(&text).ok()
}

/// Whether `dir` already holds `section` at every temperature `requested`
/// resolves to.
///
/// The whole object covers everything. Otherwise the ranged copy covers the
/// request when it holds every label the request resolves to. A request that
/// does not resolve is not covered, so the fetch runs and reports why.
#[cfg(feature = "download")]
fn temperatures_covered(dir: &std::path::Path, section: &str, requested: &HashSet<String>) -> bool {
    if dir.join(section).exists() {
        return true;
    }
    if !dir.join(ranged_section(section)).exists() {
        return false;
    }
    let (Some(index), Some(held)) = (cached_index(dir), cached_labels(dir, section)) else {
        return false;
    };
    labels_for(&index, requested).is_ok_and(|wanted| wanted.is_subset(&held))
}

/// The MTs a cached subset holds, as recorded beside it.
#[cfg(feature = "download")]
fn cached_subset_mts(dir: &std::path::Path) -> Option<HashSet<i32>> {
    let text = fs::read_to_string(dir.join(SUBSET_MTS)).ok()?;
    serde_json::from_str::<Vec<i32>>(&text)
        .ok()
        .map(|v| v.into_iter().collect())
}

/// Whether `dir` already holds the reactions data a `wanted` MT set needs.
///
/// The whole object covers everything. Otherwise a cached subset covers this
/// request when it holds every MT of `wanted` that this nuclide actually
/// publishes, which is why both sides are narrowed through the index rather
/// than compared as asked-for sets: an MT the chain names and the nuclide has
/// no channel for would otherwise look uncovered on every load and refetch
/// forever.
#[cfg(feature = "download")]
fn reactions_covered(dir: &std::path::Path, wanted: &HashSet<i32>) -> bool {
    if dir.join(REACTIONS).exists() {
        return true;
    }
    let Some(held) = cached_subset_mts(dir) else {
        return false;
    };
    // No index means the next fetch cannot be a ranged one, so a subset can
    // never be shown to cover the request and the whole file is required.
    let Some(index) = cached_index(dir) else {
        return false;
    };
    index
        .present(|mt| wanted.contains(&mt))
        .into_iter()
        .all(|mt| held.contains(&mt))
}

/// Suffix marking an optional section that the origin answered 404 for.
///
/// Without this a partially populated cache dir cannot distinguish "we have not
/// asked for `urr.arrow` yet" from "this nuclide has none", and every later
/// transport load would re-request it. A zero-byte marker file avoids the
/// read-modify-write race a shared JSON manifest would have between processes.
#[cfg(feature = "download")]
const ABSENT_SUFFIX: &str = ".absent";

/// Whether `section` has already been settled in `dir`, either fetched or
/// recorded as absent upstream.
#[cfg(feature = "download")]
fn section_is_resolved(dir: &std::path::Path, section: &str) -> bool {
    dir.join(section).exists() || dir.join(format!("{section}{ABSENT_SUFFIX}")).exists()
}

/// Whether `dir` already holds `section` as far as this load needs it.
#[cfg(feature = "download")]
pub(super) fn section_covered(dir: &std::path::Path, section: &str, wanted: Ranged<'_>) -> bool {
    match wanted {
        Ranged::Mts(mts) if section == REACTIONS => reactions_covered(dir, mts),
        Ranged::Temperatures(temperatures) if section == REACTIONS || section == ENERGY => {
            temperatures_covered(dir, section, temperatures)
        }
        _ => section_is_resolved(dir, section),
    }
}

/// Whether `dir` already holds every one of `sections`.
///
/// Per-section rather than "does the directory exist": a dir populated by an
/// earlier transmutation load holds only some of what transport now wants.
#[cfg(feature = "download")]
fn have_all_sections(dir: &std::path::Path, sections: &[(&str, bool)], wanted: Ranged<'_>) -> bool {
    dir.is_dir()
        && sections
            .iter()
            .all(|(name, _)| section_covered(dir, name, wanted))
}

/// The subset of `sections` still to fetch into `dir`.
///
/// A dir that does not exist yet needs all of them; otherwise only the ones not
/// already settled, which is what makes a second load top up rather than
/// refetch.
#[cfg(feature = "download")]
fn sections_to_fetch<'a>(
    dir: &std::path::Path,
    sections: &[(&'a str, bool)],
    wanted: Ranged<'_>,
) -> Vec<(&'a str, bool)> {
    let existing = dir.is_dir();
    sections
        .iter()
        .copied()
        .filter(|(name, _)| !existing || !section_covered(dir, name, wanted))
        .collect()
}

/// What one GET came back with.
#[cfg(feature = "download")]
enum Fetched {
    /// The bytes asked for. `partial` records whether the origin honoured a
    /// `Range` header (206) or ignored it and sent the whole object (200).
    Body { bytes: Vec<u8>, partial: bool },
    /// A definitive 404. The section is absent upstream.
    Absent,
}

/// Fetch `url`, optionally just some byte ranges, retrying transient failures.
///
/// A 200 answer to a ranged request is not an error: some proxy dropped the
/// header and sent the whole object, which is more than was wanted and still
/// the right bytes. It is reported through `partial` so the caller can cache it
/// as the whole file rather than splicing it as though it were a slice. That
/// distinction matters: spliced after a schema message, a whole object's
/// `ARROW1` magic is read as the start of a record batch, and the framing walks
/// on into numbers that decode but are not the cross sections asked for.
///
/// A transport failure is retried like a 5xx: a request sent on a pooled
/// connection the origin has already closed, or a body cut off mid-transfer,
/// succeeds on a fresh attempt. A fresh cache issues hundreds of these in
/// parallel, so without the retry a first run failed on whichever one hit it.
#[cfg(feature = "download")]
fn fetch(url: &str, spans: &[Range]) -> Result<Fetched, Box<dyn std::error::Error>> {
    const RETRY_DELAYS_MS: &[u64] = &[200, 500, 1000];
    // A client that failed to build will not build on a retry either.
    client()?;
    let mut last_failure = String::new();
    for &delay_ms in std::iter::once(&0u64).chain(RETRY_DELAYS_MS.iter()) {
        if delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }
        let r = match blocking_get(url, spans) {
            Ok(r) => r,
            Err(e) => {
                last_failure = e.to_string();
                continue;
            }
        };
        let status = r.status;
        if (200..300).contains(&status) {
            match r.body {
                Ok(bytes) => {
                    return Ok(Fetched::Body {
                        partial: status == 206,
                        bytes,
                    });
                }
                Err(e) => {
                    last_failure = e;
                    continue;
                }
            }
        }
        if status == 404 {
            // Definitively absent: R2 answers authoritatively, no retry.
            return Ok(Fetched::Absent);
        }
        last_failure = format!("HTTP {status}");
    }
    Err(format!(
        "Failed to download {} after {} attempts: {}",
        url,
        RETRY_DELAYS_MS.len() + 1,
        last_failure
    )
    .into())
}

/// What a ranged fetch came back with.
#[cfg(feature = "download")]
enum Spans {
    /// The exact bytes of each span asked for, in order.
    Parts(Vec<Vec<u8>>),
    /// The whole object: the origin ignored the `Range` header and answered
    /// 200. More than was wanted and still the right bytes, so the caller
    /// caches it under the canonical name rather than splicing it.
    Whole(Vec<u8>),
}

/// The most ranges named in one request.
///
/// A transport load at one temperature wants one batch per MT, which is over a
/// hundred spans on a heavy nuclide, and a request each is a round trip each:
/// in the browser, where every request is a synchronous one made in turn, that
/// cost more time than the bytes it saved. Naming them all in one `Range`
/// header makes it one round trip. Capped so the header stays far below the
/// limits servers put on one (about 20 bytes a range, so 4 kB here).
#[cfg(feature = "download")]
const MAX_RANGES_PER_REQUEST: usize = 200;

/// Fetch `spans` of `url`, as few requests as the origin allows.
///
/// Up to [`MAX_RANGES_PER_REQUEST`] spans go in each request, and the
/// `multipart/byteranges` answer is cut back into the spans. An origin that
/// answers a multi-range request some other way (one range, or a body that
/// does not parse) is asked again a span at a time, which every origin that
/// serves ranges at all supports.
#[cfg(feature = "download")]
fn fetch_spans(url: &str, spans: &[Range]) -> Result<Spans, Box<dyn std::error::Error>> {
    let mut bodies = Vec::with_capacity(spans.len());
    for chunk in spans.chunks(MAX_RANGES_PER_REQUEST) {
        if chunk.len() > 1 {
            match fetch(url, chunk)? {
                Fetched::Body {
                    bytes,
                    partial: false,
                } => return Ok(Spans::Whole(bytes)),
                Fetched::Body {
                    bytes,
                    partial: true,
                } => {
                    if let Some(parts) =
                        parse_byteranges(&bytes).and_then(|parts| cut_spans(&parts, chunk))
                    {
                        bodies.extend(parts);
                        continue;
                    }
                }
                Fetched::Absent => {
                    return Err(
                        format!("{url}: 404, but its version.json names byte ranges").into(),
                    );
                }
            }
        }
        for span in chunk {
            match fetch_one_span(url, *span)? {
                Spans::Parts(mut part) => bodies.append(&mut part),
                whole => return Ok(whole),
            }
        }
    }
    Ok(Spans::Parts(bodies))
}

/// Fetch one span of `url` on its own request.
#[cfg(feature = "download")]
fn fetch_one_span(url: &str, span: Range) -> Result<Spans, Box<dyn std::error::Error>> {
    match fetch(url, &[span])? {
        Fetched::Body {
            bytes,
            partial: true,
        } => {
            if bytes.len() as u64 != span.1 {
                // A short span is not recoverable by splicing it: the framing
                // still walks, and the message header at the cut is read as
                // whatever the next bytes happen to be.
                return Err(format!(
                    "{url}: asked for {} bytes at {} and got {}",
                    span.1,
                    span.0,
                    bytes.len()
                )
                .into());
            }
            Ok(Spans::Parts(vec![bytes]))
        }
        // Range ignored somewhere in the path: what is in hand IS the whole
        // object, so cache it as one rather than splicing it as a slice.
        Fetched::Body {
            bytes,
            partial: false,
        } => Ok(Spans::Whole(bytes)),
        // The nuclide has a version.json naming ranges but no section to range
        // into, which is a broken publish rather than an absent optional
        // section.
        Fetched::Absent => {
            Err(format!("{url}: 404, but its version.json names byte ranges").into())
        }
    }
}

/// Read a `multipart/byteranges` body into `(offset, bytes)` parts.
///
/// The boundary is taken from the body's first line rather than from the
/// `Content-Type` header, so a host fetcher that hands back only the body (the
/// browser's) is read the same way. Each part is read by the length its
/// `Content-Range` gives rather than by searching for the next boundary, which
/// binary data could contain.
///
/// `None` for anything else, which is what a single-range answer looks like:
/// Arrow messages start with the `0xFFFFFFFF` continuation marker, never
/// with `--`.
#[cfg(feature = "download")]
fn parse_byteranges(body: &[u8]) -> Option<Vec<(u64, Vec<u8>)>> {
    fn line_end(body: &[u8], from: usize) -> Option<usize> {
        body.get(from..)?
            .windows(2)
            .position(|w| w == b"\r\n")
            .map(|i| from + i)
    }
    let mut pos = 0;
    while body.get(pos..pos + 2) == Some(&b"\r\n"[..]) {
        pos += 2;
    }
    let end = line_end(body, pos)?;
    let boundary = body.get(pos..end)?.strip_prefix(b"--")?;
    if boundary.is_empty() {
        return None;
    }
    pos = end + 2;

    let mut parts = Vec::new();
    loop {
        // The part's headers, up to the blank line.
        let mut range: Option<(u64, u64)> = None;
        loop {
            let end = line_end(body, pos)?;
            let line = std::str::from_utf8(&body[pos..end]).ok()?;
            pos = end + 2;
            if line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.trim().eq_ignore_ascii_case("content-range")
            {
                // `bytes 64-767/10442154`
                let spec = value.trim().strip_prefix("bytes ")?;
                let (first_last, _) = spec.split_once('/')?;
                let (first, last) = first_last.split_once('-')?;
                range = Some((first.parse().ok()?, last.parse().ok()?));
            }
        }
        let (first, last) = range?;
        let len = usize::try_from(last.checked_sub(first)? + 1).ok()?;
        parts.push((first, body.get(pos..pos + len)?.to_vec()));
        pos += len;

        // CRLF, then `--boundary`, then either `--` (the end) or CRLF.
        let rest = body.get(pos..)?.strip_prefix(b"\r\n--")?;
        let rest = rest.strip_prefix(boundary)?;
        if rest.starts_with(b"--") {
            return Some(parts);
        }
        rest.strip_prefix(b"\r\n")?;
        pos = body.len() - rest.len() + 2;
    }
}

/// The bytes of each of `spans`, cut from the parts an origin sent back.
///
/// An origin may merge ranges it was asked for into fewer, larger parts, so
/// each span is looked for inside whichever part covers it. `None` if one is
/// not covered, and the caller asks for the spans one at a time instead.
#[cfg(feature = "download")]
fn cut_spans(parts: &[(u64, Vec<u8>)], spans: &[Range]) -> Option<Vec<Vec<u8>>> {
    spans
        .iter()
        .map(|&(offset, len)| {
            parts.iter().find_map(|(first, bytes)| {
                let start = usize::try_from(offset.checked_sub(*first)?).ok()?;
                bytes
                    .get(start..start + usize::try_from(len).ok()?)
                    .map(<[u8]>::to_vec)
            })
        })
        .collect()
}

/// What a ranged fetch of one section staged, and what it cost.
#[cfg(feature = "download")]
pub(super) struct RangedFetch {
    /// The staged paths, relative to the staging directory, in the order they
    /// are to be published: the section before its record. Just the canonical
    /// section name when the origin ignored the `Range` header and sent the
    /// whole object.
    pub staged: Vec<String>,
    /// The ranges requested, after adjacent ones were merged.
    pub spans: usize,
    /// Bytes received.
    pub bytes: u64,
}

#[cfg(feature = "download")]
impl RangedFetch {
    /// Whether what was staged is the whole object rather than a ranged copy.
    pub(super) fn is_whole(&self) -> bool {
        self.staged.len() == 1 && !self.staged[0].contains('/')
    }
}

/// What the ranged copy of a section records beside it.
#[cfg(feature = "download")]
enum Holds {
    /// The MTs a `subset/reactions.arrow` holds.
    Mts(Vec<i32>),
    /// The temperature labels a `temperatures/<section>` holds.
    Labels(BTreeSet<String>),
}

/// The byte ranges a ranged fetch of one section asks for, planned from the
/// nuclide's `version.json`.
///
/// Made by [`plan_ranges`], which both [`fetch_ranged`] and [`prefetch`] call,
/// so what is fetched ahead is exactly what the fetch then asks for.
#[cfg(feature = "download")]
pub(super) struct RangePlan {
    /// The spans to request: the index entries wanted, merged where they touch.
    pub spans: Vec<Range>,
    /// Each index entry the spans cover, with its name for messages.
    pieces: Vec<(Range, String)>,
    /// The index's sha256 for each entry it has one for.
    hashes: nuclear_data_schema::reaction_ranges::Hashes,
    holds: Holds,
}

#[cfg(feature = "download")]
impl RangePlan {
    /// Whether every entry fetched can be checked against a hash. A fetch that
    /// is to be verified ranges only when it can.
    pub(super) fn verifiable(&self) -> bool {
        self.pieces.iter().all(|(r, _)| self.hashes.contains_key(r))
    }
}

/// The byte ranges of `section` a load reading `ranging` of it needs, given
/// the nuclide's `version.json` and what `target_dir` already holds.
///
/// `Ok(None)` means fetch the section whole: it is not one `ranging` ranges,
/// `version` has no index (data published before it), or the labels wanted
/// are every temperature there is, which one plain GET serves better than a
/// span per batch.
///
/// For temperatures, labels the cache already holds for this section are
/// fetched again with the new ones, so the rewritten file still serves the
/// loads that wrote the old one. That refetch is the price of keeping one file
/// per section: a model with the same nuclide at two temperatures asks for the
/// second one at most once. The energy section also keeps every grid whose
/// label is not a temperature of the nuclide (the 0 K grid the NJOY route
/// publishes), because the reader keeps those on any load and a ranged file
/// must provide what a whole one does.
#[cfg(feature = "download")]
pub(super) fn plan_ranges(
    version: &serde_json::Value,
    target_dir: &std::path::Path,
    section: &str,
    ranging: Ranged<'_>,
    nuclide_name: &str,
) -> Result<Option<RangePlan>, Box<dyn std::error::Error>> {
    if !ranges_section(section, ranging) {
        return Ok(None);
    }
    let Some(index) = ReactionRanges::from_version_json(version) else {
        return Ok(None);
    };
    let requested = match ranging {
        Ranged::Mts(wanted) => {
            let wants = |mt: i32| wanted.contains(&mt);
            return Ok(Some(RangePlan {
                spans: index.spans_for(wants),
                pieces: index.pieces_where(|mt, _| wants(mt)),
                // The MTs the nuclide actually published, not the ones asked
                // for. Recording the request would leave any MT this nuclide
                // has no channel for looking uncovered on every later load.
                holds: Holds::Mts(index.present(wants)),
                hashes: index.sha256,
            }));
        }
        Ranged::Temperatures(requested) => requested,
        Ranged::Whole => return Ok(None),
    };
    let ladder = published_labels(&index);
    let mut labels = labels_for(&index, requested).map_err(|e| format!("{nuclide_name}: {e}"))?;
    if target_dir.join(ranged_section(section)).exists() {
        labels.extend(cached_labels(target_dir, section).unwrap_or_default());
    }
    if labels.is_empty() || labels.is_superset(&ladder) {
        return Ok(None);
    }
    let (spans, pieces, hashes) = if section == REACTIONS {
        let wants = |_: i32, label: &str| labels.contains(label);
        (
            index.spans_where(wants),
            index.pieces_where(wants),
            index.sha256,
        )
    } else {
        let Some(energy) = EnergyRanges::from_version_json(version) else {
            return Ok(None);
        };
        let wants = |label: &str| labels.contains(label) || !ladder.contains(label);
        (
            energy.spans_for(wants),
            energy.pieces_for(wants),
            energy.sha256,
        )
    };
    Ok(Some(RangePlan {
        spans,
        pieces,
        hashes,
        holds: Holds::Labels(labels),
    }))
}

/// Fetch `section` of the nuclide at `base_url` as the byte ranges `ranging`
/// needs, into `staging`, when it can be.
///
/// Writes the spliced Arrow IPC stream and the record of what it holds:
/// `subset/reactions.arrow` and `subset/mts.json` for MTs,
/// `temperatures/<section>` and `temperatures/<section stem>.json` for
/// temperatures (see [`plan_ranges`] for which).
///
/// `Ok(None)` means fetch it whole instead, which is a normal answer: see
/// [`plan_ranges`], and with `verified`, some entry needed has no hash to check
/// it against.
///
/// `verified` is the release manifest's entry for the section, given on the
/// keyword path. With it every byte received is checked before anything is
/// staged: each index entry against the sha256 `version.json` gives it, or, if
/// the origin ignored the `Range` header and sent the whole object, that
/// object against the manifest. A mismatch is the error, naming the nuclide,
/// the file, the entry, and both hashes. It is not retried: the origin
/// answered in full with the wrong bytes, and asking again would only make the
/// error slower. Without `verified` (a raw URL, which has no manifest) the
/// bytes are taken as they come, as they always have been. A 200 answer to a
/// ranged request is staged whole under the canonical name and returned as
/// that path.
///
/// `version.json` is read from `staging` or else `target_dir`: it is first in
/// every section list, so it is either staged by the same download already or
/// in the cache from an earlier one. On the keyword path it is a verified file
/// either way, which is what makes the hashes inside it worth checking against.
#[cfg(feature = "download")]
pub(super) fn fetch_ranged(
    base_url: &str,
    staging: &std::path::Path,
    target_dir: &std::path::Path,
    section: &str,
    ranging: Ranged<'_>,
    nuclide_name: &str,
    verified: Option<&super::release::ManifestFile>,
) -> Result<Option<RangedFetch>, Box<dyn std::error::Error>> {
    let Some(version) = cached_version(staging).or_else(|| cached_version(target_dir)) else {
        return Ok(None);
    };
    let Some(plan) = plan_ranges(&version, target_dir, section, ranging, nuclide_name)? else {
        return Ok(None);
    };
    let check = verified.map(|file| Check {
        file,
        nuclide: nuclide_name,
    });
    if check.is_some() && !plan.verifiable() {
        // Those bytes could not be verified, so the caller fetches the whole
        // file, which the manifest can verify.
        return Ok(None);
    }
    let url = format!("{base_url}/{section}");
    let bodies = match fetch_spans(&url, &plan.spans)? {
        Spans::Parts(bodies) => {
            if let Some(check) = check {
                check_pieces(&plan, &bodies, check)?;
            }
            bodies
        }
        Spans::Whole(bytes) => {
            if let Some(check) = check {
                super::release::verify(
                    &format!("{url} ({})", check.file.path),
                    check.file,
                    bytes.len() as u64,
                    &super::release::sha256_hex(&bytes),
                )?;
            }
            fs::write(staging.join(section), &bytes)?;
            return Ok(Some(RangedFetch {
                staged: vec![section.to_string()],
                spans: plan.spans.len(),
                bytes: bytes.len() as u64,
            }));
        }
    };

    let (file, record, held) = match &plan.holds {
        Holds::Mts(mts) => (
            SUBSET_REACTIONS.to_string(),
            SUBSET_MTS.to_string(),
            serde_json::to_vec(mts)?,
        ),
        Holds::Labels(labels) => (
            ranged_section(section),
            ranged_labels_file(section),
            serde_json::to_vec(labels)?,
        ),
    };
    let bytes = bodies.iter().map(|b| b.len() as u64).sum();
    if let Some(parent) = staging.join(&file).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(staging.join(&file), splice_spans(&bodies))?;
    fs::write(staging.join(&record), held)?;
    // The section before its record, so a publish interrupted between the two
    // renames leaves a record naming no more than the file holds.
    Ok(Some(RangedFetch {
        staged: vec![file, record],
        spans: plan.spans.len(),
        bytes,
    }))
}

/// Whether `ranging` fetches `section` as byte ranges when it can.
#[cfg(feature = "download")]
pub(super) fn ranges_section(section: &str, ranging: Ranged<'_>) -> bool {
    match ranging {
        Ranged::Mts(_) => section == REACTIONS,
        Ranged::Temperatures(_) => section == REACTIONS || section == ENERGY,
        Ranged::Whole => false,
    }
}

/// What a ranged fetch on the keyword path is checked against.
#[cfg(feature = "download")]
#[derive(Clone, Copy)]
struct Check<'a> {
    /// The release manifest's entry for the whole section.
    file: &'a super::release::ManifestFile,
    nuclide: &'a str,
}

/// Check each index entry a ranged fetch covers against its hash.
///
/// The spans requested are entries merged where they touch, so each entry is
/// cut back out of the span that holds it before it is hashed. Every entry is
/// checked, not just the first of a span: a batch the origin got wrong is only
/// wrong where it is.
#[cfg(feature = "download")]
fn check_pieces(plan: &RangePlan, bodies: &[Vec<u8>], check: Check<'_>) -> Result<(), String> {
    for ((off, len), name) in &plan.pieces {
        let piece = plan
            .spans
            .iter()
            .zip(bodies)
            .find(|((start, span_len), _)| *start <= *off && off + len <= start + span_len)
            .and_then(|((start, _), body)| {
                let at = usize::try_from(off - start).ok()?;
                body.get(at..at + usize::try_from(*len).ok()?)
            })
            .ok_or_else(|| {
                format!(
                    "{}: {}: no span fetched covers {name} (bytes {off}..{})",
                    check.nuclide,
                    check.file.path,
                    off + len
                )
            })?;
        let expected = &plan.hashes[&(*off, *len)];
        let got = super::release::sha256_hex(piece);
        if !got.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "{}: {}: sha256 mismatch in {name} (bytes {off}..{}): the release's \
                 version.json says {expected} and the received bytes hash to {got}. Nothing \
                 was cached.",
                check.nuclide,
                check.file.path,
                off + len
            ));
        }
    }
    Ok(())
}

/// Drop the ranged copies a newly published whole section supersedes.
///
/// The reader prefers the whole object, so what is left is dead bytes and a
/// second copy of the same cross sections to puzzle over in a cache
/// directory. Called after the rename, so a failure before it leaves the
/// ranged copy in place and the directory still usable.
#[cfg(feature = "download")]
pub(super) fn drop_superseded(dir: &std::path::Path, published: &[String]) {
    if published.iter().any(|name| name == REACTIONS) {
        fs::remove_dir_all(dir.join(SUBSET_DIR)).ok();
    }
    for section in [REACTIONS, ENERGY] {
        if published.iter().any(|name| name == section) {
            fs::remove_file(dir.join(ranged_section(section))).ok();
            fs::remove_file(dir.join(ranged_labels_file(section))).ok();
        }
    }
}

/// Fetch one whole section object to `dest`. Returns `Ok(true)` if written,
/// `Ok(false)` on a definitive 404 (absent optional section), `Err` on any
/// other failure after retrying transient (5xx / transport) errors.
#[cfg(feature = "download")]
fn fetch_section_to_file(
    url: &str,
    dest: &std::path::Path,
) -> Result<bool, Box<dyn std::error::Error>> {
    match fetch(url, &[])? {
        Fetched::Body { bytes, .. } => {
            fs::write(dest, bytes)?;
            Ok(true)
        }
        Fetched::Absent => Ok(false),
    }
}

/// Download the tar-free per-section object set (option-D) for one nuclide /
/// element into `target_dir`, fetching only the sections not already settled
/// there.
///
/// Sections are staged on the same filesystem and then published by rename, so
/// a killed or 404'd download never leaves a half-written section that a later
/// read would treat as a hit. Which rename depends on whether the cache dir
/// exists yet:
///
/// * absent: stage everything and rename the whole directory into place, as
///   before. Nothing can observe a partial directory.
/// * present: rename each fetched section in individually. This is the additive
///   case and it is why the whole directory is not replaced:
///   a user who transmutes today fetches three sections, and a transport run
///   tomorrow tops up the rest instead of refetching all of them.
#[cfg(feature = "download")]
fn download_sections(
    base_url: &str,
    target_dir: &std::path::Path,
    sections: &[(&str, bool)],
    source: &str,
    nuclide_name: &str,
    ranging: Ranged<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let existing = target_dir.is_dir();
    let wanted = sections_to_fetch(target_dir, sections, ranging);
    if wanted.is_empty() {
        return Ok(());
    }

    // A prefetch counts nuclides instead, which reads better than a line
    // each when there are hundreds.
    if !batch::quiet() {
        println!(
            "Downloading {} section(s) to cache: {} -> {:?}",
            wanted.len(),
            base_url,
            target_dir
        );
    }
    let cache_dir = target_dir
        .parent()
        .ok_or("cache target has no parent directory")?;
    let staging = cache_dir.join(format!(
        ".staging-{}-{}",
        target_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("nuc"),
        std::process::id()
    ));
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;

    // Names actually written into staging, paired with their destination name.
    // An absent optional section contributes a zero-byte marker instead.
    let build = || -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut staged = Vec::new();
        for (name, required) in &wanted {
            // reactions.arrow, for a load that named its MTs, and reactions.arrow
            // and energy.arrow, for a transport load at named temperatures, are
            // fetched as the few byte ranges the load reads. With no index
            // (data published before it) they are fetched whole, exactly as
            // before.
            if let Some(fetched) = fetch_ranged(
                base_url,
                &staging,
                target_dir,
                name,
                ranging,
                nuclide_name,
                None,
            )? {
                staged.extend(fetched.staged);
                continue;
            }
            let url = format!("{}/{}", base_url, name);
            let fetched = fetch_section_to_file(&url, &staging.join(name))?;
            if fetched {
                staged.push((*name).to_string());
            } else if *required {
                // A missing required section on the first request usually means
                // the nuclide itself is absent from the library.
                return Err(download_error(&url, "404 Not Found", source, nuclide_name));
            } else {
                let marker = format!("{name}{ABSENT_SUFFIX}");
                fs::File::create(staging.join(&marker))?;
                staged.push(marker);
            }
        }
        Ok(staged)
    };

    match build() {
        Ok(staged) => {
            if existing {
                // Additive publish: move each section in on its own. Same
                // filesystem, so each rename is atomic; a reader either sees the
                // old state or the complete new section, never a partial file.
                for name in &staged {
                    let to = target_dir.join(name);
                    // `subset/` is a staged path rather than a bare filename, so
                    // its parent may not exist in an already-published dir.
                    if let Some(parent) = to.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::rename(staging.join(name), to)?;
                }
                // The whole object supersedes any ranged copy beside it.
                drop_superseded(target_dir, &staged);
                fs::remove_dir_all(&staging).ok();
            } else {
                // First fetch. Publish the directory atomically. Another racing
                // thread holds a different pid staging dir; the path lock in
                // download_and_cache serializes the rename onto target_dir.
                if target_dir.exists() {
                    fs::remove_dir_all(target_dir).ok();
                }
                fs::rename(&staging, target_dir)?;
            }
            Ok(())
        }
        Err(e) => {
            fs::remove_dir_all(&staging).ok();
            Err(e)
        }
    }
}

/// The directory yamc caches downloaded nuclear data in, without creating it.
///
/// The platform's per-user cache directory with `yamc` appended, as
/// `etcetera`'s native strategy resolves it:
///
/// - Linux and other unix: `$XDG_CACHE_HOME/yamc`, which is `~/.cache/yamc`
///   unless the desktop's XDG base directory says otherwise.
/// - macOS: `~/Library/Caches/yamc`.
/// - Windows: `%LOCALAPPDATA%\yamc` (`C:\Users\<user>\AppData\Local\yamc`),
///   resolved through `SHGetKnownFolderPath`.
///
/// There is no yamc setting or environment variable that moves it. A user
/// who wants data somewhere else points the data source at a local directory
/// instead, which bypasses the cache entirely.
///
/// `None` means the platform resolved no home directory, which is a machine
/// with no cache location rather than a machine with an empty cache. Callers
/// that need a path say what that means for them: [`get_cache_dir`] reports it
/// as an error, and the tests treat it as a failure rather than as absent data.
pub fn cache_root() -> Option<PathBuf> {
    if let Some(root) = CACHE_ROOT_FOR_TESTS
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
    {
        return Some(root);
    }
    default_cache_root()
}

/// The platform cache directory with `yamc` appended (see [`cache_root`]).
fn default_cache_root() -> Option<PathBuf> {
    use etcetera::base_strategy::BaseStrategy;
    Some(
        etcetera::base_strategy::choose_native_strategy()
            .ok()?
            .cache_dir()
            .join("yamc"),
    )
}

/// A cache root that replaces the platform one for the whole process, so a
/// test can download into a scratch directory. Not a user setting.
static CACHE_ROOT_FOR_TESTS: std::sync::RwLock<Option<PathBuf>> = std::sync::RwLock::new(None);

/// Point [`cache_root`] at `root` for the rest of the process (`None` restores
/// the platform default). For the test suites of this workspace only: it is
/// process-global, races every other download in the process, and is not part
/// of the supported API.
#[doc(hidden)]
pub fn set_cache_root_for_tests(root: Option<PathBuf>) {
    *CACHE_ROOT_FOR_TESTS
        .write()
        .unwrap_or_else(|p| p.into_inner()) = root;
}

/// The user's home directory.
///
/// `etcetera::home_dir` over `std::env::home_dir`: `HOME` on unix with the
/// passwd entry behind it, `USERPROFILE` on Windows with
/// `SHGetKnownFolderPath` behind it. Deliberately NOT a hand-rolled read of
/// those two variables. A container, a service or an `env -i` shell has a home
/// directory by the fallback and not by the variable, and resolving to nothing
/// there would break every download for the sake of a rule this crate has no
/// business restating.
///
/// Tests must resolve the home directory through here too, rather than reading
/// `HOME` themselves: `HOME` is not a Windows variable, so a direct read
/// resolves to nothing there.
pub fn home_dir() -> Option<PathBuf> {
    etcetera::home_dir().ok()
}

/// The release each library keyword resolved to in this process, keyed by
/// keyword, and the release of each local release folder data was read from,
/// keyed by the folder's absolute path: the release identifier, its manifest
/// sha256, and whether the origin was unreachable so a cached release was used
/// (never, for a local folder). Results record this as their nuclear-data
/// provenance.
pub fn data_releases() -> std::collections::BTreeMap<String, super::release::DataRelease> {
    #[cfg(feature = "download")]
    let mut releases = super::release_cache::data_releases();
    #[cfg(not(feature = "download"))]
    let mut releases = std::collections::BTreeMap::new();
    releases.extend(super::local_release::used());
    releases
}

/// Whether `path`, a path data was loaded from, lies in the local release
/// folder that [`data_releases`] records under `key`. Never true of a keyword.
pub fn local_release_holds(key: &str, path: &std::path::Path) -> bool {
    super::local_release::holds(key, path)
}

/// Get the cache directory for yamc, creating it if it does not exist.
#[cfg(feature = "download")]
pub fn get_cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = cache_root()
        .ok_or("Could not find a cache directory: the platform resolved no home directory")?;

    // Create the cache directory if it doesn't exist
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)?;
    }

    Ok(cache_dir)
}

/// Resolve a nuclide (or element) name within a directory.
///
/// A directory holding a release `manifest.json` is a release folder, laid out
/// `neutron/<Name>.arrow/` and `photon/<Name>.arrow/` (a library downloaded or
/// built whole, or a cached release folder; see [`super::local_release`]). The
/// files of the entry that the manifest lists and that are present are checked
/// against their manifest sizes, which costs a `stat` each and catches a
/// truncated copy; a full hash check is `verify_library`'s job. Any other
/// directory holds `<Name>.arrow/` directly.
fn resolve_nuclide_in_dir(
    dir: &std::path::Path,
    nuclide_name: &str,
    kind: DataKind,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    match super::local_release::open(dir)? {
        Some(release) => Ok(release.particle(kind, nuclide_name)?),
        None => Ok(dir.join(format!("{nuclide_name}.arrow"))),
    }
}

/// Check if a string looks like a URL (starts with http:// or https://)
pub fn is_url(path_or_url: &str) -> bool {
    path_or_url.starts_with("http://") || path_or_url.starts_with("https://")
}

/// Resolve a path, URL, keyword, or directory to a local file path.
/// Resolution order: keyword → directory → URL → file path.
/// If it's a keyword, resolve it to the library's current release and serve
/// the `kind`-specific folder from the verified cache, downloading what it
/// lacks.
/// If it's a directory, resolve to `<dir>/<nuclide_name>.arrow`, or in a
/// release folder to its `neutron/` or `photon/` entry.
/// If it's a URL, download and cache it.
/// If it's a local file path, return as-is.
/// A bare name that is both a keyword and a directory is refused.
#[cfg(feature = "download")]
pub fn resolve_path_or_url(
    path_url_or_keyword: &str,
    nuclide_name: &str,
    kind: DataKind,
    scope: &crate::load_scope::LoadScope,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    refuse_shadowed_keyword(path_url_or_keyword)?;
    if is_branching_only(path_url_or_keyword) {
        return Err(branching_only_error(path_url_or_keyword, data_role(kind)).into());
    }
    if is_keyword(path_url_or_keyword) {
        // A library keyword: the current release, verified (release_cache).
        Ok(super::release_cache::fetch_particle(
            &super::release_cache::REGISTRY,
            path_url_or_keyword,
            nuclide_name,
            kind,
            scope,
        )?)
    } else if std::path::Path::new(path_url_or_keyword).is_dir() {
        let dir = PathBuf::from(path_url_or_keyword);
        // If this IS already an Arrow nuclide directory, return it directly
        if dir.extension().and_then(|e| e.to_str()) == Some("arrow") {
            Ok(dir)
        } else {
            // It's a parent directory - resolve to nuclide data file/directory inside it
            resolve_nuclide_in_dir(&dir, nuclide_name, kind)
        }
    } else if is_url(path_url_or_keyword) {
        // It's a direct URL
        download_and_cache(
            path_url_or_keyword,
            path_url_or_keyword,
            nuclide_name,
            kind,
            scope,
        )
    } else {
        // It's a local file path
        Ok(PathBuf::from(path_url_or_keyword))
    }
}

/// Resolve a data path for the nuclide / element loaders to a concrete local
/// path string, mirroring how both loaders previously inlined this logic.
///
/// Resolution:
/// - If `path` is already an Arrow data directory (`*.arrow/`), return it verbatim.
/// - Otherwise, when the `download` feature is enabled, keywords, URLs, and parent
///   directories are resolved via [`resolve_path_or_url`]; without the feature only
///   parent directories are resolved (keywords / URLs are left to error downstream).
/// - Anything else (a plain local file path) is returned verbatim.
///
/// `name` is the nuclide / element name needed to resolve keywords, URLs, and
/// directories. It may be `None` for verbatim paths; if a name is required to
/// resolve and none is supplied, this returns an error.
pub fn resolve_data_path(
    path: &str,
    name: Option<&str>,
    kind: DataKind,
    scope: &crate::load_scope::LoadScope,
) -> Result<String, Box<dyn std::error::Error>> {
    let p = std::path::Path::new(path);
    let is_arrow_dir = p.is_dir() && p.extension().and_then(|e| e.to_str()) == Some("arrow");
    if is_arrow_dir {
        // Already an Arrow data directory -- use directly.
        return Ok(path.to_string());
    }

    #[cfg(feature = "download")]
    let needs_resolution = is_keyword(path) || is_url(path) || p.is_dir();
    #[cfg(not(feature = "download"))]
    let needs_resolution = p.is_dir();

    if needs_resolution {
        let name = name.ok_or({
            #[cfg(feature = "download")]
            {
                "Nuclide name is required for keyword/URL/directory resolution"
            }
            #[cfg(not(feature = "download"))]
            {
                "Nuclide name is required for directory resolution"
            }
        })?;
        Ok(resolve_path_or_url(path, name, kind, scope)?
            .to_string_lossy()
            .to_string())
    } else {
        // Plain local file path -- no name needed.
        Ok(path.to_string())
    }
}

/// WASM fallback for subsection resolution -- only local paths are supported.
#[cfg(not(feature = "download"))]
pub fn resolve_subsection(
    source: &str,
    subsection: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    refuse_shadowed_keyword(source)?;
    if is_branching_only(source) && subsection != "branching" {
        return Err(branching_only_error(
            source,
            &format!("the source of the '{subsection}' transmutation subsection"),
        )
        .into());
    }
    if is_keyword(source) {
        Err(format!(
            "Library keyword '{source}' cannot be resolved to a transmutation \
             subsection in WASM builds. Please pass a local path."
        )
        .into())
    } else {
        resolve_local_subsection(source, subsection)
    }
}

/// WASM fallback - only supports local paths and directories, not URLs or keywords
#[cfg(not(feature = "download"))]
#[allow(dead_code)]
pub fn resolve_path_or_url(
    path_url_or_keyword: &str,
    nuclide_name: &str,
    kind: DataKind,
    _scope: &crate::load_scope::LoadScope,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    refuse_shadowed_keyword(path_url_or_keyword)?;
    if is_branching_only(path_url_or_keyword) {
        return Err(branching_only_error(path_url_or_keyword, data_role(kind)).into());
    }
    if is_keyword(path_url_or_keyword) {
        Err("URL downloading and keywords are not supported in WASM builds. Please use local file paths.".into())
    } else if std::path::Path::new(path_url_or_keyword).is_dir() {
        let dir = PathBuf::from(path_url_or_keyword);
        // If this IS already an Arrow nuclide directory, return it directly
        if dir.extension().and_then(|e| e.to_str()) == Some("arrow") {
            Ok(dir)
        } else {
            // It's a parent directory - resolve to nuclide data file/directory inside it
            resolve_nuclide_in_dir(&dir, nuclide_name, kind)
        }
    } else if is_url(path_url_or_keyword) {
        Err("URL downloading is not supported in WASM builds. Please use local file paths.".into())
    } else {
        // It's a local file path
        Ok(PathBuf::from(path_url_or_keyword))
    }
}

#[cfg(all(test, feature = "download"))]
mod tests {
    use super::*;

    /// The MF=40 covariance is optional: a library without one answers 404, so
    /// a required entry would fail every branching download there, while an
    /// optional one settles as a `.absent` marker and is fetched where it exists.
    #[test]
    fn the_branching_subsection_lists_its_covariance_as_optional() {
        let sections = transmutation_sections("branching");
        assert!(
            sections.contains(&("branching.arrow", true)),
            "{sections:?}"
        );
        assert!(
            sections.contains(&("branching_covariance.arrow", false)),
            "{sections:?}"
        );
        assert_eq!(
            sections.last(),
            Some(&("provenance.json", false)),
            "provenance.json stays last, as in every other subsection"
        );
    }

    /// Optional, so a library published before the evaluated yields existed
    /// settles as `.absent` on a 404 rather than failing every download.
    #[test]
    fn the_fission_yields_subsection_lists_its_evaluated_yields_as_optional() {
        let sections = transmutation_sections("fission_yields");
        assert!(
            sections.contains(&("fission_yields.arrow", true)),
            "{sections:?}"
        );
        assert!(
            sections.contains(&("evaluated_yields.arrow", false)),
            "{sections:?}"
        );
        assert_eq!(
            sections.last(),
            Some(&("provenance.json", false)),
            "provenance.json stays last, as in every other subsection"
        );
    }

    // ---- scoped, additive section fetching ----

    /// A scratch directory that removes itself. The crate has no dev-dependencies
    /// and these tests only need a few empty files.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "yamc-url-cache-{}-{}-{:?}",
                tag,
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).expect("create temp dir");
            TempDir(p)
        }
        fn touch(&self, name: &str) {
            fs::File::create(self.0.join(name)).expect("touch");
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Write a `version.json` carrying a byte-range index for `mts`, one batch
    /// per (MT, temperature) at two temperatures, laid out back to back the way
    /// the converter writes the batches.
    fn write_index(dir: &std::path::Path, mts: &[i32]) {
        let mut ranges: std::collections::BTreeMap<
            i32,
            std::collections::BTreeMap<String, (u64, u64)>,
        > = std::collections::BTreeMap::new();
        let mut at = 768u64;
        for mt in mts {
            for temperature in ["294K", "600K"] {
                ranges
                    .entry(*mt)
                    .or_default()
                    .insert(temperature.to_string(), (at, 100u64));
                at += 100;
            }
        }
        let index = ReactionRanges {
            schema: (64, 704),
            mts: ranges,
            sha256: Default::default(),
        };
        fs::write(
            dir.join("version.json"),
            serde_json::json!({"reaction_ranges": index.to_json()}).to_string(),
        )
        .expect("write version.json");
    }

    fn write_subset(dir: &std::path::Path, held: &[i32]) {
        fs::create_dir_all(dir.join(SUBSET_DIR)).expect("subset dir");
        fs::write(dir.join(SUBSET_REACTIONS), b"stream").expect("subset stream");
        fs::write(dir.join(SUBSET_MTS), serde_json::to_vec(&held).unwrap()).expect("subset mts");
    }

    fn mts(values: &[i32]) -> HashSet<i32> {
        values.iter().copied().collect()
    }

    /// The whole object answers any MT set, so a directory topped up for
    /// transport never refetches on behalf of an activation load.
    #[test]
    fn the_whole_reactions_object_covers_every_subset() {
        let dir = TempDir::new("covers-whole");
        dir.touch(REACTIONS);
        assert!(reactions_covered(&dir.0, &mts(&[16, 102, 999])));
    }

    /// A subset covers a request when it holds every wanted MT the nuclide
    /// publishes, and not otherwise.
    #[test]
    fn a_subset_covers_only_what_it_holds() {
        let dir = TempDir::new("covers-subset");
        write_index(&dir.0, &[16, 102, 103]);
        write_subset(&dir.0, &[16, 102]);

        assert!(
            reactions_covered(&dir.0, &mts(&[16])),
            "a narrower request is covered"
        );
        assert!(
            reactions_covered(&dir.0, &mts(&[16, 102])),
            "the exact request is covered"
        );
        assert!(
            !reactions_covered(&dir.0, &mts(&[16, 102, 103])),
            "103 is published but not held, so this needs a fetch"
        );
    }

    /// An MT the chain names that this nuclide has no channel for must not keep
    /// the subset looking uncovered. It has no batch to fetch, so a load asking
    /// for it would otherwise refetch on every single call, forever.
    #[test]
    fn an_unpublished_mt_does_not_make_a_subset_look_stale() {
        let dir = TempDir::new("covers-unpublished");
        write_index(&dir.0, &[16, 102]);
        write_subset(&dir.0, &[16, 102]);
        assert!(
            reactions_covered(&dir.0, &mts(&[16, 102, 107, 111])),
            "107 and 111 are not in this nuclide's index, so nothing is missing"
        );
    }

    /// With no index there is no way to range, so a subset can never be shown to
    /// cover a request and the whole object is required.
    #[test]
    fn without_an_index_only_the_whole_object_will_do() {
        let dir = TempDir::new("covers-noindex");
        fs::write(dir.0.join("version.json"), r#"{"data_version": "1"}"#).expect("write");
        write_subset(&dir.0, &[16, 102]);
        assert!(!reactions_covered(&dir.0, &mts(&[16])));
    }

    /// The gate a transport load goes through is untouched by a cached subset:
    /// `subset/reactions.arrow` is not `reactions.arrow`, so the whole object is
    /// still fetched. This is the poisoning case, and it is closed by the layout
    /// rather than by a check that could be forgotten.
    #[test]
    fn a_cached_subset_does_not_satisfy_a_transport_load() {
        let dir = TempDir::new("subset-vs-transport");
        write_index(&dir.0, &[16, 102]);
        write_subset(&dir.0, &[16, 102]);
        for name in [
            "nuclide.arrow",
            "energy.arrow",
            "products.arrow",
            "distributions.arrow",
        ] {
            dir.touch(name);
        }
        let full = sections_for(DataKind::Neutron, &crate::LoadScope::full());
        assert!(
            !have_all_sections(&dir.0, &full, Ranged::Whole),
            "transport must not be satisfied by a subset"
        );
        let todo: Vec<&str> = sections_to_fetch(&dir.0, &full, Ranged::Whole)
            .iter()
            .map(|(n, _)| *n)
            .collect();
        assert!(
            todo.contains(&REACTIONS),
            "the whole reactions object must be on the list, got {todo:?}"
        );
    }

    /// And the activation load that wrote it is satisfied, so the subset is not
    /// merely safe but actually load-bearing.
    #[test]
    fn a_cached_subset_satisfies_the_activation_load_that_wrote_it() {
        let dir = TempDir::new("subset-satisfies");
        write_index(&dir.0, &[16, 102]);
        write_subset(&dir.0, &[16, 102]);
        dir.touch("nuclide.arrow");
        // The union energy grids are their own required section, so a folder
        // without one is not a satisfied activation load.
        dir.touch("energy.arrow");
        let wanted = mts(&[16, 102]);
        let xs_only = sections_for(
            DataKind::Neutron,
            &crate::LoadScope::activation(wanted.clone()),
        );
        assert!(have_all_sections(&dir.0, &xs_only, Ranged::Mts(&wanted)));
        assert!(sections_to_fetch(&dir.0, &xs_only, Ranged::Mts(&wanted)).is_empty());
    }

    /// Write a ranged copy of `section` holding `labels`, as a temperature-ranged
    /// fetch leaves it.
    fn write_ranged(dir: &std::path::Path, section: &str, labels: &[&str]) {
        fs::create_dir_all(dir.join(RANGED_DIR)).expect("ranged dir");
        fs::write(dir.join(ranged_section(section)), b"stream").expect("ranged stream");
        fs::write(
            dir.join(ranged_labels_file(section)),
            serde_json::to_vec(&labels).unwrap(),
        )
        .expect("ranged labels");
    }

    fn temperatures(labels: &[&str]) -> HashSet<String> {
        labels.iter().map(|l| l.to_string()).collect()
    }

    /// A ranged copy covers the temperatures it holds, spelled either way, and
    /// a temperature it brackets only when it holds both neighbours.
    #[test]
    fn a_ranged_copy_covers_only_the_temperatures_it_holds() {
        let dir = TempDir::new("covers-temperatures");
        write_index(&dir.0, &[2, 102]);
        write_ranged(&dir.0, REACTIONS, &["294K"]);

        assert!(temperatures_covered(
            &dir.0,
            REACTIONS,
            &temperatures(&["294"])
        ));
        assert!(temperatures_covered(
            &dir.0,
            REACTIONS,
            &temperatures(&["294K"])
        ));
        assert!(!temperatures_covered(
            &dir.0,
            REACTIONS,
            &temperatures(&["600"])
        ));
        assert!(
            !temperatures_covered(&dir.0, REACTIONS, &temperatures(&["400"])),
            "400 K blends 294 K and 600 K, and 600 K is not held"
        );

        write_ranged(&dir.0, REACTIONS, &["294K", "600K"]);
        assert!(temperatures_covered(
            &dir.0,
            REACTIONS,
            &temperatures(&["400"])
        ));
        assert!(
            !temperatures_covered(&dir.0, REACTIONS, &temperatures(&["5000"])),
            "out of range is never covered, so the fetch runs and says why"
        );
    }

    /// The whole object covers every temperature, and with no index a ranged
    /// copy can never be shown to cover anything.
    #[test]
    fn the_whole_object_covers_every_temperature_and_no_index_covers_none() {
        let dir = TempDir::new("covers-temperatures-whole");
        dir.touch(ENERGY);
        assert!(temperatures_covered(
            &dir.0,
            ENERGY,
            &temperatures(&["294"])
        ));

        fs::write(dir.0.join("version.json"), r#"{"data_version": "1"}"#).expect("write");
        write_ranged(&dir.0, REACTIONS, &["294K"]);
        assert!(!temperatures_covered(
            &dir.0,
            REACTIONS,
            &temperatures(&["294"])
        ));
    }

    /// A transport load at a held temperature is satisfied by the ranged
    /// copies, and a load with no temperature, which reads every one, is not.
    #[test]
    fn a_ranged_copy_satisfies_only_a_load_at_its_temperatures() {
        let dir = TempDir::new("ranged-vs-transport");
        write_index(&dir.0, &[2, 102]);
        write_ranged(&dir.0, REACTIONS, &["294K"]);
        write_ranged(&dir.0, ENERGY, &["294K"]);
        for name in ["nuclide.arrow", "products.arrow", "distributions.arrow"] {
            dir.touch(name);
        }
        for name in ["urr.arrow", "total_nu.arrow", "fission_photon.arrow"] {
            dir.touch(&format!("{name}{ABSENT_SUFFIX}"));
        }
        let full = sections_for(DataKind::Neutron, &crate::LoadScope::full());

        let at_294 = temperatures(&["294"]);
        assert!(have_all_sections(
            &dir.0,
            &full,
            Ranged::Temperatures(&at_294)
        ));

        let at_600 = temperatures(&["600"]);
        let todo: Vec<&str> = sections_to_fetch(&dir.0, &full, Ranged::Temperatures(&at_600))
            .iter()
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(
            todo,
            [ENERGY, REACTIONS],
            "only the two ranged sections top up"
        );

        let todo: Vec<&str> = sections_to_fetch(&dir.0, &full, Ranged::Whole)
            .iter()
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(
            todo,
            [ENERGY, REACTIONS],
            "a load at every temperature must not be satisfied by a ranged copy"
        );
    }

    /// A multipart body as an origin writes one, for parts at `(offset, bytes)`.
    fn multipart(parts: &[(u64, &[u8])], total: u64) -> Vec<u8> {
        let mut body = Vec::new();
        for (offset, bytes) in parts {
            body.extend_from_slice(b"\r\n--b0und4ry\r\nContent-Type: application/octet-stream\r\n");
            body.extend_from_slice(
                format!(
                    "Content-Range: bytes {}-{}/{total}\r\n\r\n",
                    offset,
                    offset + bytes.len() as u64 - 1
                )
                .as_bytes(),
            );
            body.extend_from_slice(bytes);
        }
        body.extend_from_slice(b"\r\n--b0und4ry--\r\n");
        body
    }

    /// Parts are read by their `Content-Range` length, so a part carrying the
    /// boundary text in its own bytes is still cut in the right place.
    #[test]
    fn a_multipart_body_parses_into_its_parts() {
        let tricky: &[u8] = b"ab\r\n--b0und4ry\r\ncd";
        let body = multipart(&[(10, b"hello"), (100, tricky)], 1000);
        let parts = parse_byteranges(&body).expect("parses");
        assert_eq!(parts, vec![(10, b"hello".to_vec()), (100, tricky.to_vec())]);

        // Without the leading CRLF, as some origins write it.
        let parts = parse_byteranges(&body[2..]).expect("parses");
        assert_eq!(parts.len(), 2);
    }

    /// A single-range answer, a truncated body and a part shorter than its
    /// `Content-Range` are not multipart bodies, so the caller asks again a
    /// span at a time.
    #[test]
    fn anything_else_is_not_a_multipart_body() {
        assert!(parse_byteranges(&[0xff, 0xff, 0xff, 0xff, 0x10, 0, 0, 0]).is_none());
        let body = multipart(&[(10, b"hello"), (100, b"world")], 1000);
        assert!(parse_byteranges(&body[..body.len() - 12]).is_none());
        let short = String::from_utf8(body.clone())
            .unwrap()
            .replace("bytes 100-104", "bytes 100-140");
        assert!(parse_byteranges(short.as_bytes()).is_none());
    }

    /// Spans are cut from whichever part covers them, so an origin that merged
    /// ranges into one part still answers each span.
    #[test]
    fn spans_are_cut_from_the_parts_that_cover_them() {
        let parts = vec![(10, b"0123456789".to_vec()), (100, b"abc".to_vec())];
        assert_eq!(
            cut_spans(&parts, &[(10, 2), (15, 3), (101, 2)]),
            Some(vec![b"01".to_vec(), b"567".to_vec(), b"bc".to_vec()])
        );
        assert_eq!(
            cut_spans(&parts, &[(18, 5)]),
            None,
            "runs off the end of a part"
        );
        assert_eq!(cut_spans(&parts, &[(50, 1)]), None, "in no part at all");
    }

    #[test]
    fn the_range_header_names_every_span_inclusively() {
        assert_eq!(range_header(&[(0, 100), (200, 1)]), "bytes=0-99,200-200");
    }

    /// Only a neutron load that named its MTs or its temperatures may range.
    /// Photon data has no per-MT batches, and transport with no temperature
    /// reads every temperature there is.
    #[test]
    fn only_a_neutron_scope_naming_mts_or_temperatures_ranges() {
        let wanted = mts(&[102]);
        let activation = crate::LoadScope::activation(wanted.clone());
        assert!(matches!(
            ranged(DataKind::Neutron, &activation),
            Ranged::Mts(m) if *m == wanted
        ));
        assert!(matches!(
            ranged(DataKind::Photon, &activation),
            Ranged::Whole
        ));
        assert!(matches!(
            ranged(DataKind::Neutron, &crate::LoadScope::full()),
            Ranged::Whole
        ));
        let at_294 = crate::LoadScope::full().with_temperatures(Some(["294".to_string()].into()));
        assert!(matches!(
            ranged(DataKind::Neutron, &at_294),
            Ranged::Temperatures(t) if t.contains("294")
        ));
        assert!(matches!(ranged(DataKind::Photon, &at_294), Ranged::Whole));
        // XsOnly with no MT filter reads every MT, so there is nothing to narrow.
        let all_mts = crate::LoadScope {
            sections: crate::load_scope::SectionScope::XsOnly,
            mts: None,
            temperatures: None,
            covariance: false,
            angular_covariance: false,
            fission_covariance: false,
            resonance_parameters: false,
        };
        assert!(matches!(ranged(DataKind::Neutron, &all_mts), Ranged::Whole));
    }

    #[test]
    fn an_activation_scope_asks_for_four_neutron_sections() {
        let xs = sections_for(
            DataKind::Neutron,
            &crate::LoadScope::activation([102].into()),
        );
        let names: Vec<&str> = xs.iter().map(|(n, _)| *n).collect();
        // energy.arrow holds the union grids. An activation collapse folds
        // cross sections onto
        // that grid, so it is as required here as the cross sections are.
        assert_eq!(
            names,
            [
                "version.json",
                "nuclide.arrow",
                "energy.arrow",
                "reactions.arrow"
            ]
        );
        assert!(
            xs.iter().all(|(_, required)| *required),
            "an activation fetch has no optional section, so no 404 to reason about"
        );

        // Everything else is unchanged.
        assert_eq!(
            sections_for(DataKind::Neutron, &crate::LoadScope::full()).len(),
            NEUTRON_SECTIONS.len()
        );
        assert_eq!(
            sections_for(
                DataKind::Photon,
                &crate::LoadScope::activation([102].into())
            )
            .len(),
            PHOTON_SECTIONS.len(),
            "photon data has no transmutation path and is always fetched whole"
        );
    }

    /// Covariance costs a download-path user nothing unless they ask for it.
    ///
    /// This is the promise the whole optional-section design rests on: the
    /// matrices are megabytes, and a run that does not want uncertainty must
    /// fetch the same objects it always did. Downloads are per section, so the
    /// list below IS the request, and a section missing from it is never
    /// fetched. Worth pinning rather than reasoning about, because adding a
    /// file to the wrong constant would raise every user's download silently.
    #[test]
    fn covariance_is_fetched_only_when_the_scope_asks_for_it() {
        let names = |scope: &crate::LoadScope| -> Vec<&str> {
            sections_for(DataKind::Neutron, scope)
                .iter()
                .map(|(n, _)| *n)
                .collect()
        };

        // The default paths, unchanged.
        assert!(!names(&crate::LoadScope::activation([102].into())).contains(&"covariance.arrow"));
        assert!(!names(&crate::LoadScope::full()).contains(&"covariance.arrow"));

        // And only when asked, on either scope.
        let xs = crate::LoadScope::activation([102].into()).with_covariance(true);
        assert_eq!(
            names(&xs),
            [
                "version.json",
                "nuclide.arrow",
                "energy.arrow",
                "reactions.arrow",
                "covariance.arrow"
            ]
        );
        assert!(
            names(&crate::LoadScope::full().with_covariance(true)).contains(&"covariance.arrow")
        );

        // Optional, so a nuclide whose evaluation has no MF=33 answers 404 and
        // the miss is recorded rather than retried on every later load.
        let cov = sections_for(DataKind::Neutron, &xs)
            .iter()
            .find(|(n, _)| *n == "covariance.arrow")
            .copied();
        assert_eq!(cov, Some(("covariance.arrow", false)));
    }

    /// MF=34 is fetched only when asked for, on its own axis: an uncertainty
    /// run that wants MF=33 does not pull it, and it is optional.
    #[test]
    fn angular_covariance_is_fetched_only_when_the_scope_asks_for_it() {
        let names = |scope: &crate::LoadScope| -> Vec<&str> {
            sections_for(DataKind::Neutron, scope)
                .iter()
                .map(|(n, _)| *n)
                .collect()
        };
        let angular = "angular_covariance.arrow";
        assert!(!names(&crate::LoadScope::full()).contains(&angular));
        let mf33 = crate::LoadScope::activation([102].into()).with_covariance(true);
        assert!(!names(&mf33).contains(&angular));
        let asked = crate::LoadScope::full().with_angular_covariance(true);
        assert!(names(&asked).contains(&angular));
        assert!(sections_for(DataKind::Neutron, &asked).contains(&(angular, false)));
        assert!(
            !sections_for(DataKind::Photon, &asked)
                .iter()
                .any(|(n, _)| *n == angular),
            "photon data has no MF=34"
        );
    }

    /// MF=31 and MF=35 are fetched only when asked for, together, on their
    /// own axis: neither transport, nor an uncertainty run that wants MF=33 or
    /// MF=34, pulls them, and both are optional.
    #[test]
    fn fission_covariance_is_fetched_only_when_the_scope_asks_for_it() {
        let names = |scope: &crate::LoadScope| -> Vec<&str> {
            sections_for(DataKind::Neutron, scope)
                .iter()
                .map(|(n, _)| *n)
                .collect()
        };
        let files = ["nubar_covariance.arrow", "spectrum_covariance.arrow"];
        let other_axes = crate::LoadScope::full()
            .with_covariance(true)
            .with_angular_covariance(true);
        for scope in [crate::LoadScope::full(), other_axes] {
            let got = names(&scope);
            assert!(files.iter().all(|f| !got.contains(f)), "{got:?}");
        }
        let asked = crate::LoadScope::activation([102].into()).with_fission_covariance(true);
        for file in files {
            assert!(sections_for(DataKind::Neutron, &asked).contains(&(file, false)));
            assert!(
                !sections_for(DataKind::Photon, &asked)
                    .iter()
                    .any(|(n, _)| *n == file),
                "photon data has no {file}"
            );
        }
    }

    /// MF=2 and MF=32 are fetched only when asked for, on their own axis: no
    /// other scope pulls them, however wide, and the file is optional.
    #[test]
    fn resonance_parameters_are_fetched_only_when_the_scope_asks_for_them() {
        let file = "resonance_parameters.arrow";
        let every_other_axis = crate::LoadScope::full()
            .with_covariance(true)
            .with_angular_covariance(true)
            .with_fission_covariance(true);
        for scope in [crate::LoadScope::full(), every_other_axis] {
            assert!(
                !sections_for(DataKind::Neutron, &scope)
                    .iter()
                    .any(|(n, _)| *n == file),
                "{scope:?} fetched {file}"
            );
        }
        let asked = crate::LoadScope::activation([102].into()).with_resonance_parameters(true);
        assert!(sections_for(DataKind::Neutron, &asked).contains(&(file, false)));
        assert!(
            !sections_for(DataKind::Photon, &asked)
                .iter()
                .any(|(n, _)| *n == file),
            "photon data has no {file}"
        );
    }

    /// Every other layer ships the fission photon release (the
    /// converter writes it, the schema declares it, the Arrow reader parses
    /// it), but a download-path user only ever sees a section named in this
    /// list. Left out, the read silently returns `None` and actinide fission
    /// photon production stays ~38% low.
    #[test]
    fn a_transport_fetch_asks_for_the_fission_photon_section() {
        let full = sections_for(DataKind::Neutron, &crate::LoadScope::full());
        let (_, required) = full
            .iter()
            .find(|(name, _)| *name == "fission_photon.arrow")
            .expect("transport reads fission_photon.arrow, so it must be fetched");
        assert!(
            !required,
            "only an evaluation with fission_energy_release publishes it, so a \
             404 is expected everywhere else and must settle as an absent marker"
        );
    }

    #[test]
    fn a_transport_load_tops_up_what_a_transmutation_load_left() {
        let dir = TempDir::new("topup");
        let xs_only = sections_for(
            DataKind::Neutron,
            &crate::LoadScope::activation([102].into()),
        );
        let full = sections_for(DataKind::Neutron, &crate::LoadScope::full());

        // Nothing cached yet: both scopes want all of their own sections.
        let empty = dir.0.join("missing.arrow");
        assert_eq!(
            sections_to_fetch(&empty, &xs_only, Ranged::Whole).len(),
            xs_only.len()
        );
        assert!(!have_all_sections(&empty, &xs_only, Ranged::Whole));

        // Simulate the transmutation load having run.
        for (name, _) in &xs_only {
            dir.touch(name);
        }
        assert!(
            have_all_sections(&dir.0, &xs_only, Ranged::Whole),
            "the activation scope is satisfied"
        );
        assert!(
            !have_all_sections(&dir.0, &full, Ranged::Whole),
            "but transport still needs the rest"
        );

        // The follow-up transport fetch asks only for what is missing, which is
        // the whole point: the earlier three are not refetched.
        let todo: Vec<&str> = sections_to_fetch(&dir.0, &full, Ranged::Whole)
            .iter()
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(
            todo,
            [
                "products.arrow",
                "distributions.arrow",
                "urr.arrow",
                "total_nu.arrow",
                "fission_photon.arrow"
            ]
        );
    }

    #[test]
    fn an_absent_marker_settles_an_optional_section() {
        let dir = TempDir::new("absent");
        let full = sections_for(DataKind::Neutron, &crate::LoadScope::full());
        for (name, required) in &full {
            // Fe56 has no URR, nu or fission-photon tables upstream: those 404
            // and get a marker.
            if *required {
                dir.touch(name);
            }
        }
        assert!(
            !have_all_sections(&dir.0, &full, Ranged::Whole),
            "an unmarked optional section still looks unfetched"
        );

        dir.touch(&format!("urr.arrow{ABSENT_SUFFIX}"));
        dir.touch(&format!("total_nu.arrow{ABSENT_SUFFIX}"));
        dir.touch(&format!("fission_photon.arrow{ABSENT_SUFFIX}"));
        assert!(
            have_all_sections(&dir.0, &full, Ranged::Whole),
            "a marker means the origin answered 404, so stop asking"
        );
        assert!(sections_to_fetch(&dir.0, &full, Ranged::Whole).is_empty());
    }
}

#[cfg(test)]
mod cache_root_tests {
    use super::{cache_root, default_cache_root, home_dir};

    /// The root is the platform cache directory with `yamc` appended, and sits
    /// under the home directory on every platform CI runs on, including the
    /// Windows runner where `HOME` is unset and the known-folder API carries
    /// it. A `HOME`-only read is what returned nothing there.
    #[test]
    fn this_machine_has_a_home_and_therefore_a_cache_root() {
        let home = home_dir().expect("no home directory resolved");
        let root = default_cache_root().expect("no cache root resolved");
        assert_eq!(root.file_name().and_then(|n| n.to_str()), Some("yamc"));
        assert!(cache_root().is_some());
        if cfg!(target_os = "macos") {
            assert_eq!(root, home.join("Library").join("Caches").join("yamc"));
        } else if cfg!(windows) {
            assert!(
                root.ends_with("AppData/Local/yamc") || root.ends_with("AppData\\Local\\yamc"),
                "{root:?}"
            );
        } else if std::env::var_os("XDG_CACHE_HOME").is_none() {
            assert_eq!(root, home.join(".cache").join("yamc"));
        }
    }
}

#[cfg(test)]
mod branching_only_tests {
    use super::{
        cross_section_keywords, is_branching_only, is_keyword, resolve_path_or_url,
        resolve_subsection, DataKind,
    };

    /// IRDFF-II is a keyword, so it resolves to its published release, and a
    /// branching-only one, so it is offered nowhere but as a branching source.
    #[test]
    fn irdff_ii_is_a_branching_only_keyword() {
        assert!(is_keyword("irdff-ii"));
        assert!(is_branching_only("irdff-ii"));
        assert!(!is_branching_only("endf-b8.1"));
        assert!(!cross_section_keywords().contains(&"irdff-ii"));
        assert!(cross_section_keywords().contains(&"endf-b8.1"));
    }

    /// Asked for cross sections or for another subsection, it is refused
    /// before anything is fetched, with a message saying what it is for. No
    /// network is involved: the refusal comes first.
    #[test]
    fn irdff_ii_is_refused_as_anything_but_a_branching_source() {
        let scope = crate::load_scope::LoadScope::full();
        for kind in [DataKind::Neutron, DataKind::Photon] {
            let err = resolve_path_or_url("irdff-ii", "Fe56", kind, &scope)
                .expect_err("not a cross-section library")
                .to_string();
            assert!(err.contains("supplies isomeric branching only"), "{err}");
            assert!(err.contains("transmutation_branch_ratios"), "{err}");
        }
        for subsection in ["decay", "reactions", "fission_yields"] {
            let err = resolve_subsection("irdff-ii", subsection)
                .expect_err("not that subsection's source")
                .to_string();
            assert!(err.contains("supplies isomeric branching only"), "{err}");
            assert!(err.contains(subsection), "{err}");
        }
    }
}

#[cfg(all(test, feature = "download"))]
mod fetch_retry_tests {
    use super::{fetch, Fetched};
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// How a connection the test origin accepts is answered.
    #[derive(Clone, Copy)]
    enum Fault {
        /// Close without a response, as an origin does to an idle pooled
        /// connection.
        Drop,
        /// Promise more bytes than are sent, then close.
        Truncate,
    }

    /// A one-object HTTP origin whose first `faults` connections fail the
    /// given way and every later one is answered in full. Returns the URL and
    /// the count of connections accepted.
    fn origin(faults: usize, fault: Fault, body: &'static [u8]) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/section.arrow", listener.local_addr().unwrap());
        let accepted = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&accepted);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream: TcpStream = stream.unwrap();
                let n = seen.fetch_add(1, Ordering::SeqCst);
                let mut request = [0u8; 4096];
                let _ = stream.read(&mut request);
                if n < faults {
                    if let Fault::Truncate = fault {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len() + 100
                        );
                        let _ = stream.write_all(head.as_bytes());
                        let _ = stream.write_all(body);
                    }
                    continue;
                }
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body);
            }
        });
        (url, accepted)
    }

    fn body_of(fetched: Fetched) -> Vec<u8> {
        match fetched {
            Fetched::Body { bytes, .. } => bytes,
            Fetched::Absent => panic!("expected a body"),
        }
    }

    #[test]
    fn a_dropped_connection_is_retried() {
        let (url, accepted) = origin(2, Fault::Drop, b"sections");
        assert_eq!(body_of(fetch(&url, &[]).unwrap()), b"sections");
        assert_eq!(accepted.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn a_truncated_body_is_retried() {
        let (url, accepted) = origin(1, Fault::Truncate, b"sections");
        assert_eq!(body_of(fetch(&url, &[]).unwrap()), b"sections");
        assert_eq!(accepted.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_persistent_failure_gives_up_and_says_why() {
        let (url, accepted) = origin(usize::MAX, Fault::Drop, b"sections");
        let err = fetch(&url, &[])
            .err()
            .expect("every attempt fails")
            .to_string();
        assert!(err.contains("after 4 attempts"), "{err}");
        assert_eq!(accepted.load(Ordering::SeqCst), 4);
    }
}
