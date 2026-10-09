#[cfg(feature = "download")]
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[cfg(feature = "download")]
use std::fs;

#[cfg(feature = "download")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "download")]
use once_cell::sync::Lazy;

#[cfg(feature = "download")]
use nuclear_data_schema::reaction_ranges::{splice_spans, ReactionRanges};

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
#[cfg(feature = "download-tls")]
macro_rules! data_origin {
    () => {
        "https://yamc-data.xsplot.com/"
    };
}
#[cfg(all(feature = "download", not(feature = "download-tls")))]
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
#[cfg(feature = "download")]
static CLIENT: Lazy<reqwest::Result<reqwest::blocking::Client>> =
    Lazy::new(|| super::release_cache::build_client(super::release_cache::Timeouts::DEFAULT));

/// The shared client, or the error it failed to build with.
#[cfg(feature = "download")]
fn client() -> Result<&'static reqwest::blocking::Client, Box<dyn std::error::Error>> {
    CLIENT
        .as_ref()
        .map_err(|e| format!("could not build an HTTP client: {e}").into())
}

/// Issue a blocking GET over the shared client, optionally for one byte range.
///
/// `span` is `(offset, length)`; the header is inclusive of both ends, so the
/// last byte is `offset + length - 1`.
#[cfg(feature = "download")]
fn blocking_get(
    url: &str,
    span: Option<(u64, u64)>,
) -> Result<reqwest::blocking::Response, Box<dyn std::error::Error>> {
    let mut request = client()?.get(url);
    if let Some((offset, len)) = span {
        request = request.header(
            reqwest::header::RANGE,
            format!("bytes={}-{}", offset, offset + len - 1),
        );
    }
    Ok(request.send()?)
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
/// - `fendl-3.2d`: neutron and photon cross sections, no transmutation data.
/// - `endf-b8.1`: neutron and photon (with atomic relaxation) cross sections
///   and all four transmutation subsections.
/// - `jeff-4.0`: neutron cross sections and all four transmutation subsections.
///   Its photon sublibrary is photonuclear data, not the photoatomic and
///   relaxation pair the photon loader reads, so it publishes none.
/// - `jendl-5.0`: neutron data (most evaluations to 200 MeV), the EPICS2017
///   photon pair, and all four transmutation subsections.
pub(crate) const KEYWORDS: &[&str] = &[
    "tendl-2025",
    "tendl-2017",
    "fendl-3.2d",
    "endf-b8.1",
    "jeff-4.0",
    "jendl-5.0",
];

/// Library keyword used as the default PHOTON data source when the user has
/// not explicitly specified one. endf-b8.1 is chosen because it ships full
/// atomic-relaxation tables (binding energies + fluorescence / Auger
/// transitions), so photon transport produces fluorescence and Auger
/// electrons by default. Other libraries (e.g. fendl-3.2d) publish photon
/// cross sections with no atomic relaxation, which would silently suppress
/// fluorescence. Decoupling the photon default from the neutron library lets
/// the neutron data stay tendl/fendl while photons still get relaxation.
pub const DEFAULT_PHOTON_LIBRARY: &str = "endf-b8.1";

/// Which particle's data file to resolve under a library keyword. Different
/// libraries lay out their files differently; the loader uses this to pick
/// the right per-particle subdirectory under `url_stem`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataKind {
    Neutron,
    Photon,
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
#[cfg(feature = "download")]
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
/// - Path to a converter root (contains a `{subsection}/` subdir): return that
///   subdir.
/// - Path already pointing at the subsection dir: return it as-is.
#[cfg(feature = "download")]
pub fn resolve_subsection(
    source: &str,
    subsection: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if is_keyword(source) {
        Ok(super::release_cache::fetch_subsection(
            &super::release_cache::REGISTRY,
            source,
            subsection,
        )?)
    } else {
        let base = PathBuf::from(source);
        let nested = base.join(subsection);
        Ok(if nested.is_dir() { nested } else { base })
    }
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
    // Which MTs, if this load reads only some of them. Decides whether
    // reactions.arrow is fetched whole or as the byte ranges those MTs occupy.
    let subset = subset_mts(kind, scope);

    // Fast path: cache hit without any locking. The gate is per-section rather
    // than per-directory, because a cache dir populated by an earlier
    // transmutation load holds only some of what transport now wants. Each
    // section is published by rename, so a section that is present
    // is complete.
    if have_all_sections(&local_path, sections, subset) {
        return Ok(local_path);
    }

    // Serialize concurrent downloads of the same nuclide. Re-check after
    // acquiring the lock in case another thread finished the download while
    // we were waiting.
    let path_lock = get_path_lock(&local_path);
    let _guard = path_lock.lock().unwrap_or_else(|p| p.into_inner());
    if have_all_sections(&local_path, sections, subset) {
        return Ok(local_path);
    }

    download_sections(url, &local_path, sections, source, nuclide_name, subset)?;
    Ok(local_path)
}

/// The error for a required section a raw URL answered 404 for, which usually
/// means the nuclide itself is not published there.
#[cfg(feature = "download")]
fn download_error(
    url: &str,
    status: reqwest::StatusCode,
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
pub(crate) fn sections_for(kind: DataKind, scope: &crate::load_scope::LoadScope) -> Vec<(&'static str, bool)> {
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

/// The MTs a scope wants out of `reactions.arrow`, when it wants only some.
///
/// `None` means fetch the section whole, which is every case that is not a
/// transport-free load with a named MT set: photon data, transport, and an
/// activation load that asked for every MT.
#[cfg(feature = "download")]
fn subset_mts(kind: DataKind, scope: &crate::load_scope::LoadScope) -> Option<&HashSet<i32>> {
    match kind {
        DataKind::Neutron if !scope.wants_transport_sections() => scope.mts.as_ref(),
        _ => None,
    }
}

/// The byte-range index in a nuclide directory's `version.json`.
///
/// `None` whenever ranging is not possible: no marker yet, unreadable JSON, or
/// data published before the index existed. Every one of those means the same
/// thing to the caller, which is to fetch the section whole.
#[cfg(feature = "download")]
fn cached_index(dir: &std::path::Path) -> Option<ReactionRanges> {
    let text = fs::read_to_string(dir.join("version.json")).ok()?;
    ReactionRanges::from_version_json(&serde_json::from_str(&text).ok()?)
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

/// Whether `dir` already holds every one of `sections`.
///
/// Per-section rather than "does the directory exist": a dir populated by an
/// earlier transmutation load holds only some of what transport now wants.
#[cfg(feature = "download")]
fn have_all_sections(
    dir: &std::path::Path,
    sections: &[(&str, bool)],
    subset: Option<&HashSet<i32>>,
) -> bool {
    dir.is_dir()
        && sections.iter().all(|(name, _)| match subset {
            Some(wanted) if *name == REACTIONS => reactions_covered(dir, wanted),
            _ => section_is_resolved(dir, name),
        })
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
    subset: Option<&HashSet<i32>>,
) -> Vec<(&'a str, bool)> {
    let existing = dir.is_dir();
    sections
        .iter()
        .copied()
        .filter(|(name, _)| {
            !existing
                || match subset {
                    Some(wanted) if *name == REACTIONS => !reactions_covered(dir, wanted),
                    _ => !section_is_resolved(dir, name),
                }
        })
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

/// Fetch `url`, optionally just one byte range, retrying transient failures.
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
fn fetch(url: &str, span: Option<(u64, u64)>) -> Result<Fetched, Box<dyn std::error::Error>> {
    const RETRY_DELAYS_MS: &[u64] = &[200, 500, 1000];
    // A client that failed to build will not build on a retry either.
    client()?;
    let mut last_failure = String::new();
    for &delay_ms in std::iter::once(&0u64).chain(RETRY_DELAYS_MS.iter()) {
        if delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }
        let r = match blocking_get(url, span) {
            Ok(r) => r,
            Err(e) => {
                last_failure = e.to_string();
                continue;
            }
        };
        let status = r.status();
        if status.is_success() {
            match r.bytes() {
                Ok(bytes) => {
                    return Ok(Fetched::Body {
                        partial: status == reqwest::StatusCode::PARTIAL_CONTENT,
                        bytes: bytes.to_vec(),
                    })
                }
                Err(e) => {
                    last_failure = format!("reading the response body: {e}");
                    continue;
                }
            }
        }
        if status == reqwest::StatusCode::NOT_FOUND {
            // Definitively absent: R2 answers authoritatively, no retry.
            return Ok(Fetched::Absent);
        }
        last_failure = status.to_string();
    }
    Err(format!(
        "Failed to download {} after {} attempts: {}",
        url,
        RETRY_DELAYS_MS.len() + 1,
        last_failure
    )
    .into())
}

/// Fetch just the MTs `wanted` names out of `reactions.arrow`, into `staging`.
///
/// Writes `subset/reactions.arrow` (a spliced Arrow IPC stream) and
/// `subset/mts.json` (the MTs it holds), and returns the staged relative paths.
///
/// `Ok(None)` means the caller should fetch the section whole after all, which
/// happens two ways: the origin ignored the `Range` header and answered 200
/// with the entire object, or the spans coalesced into so much of the file that
/// ranging buys nothing. Both are normal, and neither is worth a second attempt.
#[cfg(feature = "download")]
fn fetch_reactions_subset(
    base_url: &str,
    staging: &std::path::Path,
    index: &ReactionRanges,
    wanted: &HashSet<i32>,
) -> Result<Option<Vec<String>>, Box<dyn std::error::Error>> {
    let url = format!("{base_url}/{REACTIONS}");
    let spans = index.spans_for(|mt| wanted.contains(&mt));

    let mut bodies = Vec::with_capacity(spans.len());
    for span in &spans {
        match fetch(&url, Some(*span))? {
            Fetched::Body {
                bytes,
                partial: true,
            } => {
                if bytes.len() as u64 != span.1 {
                    // A short span is not recoverable by splicing it: the
                    // framing still walks, and the message header at the cut is
                    // read as whatever the next bytes happen to be.
                    return Err(format!(
                        "{url}: asked for {} bytes at {} and got {}",
                        span.1,
                        span.0,
                        bytes.len()
                    )
                    .into());
                }
                bodies.push(bytes);
            }
            // Range ignored somewhere in the path: what is in hand IS the whole
            // object, so cache it as one rather than splicing it as a slice.
            Fetched::Body {
                bytes,
                partial: false,
            } => {
                fs::write(staging.join(REACTIONS), bytes)?;
                return Ok(None);
            }
            // The nuclide has a version.json naming ranges but no
            // reactions.arrow to range into, which is a broken publish rather
            // than an absent optional section.
            Fetched::Absent => {
                return Err(format!("{url}: 404, but its version.json names byte ranges").into())
            }
        }
    }

    fs::create_dir_all(staging.join(SUBSET_DIR))?;
    fs::write(staging.join(SUBSET_REACTIONS), splice_spans(&bodies))?;
    // The MTs the nuclide actually published, not the ones asked for. Recording
    // the request would leave any MT this nuclide has no channel for looking
    // uncovered on every later load.
    fs::write(
        staging.join(SUBSET_MTS),
        serde_json::to_vec(&index.present(|mt| wanted.contains(&mt)))?,
    )?;
    Ok(Some(vec![
        SUBSET_REACTIONS.to_string(),
        SUBSET_MTS.to_string(),
    ]))
}

/// Fetch one whole section object to `dest`. Returns `Ok(true)` if written,
/// `Ok(false)` on a definitive 404 (absent optional section), `Err` on any
/// other failure after retrying transient (5xx / transport) errors.
#[cfg(feature = "download")]
fn fetch_section_to_file(
    url: &str,
    dest: &std::path::Path,
) -> Result<bool, Box<dyn std::error::Error>> {
    match fetch(url, None)? {
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
    subset: Option<&HashSet<i32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let existing = target_dir.is_dir();
    let wanted = sections_to_fetch(target_dir, sections, subset);
    if wanted.is_empty() {
        return Ok(());
    }

    println!(
        "Downloading {} section(s) to cache: {} -> {:?}",
        wanted.len(),
        base_url,
        target_dir
    );
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
            // reactions.arrow, for a load that named its MTs, is fetched as the
            // few byte ranges those MTs occupy. The index comes from
            // version.json, which is first in every section list and so is
            // either already staged by this loop or already in the cache dir.
            if let (Some(mts), true) = (subset, *name == REACTIONS) {
                let index = cached_index(&staging).or_else(|| cached_index(target_dir));
                if let Some(index) = index {
                    if let Some(paths) = fetch_reactions_subset(base_url, &staging, &index, mts)? {
                        staged.extend(paths);
                        continue;
                    }
                    // Fell back to the whole object, already staged under the
                    // canonical name by the fetch above.
                    staged.push((*name).to_string());
                    continue;
                }
                // No index: data published before it existed. Fetch it whole,
                // exactly as this did before.
            }
            let url = format!("{}/{}", base_url, name);
            let fetched = fetch_section_to_file(&url, &staging.join(name))?;
            if fetched {
                staged.push((*name).to_string());
            } else if *required {
                // A missing required section on the first request usually means
                // the nuclide itself is absent from the library.
                return Err(download_error(
                    &url,
                    reqwest::StatusCode::NOT_FOUND,
                    source,
                    nuclide_name,
                ));
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
                // The whole object supersedes any subset beside it: the reader
                // prefers it, so what is left is dead bytes and a second copy of
                // the same cross sections to puzzle over in a cache directory.
                // Dropped after the rename, so a failure above leaves the subset
                // in place and the directory still usable.
                if staged.iter().any(|name| name == REACTIONS) {
                    fs::remove_dir_all(target_dir.join(SUBSET_DIR)).ok();
                }
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
    Some(etcetera::base_strategy::choose_native_strategy().ok()?.cache_dir().join("yamc"))
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
/// keyword: the release identifier, its manifest sha256, and whether the
/// origin was unreachable so a cached release was used. Results record this as
/// their nuclear-data provenance. Empty in a build without downloads.
pub fn data_releases() -> std::collections::BTreeMap<String, super::release::DataRelease> {
    #[cfg(feature = "download")]
    {
        super::release_cache::data_releases()
    }
    #[cfg(not(feature = "download"))]
    {
        std::collections::BTreeMap::new()
    }
}

/// Get the cache directory for yamc, creating it if it does not exist.
#[cfg(feature = "download")]
pub fn get_cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = cache_root().ok_or(
        "Could not find a cache directory: the platform resolved no home directory",
    )?;

    // Create the cache directory if it doesn't exist
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)?;
    }

    Ok(cache_dir)
}

/// Resolve a nuclide name within a directory.
/// Checks for Arrow directory.
fn resolve_nuclide_in_dir(
    dir: &std::path::Path,
    nuclide_name: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let arrow_path = dir.join(format!("{}.arrow", nuclide_name));
    Ok(arrow_path)
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
/// If it's a directory, resolve to `<dir>/<nuclide_name>.arrow`.
/// If it's a URL, download and cache it.
/// If it's a local file path, return as-is.
#[cfg(feature = "download")]
pub fn resolve_path_or_url(
    path_url_or_keyword: &str,
    nuclide_name: &str,
    kind: DataKind,
    scope: &crate::load_scope::LoadScope,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
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
            resolve_nuclide_in_dir(&dir, nuclide_name)
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
    if is_keyword(source) {
        Err(format!(
            "Library keyword '{source}' cannot be resolved to a transmutation \
             subsection in WASM builds. Please pass a local path."
        )
        .into())
    } else {
        let base = PathBuf::from(source);
        let nested = base.join(subsection);
        Ok(if nested.is_dir() { nested } else { base })
    }
}

/// WASM fallback - only supports local paths and directories, not URLs or keywords
#[cfg(not(feature = "download"))]
#[allow(dead_code)]
pub fn resolve_path_or_url(
    path_url_or_keyword: &str,
    nuclide_name: &str,
    _kind: DataKind,
    _scope: &crate::load_scope::LoadScope,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if is_keyword(path_url_or_keyword) {
        Err("URL downloading and keywords are not supported in WASM builds. Please use local file paths.".into())
    } else if std::path::Path::new(path_url_or_keyword).is_dir() {
        let dir = PathBuf::from(path_url_or_keyword);
        // If this IS already an Arrow nuclide directory, return it directly
        if dir.extension().and_then(|e| e.to_str()) == Some("arrow") {
            Ok(dir)
        } else {
            // It's a parent directory - resolve to nuclide data file/directory inside it
            resolve_nuclide_in_dir(&dir, nuclide_name)
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
            !have_all_sections(&dir.0, &full, None),
            "transport must not be satisfied by a subset"
        );
        let todo: Vec<&str> = sections_to_fetch(&dir.0, &full, None)
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
        assert!(have_all_sections(&dir.0, &xs_only, Some(&wanted)));
        assert!(sections_to_fetch(&dir.0, &xs_only, Some(&wanted)).is_empty());
    }

    /// Only a neutron load that named its MTs may range. Photon data has no
    /// per-MT batches, and transport reads every MT there is.
    #[test]
    fn only_a_named_activation_neutron_scope_ranges() {
        let wanted = mts(&[102]);
        assert_eq!(
            subset_mts(
                DataKind::Neutron,
                &crate::LoadScope::activation(wanted.clone())
            ),
            Some(&wanted),
        );
        assert!(subset_mts(DataKind::Neutron, &crate::LoadScope::full()).is_none());
        assert!(subset_mts(DataKind::Photon, &crate::LoadScope::activation(wanted)).is_none());
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
        assert!(subset_mts(DataKind::Neutron, &all_mts).is_none());
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
            sections_to_fetch(&empty, &xs_only, None).len(),
            xs_only.len()
        );
        assert!(!have_all_sections(&empty, &xs_only, None));

        // Simulate the transmutation load having run.
        for (name, _) in &xs_only {
            dir.touch(name);
        }
        assert!(
            have_all_sections(&dir.0, &xs_only, None),
            "the activation scope is satisfied"
        );
        assert!(
            !have_all_sections(&dir.0, &full, None),
            "but transport still needs the rest"
        );

        // The follow-up transport fetch asks only for what is missing, which is
        // the whole point: the earlier three are not refetched.
        let todo: Vec<&str> = sections_to_fetch(&dir.0, &full, None)
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
            !have_all_sections(&dir.0, &full, None),
            "an unmarked optional section still looks unfetched"
        );

        dir.touch(&format!("urr.arrow{ABSENT_SUFFIX}"));
        dir.touch(&format!("total_nu.arrow{ABSENT_SUFFIX}"));
        dir.touch(&format!("fission_photon.arrow{ABSENT_SUFFIX}"));
        assert!(
            have_all_sections(&dir.0, &full, None),
            "a marker means the origin answered 404, so stop asking"
        );
        assert!(sections_to_fetch(&dir.0, &full, None).is_empty());
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
            assert!(root.ends_with("AppData/Local/yamc") || root.ends_with("AppData\\Local\\yamc"), "{root:?}");
        } else if std::env::var_os("XDG_CACHE_HOME").is_none() {
            assert_eq!(root, home.join(".cache").join("yamc"));
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
        assert_eq!(body_of(fetch(&url, None).unwrap()), b"sections");
        assert_eq!(accepted.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn a_truncated_body_is_retried() {
        let (url, accepted) = origin(1, Fault::Truncate, b"sections");
        assert_eq!(body_of(fetch(&url, None).unwrap()), b"sections");
        assert_eq!(accepted.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_persistent_failure_gives_up_and_says_why() {
        let (url, accepted) = origin(usize::MAX, Fault::Drop, b"sections");
        let err = fetch(&url, None)
            .err()
            .expect("every attempt fails")
            .to_string();
        assert!(err.contains("after 4 attempts"), "{err}");
        assert_eq!(accepted.load(Ordering::SeqCst), 4);
    }
}
