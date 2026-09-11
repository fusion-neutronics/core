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
fn get_path_lock(path: &std::path::Path) -> Arc<Mutex<()>> {
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
fn ensure_tls_provider() {
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
#[cfg(feature = "download")]
static CLIENT: Lazy<reqwest::Result<reqwest::blocking::Client>> = Lazy::new(|| {
    #[cfg(feature = "download-tls")]
    ensure_tls_provider();
    reqwest::blocking::Client::builder().build()
});

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

/// All recognized keywords -- keep in sync with `get_keyword_info_mapping`.
const KEYWORDS: &[&str] = &[
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

/// Metadata for a keyword-based data source
#[cfg(feature = "download")]
#[derive(Debug, Clone, Copy)]
struct KeywordInfo {
    url_stem: &'static str,
    /// Subdirectory under `url_stem` for per-nuclide neutron data. Empty when
    /// the library uses a flat layout (`<Nuclide>.arrow/` section dirs directly
    /// under `url_stem`).
    neutron_subdir: &'static str,
    /// Subdirectory under `url_stem` for per-element photon data.
    photon_subdir: &'static str,
}

/// Get the mapping of keywords to their download info
#[cfg(feature = "download")]
fn get_keyword_info_mapping() -> HashMap<&'static str, KeywordInfo> {
    let mut map = HashMap::new();
    map.insert(
        "tendl-2025",
        KeywordInfo {
            // Hosted on Cloudflare R2 behind the xsplot.com custom domain, the
            // same host as endf-b8.1. Option-D layout: per-section objects under
            // `tendl-2025/neutron/<Name>.arrow/` (nuclide.arrow, reactions.arrow,
            // ...). TENDL is neutron-only: no photon subdir. No index file is
            // published -- unavailable nuclides surface as a download 404.
            url_stem: concat!(data_origin!(), "tendl-2025/"),
            neutron_subdir: "neutron/",
            photon_subdir: "",
        },
    );
    map.insert(
        "tendl-2017",
        KeywordInfo {
            // Same R2 host/layout as tendl-2025. Publishes per-nuclide neutron
            // cross sections (`tendl-2017/neutron/<Name>.arrow/` section dirs,
            // full endf-b8.1-parity coverage) plus the isomeric-branching and
            // reactions transmutation subsections under `transmutation/`. No
            // photon data. Library-matched to the TENDL-2017 activation chains,
            // so it is the apples-to-apples transport library for FISPACT-II
            // comparisons (e.g. the FNS decay-heat benchmark).
            url_stem: concat!(data_origin!(), "tendl-2017/"),
            neutron_subdir: "neutron/",
            photon_subdir: "",
        },
    );
    map.insert(
        "fendl-3.2d",
        KeywordInfo {
            // Hosted on Cloudflare R2 behind the xsplot.com custom domain, the
            // same host as endf-b8.1. FENDL 3.2d ships neutron + photon data.
            // Option-D layout: `fendl-3.2d/{neutron,photon}/<Name>.arrow/`
            // section dirs. No index file -- unavailable nuclides surface as a
            // download 404.
            url_stem: concat!(data_origin!(), "fendl-3.2d/"),
            neutron_subdir: "neutron/",
            photon_subdir: "photon/",
        },
    );
    map.insert(
        "endf-b8.1",
        KeywordInfo {
            // Hosted on Cloudflare R2 behind a custom domain on the xsplot.com
            // zone; CORS-friendly (ACAO: *) so browser-WASM builds can fetch
            // directly. Option-D layout: `endf-b8.1/{neutron,photon}/<Name>.arrow/`
            // section dirs, plus per-subsection transmutation section dirs under
            // `transmutation/<subsection>.arrow/`.
            url_stem: concat!(data_origin!(), "endf-b8.1/"),
            neutron_subdir: "neutron/",
            photon_subdir: "photon/",
        },
    );
    map.insert(
        "jeff-4.0",
        KeywordInfo {
            // Same R2 host/layout as endf-b8.1. JEFF-4.0 is neutron-only here:
            // its photon sublibrary is photonuclear data adopted from
            // TENDL-2023, not the photoatomic + atomic relaxation pair the
            // photon loader reads, so there is no photon subdir. The
            // transmutation chain is built from JEFF's own decay, fission-yield
            // and neutron sublibraries, so all four subsections are
            // library-consistent.
            url_stem: concat!(data_origin!(), "jeff-4.0/"),
            neutron_subdir: "neutron/",
            photon_subdir: "",
        },
    );
    map.insert(
        "jendl-5.0",
        KeywordInfo {
            // Same R2 host/layout as endf-b8.1. JENDL-5 is the third library
            // here that serves a coupled neutron-photon calculation and a
            // complete chain from one keyword: 795 neutron evaluations, most of
            // them to 200 MeV where the other general-purpose libraries stop at
            // 20, plus a photon pair and its own decay and fission-yield
            // sublibraries.
            //
            // Its photon data is EPICS2017 adopted by JAEA rather than a JAEA
            // evaluation, so it is the same photoatomic and atomic-relaxation
            // pair endf-b8.1 publishes. That makes it a viable photon source
            // rather than a relaxation-free one like fendl-3.2d, though
            // DEFAULT_PHOTON_LIBRARY stays endf-b8.1: identical data is no
            // reason to move the default.
            url_stem: concat!(data_origin!(), "jendl-5.0/"),
            neutron_subdir: "neutron/",
            photon_subdir: "photon/",
        },
    );
    map
}

/// Check if a string is a known keyword
pub fn is_keyword(input: &str) -> bool {
    KEYWORDS.contains(&input)
}

/// Expand a keyword to a full URL for a specific nuclide / element. The
/// particle `kind` selects the per-particle subdirectory under `url_stem`
/// (empty for libraries with a flat layout; all current libraries use a subdir).
#[cfg(feature = "download")]
pub fn expand_keyword_to_url(keyword: &str, nuclide_name: &str, kind: DataKind) -> Option<String> {
    get_keyword_info_mapping().get(keyword).map(|info| {
        let subdir = match kind {
            DataKind::Neutron => info.neutron_subdir,
            DataKind::Photon => info.photon_subdir,
        };
        // Option-D (#224): the base prefix of the tar-free per-section object
        // set (`<stem><subdir><Name>.arrow/`); the downloader appends each
        // section filename. No `.tar` suffix.
        format!("{}{}{}.arrow", info.url_stem, subdir, nuclide_name)
    })
}

/// Expand a keyword to the per-section base URL of one transmutation
/// *subsection* (`{url_stem}transmutation/{subsection}.arrow`), e.g. subsection
/// `"decay"`, `"reactions"`, `"fission_yields"`, or `"branching"`. The
/// downloader appends each section filename (option-D, no `.tar`). Returns
/// `None` for an unknown keyword. A library that does not publish the
/// subsection surfaces as a download 404 at fetch time.
#[cfg(feature = "download")]
pub fn expand_keyword_to_subsection_url(keyword: &str, subsection: &str) -> Option<String> {
    get_keyword_info_mapping()
        .get(keyword)
        .map(|info| format!("{}transmutation/{}.arrow", info.url_stem, subsection))
}

/// Section files published in each transmutation subsection dir (option-D),
/// `(filename, required)`. The primary section is required; auxiliary sections
/// and `provenance.json` are optional and 404 cleanly. Mirrors the filenames
/// the yani chain loader reads. Returns an empty slice for an unknown
/// subsection (callers gate on [`keyword_transmutation_subsections`] first).
#[cfg(feature = "download")]
fn transmutation_sections(subsection: &str) -> &'static [(&'static str, bool)] {
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
            ("provenance.json", false),
        ],
        "branching" => &[("branching.arrow", true), ("provenance.json", false)],
        _ => &[],
    }
}

/// The transmutation subsections a library keyword publishes on Cloudflare R2.
///
/// Returns `None` for an unknown keyword (the caller reports that separately).
/// Used to fail fast with a helpful message when a transmutation part points at
/// a library that does not provide it. Refresh when a release adds subsections.
///
/// - `endf-b8.1` / `jeff-4.0` / `jendl-5.0`: all four parts.
/// - `tendl-2025` / `tendl-2017`: `branching` + `reactions` (TENDL is
///   neutron-only, so it has no decay or fission-yield evaluations).
/// - `fendl-3.2d`: none (cross sections only).
#[cfg(feature = "download")]
fn keyword_transmutation_subsections(keyword: &str) -> Option<&'static [&'static str]> {
    Some(match keyword {
        "endf-b8.1" | "jeff-4.0" | "jendl-5.0" => {
            &["decay", "reactions", "fission_yields", "branching"]
        }
        "tendl-2025" | "tendl-2017" => &["branching", "reactions"],
        "fendl-3.2d" => &[],
        _ => return None,
    })
}

/// Resolve a transmutation subsection source (library keyword or path) to a
/// local directory containing that subsection's arrow files.
///
/// - Keyword (e.g. `"endf-b8.1"`): download+cache the per-section object set
///   under `{url_stem}transmutation/{subsection}.arrow/`.
/// - Path to a converter root (contains a `{subsection}/` subdir): return that
///   subdir.
/// - Path already pointing at the subsection dir: return it as-is.
#[cfg(feature = "download")]
pub fn resolve_subsection(
    source: &str,
    subsection: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if is_keyword(source) {
        // Fail fast with a helpful message if the library does not publish this
        // subsection, rather than letting the download hit a bare 404. TENDL is
        // neutron-only, so it has no decay/fission_yields; fendl-3.2d ships no
        // transmutation data at all.
        if let Some(available) = keyword_transmutation_subsections(source) {
            if !available.contains(&subsection) {
                let provides = if available.is_empty() {
                    "no transmutation subsections".to_string()
                } else {
                    available.join(", ")
                };
                return Err(format!(
                    "Library '{source}' does not provide a '{subsection}' \
                     transmutation subsection (it provides: {provides}). Point this \
                     transmutation part at a library that provides '{subsection}' \
                     (endf-b8.1 provides all parts) or at a local path."
                )
                .into());
            }
        }
        let url = expand_keyword_to_subsection_url(source, subsection)
            .ok_or_else(|| format!("Unknown keyword: {source}"))?;
        let cache_name = format!("{source}-transmutation-{subsection}.arrow");
        download_and_cache_named(
            &url,
            &cache_name,
            source,
            transmutation_sections(subsection),
        )
    } else {
        let base = PathBuf::from(source);
        let nested = base.join(subsection);
        Ok(if nested.is_dir() { nested } else { base })
    }
}

/// Comma-separated list of nuclide and element names published in the
/// endf-b8.1 cross-section release. Embedded at compile time so we can
/// short-circuit guaranteed-404 downloads (issue #46) without a runtime
/// index fetch. Refresh from the R2 bucket index file (currently uploaded
/// alongside the data on <https://yamc-data.xsplot.com/endf-b8.1/>) when
/// the underlying release changes.
#[cfg(feature = "download")]
const ENDF_B81_INDEX: &str = "Ac,Ac225,Ac226,Ac227,Ag,Ag107,Ag108,Ag109,Ag110_m1,Ag111,Ag112,Ag113,Ag114,Ag115,Ag116,Ag117,Ag118_m1,Al,Al26_m1,Al27,Am,Am240,Am241,Am242,Am242_m1,Am243,Am244,Am244_m1,Ar,Ar36,Ar37,Ar38,Ar39,Ar40,Ar41,As,As73,As74,As75,At,Au,Au197,B,B10,B11,Ba,Ba130,Ba131,Ba132,Ba133,Ba134,Ba135,Ba136,Ba137,Ba138,Ba139,Ba140,Be,Be7,Be9,Bi,Bi209,Bi210_m1,Bk,Bk245,Bk246,Bk247,Bk248,Bk249,Bk250,Br,Br79,Br80,Br81,C,C12,C13,Ca,Ca40,Ca41,Ca42,Ca43,Ca44,Ca45,Ca46,Ca47,Ca48,Cd,Cd106,Cd107,Cd108,Cd109,Cd110,Cd111,Cd112,Cd113,Cd114,Cd115_m1,Cd116,Ce,Ce136,Ce137,Ce137_m1,Ce138,Ce139,Ce140,Ce141,Ce142,Ce143,Ce144,Cf,Cf246,Cf247,Cf248,Cf249,Cf250,Cf251,Cf252,Cf253,Cf254,Cl,Cl35,Cl36,Cl37,Cm,Cm240,Cm241,Cm242,Cm243,Cm244,Cm245,Cm246,Cm247,Cm248,Cm249,Cm250,Co,Co58,Co58_m1,Co59,Cr,Cr50,Cr51,Cr52,Cr53,Cr54,Cs,Cs133,Cs134,Cs135,Cs136,Cs137,Cu,Cu63,Cu64,Cu65,Dy,Dy154,Dy155,Dy156,Dy157,Dy158,Dy159,Dy160,Dy161,Dy162,Dy163,Dy164,Er,Er162,Er163,Er164,Er165,Er166,Er167,Er168,Er169,Er170,Es,Es251,Es252,Es253,Es254,Es254_m1,Es255,Eu,Eu151,Eu152,Eu153,Eu154,Eu155,Eu156,Eu157,F,F19,Fe,Fe54,Fe55,Fe56,Fe57,Fe58,Fm,Fm255,Fr,Ga,Ga69,Ga70,Ga71,Gd,Gd152,Gd153,Gd154,Gd155,Gd156,Gd157,Gd158,Gd159,Gd160,Ge,Ge70,Ge71,Ge72,Ge73,Ge74,Ge75,Ge76,H,H1,H2,H3,He,He3,He4,Hf,Hf174,Hf175,Hf176,Hf177,Hf178,Hf179,Hf180,Hf181,Hf182,Hg,Hg196,Hg197,Hg197_m1,Hg198,Hg199,Hg200,Hg201,Hg202,Hg203,Hg204,Ho,Ho165,Ho166_m1,I,I127,I128,I129,I130,I131,I132,I132_m1,I133,I134,I135,In,In113,In114,In115,Ir,Ir191,Ir192,Ir193,Ir194_m1,K,K39,K40,K41,Kr,Kr78,Kr79,Kr80,Kr81,Kr82,Kr83,Kr84,Kr85,Kr86,La,La138,La139,La140,Li,Li6,Li7,Lu,Lu175,Lu176,Mg,Mg24,Mg25,Mg26,Mn,Mn54,Mn55,Mo,Mo100,Mo92,Mo93,Mo94,Mo95,Mo96,Mo97,Mo98,Mo99,N,N14,N15,Na,Na22,Na23,Nb,Nb93,Nb94,Nb95,Nd,Nd142,Nd143,Nd144,Nd145,Nd146,Nd147,Nd148,Nd149,Nd150,Ne,Ne20,Ne21,Ne22,Ni,Ni58,Ni59,Ni60,Ni61,Ni62,Ni63,Ni64,Np,Np234,Np235,Np236,Np236_m1,Np237,Np238,Np239,O,O16,O17,O18,Os,Os184,Os185,Os186,Os187,Os188,Os189,Os190,Os191,Os192,P,P31,Pa,Pa229,Pa230,Pa231,Pa232,Pa233,Pb,Pb204,Pb205,Pb206,Pb207,Pb208,Pd,Pd102,Pd103,Pd104,Pd105,Pd106,Pd107,Pd108,Pd109,Pd110,Pm,Pm143,Pm144,Pm145,Pm146,Pm147,Pm148,Pm148_m1,Pm149,Pm150,Pm151,Po,Po208,Po209,Po210,Pr,Pr141,Pr142,Pr143,Pt,Pt190,Pt191,Pt192,Pt193,Pt194,Pt195,Pt196,Pt197,Pt198,Pu,Pu236,Pu237,Pu238,Pu239,Pu240,Pu241,Pu242,Pu243,Pu244,Pu245,Pu246,Ra,Ra223,Ra224,Ra225,Ra226,Rb,Rb85,Rb86,Rb87,Re,Re185,Re186_m1,Re187,Rh,Rh103,Rh104,Rh105,Rn,Ru,Ru100,Ru101,Ru102,Ru103,Ru104,Ru105,Ru106,Ru96,Ru97,Ru98,Ru99,S,S32,S33,S34,S35,S36,Sb,Sb121,Sb122,Sb123,Sb124,Sb125,Sb126,Sc,Sc45,Se,Se74,Se75,Se76,Se77,Se78,Se79,Se80,Se81,Se82,Si,Si28,Si29,Si30,Si31,Si32,Sm,Sm144,Sm145,Sm146,Sm147,Sm148,Sm149,Sm150,Sm151,Sm152,Sm153,Sm154,Sn,Sn112,Sn113,Sn114,Sn115,Sn116,Sn117,Sn118,Sn119,Sn120,Sn121_m1,Sn122,Sn123,Sn124,Sn125,Sn126,Sr,Sr84,Sr85,Sr86,Sr87,Sr88,Sr89,Sr90,Ta,Ta180,Ta180_m1,Ta181,Ta182,Tb,Tb158,Tb159,Tb160,Tb161,Tc,Tc98,Tc99,Te,Te120,Te121,Te121_m1,Te122,Te123,Te124,Te125,Te126,Te127_m1,Te128,Te129_m1,Te130,Te131,Te131_m1,Te132,Th,Th227,Th228,Th229,Th230,Th231,Th232,Th233,Th234,Ti,Ti46,Ti47,Ti48,Ti49,Ti50,Tl,Tl203,Tl204,Tl205,Tm,Tm168,Tm169,Tm170,Tm171,U,U230,U231,U232,U233,U234,U235,U236,U237,U238,U239,U240,U241,V,V49,V50,V51,W,W180,W181,W182,W183,W184,W185,W186,Xe,Xe123,Xe124,Xe125,Xe126,Xe127,Xe128,Xe129,Xe130,Xe131,Xe132,Xe133,Xe134,Xe135,Xe136,Y,Y89,Y90,Y91,Yb,Yb168,Yb169,Yb170,Yb171,Yb172,Yb173,Yb174,Yb175,Yb176,Zn,Zn64,Zn65,Zn66,Zn67,Zn68,Zn69,Zn70,Zr,Zr90,Zr91,Zr92,Zr93,Zr94,Zr95,Zr96";

/// Comma-separated nuclide + element names published in the fendl-3.2d
/// Cloudflare R2 release (192 neutron nuclides + 61 photon elements).
/// Regenerate from the published data when the release changes.
#[cfg(feature = "download")]
const FENDL_32D_INDEX: &str = "Ag,Ag107,Ag109,Al,Al27,Ar,Ar36,Ar38,Ar40,Au,Au197,B,B10,B11,Ba,Ba130,Ba132,Ba134,Ba135,Ba136,Ba137,Ba138,Be,Be9,Bi,Bi209,Br,Br79,Br81,C,C12,C13,Ca,Ca40,Ca42,Ca43,Ca44,Ca46,Ca48,Cd,Cd106,Cd108,Cd110,Cd111,Cd112,Cd113,Cd114,Cd116,Ce,Ce136,Ce138,Ce140,Ce142,Cl,Cl35,Cl37,Co,Co59,Cr,Cr50,Cr52,Cr53,Cr54,Cs,Cs133,Cu,Cu63,Cu65,Er,Er162,Er164,Er166,Er167,Er168,Er170,F,F19,Fe,Fe54,Fe56,Fe57,Fe58,Ga,Ga69,Ga71,Gd,Gd152,Gd154,Gd155,Gd156,Gd157,Gd158,Gd160,Ge,Ge70,Ge72,Ge73,Ge74,Ge76,H,H1,H2,H3,He,He3,He4,Hf,Hf174,Hf176,Hf177,Hf178,Hf179,Hf180,I,I127,K,K39,K40,K41,La,La138,La139,Li,Li6,Li7,Lu,Lu175,Lu176,Mg,Mg24,Mg25,Mg26,Mn,Mn55,Mo,Mo100,Mo92,Mo94,Mo95,Mo96,Mo97,Mo98,N,N14,N15,Na,Na23,Nb,Nb93,Ne,Ne20,Ne21,Ne22,Ni,Ni58,Ni60,Ni61,Ni62,Ni64,O,O16,O17,O18,P,P31,Pb,Pb204,Pb206,Pb207,Pb208,Pt,Pt190,Pt192,Pt194,Pt195,Pt196,Pt198,Re,Re185,Re187,Rh,Rh103,S,S32,S33,S34,S36,Sb,Sb121,Sb123,Sc,Sc45,Si,Si28,Si29,Si30,Sm,Sm144,Sm147,Sm148,Sm149,Sm150,Sm152,Sm154,Sn,Sn112,Sn114,Sn115,Sn116,Sn117,Sn118,Sn119,Sn120,Sn122,Sn124,Ta,Ta180_m1,Ta181,Th,Th232,Ti,Ti46,Ti47,Ti48,Ti49,Ti50,U,U234,U235,U238,V,V50,V51,W,W180,W182,W183,W184,W186,Y,Y89,Zn,Zn64,Zn66,Zn67,Zn68,Zn70,Zr,Zr90,Zr91,Zr92,Zr94,Zr96";

/// Parsed embedded indexes per keyword. Only libraries whose published
/// list is shipped inside the wheel appear here.
#[cfg(feature = "download")]
static EMBEDDED_INDEX: Lazy<HashMap<&'static str, HashSet<&'static str>>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("endf-b8.1", parse_embedded_index(ENDF_B81_INDEX));
    m.insert("fendl-3.2d", parse_embedded_index(FENDL_32D_INDEX));
    m
});

/// The `data_version` each published library is expected to carry (issue #366).
///
/// The R2 objects are overwritten in place when a library is rebuilt, so the
/// URL and the cache key `{keyword}-{nuclide}.arrow` are identical before and
/// after. Existence alone therefore cannot tell a current cache from a stale
/// one, and after a re-publish a fresh install gets corrected data while an
/// existing install keeps the old data indefinitely with nothing to tell them
/// apart.
///
/// The value here is the release identifier the converter stamps into every
/// `version.json` it writes (`yamc_convert::entry::write_version`, and its
/// transmutation counterpart in `yani-convert`). A cached directory whose stamp
/// differs, or which predates stamping and has none, is evicted and refetched.
///
/// Compiled in rather than fetched, for the same reason as [`EMBEDDED_INDEX`]:
/// the cache-hit path stays a zero-cost offline path with no round trip per
/// nuclide, which matters when a model loads hundreds of them. The cost is that
/// invalidation ships with a yamc release, so a re-publish needs an entry here
/// bumped in the same release to reach existing installs.
///
/// A keyword with no entry is not checked at all, which is the behaviour every
/// library had before this existed. Add an entry when a library is published
/// with a stamp; do not add one speculatively, because an entry whose value no
/// published data carries invalidates every cache on the first load and then
/// fails, by design (see `download_and_cache`).
#[cfg(feature = "download")]
const EXPECTED_DATA_VERSION: &[(&str, &str)] = &[
    // The 2026-09-02 republish, which restamped every library in the same run,
    // so they share a value rather than drifting per library. Confirmed against
    // the origin before pinning, because the warning above is real: an entry no
    // published data carries invalidates every cache on first load and then
    // fails.
    //
    // Swept over neutron for all six, photon for the three that publish it, and
    // every transmutation subsection each library provides (four for endf-b8.1
    // / jeff-4.0 / jendl-5.0, two for the TENDLs, none for fendl-3.2d); all 25
    // published paths read "2026-09-02". jeff-4.0 publishes no photon data at
    // all (404 on element.arrow, not an unstamped directory), so there is
    // nothing to pin for it there.
    //
    // This pin has to move with a republish. The previous value was
    // "2026-09-02", and leaving it behind does not serve stale data: it fails
    // every fresh download outright, because the stamp the origin now carries
    // no longer matches what this build expects.
    //
    // The 2026-09-08 rebuild is the first carrying the placeholder and decay
    // consistency records in the transmutation provenance, isomer excitation
    // energies read from the decay headers, TENDL branching scoped to TENDL's
    // own parents, and photon sections without the heating column this schema
    // stopped declaring. A released wheel pinning 2026-09-02 can read none of
    // it, and one pinning this can read none of what came before, which is the
    // coupling issue #366 accepted and #28 in the generation scripts is about.
    ("tendl-2025", "2026-09-08"),
    ("tendl-2017", "2026-09-08"),
    ("fendl-3.2d", "2026-09-08"),
    ("endf-b8.1", "2026-09-08"),
    ("jeff-4.0", "2026-09-08"),
    ("jendl-5.0", "2026-09-08"),
];

/// The `data_version` this build expects for `source`, if it pins one.
#[cfg(feature = "download")]
fn expected_data_version(source: &str) -> Option<&'static str> {
    EXPECTED_DATA_VERSION
        .iter()
        .find(|(keyword, _)| *keyword == source)
        .map(|(_, version)| *version)
}

/// The marker files a cached directory can record its release stamp in.
///
/// Two layouts, because two writers: a nuclide or element directory carries
/// `version.json` (written last, as the completion marker), while a
/// transmutation subsection carries `provenance.json`. Both are cached through
/// this module and both are invalidated the same way, so the stamp is read from
/// whichever is present rather than duplicating the logic per caller.
#[cfg(feature = "download")]
const STAMP_FILES: &[&str] = &["version.json", "provenance.json"];

/// The `data_version` recorded in a cached directory's marker file.
///
/// `None` covers every way the answer can be "no stamp to compare": no
/// directory, no marker, unreadable or malformed JSON, a `data_version` that is
/// not a string, or a marker written before the field existed. All of them mean
/// the same thing to the caller, and none of them should panic or fail a load
/// on their own.
#[cfg(feature = "download")]
fn cached_data_version(dir: &std::path::Path) -> Option<String> {
    STAMP_FILES.iter().find_map(|marker| {
        let text = fs::read_to_string(dir.join(marker)).ok()?;
        let value: serde_json::Value = serde_json::from_str(&text).ok()?;
        value
            .get("data_version")
            .and_then(|v| v.as_str())
            // An empty stamp is what the converter writes when the build did
            // not supply one, and it means exactly what an absent field means.
            .filter(|v| !v.is_empty())
            .map(str::to_owned)
    })
}

/// Whether a cached directory carries the `data_version` this build expects.
///
/// Split from [`cache_is_current`] so the comparison can be exercised against
/// an explicit expectation without a published library having to exist.
#[cfg(feature = "download")]
fn data_version_matches(dir: &std::path::Path, expected: Option<&str>) -> bool {
    match expected {
        // Nothing pinned for this keyword: unchanged pre-#366 behaviour.
        None => true,
        Some(expected) => cached_data_version(dir).as_deref() == Some(expected),
    }
}

/// Whether the cached copy of `source` at `dir` is the one this build expects.
#[cfg(feature = "download")]
fn cache_is_current(dir: &std::path::Path, source: &str) -> bool {
    data_version_matches(dir, expected_data_version(source))
}

/// Remove a cached directory that holds a different release than this build
/// expects, so the section top-up refetches all of it.
///
/// Eviction rather than overwrite is required: `sections_to_fetch` skips any
/// section already resolved on disk, so a stale but complete directory would
/// otherwise download nothing and keep every stale byte.
///
/// Only ever called with a path built as `cache_dir.join(name)`, and it
/// re-checks that the parent really is the resolved cache directory before
/// removing anything, so a caller mistake cannot turn this into a delete of
/// somewhere else.
#[cfg(feature = "download")]
fn evict_if_stale(dir: &std::path::Path, source: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !dir.is_dir() || cache_is_current(dir, source) {
        return Ok(());
    }
    let cache_dir = get_cache_dir()?;
    if dir.parent() != Some(cache_dir.as_path()) {
        return Err(format!(
            "refusing to evict {}: not a direct child of the cache directory {}",
            dir.display(),
            cache_dir.display()
        )
        .into());
    }
    println!(
        "Cached '{}' data at {} is from a different release than this build of yamc expects \
         (found {:?}, expected {:?}); re-downloading it.",
        source,
        dir.display(),
        cached_data_version(dir).unwrap_or_else(|| "no data_version".to_string()),
        expected_data_version(source).unwrap_or("none"),
    );
    fs::remove_dir_all(dir)?;
    Ok(())
}

/// Error for data that is still not the expected release after a fresh
/// download, which means the published objects were never stamped with the
/// version this build pins.
///
/// Loud on purpose. The alternative is evicting and refetching the same bytes
/// on every single load, which looks like a network problem rather than a
/// publishing mistake.
#[cfg(feature = "download")]
fn stale_after_download_error(source: &str, dir: &std::path::Path) -> Box<dyn std::error::Error> {
    format!(
        "'{}' was downloaded fresh and still reports data_version {:?}, but this build of yamc \
         expects {:?}. The published data has not been stamped with the version this yamc \
         release pins, so either the publish or EXPECTED_DATA_VERSION in url_cache.rs is wrong. \
         Cached at {}.",
        source,
        cached_data_version(dir).unwrap_or_else(|| "no data_version".to_string()),
        expected_data_version(source).unwrap_or("none"),
        dir.display(),
    )
    .into()
}

#[cfg(feature = "download")]
fn parse_embedded_index(text: &'static str) -> HashSet<&'static str> {
    text.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

/// `Some(true)` if the nuclide is known to be published in the keyword's
/// release, `Some(false)` if it is known to be unpublished, `None` if
/// no embedded index is shipped for the keyword (caller should fall back
/// to attempting the download).
#[cfg(feature = "download")]
fn nuclide_is_in_embedded_index(keyword: &str, nuclide_name: &str) -> Option<bool> {
    EMBEDDED_INDEX
        .get(keyword)
        .map(|set| set.contains(nuclide_name))
}

/// Error returned when the embedded index says the requested nuclide
/// isn't published -- used to short-circuit the otherwise-guaranteed 404.
#[cfg(feature = "download")]
fn not_in_embedded_index_error(source: &str, nuclide_name: &str) -> Box<dyn std::error::Error> {
    let mut msg = format!(
        "Nuclide '{}' is not available in '{}'.",
        nuclide_name, source
    );
    if let Some(set) = EMBEDDED_INDEX.get(source) {
        let mut available: Vec<&str> = set.iter().copied().collect();
        available.sort_unstable();
        msg.push_str(&format!(
            "\n\nAvailable nuclides/elements:\n{}",
            available.join(", ")
        ));
    }
    msg.into()
}

/// Generate a cache path for keyword-based downloads (always Arrow tar)
fn generate_cache_name(source: &str, nuclide_name: &str) -> String {
    if is_keyword(source) {
        format!("{}-{}.arrow", source, nuclide_name)
    } else {
        format!("{}.arrow", nuclide_name)
    }
}

/// Download a file from URL to cache directory, return the local path.
/// Downloads the tar, extracts the arrow directory in-memory,
/// and writes the contents to cache.
#[cfg(feature = "download")]
pub fn download_and_cache(
    url: &str,
    source: &str,
    nuclide_name: &str,
    kind: DataKind,
    scope: &crate::load_scope::LoadScope,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = get_cache_dir()?;
    let cache_name = generate_cache_name(source, nuclide_name);
    let local_path = cache_dir.join(&cache_name);
    let sections = sections_for(kind, scope);
    // Which MTs, if this load reads only some of them. Decides whether
    // reactions.arrow is fetched whole or as the byte ranges those MTs occupy.
    let subset = subset_mts(kind, scope);

    // Fast path: cache hit without any locking. The gate is per-section rather
    // than per-directory, because a cache dir populated by an earlier
    // transmutation load holds only some of what transport now wants (issue
    // #389). Each section is published by rename, so a section that is present
    // is complete. `cache_is_current` additionally requires the release stamp
    // to be the one this build expects, so a re-published library invalidates
    // rather than hitting forever (issue #366); it is a local `version.json`
    // read, so the offline zero-round-trip property is unchanged.
    if have_all_sections(&local_path, sections, subset) && cache_is_current(&local_path, source) {
        return Ok(local_path);
    }

    // Serialize concurrent downloads of the same nuclide. Re-check after
    // acquiring the lock in case another thread finished the download while
    // we were waiting.
    let path_lock = get_path_lock(&local_path);
    let _guard = path_lock.lock().unwrap_or_else(|p| p.into_inner());
    if have_all_sections(&local_path, sections, subset) && cache_is_current(&local_path, source) {
        return Ok(local_path);
    }

    // A stale directory has to go before the top-up, not after: the top-up only
    // fetches sections that are not already resolved, so a complete stale copy
    // would fetch nothing at all.
    evict_if_stale(&local_path, source)?;

    // Cache miss. If we ship an index of available nuclides for this
    // library and the requested name isn't in it, fail fast -- saves a
    // guaranteed 404 round trip (issue #46). Falls through when no
    // embedded index is shipped for `source` (e.g. raw URLs, or
    // libraries we haven't shipped a list for).
    if matches!(
        nuclide_is_in_embedded_index(source, nuclide_name),
        Some(false)
    ) {
        return Err(not_in_embedded_index_error(source, nuclide_name));
    }

    download_sections(url, &local_path, sections, source, nuclide_name, subset)?;

    if !cache_is_current(&local_path, source) {
        return Err(stale_after_download_error(source, &local_path));
    }

    Ok(local_path)
}

/// Download an option-D per-section object set to the cache under an explicit
/// cache directory name and return the local path. Used for assets that don't
/// follow the `<source>-<nuclide>.arrow` cache layout -- currently the
/// transmutation subsections (one dir per library subsection). `sections` is
/// the file list for the subsection (see [`transmutation_sections`]).
#[cfg(feature = "download")]
pub fn download_and_cache_named(
    url: &str,
    cache_name: &str,
    source: &str,
    sections: &[(&str, bool)],
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = get_cache_dir()?;
    let local_path = cache_dir.join(cache_name);

    if local_path.exists() && cache_is_current(&local_path, source) {
        return Ok(local_path);
    }

    let path_lock = get_path_lock(&local_path);
    let _guard = path_lock.lock().unwrap_or_else(|p| p.into_inner());
    if local_path.exists() && cache_is_current(&local_path, source) {
        return Ok(local_path);
    }

    evict_if_stale(&local_path, source)?;

    // `None`, and load-bearing rather than incidental: the chain's `reactions`
    // subsection publishes a file that is also called `reactions.arrow`, and it
    // is a different table with no per-MT batches and no byte-range index. A
    // subset here would try to range into it.
    download_sections(url, &local_path, sections, source, cache_name, None)?;

    if !cache_is_current(&local_path, source) {
        return Err(stale_after_download_error(source, &local_path));
    }

    Ok(local_path)
}

/// Build a helpful error message when a download fails, including available nuclides if an index exists
#[cfg(feature = "download")]
fn download_error(
    url: &str,
    status: reqwest::StatusCode,
    source: &str,
    nuclide_name: &str,
) -> Box<dyn std::error::Error> {
    let mut msg = format!("Failed to download {}: {}", url, status);

    // If we ship an embedded index for this library, list what's available to
    // help the caller (libraries without an embedded index just 404).
    if let Some(set) = EMBEDDED_INDEX.get(source) {
        let mut available: Vec<&str> = set.iter().copied().collect();
        available.sort_unstable();
        msg.push_str(&format!(
            "\n\nNuclide '{}' is not available in '{}'. Available nuclides/elements:\n{}",
            nuclide_name,
            source,
            available.join(", ")
        ));
    }

    msg.into()
}

/// Per-nuclide neutron section objects (option-D). `(filename, required)`:
/// optional sections are absent for many nuclides (urr only in resonance
/// nuclides, total_nu and fission_photon only in fissionables) and a 404 there
/// is expected.
#[cfg(feature = "download")]
const NEUTRON_SECTIONS: &[(&str, bool)] = &[
    ("version.json", true),
    ("nuclide.arrow", true),
    ("reactions.arrow", true),
    ("products.arrow", true),
    ("distributions.arrow", true),
    ("fast_xs.arrow", true),
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
    ("reactions.arrow", true),
    ("products.arrow", true),
    ("distributions.arrow", true),
    ("fast_xs.arrow", true),
    ("urr.arrow", false),
    ("total_nu.arrow", false),
    ("fission_photon.arrow", false),
    // Optional: most evaluations carry no MF=33, so a 404 here is expected and
    // is recorded as settled rather than retried on every later load.
    ("covariance.arrow", false),
];

/// Per-element photon section objects (option-D).
#[cfg(feature = "download")]
const PHOTON_SECTIONS: &[(&str, bool)] = &[
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
    ("reactions.arrow", true),
    ("covariance.arrow", false),
];

/// Which section objects a scope needs.
///
/// Photon data has no transmutation path, so it is always fetched whole.
#[cfg(feature = "download")]
fn sections_for(
    kind: DataKind,
    scope: &crate::load_scope::LoadScope,
) -> &'static [(&'static str, bool)] {
    match kind {
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
    }
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
/// `build_fast_xs_from_arrow` drops any MT missing from the reactions map with
/// a bare `if let Some(..)`, so a nuclide can come back with no elastic
/// scattering, no `fissionable` flag and no MT 101 absorption, and nothing
/// anywhere says so.
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
#[cfg(feature = "download")]
fn fetch(url: &str, span: Option<(u64, u64)>) -> Result<Fetched, Box<dyn std::error::Error>> {
    const RETRY_DELAYS_MS: &[u64] = &[200, 500, 1000];
    let mut last_status: Option<reqwest::StatusCode> = None;
    for &delay_ms in std::iter::once(&0u64).chain(RETRY_DELAYS_MS.iter()) {
        if delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }
        let r = blocking_get(url, span)?;
        let status = r.status();
        if status.is_success() {
            return Ok(Fetched::Body {
                partial: status == reqwest::StatusCode::PARTIAL_CONTENT,
                bytes: r.bytes()?.to_vec(),
            });
        }
        if status == reqwest::StatusCode::NOT_FOUND {
            // Definitively absent: R2 answers authoritatively, no retry.
            return Ok(Fetched::Absent);
        }
        last_status = Some(status);
    }
    Err(format!(
        "Failed to download {}: {}",
        url,
        last_status.unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR)
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
///   case (issue #389) and it is why the whole directory is no longer replaced:
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
/// `YAMC_CACHE_DIR` wins when it is set and non-empty, and is taken verbatim:
/// it names the cache root itself, not a parent to append `.cache/yamc` to.
/// That is the supported way to point a process at a cache somewhere else, and
/// it is what `scripts/fetch_test_fixtures.py` already reads on the Python
/// side.
///
/// Without it the root is `<home>/.cache/yamc`, where home is [`home_dir`].
/// Resolving it there rather than through `HOME` alone is the whole point:
/// `HOME` is not a Windows variable, so a `HOME`-only lookup silently resolves
/// to nothing there (issue #544).
///
/// `None` means neither the override nor a home directory resolved, which is a
/// machine with no cache location rather than a machine with an empty cache.
/// Callers that need a path say what that means for them: [`get_cache_dir`]
/// reports it as an error, and the tests treat it as a failure rather than as
/// absent data.
pub fn cache_root() -> Option<PathBuf> {
    cache_root_from(real_env, home_dir())
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
/// Issue #544 was never about this function. It was about the TESTS reading
/// `HOME` themselves, which is not a Windows variable, so they resolved to
/// nothing there while this resolved correctly. The fix is that everything
/// comes through here.
pub fn home_dir() -> Option<PathBuf> {
    etcetera::home_dir().ok()
}

/// The real process environment, with an empty variable treated as unset.
///
/// A shell that exports `YAMC_CACHE_DIR=` would otherwise name the empty path.
fn real_env(key: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(key).filter(|v| !v.is_empty())
}

/// [`cache_root`] over a supplied environment and home directory.
///
/// Split out so the override rule can be tested without moving the process's
/// own environment, which would race every other test in the binary.
fn cache_root_from(
    get: impl Fn(&str) -> Option<std::ffi::OsString>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(dir) = get("YAMC_CACHE_DIR") {
        return Some(PathBuf::from(dir));
    }
    Some(home?.join(".cache").join("yamc"))
}

/// Where a cached entry for `nuclide` from `source` sits on disk, e.g.
/// `<root>/endf-b8.1-Fe58.arrow`.
///
/// The naming rule is the one the downloader writes with, so a caller can find
/// an entry the cache already holds without re-deriving the layout. `None` when
/// [`cache_root`] resolves to nothing.
pub fn cached_entry_path(source: &str, nuclide: &str) -> Option<PathBuf> {
    Some(cache_root()?.join(generate_cache_name(source, nuclide)))
}

#[cfg(test)]
mod cache_root_tests {
    use super::{cache_root, cache_root_from, generate_cache_name, home_dir};
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::path::PathBuf;

    /// A fixed environment, so these say nothing about the machine they run on.
    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key| map.get(key).filter(|v| !v.is_empty()).map(OsString::from)
    }

    fn home(path: &str) -> Option<PathBuf> {
        Some(PathBuf::from(path))
    }

    #[test]
    fn the_root_hangs_off_the_home_directory() {
        assert_eq!(
            cache_root_from(env(&[]), home("/home/someone")),
            Some(PathBuf::from("/home/someone/.cache/yamc"))
        );
    }

    /// The Windows spelling, which is the shape issue #544 was about. Only the
    /// join is ours: which variable produced the home is `etcetera`'s business,
    /// and taking it back off it is what broke a process with no exported HOME.
    #[test]
    fn a_windows_home_joins_the_same_way() {
        assert_eq!(
            cache_root_from(env(&[]), home("C:/Users/runneradmin")),
            Some(PathBuf::from("C:/Users/runneradmin/.cache/yamc"))
        );
    }

    /// The override names the cache root itself, with no `.cache/yamc`
    /// appended: it is a cache directory, not a home directory to derive one
    /// from. And it wins over a home that resolves perfectly well, which is
    /// what makes it usable for an isolated test cache.
    #[test]
    fn the_override_wins_verbatim() {
        assert_eq!(
            cache_root_from(
                env(&[("YAMC_CACHE_DIR", "/tmp/isolated")]),
                home("/home/someone")
            ),
            Some(PathBuf::from("/tmp/isolated"))
        );
    }

    /// An empty variable is an unset one. A shell that exports
    /// `YAMC_CACHE_DIR=` would otherwise name the empty path, and the cache
    /// would land relative to the working directory.
    #[test]
    fn an_empty_override_is_no_override() {
        assert_eq!(
            cache_root_from(env(&[("YAMC_CACHE_DIR", "")]), home("/home/someone")),
            Some(PathBuf::from("/home/someone/.cache/yamc"))
        );
    }

    /// No home and no override is a machine with nowhere to put a cache, which
    /// a caller must be able to tell from a machine whose cache is empty.
    #[test]
    fn no_home_and_no_override_resolves_nothing() {
        assert_eq!(cache_root_from(env(&[]), None), None);
    }

    /// And the override still answers without a home, which is the case it
    /// exists for: a container or a service account.
    #[test]
    fn the_override_answers_without_a_home() {
        assert_eq!(
            cache_root_from(env(&[("YAMC_CACHE_DIR", "/srv/cache")]), None),
            Some(PathBuf::from("/srv/cache"))
        );
    }

    #[test]
    fn an_entry_sits_directly_under_the_root() {
        let root = cache_root_from(env(&[("YAMC_CACHE_DIR", "/tmp/isolated")]), None).unwrap();
        assert_eq!(
            root.join(generate_cache_name("endf-b8.1", "Fe58")),
            PathBuf::from("/tmp/isolated/endf-b8.1-Fe58.arrow")
        );
    }

    /// `etcetera` resolves a home on every platform CI runs on, including the
    /// Windows runner where `HOME` is unset and `USERPROFILE` carries it. A
    /// `HOME`-only read is what returned nothing there (issue #544).
    #[test]
    fn this_machine_has_a_home_and_therefore_a_cache_root() {
        assert!(home_dir().is_some(), "no home directory resolved");
        assert!(cache_root().is_some());
    }
}

/// Get the cache directory for yamc, creating it if it does not exist.
#[cfg(feature = "download")]
pub fn get_cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = cache_root().ok_or(
        "Could not find a cache directory: neither YAMC_CACHE_DIR nor a home \
         directory is set",
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
/// If it's a keyword, expand it to URL (picking the `kind`-specific
/// subdirectory if the library splits neutron / photon data) and download.
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
        // It's a keyword, expand to URL and download
        let url = expand_keyword_to_url(path_url_or_keyword, nuclide_name, kind)
            .ok_or_else(|| format!("Unknown keyword: {}", path_url_or_keyword))?;
        download_and_cache(&url, path_url_or_keyword, nuclide_name, kind, scope)
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

    #[test]
    fn endf_b81_nuclide_url_uses_neutron_subdir() {
        let url = expand_keyword_to_url("endf-b8.1", "Fe56", DataKind::Neutron)
            .expect("endf-b8.1 should expand for Fe56");
        assert_eq!(url, concat!(data_origin!(), "endf-b8.1/neutron/Fe56.arrow"));
    }

    #[test]
    fn endf_b81_element_url_uses_photon_subdir() {
        let url = expand_keyword_to_url("endf-b8.1", "Fe", DataKind::Photon)
            .expect("endf-b8.1 should expand for photon Fe");
        assert_eq!(url, concat!(data_origin!(), "endf-b8.1/photon/Fe.arrow"));
    }

    #[test]
    fn fendl_3_2d_uses_r2_neutron_and_photon_subdirs() {
        // fendl-3.2d is hosted on Cloudflare R2 with neutron/ + photon/ subdirs,
        // like endf-b8.1.
        let n = expand_keyword_to_url("fendl-3.2d", "Fe56", DataKind::Neutron)
            .expect("fendl-3.2d should expand for neutron Fe56");
        assert_eq!(n, concat!(data_origin!(), "fendl-3.2d/neutron/Fe56.arrow"));
        let p = expand_keyword_to_url("fendl-3.2d", "Fe", DataKind::Photon)
            .expect("fendl-3.2d should expand for photon Fe");
        assert_eq!(p, concat!(data_origin!(), "fendl-3.2d/photon/Fe.arrow"));
    }

    #[test]
    fn tendl_2017_is_a_keyword_and_resolves_neutron_on_r2() {
        assert!(is_keyword("tendl-2017"), "tendl-2017 must be recognized");
        let n = expand_keyword_to_url("tendl-2017", "Fe56", DataKind::Neutron)
            .expect("tendl-2017 should expand for neutron Fe56");
        assert_eq!(n, concat!(data_origin!(), "tendl-2017/neutron/Fe56.arrow"));
        // TENDL is neutron-only: empty photon subdir (no photon/ path segment).
        let p = expand_keyword_to_url("tendl-2017", "Fe", DataKind::Photon)
            .expect("tendl-2017 expands with an empty photon subdir");
        assert_eq!(p, concat!(data_origin!(), "tendl-2017/Fe.arrow"));
        // The branching transmutation subsection is published for tendl-2017.
        let b = expand_keyword_to_subsection_url("tendl-2017", "branching")
            .expect("tendl-2017 branching subsection URL");
        assert_eq!(
            b,
            concat!(data_origin!(), "tendl-2017/transmutation/branching.arrow")
        );
    }

    #[test]
    fn jeff_4_0_is_a_keyword_and_resolves_neutron_on_r2() {
        assert!(is_keyword("jeff-4.0"), "jeff-4.0 must be recognized");
        let n = expand_keyword_to_url("jeff-4.0", "Fe56", DataKind::Neutron)
            .expect("jeff-4.0 should expand for neutron Fe56");
        assert_eq!(n, concat!(data_origin!(), "jeff-4.0/neutron/Fe56.arrow"));
        // JEFF-4.0 ships no photoatomic data: empty photon subdir.
        let p = expand_keyword_to_url("jeff-4.0", "Fe", DataKind::Photon)
            .expect("jeff-4.0 expands with an empty photon subdir");
        assert_eq!(p, concat!(data_origin!(), "jeff-4.0/Fe.arrow"));
        // Unlike TENDL, the chain is complete: JEFF has its own decay and
        // fission-yield sublibraries.
        assert_eq!(
            keyword_transmutation_subsections("jeff-4.0"),
            Some(&["decay", "reactions", "fission_yields", "branching"][..])
        );
    }

    /// `KEYWORDS` and `get_keyword_info_mapping` are two lists of the same
    /// thing, and the comment on `KEYWORDS` asks for them to be kept in sync by
    /// hand. A keyword in one and not the other is accepted by `is_keyword` and
    /// then fails to expand, so check it rather than trust it.
    #[test]
    fn every_keyword_has_download_info() {
        for keyword in KEYWORDS {
            assert!(
                expand_keyword_to_url(keyword, "Fe56", DataKind::Neutron).is_some(),
                "{keyword:?} is in KEYWORDS but has no entry in \
                 get_keyword_info_mapping"
            );
            assert!(
                keyword_transmutation_subsections(keyword).is_some(),
                "{keyword:?} is in KEYWORDS but keyword_transmutation_subsections \
                 does not know it, so a chain request reports it as unknown"
            );
        }
        assert_eq!(
            KEYWORDS.len(),
            get_keyword_info_mapping().len(),
            "get_keyword_info_mapping holds an entry KEYWORDS does not list"
        );
    }

    #[test]
    fn jendl_5_0_is_a_keyword_and_resolves_both_particles_on_r2() {
        assert!(is_keyword("jendl-5.0"), "jendl-5.0 must be recognized");
        let n = expand_keyword_to_url("jendl-5.0", "Fe56", DataKind::Neutron)
            .expect("jendl-5.0 should expand for neutron Fe56");
        assert_eq!(n, concat!(data_origin!(), "jendl-5.0/neutron/Fe56.arrow"));
        // Unlike jeff-4.0, JENDL publishes the photoatomic + atomic relaxation
        // pair the photon loader reads, so the photon subdir is not empty.
        let p = expand_keyword_to_url("jendl-5.0", "Fe", DataKind::Photon)
            .expect("jendl-5.0 should expand for photon Fe");
        assert_eq!(p, concat!(data_origin!(), "jendl-5.0/photon/Fe.arrow"));
        // The chain is built from JENDL's own decay and fission-yield
        // sublibraries, so all four subsections are library-consistent.
        assert_eq!(
            keyword_transmutation_subsections("jendl-5.0"),
            Some(&["decay", "reactions", "fission_yields", "branching"][..])
        );
    }

    #[test]
    fn tendl_2025_uses_r2_neutron_subdir() {
        // tendl-2025 is hosted on Cloudflare R2 with a neutron/ subdir,
        // mirroring endf-b8.1.
        let url = expand_keyword_to_url("tendl-2025", "Fe56", DataKind::Neutron)
            .expect("tendl-2025 should expand for Fe56");
        assert_eq!(
            url,
            concat!(data_origin!(), "tendl-2025/neutron/Fe56.arrow")
        );
    }

    /// End-to-end network test: download + extract a real tendl-2025 nuclide
    /// from Cloudflare R2. Ignored by default (needs network and the data to be
    /// published). Run with:
    ///   cargo test -p yamc-nuclide --features download -- --ignored tendl_2025_fetch_li6
    #[test]
    #[ignore]
    fn tendl_2025_fetch_li6_from_r2() {
        let url = expand_keyword_to_url("tendl-2025", "Li6", DataKind::Neutron).unwrap();
        let path = download_and_cache(
            &url,
            "tendl-2025",
            "Li6",
            DataKind::Neutron,
            &crate::LoadScope::full(),
        )
        .expect("download Li6");
        assert!(
            path.join("nuclide.arrow").exists(),
            "expected nuclide.arrow under {path:?}"
        );
    }

    /// End-to-end network test: download a real fendl-3.2d neutron + photon file
    /// from Cloudflare R2. Ignored by default (needs network and published data).
    /// Run with:
    ///   cargo test -p yamc-nuclide --features download -- --ignored fendl_3_2d_fetch
    #[test]
    #[ignore]
    fn fendl_3_2d_fetch_from_r2() {
        let n = expand_keyword_to_url("fendl-3.2d", "Li6", DataKind::Neutron).unwrap();
        let np = download_and_cache(
            &n,
            "fendl-3.2d",
            "Li6",
            DataKind::Neutron,
            &crate::LoadScope::full(),
        )
        .expect("download neutron Li6");
        assert!(
            np.join("nuclide.arrow").exists(),
            "missing nuclide.arrow under {np:?}"
        );
        let p = expand_keyword_to_url("fendl-3.2d", "Fe", DataKind::Photon).unwrap();
        let pp = download_and_cache(
            &p,
            "fendl-3.2d",
            "Fe",
            DataKind::Photon,
            &crate::LoadScope::full(),
        )
        .expect("download photon Fe");
        assert!(
            pp.join("element.arrow").exists(),
            "missing element.arrow under {pp:?}"
        );
    }

    #[test]
    fn subsection_url_expands_under_transmutation_prefix() {
        assert_eq!(
            expand_keyword_to_subsection_url("endf-b8.1", "decay"),
            Some(concat!(data_origin!(), "endf-b8.1/transmutation/decay.arrow").to_string())
        );
        assert_eq!(
            expand_keyword_to_subsection_url("tendl-2025", "branching"),
            Some(concat!(data_origin!(), "tendl-2025/transmutation/branching.arrow").to_string())
        );
        assert!(expand_keyword_to_subsection_url("not-a-library", "decay").is_none());
    }

    #[test]
    fn keyword_transmutation_subsections_map() {
        assert_eq!(
            keyword_transmutation_subsections("endf-b8.1"),
            Some(&["decay", "reactions", "fission_yields", "branching"][..])
        );
        assert_eq!(
            keyword_transmutation_subsections("jendl-5.0"),
            Some(&["decay", "reactions", "fission_yields", "branching"][..])
        );
        assert_eq!(
            keyword_transmutation_subsections("tendl-2025"),
            Some(&["branching", "reactions"][..])
        );
        assert_eq!(
            keyword_transmutation_subsections("tendl-2017"),
            Some(&["branching", "reactions"][..])
        );
        assert_eq!(
            keyword_transmutation_subsections("fendl-3.2d"),
            Some(&[][..])
        );
        assert!(keyword_transmutation_subsections("not-a-library").is_none());
    }

    #[test]
    fn resolve_subsection_fails_fast_on_unavailable_subsection() {
        // TENDL is neutron-only: no decay subsection. Must fail before any
        // download with a message that lists what the library does provide.
        let err = resolve_subsection("tendl-2017", "decay")
            .expect_err("tendl-2017 has no decay subsection")
            .to_string();
        assert!(err.contains("tendl-2017"), "{err}");
        assert!(err.contains("does not provide"), "{err}");
        assert!(
            err.contains("branching") && err.contains("reactions"),
            "error should list available subsections: {err}"
        );
        // fendl-3.2d publishes no transmutation subsections at all.
        let err2 = resolve_subsection("fendl-3.2d", "reactions")
            .expect_err("fendl-3.2d has no transmutation subsections")
            .to_string();
        assert!(err2.contains("no transmutation subsections"), "{err2}");
    }

    #[test]
    fn embedded_endf_b81_index_contains_published_nuclides() {
        // Spot-check entries that should be in the published list.
        assert_eq!(
            nuclide_is_in_embedded_index("endf-b8.1", "Fe56"),
            Some(true)
        );
        assert_eq!(nuclide_is_in_embedded_index("endf-b8.1", "Li6"), Some(true));
        assert_eq!(nuclide_is_in_embedded_index("endf-b8.1", "Be9"), Some(true));
        // Element symbols are also published.
        assert_eq!(nuclide_is_in_embedded_index("endf-b8.1", "Fe"), Some(true));
    }

    #[test]
    fn embedded_endf_b81_index_excludes_unpublished_nuclides() {
        // Examples called out in issue #46 -- these should fail fast
        // instead of triggering a 404.
        for unpublished in [
            "Ag104", "Ag105", "Ag106", "Ag107_m1", "Ag108_m1", "Ag109_m1", "Ag110", "Cd104",
            "Cd105", "Mo101", "Mo102", "Nb91", "Nb92",
        ] {
            assert_eq!(
                nuclide_is_in_embedded_index("endf-b8.1", unpublished),
                Some(false),
                "{unpublished} should be marked unpublished in embedded index"
            );
        }
    }

    #[test]
    fn embedded_index_returns_none_for_libraries_without_one() {
        // tendl-2025 ships no embedded index yet, so it falls back to the
        // network path; unknown keywords likewise return None.
        assert_eq!(nuclide_is_in_embedded_index("tendl-2025", "Fe56"), None);
        assert_eq!(nuclide_is_in_embedded_index("not-a-library", "Fe56"), None);
    }

    #[test]
    fn embedded_fendl_3_2d_index_matches_published_set() {
        // Published neutron nuclides + photon elements.
        assert_eq!(
            nuclide_is_in_embedded_index("fendl-3.2d", "Fe56"),
            Some(true)
        );
        assert_eq!(
            nuclide_is_in_embedded_index("fendl-3.2d", "Li6"),
            Some(true)
        );
        assert_eq!(nuclide_is_in_embedded_index("fendl-3.2d", "Fe"), Some(true));
        // Cf is not in the FENDL 3.2d fusion library -> fail fast.
        assert_eq!(
            nuclide_is_in_embedded_index("fendl-3.2d", "Cf252"),
            Some(false)
        );
    }

    #[test]
    fn unpublished_nuclide_errors_without_network() {
        // resolve_path_or_url must fail fast for known-unpublished
        // names without ever hitting the network.
        let err = resolve_path_or_url(
            "endf-b8.1",
            "Ag104",
            DataKind::Neutron,
            &crate::LoadScope::full(),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Ag104"), "msg lacks nuclide: {msg}");
        assert!(msg.contains("endf-b8.1"), "msg lacks library: {msg}");
        assert!(
            msg.contains("not available"),
            "msg lacks reason text: {msg}"
        );
    }

    // ---- issue #389: scoped, additive section fetching ----

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

    /// Write a `version.json` carrying a byte-range index for `mts`, laid out
    /// back to back the way the converter writes the batches.
    fn write_index(dir: &std::path::Path, mts: &[i32]) {
        let mut ranges = std::collections::BTreeMap::new();
        let mut at = 768u64;
        for mt in mts {
            ranges.insert(*mt, (at, 100u64));
            at += 100;
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
            "products.arrow",
            "distributions.arrow",
            "fast_xs.arrow",
        ] {
            dir.touch(name);
        }
        let full = sections_for(DataKind::Neutron, &crate::LoadScope::full());
        assert!(
            !have_all_sections(&dir.0, full, None),
            "transport must not be satisfied by a subset"
        );
        let todo: Vec<&str> = sections_to_fetch(&dir.0, full, None)
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
        let wanted = mts(&[16, 102]);
        let xs_only = sections_for(
            DataKind::Neutron,
            &crate::LoadScope::activation(wanted.clone()),
        );
        assert!(have_all_sections(&dir.0, xs_only, Some(&wanted)));
        assert!(sections_to_fetch(&dir.0, xs_only, Some(&wanted)).is_empty());
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
        };
        assert!(subset_mts(DataKind::Neutron, &all_mts).is_none());
    }

    #[test]
    fn an_activation_scope_asks_for_three_neutron_sections() {
        let xs = sections_for(
            DataKind::Neutron,
            &crate::LoadScope::activation([102].into()),
        );
        let names: Vec<&str> = xs.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, ["version.json", "nuclide.arrow", "reactions.arrow"]);
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

    /// Issue #369. Every other layer shipped the fission photon release (the
    /// converter writes it, the schema declares it, the Arrow reader parses
    /// it), but a download-path user only ever sees a section named in this
    /// list. Left out, the read silently returns `None` and actinide fission
    /// photon production stays ~38% low -- the exact deficit #369 reported.
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
            sections_to_fetch(&empty, xs_only, None).len(),
            xs_only.len()
        );
        assert!(!have_all_sections(&empty, xs_only, None));

        // Simulate the transmutation load having run.
        for (name, _) in xs_only {
            dir.touch(name);
        }
        assert!(
            have_all_sections(&dir.0, xs_only, None),
            "the activation scope is satisfied"
        );
        assert!(
            !have_all_sections(&dir.0, full, None),
            "but transport still needs the rest"
        );

        // The follow-up transport fetch asks only for what is missing, which is
        // the whole point: the earlier three are not refetched.
        let todo: Vec<&str> = sections_to_fetch(&dir.0, full, None)
            .iter()
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(
            todo,
            [
                "products.arrow",
                "distributions.arrow",
                "fast_xs.arrow",
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
        for (name, required) in full {
            // Fe56 has no URR, nu or fission-photon tables upstream: those 404
            // and get a marker.
            if *required {
                dir.touch(name);
            }
        }
        assert!(
            !have_all_sections(&dir.0, full, None),
            "an unmarked optional section still looks unfetched"
        );

        dir.touch(&format!("urr.arrow{ABSENT_SUFFIX}"));
        dir.touch(&format!("total_nu.arrow{ABSENT_SUFFIX}"));
        dir.touch(&format!("fission_photon.arrow{ABSENT_SUFFIX}"));
        assert!(
            have_all_sections(&dir.0, full, None),
            "a marker means the origin answered 404, so stop asking"
        );
        assert!(sections_to_fetch(&dir.0, full, None).is_empty());
    }

    /// End-to-end network test for the additive path: fetch one nuclide at an
    /// activation scope, then at full scope, and check the second call topped
    /// the same directory up instead of replacing it. Ignored by default.
    ///   cargo test -p yamc-nuclide --features download -- --ignored additive
    #[test]
    #[ignore]
    fn additive_fetch_tops_up_the_same_cache_dir() {
        let url = expand_keyword_to_url("tendl-2025", "Li6", DataKind::Neutron).unwrap();
        let narrow = download_and_cache(
            &url,
            "tendl-2025",
            "Li6",
            DataKind::Neutron,
            &crate::LoadScope::activation([102].into()),
        )
        .expect("activation fetch");
        // The ranged path splices the MTs the chain names into `subset/`
        // (alongside a `subset/mts.json` naming them) rather than writing a
        // whole `reactions.arrow`, so that is where a narrow fetch lands. The
        // top-level file appears only once something asks for the whole thing.
        let subset_reactions = narrow.join("subset").join("reactions.arrow");
        assert!(
            subset_reactions.exists(),
            "an activation fetch should splice its MTs into subset/"
        );
        assert!(
            !narrow.join("fast_xs.arrow").exists(),
            "an activation fetch must not pull the transport accelerator"
        );
        // `nuclide.arrow` is the additive-ness probe rather than the reactions
        // table: a full fetch supersedes the spliced `subset/` with the whole
        // `reactions.arrow` and removes it, so the reactions table is expected
        // to move. `nuclide.arrow` is wanted by both scopes and must survive
        // the second call untouched.
        let stamp = fs::metadata(narrow.join("nuclide.arrow"))
            .and_then(|m| m.modified())
            .expect("mtime");

        let full = download_and_cache(
            &url,
            "tendl-2025",
            "Li6",
            DataKind::Neutron,
            &crate::LoadScope::full(),
        )
        .expect("full fetch");
        assert_eq!(narrow, full, "the same cache dir should be reused");
        assert!(full.join("fast_xs.arrow").exists(), "topped up");
        assert!(
            full.join("reactions.arrow").exists(),
            "a full fetch wants every MT, so it takes the whole reactions table"
        );
        assert!(
            !subset_reactions.exists(),
            "the spliced subset is superseded by the whole table, not left beside it"
        );
        assert_eq!(
            fs::metadata(full.join("nuclide.arrow"))
                .and_then(|m| m.modified())
                .expect("mtime"),
            stamp,
            "already-present sections must not be refetched"
        );
    }

    // ----- data_version cache invalidation (issue #366) --------------------

    /// Write a `version.json` holding `data_version`, or one without the field
    /// when `version` is `None` (which is what every directory published before
    /// #366 looks like).
    fn write_marker(dir: &std::path::Path, version: Option<&str>) {
        fs::create_dir_all(dir).expect("create marker dir");
        let body = match version {
            Some(v) => {
                format!(r#"{{"format_version": 1, "library": "test", "data_version": "{v}"}}"#)
            }
            None => r#"{"format_version": 1, "library": "test"}"#.to_string(),
        };
        fs::write(dir.join("version.json"), body).expect("write marker");
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("yamc-data-version-{name}"));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_keyword_with_no_pinned_version_is_never_stale() {
        let dir = scratch("unpinned");
        write_marker(&dir, Some("anything-at-all"));
        assert!(
            data_version_matches(&dir, None),
            "a library this build pins no version for must keep its pre-#366 behaviour"
        );
        // Including one with no marker at all.
        let bare = scratch("unpinned-bare");
        fs::create_dir_all(&bare).expect("create dir");
        assert!(data_version_matches(&bare, None));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&bare);
    }

    #[test]
    fn a_matching_stamp_is_current() {
        let dir = scratch("match");
        write_marker(&dir, Some("2026-08-09.1"));
        assert_eq!(cached_data_version(&dir).as_deref(), Some("2026-08-09.1"));
        assert!(data_version_matches(&dir, Some("2026-08-09.1")));
        let _ = fs::remove_dir_all(&dir);
    }

    /// The case the issue was filed for: the library is re-published, the URL
    /// and the cache key are unchanged, and only the stamp differs.
    #[test]
    fn a_different_stamp_is_stale() {
        let dir = scratch("mismatch");
        write_marker(&dir, Some("2026-06-13.1"));
        assert!(
            !data_version_matches(&dir, Some("2026-08-09.1")),
            "a re-published library must not hit the cache"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Data published before stamping existed has no `data_version` at all, and
    /// must be treated as stale once a version is pinned. This is what makes the
    /// first pinned release actually reach existing installs.
    #[test]
    fn an_unstamped_cache_is_stale_once_a_version_is_pinned() {
        let dir = scratch("unstamped");
        write_marker(&dir, None);
        assert_eq!(cached_data_version(&dir), None);
        assert!(!data_version_matches(&dir, Some("2026-08-09.1")));
        let _ = fs::remove_dir_all(&dir);
    }

    /// A marker that is missing, empty or not JSON must read as "no stamp"
    /// rather than panicking or erroring: it is a cache directory, and the
    /// answer to "is this the right release" is simply no.
    #[test]
    fn an_unreadable_marker_reads_as_no_stamp() {
        let missing = scratch("no-marker");
        fs::create_dir_all(&missing).expect("create dir");
        assert_eq!(cached_data_version(&missing), None);

        let broken = scratch("broken-marker");
        fs::create_dir_all(&broken).expect("create dir");
        fs::write(broken.join("version.json"), "not json at all").expect("write");
        assert_eq!(cached_data_version(&broken), None);

        let wrong_type = scratch("wrong-type-marker");
        fs::create_dir_all(&wrong_type).expect("create dir");
        fs::write(wrong_type.join("version.json"), r#"{"data_version": 3}"#).expect("write");
        assert_eq!(cached_data_version(&wrong_type), None);

        for d in [missing, broken, wrong_type] {
            let _ = fs::remove_dir_all(d);
        }
    }

    /// Eviction must refuse a path that is not a direct child of the cache
    /// directory, so a caller passing the wrong path cannot delete elsewhere.
    #[test]
    fn eviction_refuses_a_path_outside_the_cache() {
        let outside = scratch("outside-cache");
        write_marker(&outside, Some("stale"));
        // No keyword pins a version, so this is a no-op and returns Ok without
        // touching anything; the guard is exercised by the assertion that the
        // directory survives either way.
        let _ = evict_if_stale(&outside, "definitely-not-a-library");
        assert!(
            outside.join("version.json").is_file(),
            "a directory outside the cache must never be removed"
        );
        let _ = fs::remove_dir_all(&outside);
    }

    /// Every pinned entry must name a keyword that can actually be expanded to
    /// a URL. A typo here would silently pin nothing, and the check it is meant
    /// to perform would never run.
    #[test]
    fn every_pinned_keyword_is_a_real_keyword() {
        for (keyword, version) in EXPECTED_DATA_VERSION {
            assert!(
                is_keyword(keyword),
                "EXPECTED_DATA_VERSION pins {keyword:?} (version {version:?}), \
                 which is not a library keyword"
            );
            assert!(
                !version.is_empty(),
                "EXPECTED_DATA_VERSION pins an empty version for {keyword:?}"
            );
        }
    }

    /// A transmutation subsection stamps `provenance.json` rather than
    /// `version.json`, and must be invalidated by the same comparison.
    #[test]
    fn a_provenance_stamp_is_read_too() {
        let dir = scratch("provenance");
        fs::create_dir_all(&dir).expect("create dir");
        fs::write(
            dir.join("provenance.json"),
            r#"{"subsection": "decay", "data_version": "2026-08-09.1"}"#,
        )
        .expect("write provenance");
        assert_eq!(cached_data_version(&dir).as_deref(), Some("2026-08-09.1"));
        assert!(data_version_matches(&dir, Some("2026-08-09.1")));
        assert!(!data_version_matches(&dir, Some("2026-06-13.1")));
        let _ = fs::remove_dir_all(&dir);
    }
}
