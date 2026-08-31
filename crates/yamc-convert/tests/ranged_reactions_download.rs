//! Fetching only the MTs a chain names must load the same cross sections as
//! fetching the whole `reactions.arrow`.
//!
//! The converter publishes a byte range for every MT's record batch, so an
//! activation load fetches a few ranges instead of a multi-megabyte object. The
//! bytes are spliced into an Arrow IPC stream and cached under `subset/`, which
//! is a different framing and a different path from anything the loader read
//! before. Both have to come out at the same numbers, and the cache must not let
//! a later transport load mistake the subset for the whole section.
//!
//! Lives in this crate rather than beside the loader because it needs both
//! halves: `index_reactions` here to build the index, and `download_and_cache`
//! there to consume it. The dependency already runs this way (yamc-convert
//! dev-depends on yamc-nuclide to check its output loads), and adding the
//! reverse edge would make a dev-dependency cycle that breaks doc-tests with
//! E0460.
//!
//! Served from a local socket rather than mocked: the `Range` handling, the 206,
//! and the splice are the parts most likely to be subtly wrong, and a fake that
//! returns the right bytes would exercise none of them.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use yamc_nuclide::LoadScope;

/// MTs a typical activation chain asks Fe56 for.
const ACTIVATION_MTS: &[i32] = &[16, 102, 103, 107];

fn fixture() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc/tests/Fe56.arrow");
    dir.join("reactions.arrow").exists().then_some(dir)
}

/// A copy of the fixture served with, or without, the byte-range index.
///
/// BOTH cases are constructed rather than assumed. The fixture is not committed
/// -- `crates/yamc/tests/*.arrow` is gitignored and fetched by
/// `scripts/fetch_test_fixtures.py` -- so whether it carries an index depends on
/// when the cache was last filled, and the published `Fe56.arrow` has carried
/// one since the 2026-08-21 republish. Serving it unmodified therefore used to
/// mean "no index" and now means "index", silently turning the no-index test
/// into a second copy of the indexed one (issue #580).
///
/// With the index, it is built by the converter's own `index_reactions` rather
/// than by a copy of its footer walk, so what the origin serves is what a
/// published nuclide serves. Without it, `reaction_ranges` is removed from
/// `version.json`, which is exactly what a library published before the index
/// looks like.
struct Served {
    dir: PathBuf,
    _tmp: Option<PathBuf>,
}

impl Served {
    fn new(tag: &str, src: &Path, with_index: bool) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "yamc-served-{}-{}-{:?}",
            tag,
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("served dir");
        for entry in std::fs::read_dir(src).expect("read fixture") {
            let entry = entry.expect("entry");
            if entry.path().is_file() {
                std::fs::copy(entry.path(), dir.join(entry.file_name())).expect("copy");
            }
        }
        if with_index {
            yamc_convert::reaction_ranges::write_reaction_ranges(&dir).expect("write index");
        } else {
            strip_reaction_ranges(&dir);
        }
        Self {
            dir: dir.clone(),
            _tmp: Some(dir),
        }
    }
}

/// Remove `reaction_ranges` from a copied fixture's `version.json`.
///
/// The inverse of `write_reaction_ranges`, and deliberately the same
/// write-then-rename: a half-written `version.json` is what a resume would take
/// as proof the conversion finished.
fn strip_reaction_ranges(dir: &Path) {
    let version_path = dir.join("version.json");
    let mut version: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&version_path).expect("read version.json"))
            .expect("parse version.json");
    version
        .as_object_mut()
        .expect("version.json is not a JSON object")
        .remove("reaction_ranges");
    let tmp = dir.join("version.json.tmp");
    std::fs::write(
        &tmp,
        serde_json::to_string_pretty(&version).expect("serialize version.json"),
    )
    .expect("write version.json");
    std::fs::rename(tmp, &version_path).expect("rename version.json");
}

impl Drop for Served {
    fn drop(&mut self) {
        if let Some(tmp) = &self._tmp {
            let _ = std::fs::remove_dir_all(tmp);
        }
    }
}

/// A one-nuclide origin, serving the fixture with byte-range support.
///
/// Counts the bytes it has written for `reactions.arrow`, which is what the
/// saving is actually measured on.
struct Origin {
    port: u16,
    reactions_bytes: Arc<AtomicUsize>,
    /// Answer ranged requests with the whole object and a 200, the way a proxy
    /// that strips the header does.
    ignore_ranges: bool,
}

impl Origin {
    fn start(dir: PathBuf, ignore_ranges: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let reactions_bytes = Arc::new(AtomicUsize::new(0));
        let counter = reactions_bytes.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let _ = serve(stream, &dir, &counter, ignore_ranges);
            }
        });
        Self {
            port,
            reactions_bytes,
            ignore_ranges,
        }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}/Fe56.arrow", self.port)
    }
}

/// Serve one connection, keeping it open for further requests.
///
/// The loader now holds one `reqwest` client for the process, so it reuses a
/// connection across sections and across nuclides. A server that closed after
/// one response would reset those reused connections, so speaking keep-alive is
/// what makes this an origin rather than a one-shot fake, and it exercises the
/// reuse the shared client exists for.
fn serve(
    mut stream: TcpStream,
    dir: &Path,
    counter: &AtomicUsize,
    ignore_ranges: bool,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    loop {
        let mut request = String::new();
        if reader.read_line(&mut request)? == 0 {
            return Ok(()); // client hung up
        }
        let target = request.split_whitespace().nth(1).unwrap_or("/").to_string();
        let name = target.rsplit('/').next().unwrap_or("").to_string();

        let mut range: Option<(usize, usize)> = None;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 || line.trim().is_empty() {
                break;
            }
            if let Some(spec) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                if let Some((a, b)) = spec.trim().split_once('-') {
                    if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                        range = Some((a, b));
                    }
                }
            }
        }

        let Ok(body) = std::fs::read(dir.join(&name)) else {
            stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
            )?;
            continue;
        };

        let (status, slice) = match range {
            // Inclusive of both ends, as the header specifies.
            Some((a, b)) if !ignore_ranges && b < body.len() => {
                ("206 Partial Content", &body[a..=b])
            }
            _ => ("200 OK", &body[..]),
        };
        if name == "reactions.arrow" {
            counter.fetch_add(slice.len(), Ordering::Relaxed);
        }
        stream.write_all(
            format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\n\r\n",
                slice.len()
            )
            .as_bytes(),
        )?;
        stream.write_all(slice)?;
        stream.flush()?;
    }
}

/// A cache directory of this test's own, so a run never touches the real one.
struct Cache(PathBuf);

impl Cache {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "yamc-ranged-{}-{}-{:?}",
            tag,
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("cache dir");
        // `YAMC_CACHE_DIR` names the cache root verbatim, which is the whole
        // reason it exists: this used to redirect by moving the home directory
        // out from under the library, and that is a fight the test cannot win
        // on every platform at once. Setting only HOME left Windows resolving
        // to the real profile, so the library cached there while the
        // assertions looked in this temp directory and found nothing.
        //
        // The tests in this binary each get their own root, so they must not
        // run in parallel against one variable.
        std::env::set_var("YAMC_CACHE_DIR", &p);
        Cache(p)
    }
    /// Where `download_and_cache` puts a nuclide fetched from a raw URL.
    ///
    /// Directly under the root, with no `.cache/yamc` below it: the override
    /// IS the root rather than a home directory to derive one from.
    fn nuclide_dir(&self) -> PathBuf {
        self.0.join("Fe56.arrow")
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Serialises the tests: they share `YAMC_CACHE_DIR` and the process-global
/// nuclide cache.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

fn activation_scope() -> LoadScope {
    LoadScope::activation(ACTIVATION_MTS.iter().copied().collect())
}

/// Download at `scope` from `origin` and load what lands in the cache.
fn download_and_load(
    origin: &Origin,
    scope: &LoadScope,
) -> Result<yamc_nuclide::nuclide::Nuclide, Box<dyn std::error::Error>> {
    let path = yamc_nuclide::url_cache::download_and_cache(
        &origin.url(),
        &origin.url(),
        "Fe56",
        yamc_nuclide::url_cache::DataKind::Neutron,
        scope,
    )?;
    // Straight off disk rather than through the global cache, so each assertion
    // is about what was downloaded rather than what a previous test parsed.
    yamc_nuclide::nuclide::load_nuclide(&path, scope)
}

/// The cross sections, temperature by temperature, for one MT. Empty vectors
/// where the load did not materialize it, which is what an MT outside the scope
/// looks like.
fn xs(nuclide: &yamc_nuclide::nuclide::Nuclide, mt: i32) -> Vec<Vec<f64>> {
    nuclide
        .loaded_temperatures
        .iter()
        .map(|t| {
            let idx = nuclide.get_temp_idx(t).expect("temp idx");
            nuclide.reactions[idx]
                .get(&mt)
                .map(|r| r.cross_section.to_vec())
                .unwrap_or_default()
        })
        .collect()
}

#[test]
fn a_ranged_download_loads_the_same_cross_sections_as_the_whole_file() {
    let _guard = exclusive();
    let Some(dir) = fixture() else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };
    let scope = activation_scope();

    let served = Served::new("equiv", &dir, true);
    let whole = {
        let _cache = Cache::new("whole");
        let origin = Origin::start(served.dir.clone(), false);
        // A full-section scope fetches reactions.arrow whole, as it always has.
        let nuclide = download_and_load(&origin, &LoadScope::full()).expect("whole download");
        let bytes = origin.reactions_bytes.load(Ordering::Relaxed);
        (
            ACTIVATION_MTS
                .iter()
                .map(|mt| xs(&nuclide, *mt))
                .collect::<Vec<_>>(),
            bytes,
        )
    };

    let cache = Cache::new("ranged");
    let origin = Origin::start(served.dir.clone(), false);
    let nuclide = download_and_load(&origin, &scope).expect("ranged download");
    let ranged_bytes = origin.reactions_bytes.load(Ordering::Relaxed);

    // The point of the exercise.
    assert!(
        ranged_bytes * 4 < whole.1,
        "ranged fetch pulled {ranged_bytes} bytes against {} for the whole file",
        whole.1,
    );

    // The whole point of the exercise: same numbers.
    for (i, mt) in ACTIVATION_MTS.iter().enumerate() {
        assert!(!whole.0[i].is_empty(), "fixture should carry MT {mt}");
        assert_eq!(
            xs(&nuclide, *mt),
            whole.0[i],
            "MT {mt} differs between a ranged and a whole download",
        );
    }

    // Cached where a partial belongs, and NOT under the canonical name: that is
    // what keeps every existing completeness gate honest.
    let cached = cache.nuclide_dir();
    assert!(
        cached.join("subset/reactions.arrow").exists(),
        "the spliced subset should be cached under subset/",
    );
    assert!(
        !cached.join("reactions.arrow").exists(),
        "a partial must never be written under the canonical name",
    );
}

/// The cache-poisoning case, end to end. A transport load arriving after an
/// activation load must fetch the whole object rather than reading the subset.
#[test]
fn a_transport_load_after_a_ranged_one_refetches_the_whole_object() {
    let _guard = exclusive();
    let Some(dir) = fixture() else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };
    let cache = Cache::new("topup");
    let served = Served::new("topup", &dir, true);
    let origin = Origin::start(served.dir.clone(), false);

    download_and_load(&origin, &activation_scope()).expect("ranged download");
    let after_ranged = origin.reactions_bytes.load(Ordering::Relaxed);

    let full = download_and_load(&origin, &LoadScope::full()).expect("transport download");
    let whole_bytes = origin.reactions_bytes.load(Ordering::Relaxed) - after_ranged;

    assert!(
        whole_bytes > after_ranged * 4,
        "transport should have refetched the whole object, got {whole_bytes} bytes",
    );
    assert!(
        cache.nuclide_dir().join("reactions.arrow").exists(),
        "the whole object should now be cached",
    );
    assert!(
        !cache.nuclide_dir().join("subset").exists(),
        "the superseded subset should have been dropped",
    );

    // MT 2 is elastic scattering. It is not in the activation set, so if
    // transport were reading the subset it would be silently absent, which is
    // the failure this layout exists to prevent.
    assert!(
        !xs(&full, 2).iter().all(Vec::is_empty),
        "transport load must carry elastic scattering",
    );
}

/// A second activation load asking for MTs the first did not fetch must top the
/// subset up rather than read a set that does not cover it.
#[test]
fn a_wider_activation_load_refetches() {
    let _guard = exclusive();
    let Some(dir) = fixture() else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };
    let _cache = Cache::new("widen");
    let served = Served::new("widen", &dir, true);
    let origin = Origin::start(served.dir.clone(), false);

    let narrow = LoadScope::activation([102].into());
    download_and_load(&origin, &narrow).expect("narrow download");
    let after_narrow = origin.reactions_bytes.load(Ordering::Relaxed);

    // Asking again for the same MT must not fetch a single byte.
    download_and_load(&origin, &narrow).expect("second narrow download");
    assert_eq!(
        origin.reactions_bytes.load(Ordering::Relaxed),
        after_narrow,
        "a covered request must be served from the cache",
    );

    // A wider one must, and must come back with the MT it added.
    let wider = activation_scope();
    let nuclide = download_and_load(&origin, &wider).expect("wider download");
    assert!(
        origin.reactions_bytes.load(Ordering::Relaxed) > after_narrow,
        "a wider MT set must fetch",
    );
    assert!(
        !xs(&nuclide, 103).iter().all(Vec::is_empty),
        "the widened load must carry MT 103",
    );
}

/// A proxy that strips the `Range` header answers 200 with the whole object.
/// That is more bytes than were wanted and still the right ones, so it must be
/// cached as the whole section rather than spliced as though it were a slice.
#[test]
fn an_ignored_range_header_caches_the_whole_object() {
    let _guard = exclusive();
    let Some(dir) = fixture() else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };
    let cache = Cache::new("noranges");
    let served = Served::new("noranges", &dir, true);
    let origin = Origin::start(served.dir.clone(), true);
    assert!(origin.ignore_ranges);

    let nuclide = download_and_load(&origin, &activation_scope()).expect("download");

    assert!(
        cache.nuclide_dir().join("reactions.arrow").exists(),
        "a 200 answer is the whole file and belongs under the canonical name",
    );
    assert!(
        !cache.nuclide_dir().join("subset/reactions.arrow").exists(),
        "nothing should have been spliced",
    );
    for mt in ACTIVATION_MTS {
        assert!(
            !xs(&nuclide, *mt).iter().all(Vec::is_empty),
            "MT {mt} should have loaded",
        );
    }
}

/// A library published before the index exists has no `reaction_ranges` in its
/// `version.json`. That is a normal answer, not an error: the section is fetched
/// whole, exactly as it was before any of this.
#[test]
fn a_library_with_no_index_still_downloads_whole() {
    let _guard = exclusive();
    let Some(dir) = fixture() else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };
    let cache = Cache::new("noindex");
    // Served with `reaction_ranges` stripped, so this is the pre-index case
    // whatever the fetched fixture happens to carry.
    let served = Served::new("noindex", &dir, false);
    let origin = Origin::start(served.dir.clone(), false);

    let nuclide = download_and_load(&origin, &activation_scope()).expect("download");

    assert!(
        cache.nuclide_dir().join("reactions.arrow").exists(),
        "with no index the whole object is the only option",
    );
    assert!(!cache.nuclide_dir().join("subset").exists());
    for mt in ACTIVATION_MTS {
        assert!(
            !xs(&nuclide, *mt).iter().all(Vec::is_empty),
            "MT {mt} should have loaded",
        );
    }
}
