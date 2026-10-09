//! The release-aware downloader against a local HTTP origin serving a
//! synthetic release tree, so every case runs without the network.

use super::*;
use crate::storage::release::sha256_hex;
use crate::LoadScope;
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

/// How the origin answers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Behaviour {
    Serve,
    /// Accept the connection and never answer, as a black-holed network does.
    Silent,
}

/// A one-thread-per-connection HTTP/1.1 origin over a path -> bytes map.
struct Origin {
    url: String,
    files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    hits: Arc<Mutex<Vec<String>>>,
    behaviour: Arc<Mutex<Behaviour>>,
    /// Paths whose next answer promises more bytes than it sends, once each.
    cut_once: Arc<Mutex<HashSet<String>>>,
}

impl Origin {
    fn start() -> Origin {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let origin = Origin {
            url,
            files: Arc::default(),
            hits: Arc::default(),
            behaviour: Arc::new(Mutex::new(Behaviour::Serve)),
            cut_once: Arc::default(),
        };
        let (files, hits, behaviour, cut_once) = (
            Arc::clone(&origin.files),
            Arc::clone(&origin.hits),
            Arc::clone(&origin.behaviour),
            Arc::clone(&origin.cut_once),
        );
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let (files, hits, behaviour, cut_once) = (
                    Arc::clone(&files),
                    Arc::clone(&hits),
                    Arc::clone(&behaviour),
                    Arc::clone(&cut_once),
                );
                std::thread::spawn(move || answer(stream, &files, &hits, &behaviour, &cut_once));
            }
        });
        origin
    }

    fn put(&self, path: &str, bytes: &[u8]) {
        self.files
            .lock()
            .unwrap()
            .insert(path.to_string(), bytes.to_vec());
    }

    fn hits(&self) -> Vec<String> {
        self.hits.lock().unwrap().clone()
    }

    fn set(&self, behaviour: Behaviour) {
        *self.behaviour.lock().unwrap() = behaviour;
    }

    /// Publish a release of `keyword`: its files, then its manifest, then the
    /// pointer, in the order the publishing scripts upload them.
    fn publish(&self, keyword: &str, release: &str, files: &[(&str, &[u8])]) {
        let entries: Vec<serde_json::Value> = files
            .iter()
            .map(|(path, bytes)| {
                self.put(&format!("{keyword}/{release}/{path}"), bytes);
                serde_json::json!({"path": path, "bytes": bytes.len(), "sha256": sha256_hex(bytes)})
            })
            .collect();
        let manifest = serde_json::to_vec(&serde_json::json!({
            "schema": "test",
            "keyword": keyword,
            "release": release,
            "format_version": 2,
            "converter_version": "test",
            "created": "2026-10-01T00:00:00Z",
            "files": entries,
        }))
        .unwrap();
        self.put(&format!("{keyword}/{release}/manifest.json"), &manifest);
        self.point(keyword, release, &manifest, 2);
    }

    fn point(&self, keyword: &str, release: &str, manifest: &[u8], format_version: u32) {
        let latest = serde_json::json!({
            "release": release,
            "format_version": format_version,
            "manifest": format!("{release}/manifest.json"),
            "manifest_sha256": sha256_hex(manifest),
            "manifest_bytes": manifest.len(),
        });
        self.put(&format!("{keyword}/latest.json"), latest.to_string().as_bytes());
    }
}

fn answer(
    mut stream: TcpStream,
    files: &Mutex<HashMap<String, Vec<u8>>>,
    hits: &Mutex<Vec<String>>,
    behaviour: &Mutex<Behaviour>,
    cut_once: &Mutex<HashSet<String>>,
) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).map_or(true, |n| n == 0) || line == "\r\n" {
            break;
        }
    }
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .trim_start_matches('/')
        .split('?')
        .next()
        .unwrap_or("")
        .to_string();
    hits.lock().unwrap().push(path.clone());
    if *behaviour.lock().unwrap() == Behaviour::Silent {
        std::thread::sleep(Duration::from_secs(20));
        return;
    }
    let body = files.lock().unwrap().get(&path).cloned();
    let Some(body) = body else {
        let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    };
    let cut = cut_once.lock().unwrap().remove(&path);
    let promised = if cut { body.len() + 100 } else { body.len() };
    let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {promised}\r\nConnection: close\r\n\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&body);
}

/// A scratch cache root that removes itself.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "yamc-release-cache-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        Scratch(p)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const FAST: Timeouts = Timeouts {
    connect: Duration::from_millis(500),
    pointer: Duration::from_millis(300),
    pointer_attempts: 2,
    read: Duration::from_secs(5),
};

fn registry(origin: &str, root: &Scratch) -> Registry {
    Registry::new(origin, Some(root.0.clone()), FAST)
}

/// An origin that refuses connections: a port that was bound and released.
fn dead_origin() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/", listener.local_addr().unwrap());
    drop(listener);
    url
}

/// The required neutron sections of a transport load, with contents that say
/// which nuclide and release they are.
fn nuclide_files(name: &str, release: &str) -> Vec<(String, Vec<u8>)> {
    [
        "version.json",
        "nuclide.arrow",
        "energy.arrow",
        "reactions.arrow",
        "products.arrow",
        "distributions.arrow",
    ]
    .iter()
    .map(|section| {
        (
            format!("neutron/{name}.arrow/{section}"),
            format!("{name} {section} of {release}").into_bytes(),
        )
    })
    .collect()
}

fn publish(origin: &Origin, release: &str, nuclides: &[&str], extra: &[(&str, &[u8])]) {
    let mut owned: Vec<(String, Vec<u8>)> = nuclides
        .iter()
        .flat_map(|n| nuclide_files(n, release))
        .collect();
    owned.extend(extra.iter().map(|(p, b)| (p.to_string(), b.to_vec())));
    let files: Vec<(&str, &[u8])> = owned.iter().map(|(p, b)| (p.as_str(), b.as_slice())).collect();
    origin.publish("endf-b8.1", release, &files);
}

fn fetch(reg: &Registry, name: &str) -> Result<PathBuf, String> {
    fetch_particle(reg, "endf-b8.1", name, DataKind::Neutron, &LoadScope::full())
}

#[test]
fn a_release_file_is_downloaded_verified_and_laid_out_by_release() {
    let origin = Origin::start();
    publish(&origin, "2026-10-01", &["Fe56", "Li6"], &[]);
    let root = Scratch::new("verified");
    let reg = registry(&origin.url, &root);

    let dir = fetch(&reg, "Fe56").expect("download");
    assert_eq!(
        dir,
        root.0.join("endf-b8.1/2026-10-01/neutron/Fe56.arrow"),
        "the cache is <root>/<keyword>/<release>/<path>"
    );
    assert_eq!(
        fs::read(dir.join("reactions.arrow")).unwrap(),
        b"Fe56 reactions.arrow of 2026-10-01"
    );
    assert!(root.0.join("endf-b8.1/2026-10-01/manifest.json").is_file());
    // Optional sections the manifest does not list are absent: not requested.
    assert!(
        !origin.hits().iter().any(|h| h.ends_with("urr.arrow")),
        "{:?}",
        origin.hits()
    );
    let used = reg.data_releases();
    let record = &used["endf-b8.1"];
    assert_eq!(record.release, "2026-10-01");
    assert!(!record.offline);
    assert_eq!(
        record.manifest_sha256,
        sha256_hex(&fs::read(root.0.join("endf-b8.1/2026-10-01/manifest.json")).unwrap())
    );
    // Nothing is left staged.
    let leftovers: Vec<_> = fs::read_dir(root.0.join("endf-b8.1/2026-10-01"))
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with('.'))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn a_corrupted_byte_is_refused_and_never_cached() {
    let origin = Origin::start();
    publish(&origin, "2026-10-01", &["Fe56"], &[]);
    let mut bad = b"Fe56 reactions.arrow of 2026-10-01".to_vec();
    bad[3] ^= 1;
    origin.put("endf-b8.1/2026-10-01/neutron/Fe56.arrow/reactions.arrow", &bad);
    let root = Scratch::new("corrupt");
    let err = fetch(&registry(&origin.url, &root), "Fe56").unwrap_err();
    assert!(err.contains("sha256 mismatch"), "{err}");
    assert!(err.contains("neutron/Fe56.arrow/reactions.arrow"), "{err}");
    assert!(err.contains(&sha256_hex(&bad)), "names what it got: {err}");
    assert!(!root
        .0
        .join("endf-b8.1/2026-10-01/neutron/Fe56.arrow/reactions.arrow")
        .exists());
    // A mismatch is the origin's answer, not a dropped connection: no retry.
    let asks = origin
        .hits()
        .iter()
        .filter(|h| h.ends_with("Fe56.arrow/reactions.arrow"))
        .count();
    assert_eq!(asks, 1);
}

#[test]
fn a_truncated_file_is_refused() {
    let origin = Origin::start();
    publish(&origin, "2026-10-01", &["Fe56"], &[]);
    origin.put(
        "endf-b8.1/2026-10-01/neutron/Fe56.arrow/products.arrow",
        b"Fe56 products",
    );
    let root = Scratch::new("truncated");
    let err = fetch(&registry(&origin.url, &root), "Fe56").unwrap_err();
    assert!(err.contains("size mismatch"), "{err}");
    assert!(err.contains("products.arrow"), "{err}");
}

/// A transfer cut off mid-body is a transport failure and is retried, unlike a
/// body that arrives whole and is wrong.
#[test]
fn a_transfer_cut_short_is_retried() {
    let origin = Origin::start();
    publish(&origin, "2026-10-01", &["Fe56"], &[]);
    origin
        .cut_once
        .lock()
        .unwrap()
        .insert("endf-b8.1/2026-10-01/neutron/Fe56.arrow/nuclide.arrow".into());
    let root = Scratch::new("cut");
    let dir = fetch(&registry(&origin.url, &root), "Fe56").expect("retried");
    assert_eq!(
        fs::read(dir.join("nuclide.arrow")).unwrap(),
        b"Fe56 nuclide.arrow of 2026-10-01"
    );
}

#[test]
fn a_manifest_that_does_not_match_its_pointer_is_refused() {
    let origin = Origin::start();
    publish(&origin, "2026-10-01", &["Fe56"], &[]);
    origin.point("endf-b8.1", "2026-10-01", b"some other manifest", 2);
    let root = Scratch::new("manifest-hash");
    let err = fetch(&registry(&origin.url, &root), "Fe56").unwrap_err();
    assert!(err.contains("manifest"), "{err}");
    assert!(err.contains("mismatch"), "{err}");
    assert!(!root.0.join("endf-b8.1/2026-10-01/neutron").exists());
}

#[test]
fn a_format_this_build_cannot_read_is_refused_before_any_download() {
    let origin = Origin::start();
    publish(&origin, "2027-01-01", &["Fe56"], &[]);
    let manifest = origin
        .files
        .lock()
        .unwrap()
        .get("endf-b8.1/2027-01-01/manifest.json")
        .cloned()
        .unwrap();
    origin.point("endf-b8.1", "2027-01-01", &manifest, 3);
    let root = Scratch::new("format");
    let err = fetch(&registry(&origin.url, &root), "Fe56").unwrap_err();
    assert!(err.contains("format_version 3"), "{err}");
    assert!(err.contains("reads format_version 2"), "{err}");
    assert_eq!(origin.hits(), ["endf-b8.1/latest.json"]);
}

#[test]
fn a_library_not_yet_in_the_release_layout_says_so() {
    let origin = Origin::start();
    let root = Scratch::new("unpublished");
    let err = fetch(&registry(&origin.url, &root), "Fe56").unwrap_err();
    assert!(
        err.contains("has not been published in the release layout"),
        "{err}"
    );
    assert!(err.contains("local directory"), "{err}");
}

#[test]
fn a_nuclide_the_release_lacks_is_refused_listing_what_it_has() {
    let origin = Origin::start();
    publish(&origin, "2026-10-01", &["Fe56", "Li6"], &[]);
    let root = Scratch::new("absent-nuclide");
    let err = fetch(&registry(&origin.url, &root), "Ag104").unwrap_err();
    assert!(err.contains("Ag104") && err.contains("not available"), "{err}");
    assert!(err.contains("Fe56, Li6"), "{err}");
    assert!(
        !origin.hits().iter().any(|h| h.contains("Ag104")),
        "the manifest is the index, so nothing is probed"
    );
}

#[test]
fn a_subsection_the_release_lacks_is_refused_listing_what_it_has() {
    let origin = Origin::start();
    publish(
        &origin,
        "2026-10-01",
        &["Fe56"],
        &[("transmutation/branching.arrow/branching.arrow", b"b")],
    );
    let root = Scratch::new("subsection");
    let reg = registry(&origin.url, &root);
    let err = fetch_subsection(&reg, "endf-b8.1", "decay").unwrap_err();
    assert!(err.contains("does not provide a 'decay'"), "{err}");
    assert!(err.contains("branching"), "{err}");
    let dir = fetch_subsection(&reg, "endf-b8.1", "branching").expect("published");
    assert_eq!(fs::read(dir.join("branching.arrow")).unwrap(), b"b");
}

#[test]
fn a_warm_cache_is_reused_and_needs_no_network() {
    let origin = Origin::start();
    publish(&origin, "2026-10-01", &["Fe56"], &[]);
    let root = Scratch::new("warm");
    fetch(&registry(&origin.url, &root), "Fe56").expect("first download");

    // A new process with the origin up: the pointer is asked, nothing else.
    let before = origin.hits().len();
    fetch(&registry(&origin.url, &root), "Fe56").expect("cache hit");
    assert_eq!(&origin.hits()[before..], ["endf-b8.1/latest.json"]);

    // And with no network at all.
    let reg = registry(&dead_origin(), &root);
    let dir = fetch(&reg, "Fe56").expect("offline from cache");
    assert_eq!(
        fs::read(dir.join("version.json")).unwrap(),
        b"Fe56 version.json of 2026-10-01"
    );
    let record = &reg.data_releases()["endf-b8.1"];
    assert_eq!(record.release, "2026-10-01");
    assert!(record.offline);
}

#[test]
fn a_new_release_replaces_the_old_for_every_file_and_never_mixes() {
    let origin = Origin::start();
    publish(&origin, "2026-10-01", &["Fe56", "Li6"], &[]);
    let root = Scratch::new("switch");
    let old = registry(&origin.url, &root);
    fetch(&old, "Fe56").unwrap();
    fetch(&old, "Li6").unwrap();

    publish(&origin, "2026-11-15", &["Fe56", "Li6"], &[]);
    let new = registry(&origin.url, &root);
    for name in ["Fe56", "Li6"] {
        let dir = fetch(&new, name).unwrap();
        assert_eq!(dir, root.0.join(format!("endf-b8.1/2026-11-15/neutron/{name}.arrow")));
        assert_eq!(
            fs::read(dir.join("reactions.arrow")).unwrap(),
            format!("{name} reactions.arrow of 2026-11-15").into_bytes(),
            "every file comes from the new release, although the old one is cached"
        );
    }
    assert_eq!(new.data_releases()["endf-b8.1"].release, "2026-11-15");
    // The old release stays on disk for offline use, untouched.
    assert_eq!(
        fs::read(root.0.join("endf-b8.1/2026-10-01/neutron/Li6.arrow/reactions.arrow")).unwrap(),
        b"Li6 reactions.arrow of 2026-10-01"
    );
}

/// The offline choice is the newest cached release holding every file the
/// first request needs, and then that release for the rest of the process.
#[test]
fn offline_uses_the_newest_complete_cached_release_and_sticks_to_it() {
    let origin = Origin::start();
    publish(&origin, "2026-10-01", &["Fe56", "Li6"], &[]);
    let root = Scratch::new("offline");
    let reg = registry(&origin.url, &root);
    fetch(&reg, "Fe56").unwrap();
    fetch(&reg, "Li6").unwrap();
    publish(&origin, "2026-11-15", &["Fe56", "Li6"], &[]);
    fetch(&registry(&origin.url, &root), "Fe56").unwrap();
    // Cache: 2026-10-01 holds Fe56 and Li6, 2026-11-15 holds Fe56 only.

    let dead = dead_origin();
    let reg = registry(&dead, &root);
    let li6 = fetch(&reg, "Li6").expect("only 2026-10-01 holds Li6");
    assert!(li6.starts_with(root.0.join("endf-b8.1/2026-10-01")));
    let fe56 = fetch(&reg, "Fe56").expect("from the same release");
    assert!(
        fe56.starts_with(root.0.join("endf-b8.1/2026-10-01")),
        "no mixing: Fe56 comes from the release Li6 did, not the newer one"
    );
    let record = &reg.data_releases()["endf-b8.1"];
    assert_eq!((record.release.as_str(), record.offline), ("2026-10-01", true));

    // Asked for Fe56 first, the newest release holding it wins, and a later
    // Li6 is the clear error rather than a silent switch.
    let reg = registry(&dead, &root);
    let fe56 = fetch(&reg, "Fe56").unwrap();
    assert!(fe56.starts_with(root.0.join("endf-b8.1/2026-11-15")));
    let err = fetch(&reg, "Li6").unwrap_err();
    assert!(err.contains("no connection"), "{err}");
    assert!(err.contains("neutron/Li6.arrow/"), "names the file: {err}");
    assert!(err.contains("2026-11-15"), "{err}");
}

#[test]
fn offline_with_nothing_cached_says_no_connection_was_available() {
    let root = Scratch::new("offline-empty");
    let err = fetch(&registry(&dead_origin(), &root), "Fe56").unwrap_err();
    assert!(err.contains("could not be reached"), "{err}");
    assert!(err.contains("no release of 'endf-b8.1'"), "{err}");
}

/// A network that swallows packets costs the pointer budget and no more,
/// then the cache answers.
#[test]
fn a_silent_origin_falls_back_within_the_pointer_timeout() {
    let origin = Origin::start();
    publish(&origin, "2026-10-01", &["Fe56"], &[]);
    let root = Scratch::new("silent");
    fetch(&registry(&origin.url, &root), "Fe56").unwrap();
    origin.set(Behaviour::Silent);

    let started = Instant::now();
    let dir = fetch(&registry(&origin.url, &root), "Fe56").expect("from cache");
    let took = started.elapsed();
    assert!(dir.starts_with(root.0.join("endf-b8.1/2026-10-01")));
    let budget = FAST.pointer * FAST.pointer_attempts as u32 + Duration::from_secs(2);
    assert!(took < budget, "took {took:?}, budget {budget:?}");
}

/// The documented first-contact budget: an offline machine waits at most a
/// few seconds before the cache is used.
#[test]
fn the_default_timeouts_bound_the_first_contact() {
    let t = Timeouts::DEFAULT;
    assert!(t.connect <= Duration::from_secs(5));
    assert!(t.pointer <= Duration::from_secs(5));
    assert!((1..=2).contains(&t.pointer_attempts));
    assert!(t.pointer.max(t.connect) * t.pointer_attempts as u32 <= Duration::from_secs(10));
    assert!(t.read >= Duration::from_secs(10), "a slow link must not fail a large file");
}

/// The live origin serves, for every keyword, a `latest.json` whose
/// `format_version` this build reads, naming a manifest whose size and sha256
/// match. A keyword whose `latest.json` is absent has not been republished in
/// the release layout yet and is skipped with a notice; once every library is
/// published that way the skip never fires.
///
/// Ignored by default because it needs the network. CI runs it on every pull
/// request and push and in the release gate:
///   cargo test -p yamc-nuclide --features download-tls --lib -- \
///       --ignored origin_serves_a_readable_release_for_every_keyword
#[test]
#[ignore]
fn origin_serves_a_readable_release_for_every_keyword() {
    // A query string the CDN has not seen, so the answer is the origin's.
    let bust = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let client = build_client(Timeouts::DEFAULT).expect("client");
    let get = |url: &str| -> Result<Option<Vec<u8>>, String> {
        let r = client
            .get(format!("{url}?release-check={bust}"))
            .send()
            .map_err(|e| describe(&e))?;
        if r.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let r = r.error_for_status().map_err(|e| describe(&e))?;
        Ok(Some(r.bytes().map_err(|e| describe(&e))?.to_vec()))
    };
    let mut failures = Vec::new();
    for keyword in super::super::url_cache::KEYWORDS {
        let latest_url = format!("{}{keyword}/{LATEST}", super::super::url_cache::ORIGIN);
        let pointer = match get(&latest_url) {
            Ok(Some(bytes)) => match parse_latest(keyword, &bytes) {
                Ok(p) => p,
                Err(e) => {
                    failures.push(e);
                    continue;
                }
            },
            Ok(None) => {
                println!(
                    "::notice::{keyword}: {latest_url} is absent, so the library has not been \
                     published in the release layout yet; skipped. The next data publish turns \
                     this check on."
                );
                continue;
            }
            Err(e) => {
                failures.push(format!("{latest_url}: {e}"));
                continue;
            }
        };
        let manifest_url = format!(
            "{}{keyword}/{}",
            super::super::url_cache::ORIGIN,
            pointer.manifest
        );
        match get(&manifest_url) {
            Ok(Some(bytes)) => match parse_manifest(keyword, &bytes, Some(&pointer)) {
                Ok(release) => println!(
                    "{keyword}: release {} (format_version {}), manifest sha256 {}, {} files",
                    pointer.release,
                    pointer.format_version,
                    release.manifest_sha256,
                    release.manifest.files.len()
                ),
                Err(e) => failures.push(e),
            },
            Ok(None) => failures.push(format!("{manifest_url}: 404, named by {latest_url}")),
            Err(e) => failures.push(format!("{manifest_url}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "the origin does not serve a release this build reads:\n  {}",
        failures.join("\n  ")
    );
}
