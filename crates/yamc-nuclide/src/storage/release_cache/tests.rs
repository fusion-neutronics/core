//! The release-aware downloader against a local HTTP origin serving a
//! synthetic release tree, so every case runs without the network.

use super::*;
use crate::storage::release::{sha256_hex, CACHE_BUST_PARAM};
use crate::LoadScope;
use nuclear_data_schema::energy_ranges::EnergyRanges;
use nuclear_data_schema::reaction_ranges::{Hashes, Range, ReactionRanges};
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

/// How the origin answers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Behaviour {
    /// Serve whole objects, and byte ranges when asked: one range as a plain
    /// 206, several as `multipart/byteranges`, as R2 behind Cloudflare does.
    Serve,
    /// Serve whole objects and ignore every `Range` header, as a proxy that
    /// drops it does.
    IgnoreRanges,
    /// Accept the connection and close it without answering.
    Disconnect,
    /// Accept the connection and never answer, as a black-holed network does.
    Silent,
}

/// One answered request: the path, the `Range` header it carried, and how
/// many bytes of the object the answer carried.
#[derive(Clone, Debug)]
struct Served {
    path: String,
    range: Option<String>,
    bytes: usize,
}

/// A one-thread-per-connection HTTP/1.1 origin over a path -> bytes map.
struct Origin {
    url: String,
    files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    /// What a CDN edge in front of the origin holds, by request target (the
    /// path and the query string, as the edge keys its cache). A request for
    /// one of these is answered from here and never reaches `files`.
    edge: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    /// Every request target asked for, query string included.
    hits: Arc<Mutex<Vec<String>>>,
    served: Arc<Mutex<Vec<Served>>>,
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
            edge: Arc::default(),
            hits: Arc::default(),
            served: Arc::default(),
            behaviour: Arc::new(Mutex::new(Behaviour::Serve)),
            cut_once: Arc::default(),
        };
        let state = State {
            files: Arc::clone(&origin.files),
            edge: Arc::clone(&origin.edge),
            hits: Arc::clone(&origin.hits),
            served: Arc::clone(&origin.served),
            behaviour: Arc::clone(&origin.behaviour),
            cut_once: Arc::clone(&origin.cut_once),
        };
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let state = state.clone();
                std::thread::spawn(move || answer(stream, &state));
            }
        });
        origin
    }

    /// Every answered request for an object whose path ends with `suffix`.
    fn served(&self, suffix: &str) -> Vec<Served> {
        self.served
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.path.ends_with(suffix))
            .cloned()
            .collect()
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
        self.publish_listing(keyword, release, files, Some(&["test"]));
    }

    /// [`Origin::publish`], with the manifest's `converters` given,
    /// or left out for `None`.
    fn publish_listing(
        &self,
        keyword: &str,
        release: &str,
        files: &[(&str, &[u8])],
        converters: Option<&[&str]>,
    ) {
        let entries: Vec<serde_json::Value> = files
            .iter()
            .map(|(path, bytes)| {
                self.put(&format!("{keyword}/{release}/{path}"), bytes);
                serde_json::json!({"path": path, "bytes": bytes.len(), "sha256": sha256_hex(bytes)})
            })
            .collect();
        let mut manifest = serde_json::json!({
            "schema": "test",
            "keyword": keyword,
            "release": release,
            "format_version": 2,
            "created": "2026-10-01T00:00:00Z",
            "files": entries,
        });
        if let Some(converters) = converters {
            manifest["converters"] = serde_json::json!(converters);
        }
        let manifest = serde_json::to_vec(&manifest).unwrap();
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
        self.put(
            &format!("{keyword}/latest.json"),
            latest.to_string().as_bytes(),
        );
    }
}

/// What every connection thread shares.
#[derive(Clone)]
struct State {
    files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    edge: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    hits: Arc<Mutex<Vec<String>>>,
    served: Arc<Mutex<Vec<Served>>>,
    behaviour: Arc<Mutex<Behaviour>>,
    cut_once: Arc<Mutex<HashSet<String>>>,
}

/// The `(first, last)` byte positions a `Range` header names, inclusive.
fn parse_range(header: &str) -> Option<Vec<(usize, usize)>> {
    header
        .trim()
        .strip_prefix("bytes=")?
        .split(',')
        .map(|r| {
            let (first, last) = r.trim().split_once('-')?;
            Some((first.parse().ok()?, last.parse().ok()?))
        })
        .collect()
}

fn answer(mut stream: TcpStream, state: &State) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let mut range = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).map_or(true, |n| n == 0) || line == "\r\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("range")
        {
            range = Some(value.trim().to_string());
        }
    }
    let target = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .trim_start_matches('/')
        .to_string();
    // The origin answers by path; only the edge tells queries apart.
    let path = target.split('?').next().unwrap_or("").to_string();
    state.hits.lock().unwrap().push(target.clone());
    let behaviour = *state.behaviour.lock().unwrap();
    if behaviour == Behaviour::Disconnect {
        return;
    }
    if behaviour == Behaviour::Silent {
        std::thread::sleep(Duration::from_secs(20));
        return;
    }
    let cached = state.edge.lock().unwrap().get(&target).cloned();
    let body = cached.or_else(|| state.files.lock().unwrap().get(&path).cloned());
    let Some(body) = body else {
        let _ = stream
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    };
    let wanted = range
        .as_deref()
        .filter(|_| behaviour == Behaviour::Serve)
        .and_then(parse_range);
    let record = |bytes: usize| {
        state.served.lock().unwrap().push(Served {
            path: path.clone(),
            range: range.clone(),
            bytes,
        })
    };
    match wanted {
        Some(ranges) if ranges.iter().any(|&(f, l)| f > l || l >= body.len()) => {
            let _ = stream.write_all(
                b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
        }
        Some(ranges) if ranges.len() == 1 => {
            let (first, last) = ranges[0];
            let part = &body[first..=last];
            record(part.len());
            let head = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {first}-{last}/{}\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len(),
                part.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(part);
        }
        Some(ranges) => {
            const BOUNDARY: &str = "yamc-test-boundary";
            let mut out = Vec::new();
            for &(first, last) in &ranges {
                out.extend_from_slice(
                    format!(
                        "\r\n--{BOUNDARY}\r\nContent-Type: application/octet-stream\r\n\
                         Content-Range: bytes {first}-{last}/{}\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                );
                out.extend_from_slice(&body[first..=last]);
            }
            out.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
            record(ranges.iter().map(|&(f, l)| l - f + 1).sum());
            let head = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Type: multipart/byteranges; \
                 boundary={BOUNDARY}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                out.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&out);
        }
        None => {
            record(body.len());
            let cut = state.cut_once.lock().unwrap().remove(&path);
            let promised = if cut { body.len() + 100 } else { body.len() };
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {promised}\r\nConnection: close\r\n\r\n"
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&body);
        }
    }
}

/// `hits` with the value of every cache-busting query replaced by `*`, each
/// checked first to be the 16 hex digits [`pointer_url`] writes, so a test can
/// compare whole request lists although every pointer request carries a value
/// of its own.
fn masked(hits: &[String]) -> Vec<String> {
    let marker = format!("?{CACHE_BUST_PARAM}=");
    hits.iter()
        .map(|hit| match hit.split_once(&marker) {
            Some((path, value)) => {
                assert!(
                    value.len() == 16 && value.chars().all(|c| c.is_ascii_hexdigit()),
                    "{hit}: the cache-busting value is not 16 hex digits"
                );
                format!("{path}{marker}*")
            }
            None => hit.clone(),
        })
        .collect()
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

/// An origin that closes connections without answering, keeping its port bound.
fn dead_origin() -> String {
    let origin = Origin::start();
    origin.set(Behaviour::Disconnect);
    origin.url
}

#[test]
fn an_offline_origin_keeps_its_port_reserved() {
    let origin = dead_origin();
    let address = origin
        .strip_prefix("http://")
        .unwrap()
        .trim_end_matches('/');
    assert!(
        TcpListener::bind(address).is_err(),
        "another test must not be able to serve responses from the offline origin"
    );
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
    let files: Vec<(&str, &[u8])> = owned
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_slice()))
        .collect();
    origin.publish("endf-b8.1", release, &files);
}

fn fetch(reg: &Registry, name: &str) -> Result<PathBuf, String> {
    fetch_particle(
        reg,
        "endf-b8.1",
        name,
        DataKind::Neutron,
        &LoadScope::full(),
    )
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
    origin.put(
        "endf-b8.1/2026-10-01/neutron/Fe56.arrow/reactions.arrow",
        &bad,
    );
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
    assert_eq!(
        masked(&origin.hits()),
        ["endf-b8.1/latest.json?release-check=*"]
    );
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
    assert!(
        err.contains("Ag104") && err.contains("not available"),
        "{err}"
    );
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
    assert_eq!(
        masked(&origin.hits()[before..]),
        ["endf-b8.1/latest.json?release-check=*"]
    );

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
        assert_eq!(
            dir,
            root.0
                .join(format!("endf-b8.1/2026-11-15/neutron/{name}.arrow"))
        );
        assert_eq!(
            fs::read(dir.join("reactions.arrow")).unwrap(),
            format!("{name} reactions.arrow of 2026-11-15").into_bytes(),
            "every file comes from the new release, although the old one is cached"
        );
    }
    assert_eq!(new.data_releases()["endf-b8.1"].release, "2026-11-15");
    // The old release stays on disk for offline use, untouched.
    assert_eq!(
        fs::read(
            root.0
                .join("endf-b8.1/2026-10-01/neutron/Li6.arrow/reactions.arrow")
        )
        .unwrap(),
        b"Li6 reactions.arrow of 2026-10-01"
    );
}

/// Every resolution asks for the pointer with a cache-busting value of its
/// own, and for nothing else: the manifest and the release files never change,
/// and asking for them plainly is what lets the CDN's edge cache them.
#[test]
fn each_resolution_busts_the_cache_for_the_pointer_and_only_the_pointer() {
    let origin = Origin::start();
    publish(&origin, "2026-10-01", &["Fe56"], &[]);
    let root = Scratch::new("bust");
    // Two processes, the second reading what the first cached.
    fetch(&registry(&origin.url, &root), "Fe56").expect("download");
    fetch(&registry(&origin.url, &root), "Fe56").expect("cache hit");

    let hits = origin.hits();
    let (pointers, rest): (Vec<String>, Vec<String>) = hits
        .iter()
        .cloned()
        .partition(|hit| hit.starts_with("endf-b8.1/latest.json"));
    assert_eq!(
        masked(&pointers),
        ["endf-b8.1/latest.json?release-check=*"; 2]
    );
    assert_ne!(pointers[0], pointers[1], "two resolutions, two values");
    assert!(
        rest.iter().any(|hit| hit.ends_with("/manifest.json"))
            && rest.iter().any(|hit| hit.ends_with("/reactions.arrow")),
        "{rest:?}"
    );
    assert!(
        rest.iter().all(|hit| !hit.contains('?')),
        "the manifest and the files are asked for plainly: {rest:?}"
    );
}

/// The CDN's edge once answered `latest.json` with a copy 5.8 days old that
/// named a release which had since been deleted, so every download from an
/// empty cache failed with a 404. The edge holds no copy for a query it has
/// not seen, so with a cache-busting value the origin answers, with the
/// release that is current.
#[test]
fn a_stale_pointer_at_the_edge_does_not_decide_the_release() {
    let origin = Origin::start();
    publish(&origin, "2026-10-10", &["Fe56"], &[]);
    let stale = published(&origin, "endf-b8.1/latest.json");
    // The next release goes up, and the old one is deleted ...
    publish(&origin, "2026-10-12", &["Fe56"], &[]);
    origin
        .files
        .lock()
        .unwrap()
        .retain(|path, _| !path.starts_with("endf-b8.1/2026-10-10/"));
    // ... while the edge still answers the plain URL with the old pointer.
    origin
        .edge
        .lock()
        .unwrap()
        .insert("endf-b8.1/latest.json".into(), stale);

    let root = Scratch::new("stale-edge");
    let reg = registry(&origin.url, &root);
    let dir = fetch(&reg, "Fe56").expect("the pointer the origin serves now");
    assert_eq!(dir, root.0.join("endf-b8.1/2026-10-12/neutron/Fe56.arrow"));
    assert_eq!(reg.data_releases()["endf-b8.1"].release, "2026-10-12");
    assert!(
        !origin.hits().iter().any(|hit| hit.contains("2026-10-10")),
        "{:?}",
        origin.hits()
    );
}

/// The builds that wrote a release, as its manifest lists them: a full
/// library, and the chain-only irdff-ii.
const CONVERTERS: [&str; 2] = [
    "yani-python 0.22.1 (core 1111111)",
    "yani-python 0.22.1 (core 2222222)",
];
const IRDFF_II_CONVERTERS: [&str; 1] = ["yani-python 0.22.1 (core 1111111)"];

/// A run records the converters its release's manifest lists, online and
/// from the cache when the origin is out of reach.
#[test]
fn a_run_records_the_converters_that_wrote_its_release() {
    let origin = Origin::start();
    let files = nuclide_files("Fe56", "2026-10-12.1");
    let files: Vec<(&str, &[u8])> = files
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_slice()))
        .collect();
    origin.publish_listing("endf-b8.1", "2026-10-12.1", &files, Some(&CONVERTERS));
    let root = Scratch::new("converters");

    let online = registry(&origin.url, &root);
    fetch(&online, "Fe56").expect("download");
    let record = &online.data_releases()["endf-b8.1"];
    assert_eq!(
        (record.release.as_str(), record.offline),
        ("2026-10-12.1", false)
    );
    assert_eq!(record.converters, CONVERTERS);

    let offline = registry(&dead_origin(), &root);
    fetch(&offline, "Fe56").expect("from the cache");
    let record = &offline.data_releases()["endf-b8.1"];
    assert!(record.offline);
    assert_eq!(record.converters, CONVERTERS);
}

/// A library that publishes only a transmutation subsection, as irdff-ii
/// does, records its converter the same way.
#[test]
fn a_chain_only_library_records_its_converter() {
    let origin = Origin::start();
    origin.publish_listing(
        "irdff-ii",
        "2026-10-12.1",
        &[("transmutation/branching.arrow/branching.arrow", b"b")],
        Some(&IRDFF_II_CONVERTERS),
    );
    let root = Scratch::new("converters-irdff");
    let reg = registry(&origin.url, &root);
    fetch_subsection(&reg, "irdff-ii", "branching").expect("published");
    assert_eq!(
        reg.data_releases()["irdff-ii"].converters,
        IRDFF_II_CONVERTERS
    );
}

/// A cached release whose manifest lists no converters (one written before
/// the publishing scripts listed them) still loads, online and offline, and
/// records an empty list.
#[test]
fn a_release_whose_manifest_lists_no_converters_records_an_empty_list() {
    let origin = Origin::start();
    let files = nuclide_files("Fe56", "2026-10-01");
    let files: Vec<(&str, &[u8])> = files
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_slice()))
        .collect();
    origin.publish_listing("endf-b8.1", "2026-10-01", &files, None);
    let root = Scratch::new("converters-none");

    let online = registry(&origin.url, &root);
    fetch(&online, "Fe56").expect("download");
    assert!(online.data_releases()["endf-b8.1"].converters.is_empty());

    let offline = registry(&dead_origin(), &root);
    fetch(&offline, "Fe56").expect("from the cache");
    let record = &offline.data_releases()["endf-b8.1"];
    assert!(record.offline && record.converters.is_empty());
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
    assert_eq!(
        (record.release.as_str(), record.offline),
        ("2026-10-01", true)
    );

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
    assert!(
        t.read >= Duration::from_secs(10),
        "a slow link must not fail a large file"
    );
}

/// The live origin serves, for every keyword, a `latest.json` whose
/// `format_version` this build reads, naming a manifest whose size and sha256
/// match. Every keyword in `KEYWORDS` must be published: one whose
/// `latest.json` is absent fails, since a wheel released then would offer a
/// library it cannot download. With `YAMC_DATA_RELEASE_ALLOW_UNPUBLISHED=1` an
/// absent keyword is reported with a notice instead, so a pull request is not
/// blocked by the state of the data publish, while a keyword that is published
/// is still checked in full.
///
/// Ignored by default because it needs the network. CI runs it on every pull
/// request (allowing unpublished keywords) and push, and strictly in the
/// release gate:
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
    let allow_unpublished =
        std::env::var("YAMC_DATA_RELEASE_ALLOW_UNPUBLISHED").is_ok_and(|v| v == "1");
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
            Ok(None) if allow_unpublished => {
                println!(
                    "::notice::{keyword}: {latest_url} is absent, so the library has not been \
                     published in the release layout yet. Allowed here; a release is blocked \
                     until it is."
                );
                continue;
            }
            Ok(None) => {
                failures.push(format!(
                    "{latest_url}: 404, so {keyword} is not published in the release layout"
                ));
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

/// A run against the live origin records the converters its release's
/// manifest lists, for a full library (endf-b8.1) and for the chain-only
/// irdff-ii, each read through a transmutation subsection into an empty cache.
/// A release published before the build stamps lists none under
/// `converters`, and records an empty list.
///
/// Ignored by default because it needs the network and downloads about 7 MB:
///   cargo test -p yamc-nuclide --features download-tls --lib -- \
///       --ignored a_run_on_the_live_origin_records_its_converters --nocapture
#[test]
#[ignore]
fn a_run_on_the_live_origin_records_its_converters() {
    let root = Scratch::new("live-converters");
    let reg = Registry::new(
        super::super::url_cache::ORIGIN,
        Some(root.0.clone()),
        Timeouts::DEFAULT,
    );
    for keyword in ["endf-b8.1", "irdff-ii"] {
        fetch_subsection(&reg, keyword, "branching").expect("download");
        let record = &reg.data_releases()[keyword];
        assert!(!record.offline, "{keyword}: the origin was not reached");
        let manifest: serde_json::Value = serde_json::from_slice(
            &fs::read(root.0.join(keyword).join(&record.release).join(MANIFEST)).unwrap(),
        )
        .unwrap();
        let listed = manifest.get("converters").cloned();
        assert_eq!(
            serde_json::json!(record.converters),
            listed.unwrap_or(serde_json::json!([])),
            "{keyword}: the record is the manifest's list"
        );
        println!(
            "{keyword}: release {} converters {:?}",
            record.release, record.converters
        );
    }
}

/// The MTs and temperatures of the synthetic indexed nuclide.
const MTS: [i32; 4] = [1, 2, 102, 105];
const TEMPERATURES: [&str; 3] = ["294K", "600K", "900K"];
/// A batch's length, large beside the schema so a ranged fetch's saving shows.
const BATCH: usize = 4000;

/// The bytes of one synthetic message, distinct for every nuclide and entry.
fn message(name: &str, what: &str, len: usize) -> Vec<u8> {
    let seed = format!("{name} {what} ");
    seed.bytes().cycle().take(len).collect()
}

/// A section laid out as the converter writes one, with its index entries:
/// 8 bytes of magic, a 64-byte schema message, then one batch per key. The
/// bytes are not Arrow, which nothing here parses: the downloader only cuts,
/// checks and splices them.
fn indexed_section(name: &str, keys: &[String]) -> (Vec<u8>, Range, Vec<(String, Range)>) {
    let mut bytes = b"ARROW1\0\0".to_vec();
    let schema = (bytes.len() as u64, 64);
    bytes.extend(message(name, "schema", 64));
    let mut entries = Vec::new();
    for key in keys {
        entries.push((key.clone(), (bytes.len() as u64, BATCH as u64)));
        bytes.extend(message(name, key, BATCH));
    }
    // A footer the index does not name.
    bytes.extend(message(name, "footer", 32));
    (bytes, schema, entries)
}

/// The hashes of every entry of a section.
fn hashes_of(bytes: &[u8], ranges: impl IntoIterator<Item = Range>) -> Hashes {
    ranges
        .into_iter()
        .map(|(off, len)| {
            let piece = &bytes[off as usize..(off + len) as usize];
            ((off, len), sha256_hex(piece))
        })
        .collect()
}

/// A nuclide whose `reactions.arrow` and `energy.arrow` carry a byte-range
/// index in `version.json`, with hashes when `hashed`. `energy.arrow` also has
/// a 0 K grid, which is no temperature of the nuclide and rides with every
/// ranged load.
fn indexed_nuclide(name: &str, hashed: bool) -> Vec<(String, Vec<u8>)> {
    let reaction_keys: Vec<String> = MTS
        .iter()
        .flat_map(|mt| TEMPERATURES.iter().map(move |t| format!("{mt}@{t}")))
        .collect();
    let (reactions, schema, entries) = indexed_section(name, &reaction_keys);
    let mut mts: BTreeMap<i32, BTreeMap<String, Range>> = BTreeMap::new();
    for (key, range) in &entries {
        let (mt, t) = key.split_once('@').unwrap();
        mts.entry(mt.parse().unwrap())
            .or_default()
            .insert(t.to_string(), *range);
    }
    let mut reaction_index = ReactionRanges {
        schema,
        mts,
        sha256: Hashes::new(),
    };

    let energy_keys: Vec<String> = std::iter::once("0K")
        .chain(TEMPERATURES)
        .map(str::to_string)
        .collect();
    let (energy, schema, entries) = indexed_section(name, &energy_keys);
    let mut energy_index = EnergyRanges {
        schema,
        temperatures: entries.into_iter().collect(),
        sha256: Hashes::new(),
    };
    if hashed {
        reaction_index.sha256 = hashes_of(
            &reactions,
            std::iter::once(reaction_index.schema).chain(
                reaction_index
                    .mts
                    .values()
                    .flat_map(|t| t.values().copied()),
            ),
        );
        energy_index.sha256 = hashes_of(
            &energy,
            std::iter::once(energy_index.schema).chain(energy_index.temperatures.values().copied()),
        );
    }
    let version = serde_json::json!({
        "format_version": 2,
        "reaction_ranges": reaction_index.to_json(),
        "energy_ranges": energy_index.to_json(),
    });
    let dir = format!("neutron/{name}.arrow");
    vec![
        (
            format!("{dir}/version.json"),
            serde_json::to_vec_pretty(&version).unwrap(),
        ),
        (
            format!("{dir}/nuclide.arrow"),
            message(name, "nuclide", 100),
        ),
        (format!("{dir}/energy.arrow"), energy),
        (format!("{dir}/reactions.arrow"), reactions),
        (
            format!("{dir}/products.arrow"),
            message(name, "products", 100),
        ),
        (
            format!("{dir}/distributions.arrow"),
            message(name, "distributions", 100),
        ),
    ]
}

fn publish_indexed(origin: &Origin, release: &str, name: &str, hashed: bool) {
    let owned = indexed_nuclide(name, hashed);
    let files: Vec<(&str, &[u8])> = owned
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_slice()))
        .collect();
    origin.publish("endf-b8.1", release, &files);
}

/// The bytes of one published object.
fn published(origin: &Origin, path: &str) -> Vec<u8> {
    origin.files.lock().unwrap()[path].clone()
}

/// The bytes of `section` at `ranges`, spliced as the downloader splices them.
fn spliced(bytes: &[u8], ranges: &[Range]) -> Vec<u8> {
    let bodies: Vec<Vec<u8>> = ranges
        .iter()
        .map(|&(off, len)| bytes[off as usize..(off + len) as usize].to_vec())
        .collect();
    nuclear_data_schema::reaction_ranges::splice_spans(&bodies)
}

fn at_294() -> LoadScope {
    LoadScope::full().with_temperatures(Some(["294".to_string()].into()))
}

fn fetch_scoped(reg: &Registry, name: &str, scope: &LoadScope) -> Result<PathBuf, String> {
    fetch_particle(reg, "endf-b8.1", name, DataKind::Neutron, scope)
}

const FE56: &str = "endf-b8.1/2026-10-01/neutron/Fe56.arrow";

/// A transport load at one temperature fetches just that temperature's
/// batches of both indexed sections, each checked against its hash, and caches
/// them under `temperatures/` where nothing takes them for the whole files.
#[test]
fn a_ranged_fetch_is_verified_and_cached_under_its_own_folder() {
    let origin = Origin::start();
    publish_indexed(&origin, "2026-10-01", "Fe56", true);
    let root = Scratch::new("ranged");
    let reg = registry(&origin.url, &root);

    let dir = fetch_scoped(&reg, "Fe56", &at_294()).expect("ranged download");
    assert!(
        !dir.join("reactions.arrow").exists() && !dir.join("energy.arrow").exists(),
        "the canonical names are kept for the whole verified files"
    );

    let reactions = published(&origin, &format!("{FE56}/reactions.arrow"));
    let version: serde_json::Value =
        serde_json::from_slice(&published(&origin, &format!("{FE56}/version.json"))).unwrap();
    let index = ReactionRanges::from_version_json(&version).unwrap();
    let mut want = vec![index.schema];
    want.extend(MTS.iter().map(|mt| index.mts[mt]["294K"]));
    assert_eq!(
        fs::read(dir.join("temperatures/reactions.arrow")).unwrap(),
        spliced(&reactions, &want),
        "every MT at 294K, after the schema, and nothing else"
    );
    let labels: Vec<String> =
        serde_json::from_slice(&fs::read(dir.join("temperatures/reactions.json")).unwrap())
            .unwrap();
    assert_eq!(labels, ["294K"]);

    let energy = published(&origin, &format!("{FE56}/energy.arrow"));
    let grids = EnergyRanges::from_version_json(&version).unwrap();
    assert_eq!(
        fs::read(dir.join("temperatures/energy.arrow")).unwrap(),
        spliced(
            &energy,
            &[
                grids.schema,
                grids.temperatures["0K"],
                grids.temperatures["294K"]
            ]
        ),
        "the 294K grid, and the 0 K grid every load keeps"
    );

    // Ranged requests, for exactly the entries wanted: a third of the
    // reactions, and two grids of four.
    for (section, whole, batches) in [
        ("reactions.arrow", &reactions, MTS.len()),
        ("energy.arrow", &energy, 2),
    ] {
        let served = origin.served(&format!("Fe56.arrow/{section}"));
        assert!(
            served.iter().all(|s| s.range.is_some()),
            "{section}: {served:?}"
        );
        let bytes: usize = served.iter().map(|s| s.bytes).sum();
        assert_eq!(
            bytes,
            64 + batches * BATCH,
            "{section}: {bytes} of {} bytes fetched",
            whole.len()
        );
    }
    assert!(origin.served("Fe56.arrow/reactions.arrow")[0].bytes * 2 < reactions.len());

    // The ranged copy serves the same load later, online and offline.
    let before = origin.hits().len();
    fetch_scoped(&registry(&origin.url, &root), "Fe56", &at_294()).expect("cache hit");
    assert_eq!(
        masked(&origin.hits()[before..]),
        ["endf-b8.1/latest.json?release-check=*"]
    );
    let offline = registry(&dead_origin(), &root);
    fetch_scoped(&offline, "Fe56", &at_294()).expect("offline from the ranged copy");
    // A load that needs every temperature is not served by it offline.
    let err = fetch_scoped(&offline, "Fe56", &LoadScope::full()).unwrap_err();
    assert!(err.contains("reactions.arrow"), "{err}");

    // Online, the whole files are downloaded, verified, and supersede it.
    let dir = fetch_scoped(&reg, "Fe56", &LoadScope::full()).expect("whole download");
    assert_eq!(fs::read(dir.join("reactions.arrow")).unwrap(), reactions);
    assert!(!dir.join("temperatures/reactions.arrow").exists());
    assert!(!dir.join("temperatures/energy.arrow").exists());
}

/// An activation load ranges `reactions.arrow` by MT at every temperature,
/// verified the same way, and caches the result under `subset/`.
#[test]
fn an_activation_load_fetches_its_mts_verified() {
    let origin = Origin::start();
    publish_indexed(&origin, "2026-10-01", "Li6", true);
    let root = Scratch::new("ranged-mts");
    let scope = LoadScope::activation([105, 16].into());
    let dir = fetch_scoped(&registry(&origin.url, &root), "Li6", &scope).expect("subset");

    let path = "endf-b8.1/2026-10-01/neutron/Li6.arrow";
    let reactions = published(&origin, &format!("{path}/reactions.arrow"));
    let version: serde_json::Value =
        serde_json::from_slice(&published(&origin, &format!("{path}/version.json"))).unwrap();
    let index = ReactionRanges::from_version_json(&version).unwrap();
    let mut want = vec![index.schema];
    want.extend(TEMPERATURES.iter().map(|t| index.mts[&105][*t]));
    assert_eq!(
        fs::read(dir.join("subset/reactions.arrow")).unwrap(),
        spliced(&reactions, &want)
    );
    assert_eq!(
        fs::read_to_string(dir.join("subset/mts.json")).unwrap(),
        "[105]",
        "what the nuclide publishes, not what was asked"
    );
    assert!(!dir.join("reactions.arrow").exists());
    // The activation scope reads energy.arrow at every temperature: whole.
    assert!(dir.join("energy.arrow").is_file());

    let served = origin.served("Li6.arrow/reactions.arrow");
    assert_eq!(served.len(), 1, "one request for the one span: {served:?}");
    let bytes = served[0].bytes;
    assert_eq!(bytes, 64 + 3 * BATCH);
    assert!(
        bytes * 3 < reactions.len(),
        "{bytes} of {}",
        reactions.len()
    );
}

/// A batch the origin serves wrong is refused, naming the nuclide, the file,
/// the cross section and both hashes. It is not asked for again and nothing
/// of the section is cached.
///
/// MT 1 at 294K directly follows the schema, so the two arrive as one merged
/// span: the check has to cut the batch back out of it to catch this.
#[test]
fn a_corrupted_batch_is_refused_naming_it() {
    let origin = Origin::start();
    publish_indexed(&origin, "2026-10-01", "Fe56", true);
    let path = format!("{FE56}/reactions.arrow");
    let good = published(&origin, &path);
    let version: serde_json::Value =
        serde_json::from_slice(&published(&origin, &format!("{FE56}/version.json"))).unwrap();
    let index = ReactionRanges::from_version_json(&version).unwrap();
    let (off, len) = index.mts[&1]["294K"];
    assert_eq!(
        index.schema.0 + index.schema.1,
        off,
        "adjacent to the schema"
    );
    let mut bad = good.clone();
    bad[off as usize + 10] ^= 1;
    origin.put(&path, &bad);

    let root = Scratch::new("ranged-corrupt");
    let err = fetch_scoped(&registry(&origin.url, &root), "Fe56", &at_294()).unwrap_err();
    let expected = sha256_hex(&good[off as usize..(off + len) as usize]);
    let got = sha256_hex(&bad[off as usize..(off + len) as usize]);
    for needle in [
        "Fe56",
        "neutron/Fe56.arrow/reactions.arrow",
        "MT 1 at 294K",
        "sha256 mismatch",
        &expected,
        &got,
    ] {
        assert!(err.contains(needle), "missing {needle:?}: {err}");
    }
    assert_eq!(
        origin.served("Fe56.arrow/reactions.arrow").len(),
        1,
        "no retry"
    );
    let dir = root.0.join(FE56);
    assert!(!dir.join("temperatures/reactions.arrow").exists());
    assert!(!dir.join("reactions.arrow").exists());
}

/// An index from before hashes cannot verify a range, so the section is
/// fetched whole and checked against the manifest, as it always was.
#[test]
fn an_index_without_hashes_falls_back_to_the_whole_verified_file() {
    let origin = Origin::start();
    publish_indexed(&origin, "2026-10-01", "Fe56", false);
    let root = Scratch::new("ranged-unhashed");
    let dir = fetch_scoped(&registry(&origin.url, &root), "Fe56", &at_294()).expect("whole");
    assert_eq!(
        fs::read(dir.join("reactions.arrow")).unwrap(),
        published(&origin, &format!("{FE56}/reactions.arrow"))
    );
    assert!(dir.join("energy.arrow").is_file());
    assert!(!dir.join("temperatures").exists());
    assert!(
        origin.served("").iter().all(|s| s.range.is_none()),
        "no range was asked for: {:?}",
        origin.served("")
    );
}

/// An origin that ignores `Range` sends the whole object, which is checked
/// against the manifest and cached under its canonical name.
#[test]
fn a_whole_answer_to_a_range_request_is_verified_against_the_manifest() {
    let origin = Origin::start();
    publish_indexed(&origin, "2026-10-01", "Fe56", true);
    origin.set(Behaviour::IgnoreRanges);
    let root = Scratch::new("ranged-ignored");
    let dir = fetch_scoped(&registry(&origin.url, &root), "Fe56", &at_294()).expect("whole");
    assert_eq!(
        fs::read(dir.join("reactions.arrow")).unwrap(),
        published(&origin, &format!("{FE56}/reactions.arrow"))
    );
    assert!(!dir.join("temperatures/reactions.arrow").exists());
    assert!(origin
        .served("Fe56.arrow/reactions.arrow")
        .iter()
        .all(|s| s.range.is_some()));

    // And a whole answer that does not match the manifest is refused.
    let origin = Origin::start();
    publish_indexed(&origin, "2026-10-01", "Fe56", true);
    origin.set(Behaviour::IgnoreRanges);
    let path = format!("{FE56}/energy.arrow");
    let mut bad = published(&origin, &path);
    bad[100] ^= 1;
    origin.put(&path, &bad);
    let root = Scratch::new("ranged-ignored-corrupt");
    let err = fetch_scoped(&registry(&origin.url, &root), "Fe56", &at_294()).unwrap_err();
    assert!(err.contains("sha256 mismatch"), "{err}");
    assert!(err.contains("neutron/Fe56.arrow/energy.arrow"), "{err}");
    assert!(!root.0.join(FE56).join("energy.arrow").exists());
}

/// Publish one release holding every one of `names` as [`indexed_nuclide`].
fn publish_indexed_many(origin: &Origin, release: &str, names: &[&str], hashed: bool) {
    let owned: Vec<(String, Vec<u8>)> = names
        .iter()
        .flat_map(|name| indexed_nuclide(name, hashed))
        .collect();
    let files: Vec<(&str, &[u8])> = owned
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_slice()))
        .collect();
    origin.publish("endf-b8.1", release, &files);
}

/// Prefetch `names` of `endf-b8.1` at `scope` through `reg`, as a load of
/// many nuclides does before loading them one at a time.
fn prefetch(reg: &Registry, names: &[&str], scope: &LoadScope) {
    let requests: Vec<crate::storage::url_cache::DataRequest> = names
        .iter()
        .map(|name| crate::storage::url_cache::DataRequest {
            source: "endf-b8.1",
            name,
            kind: DataKind::Neutron,
            scope,
        })
        .collect();
    crate::storage::url_cache::batch::prefetch(reg, &requests);
}

/// Every request `origin` answered from a prefetched reply: `(path, range)`.
fn taken_from(origin: &Origin) -> Vec<(String, Option<String>)> {
    crate::storage::url_cache::batch::TAKEN
        .lock()
        .unwrap()
        .iter()
        .filter_map(|(url, range)| {
            Some((url.strip_prefix(&origin.url)?.to_string(), range.clone()))
        })
        .collect()
}

/// Every request `origin` answered: `(path, range)`, sorted.
fn served_all(origin: &Origin) -> Vec<(String, Option<String>)> {
    let mut served: Vec<_> = origin
        .served("")
        .into_iter()
        .map(|s| (s.path, s.range))
        .collect();
    served.sort();
    served
}

const NAMES: [&str; 3] = ["Fe56", "Li6", "O16"];

/// A keyword prefetch of several nuclides resolves the release once, first,
/// and then asks for exactly what the release downloads would: every file
/// and range once. The downloads take every one of those replies rather than
/// asking again, and cache what they cache without a prefetch.
#[test]
fn a_keyword_prefetch_serves_the_release_downloads_from_its_replies() {
    let origin = Origin::start();
    publish_indexed_many(&origin, "2026-10-01", &NAMES, true);
    let root = Scratch::new("prefetch");
    let reg = registry(&origin.url, &root);
    prefetch(&reg, &NAMES, &at_294());

    let hits = origin.hits();
    assert_eq!(
        masked(&hits[..2]),
        [
            "endf-b8.1/latest.json?release-check=*",
            "endf-b8.1/2026-10-01/manifest.json"
        ]
    );
    let served = served_all(&origin);
    assert_eq!(served.len(), hits.len(), "{hits:?}");
    let mut unique = served.clone();
    unique.dedup();
    assert_eq!(unique, served, "each file and range is asked for once");

    // Every request after the pointer and the manifest was prefetched, and
    // every one of those replies was taken by a download.
    let mut files: Vec<_> = served
        .iter()
        .filter(|(path, _)| path.contains("/neutron/"))
        .cloned()
        .collect();
    assert_eq!(files.len(), served.len() - 2);
    let mut taken = taken_from(&origin);
    files.sort();
    taken.sort();
    assert_eq!(taken, files, "every download was served from the prefetch");
    for name in NAMES {
        for section in ["reactions.arrow", "energy.arrow"] {
            let asked = origin.served(&format!("{name}.arrow/{section}"));
            assert_eq!(asked.len(), 1, "{name} {section}: {asked:?}");
            assert!(asked[0].range.is_some(), "{name} {section} is ranged");
        }
    }

    // The same cache as one nuclide at a time, from the same requests.
    let sequential = Origin::start();
    publish_indexed_many(&sequential, "2026-10-01", &NAMES, true);
    let one_by_one = Scratch::new("prefetch-sequential");
    let seq = registry(&sequential.url, &one_by_one);
    for name in NAMES {
        fetch_scoped(&seq, name, &at_294()).expect("sequential download");
    }
    assert_eq!(served_all(&sequential), served);
    for name in NAMES {
        let dir = format!("endf-b8.1/2026-10-01/neutron/{name}.arrow");
        for file in [
            "version.json",
            "nuclide.arrow",
            "products.arrow",
            "distributions.arrow",
            "temperatures/reactions.arrow",
            "temperatures/reactions.json",
            "temperatures/energy.arrow",
            "temperatures/energy.json",
        ] {
            assert_eq!(
                fs::read(root.0.join(&dir).join(file)).unwrap(),
                fs::read(one_by_one.0.join(&dir).join(file)).unwrap(),
                "{dir}/{file}"
            );
        }
    }

    // And the loads that follow find everything cached.
    let before = origin.hits().len();
    for name in NAMES {
        fetch_scoped(&reg, name, &at_294()).expect("cache hit");
    }
    assert_eq!(origin.hits().len(), before);
}

/// A prefetched whole file is checked against the manifest as one from the
/// network is: a corrupted one is refused, not asked for again, and nothing
/// of its nuclide is cached. The others are.
#[test]
fn a_corrupted_prefetched_file_is_refused() {
    let origin = Origin::start();
    publish_indexed_many(&origin, "2026-10-01", &NAMES, true);
    let path = "endf-b8.1/2026-10-01/neutron/Li6.arrow/nuclide.arrow";
    let mut bad = published(&origin, path);
    bad[10] ^= 1;
    origin.put(path, &bad);
    let root = Scratch::new("prefetch-corrupt-file");
    let reg = registry(&origin.url, &root);
    prefetch(&reg, &NAMES, &at_294());

    assert_eq!(origin.served("Li6.arrow/nuclide.arrow").len(), 1);
    assert!(taken_from(&origin)
        .iter()
        .any(|(p, r)| p == path && r.is_none()));
    let li6 = root.0.join("endf-b8.1/2026-10-01/neutron/Li6.arrow");
    assert!(!li6.join("nuclide.arrow").exists());
    assert!(!li6.join("version.json").exists());
    let fe56 = root.0.join(FE56);
    assert!(fe56.join("nuclide.arrow").is_file());
    assert!(fe56.join("temperatures/reactions.arrow").is_file());

    // The load that follows meets the same bytes and says why.
    let err = fetch_scoped(&reg, "Li6", &at_294()).unwrap_err();
    assert!(err.contains("sha256 mismatch"), "{err}");
    assert!(err.contains("neutron/Li6.arrow/nuclide.arrow"), "{err}");
}

/// A prefetched range is checked against its index hash as one from the
/// network is: a corrupted batch is refused, not asked for again, and the
/// section is not cached.
#[test]
fn a_corrupted_prefetched_range_is_refused() {
    let origin = Origin::start();
    publish_indexed_many(&origin, "2026-10-01", &NAMES, true);
    let path = format!("{FE56}/reactions.arrow");
    let version: serde_json::Value =
        serde_json::from_slice(&published(&origin, &format!("{FE56}/version.json"))).unwrap();
    let (off, _) = ReactionRanges::from_version_json(&version).unwrap().mts[&1]["294K"];
    let mut bad = published(&origin, &path);
    bad[off as usize + 10] ^= 1;
    origin.put(&path, &bad);
    let root = Scratch::new("prefetch-corrupt-range");
    let reg = registry(&origin.url, &root);
    prefetch(&reg, &NAMES, &at_294());

    let served = origin.served("Fe56.arrow/reactions.arrow");
    assert_eq!(served.len(), 1, "no retry: {served:?}");
    assert!(taken_from(&origin)
        .iter()
        .any(|(p, r)| *p == path && *r == served[0].range));
    let fe56 = root.0.join(FE56);
    assert!(!fe56.join("temperatures/reactions.arrow").exists());
    assert!(!fe56.join("reactions.arrow").exists());
    let li6 = root.0.join("endf-b8.1/2026-10-01/neutron/Li6.arrow");
    assert!(li6.join("temperatures/reactions.arrow").is_file());

    let err = fetch_scoped(&reg, "Fe56", &at_294()).unwrap_err();
    assert!(err.contains("MT 1 at 294K"), "{err}");
}

/// An index without hashes cannot verify a range, so the prefetch plans the
/// indexed sections whole, as the download then fetches them.
#[test]
fn without_hashes_a_prefetch_plans_the_indexed_sections_whole() {
    let origin = Origin::start();
    publish_indexed_many(&origin, "2026-10-01", &NAMES, false);
    let root = Scratch::new("prefetch-unhashed");
    let reg = registry(&origin.url, &root);
    prefetch(&reg, &NAMES, &at_294());

    let served = served_all(&origin);
    assert!(served.iter().all(|(_, r)| r.is_none()), "{served:?}");
    let mut files: Vec<_> = served
        .into_iter()
        .filter(|(p, _)| p.contains("/neutron/"))
        .collect();
    let mut taken = taken_from(&origin);
    files.sort();
    taken.sort();
    assert_eq!(taken, files);
    for name in NAMES {
        let dir = root
            .0
            .join(format!("endf-b8.1/2026-10-01/neutron/{name}.arrow"));
        assert_eq!(
            fs::read(dir.join("reactions.arrow")).unwrap(),
            published(
                &origin,
                &format!("endf-b8.1/2026-10-01/neutron/{name}.arrow/reactions.arrow")
            )
        );
        assert!(dir.join("energy.arrow").is_file());
        assert!(!dir.join("temperatures").exists());
    }
}

const THERMAL_LABELS: [&str; 4] = ["296K", "400K", "500K", "600K"];

/// A thermal table whose per-temperature sections are indexed under
/// `temperature_ranges`, each entry hashed, as the converter writes them.
fn publish_thermal(origin: &Origin, release: &str, name: &str) {
    let keys: Vec<String> = THERMAL_LABELS.iter().map(|s| s.to_string()).collect();
    let mut ranges = serde_json::Map::new();
    let mut files = Vec::new();
    let dir = format!("thermal/{name}.arrow");
    for section in ["inelastic.arrow", "elastic.arrow"] {
        let (bytes, schema, entries) = indexed_section(name, &keys);
        let mut index = EnergyRanges {
            schema,
            temperatures: entries.into_iter().collect(),
            sha256: Hashes::new(),
        };
        index.sha256 = hashes_of(
            &bytes,
            std::iter::once(index.schema).chain(index.temperatures.values().copied()),
        );
        ranges.insert(section.to_string(), index.to_json());
        files.push((format!("{dir}/{section}"), bytes));
    }
    let version = serde_json::json!({
        "format_version": 2,
        "particle": "thermal",
        "temperature_ranges": ranges,
    });
    files.push((
        format!("{dir}/version.json"),
        serde_json::to_vec_pretty(&version).unwrap(),
    ));
    files.push((
        format!("{dir}/thermal.arrow"),
        message(name, "thermal", 100),
    ));
    let refs: Vec<(&str, &[u8])> = files
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_slice()))
        .collect();
    origin.publish("endf-b8.1", release, &refs);
}

fn fetch_thermal(reg: &Registry, name: &str, temperature: &str) -> Result<PathBuf, String> {
    let scope = LoadScope::full().with_temperatures(Some([temperature.to_string()].into()));
    fetch_particle(reg, "endf-b8.1", name, DataKind::Thermal, &scope)
}

/// A material at one table temperature fetches that temperature's batch of
/// each per-temperature section alone; one between two fetches both; and the
/// tabulated law (`sab.arrow`) is never fetched by transport.
#[test]
fn a_thermal_table_is_fetched_one_temperature_at_a_time() {
    let origin = Origin::start();
    publish_thermal(&origin, "2026-11-07", "c_Graphite");
    let root = Scratch::new("thermal-ranged");
    let reg = registry(&origin.url, &root);

    let dir = fetch_thermal(&reg, "c_Graphite", "400").expect("ranged thermal download");
    assert!(dir.join("thermal.arrow").exists() && dir.join("version.json").exists());
    assert!(
        !dir.join("inelastic.arrow").exists(),
        "only a ranged copy is kept"
    );
    let base = "endf-b8.1/2026-11-07/thermal/c_Graphite.arrow";
    let version: serde_json::Value =
        serde_json::from_slice(&published(&origin, &format!("{base}/version.json"))).unwrap();
    for section in ["inelastic", "elastic"] {
        let index =
            EnergyRanges::from_json(&version["temperature_ranges"][format!("{section}.arrow")])
                .unwrap();
        let whole = published(&origin, &format!("{base}/{section}.arrow"));
        assert_eq!(
            fs::read(dir.join(format!("temperatures/{section}.arrow"))).unwrap(),
            spliced(&whole, &[index.schema, index.temperatures["400K"]]),
            "{section}: the 400K batch after the schema, and nothing else"
        );
    }
    assert!(!dir.join("sab.arrow").exists());

    // Between two table temperatures: both are fetched, the 400K batch kept.
    let dir = fetch_thermal(&reg, "c_Graphite", "450").expect("bracketing download");
    let labels: Vec<String> =
        serde_json::from_slice(&fs::read(dir.join("temperatures/inelastic.json")).unwrap())
            .unwrap();
    assert_eq!(labels, ["400K", "500K"]);
}

/// Below the table's lowest temperature is refused before anything is
/// fetched, naming what the table has.
#[test]
fn a_thermal_table_is_not_extrapolated() {
    let origin = Origin::start();
    publish_thermal(&origin, "2026-11-07", "c_Graphite");
    let root = Scratch::new("thermal-outside");
    let reg = registry(&origin.url, &root);
    let err = fetch_thermal(&reg, "c_Graphite", "250").unwrap_err();
    assert!(err.contains("296K"), "{err}");
}
