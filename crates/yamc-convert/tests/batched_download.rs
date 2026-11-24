//! `url_cache::prefetch` must cache exactly what downloading the nuclides one
//! at a time caches, with the same requests, but with them in flight together.
//!
//! Served from a local socket, as `ranged_reactions_download.rs` does and for
//! the same reason: the `Range` headers, the `multipart/byteranges` answers and
//! the splice are what the batch plans ahead, and a fake that returned the
//! right bytes would check none of that planning. Each response is held back a
//! little, so requests that are sent together are seen in flight together.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use yamc_nuclide::url_cache::{download_and_cache, prefetch, DataKind, DataRequest};
use yamc_nuclide::LoadScope;

/// Nuclides `scripts/fetch_test_fixtures.py` fetches, all of which carry the
/// byte-range indexes.
const NUCLIDES: &[&str] = &["Fe56", "Li6", "O16", "H1"];

fn fixtures() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc/tests");
    NUCLIDES
        .iter()
        .all(|n| dir.join(format!("{n}.arrow/version.json")).exists())
        .then_some(dir)
}

/// What the origin saw: every request as `(path, range)`, and the most it had
/// in flight at once.
#[derive(Default)]
struct Seen {
    requests: Vec<(String, Option<String>)>,
    in_flight: usize,
    peak: usize,
}

struct Origin {
    port: u16,
    seen: Arc<Mutex<Seen>>,
}

impl Origin {
    /// Serve `<dir>/<Name>.arrow/<section>`, one thread per connection.
    fn start(dir: PathBuf) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Seen::default()));
        let shared = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (dir, seen) = (dir.clone(), shared.clone());
                std::thread::spawn(move || serve(stream, &dir, &seen));
            }
        });
        Self { port, seen }
    }

    fn source(&self, nuclide: &str) -> String {
        format!("http://127.0.0.1:{}/{nuclide}.arrow", self.port)
    }

    fn requests(&self) -> Vec<(String, Option<String>)> {
        let mut r = self.seen.lock().unwrap().requests.clone();
        r.sort();
        r
    }

    fn peak(&self) -> usize {
        self.seen.lock().unwrap().peak
    }
}

/// Answer requests on one keep-alive connection, `Range` headers included.
fn serve(stream: TcpStream, dir: &Path, seen: &Mutex<Seen>) -> std::io::Result<()> {
    let mut stream = stream;
    let mut reader = BufReader::new(stream.try_clone()?);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Ok(());
        }
        let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
        let mut range = None;
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header)? == 0 || header.trim().is_empty() {
                break;
            }
            if let Some(value) = header.to_ascii_lowercase().strip_prefix("range:") {
                range = Some(value.trim().to_string());
            }
        }
        {
            let mut s = seen.lock().unwrap();
            s.requests.push((path.clone(), range.clone()));
            s.in_flight += 1;
            s.peak = s.peak.max(s.in_flight);
        }
        std::thread::sleep(Duration::from_millis(20));
        let body = std::fs::read(dir.join(path.trim_start_matches('/')));
        let (status, content_type, payload) = match (body, &range) {
            (Err(_), _) => ("404 Not Found", "text/plain".to_string(), Vec::new()),
            (Ok(body), None) => ("200 OK", "application/octet-stream".to_string(), body),
            (Ok(body), Some(range)) => {
                let spans: Vec<(usize, usize)> = range
                    .trim_start_matches("bytes=")
                    .split(',')
                    .map(|one| {
                        let (a, b) = one.split_once('-').unwrap();
                        (a.parse().unwrap(), b.parse().unwrap())
                    })
                    .collect();
                if let [(a, b)] = spans[..] {
                    (
                        "206 Partial Content",
                        "application/octet-stream".to_string(),
                        body[a..=b].to_vec(),
                    )
                } else {
                    let mut multipart = Vec::new();
                    for (a, b) in &spans {
                        multipart.extend_from_slice(
                            format!(
                                "\r\n--BOUNDARY\r\nContent-Type: application/octet-stream\r\n\
                                 Content-Range: bytes {a}-{b}/{}\r\n\r\n",
                                body.len()
                            )
                            .as_bytes(),
                        );
                        multipart.extend_from_slice(&body[*a..=*b]);
                    }
                    multipart.extend_from_slice(b"\r\n--BOUNDARY--\r\n");
                    (
                        "206 Partial Content",
                        "multipart/byteranges; boundary=BOUNDARY".to_string(),
                        multipart,
                    )
                }
            }
        };
        seen.lock().unwrap().in_flight -= 1;
        stream.write_all(
            format!(
                "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
                payload.len()
            )
            .as_bytes(),
        )?;
        stream.write_all(&payload)?;
        stream.flush()?;
    }
}

/// A cache root of this test's own, named through the test hook.
struct Cache(PathBuf);

impl Cache {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("yamc-batched-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("cache dir");
        yamc_nuclide::url_cache::set_cache_root_for_tests(Some(p.clone()));
        Cache(p)
    }

    /// Every file in the cache, by path relative to the root, with its bytes.
    fn contents(&self) -> BTreeMap<String, Vec<u8>> {
        fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
            for entry in std::fs::read_dir(dir).expect("read cache") {
                let path = entry.expect("entry").path();
                if path.is_dir() {
                    walk(root, &path, out);
                } else {
                    let rel = path
                        .strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
                    out.insert(rel, std::fs::read(&path).expect("read cached file"));
                }
            }
        }
        let mut out = BTreeMap::new();
        walk(&self.0, &self.0, &mut out);
        out
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Serialises the tests: they share the cache-root hook.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// Download `nuclides` at `scope` one at a time, the way a load loop does,
/// optionally after prefetching them all. Answers what was cached and the
/// origin that served it.
fn download(
    dir: &Path,
    tag: &str,
    nuclides: &[&str],
    scope: &LoadScope,
    batched: bool,
) -> (BTreeMap<String, Vec<u8>>, Origin) {
    let cache = Cache::new(tag);
    let origin = Origin::start(dir.to_path_buf());
    let sources: Vec<String> = nuclides.iter().map(|n| origin.source(n)).collect();
    if batched {
        let requests: Vec<DataRequest> = nuclides
            .iter()
            .zip(&sources)
            .map(|(name, source)| DataRequest {
                source,
                name,
                kind: DataKind::Neutron,
                scope,
            })
            .collect();
        prefetch(&requests);
    }
    let before = origin.seen.lock().unwrap().requests.len();
    for (name, source) in nuclides.iter().zip(&sources) {
        download_and_cache(source, source, name, DataKind::Neutron, scope).expect("download");
    }
    if batched {
        assert_eq!(
            origin.seen.lock().unwrap().requests.len(),
            before,
            "the loads after a prefetch went to the network"
        );
    }
    (cache.contents(), origin)
}

/// Prefetching caches byte for byte what the one-at-a-time loop caches, with
/// the same requests and no others, and has more than one in flight.
fn same_as_sequential(tag: &str, scope: LoadScope) {
    let _guard = exclusive();
    let Some(dir) = fixtures() else {
        eprintln!("skip: fixtures not present");
        return;
    };
    let (one_at_a_time, sequential) =
        download(&dir, &format!("{tag}-seq"), NUCLIDES, &scope, false);
    let (batched, origin) = download(&dir, &format!("{tag}-batch"), NUCLIDES, &scope, true);

    assert!(!one_at_a_time.is_empty());
    assert_eq!(
        one_at_a_time.keys().collect::<Vec<_>>(),
        batched.keys().collect::<Vec<_>>(),
        "a prefetch cached different files"
    );
    for (path, bytes) in &one_at_a_time {
        assert!(batched[path] == *bytes, "{path} differs after a prefetch");
    }
    assert_eq!(
        sequential.requests(),
        origin.requests(),
        "a prefetch made different requests"
    );
    assert_eq!(sequential.peak(), 1);
    assert!(
        origin.peak() > 1,
        "a prefetch had only one request in flight"
    );
}

#[test]
fn an_activation_load_caches_the_same_with_and_without_a_prefetch() {
    let mts = [16, 102, 103, 105, 107].into_iter().collect();
    same_as_sequential("activation", LoadScope::activation(mts));
}

#[test]
fn a_transport_load_at_one_temperature_caches_the_same_with_and_without_a_prefetch() {
    let scope = LoadScope::full().with_temperatures(Some(["294".to_string()].into()));
    same_as_sequential("temperature", scope);
}

#[test]
fn a_whole_transport_load_caches_the_same_with_and_without_a_prefetch() {
    same_as_sequential("whole", LoadScope::full());
}

/// A prefetch over a cache that already holds some of the data fetches only
/// the rest, as the loop would: the ranged sections are planned from the
/// cached index in one round.
#[test]
fn a_top_up_after_a_prefetch_matches_the_loop() {
    let _guard = exclusive();
    let Some(dir) = fixtures() else {
        eprintln!("skip: fixtures not present");
        return;
    };
    let narrow = LoadScope::activation([102].into_iter().collect());
    let wide = LoadScope::full().with_temperatures(Some(["294".to_string()].into()));
    let run = |tag: &str, batched: bool| {
        let cache = Cache::new(tag);
        let origin = Origin::start(dir.clone());
        let sources: Vec<String> = NUCLIDES.iter().map(|n| origin.source(n)).collect();
        for scope in [&narrow, &wide] {
            if batched {
                let requests: Vec<DataRequest> = NUCLIDES
                    .iter()
                    .zip(&sources)
                    .map(|(name, source)| DataRequest {
                        source,
                        name,
                        kind: DataKind::Neutron,
                        scope,
                    })
                    .collect();
                prefetch(&requests);
            }
            for (name, source) in NUCLIDES.iter().zip(&sources) {
                download_and_cache(source, source, name, DataKind::Neutron, scope)
                    .expect("download");
            }
        }
        (cache.contents(), origin.requests())
    };
    let (seq_files, seq_requests) = run("topup-seq", false);
    let (batch_files, batch_requests) = run("topup-batch", true);
    assert_eq!(seq_requests, batch_requests);
    assert!(
        seq_files == batch_files,
        "a top-up after a prefetch cached different files"
    );
}

/// A nuclide the origin does not publish fails the load exactly as it does
/// without a prefetch, and does not stop the others from being fetched.
#[test]
fn an_unpublished_nuclide_still_fails_its_load() {
    let _guard = exclusive();
    let Some(dir) = fixtures() else {
        eprintln!("skip: fixtures not present");
        return;
    };
    let _cache = Cache::new("absent");
    let origin = Origin::start(dir.clone());
    let scope = LoadScope::full();
    let names = ["Li6", "Xx999"];
    let sources: Vec<String> = names.iter().map(|n| origin.source(n)).collect();
    let requests: Vec<DataRequest> = names
        .iter()
        .zip(&sources)
        .map(|(name, source)| DataRequest {
            source,
            name,
            kind: DataKind::Neutron,
            scope: &scope,
        })
        .collect();
    prefetch(&requests);
    download_and_cache(&sources[0], &sources[0], "Li6", DataKind::Neutron, &scope)
        .expect("Li6 is published");
    let err = download_and_cache(&sources[1], &sources[1], "Xx999", DataKind::Neutron, &scope)
        .expect_err("Xx999 is not published");
    assert!(err.to_string().contains("404"), "{err}");
}
