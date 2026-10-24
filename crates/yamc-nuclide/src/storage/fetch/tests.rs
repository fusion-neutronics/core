//! [`download_file`] against a local HTTP origin, so every case runs without
//! the network.
//!
//! The origin is the one-thread-per-connection kind `release_cache`'s tests
//! use, answering each path from a list of replies. That origin answers only
//! 200 or 404, always with a Content-Length; these cases need a 403, a
//! `text/html` page, and chunked bodies with no Content-Length, one of which
//! reports how much of itself the client read.

use super::*;
use crate::storage::release::sha256_hex;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

/// The size of a [`Reply::Flood`] body: far more than the test file, and
/// more than the socket buffers between the origin and the client hold.
const FLOOD: usize = 64 << 20;

/// How the origin answers one request.
#[derive(Clone)]
enum Reply {
    /// `200 OK` with a Content-Length.
    File(Vec<u8>),
    /// `200 OK`, chunked, so with no Content-Length, as the NEA data bank
    /// serves the JEFF-4.0 tapes.
    Chunked(Vec<u8>),
    /// A status line, a Content-Type and a body with a Content-Length.
    Status(&'static str, &'static str, Vec<u8>),
    /// `200 OK` with the body's Content-Length, closed after the first 3000
    /// bytes: a transfer cut short.
    Cut(Vec<u8>),
    /// `200 OK`, chunked, [`FLOOD`] bytes long. How many of them were sent
    /// before the client stopped reading goes back on the channel.
    Flood(Sender<usize>),
    /// Close the connection without answering.
    Drop,
}

/// The replies for each path, in turn, the last one for good.
type Replies = Arc<Mutex<HashMap<String, Vec<Reply>>>>;

/// A local HTTP/1.1 origin that records every path asked for.
struct Origin {
    url: String,
    replies: Replies,
    hits: Arc<Mutex<Vec<String>>>,
}

impl Origin {
    fn start() -> Origin {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = Origin {
            url: format!("http://{}/", listener.local_addr().unwrap()),
            replies: Arc::default(),
            hits: Arc::default(),
        };
        let (replies, hits) = (Arc::clone(&origin.replies), Arc::clone(&origin.hits));
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (replies, hits) = (Arc::clone(&replies), Arc::clone(&hits));
                std::thread::spawn(move || answer(stream, &replies, &hits));
            }
        });
        origin
    }

    /// Answer `path` with `replies` in turn, the last one for good, and
    /// return its URL.
    fn serve(&self, path: &str, replies: Vec<Reply>) -> String {
        self.replies
            .lock()
            .unwrap()
            .insert(path.to_string(), replies);
        format!("{}{path}", self.url)
    }

    /// How many requests for `path` arrived.
    fn hits(&self, path: &str) -> usize {
        self.hits
            .lock()
            .unwrap()
            .iter()
            .filter(|hit| *hit == path)
            .count()
    }
}

fn answer(mut stream: TcpStream, replies: &Replies, hits: &Mutex<Vec<String>>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request = String::new();
    if reader.read_line(&mut request).is_err() {
        return;
    }
    // The headers, up to the blank line.
    let mut line = String::new();
    while reader.read_line(&mut line).is_ok_and(|n| n > 2) {
        line.clear();
    }
    let path = request
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .trim_start_matches('/')
        .to_string();
    hits.lock().unwrap().push(path.clone());
    let reply = match replies.lock().unwrap().get_mut(&path) {
        Some(queue) if queue.len() > 1 => queue.remove(0),
        Some(queue) => queue[0].clone(),
        None => Reply::Status("404 Not Found", "text/plain", Vec::new()),
    };
    let head = |status: &str, content_type: &str, framing: String| {
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n{framing}\r\n\
             Connection: close\r\n\r\n"
        )
        .into_bytes()
    };
    let sized = |status, content_type, body: Vec<u8>, promised: usize| {
        let framing = format!("Content-Length: {promised}");
        [head(status, content_type, framing), body].concat()
    };
    let response = match reply {
        Reply::Drop => return,
        Reply::Flood(report) => {
            let framing = "Transfer-Encoding: chunked".to_string();
            let chunk = [b"400\r\n".as_slice(), &[b'x'; 1024], b"\r\n"].concat();
            let mut sent = 0;
            if stream
                .write_all(&head("200 OK", "application/octet-stream", framing))
                .is_ok()
            {
                while sent < FLOOD && stream.write_all(&chunk).is_ok() {
                    sent += 1024;
                }
                let _ = stream.write_all(b"0\r\n\r\n");
            }
            let _ = report.send(sent);
            return;
        }
        Reply::File(body) => {
            let n = body.len();
            sized("200 OK", "application/octet-stream", body, n)
        }
        Reply::Status(status, content_type, body) => {
            let n = body.len();
            sized(status, content_type, body, n)
        }
        Reply::Cut(body) => {
            let n = body.len();
            let mut out = sized("200 OK", "application/octet-stream", body, n);
            out.truncate(out.len() - n + 3000.min(n));
            out
        }
        Reply::Chunked(body) => {
            let framing = "Transfer-Encoding: chunked".to_string();
            let mut out = head("200 OK", "application/octet-stream", framing);
            for chunk in body.chunks(1000) {
                out.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
                out.extend_from_slice(chunk);
                out.extend_from_slice(b"\r\n");
            }
            out.extend_from_slice(b"0\r\n\r\n");
            out
        }
    };
    let _ = stream.write_all(&response);
}

/// A scratch folder that removes itself.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "yamc-fetch-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    /// Where the library builder keeps its archives, `.build/downloads/`.
    /// Joined one part at a time, so it has this platform's separators: on
    /// Windows `std::path::absolute` turns `/` into `\`, so the folder an
    /// error names would not match `.build/downloads` written as one.
    fn folder(&self) -> PathBuf {
        self.0.join(".build").join("downloads")
    }

    /// `name` in [`Scratch::folder`].
    fn downloads(&self, name: &str) -> PathBuf {
        self.folder().join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The file the origin serves: longer than the head read first, so a
/// transfer cut short has written some of it.
fn file() -> Vec<u8> {
    (0..5000u32).map(|i| (i % 251) as u8).collect()
}

/// The size and sha256 the file is pinned to.
fn pins() -> (u64, String) {
    let file = file();
    (file.len() as u64, sha256_hex(&file))
}

/// The page JAEA answered in place of `jendl5-n.tar.gz` in October 2026, as
/// `200 OK` and `text/html`, cut short here.
const JAEA_PAGE: &[u8] = b"<html style=\"height:100%\"><head><META NAME=\"ROBOTS\" \
    CONTENT=\"NOINDEX, NOFOLLOW\"></head><body></body></html>";

/// The start of the challenge page the IAEA answered with a 403 for the
/// FENDL and IRDFF zips in October 2026.
const CHALLENGE: &[u8] =
    b"<!DOCTYPE html><html lang=\"en-US\"><head><title>Just a moment...</title>";

/// The error says how to supply the file instead: where to put it, under
/// which name, and the size and sha256 it must have.
fn assert_names_the_way_out(err: &str, url: &str, dest: &Path) {
    let (bytes, sha256) = pins();
    let name = dest.file_name().unwrap().to_string_lossy().into_owned();
    let folder = dest.parent().unwrap().display().to_string();
    let size = format!("{bytes} bytes with sha256 {sha256}");
    for part in [url, &name, &folder, &size, "web browser"] {
        assert!(err.contains(part), "{part:?} is missing from: {err}");
    }
}

/// Nothing is kept of a failed download.
fn assert_nothing_kept(dest: &Path) {
    assert!(!dest.exists(), "{}", dest.display());
    assert!(!part_path(dest).exists(), "{}", dest.display());
}

/// The file is streamed in, checked and renamed into place, its folder made
/// on the way, and nothing is left beside it. A chunked reply with no
/// Content-Length, from a URL that does not end in the file's name, as the
/// NEA serves JEFF-4.0, is read the same way.
#[test]
fn a_file_is_downloaded_checked_and_renamed_into_place() {
    let origin = Origin::start();
    let (bytes, sha256) = pins();
    let scratch = Scratch::new("downloaded");
    let jeff = "api/records/tw0c6-t1386/files/Radioactive_Decay_Data_JEFF-40.txt/content";
    for (path, name, reply) in [
        ("TENDL-g.tgz", "TENDL-g.tgz", Reply::File(file())),
        (
            jeff,
            "Radioactive_Decay_Data_JEFF-40.txt",
            Reply::Chunked(file()),
        ),
    ] {
        let url = origin.serve(path, vec![reply]);
        let dest = scratch.downloads(name);
        assert_eq!(
            download_file(&url, &dest, bytes, &sha256),
            Ok(Outcome::Downloaded)
        );
        assert_eq!(fs::read(&dest).unwrap(), file());
        assert_eq!(origin.hits(path), 1);
    }
    let mut left: Vec<_> = fs::read_dir(scratch.folder())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    left.sort();
    assert_eq!(left, ["Radioactive_Decay_Data_JEFF-40.txt", "TENDL-g.tgz"]);
}

/// A body of the pinned size with another sha256 is refused, naming both
/// hashes, and nothing of it is kept. It arrived whole, so it is not asked
/// for again.
#[test]
fn a_body_with_another_sha256_is_refused() {
    let origin = Origin::start();
    let (bytes, sha256) = pins();
    let mut other = file();
    other[5] ^= 1;
    let url = origin.serve("jendl5-n.tar.gz", vec![Reply::File(other.clone())]);
    let scratch = Scratch::new("sha256");
    let dest = scratch.downloads("jendl5-n.tar.gz");

    let err = download_file(&url, &dest, bytes, &sha256).unwrap_err();
    let got = format!(
        "the host sent a different file, with sha256 {}",
        sha256_hex(&other)
    );
    assert!(err.contains(&got), "{err}");
    assert_names_the_way_out(&err, &url, &dest);
    assert_nothing_kept(&dest);
    assert_eq!(origin.hits("jendl5-n.tar.gz"), 1);
}

/// A reply that announces another size is refused before its body is
/// streamed, and one with no Content-Length that ends short is refused as
/// short.
#[test]
fn a_body_of_another_size_is_refused() {
    let origin = Origin::start();
    let (bytes, sha256) = pins();
    let scratch = Scratch::new("size");
    for (path, reply, why) in [
        (
            "announced",
            Reply::File([file(), b"!".to_vec()].concat()),
            format!(
                "the host announced {} bytes, not the expected {bytes}",
                bytes + 1
            ),
        ),
        (
            "short",
            Reply::Chunked(file()[..10].to_vec()),
            format!("the host sent 10 bytes, not the expected {bytes}"),
        ),
    ] {
        let url = origin.serve(path, vec![reply]);
        let dest = scratch.downloads(path);
        let err = download_file(&url, &dest, bytes, &sha256).unwrap_err();
        assert!(err.contains(&why), "{err}");
        assert_names_the_way_out(&err, &url, &dest);
        assert_nothing_kept(&dest);
        assert_eq!(origin.hits(path), 1, "{path}");
    }
}

/// A body with no Content-Length that runs past the pinned size is read no
/// further than one byte past it, rather than to its end or until the disk is
/// full: the host is left unable to send most of it.
#[test]
fn a_body_that_runs_on_is_not_read_past_the_pinned_size() {
    let origin = Origin::start();
    let (bytes, sha256) = pins();
    let (report, sent) = channel();
    let url = origin.serve("TENDL-n.tgz", vec![Reply::Flood(report)]);
    let scratch = Scratch::new("flood");
    let dest = scratch.downloads("TENDL-n.tgz");

    let err = download_file(&url, &dest, bytes, &sha256).unwrap_err();
    let why = format!("the host sent more than the expected {bytes} bytes");
    assert!(err.contains(&why), "{err}");
    assert_names_the_way_out(&err, &url, &dest);
    assert_nothing_kept(&dest);
    let sent = sent
        .recv_timeout(Duration::from_secs(30))
        .expect("the origin stops sending");
    assert!(sent < FLOOD, "all {sent} bytes of the body were read");
}

/// An HTML page in place of the file is named as one: JAEA's, whose
/// Content-Length already shows it is not the file, and one sent with no
/// Content-Length and a byte-order mark, told by how its body starts.
#[test]
fn an_html_page_in_place_of_the_file_is_named_as_one() {
    let origin = Origin::start();
    let (bytes, sha256) = pins();
    let scratch = Scratch::new("html");
    let marked = [b"\xef\xbb\xbf\r\n  ".as_slice(), CHALLENGE].concat();
    for (path, reply) in [
        (
            "jendl5-n.tar.gz",
            Reply::Status("200 OK", "text/html", JAEA_PAGE.to_vec()),
        ),
        ("jendl5-pa.tar.gz", Reply::Chunked(marked)),
    ] {
        let url = origin.serve(path, vec![reply]);
        let dest = scratch.downloads(path);
        let err = download_file(&url, &dest, bytes, &sha256).unwrap_err();
        assert!(err.contains("an HTML page in place of the file"), "{err}");
        assert_names_the_way_out(&err, &url, &dest);
        assert_nothing_kept(&dest);
        assert_eq!(origin.hits(path), 1, "{path}");
    }
}

/// A 403, as the IAEA's bot check answers a script, is refused at once and
/// not asked again, and so is any other answer that is not a 5xx.
#[test]
fn a_403_is_refused_at_once() {
    let origin = Origin::start();
    let (bytes, sha256) = pins();
    let scratch = Scratch::new("403");
    for (path, status, content_type, body) in [
        (
            "IRDFF/IRDFF-II_ENDF.zip",
            "403 Forbidden",
            "text/html; charset=UTF-8",
            CHALLENGE.to_vec(),
        ),
        ("moved.zip", "404 Not Found", "text/plain", Vec::new()),
    ] {
        let url = origin.serve(path, vec![Reply::Status(status, content_type, body)]);
        let dest = scratch.downloads(path.rsplit('/').next().unwrap());
        let err = download_file(&url, &dest, bytes, &sha256).unwrap_err();
        assert!(
            err.contains(&format!("the host answered HTTP {status}")),
            "{err}"
        );
        assert_names_the_way_out(&err, &url, &dest);
        assert_nothing_kept(&dest);
        assert_eq!(origin.hits(path), 1, "{path}");
    }
}

/// A copy already in place, as a browser download leaves it, is used with no
/// request, even when the host refuses scripts, and what an interrupted
/// download left beside it goes. A different file there is not taken for it:
/// the download is tried, its failure says what was found and leaves it as it
/// was, and a checked download replaces it.
#[test]
fn a_pinned_file_in_place_is_used_without_a_request() {
    let origin = Origin::start();
    let (bytes, sha256) = pins();
    let refused = Reply::Status("403 Forbidden", "text/html", CHALLENGE.to_vec());
    let url = origin.serve("IRDFF-II_ENDF.zip", vec![refused]);
    let scratch = Scratch::new("reuse");
    let dest = scratch.downloads("IRDFF-II_ENDF.zip");
    fs::create_dir_all(dest.parent().unwrap()).unwrap();
    fs::write(&dest, file()).unwrap();
    fs::write(part_path(&dest), &file()[..3000]).unwrap();

    assert_eq!(
        download_file(&url, &dest, bytes, &sha256),
        Ok(Outcome::Reused)
    );
    assert!(!part_path(&dest).exists());
    let upper = sha256.to_ascii_uppercase();
    assert_eq!(
        download_file(&url, &dest, bytes, &upper),
        Ok(Outcome::Reused)
    );
    assert_eq!(origin.hits("IRDFF-II_ENDF.zip"), 0);

    let partial = file()[..1000].to_vec();
    let mut altered = file();
    altered[0] ^= 1;
    for (wrong, found) in [
        (partial, "is 1000 bytes, so it was not used".to_string()),
        (
            altered.clone(),
            format!(
                "is {bytes} bytes with sha256 {}, so it was not used",
                sha256_hex(&altered)
            ),
        ),
    ] {
        fs::write(&dest, &wrong).unwrap();
        let err = download_file(&url, &dest, bytes, &sha256).unwrap_err();
        assert!(err.contains(&found), "{err}");
        assert!(err.contains("HTTP 403 Forbidden"), "{err}");
        assert_names_the_way_out(&err, &url, &dest);
        assert_eq!(fs::read(&dest).unwrap(), wrong, "left as it was");
    }
    assert_eq!(origin.hits("IRDFF-II_ENDF.zip"), 2);

    origin.serve("IRDFF-II_ENDF.zip", vec![Reply::File(file())]);
    assert_eq!(
        download_file(&url, &dest, bytes, &sha256),
        Ok(Outcome::Downloaded)
    );
    assert_eq!(fs::read(&dest).unwrap(), file());
}

/// A `.part` the host refused before sending any of the file is left as it
/// was: it may be a web browser's download still in progress, which Firefox
/// keeps as `<name>.part`, whether or not an empty file holds the name.
#[test]
fn a_refusal_leaves_a_browser_download_in_progress_alone() {
    let origin = Origin::start();
    let (bytes, sha256) = pins();
    let scratch = Scratch::new("browser");
    fs::create_dir_all(scratch.folder()).unwrap();
    let begun = &file()[..3000];
    for (path, reply, placeholder) in [
        (
            "fendl-3.2d-T1-atom-endf.zip",
            Reply::Status("403 Forbidden", "text/html", CHALLENGE.to_vec()),
            false,
        ),
        (
            "jendl5-n_upd17.tar.gz",
            Reply::Status("200 OK", "text/html", JAEA_PAGE.to_vec()),
            true,
        ),
    ] {
        let url = origin.serve(path, vec![reply]);
        let dest = scratch.downloads(path);
        if placeholder {
            fs::write(&dest, b"").unwrap();
        }
        fs::write(part_path(&dest), begun).unwrap();
        let err = download_file(&url, &dest, bytes, &sha256).unwrap_err();
        assert_names_the_way_out(&err, &url, &dest);
        assert_eq!(fs::read(part_path(&dest)).unwrap(), begun, "{path}");
    }
}

/// A dropped connection, a transfer cut short and a 5xx are retried, each
/// from the start. As many failures in a row as there are attempts give up,
/// naming the last.
#[test]
fn transfer_failures_are_retried_from_the_start() {
    let origin = Origin::start();
    let (bytes, sha256) = pins();
    let scratch = Scratch::new("retry");
    let attempts = RETRY_DELAYS_MS.len() + 1;
    let mut replies = vec![
        Reply::Drop,
        Reply::Cut(file()),
        Reply::Status("503 Service Unavailable", "text/plain", Vec::new()),
    ];
    replies.truncate(attempts - 1);
    replies.push(Reply::File(file()));
    let url = origin.serve("TENDL-n.tgz", replies);
    let dest = scratch.downloads("TENDL-n.tgz");
    assert_eq!(
        download_file(&url, &dest, bytes, &sha256),
        Ok(Outcome::Downloaded)
    );
    assert_eq!(
        fs::read(&dest).unwrap(),
        file(),
        "nothing of the cut attempt"
    );
    assert_eq!(origin.hits("TENDL-n.tgz"), attempts);

    let url = origin.serve("TENDL-g.tgz", vec![Reply::Drop]);
    let dest = scratch.downloads("TENDL-g.tgz");
    let err = download_file(&url, &dest, bytes, &sha256).unwrap_err();
    assert!(
        err.contains(&format!("{attempts} attempts failed")),
        "{err}"
    );
    assert_names_the_way_out(&err, &url, &dest);
    assert_nothing_kept(&dest);
    assert_eq!(origin.hits("TENDL-g.tgz"), attempts);
}

/// A pin that is not a sha256 is refused before anything is asked.
#[test]
fn a_pin_that_is_not_a_sha256_is_refused_before_any_request() {
    let origin = Origin::start();
    let url = origin.serve("x.zip", vec![Reply::File(file())]);
    let scratch = Scratch::new("pin");
    let dest = scratch.downloads("x.zip");
    let not_hex = "g".repeat(64);
    let too_long = format!("{}0", sha256_hex(b"x"));
    for pin in ["", "abc", not_hex.as_str(), too_long.as_str()] {
        let err = download_file(&url, &dest, 5000, pin).unwrap_err();
        assert!(err.contains("is not 64 hex digits"), "{err}");
    }
    assert_eq!(origin.hits("x.zip"), 0);
}

/// An address no request can be sent to, a `file://` URL or one with no
/// scheme, fails the same way every time, so it is refused at once rather
/// than retried, with the same way out.
#[test]
fn an_address_that_cannot_be_requested_is_refused_at_once() {
    let (bytes, sha256) = pins();
    let scratch = Scratch::new("address");
    let dest = scratch.downloads("TENDL-g.tgz");
    for url in [
        "file:///data/TENDL-g.tgz",
        "tendl.imperial.ac.uk/tendl_2025/tar_files/TENDL-g.tgz",
    ] {
        let err = download_file(url, &dest, bytes, &sha256).unwrap_err();
        assert!(!err.contains("attempts failed"), "{err}");
        assert_names_the_way_out(&err, url, &dest);
        assert_nothing_kept(&dest);
    }
}

/// Calls for one file at once take turns: one downloads it, and the others
/// find it in place.
#[test]
fn concurrent_calls_for_one_file_make_one_request() {
    let origin = Origin::start();
    let (bytes, sha256) = pins();
    let url = origin.serve("jendl5-dec.tar.gz", vec![Reply::File(file())]);
    let scratch = Scratch::new("turns");
    let dest = scratch.downloads("jendl5-dec.tar.gz");
    let outcomes: Vec<Outcome> = std::thread::scope(|scope| {
        let calls: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| download_file(&url, &dest, bytes, &sha256).unwrap()))
            .collect();
        calls.into_iter().map(|call| call.join().unwrap()).collect()
    });
    assert_eq!(origin.hits("jendl5-dec.tar.gz"), 1);
    let downloads = outcomes
        .iter()
        .filter(|o| **o == Outcome::Downloaded)
        .count();
    assert_eq!(downloads, 1, "{outcomes:?}");
}

/// A `.part` file that cannot be written is this machine's problem, not the
/// host's: it is reported as it is, at once, without the advice to fetch the
/// file another way.
#[cfg(target_os = "linux")]
#[test]
fn a_full_disk_is_reported_at_once() {
    let origin = Origin::start();
    let (bytes, sha256) = pins();
    let url = origin.serve("TENDL-n.tgz", vec![Reply::File(file())]);
    let scratch = Scratch::new("full");
    let dest = scratch.downloads("TENDL-n.tgz");
    fs::create_dir_all(dest.parent().unwrap()).unwrap();
    // Every write to /dev/full fails with ENOSPC, as on a full disk.
    std::os::unix::fs::symlink("/dev/full", part_path(&dest)).unwrap();

    let err = download_file(&url, &dest, bytes, &sha256).unwrap_err();
    assert!(err.contains("could not write"), "{err}");
    assert!(err.contains("No space left on device"), "{err}");
    assert!(!err.contains("web browser"), "{err}");
    assert_eq!(origin.hits("TENDL-n.tgz"), 1);
    assert_nothing_kept(&dest);
}
