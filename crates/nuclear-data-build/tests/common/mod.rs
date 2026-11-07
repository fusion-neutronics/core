//! Shared test helpers: scratch folders, pins of in-memory files, recipes
//! with test URLs, and a local HTTP origin, so nothing reaches the network.

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use nuclear_data_build::recipe::{Pin, Recipe, Source, Sublibrary};
use sha2::{Digest, Sha256};

/// A folder of its own under the system temporary folder, removed on drop.
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!(
            "nuclear-data-build-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    pub fn dir(&self, name: &str) -> PathBuf {
        let d = self.0.join(name);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

pub fn pin_of(body: &[u8]) -> Pin {
    let sha: String = Sha256::digest(body)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Pin {
        bytes: body.len() as u64,
        sha256: leak(sha),
    }
}

/// A source of `body` served at `url`, saved as `name`.
pub fn source(name: &str, url: &str, body: &[u8], sublibrary: Sublibrary) -> Source {
    Source {
        name: leak(name.to_string()),
        url: leak(url.to_string()),
        pin: pin_of(body),
        sublibrary,
        keep: |_| true,
    }
}

pub fn recipe(keyword: &str, sources: Vec<Source>) -> Recipe<'static> {
    Recipe {
        keyword: leak(keyword.to_string()),
        sources: Box::leak(sources.into_boxed_slice()),
        parts: &[],
        njoy: endf::njoy::Flavour::Upstream,
        decay_library: None,
        unconvertible: &[],
        disk_bytes: 0,
    }
}

/// How the origin answers a path: a status line, extra header lines and a
/// body.
#[derive(Clone)]
pub struct Reply {
    pub status: &'static str,
    pub headers: &'static str,
    pub body: Vec<u8>,
}

impl Reply {
    pub fn file(body: &[u8]) -> Reply {
        Reply {
            status: "200 OK",
            headers: "Content-Type: application/octet-stream\r\n",
            body: body.to_vec(),
        }
    }
}

/// A local HTTP/1.1 origin answering each path from a fixed reply, and
/// counting the requests for each.
pub struct Origin {
    pub url: String,
    replies: Arc<Mutex<HashMap<String, Reply>>>,
    hits: Arc<Mutex<HashMap<String, usize>>>,
}

impl Origin {
    pub fn start() -> Origin {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = Origin {
            url: format!("http://{}/", listener.local_addr().unwrap()),
            replies: Arc::default(),
            hits: Arc::default(),
        };
        let (replies, hits) = (origin.replies.clone(), origin.hits.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let (replies, hits) = (replies.clone(), hits.clone());
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut request = String::new();
                    if reader.read_line(&mut request).is_err() {
                        return;
                    }
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
                    *hits.lock().unwrap().entry(path.clone()).or_default() += 1;
                    let reply = replies
                        .lock()
                        .unwrap()
                        .get(&path)
                        .cloned()
                        .unwrap_or(Reply {
                            status: "404 Not Found",
                            headers: "Content-Type: text/plain\r\n",
                            body: Vec::new(),
                        });
                    let head = format!(
                        "HTTP/1.1 {}\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n",
                        reply.status,
                        reply.headers,
                        reply.body.len()
                    );
                    let _ = stream.write_all(&[head.into_bytes(), reply.body].concat());
                });
            }
        });
        origin
    }

    /// Serve `reply` at `path`, returning its URL.
    pub fn serve(&self, path: &str, reply: Reply) -> String {
        self.replies.lock().unwrap().insert(path.to_string(), reply);
        format!("{}{path}", self.url)
    }

    pub fn hits(&self, path: &str) -> usize {
        self.hits.lock().unwrap().get(path).copied().unwrap_or(0)
    }
}

/// A file's contents.
pub fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap()
}
