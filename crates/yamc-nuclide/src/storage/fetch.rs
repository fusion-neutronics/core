//! One file downloaded and checked against the size and sha256 it is pinned
//! to: the source archives a local library build starts from.
//!
//! [`download_file`] streams the body into `<dest>.part`, counting and hashing
//! it as it arrives with the same streaming copy the release downloads use,
//! and renames it to `dest` only once both match, so a file at `dest` is the
//! pinned one.
//!
//! Some hosts refuse scripted downloads. In October 2026 the IAEA answered
//! HTTP 403 (a Cloudflare challenge) for the FENDL and IRDFF zips, and JAEA
//! answered `200 OK` with an HTML page of under 1 kB in place of
//! `jendl5-n.tar.gz`, with or without a User-Agent. So a file already at
//! `dest` with the pinned size and sha256 is used without a request, and every
//! failure names the URL, the file, its size and sha256, and `dest`'s folder,
//! where a copy fetched with a web browser is accepted.

use std::fs;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::StatusCode;

use super::release_cache::{
    build_client, copy_hashed, describe, stream_to_file, Timeouts, RETRY_DELAYS_MS,
};
use super::url_cache::get_path_lock;

/// What [`download_file`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// `dest` already held the pinned file, so nothing was requested.
    Reused,
    /// The file was downloaded, checked and renamed into place.
    Downloaded,
}

/// Why [`download_file`] did not leave the pinned file at `dest`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadError {
    /// The host refused a scripted download in a way a web browser gets
    /// past: HTTP 403, a Cloudflare challenge, or an HTML page in place of
    /// the file. A copy saved by a browser is what this file then needs.
    pub needs_browser: bool,
    /// What went wrong, naming the URL, the file and how to supply it.
    pub message: String,
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DownloadError {}

impl From<String> for DownloadError {
    fn from(message: String) -> Self {
        DownloadError {
            needs_browser: false,
            message,
        }
    }
}

/// How much of a body is read before the rest, to tell an HTML page from the
/// file when the body is not the file.
const HEAD: u64 = 1024;

const HTML_PAGE: &str =
    "the host answered with an HTML page in place of the file, as a bot check or a login page does";

/// Make `dest` hold the file at `url` that is `bytes` long with hex sha256
/// `sha256`, creating `dest`'s folder if needed.
///
/// A file already at `dest` with that size and sha256 is used as it is, with
/// no request. Otherwise the body is streamed into `<dest>.part`, counted and
/// hashed as it arrives, and renamed to `dest` once both match. A reply that
/// announces another Content-Length is refused after reading at most the
/// first kilobyte of its body, and no body is read past `bytes + 1`. A
/// different file found at `dest` is replaced only by a checked download. A
/// `.part` an interrupted run left is overwritten by the next download, or
/// removed once `dest` holds the file. One is never removed by an attempt
/// the host refused before sending any of the file, since it may be a web
/// browser's download still in progress.
///
/// A transport failure, a body cut short and a 5xx are retried from the
/// start, with the release downloads' delays. Anything else the host answers
/// is final: another status (a 403 from a bot check, a 404), an HTML page in
/// place of the file, and a body of another size or sha256. That error names
/// the URL, the file, its size and sha256, and `dest`'s folder, where a copy
/// fetched another way (a web browser, another machine) is used when it
/// matches. A `.part` file that cannot be written (a full disk, a quota, no
/// permission) is reported as that, at once.
///
/// [`DownloadError::needs_browser`] is set when the host refused in a way a
/// web browser gets past: HTTP 403, a Cloudflare challenge (answered with a
/// `cf-mitigated: challenge` header, which is not retried even on a 503), or
/// an HTML page in place of the file.
///
/// Concurrent calls for one `dest` in a process take turns, so the later ones
/// find the file in place. Separate processes must not download one `dest`
/// at once; the library builder locks its folder for that.
pub fn download_file(
    url: &str,
    dest: &Path,
    bytes: u64,
    sha256: &str,
) -> Result<Outcome, DownloadError> {
    let dest = std::path::absolute(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let Some(name) = dest.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        return Err(format!("{} is not a path to a file", dest.display()).into());
    };
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("the sha256 given for {name}, {sha256:?}, is not 64 hex digits").into());
    }
    let sha256 = sha256.to_ascii_lowercase();

    let lock = get_path_lock(&dest);
    let _turn = lock.lock().unwrap_or_else(|p| p.into_inner());
    let part = part_path(&dest);
    let found = match existing(&dest, bytes, &sha256)? {
        Existing::Pinned => {
            // What an interrupted download left, which none will now replace.
            fs::remove_file(&part).ok();
            return Ok(Outcome::Reused);
        }
        Existing::Absent => None,
        Existing::Other(what) => Some(what),
    };
    if let Some(folder) = dest.parent() {
        fs::create_dir_all(folder)
            .map_err(|e| format!("could not create {}: {e}", folder.display()))?;
    }
    let client = build_client(Timeouts::DEFAULT)
        .map_err(|e| format!("could not build an HTTP client: {e}"))?;
    let mut last = String::new();
    for &delay_ms in std::iter::once(&0u64).chain(RETRY_DELAYS_MS) {
        if delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(delay_ms));
        }
        match attempt(&client, url, &part, bytes, &sha256) {
            Ok(()) => {
                return fs::rename(&part, &dest)
                    .map(|()| Outcome::Downloaded)
                    .map_err(|e| {
                        fs::remove_file(&part).ok();
                        format!(
                            "could not rename {} to {}: {e}",
                            part.display(),
                            dest.display()
                        )
                        .into()
                    });
            }
            Err(Failure::Transient(why)) => last = why,
            Err(Failure::Refused { why, needs_browser }) => {
                return Err(DownloadError {
                    needs_browser,
                    message: failed(url, &dest, bytes, &sha256, &why, found.as_deref()),
                });
            }
            Err(Failure::Local(why)) => return Err(why.into()),
        }
    }
    let why = format!(
        "{} attempts failed, the last with: {last}",
        RETRY_DELAYS_MS.len() + 1
    );
    Err(failed(url, &dest, bytes, &sha256, &why, found.as_deref()).into())
}

/// What `dest` holds before a download.
enum Existing {
    Absent,
    /// The pinned file.
    Pinned,
    /// Another file, described for the error should the download fail.
    Other(String),
}

/// Whether `dest` already holds the pinned file: the size first, then, only
/// if that matches, the sha256 of the whole file.
fn existing(dest: &Path, bytes: u64, sha256: &str) -> Result<Existing, String> {
    let unreadable = |e: std::io::Error| format!("could not read {}: {e}", dest.display());
    let meta = match fs::metadata(dest) {
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Existing::Absent),
        meta => meta.map_err(unreadable)?,
    };
    if meta.is_dir() {
        return Err(format!(
            "{} is a folder, so the download cannot be saved there",
            dest.display()
        ));
    }
    if meta.len() != bytes {
        return Ok(Existing::Other(format!("{} bytes", meta.len())));
    }
    let mut file = fs::File::open(dest).map_err(unreadable)?;
    let (n, got) = copy_hashed(&mut file, &mut std::io::sink()).map_err(unreadable)?;
    Ok(if n == bytes && got == sha256 {
        Existing::Pinned
    } else {
        Existing::Other(format!("{n} bytes with sha256 {got}"))
    })
}

/// Why an attempt did not leave the pinned file in `<dest>.part`.
enum Failure {
    /// The host's answer, which asking again would not change, and whether a
    /// web browser would get past it.
    Refused { why: String, needs_browser: bool },
    /// A transport failure, a body cut short or a 5xx: worth another attempt.
    Transient(String),
    /// The `.part` file could not be written, which no host can fix.
    Local(String),
}

/// One GET of `url` into `part`, checked against the pins as it streams.
fn attempt(
    client: &Client,
    url: &str,
    part: &Path,
    bytes: u64,
    sha256: &str,
) -> Result<(), Failure> {
    let mut response = client.get(url).send().map_err(|e| {
        // An address reqwest cannot request at all (a scheme it does not
        // speak) fails the same way every time.
        if e.is_builder() {
            Failure::Refused {
                why: describe(&e),
                needs_browser: false,
            }
        } else {
            Failure::Transient(describe(&e))
        }
    })?;
    let status = response.status();
    // Cloudflare marks the page it sends in place of the file this way, with
    // a 403 or a 503. Asking again from a script gets the same page.
    if response
        .headers()
        .get("cf-mitigated")
        .is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"challenge"))
    {
        return Err(Failure::Refused {
            why: format!("the host answered HTTP {status} with a Cloudflare bot check"),
            needs_browser: true,
        });
    }
    if status.is_server_error() {
        return Err(Failure::Transient(format!("HTTP {status}")));
    }
    if !status.is_success() {
        let why = if status == StatusCode::FORBIDDEN {
            ", as a host with a bot check answers a scripted download"
        } else {
            ""
        };
        return Err(Failure::Refused {
            why: format!("the host answered HTTP {status}{why}"),
            needs_browser: status == StatusCode::FORBIDDEN,
        });
    }
    let announced = response.content_length();
    let reading = |e: std::io::Error| Failure::Transient(format!("reading the response body: {e}"));
    let mut head = Vec::new();
    (&mut response)
        .take(HEAD.min(bytes.saturating_add(1)))
        .read_to_end(&mut head)
        .map_err(reading)?;
    let page = looks_like_html(&head);
    if let Some(n) = announced
        && n != bytes
    {
        return Err(Failure::Refused {
            why: if page {
                HTML_PAGE.to_string()
            } else {
                format!("the host announced {n} bytes, not the expected {bytes}")
            },
            needs_browser: page,
        });
    }
    let mut body = (&head[..])
        .chain(&mut response)
        .take(bytes.saturating_add(1));
    let streamed = stream_to_file(&mut body, part);
    if let Ok((n, got)) = &streamed
        && *n == bytes
        && got == sha256
    {
        return Ok(());
    }
    // Only here, once this attempt has written it: a `.part` that a refusal
    // came before may be a web browser's download in progress, which
    // Firefox keeps as `<name>.part` beside where the file will go.
    fs::remove_file(part).ok();
    Err(match streamed {
        Err(e) if written_here(&e) => {
            Failure::Local(format!("could not write {}: {e}", part.display()))
        }
        Err(e) => reading(e),
        Ok((n, got)) => Failure::Refused {
            why: if page {
                HTML_PAGE.to_string()
            } else if n > bytes {
                format!("the host sent more than the expected {bytes} bytes")
            } else if n < bytes {
                format!("the host sent {n} bytes, not the expected {bytes}")
            } else {
                format!("the host sent a different file, with sha256 {got}")
            },
            needs_browser: page,
        },
    })
}

/// Whether a body starts as a web page does: with `<`, after any byte-order
/// mark and whitespace. No archive or ENDF tape starts that way.
fn looks_like_html(head: &[u8]) -> bool {
    head.strip_prefix(b"\xef\xbb\xbf")
        .unwrap_or(head)
        .trim_ascii_start()
        .starts_with(b"<")
}

/// Whether an I/O error came from writing the file here rather than from the
/// transfer, so that retrying the download would only fail again.
fn written_here(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        ErrorKind::StorageFull
            | ErrorKind::QuotaExceeded
            | ErrorKind::FileTooLarge
            | ErrorKind::ReadOnlyFilesystem
            | ErrorKind::PermissionDenied
            | ErrorKind::IsADirectory
    )
}

/// `<dest>.part`, beside `dest`, so the rename stays on one filesystem.
fn part_path(dest: &Path) -> PathBuf {
    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    PathBuf::from(part)
}

/// The error for a download that did not happen, saying how to supply the
/// file instead. `found` describes a different file already at `dest`.
fn failed(
    url: &str,
    dest: &Path,
    bytes: u64,
    sha256: &str,
    why: &str,
    found: Option<&str>,
) -> String {
    let name = dest.file_name().unwrap_or_default().to_string_lossy();
    let folder = dest.parent().unwrap_or(dest).display();
    let found = found
        .map(|f| {
            format!(
                " The file already at {} is {f}, so it was not used.",
                dest.display()
            )
        })
        .unwrap_or_default();
    format!(
        "Could not download {name} from {url}: {why}.{found} Download it with a web browser \
         (or on another machine) and put it in {folder} as {name}: a file there is used, \
         without downloading, when it is {bytes} bytes with sha256 {sha256}."
    )
}

#[cfg(test)]
mod tests;
