//! Getting every source file of a recipe onto local disk, checked against its
//! pin, with one message for whatever has to be fetched with a web browser.
//!
//! Each source is looked for, in this order:
//!
//! 1. in the folder passed as `downloads`, used where it is (a shared copy on
//!    a cluster need not be copied into every build);
//! 2. in `<path>/.build/downloads/`, where earlier runs put it;
//! 3. in the operating system's Downloads folder, where a web browser saves
//!    it, copied into `<path>/.build/downloads/` (the user may clear the
//!    Downloads folder later);
//! 4. at its host, downloaded into `<path>/.build/downloads/`. This is tried
//!    on every run, also for hosts that refuse scripts today, so the manual
//!    step goes away by itself if a host stops refusing.
//!
//! In folders 1 and 3 a file is recognised by its size and sha256, not its
//! name, so a browser's renamed copy (`jendl5-n (1).tar.gz`) is found.
//!
//! Every source is tried before anything fails, and the files that their
//! hosts only serve to a web browser are listed together in one message, so
//! a user downloads them in one go rather than meeting them one run at a
//! time. The files that did download are kept for the next run.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use yamc_nuclide::storage::fetch::download_file;

use crate::recipe::{Pin, Recipe, Source};

/// Where source files are looked for before their hosts are asked.
#[derive(Debug, Clone, Default)]
pub struct Search {
    /// The folder the caller passed as `downloads`, searched first.
    pub downloads: Option<PathBuf>,
    /// The operating system's Downloads folder, if it has one.
    pub os_downloads: Option<PathBuf>,
}

impl Search {
    /// `downloads`, then the operating system's own Downloads folder: the
    /// XDG user directory on Linux (which desktops name in the user's
    /// language), `~/Downloads` on macOS, and the Downloads known folder on
    /// Windows (which users can move to another drive).
    pub fn new(downloads: Option<&Path>) -> Self {
        Search {
            downloads: downloads.map(Path::to_path_buf),
            os_downloads: dirs::download_dir(),
        }
    }
}

/// A source file on local disk, checked against its pin.
#[derive(Debug, Clone)]
pub struct Fetched {
    pub source: Source,
    pub path: PathBuf,
}

/// `<path>/.build/downloads/`.
pub fn downloads_dir(root: &Path) -> PathBuf {
    root.join(".build").join("downloads")
}

/// Make every source of `recipe` available on local disk, checked against
/// its pin, in recipe order.
///
/// Fails, before any request, on a recipe with a source not pinned yet.
/// Otherwise every source is tried; then, if any is missing, one error lists
/// the files whose hosts only serve them to a web browser (with links, sizes
/// and where to save them) and any that failed another way.
pub fn fetch(recipe: &Recipe, root: &Path, search: &Search) -> Result<Vec<Fetched>, String> {
    let unpinned: Vec<&str> = recipe
        .sources
        .iter()
        .filter(|s| s.pin.is_none())
        .map(|s| s.name)
        .collect();
    if !unpinned.is_empty() {
        return Err(format!(
            "{} cannot be built yet: {} of its source files are not pinned to a size and \
             sha256 in this version ({}).",
            recipe.keyword,
            unpinned.len(),
            unpinned.join(", ")
        ));
    }
    let build = downloads_dir(root);
    let mut scans = Scans::default();
    let mut fetched = Vec::new();
    let mut browser = Vec::new();
    let mut failed = Vec::new();
    for source in recipe.sources {
        let pin = source.pin.expect("checked above");
        let dest = build.join(source.name);
        if let Some(dir) = &search.downloads
            && let Some(path) = scans.find(dir, pin)?
        {
            fetched.push(Fetched {
                source: *source,
                path,
            });
            continue;
        }
        if let Some(path) = scans.find(&build, pin)? {
            fetched.push(Fetched {
                source: *source,
                path,
            });
            continue;
        }
        if let Some(dir) = &search.os_downloads
            && let Some(found) = scans.find(dir, pin)?
        {
            copy_into(&found, &dest)?;
            fetched.push(Fetched {
                source: *source,
                path: dest,
            });
            continue;
        }
        match download_file(source.url, &dest, pin.bytes, pin.sha256) {
            Ok(_) => fetched.push(Fetched {
                source: *source,
                path: dest,
            }),
            Err(e) if e.needs_browser => browser.push(*source),
            Err(e) => failed.push(e.message),
        }
    }
    if browser.is_empty() && failed.is_empty() {
        return Ok(fetched);
    }
    Err(missing_message(
        recipe.keyword,
        root,
        search,
        &browser,
        &failed,
    ))
}

/// The sha256 of a file, read in pieces.
pub(crate) fn sha256_file(path: &Path) -> Result<String, String> {
    let unreadable = |e: std::io::Error| format!("could not read {}: {e}", path.display());
    let mut file = fs::File::open(path).map_err(unreadable)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(unreadable)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Copy `from` to `to` through `<to>.part`, so `to` only ever holds the
/// whole file.
fn copy_into(from: &Path, to: &Path) -> Result<(), String> {
    if let Some(dir) = to.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    }
    let mut part = to.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    fs::copy(from, &part)
        .and_then(|_| fs::rename(&part, to))
        .map_err(|e| {
            fs::remove_file(&part).ok();
            format!("could not copy {} to {}: {e}", from.display(), to.display())
        })
}

/// The files in each searched folder, by size, with their sha256 worked out
/// only for files whose size matches a pin, once per run.
#[derive(Default)]
struct Scans {
    by_size: HashMap<PathBuf, Vec<(u64, PathBuf)>>,
    sha256: HashMap<PathBuf, String>,
}

impl Scans {
    /// A file in `dir` (not its subfolders) that is the pinned file, under
    /// any name. A folder that does not exist holds nothing.
    fn find(&mut self, dir: &Path, pin: Pin) -> Result<Option<PathBuf>, String> {
        if !self.by_size.contains_key(dir) {
            self.by_size.insert(dir.to_path_buf(), list(dir)?);
        }
        let candidates: Vec<PathBuf> = self.by_size[dir]
            .iter()
            .filter(|(n, _)| *n == pin.bytes)
            .map(|(_, p)| p.clone())
            .collect();
        for path in candidates {
            let sha = match self.sha256.get(&path) {
                Some(sha) => sha.clone(),
                None => {
                    let sha = sha256_file(&path)?;
                    self.sha256.insert(path.clone(), sha.clone());
                    sha
                }
            };
            if sha == pin.sha256 {
                return Ok(Some(path));
            }
        }
        Ok(None)
    }
}

/// `(size, path)` of every file directly in `dir`. A browser's unfinished
/// download (`.part`, `.crdownload`, `.download`) has another size than the
/// pin, so it is never mistaken for the file.
fn list(dir: &Path) -> Result<Vec<(u64, PathBuf)>, String> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("could not list {}: {e}", dir.display())),
    };
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("could not list {}: {e}", dir.display()))?;
        // Follows symbolic links, so a linked archive counts as a file.
        if let Ok(meta) = fs::metadata(entry.path())
            && meta.is_file()
        {
            files.push((meta.len(), entry.path()));
        }
    }
    files.sort();
    Ok(files)
}

/// A size for people: `1.9 GB`, `12 MB`, `950 kB`.
pub(crate) fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} kB", b / 1e3)
    } else {
        format!("{bytes} bytes")
    }
}

/// The one error for a run with sources missing.
fn missing_message(
    keyword: &str,
    root: &Path,
    search: &Search,
    browser: &[Source],
    failed: &[String],
) -> String {
    let mut out = String::new();
    if !browser.is_empty() {
        out.push_str(&if browser.len() == 1 {
            format!("{keyword} needs 1 file that its host only serves to a web browser.\n")
        } else {
            format!(
                "{keyword} needs {} files that their hosts only serve to a web browser.\n",
                browser.len()
            )
        });
        let mut folders: Vec<String> = Vec::new();
        if let Some(d) = &search.downloads {
            folders.push(d.display().to_string());
        }
        if let Some(d) = &search.os_downloads {
            folders.push(d.display().to_string());
        }
        folders.push(downloads_dir(root).display().to_string());
        out.push_str(&format!(
            "Open each link in a web browser. {} found by its contents, under any name, in {}.\n",
            if browser.len() == 1 {
                "The file is"
            } else {
                "Each file is"
            },
            or_list(&folders)
        ));
        if search.os_downloads.is_none() {
            out.push_str(
                "This system has no Downloads folder, so save them to a folder and pass it as \
                 downloads=, or into the build's own folder above.\n",
            );
        }
        out.push('\n');
        let width = browser.iter().map(|s| s.url.len()).max().unwrap_or(0);
        for s in browser {
            let size = s.pin.map(|p| human(p.bytes)).unwrap_or_default();
            out.push_str(&format!("  {:<width$}  {size}\n", s.url));
        }
        out.push_str("\nThen run the build again.\n");
    }
    if !failed.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!(
            "{keyword}: these source files could not be downloaded:\n"
        ));
        for f in failed {
            out.push_str(&format!("  - {f}\n"));
        }
    }
    out.trim_end().to_string()
}

fn or_list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{} or {last}", init.join(", ")),
    }
}
