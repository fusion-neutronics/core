//! Release folders on local disk.
//!
//! A library downloaded or built whole is laid out as its published release
//! is (see [`super::release`]): `manifest.json` at the top, then
//! `neutron/<Name>.arrow/`, `photon/<Name>.arrow/` and
//! `transmutation/<subsection>.arrow/`. Named as a data source, the folder
//! serves every setting from that one path, and it is read as a published
//! release is, with its own manifest in place of the origin's:
//!
//! - the manifest is parsed once per folder and process (again only when it
//!   is rewritten, as a rebuild does), and a release in a `format_version`
//!   this build does not read is refused;
//! - the files of each nuclide, element or subsection read from the folder are
//!   checked against the sizes the manifest lists (hashing them is a full
//!   verification's job, not every load's);
//! - a transmutation subsection the manifest does not list is refused by name,
//!   and a file it lists that the folder does not hold is refused as an
//!   incomplete copy;
//! - each folder read from is recorded for the run's provenance ([`used`]),
//!   keyed by its absolute path.
//!
//! A directory with no `manifest.json`, or with the one the converters write
//! at the top of a chain directory (it lists `subsections`, not `files`), is
//! not a release folder, and is read as it is.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use once_cell::sync::Lazy;

use super::release::{check_sizes, parse_local_manifest, DataRelease, Release, MANIFEST};
use super::url_cache::{transmutation_sections, DataKind};

/// A release folder on local disk, with its parsed manifest.
#[derive(Debug)]
pub(crate) struct LocalRelease {
    /// The folder as an absolute path (see [`absolute`]). Served paths start
    /// with it.
    root: PathBuf,
    release: Release,
    /// What [`used`] records for the folder, under `root` as a string.
    record: DataRelease,
}

/// What a folder's manifest parsed to, and the size and modification time it
/// had then. A manifest rewritten since (a folder rebuilt in the same process)
/// is read again.
struct Parsed {
    stamp: (u64, Option<SystemTime>),
    release: Option<Arc<LocalRelease>>,
}

/// Every folder whose manifest has been read, by absolute path.
static FOLDERS: Lazy<Mutex<HashMap<PathBuf, Parsed>>> = Lazy::new(Default::default);

/// Every release folder a file was served from, by absolute path.
static USED: Lazy<Mutex<BTreeMap<String, DataRelease>>> = Lazy::new(Default::default);

/// `dir` as an absolute path, its `.` components and any trailing separator
/// dropped, so every spelling of a folder is one key. Symlinks are kept: the
/// folder is recorded as it was named. `dir` itself where the platform gives
/// no working directory.
fn absolute(dir: &Path) -> PathBuf {
    std::path::absolute(dir)
        .map(|path| path.components().collect())
        .unwrap_or_else(|_| dir.to_path_buf())
}

/// The release folder at `dir`, or `None` when `dir` holds no release
/// manifest. The manifest is parsed on the first call for a folder and kept.
pub(crate) fn open(dir: &Path) -> Result<Option<Arc<LocalRelease>>, String> {
    let manifest = dir.join(MANIFEST);
    let Ok(meta) = std::fs::metadata(&manifest) else {
        return Ok(None);
    };
    if !meta.is_file() {
        return Ok(None);
    }
    let root = absolute(dir);
    let stamp = (meta.len(), meta.modified().ok());
    let mut folders = FOLDERS.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(parsed) = folders.get(&root)
        && parsed.stamp == stamp
    {
        return Ok(parsed.release.clone());
    }
    let bytes = std::fs::read(&manifest)
        .map_err(|e| format!("could not read {}: {e}", manifest.display()))?;
    let release = parse_local_manifest(&manifest, &bytes)?.map(|release| {
        let record = DataRelease {
            release: release.manifest.release.clone(),
            manifest_sha256: release.manifest_sha256.clone(),
            format_version: release.manifest.format_version,
            offline: false,
            converter_versions: release.manifest.converter_versions.clone(),
        };
        Arc::new(LocalRelease {
            root: root.clone(),
            release,
            record,
        })
    });
    folders.insert(
        root,
        Parsed {
            stamp,
            release: release.clone(),
        },
    );
    Ok(release)
}

/// Every local release folder a file was served from in this process, keyed
/// by its absolute path.
pub(crate) fn used() -> BTreeMap<String, DataRelease> {
    USED.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

/// Whether `path` lies in the folder `key` names, `key` being a [`used`] key.
pub(crate) fn holds(key: &str, path: &Path) -> bool {
    Path::new(key).is_absolute() && absolute(path).starts_with(key)
}

impl LocalRelease {
    /// The folder of nuclide (or element) `name`, with the sizes of its files
    /// checked against the manifest.
    pub(crate) fn particle(&self, kind: DataKind, name: &str) -> Result<PathBuf, String> {
        let entry = match kind {
            DataKind::Neutron => format!("neutron/{name}.arrow"),
            DataKind::Photon => format!("photon/{name}.arrow"),
            DataKind::Thermal => format!("thermal/{name}.arrow"),
        };
        check_sizes(&self.root, &self.release.manifest, &entry)?;
        self.record_use();
        Ok(self.root.join(entry))
    }

    /// The folder of transmutation `subsection`, `transmutation/<subsection>.arrow/`.
    ///
    /// Refused when the release does not carry the subsection, listing the
    /// ones it does, and when the folder lacks a file the manifest lists for
    /// it, since that loads as a subsection with less in it than the release
    /// has.
    pub(crate) fn subsection(&self, subsection: &str) -> Result<PathBuf, String> {
        let sections = transmutation_sections(subsection);
        if sections.is_empty() {
            return Err(format!("Unknown transmutation subsection '{subsection}'"));
        }
        let entry = format!("transmutation/{subsection}.arrow");
        if !self.release.has_dir(&entry) {
            let available = self.release.names_under("transmutation");
            let listing = if available.is_empty() {
                "it provides none".to_string()
            } else {
                format!("it provides {}", available.join(", "))
            };
            return Err(format!(
                "{} does not provide a '{subsection}' transmutation subsection: {listing}. \
                 {} Set that part to one of them or to a local folder holding it.",
                self.describe(),
                super::url_cache::publishers_of(subsection)
            ));
        }
        for (file, required) in sections {
            let path = format!("{entry}/{file}");
            match self.release.file(&path) {
                Some(_) if !self.root.join(&path).is_file() => {
                    return Err(format!(
                        "{} does not hold {path}, which its manifest lists: the folder is \
                         incomplete. Download or build it again.",
                        self.describe()
                    ));
                }
                None if *required => {
                    return Err(format!(
                        "{} lists {entry} in its manifest without its required file {file}",
                        self.describe()
                    ));
                }
                _ => {}
            }
        }
        check_sizes(&self.root, &self.release.manifest, &entry)?;
        self.record_use();
        Ok(self.root.join(entry))
    }

    /// The folder as messages name it.
    fn describe(&self) -> String {
        format!(
            "The local release folder {} ('{}' release {})",
            self.root.display(),
            self.release.manifest.keyword,
            self.release.manifest.release
        )
    }

    /// Record that a file was served from this folder.
    fn record_use(&self) {
        let key = self.root.to_string_lossy();
        let mut used = USED.lock().unwrap_or_else(|p| p.into_inner());
        if used.get(key.as_ref()) != Some(&self.record) {
            used.insert(key.into_owned(), self.record.clone());
        }
    }
}
