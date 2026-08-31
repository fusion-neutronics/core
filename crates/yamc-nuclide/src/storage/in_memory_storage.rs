//! [`Storage`] backend that holds file contents entirely in RAM.
//!
//! Intended for browser builds where there is no `std::fs`-style
//! filesystem. A JS host (or anyone, really) populates the backend by
//! calling [`InMemoryStorage::add_file`] with the bytes that *would*
//! have lived on disk under the matching path. Reads then go through
//! [`open_read`] / [`read_to_string`] / [`exists`] exactly like the
//! native backend, so the rest of the nuclide loader is none the wiser.
//!
//! ## Path convention
//!
//! Paths are matched as-is -- there's no scheme parsing or
//! canonicalisation. The Arrow nuclide loader requests paths like
//! `/Li6.arrow/nuclide.arrow`, `/Li6.arrow/version.json`, etc. (when
//! built with `cfg(target_arch = "wasm32")`, since the keyword resolver
//! falls through to "treat as literal path"). A JS host that has
//! fetched a per-nuclide `.arrow/` section set makes one [`add_file`]
//! call per section object.
//!
//! Container paths (e.g. `/Li6.arrow/`) report as existing if at
//! least one file lives under them.
//!
//! ## Single-instance assumption
//!
//! `yamc_nuclide::storage::set_storage` swaps a process-global
//! backend, so one `InMemoryStorage` is active per process at a time.
//! Browser hosts that want multiple independent simulations should
//! reuse the same backend and namespace by path prefix.

use std::collections::HashMap;
use std::io;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::storage::{ReadSeek, Storage};

/// In-RAM virtual file store. Cheap to clone (just bumps an
/// `Arc<RwLock<…>>` refcount) -- the canonical idiom is to keep one
/// clone for inserting files and hand a second to
/// [`yamc_nuclide::storage::set_storage`]:
///
/// ```ignore
/// let storage = InMemoryStorage::new();
/// yamc_nuclide::storage::set_storage(Box::new(storage.clone()));
/// // `storage` still owns a handle to the same backing store
/// storage.add_file("/Li6.arrow/nuclide.arrow", bytes);
/// ```
#[derive(Clone, Default)]
pub struct InMemoryStorage {
    files: Arc<RwLock<HashMap<PathBuf, Arc<[u8]>>>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) the file at `path` with `bytes`.
    pub fn add_file<P: Into<PathBuf>>(&self, path: P, bytes: Vec<u8>) {
        let mut files = self.files.write().unwrap_or_else(|p| p.into_inner());
        files.insert(path.into(), Arc::from(bytes.into_boxed_slice()));
    }

    /// Remove every file currently held. Useful for tests and for hosts
    /// that want to swap the in-memory data set between runs.
    pub fn clear(&self) {
        self.files
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }

    /// Number of files currently stored. Cheap; intended for diagnostics.
    pub fn len(&self) -> usize {
        self.files.read().unwrap_or_else(|p| p.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Storage for InMemoryStorage {
    fn open_read(&self, path: &Path) -> io::Result<Box<dyn ReadSeek + Send>> {
        let files = self.files.read().unwrap_or_else(|p| p.into_inner());
        let bytes = files.get(path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("InMemoryStorage: no file at {}", path.display()),
            )
        })?;
        // `Cursor<Arc<[u8]>>` is `Read + Seek` (any `AsRef<[u8]>` is), so the
        // reader streams straight out of the stored bytes. Copying them into a
        // `Vec` first meant the browser path held a second full copy of every
        // fetched section for the duration of its parse (issue #476).
        Ok(Box::new(Cursor::new(Arc::clone(bytes))))
    }

    fn open_write(&self, _path: &Path) -> io::Result<Box<dyn io::Write + Send>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "InMemoryStorage is read-only -- use add_file() to populate",
        ))
    }

    fn read_to_string(&self, path: &Path) -> io::Result<String> {
        let files = self.files.read().unwrap_or_else(|p| p.into_inner());
        let bytes = files.get(path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("InMemoryStorage: no file at {}", path.display()),
            )
        })?;
        String::from_utf8((**bytes).to_vec())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    fn exists(&self, path: &Path) -> bool {
        let files = self.files.read().unwrap_or_else(|p| p.into_inner());
        // Exact-file match …
        if files.contains_key(path) {
            return true;
        }
        // … or treat `path` as a directory if anything sits under it.
        // The nuclide loader probes `dir.join("version.json").exists()`
        // and similar; we want a `/Li6.arrow/` query to succeed once
        // any sub-file has been added.
        files.keys().any(|p| p.starts_with(path))
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn add_and_read_round_trip() {
        let s = InMemoryStorage::new();
        s.add_file("/foo/bar.txt", b"hello".to_vec());
        assert_eq!(
            s.read_to_string(Path::new("/foo/bar.txt")).unwrap(),
            "hello"
        );
    }

    #[test]
    fn open_read_streams_bytes() {
        let s = InMemoryStorage::new();
        s.add_file("/x.bin", vec![1, 2, 3, 4]);
        let mut r = s.open_read(Path::new("/x.bin")).unwrap();
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, vec![1, 2, 3, 4]);
    }

    #[test]
    fn missing_file_returns_not_found() {
        let s = InMemoryStorage::new();
        // `Box<dyn ReadSeek + Send>` isn't `Debug` so `unwrap_err` won't compile;
        // match the Err arm explicitly instead.
        match s.open_read(Path::new("/nope")) {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::NotFound),
            Ok(_) => panic!("expected NotFound error"),
        }
    }

    #[test]
    fn exists_matches_file_and_directory_prefix() {
        let s = InMemoryStorage::new();
        s.add_file("/Li6.arrow/nuclide.arrow", vec![]);
        s.add_file("/Li6.arrow/version.json", b"{}".to_vec());
        // exact file
        assert!(s.exists(Path::new("/Li6.arrow/nuclide.arrow")));
        // directory containing files
        assert!(s.exists(Path::new("/Li6.arrow")));
        assert!(s.exists(Path::new("/Li6.arrow/")));
        // nothing under this prefix
        assert!(!s.exists(Path::new("/Fe56.arrow")));
    }

    #[test]
    fn write_is_unsupported() {
        let s = InMemoryStorage::new();
        match s.open_write(Path::new("/anywhere")) {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::Unsupported),
            Ok(_) => panic!("expected Unsupported error"),
        }
    }
}
