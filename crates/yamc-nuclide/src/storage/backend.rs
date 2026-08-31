//! Pluggable storage backend for nuclide-data reads.
//!
//! Default implementation wraps `std::fs` for native builds. Browser builds
//! can swap in an alternative (OPFS, IndexedDB, fetched-byte buffers) by
//! calling [`set_storage`] before any nuclide load. The trait surface is
//! deliberately small -- only what `nuclide_arrow` and `arrow_helpers`
//! actually need to do their work.
//!
//! ## Why a trait
//!
//! `std::fs` compiles for `wasm32-unknown-unknown` but every call returns
//! an error at runtime, so a browser build that goes through the native
//! `std::fs` path would only ever produce "not found" errors. The trait
//! lets a browser host wire OPFS-backed reads without touching the parsers.

use std::io;
use std::io::{Read, Seek, Write};
use std::path::Path;
use std::sync::RwLock;

use once_cell::sync::Lazy;

/// Compound bound used by Arrow IPC's `FileReader::try_new` -- needs `Read + Seek`.
pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

/// Backend abstraction for nuclide-data file I/O.
pub trait Storage: Send + Sync {
    /// Open a file for streaming reads. Used for Arrow IPC files where we
    /// want `Read + Seek` semantics (avoids buffering the whole nuclide
    /// into memory before parsing).
    fn open_read(&self, path: &Path) -> io::Result<Box<dyn ReadSeek + Send>>;

    /// Open a file for sequential writes.
    fn open_write(&self, path: &Path) -> io::Result<Box<dyn Write + Send>>;

    /// Read a small text file into a `String`. Used for `version.json`-style
    /// sidecar metadata where the whole file is needed at once.
    fn read_to_string(&self, path: &Path) -> io::Result<String>;

    /// Whether the path resolves to an existing file or directory.
    fn exists(&self, path: &Path) -> bool;
}

/// Storage backend that delegates to `std::fs`. The default on native targets.
#[cfg(not(target_arch = "wasm32"))]
pub struct NativeStorage;

#[cfg(not(target_arch = "wasm32"))]
impl Storage for NativeStorage {
    fn open_read(&self, path: &Path) -> io::Result<Box<dyn ReadSeek + Send>> {
        Ok(Box::new(std::fs::File::open(path)?))
    }
    fn open_write(&self, path: &Path) -> io::Result<Box<dyn Write + Send>> {
        Ok(Box::new(std::fs::File::create(path)?))
    }
    fn read_to_string(&self, path: &Path) -> io::Result<String> {
        std::fs::read_to_string(path)
    }
    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }
}

/// wasm32 default -- every call fails. A browser host is expected to call
/// [`set_storage`] with an OPFS-backed implementation at startup.
#[cfg(target_arch = "wasm32")]
pub struct UnconfiguredStorage;

#[cfg(target_arch = "wasm32")]
impl Storage for UnconfiguredStorage {
    fn open_read(&self, _: &Path) -> io::Result<Box<dyn ReadSeek + Send>> {
        Err(io::Error::other(
            "no storage backend configured -- call yamc_nuclide::storage::set_storage(..) before loading nuclides",
        ))
    }
    fn open_write(&self, _: &Path) -> io::Result<Box<dyn Write + Send>> {
        Err(io::Error::other("no storage backend configured for wasm32"))
    }
    fn read_to_string(&self, _: &Path) -> io::Result<String> {
        Err(io::Error::other("no storage backend configured for wasm32"))
    }
    fn exists(&self, _: &Path) -> bool {
        false
    }
}

static STORAGE: Lazy<RwLock<Box<dyn Storage>>> = Lazy::new(|| {
    #[cfg(not(target_arch = "wasm32"))]
    {
        RwLock::new(Box::new(NativeStorage))
    }
    #[cfg(target_arch = "wasm32")]
    {
        RwLock::new(Box::new(UnconfiguredStorage))
    }
});

/// Install a new storage backend. Subsequent reads/writes go through it.
/// Intended for browser hosts to wire in an OPFS-backed impl at startup,
/// and for tests that want to assert against an in-memory fake.
pub fn set_storage(backend: Box<dyn Storage>) {
    *STORAGE.write().unwrap_or_else(|p| p.into_inner()) = backend;
}

/// Read `path` into bytes via the configured backend. Convenience helper for
/// callers that need a slice rather than a `Read + Seek` cursor -- uses
/// `open_read` internally so the result reflects whichever backend is active.
pub fn read_bytes(path: &Path) -> io::Result<Vec<u8>> {
    let mut reader = open_read(path)?;
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Open `path` for streaming reads via the configured backend.
pub fn open_read(path: &Path) -> io::Result<Box<dyn ReadSeek + Send>> {
    STORAGE
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .open_read(path)
}

/// Open `path` for sequential writes via the configured backend.
pub fn open_write(path: &Path) -> io::Result<Box<dyn Write + Send>> {
    STORAGE
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .open_write(path)
}

/// Read `path` as UTF-8 text via the configured backend.
pub fn read_to_string(path: &Path) -> io::Result<String> {
    STORAGE
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .read_to_string(path)
}

/// Whether `path` exists via the configured backend.
pub fn exists(path: &Path) -> bool {
    STORAGE
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .exists(path)
}
