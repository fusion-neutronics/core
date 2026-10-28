//! Whole nuclear-data libraries in a folder: checking one against its
//! manifest.
//!
//! Registered on both wheels: a library folder serves cross sections and the
//! transmutation chain alike, and either wheel reads it.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// Check a nuclear-data library folder against its ``manifest.json``.
///
/// Re-hashes every file the manifest lists and compares its size and sha256
/// with the manifest, then looks for files the manifest does not list.
/// Loading data from a library folder only checks the sizes of the files it
/// reads; this is the full check, for a library kept for offline use or built
/// locally.
///
/// Parameters
/// ----------
/// path : str
///     A library folder: ``manifest.json`` beside the ``neutron/``,
///     ``photon/``, ``photonuclear/`` and ``transmutation/`` folders it
///     describes, laid out as each release in :func:`cache_dir` is. A release
///     folder in the cache holds only the files runs have downloaded, so the
///     rest are reported missing.
///
/// Returns
/// -------
/// dict
///     ``keyword``, ``release``, ``format_version`` and ``manifest_sha256``
///     identify the release the manifest describes, and ``file_count`` and
///     ``total_bytes`` count what it lists. ``missing`` (listed but absent),
///     ``changed`` (listed, with another size or sha256) and ``unlisted``
///     (in the release's ``<Name>.arrow/`` folders, or a particle's
///     ``index.json``, but not listed) are sorted paths relative to the
///     folder. ``ok`` is ``True`` when all three are empty.
///
///     An unlisted file fails the check: results record the manifest's sha256
///     as the identity of the library a run read, and a file the manifest
///     does not list is data that identity does not cover. Files outside the
///     release's folders, such as a local build's ``.build/`` folder, are not
///     looked at.
///
/// Raises
/// ------
/// RuntimeError
///     If ``path`` is not a folder or holds no ``manifest.json``, if the
///     manifest is not a version 1 release manifest or is in a
///     ``format_version`` this build does not read, or if a listed file
///     cannot be read.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn verify_library(py: Python<'_>, path: &str) -> PyResult<Py<PyDict>> {
    let found = py
        .detach(|| yamc_nuclide::storage::release::verify_folder(std::path::Path::new(path)))
        .map_err(PyRuntimeError::new_err)?;
    let manifest = &found.release.manifest;
    let dict = PyDict::new(py);
    dict.set_item("keyword", &manifest.keyword)?;
    dict.set_item("release", &manifest.release)?;
    dict.set_item("format_version", manifest.format_version)?;
    dict.set_item("manifest_sha256", &found.release.manifest_sha256)?;
    dict.set_item("file_count", manifest.files.len())?;
    dict.set_item(
        "total_bytes",
        manifest.files.iter().map(|f| f.bytes).sum::<u64>(),
    )?;
    dict.set_item("missing", &found.missing)?;
    dict.set_item("changed", &found.changed)?;
    dict.set_item("unlisted", &found.unlisted)?;
    dict.set_item("ok", found.ok())?;
    Ok(dict.unbind())
}
