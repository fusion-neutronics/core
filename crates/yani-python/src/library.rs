//! Whole nuclear-data libraries in a folder: building one from its evaluated
//! sources, and checking one against its manifest.
//!
//! Registered on both wheels: a library folder serves cross sections and the
//! transmutation chain alike, and either wheel reads it. Building is native
//! only: nothing in a browser runs NJOY.

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

/// Build a whole nuclear-data library from its evaluated sources into a folder.
///
/// Downloads the library's source archives (each checked against the size and
/// sha256 it is pinned to), unpacks them, converts every evaluation, builds the
/// transmutation chain, checks the result and writes ``manifest.json`` last. The
/// folder is laid out as a published release, so it serves every setting:
///
/// .. code-block:: python
///
///     generate_library("endf-b8.1", "/data/endf-b8.1")
///     cross_section_data = "/data/endf-b8.1"
///     transmutation_decay_data = "/data/endf-b8.1"
///
/// Running it again resumes: finished folders are kept and only the rest are
/// converted. A folder holds one library built by one version of this package.
///
/// Some hosts serve their archives only to a web browser (the IAEA's FENDL and
/// IRDFF-II files, JAEA's JENDL-5 files). The error then lists each file with
/// its link; download them with a browser and run again. They are found in the
/// Downloads folder, in ``downloads``, or in ``<path>/.build/downloads/``.
///
/// Parameters
/// ----------
/// keyword : str
///     The library: ``"endf-b8.1"``, ``"jeff-4.0"``, ``"jendl-5.0"``,
///     ``"tendl-2025"``, ``"tendl-2017"``, ``"fendl-3.2d"`` or ``"irdff-ii"``.
/// path : str
///     The folder to build in, created if absent. It must be new, empty or an
///     earlier build of the same library, and not inside :func:`cache_dir`.
/// parts : list of str, optional
///     What to build, of ``"neutron"``, ``"photon"``, ``"photonuclear"``,
///     ``"decay"``, ``"reactions"``, ``"fission_yields"`` and ``"branching"``.
///     The default is every part the library's published release carries. Only
///     ``"neutron"`` runs NJOY, so a chain alone builds in minutes.
/// nuclides : list of str, optional
///     Only these nuclides (``"Fe56"``, ``"Am242_m1"``) and elements (``"Fe"``,
///     meaning every Fe evaluation and the Fe photon data). The transmutation
///     chain is always built whole. A name the library does not have is refused.
/// temperatures : list of float, optional
///     Neutron temperatures in Kelvin. The default is 250, 293.6, 600, 900, 1200
///     and 2500 K. The 0 K grid is always kept.
/// njoy_exec : str, optional
///     The NJOY executable. The default is the ``njoy`` installed beside this
///     Python (``pip install njoy2016``), then the one on PATH. FENDL needs the
///     IAEA-NDS fork of NJOY, and a refusal says how to build it.
/// downloads : str, optional
///     A folder of source archives fetched by hand, searched first.
///
/// Returns
/// -------
/// dict
///     ``keyword``, ``release`` (the date the build started, the folder's
///     release stamp), ``path``, ``parts``, ``converted`` and ``reused`` (sorted
///     ``"<particle>/<Name>"`` keys), ``not_converted`` (the evaluations the
///     library's recipe declares NJOY cannot process, with the reason each
///     failed), ``stale_declarations`` (declared ones that converted after all),
///     ``njoy`` (the NJOY version used, or ``None``), ``chain_nuclides``,
///     ``manifest_sha256``, ``file_count`` and ``total_bytes``.
///
/// Raises
/// ------
/// ValueError
///     An unknown keyword or part, or bad temperatures.
/// RuntimeError
///     The build cannot start (the folder, the NJOY, the disk space, or source
///     files that need a browser), or did not finish: evaluations that failed
///     and are not declared, a gate that failed, or Ctrl-C. Nothing finished is
///     lost, and running again retries only what is left. Each failed
///     evaluation's reason is in ``<path>/.build/failed/``.
#[cfg(not(target_os = "emscripten"))]
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (keyword, path, *, parts = None, nuclides = None, temperatures = None, njoy_exec = None, downloads = None))]
#[allow(clippy::too_many_arguments)]
pub fn generate_library(
    py: Python<'_>,
    keyword: &str,
    path: &str,
    parts: Option<Vec<String>>,
    nuclides: Option<Vec<String>>,
    temperatures: Option<Vec<f64>>,
    njoy_exec: Option<String>,
    downloads: Option<String>,
) -> PyResult<Py<PyDict>> {
    use nuclear_data_build::{Part, Request};
    use pyo3::exceptions::PyValueError;
    use std::sync::atomic::{AtomicBool, Ordering};

    nuclear_data_build::recipe::recipe(keyword).map_err(PyValueError::new_err)?;
    let parts: Option<Vec<Part>> = parts
        .map(|names| names.iter().map(|n| Part::parse(n)).collect())
        .transpose()
        .map_err(PyValueError::new_err)?;
    if let Some(t) = temperatures.as_deref()
        && (t.is_empty() || t.iter().any(|t| !t.is_finite() || *t <= 0.0))
    {
        return Err(PyValueError::new_err(
            "temperatures must be a non-empty list of temperatures in Kelvin above \
             0; the 0 K grid is always kept",
        ));
    }
    let njoy_exec = match njoy_exec {
        Some(n) => n,
        None => default_njoy(py)?,
    };
    let builder = format!("yani-python {}", env!("CARGO_PKG_VERSION"));
    let stop = AtomicBool::new(false);
    let path_buf = std::path::PathBuf::from(path);
    let downloads = downloads.map(std::path::PathBuf::from);
    let progress = |line: &str| eprintln!("{line}");

    // The build runs on its own thread with the GIL released, and this
    // thread checks for Ctrl-C while it runs: on one, nothing more is
    // started and the build returns once what is running finishes.
    let result = py.detach(|| {
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                nuclear_data_build::generate(&Request {
                    keyword,
                    path: &path_buf,
                    parts: parts.as_deref(),
                    nuclides: nuclides.as_deref(),
                    temperatures: temperatures.as_deref(),
                    njoy_exec: &njoy_exec,
                    downloads: downloads.as_deref(),
                    builder: &builder,
                    stop: Some(&stop),
                    progress: Some(&progress),
                })
            });
            let mut interrupted = None;
            while !worker.is_finished() {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if interrupted.is_none()
                    && let Err(e) = Python::attach(|py| py.check_signals())
                {
                    stop.store(true, Ordering::Relaxed);
                    eprintln!("interrupted: finishing the conversions that are running");
                    interrupted = Some(e);
                }
            }
            let built = worker
                .join()
                .unwrap_or_else(|_| Err("the library builder panicked".to_string()));
            (built, interrupted)
        })
    });
    let report = match result {
        (_, Some(interrupt)) => return Err(interrupt),
        (Ok(report), None) => report,
        (Err(e), None) => return Err(PyRuntimeError::new_err(e)),
    };

    let dict = PyDict::new(py);
    dict.set_item("keyword", &report.keyword)?;
    dict.set_item("release", &report.release)?;
    dict.set_item("path", report.path.to_string_lossy())?;
    dict.set_item(
        "parts",
        report.parts.iter().map(|p| p.name()).collect::<Vec<_>>(),
    )?;
    dict.set_item("converted", report.converted.iter().collect::<Vec<_>>())?;
    dict.set_item("reused", report.reused.iter().collect::<Vec<_>>())?;
    dict.set_item("not_converted", &report.not_converted)?;
    dict.set_item("stale_declarations", &report.stale_declarations)?;
    dict.set_item("njoy", &report.njoy)?;
    dict.set_item("chain_nuclides", report.chain_nuclides)?;
    dict.set_item("manifest_sha256", &report.manifest_sha256)?;
    dict.set_item("file_count", report.file_count)?;
    dict.set_item("total_bytes", report.total_bytes)?;
    Ok(dict.unbind())
}

/// The `njoy` the njoy2016 wheel installs beside this interpreter, or `njoy`
/// from PATH.
#[cfg(not(target_os = "emscripten"))]
fn default_njoy(py: Python<'_>) -> PyResult<String> {
    let sysconfig = py.import("sysconfig")?;
    let scripts: String = sysconfig
        .call_method1("get_path", ("scripts",))?
        .extract()?;
    let name = if cfg!(windows) { "njoy.exe" } else { "njoy" };
    let beside = std::path::Path::new(&scripts).join(name);
    Ok(if beside.is_file() {
        beside.to_string_lossy().into_owned()
    } else {
        "njoy".to_string()
    })
}
