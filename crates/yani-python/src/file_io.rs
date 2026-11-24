//! The one convention every save and load follows, in both wheels:
//! `to_<format>(path=None)` writes the file at `path`, or returns its bytes
//! when there is no path (a browser, wasm or Pyodide session has no file
//! system worth writing to), and `from_<format>(source)` reads a path (`str`
//! or `os.PathLike`) or those bytes.

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Write `bytes` to `path`, or hand them back when there is none.
pub fn write_or_return(
    py: Python<'_>,
    path: Option<std::path::PathBuf>,
    bytes: &[u8],
) -> PyResult<Option<Py<PyBytes>>> {
    match path {
        Some(path) => {
            std::fs::write(&path, bytes).map_err(|e| {
                PyOSError::new_err(format!("failed to write {}: {e}", path.display()))
            })?;
            Ok(None)
        }
        None => Ok(Some(PyBytes::new(py, bytes).unbind())),
    }
}

/// The contents `source` names: the bytes themselves, or the file at a path.
pub fn read_source(source: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = source.cast::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    let path: std::path::PathBuf = source.extract()?;
    std::fs::read(&path)
        .map_err(|e| PyOSError::new_err(format!("failed to read {}: {e}", path.display())))
}
