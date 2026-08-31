// Nuclide loader entry point.
//
// Arrow IPC is the only supported nuclear data format; this module is a thin
// wrapper over `nuclide_arrow::read_nuclide_from_arrow` with file-extension
// validation. A trait+enum abstraction here originally supported JSON/HDF5
// alongside Arrow, but those formats were removed.

use crate::load_scope::LoadScope;
use crate::nuclide::Nuclide;
use std::error::Error;
use std::path::Path;

/// Load a nuclide from an Arrow IPC directory.
///
/// The path is expected to be a directory ending in `.arrow/` or one that
/// contains a `version.json` (the Arrow-format convention). `scope` selects
/// which sections, MTs and temperatures to materialize; pass
/// [`LoadScope::full`] for transport.
pub fn load_nuclide<P: AsRef<Path>>(path: P, scope: &LoadScope) -> Result<Nuclide, Box<dyn Error>> {
    let path = path.as_ref();

    if !is_arrow_path(path) {
        return Err(format!(
            "Could not detect Arrow nuclear data at: {}. Expected a directory \
             ending in `.arrow/` or containing `version.json`.",
            path.display()
        )
        .into());
    }

    let nuclide = crate::nuclide_arrow::read_nuclide_from_arrow(path, scope)?;
    if crate::load_logging_enabled() {
        println!(
            "Loaded nuclear data for {}: {}",
            nuclide.name.as_deref().unwrap_or("?"),
            path.display()
        );
    }
    Ok(nuclide)
}

fn is_arrow_path(path: &Path) -> bool {
    if path.is_dir() {
        if path.extension().and_then(|e| e.to_str()) == Some("arrow") {
            return true;
        }
        if path.join("version.json").exists() {
            return true;
        }
    }
    path.extension().and_then(|e| e.to_str()) == Some("arrow")
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_arrow_path_is_rejected() {
        assert!(!is_arrow_path(Path::new("Li6.unknown")));
    }

    #[test]
    fn arrow_extension_is_accepted() {
        assert!(is_arrow_path(Path::new("Li6.arrow")));
    }
}
