//! Transmutation-chain resolution and radionuclide discovery for decay-photon
//! shutdown-dose-rate post-processing. The time-correction maths is driven by
//! `PulseSchedule` (see `schedule.rs`), which calls the `yamc-physics` core
//! directly. `get_radionuclides_from_chain` backs `Model.radionuclides()`.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use yani::LoadedChain;

/// Load the transmutation network assembled from the configured per-subsection
/// sources (`yamc.transmutation_decay_data` / `_reactions` /
/// `_fission_yields`). Each source (a library keyword or a local path) is
/// resolved independently -- keywords download their subsection tarball on
/// first use and cache it -- then the parts are merged. When
/// `yamc.transmutation_branch_ratios` is set, the isomeric-branching overlay is
/// resolved too (grafted `(n,n')` channels + verbatim curves scored at the
/// collision energy on the coupled path, or folded against the user spectrum
/// on the flux-given path).
pub fn resolve_chain() -> PyResult<LoadedChain> {
    let loaded = yani_transmute::load_configured_chain().map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to load transmutation chain: {e}"))
    })?;
    if yamc_nuclide::load_logging_enabled() {
        println!("Assembled transmutation chain from configured subsections");
    }
    Ok(loaded)
}

/// Find unstable activation products from a list of nuclide names.
///
/// Args:
///     nuclide_names (list[str]): Nuclide names present in the model materials.
///
/// The transmutation network is assembled from the configured per-subsection
/// sources (``yamc.transmutation_decay_data`` etc.).
///
/// Returns:
///     list[str]: Sorted list of unique radionuclide names.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (nuclide_names))]
pub fn get_radionuclides_from_chain(nuclide_names: Vec<String>) -> PyResult<Vec<String>> {
    let loaded = resolve_chain()?;
    Ok(yani_decay::get_radionuclides_from_chain(
        &nuclide_names,
        &loaded.chain,
    ))
}
