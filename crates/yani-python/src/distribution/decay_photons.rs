//! Transmutation-chain resolution and radionuclide discovery for decay-photon
//! shutdown-dose-rate post-processing. The time-correction maths is driven by
//! `PulseSchedule` (see `schedule.rs`), which calls the `yamc-physics` core
//! directly. `get_radionuclides_from_chain` backs `Model.radionuclides()`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use yamc_nuclide::config::{ChainUse, Config};
use yani::LoadedChain;

/// Load the transmutation network a calculation of `chain_use` reads,
/// assembled from the configured per-subsection sources
/// (`yamc.transmutation_decay_data` / `_reactions` / `_fission_yields`).
///
/// There is no default library: a source `chain_use` reads that is unset
/// raises `ValueError` before anything is resolved or downloaded, naming each
/// such setting, as an unset `cross_section_data` does. Each source (a library
/// keyword or a local path) is resolved independently (a keyword downloads its
/// subsection on first use and caches it), then the parts are merged.
/// When `yamc.transmutation_branch_ratios` names branching sources, the
/// isomeric-branching overlay is resolved too, from each in order (grafted
/// `(n,n')` channels + verbatim curves scored at the collision energy on the
/// coupled path, or folded against the user spectrum on the flux-given path).
pub fn resolve_chain(chain_use: ChainUse) -> PyResult<LoadedChain> {
    let sources = Config::global()
        .transmutation_sources(chain_use)
        .map_err(PyValueError::new_err)?;
    let loaded = yani_transmute::load_chain_sources(&sources).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to load transmutation chain: {e}"))
    })?;
    if yamc_nuclide::load_logging_enabled() {
        let sources: Vec<&str> = loaded
            .branch
            .sources()
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        if sources.is_empty() {
            println!("Assembled transmutation chain from configured subsections");
        } else {
            println!(
                "Assembled transmutation chain from configured subsections, with isomeric \
                 branching from {}",
                sources.join(", then ")
            );
        }
    }
    Ok(loaded)
}

/// Find unstable activation products from a list of nuclide names.
///
/// Args:
///     nuclide_names (list[str]): Nuclide names present in the model materials.
///
/// The transmutation network is assembled from the configured per-subsection
/// sources: the decay data (``yamc.transmutation_decay_data``) and the
/// reactions (``yamc.transmutation_reactions``), each a library keyword or a
/// path. There is no default library.
///
/// Returns:
///     list[str]: Sorted list of unique radionuclide names.
///
/// Raises:
///     ValueError: If either source is unset, naming it.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (nuclide_names))]
pub fn get_radionuclides_from_chain(nuclide_names: Vec<String>) -> PyResult<Vec<String>> {
    let loaded = resolve_chain(ChainUse::Activation)?;
    Ok(yani_decay::get_radionuclides_from_chain(
        &nuclide_names,
        &loaded.chain,
    ))
}
