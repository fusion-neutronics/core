//! `read_nuclide_from_arrow` / `read_element_from_arrow`: the readback gate.
//!
//! A conversion is checked by reading it back, and until now the only Python
//! reader was a second implementation of the format in
//! `packages/nuclear_data_to_arrow`. That is a weaker gate than it looks: a
//! directory can satisfy every schema and still be refused by the loader that
//! actually consumes it, and writer and reader sharing a vocabulary is how #379
//! happened (the transmutation writer spelling MT 18 "fission" while the Rust
//! consumer's map held only "(n,fission)", invisible to every Python test).
//! These bind the real loaders, so the gate is the consumer (issues #443,
//! #525).
//!
//! A summary rather than the whole nuclide. The gate asks "does yamc accept
//! this, and does it hold what it should", and materialising megabytes of
//! secondary distributions into Python answers neither question.
//!
//! Registered on both wheels. Reading a directory back is not transport, and
//! `yani` writes chains that want the same check.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use yamc_nuclide::load_scope::{LoadScope, SectionScope};

/// Read a nuclide Arrow directory through yamc's own loader and describe it.
///
/// Parameters
/// ----------
/// path : str
///     A ``{Nuclide}.arrow/`` section directory. Taken verbatim: no keyword
///     expansion, no directory search and no download, so a validation run
///     cannot silently check a different file from the one it named.
/// scope : str, optional
///     ``"full"`` (default) reads every section, which is what transport
///     needs. ``"xs"`` reads ``nuclide.arrow`` and ``reactions.arrow`` only,
///     which is what a transport-free reaction-rate collapse touches.
///
/// Returns
/// -------
/// dict
///     ``name``, ``atomic_number``, ``mass_number``, ``atomic_weight_ratio``,
///     ``fissionable``, ``urr_present``, ``available_temperatures``,
///     ``loaded_temperatures``, ``mts``, ``energy_points`` and
///     ``scope_loaded``.
///
///     ``energy_points`` is a dict of temperature to grid length, over the
///     loaded temperatures. Not one number: the reader also keeps the 0 K union
///     grid, which is longer than any of them and belongs to no temperature the
///     other two lists mention, so a single maximum would report a grid nothing
///     reads and hide a per-temperature grid that had been truncated.
///
///     There is no ``library`` key. The Arrow loader does not populate that
///     field, so it would report ``None`` for every directory, correct or not.
///     Read ``version.json`` for it.
///
///     ``scope_loaded`` is the one to assert on, and it is not always the
///     ``scope`` asked for: a directory holding no transport sections narrows a
///     ``"full"`` request to ``"xs"`` rather than failing it, on the theory
///     that it is a cross-sections-only conversion. A gate that asks for
///     ``"full"`` and does not check what it got will pass a directory with no
///     distributions, no products and no ``fast_xs``.
///
/// Raises
/// ------
/// ValueError
///     ``scope`` is neither ``"full"`` nor ``"xs"``.
/// RuntimeError
///     The directory is absent, is not the Arrow format version this build
///     reads, or any section fails to parse.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (path, scope = "full"))]
pub fn read_nuclide_from_arrow(py: Python, path: &str, scope: &str) -> PyResult<Py<PyAny>> {
    let requested = match scope {
        "full" => LoadScope::full(),
        "xs" => LoadScope {
            sections: SectionScope::XsOnly,
            mts: None,
            temperatures: None,
            covariance: false,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown scope {other:?}; expected \"full\" or \"xs\""
            )))
        }
    };

    // The load is the expensive part (a full-scope nuclide is hundreds of MB of
    // sections) and touches no Python, so the GIL goes back while it runs. A
    // gate reading a published tree through a thread pool otherwise gets no
    // parallelism at all, and blocks every other thread in the process for the
    // duration of each read.
    // `Box<dyn Error>` is not `Send`, so the message crosses the boundary
    // rather than the error.
    let nuclide = py
        .detach(|| {
            yamc_nuclide::nuclide_arrow::read_nuclide_from_arrow(
                std::path::Path::new(path),
                &requested,
            )
            .map_err(|e| e.to_string())
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{path}: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("name", nuclide.name.clone())?;
    dict.set_item("atomic_number", nuclide.atomic_number)?;
    dict.set_item("mass_number", nuclide.mass_number)?;
    dict.set_item("atomic_weight_ratio", nuclide.atomic_weight_ratio)?;
    dict.set_item("fissionable", nuclide.fissionable)?;
    dict.set_item("urr_present", nuclide.urr_present)?;
    dict.set_item(
        "available_temperatures",
        nuclide.available_temperatures.clone(),
    )?;
    dict.set_item("loaded_temperatures", nuclide.loaded_temperatures.clone())?;
    dict.set_item("mts", nuclide.reaction_mts())?;
    // Per temperature, and only the loaded ones. `energy` additionally holds
    // the 0 K union grid, which is the longest and is listed in neither
    // temperature list, so a maximum over the map reports a grid no load reads.
    let energy_points = PyDict::new(py);
    if let Some(grids) = nuclide.energy.as_ref() {
        for temperature in &nuclide.loaded_temperatures {
            if let Some(grid) = grids.get(temperature) {
                energy_points.set_item(temperature, grid.len())?;
            }
        }
    }
    dict.set_item("energy_points", energy_points)?;
    dict.set_item(
        "scope_loaded",
        match nuclide.load_scope.sections {
            SectionScope::Full => "full",
            SectionScope::XsOnly => "xs",
        },
    )?;
    Ok(dict.into())
}

/// Read a photoatomic Arrow directory through yamc's own loader and describe it.
///
/// Parameters
/// ----------
/// path : str
///     An ``{Element}.arrow/`` directory. Taken verbatim, and read straight
///     rather than through the process-global element store: that store is
///     keyed by element name, so validating two builds of ``Fe.arrow`` in one
///     process would get the first one back for the second call and report a
///     false pass.
///
///     Publishes none of the process-wide photon grids either. An ordinary
///     read installs the Compton momentum grid and the two TTB grids on first
///     use, so inspecting a candidate directory in an interpreter that also
///     runs transport would hand that candidate's grids to every later element
///     load and every photon collision. A different-length grid then panics
///     out of the sampler; a same-length one with different values changes
///     every Doppler sample and says nothing.
///
/// Returns
/// -------
/// dict
///     ``name``, ``atomic_number``, ``n_energy_points``, ``n_subshells``,
///     ``has_atomic_relaxation``, ``has_compton_profiles`` and
///     ``has_bremsstrahlung``.
///
///     The last three are the auxiliary tabulations that are not in any
///     evaluation, so a directory converted without them reads back fine and
///     is still not what transport wants. There is no ``scope`` here: the
///     photon reader has one mode.
///
/// Raises
/// ------
/// RuntimeError
///     The directory is absent, or any section fails to parse.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn read_element_from_arrow(py: Python, path: &str) -> PyResult<Py<PyAny>> {
    let element = py
        .detach(|| {
            yamc_element::photon_arrow::inspect_photon_interaction_from_arrow(std::path::Path::new(
                path,
            ))
            .map_err(|e| e.to_string())
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{path}: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("name", element.name.clone())?;
    dict.set_item("atomic_number", element.atomic_number)?;
    dict.set_item("n_energy_points", element.energy.len())?;
    dict.set_item("n_subshells", element.shells.len())?;
    dict.set_item("has_atomic_relaxation", element.has_atomic_relaxation)?;
    dict.set_item("has_compton_profiles", !element.profile_pdf.is_empty())?;
    dict.set_item("has_bremsstrahlung", !element.dcs.is_empty())?;
    Ok(dict.into())
}
