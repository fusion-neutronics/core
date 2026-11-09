//! What `Model.simulate_transmutation` returns: the inventories, plus the
//! model's own tallies from each step's transport.

use pyo3::exceptions::{PyIndexError, PyKeyError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yani_python::PyTransmutationResults;

use crate::simulation::{provenance_version, PySimulationResults};
use crate::tally::PyTally;

/// The results of ``Model.simulate_transmutation``: everything
/// ``TransmutationResults`` holds, plus the model's own tallies
/// (``Model(tallies=[...])``) from each step's transport, so anything that
/// changes as the materials burn (TBR as Li6 depletes, heating, flux,
/// spectra, mesh tallies) is read off the one run.
///
/// Every step holds the whole ``SimulationResults`` of that step's
/// transport, so a large mesh tally costs its size once per transport step
/// (once in all for ``method="independent"``).
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "ModelTransmutationResults", extends = PyTransmutationResults)]
pub struct PyModelTransmutationResults {
    pub tallies: yamc::StepTallies,
}

impl PyModelTransmutationResults {
    /// Wrap a finished run, stamping each step's provenance with the
    /// installed version as ``simulate_transport`` does.
    pub fn new(
        py: Python<'_>,
        results: yani_transmute::TransmutationResults,
        mut tallies: yamc::StepTallies,
    ) -> PyResult<Py<Self>> {
        let version = provenance_version();
        for run in tallies.runs.iter_mut().flat_map(|r| r.runs.iter_mut()) {
            run.yamc_version.clone_from(&version);
        }
        let base = PyTransmutationResults::from_run(py, results)?;
        Py::new(py, (Self { tallies }, base))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelTransmutationResults {
    /// The model's tallies from step ``step``'s transport, or ``None`` for a
    /// decay-only step (a ``Cooldown``, or a ``Pulse`` with rate 0), where no
    /// transport runs, and for a model with no tallies.
    ///
    /// ``step`` indexes ``timesteps``, as ``get_source_rates`` and
    /// ``get_reaction_rates`` do: step ``i`` is the transport that drove the
    /// step from composition ``i`` (``get_material(id, i)``) to composition
    /// ``i + 1``. Each step's tallies are reset before its transport, so they
    /// reflect that step's compositions only.
    ///
    /// Values are per source particle, as ``simulate_transport`` returns them;
    /// multiply by ``get_source_rates(material_id)[step]`` for absolute rates.
    /// The transmutation's internal tallies and its scouting transport are
    /// not among them.
    ///
    /// With ``method="independent"`` transport runs once, on the initial
    /// compositions, and every irradiation step returns that one run's
    /// results: they do not follow the burnup.
    ///
    /// Args:
    ///     step: The schedule step, ``0 <= step < len(timesteps)``.
    ///
    /// Examples:
    ///     >>> for step, t in enumerate(results.timesteps):
    ///     ...     step_results = results.tally_results(step)
    ///     ...     if step_results is not None:
    ///     ...         print(t, step_results[tbr].mean)
    fn tally_results(&self, step: usize) -> PyResult<Option<PySimulationResults>> {
        let found = self.tallies.get(step).map_err(PyIndexError::new_err)?;
        Ok(found.cloned().map(PySimulationResults::from))
    }

    /// ``tally``'s per-bin means and standard deviations over the steps that
    /// ran transport, in step order: ``(means, standard_deviations)``, each a
    /// list with one flat list of bins per transport step. TBR against
    /// irradiation time is one call. See ``tally_results`` for units and
    /// what each step holds.
    ///
    /// Args:
    ///     tally: One of the model's tallies.
    ///
    /// Examples:
    ///     >>> means, std_devs = results.tally_evolution(tbr)
    fn tally_evolution(&self, tally: PyTally) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        self.tallies
            .evolution(&tally.inner)
            .map_err(PyKeyError::new_err)
    }
}
