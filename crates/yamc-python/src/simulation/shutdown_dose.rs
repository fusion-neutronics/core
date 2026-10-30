//! `yamc.simulate_shutdown_dose`: the shutdown dose rate over a schedule whose
//! steps put the machine in different models. A thin wrapper: grouping the
//! Python objects into models and stages here, the method in
//! `yamc::shutdown_dose`.

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use yamc::shutdown_dose::{Placement, Stage, Step};
use yani_python::distribution::{PyDoseResult, PyPulseSchedule};

use crate::simulation::PyModel;
use crate::tally::PyTally;

/// Simulate the shutdown dose rate of a machine that changes along its
/// irradiation and cooling schedule.
///
/// Each step of ``schedule`` names the ``Model`` the machine is in
/// (``Pulse(model=...)``, ``Cooldown(model=...)``; a step without one stays in
/// the previous step's). A model can add cells, leave cells out (removed
/// components no longer emit, though what they emitted before still decays
/// with them should they come back) and hold cells somewhere else (say where
/// with the step's ``moves``). Cells are the same component across models when
/// they have the same name, so name every cell that activates.
///
/// For every model a pulse irradiates in, the neutrons are transported once
/// and each decay photon (D1S) is recorded where it is born. For every pair of
/// irradiation and observation configurations those photons are then moved
/// with their cells and transported through the observation step's model,
/// with ``tallies`` binned by parent radionuclide. The per-radionuclide time
/// correction of each irradiation's own pulses then gives the dose at the end
/// of every step. No step of the schedule needs a transport run of its own.
///
/// Runs on the CPU, on one process. ``std_dev`` is the photon transport's
/// statistical error, given the recorded decay photons; the neutron
/// transport's is not included.
///
/// Args:
///     schedule: A ``PulseSchedule`` whose steps name models.
///     tallies: Photon tallies to score the dose in, without
///         ``parent_nuclides`` (they are binned by every radionuclide the
///         irradiations make). Each needs a unique ``name``, and must be valid
///         in every model the schedule observes (a cell filter's cells must
///         exist in each).
///     total_particles: Histories per transport run: each irradiated model's
///         neutron run and each photon run.
///     seed: Base random seed.
///     threads: Worker threads; ``None`` uses all cores.
///
/// Returns:
///     ``{tally name: DoseResult}``, each with one row per schedule step:
///     ``.mean`` / ``.std_dev`` / ``.by_nuclide`` / ``.times``.
///
/// Examples:
///     >>> schedule = yamc.PulseSchedule([
///     ...     yamc.Pulse(model=dt_reactor, rate=1e18, duration=(1, "a")),
///     ...     yamc.Cooldown(model=opened, duration=(1, "d"),
///     ...                   moves={"blanket": (0, 0, 300)}),
///     ... ])
///     >>> dose = yamc.simulate_shutdown_dose(schedule, tallies=[dose_tally],
///     ...                                    total_particles=100_000)
///     >>> dose["dose"].mean[-1]
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (schedule, tallies, total_particles, seed=1, threads=None))]
pub fn simulate_shutdown_dose(
    py: Python<'_>,
    schedule: &PyPulseSchedule,
    tallies: Vec<PyRef<'_, PyTally>>,
    total_particles: usize,
    seed: u64,
    threads: Option<usize>,
) -> PyResult<BTreeMap<String, PyDoseResult>> {
    if total_particles == 0 {
        return Err(PyValueError::new_err("total_particles must be positive"));
    }
    let mut names = Vec::with_capacity(tallies.len());
    for t in &tallies {
        let name = t.inner.name.clone().ok_or_else(|| {
            PyValueError::new_err("give each tally a name: the results are keyed by it")
        })?;
        if names.contains(&name) {
            return Err(PyValueError::new_err(format!(
                "two tallies are named '{name}'"
            )));
        }
        names.push(name);
    }
    let tallies: Vec<Arc<yamc_tallies::Tally>> = tallies.iter().map(|t| t.inner.clone()).collect();

    // Distinct models by identity, distinct (model, moves) as stages.
    let mut model_objs: Vec<Py<PyAny>> = Vec::new();
    let mut models: Vec<yamc::model::Model> = Vec::new();
    let mut stages: Vec<Stage> = Vec::new();
    let mut steps: Vec<Step> = Vec::new();
    for s in schedule.staged_steps(py)? {
        let obj = s.model.bind(py);
        let model = match model_objs.iter().position(|m| m.bind(py).is(obj)) {
            Some(i) => i,
            None => {
                let m = obj.cast::<PyModel>().map_err(|_| {
                    PyTypeError::new_err("a schedule step's model must be a yamc.Model")
                })?;
                models.push(m.borrow().inner.clone());
                model_objs.push(s.model.clone_ref(py));
                models.len() - 1
            }
        };
        let placements = s
            .moves
            .iter()
            .map(|(name, m)| {
                Placement::new(*m)
                    .map(|p| (name.clone(), p))
                    .map_err(|e| PyValueError::new_err(format!("moves['{name}']: {e}")))
            })
            .collect::<PyResult<BTreeMap<_, _>>>()?;
        let stage = Stage { model, placements };
        let stage = match stages.iter().position(|st| *st == stage) {
            Some(i) => i,
            None => {
                stages.push(stage);
                stages.len() - 1
            }
        };
        steps.push(Step {
            rate: s.rate,
            duration: s.duration,
            stage,
        });
    }

    let settings = yamc::model::TransportSettings {
        total_particles: Some(total_particles),
        seed,
        threads,
        max_runtime: None,
        data_uncertainty: None,
    };
    let dose = py
        .detach(|| {
            yamc::shutdown_dose::simulate_shutdown_dose(
                &models, &stages, &steps, &tallies, &settings,
            )
        })
        .map_err(PyValueError::new_err)?;
    names
        .into_iter()
        .zip(dose.tallies)
        .map(|(name, d)| {
            PyDoseResult::from_rows(py, d.mean, d.std_dev, d.by_nuclide, dose.times.clone())
                .map(|r| (name, r))
        })
        .collect()
}
