//! `yamc.simulate_shutdown_dose`: the shutdown dose rate over a schedule whose
//! steps put the machine in different models. A thin wrapper: grouping the
//! Python objects into models and stages here, the methods in
//! `yamc::shutdown_dose` (D1S) and `yamc::r2s`.

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use yamc::shutdown_dose::{Placement, Stage, Step};
use yani_python::distribution::{PyDoseResult, PyPulseSchedule};

use crate::geometry::{PyRegularCylindricalMesh, PyRegularRectangularMesh};
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
/// and each decay photon (D1S) is recorded where it is born. For every
/// configuration the machine is observed in, each neutron history's photons,
/// from every irradiated model, are then moved with their cells and
/// transported through that model as one photon history, and that history's
/// dose at the end of every step it observes is formed from the
/// per-radionuclide time correction factors of each irradiation's own pulses.
/// No step of the schedule needs a transport run of its own.
///
/// ``std_dev`` is the standard error of the whole calculation: the spread of
/// those per-history doses, so it covers the neutron transport and the photon
/// transport, and the correlation between radionuclides and between
/// irradiations that one neutron history carries. For a machine that never
/// changes it is the error of the single-model D1S run. Runs on the CPU, on
/// one process.
///
/// With ``method="r2s"`` (cell-based rigorous two-step) the inventory of every
/// material-filled cell is solved instead, through the whole schedule with the
/// cell's own neutron spectrum, so activation products that are themselves
/// activated and burnup are included. Each step then gets a photon transport
/// of its own, with every cell emitting its decay photons at that time
/// uniformly over its volume (split cells where the flux varies steeply). The
/// pulses must all irradiate in one model; cells are matched across models by
/// name and emit wherever the step's model has them. A cell's volume is
/// estimated by sampling unless ``Cell(volume=...)`` gives it.
///
/// With ``method="r2s"`` and a ``mesh`` (cell-under-mesh R2S), every part of
/// a material-filled cell under a voxel is solved instead, with the voxel's
/// neutron spectrum (tallied in VITAMIN-J-175 groups), and emits uniformly
/// over that part, so the source follows the flux gradient through thick
/// cells without splitting them. The parts' volumes are estimated by sampling
/// each voxel. Pulses must irradiate with one set of moves, and a cell moved
/// after irradiation needs its ``moves``, as with D1S. Voxels no neutron
/// reaches are not activated.
///
/// With R2S, ``std_dev`` is the photon transport's statistical error given
/// the solved inventories; the neutron transport's is not included.
///
/// Args:
///     schedule: A ``PulseSchedule`` whose steps name models.
///     tallies: Photon tallies to score the dose in, without
///         ``parent_nuclides`` (they are binned by every radionuclide the
///         irradiations make). Each needs a unique ``name``, and must be valid
///         in every model the schedule observes (a cell filter's cells must
///         exist in each).
///     total_particles: Neutron histories per irradiated model. Every photon
///         run then has one history per neutron history (with R2S, each
///         step's photon run has this many).
///     seed: Base random seed.
///     threads: Worker threads; ``None`` uses all cores.
///     method: ``"d1s"`` (one-step, one photon run per pair of irradiation
///         and observation configurations) or ``"r2s"`` (cell inventories,
///         one photon run per step).
///     mesh: With ``method="r2s"``, a mesh over the activated components for
///         cell-under-mesh R2S, each element's part of every material-filled
///         cell activated on the element's neutron spectrum: a
///         ``RegularRectangularMesh``, a ``RegularCylindricalMesh``, or a tet
///         ``MeshGeometry`` (every tet, each inside one cell, as the model's
///         own tet mesh is). ``None`` activates whole cells.
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
#[pyo3(signature = (schedule, tallies, total_particles, seed=1, threads=None, method="d1s", mesh=None))]
pub fn simulate_shutdown_dose(
    py: Python<'_>,
    schedule: &PyPulseSchedule,
    tallies: Vec<PyRef<'_, PyTally>>,
    total_particles: usize,
    seed: u64,
    threads: Option<usize>,
    method: &str,
    #[gen_stub(override_type(
        type_repr = "RegularRectangularMesh | RegularCylindricalMesh | MeshGeometry | None"
    ))]
    mesh: Option<&Bound<'_, PyAny>>,
) -> PyResult<BTreeMap<String, PyDoseResult>> {
    let mesh = mesh.map(r2s_mesh).transpose()?;
    match (method, &mesh) {
        ("r2s", _) | ("d1s", None) => {}
        ("d1s", Some(_)) => {
            return Err(PyValueError::new_err(
                "mesh is for method=\"r2s\" (cell-under-mesh); D1S photons are born where \
                 their reactions happen and need no mesh",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "method must be \"d1s\" or \"r2s\", got \"{method}\""
            )));
        }
    }
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
        .detach(|| match method {
            "d1s" => yamc::shutdown_dose::simulate_shutdown_dose(
                &models, &stages, &steps, &tallies, &settings,
            ),
            _ => yamc::r2s::simulate_r2s(
                &models,
                &stages,
                &steps,
                &tallies,
                &settings,
                mesh.as_ref(),
            ),
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

/// The mesh R2S activates on, from the yamc mesh object given as `mesh=`.
fn r2s_mesh(mesh: &Bound<'_, PyAny>) -> PyResult<yamc::r2s::R2sMesh> {
    use yamc::r2s::R2sMesh;
    if let Ok(m) = mesh.extract::<PyRef<'_, PyRegularRectangularMesh>>() {
        return Ok(R2sMesh::Rectangular(m.internal.clone()));
    }
    if let Ok(m) = mesh.extract::<PyRef<'_, PyRegularCylindricalMesh>>() {
        return Ok(R2sMesh::Cylindrical(m.internal.clone()));
    }
    #[cfg(feature = "mesh")]
    if let Ok(m) = mesh.extract::<PyRef<'_, crate::geometry::PyMeshGeometry>>() {
        return Ok(R2sMesh::Tetrahedral(Arc::clone(&m.inner.mesh)));
    }
    Err(PyTypeError::new_err(
        "mesh must be a RegularRectangularMesh, RegularCylindricalMesh or MeshGeometry",
    ))
}
