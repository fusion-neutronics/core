//! `transmute(materials, schedules)`: several materials in one call.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::distribution::PyPulseSchedule;
use crate::material::PyMaterial;
use crate::transmutation_results::PyTransmutationResults;

/// Transmute several materials over one timeline in one call.
///
/// The plural of ``Material.transmute``, for a mesh from a transport run, a
/// component broken into regions, or a sweep over compositions. The answer for
/// each material is exactly what ``material.transmute(schedule)`` gives it, and
/// the result is one ``TransmutationResults`` keyed by each material's ``id``,
/// the same shape ``Model.simulate_transmutation`` returns.
///
/// What a Python loop would repeat is done once instead: the chain is read
/// once, each nuclide's cross sections are decoded once and shared by every
/// material that needs them, and a multigroup collapse runs once for every
/// distinct combination of spectrum, composition, temperature and shielding.
/// Cells of one steel that saw the same spectrum share a collapse, whatever
/// their flux magnitudes. ``results.collapse_reuse`` says how many collapses
/// were shared. The materials' solves then run in parallel.
///
/// Args:
///     materials (list[Material]): The materials, each with a distinct ``id``,
///         since the results are keyed by it. As with ``Material.transmute``,
///         the cross sections each needs are loaded into it and kept, and the
///         composition is not modified.
///     schedules (PulseSchedule | list[PulseSchedule]): One schedule for every
///         material, or one per material in the same order. Each material's
///         schedule carries its own spectra (the pulse sources) and flux
///         magnitudes (the pulse rates), which is how each cell of a mesh gets
///         its own flux. They must share one timeline, the same durations and
///         irradiation on the same steps, because the result has one series of
///         times.
///     data_uncertainty (DataUncertainty, optional): Nuclear-data uncertainty,
///         applied to every material as ``Material.transmute`` applies it to
///         one. The same seed perturbs a given evaluation the same way in every
///         material.
///     self_shielding_chord (float, optional): One chord length ``4V/S`` in cm,
///         for every material. See ``Material.transmute``.
///     self_shielding_shape (SphereLump | CubeLump | FoilLump | CylinderLump | WireLump, optional):
///         One lump shape for every material, turned into a chord through each
///         material's own ``volume``. Give this or ``self_shielding_chord``,
///         not both.
///
/// Returns:
///     TransmutationResults: Keyed by each material's ``id``. Per material,
///         ``get_source_rates(id)`` gives its flux magnitudes and
///         ``get_self_shielding_info(id)`` its shielding report.
///
/// Raises:
///     ValueError: If two materials share an id, the timelines differ, the
///         number of schedules does not match the number of materials, or
///         any single material would fail ``Material.transmute``, in which
///         case the message names the material.
///     TypeError: If ``schedules`` is neither a PulseSchedule nor a list of
///         them, or the same Material object is given twice.
///
/// Examples:
///     >>> schedules = [
///     ...     yani.PulseSchedule([
///     ...         yani.Pulse(rate=flux[i], duration=(1, "y"), source=spectra[i]),
///     ...         yani.Cooldown(duration=(1, "d")),
///     ...     ])
///     ...     for i in range(len(cells))
///     ... ]
///     >>> results = yani.transmute(cells, schedules)
///     >>> results.get_final_material(cells[3].id)
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (materials, schedules, data_uncertainty = None, self_shielding_chord = None, self_shielding_shape = None))]
pub fn transmute(
    py: Python<'_>,
    materials: Vec<Bound<'_, PyMaterial>>,
    #[gen_stub(override_type(type_repr = "PulseSchedule | typing.Sequence[PulseSchedule]"))]
    schedules: &Bound<'_, PyAny>,
    data_uncertainty: Option<crate::data_uncertainty::PyDataUncertainty>,
    self_shielding_chord: Option<f64>,
    // Qualified, because the generator files the lump classes under the
    // `shapes` submodule: the bare names do not resolve in the root stub.
    #[gen_stub(override_type(
        type_repr = "shapes.SphereLump | shapes.CubeLump | shapes.FoilLump | shapes.CylinderLump | shapes.WireLump | None"
    ))]
    self_shielding_shape: Option<Bound<'_, PyAny>>,
) -> PyResult<PyTransmutationResults> {
    if materials.is_empty() {
        return Err(PyValueError::new_err(
            "materials is empty: nothing to transmute",
        ));
    }

    // One schedule shared, or one per material.
    let schedules: Vec<Bound<'_, PyPulseSchedule>> = match schedules.cast::<PyPulseSchedule>() {
        Ok(one) => vec![one.clone(); materials.len()],
        Err(_) => {
            let list: Vec<Bound<'_, PyAny>> = schedules.extract().map_err(|_| {
                PyTypeError::new_err("schedules must be a PulseSchedule or a list of them")
            })?;
            if list.len() != materials.len() {
                return Err(PyValueError::new_err(format!(
                    "{} schedules for {} materials: give one schedule for all of them, or \
                     one per material",
                    list.len(),
                    materials.len()
                )));
            }
            list.into_iter()
                .enumerate()
                .map(|(i, s)| {
                    s.cast_into::<PyPulseSchedule>().map_err(|_| {
                        PyTypeError::new_err(format!("schedules[{i}] is not a PulseSchedule"))
                    })
                })
                .collect::<PyResult<_>>()?
        }
    };

    let loaded = crate::distribution::resolve_chain()?;
    let mut plans = Vec::with_capacity(materials.len());
    let mut guards = Vec::with_capacity(materials.len());
    for (i, (material, schedule)) in materials.iter().zip(&schedules).enumerate() {
        // Each irradiation pulse carries its spectrum (a Histogram-energy
        // NeutronSource) and a magnitude (rate); cooldowns are decay-only.
        let (spectra, steps) = schedule.borrow().transmute_plan(py)?;
        let guard = material.try_borrow_mut().map_err(|_| {
            PyTypeError::new_err(format!(
                "materials[{i}] appears more than once; give each material once"
            ))
        })?;
        let shielding = crate::material::shielding_request(
            &guard.internal,
            self_shielding_chord,
            self_shielding_shape.as_ref(),
        )?;
        plans.push((spectra, steps, shielding));
        guards.push(guard);
    }

    // Everything the solve needs is owned or borrowed from the guards, which
    // outlive it, so the GIL can be released for the length of the solve as
    // `Material.transmute` does.
    let uncertainty = data_uncertainty.map(|u| u.inner);
    let mut refs: Vec<&mut yamc_materials::material::Material> =
        guards.iter_mut().map(|g| &mut g.internal).collect();
    let results = py.detach(move || {
        let cases = refs
            .iter_mut()
            .zip(plans)
            .map(
                |(material, (spectra, steps, shielding))| yani_transmute::TransmuteCase {
                    material,
                    spectra,
                    steps,
                    shielding,
                },
            )
            .collect();
        yani_transmute::transmute_materials(
            cases,
            loaded.chain,
            &loaded.branch,
            loaded.parts,
            uncertainty.as_ref(),
        )
        // `Box<dyn Error>` is not `Send`, so it cannot come back out through
        // `detach`; the message is what the caller sees anyway.
        .map_err(|e| e.to_string())
    });
    let results = results.map_err(PyValueError::new_err)?;
    Ok(PyTransmutationResults { inner: results })
}
