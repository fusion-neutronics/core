//! Clearance indexes for a material, through the
//! `radiological-material-clearance-finder` crate.
//!
//! The crate holds the regulatory tables and the index arithmetic. What this
//! module adds is the inventory: a Material's atom densities, and the
//! half-lives of the configured transmutation chain, so the index is computed
//! from the same decay constants `activity()` and `decay_heat()` use rather
//! than the crate's own ENDF/B-VIII.0 table.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use radiological_material_clearance_finder as clearance;
use yamc_materials::material::Material;
use yani::ChainNuclide;

pub(crate) fn to_py_err(error: clearance::Error) -> PyErr {
    match error {
        clearance::Error::UnknownLimitSet(message) => PyKeyError::new_err(message),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// Decay data whose half-lives are the chain's.
///
/// A nuclide the chain calls stable, or does not hold at all, is stable here
/// too, which is how `yani_decay::activity_by_nuclide` already treats it.
/// Atomic masses come from the crate's AME2020 table. Chain names the crate
/// cannot read as a single nuclide are left out rather than failing the whole
/// table over one entry.
pub(crate) fn decay_data_from_chain(
    chain: &HashMap<String, ChainNuclide>,
) -> PyResult<Arc<clearance::DecayData>> {
    let half_lives = chain.iter().filter_map(|(name, nuclide)| {
        let half_life = nuclide.half_life?;
        (half_life > 0.0 && half_life.is_finite() && clearance::nuclide::is_valid(name))
            .then_some((name.as_str(), half_life))
    });
    let masses = clearance::DecayData::shared_default()
        .atomic_masses()
        .clone();
    Ok(Arc::new(
        clearance::DecayData::new(half_lives, masses).map_err(to_py_err)?,
    ))
}

/// The material as the clearance crate sees it: atom densities, the volume
/// when one is set (only the total activity sets need it) and the name.
pub(crate) fn inventory(
    material: &Material,
    chain: &HashMap<String, ChainNuclide>,
) -> PyResult<clearance::Material> {
    let densities = material
        .get_atoms_per_barn_cm()
        .map_err(PyValueError::new_err)?;
    let mut inventory = clearance::Material::from_atom_densities(densities)
        .map_err(to_py_err)?
        .with_decay_data(decay_data_from_chain(chain)?);
    if let Some(volume) = material.volume {
        inventory = inventory.with_volume(volume).map_err(to_py_err)?;
    }
    if let Some(name) = &material.name {
        inventory = inventory.with_name(name.clone());
    }
    Ok(inventory)
}

pub(crate) fn options(
    metal: bool,
    apply_default_limit: bool,
    exclude_daughters: bool,
) -> clearance::ClearanceOptions {
    clearance::ClearanceOptions {
        metal,
        apply_default_limit,
        exclude_daughters,
    }
}

fn pairs_to_dict<'py, V>(py: Python<'py>, pairs: &[(String, V)]) -> PyResult<Bound<'py, PyDict>>
where
    V: Clone + IntoPyObject<'py>,
{
    let dict = PyDict::new(py);
    for (key, value) in pairs {
        dict.set_item(key, value.clone())?;
    }
    Ok(dict)
}

/// The outcome of assessing a material against one clearance limit set.
///
/// Every regulation here uses the same arithmetic: each radionuclide's
/// activity divided by its tabulated limit, summed. The material meets the
/// limits when that sum, ``index``, is below ``threshold``. The per-nuclide
/// dicts are ordered largest first.
///
/// Examples:
///     >>> result = material.clearance_index("UK_EPR16_out_of_scope")
///     >>> result.clearable
///     False
///     >>> result.dominant(2)
///     [('Co60', 11.2...), ('Cs137', 0.029...)]
///     >>> print(result)
#[gen_stub_pyclass]
#[pyclass(name = "ClearanceResult", frozen)]
pub struct PyClearanceResult {
    pub(crate) inner: clearance::ClearanceResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyClearanceResult {
    /// Name of the limit set assessed against.
    #[getter]
    fn limit_set(&self) -> String {
        self.inner.limit_set.clone()
    }

    /// The sum of activity-to-limit ratios.
    #[getter]
    fn index(&self) -> f64 {
        self.inner.index
    }

    /// The value the index must stay below, normally 1.
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    /// Activity units the comparison was made in: ``"Bq/g"``, ``"Ci/m3"`` or
    /// ``"Bq"``.
    #[getter]
    fn units(&self) -> &'static str {
        self.inner.units.as_str()
    }

    /// Whether the material meets this set's limits.
    #[getter]
    fn clearable(&self) -> bool {
        self.inner.clearable()
    }

    /// Whether the regulation excludes this material outright, which for the
    /// UK sets means every radionuclide present is shorter lived than 100 s.
    #[getter]
    fn out_of_scope(&self) -> bool {
        self.inner.out_of_scope
    }

    /// Each nuclide's contribution to the index, largest first.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, float]"))]
    fn by_nuclide<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs_to_dict(py, &self.inner.by_nuclide)
    }

    /// Each nuclide's activity in ``units``, largest first.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, float]"))]
    fn activities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs_to_dict(py, &self.inner.activities)
    }

    /// The limit applied to each nuclide, after any metal, per-gram, dynamic or
    /// secular equilibrium adjustment.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, float]"))]
    fn limits_used<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs_to_dict(py, &self.inner.limits_used)
    }

    /// Nuclides that took the set's catch-all limit because the table does
    /// not list them.
    #[getter]
    fn defaulted(&self) -> Vec<String> {
        self.inner.defaulted.clone()
    }

    /// Nuclide to the reason its activity was left out, which is always that a
    /// parent's limit already accounts for it in full.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, str]"))]
    fn excluded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs_to_dict(py, &self.inner.excluded)
    }

    /// Nuclide to the activity a parent accounted for, where the parent could
    /// only support part of it. The remainder was assessed against the
    /// nuclide's own limit.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, float]"))]
    fn credited<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs_to_dict(py, &self.inner.credited)
    }

    /// Activity present with no limit and no catch-all, so absent from the
    /// index entirely. The number to check before trusting a comfortable index.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, float]"))]
    fn uncovered<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs_to_dict(py, &self.inner.uncovered)
    }

    /// Nuclides present that the source explicitly places no limit on.
    #[getter]
    fn unlimited(&self) -> Vec<String> {
        self.inner.unlimited.clone()
    }

    /// Total activity with no limit, in ``units``.
    #[getter]
    fn uncovered_activity(&self) -> f64 {
        self.inner.uncovered_activity()
    }

    /// Share of total activity that falls outside the sum, from 0 to 1.
    #[getter]
    fn uncovered_fraction(&self) -> f64 {
        self.inner.uncovered_fraction()
    }

    /// The material's name, carried through for reporting.
    #[getter]
    fn material_name(&self) -> String {
        self.inner.material_name.clone()
    }

    /// The activity actually charged against the limit for one nuclide: its
    /// total activity, less any part a parent accounted for.
    ///
    /// Args:
    ///     nuclide (str): Nuclide name, such as ``"Co60"``.
    fn assessed_activity(&self, nuclide: &str) -> f64 {
        self.inner.assessed_activity(nuclide)
    }

    /// The nuclides contributing most to the index, largest first.
    ///
    /// Args:
    ///     count (int): How many to return.
    #[pyo3(signature = (count=10))]
    fn dominant(&self, count: usize) -> Vec<(String, f64)> {
        self.inner.dominant(count).to_vec()
    }

    /// A JSON-serialisable copy of the result.
    #[gen_stub(override_return_type(type_repr = "dict[str, object]"))]
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = &self.inner;
        let d = PyDict::new(py);
        d.set_item("limit_set", &r.limit_set)?;
        d.set_item("material", &r.material_name)?;
        d.set_item("index", r.index)?;
        d.set_item("threshold", r.threshold)?;
        d.set_item("units", r.units.as_str())?;
        d.set_item("clearable", r.clearable())?;
        d.set_item("out_of_scope", r.out_of_scope)?;
        d.set_item("by_nuclide", pairs_to_dict(py, &r.by_nuclide)?)?;
        d.set_item("activities", pairs_to_dict(py, &r.activities)?)?;
        d.set_item("limits_used", pairs_to_dict(py, &r.limits_used)?)?;
        d.set_item("defaulted", r.defaulted.clone())?;
        d.set_item("excluded", pairs_to_dict(py, &r.excluded)?)?;
        d.set_item("credited", pairs_to_dict(py, &r.credited)?)?;
        d.set_item("uncovered", pairs_to_dict(py, &r.uncovered)?)?;
        d.set_item("unlimited", r.unlimited.clone())?;
        d.set_item("uncovered_fraction", r.uncovered_fraction())?;
        Ok(d)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClearanceResult('{}', index={}, clearable={})",
            self.inner.limit_set,
            self.inner.index,
            if self.inner.clearable() {
                "True"
            } else {
                "False"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nuclide(name: &str, half_life: Option<f64>) -> (String, ChainNuclide) {
        (
            name.to_string(),
            ChainNuclide {
                name: name.to_string(),
                half_life,
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
                decay_energy_components: Default::default(),
            },
        )
    }

    #[test]
    fn the_chain_half_lives_are_the_ones_used() {
        let chain: HashMap<String, ChainNuclide> = [
            nuclide("Co60", Some(1.0)),
            nuclide("Fe56", None),
            nuclide("Ag108_m1", Some(2.0)),
            nuclide("not a nuclide", Some(3.0)),
        ]
        .into();
        let data = decay_data_from_chain(&chain).unwrap();
        assert_eq!(data.half_life("Co60").unwrap(), Some(1.0));
        assert_eq!(data.half_life("Ag108m").unwrap(), Some(2.0));
        assert_eq!(data.half_life("Fe56").unwrap(), None);
        // Absent from the chain, so stable here, as activity() treats it.
        assert_eq!(data.half_life("Cs137").unwrap(), None);
    }

    #[test]
    fn a_cobalt_steel_index_follows_the_chain_half_life() {
        let half_life = 1.663_2e8;
        let chain: HashMap<String, ChainNuclide> =
            [nuclide("Co60", Some(half_life)), nuclide("Fe56", None)].into();
        let atoms_per_barn_cm = [("Fe56", 0.0849), ("Co60", 1e-10)];
        let inventory = clearance::Material::from_atom_densities(atoms_per_barn_cm)
            .unwrap()
            .with_decay_data(decay_data_from_chain(&chain).unwrap());
        let set = clearance::get_limit_set("UK_EPR16_out_of_scope").unwrap();
        let result =
            clearance::clearance_index(&inventory, &set, options(false, true, true)).unwrap();

        let data = clearance::DecayData::shared_default();
        let mass_per_cm3 = (0.0849 * data.atomic_mass("Fe56").unwrap()
            + 1e-10 * data.atomic_mass("Co60").unwrap())
            * 1e24
            / clearance::AVOGADRO;
        let bq_per_g = 1e-10 * 1e24 * std::f64::consts::LN_2 / half_life / mass_per_cm3;
        // UK EPR 2016 limits Co-60 to 0.1 Bq/g.
        let expected = bq_per_g / 0.1;
        assert!((result.index - expected).abs() <= 1e-12 * expected);
    }
}
