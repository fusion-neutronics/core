//! `Processing`: continuous removal, transfer between materials and feed,
//! applied by `transmute` and `Model.simulate_transmutation`. A thin wrapper
//! over `yani_transmute::processing`, which holds the unit handling and the
//! solve.

use std::collections::HashMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yani_transmute::processing::{Processing, ProcessingRun, Rate};

use crate::distribution::parse_duration;
use crate::material::PyMaterial;

/// Removals, transfers between materials and feeds, for a transmutation run.
///
/// Each term adds to the Bateman equations of the materials it names, and
/// every one is solved exactly: a first-order removal is a loss rate, a
/// transfer is that loss arriving in the destination scaled by the ratio of
/// volumes, and a feed is a source held constant over each interval. Materials
/// a transfer links are solved together each step.
///
/// A destination need not be irradiated. One that ``transmute`` or
/// ``simulate_transmutation`` is not otherwise given (a storage tank outside
/// the geometry, say) is carried unirradiated and reported in the results
/// under its own id, so its activity and contact dose read like any other
/// material's.
///
/// A rate is a number, or a table ``(times, rates)``: piecewise constant, each
/// rate holding from its time (seconds from the start of the schedule, or
/// ``(value, unit)``) to the next, and zero before the first. Steps are split
/// where a table changes, so a table is solved exactly. ``steps`` restricts a
/// term to the listed schedule steps (0-based).
///
/// Examples:
///     >>> loop = yamc.Processing()
///     >>> # tritium out of the breeder with a 2 h half-time, into a tank
///     >>> loop.add_transfer(pbli, "H3", rate=(2, "h"), rate_units="half-time",
///     ...                   destination=tank)
///     >>> loop.add_transfer(pbli, "He", rate=6, rate_units="1/h")  # vented
///     >>> loop.add_feed(pbli, {"Li6": 1.0}, rate=0.05, rate_units="kg/d", steps=[0, 2])
///     >>> # a whole liquid circulating between a core and a pool
///     >>> loop.add_transfer(water, rate=flow / water.volume, destination=pool)
///     >>> loop.add_transfer(pool, rate=flow / pool.volume, destination=water)
///     >>> results = yamc.transmute([pbli, water], schedules, processing=loop)
#[gen_stub_pyclass]
#[pyclass(name = "Processing")]
pub struct PyProcessing {
    pub inner: Processing,
    /// Every material a term names, for the run to take the ones it is not
    /// otherwise given.
    materials: Vec<Py<PyMaterial>>,
}

impl PyProcessing {
    fn remember(&mut self, material: &Bound<'_, PyMaterial>) -> PyResult<u32> {
        if !self.materials.iter().any(|m| m.is(material)) {
            self.materials.push(material.clone().unbind());
        }
        material.borrow().material_id()
    }

    /// The processing with the materials it names, skipping the ones in
    /// `given`, which the run already holds.
    pub fn run<'a>(
        &'a self,
        py: Python<'_>,
        given: &[Bound<'_, PyMaterial>],
    ) -> PyResult<ProcessingRun<'a>> {
        let receivers = self
            .materials
            .iter()
            .filter(|m| !given.iter().any(|g| m.is(g)))
            .map(|m| Ok(m.bind(py).try_borrow()?.internal.clone()))
            .collect::<PyResult<_>>()?;
        Ok(ProcessingRun {
            processing: &self.inner,
            receivers,
        })
    }
}

/// A rate: a number, a `(value, unit)` duration where `durations`, or a
/// `(times, rates)` table of either.
fn parse_rate(obj: &Bound<'_, PyAny>, durations: bool) -> PyResult<Rate> {
    let value = |v: &Bound<'_, PyAny>| -> PyResult<f64> {
        if durations {
            parse_duration(v)
        } else {
            v.extract::<f64>()
                .map_err(|_| PyTypeError::new_err("a rate must be a number"))
        }
    };
    if obj.extract::<(f64, String)>().is_ok() || obj.extract::<f64>().is_ok() {
        return value(obj).map(Rate::Constant);
    }
    let (times, values): (Vec<Bound<'_, PyAny>>, Vec<Bound<'_, PyAny>>) =
        obj.extract().map_err(|_| {
            PyTypeError::new_err(
                "rate must be a number, a (value, unit) duration for a half-time, or a \
                 (times, rates) table",
            )
        })?;
    Ok(Rate::Table {
        times: times.iter().map(parse_duration).collect::<PyResult<_>>()?,
        values: values.iter().map(value).collect::<PyResult<_>>()?,
    })
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProcessing {
    #[new]
    fn new() -> Self {
        Self {
            inner: Processing::new(),
            materials: Vec::new(),
        }
    }

    /// Remove ``component`` from ``material`` continuously, at a first-order
    /// rate, into ``destination`` when given and out of the system otherwise.
    ///
    /// Args:
    ///     material (Material): Where it is removed from.
    ///     component (str, optional): An element (``"H"``, every isotope) or a
    ///         nuclide (``"H3"``). ``None`` (the default) removes everything,
    ///         as a flowing liquid does.
    ///     rate (float | tuple): The rate, or a ``(times, rates)`` table.
    ///     rate_units (str): ``"1/s"`` (or ``1/min``, ``1/h``, ``1/d``,
    ///         ``1/a``), or ``"half-time"`` for the time to remove half, in
    ///         seconds or as ``(value, unit)``.
    ///     destination (Material, optional): Where it goes. Both materials need
    ///         a volume: what leaves is diluted by their ratio.
    ///     steps (list[int], optional): The schedule steps it acts on; every
    ///         step by default.
    #[pyo3(signature = (material, component = None, *, rate, rate_units = "1/s", destination = None, steps = None))]
    #[allow(clippy::too_many_arguments)]
    fn add_transfer(
        &mut self,
        material: &Bound<'_, PyMaterial>,
        component: Option<&str>,
        rate: &Bound<'_, PyAny>,
        rate_units: &str,
        destination: Option<&Bound<'_, PyMaterial>>,
        steps: Option<Vec<usize>>,
    ) -> PyResult<()> {
        let rate = parse_rate(rate, rate_units.trim() == "half-time")?;
        let source = self.remember(material)?;
        let destination = destination.map(|d| self.remember(d)).transpose()?;
        self.inner
            .add_transfer(source, component, rate, rate_units, destination, steps)
            .map_err(PyValueError::new_err)
    }

    /// Feed ``composition`` into ``material`` continuously.
    ///
    /// Args:
    ///     material (Material): What it is fed into.
    ///     composition (dict[str, float]): Nuclides or natural elements, by
    ///         atom fraction (normalized).
    ///     rate (float | tuple): The rate, or a ``(times, rates)`` table.
    ///     rate_units (str): An amount per time, ``"g/s"``, ``"kg/d"``,
    ///         ``"mol/h"`` or ``"atoms/s"``, for which ``material`` needs a
    ///         volume; or with ``proportional_to`` a rate constant ``"1/s"``.
    ///     proportional_to (str, optional): A nuclide of ``material``: the feed
    ///         is then ``rate`` times its inventory, which holds an enrichment
    ///         (Li6 fed in proportion to Li7, say).
    ///     steps (list[int], optional): The schedule steps it acts on; every
    ///         step by default.
    #[pyo3(signature = (material, composition, *, rate, rate_units = "g/s", proportional_to = None, steps = None))]
    fn add_feed(
        &mut self,
        material: &Bound<'_, PyMaterial>,
        composition: HashMap<String, f64>,
        rate: &Bound<'_, PyAny>,
        rate_units: &str,
        proportional_to: Option<&str>,
        steps: Option<Vec<usize>>,
    ) -> PyResult<()> {
        let rate = parse_rate(rate, false)?;
        let id = self.remember(material)?;
        self.inner
            .add_feed(id, &composition, rate, rate_units, proportional_to, steps)
            .map_err(PyValueError::new_err)
    }

    /// The ids of every material a term names.
    #[getter]
    fn material_ids(&self) -> Vec<u32> {
        self.inner.material_ids().into_iter().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Processing({} transfer(s), {} feed(s), materials={:?})",
            self.inner.transfers.len(),
            self.inner.feeds.len(),
            self.material_ids()
        )
    }
}
