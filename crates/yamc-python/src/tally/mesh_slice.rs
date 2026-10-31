//! `MeshSliceData` -- the rich 2D return type for `Tally.extract_mesh_slice()`
//! -- plus `parse_scores_arg`, the score-list parser shared by the tally
//! constructors.

use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc_tallies::tally::Score;

/// Rich return type for ``Tally.extract_mesh_slice()``.
///
/// Behaves like a 2D list of floats for backward compatibility
/// (``np.array(tally.extract_mesh_slice(...))`` still works), while also
/// exposing coordinate metadata for matplotlib plotting.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "MeshSliceData")]
pub struct PyMeshSliceData {
    #[pyo3(get)]
    pub values: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub h_edges: Vec<f64>,
    #[pyo3(get)]
    pub v_edges: Vec<f64>,
    #[pyo3(get)]
    pub extent: (f64, f64, f64, f64),
    #[pyo3(get)]
    pub h_label: String,
    #[pyo3(get)]
    pub v_label: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMeshSliceData {
    /// Backward compatibility: ``np.array(tally.extract_mesh_slice(...))`` works.
    fn __len__(&self) -> usize {
        self.values.len()
    }

    fn __getitem__(&self, idx: isize) -> PyResult<Vec<f64>> {
        let len = self.values.len() as isize;
        let i = if idx < 0 { idx + len } else { idx };
        if i < 0 || i >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            ));
        }
        Ok(self.values[i as usize].clone())
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::PyList;
        let list = PyList::new(py, &slf.values)?;
        list.call_method0("__iter__").map(|it| it.into())
    }

    /// Support ``np.array(mesh_slice_data)`` → 2D array of values.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (&self.values,))?;
        if let Some(dt) = dtype {
            return Ok(arr.call_method1("astype", (dt,))?.into());
        }
        if copy == Some(true) {
            return Ok(arr.call_method0("copy")?.into());
        }
        Ok(arr.into())
    }

    fn __repr__(&self) -> String {
        let pv = self.values.len();
        let ph = if pv > 0 { self.values[0].len() } else { 0 };
        format!(
            "MeshSliceData(shape=({pv}, {ph}), h_label='{}', v_label='{}')",
            self.h_label, self.v_label
        )
    }
}

/// Parse a Python score list (mix of ints and strings) into `Vec<Score>`.
pub(crate) fn parse_scores_arg(scores: &Bound<'_, PyAny>) -> PyResult<Vec<Score>> {
    let items: Vec<Bound<'_, PyAny>> = scores
        .extract()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("scores must be a list"))?;
    items
        .iter()
        .map(|item| {
            if item.is_instance_of::<pyo3::types::PyString>() {
                let s: String = item.extract()?;
                s.parse::<Score>()
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
            } else if let Ok(i) = item.extract::<i32>() {
                Score::from_mt_number(i).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Score must be int or string",
                ))
            }
        })
        .collect()
}
