use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use yamc_nuclide::data::{ELEMENT_NAMES, ELEMENT_NUCLIDES, NATURAL_ABUNDANCE, REACTION_MT};

/// Return a dict mapping element symbol to its naturally occurring nuclides.
///
/// Each value is a sorted list of nuclide names for that element.
///
/// Returns:
///     dict[str, list[str]]: e.g. ``{"Fe": ["Fe54", "Fe56", "Fe57", "Fe58"], ...}``
#[gen_stub_pyfunction]
#[pyfunction]
pub fn element_nuclides(py: Python) -> Py<PyAny> {
    let dict = PyDict::new(py);
    for (element, nuclides) in ELEMENT_NUCLIDES.iter() {
        let mut sorted_nuclides = nuclides.clone();
        sorted_nuclides.sort();
        dict.set_item(*element, sorted_nuclides).unwrap();
    }
    dict.into()
}

/// Return a dict of natural isotopic abundances by nuclide name.
///
/// Returns:
///     dict[str, float]: e.g. ``{"Li6": 0.07589, "Li7": 0.92411, ...}``
#[gen_stub_pyfunction]
#[pyfunction]
pub fn natural_abundance(py: Python) -> Py<PyAny> {
    let dict = PyDict::new(py);
    for (k, v) in NATURAL_ABUNDANCE.iter() {
        dict.set_item(*k, v).unwrap();
    }
    dict.into()
}

/// Return a dict mapping element symbol to full element name.
///
/// Returns:
///     dict[str, str]: e.g. ``{"Fe": "iron", "Li": "lithium", ...}``
#[gen_stub_pyfunction]
#[pyfunction]
pub fn element_names(py: Python) -> Py<PyAny> {
    let dict = PyDict::new(py);
    for (symbol, name) in ELEMENT_NAMES.iter() {
        dict.set_item(*symbol, *name).unwrap();
    }
    dict.into()
}

/// Return a dict mapping MT numbers (int) to ENDF reaction notation strings.
/// e.g. {2: "(n,elastic)", 18: "(n,fission)", 102: "(n,gamma)", ...}
///
/// Each MT's own name, not an inversion of `REACTION_MT`. Several names map to
/// one MT (`"fission"` and `"(n,fission)"` are both MT 18), so inverting a
/// HashMap picked whichever the iteration order reached last: the same build
/// could return `{18: "fission"}` on one run and `{18: "(n,fission)"}` on the
/// next.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn reaction_names(py: Python) -> Py<PyAny> {
    let dict = PyDict::new(py);
    let mut mts: Vec<i32> = REACTION_MT.values().copied().collect();
    mts.sort_unstable();
    mts.dedup();
    for mt in mts {
        if let Some(name) = endf::reaction_name(mt) {
            dict.set_item(mt, name).unwrap();
        }
    }
    dict.into()
}

/// Return the chemical symbol of an atomic number.
///
/// Bound so that a caller naming files or directories from ENDF data does not
/// have to carry its own copy of the periodic table, or install a second
/// nuclear-data package to borrow one.
///
/// Args:
///     z (int): Atomic number, 0 to 118.
///
/// Returns:
///     str: The chemical symbol, e.g. ``"Fe"``.
///
/// Raises:
///     ValueError: If there is no element with that atomic number.
///
/// Examples:
///     >>> yani.data.atomic_symbol(26)
///     'Fe'
///     >>> yani.data.atomic_symbol(0)
///     'n'
///
/// Note:
///     Z = 0 is the neutron, ``"n"``, and is not an error. ENDF libraries
///     really do ship a neutron material (JENDL is one), so a caller walking a
///     sublibrary will meet it.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn atomic_symbol(z: u32) -> pyo3::PyResult<String> {
    endf::data::ATOMIC_SYMBOL
        .get(z as usize)
        .map(|symbol| (*symbol).to_string())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("no element with atomic number {z}"))
        })
}

/// Return the atomic number of a chemical symbol.
///
/// The inverse of :func:`atomic_symbol`.
///
/// Args:
///     symbol (str): A chemical symbol, e.g. ``"Fe"``. Case sensitive, and the
///         symbol only: ``"fe"`` and ``"iron"`` are both refused.
///
/// Returns:
///     int: The atomic number.
///
/// Raises:
///     ValueError: If the symbol names no element.
///
/// Examples:
///     >>> yani.data.atomic_number("Fe")
///     26
#[gen_stub_pyfunction]
#[pyfunction]
pub fn atomic_number(symbol: &str) -> pyo3::PyResult<u32> {
    endf::data::atomic_number(symbol).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("{symbol:?} is not a chemical symbol"))
    })
}

/// Split a nuclide name into its element symbol, mass number and isomeric state.
///
/// The one place downstream code should ask, because the obvious way of doing
/// it by hand is wrong for exactly the entry that is easiest to forget:
/// stripping the trailing digits off ``"Ta180_m1"`` leaves ``"Ta180_m"`` and a
/// mass number of 1, and natural tantalum really does contain Ta180m, so a
/// composition built from :func:`natural_abundance` will hand you that name.
///
/// Args:
///     name (str): A nuclide name, e.g. ``"Fe56"`` or ``"Ta180_m1"``.
///
/// Returns:
///     tuple[str, int, int]: Element symbol, mass number, and isomeric state
///         (0 for a ground state, 1 for the first metastable, and so on).
///
/// Raises:
///     ValueError: If the name is not a nuclide name.
///
/// Examples:
///     >>> yani.data.split_nuclide("Fe56")
///     ('Fe', 56, 0)
///     >>> yani.data.split_nuclide("Ta180_m1")
///     ('Ta', 180, 1)
#[gen_stub_pyfunction]
#[pyfunction]
pub fn split_nuclide(name: &str) -> pyo3::PyResult<(String, u32, u32)> {
    let (z, a, m) = endf::data::zam(name).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{name:?} is not a nuclide name (expected something like 'Fe56' or 'Ta180_m1')"
        ))
    })?;
    let symbol = endf::data::ATOMIC_SYMBOL.get(z as usize).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("no element with atomic number {z}"))
    })?;
    Ok(((*symbol).to_string(), a, m))
}
