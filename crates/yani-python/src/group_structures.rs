//! The built-in energy group structures, as data rather than as a spelling.
//!
//! A name like `"CCFE-709"` was previously only usable as an argument (a
//! `Histogram`'s `boundaries`, a `Tally`'s `energy_group_structure`), which
//! left a caller holding 709 flux values with no access to the energies they
//! belong to. These two functions hand over the edges themselves, so a
//! spectrum can be plotted, tabulated, or folded against a cross section on
//! its own grid. Issue #492.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use yamc_nuclide::group_structures::{available_group_structures, get_group_structure};

/// Energy boundaries of a built-in group structure, in eV.
///
/// Args:
///     name: Name of the group structure, case-sensitive, one of the names
///         :func:`group_structure_names` returns (e.g. ``"UKAEA-1102"``).
///
/// Returns:
///     list[float]: The bin edges in eV, ascending. One more than the number
///     of groups, so ``len(edges) - 1`` groups.
///
/// Raises:
///     ValueError: If the name is not a built-in group structure. The message
///                 lists the ones that are.
///
/// Examples:
///     >>> import yamc
///     >>> edges = yamc.group_structure("UKAEA-1102")
///     >>> len(edges), edges[0], edges[-1]
///     (1103, 1e-05, 1000000000.0)
///     >>> # the energies a 1102-group spectrum belongs to
///     >>> midpoints = [(lo + hi) / 2 for lo, hi in zip(edges, edges[1:])]
#[gen_stub_pyfunction]
#[pyfunction]
pub fn group_structure(name: &str) -> PyResult<Vec<f64>> {
    get_group_structure(name)
        .map(|edges| edges.to_vec())
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

/// Names of the built-in group structures.
///
/// Neutron structures first by increasing group count, then photon. Every name
/// except ``"CCFE-24-PHOTON"`` carries the same boundaries as OpenMC's
/// ``openmc.mgxs.GROUP_STRUCTURES`` entry of that name.
///
/// Returns:
///     list[str]: The names accepted by :func:`group_structure`, and anywhere
///     else a group structure can be named.
///
/// Examples:
///     >>> import yamc
///     >>> names = yamc.group_structure_names()
///     >>> "CCFE-709" in names
///     True
///     >>> {n: len(yamc.group_structure(n)) - 1 for n in names[:2]}
///     {'XMAS-172': 172, 'VITAMIN-J-175': 175}
#[gen_stub_pyfunction]
#[pyfunction]
pub fn group_structure_names() -> Vec<String> {
    available_group_structures()
        .into_iter()
        .map(String::from)
        .collect()
}
