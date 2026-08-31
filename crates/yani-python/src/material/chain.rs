use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::HashMap;
use std::sync::Arc;

use yani::{ChainNuclide, DecaySourceDistribution};

/// A parsed transmutation chain.
///
/// Wraps the parsed chain data so it can be reused without re-parsing it.
///
/// Examples:
///     >>> chain = yamc.TransmutationChain("transmutation-endf-b8.1-sfr.arrow")
///     >>> reduced = chain.reduce(["Fe56"])
///     >>> reduced.nuclide_names
#[gen_stub_pyclass]
#[pyclass(name = "TransmutationChain", from_py_object)]
#[derive(Clone)]
pub struct PyChain {
    pub inner: Arc<HashMap<String, ChainNuclide>>,
    /// Source data library (e.g. "endf-b8.1"), read from the chain's
    /// `version.json` at load and preserved through `reduce()` so
    /// `export_to_arrow` can record the real library rather than "unknown".
    pub library: Option<String>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyChain {
    /// Parse a transmutation chain from a split (v2) chain directory that
    /// holds `decay/`, `reactions/` and `fission_yields/` subsection subdirs.
    ///
    /// Args:
    ///     path (str): Path to the chain root directory.
    #[new]
    #[pyo3(text_signature = "(path)")]
    pub fn new(path: &str) -> PyResult<Self> {
        let root = std::path::Path::new(path);
        if !root.is_dir() {
            return Err(pyo3::exceptions::PyIOError::new_err(format!(
                "chain directory not found: {path}"
            )));
        }
        // The Python-facing chain object holds the base three-part merge; the
        // isomeric-branching overlay (and its `(n,n')` grafts) is applied later
        // in `load_configured_chain` from the configured `branch_ratios`
        // source, so `reduce()`/`export_to_arrow()` round-trip the base chain.
        let branch_root = root.join("branching");
        let (chain, _branch) = yani::parse_chain_parts(
            &root.join("decay"),
            Some(&root.join("reactions")),
            Some(&root.join("fission_yields")),
            branch_root.is_dir().then_some(branch_root.as_path()),
        )
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to load chain: {e}")))?;
        let library = read_chain_library(path);
        Ok(PyChain {
            inner: std::sync::Arc::new(chain),
            library,
        })
    }

    /// Reduce the chain to nuclides reachable from seed nuclides.
    ///
    /// By default this walks until the reachable set stops growing, so the
    /// result holds every nuclide the seeds can ever produce, no matter how
    /// long they are irradiated. Pass ``level`` only to build a deliberately
    /// truncated chain (a small test fixture, say); a truncated chain will
    /// miss inventory at high fluence.
    ///
    /// Args:
    ///     nuclides (list[str]): Seed nuclide names.
    ///     level (int | None): Cap on the number of reaction/decay steps to
    ///         walk. Default None, meaning walk to saturation.
    ///
    /// Returns:
    ///     TransmutationChain: Reduced chain containing only reachable nuclides.
    #[pyo3(signature = (nuclides, level=None))]
    pub fn reduce(&self, nuclides: Vec<String>, level: Option<usize>) -> PyChain {
        let refs: Vec<&str> = nuclides.iter().map(|s| s.as_str()).collect();
        let reduced = yani::reduce_chain(&self.inner, &refs, level.unwrap_or(usize::MAX));
        PyChain {
            inner: Arc::new(reduced),
            library: self.library.clone(),
        }
    }

    /// List of nuclide names in the chain.
    #[getter]
    pub fn nuclide_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.inner.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of nuclides in the chain.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("TransmutationChain({} nuclides)", self.inner.len())
    }

    /// Write the chain to a split (v2) chain directory with `decay/`,
    /// `reactions/` and `fission_yields/` subsection subdirs.
    ///
    /// Args:
    ///     path (str): Output directory path.
    pub fn export_to_arrow(&self, path: &str) -> PyResult<()> {
        yani::export_chain_parts(&self.inner, path, self.library.as_deref()).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to export chain: {e}"))
        })
    }

    /// Half-lives of the unstable nuclides in the chain.
    ///
    /// Returns:
    ///     dict[str, float]: nuclide name -> half-life in seconds. Stable
    ///     nuclides (no half-life) are omitted.
    #[getter]
    pub fn half_lives(&self, py: Python) -> Py<PyAny> {
        let dict = PyDict::new(py);
        for (name, nuclide) in self.inner.iter() {
            if let Some(hl) = nuclide.half_life {
                dict.set_item(name.as_str(), hl).unwrap();
            }
        }
        dict.into()
    }

    /// Mean decay energies of nuclides in the chain.
    ///
    /// Returns:
    ///     dict[str, float]: nuclide name -> mean decay energy in eV. Nuclides
    ///     with zero decay energy are omitted.
    #[getter]
    pub fn decay_energies(&self, py: Python) -> Py<PyAny> {
        let dict = PyDict::new(py);
        for (name, nuclide) in self.inner.iter() {
            if nuclide.decay_energy > 0.0 {
                dict.set_item(name.as_str(), nuclide.decay_energy).unwrap();
            }
        }
        dict.into()
    }

    /// Neutron-induced reactions of each nuclide in the chain.
    ///
    /// Every nuclide in the chain is a key, including those with no reactions.
    /// For the decay half of a production route see :attr:`decays`.
    ///
    /// Returns:
    ///     dict[str, list[tuple[str, str | None, float]]]: nuclide name ->
    ///     list of (reaction_type, target, branching_ratio). ``target`` is
    ///     ``None`` for a channel that names no single product, which in
    ///     practice means fission.
    #[getter]
    pub fn reactions(&self, py: Python) -> Py<PyAny> {
        let dict = PyDict::new(py);
        for (name, nuclide) in self.inner.iter() {
            dict.set_item(name.as_str(), edge_list(py, &nuclide.reactions))
                .unwrap();
        }
        dict.into()
    }

    /// Decay modes of each unstable nuclide in the chain.
    ///
    /// The other half of a production route. :attr:`reactions` gives the
    /// neutron-induced edge, and a walk with only that stops at the first
    /// product: it reaches the Hf183 in ``W186(n,a)Hf183`` and not the
    /// ``beta-`` step that carries it on to Ta183, or the ``IT`` step that
    /// turns an isomer into its own ground state. Multi-step routes are what a
    /// decay heat pathway analysis is made of.
    ///
    /// Same tuple shape as :attr:`reactions`, and the same underlying
    /// ``ChainReaction``, so a walk can concatenate the two into one edge list.
    ///
    /// Returns:
    ///     dict[str, list[tuple[str, str | None, float]]]: nuclide name ->
    ///     list of (decay_mode, daughter, branching_ratio), the branchings for
    ///     one parent summing to 1. Modes carry the evaluation's own
    ///     spellings: ``"beta-"``, ``"ec/beta+"``, ``"alpha"``, ``"IT"``,
    ///     ``"sf"``, ``"p"``, ``"n"``, and multi-particle emissions written as
    ///     ``"beta-,n"``. ``daughter`` is the parent itself on ``"sf"``, whose
    ///     products come from the fission yields rather than from the edge, and
    ///     ``None`` where the mode's product is outside the chain. Stable
    ///     nuclides have no decay modes and are omitted, as they are from
    ///     :attr:`half_lives`.
    #[getter]
    pub fn decays(&self, py: Python) -> Py<PyAny> {
        let dict = PyDict::new(py);
        for (name, nuclide) in self.inner.iter() {
            if nuclide.decays.is_empty() {
                continue;
            }
            dict.set_item(name.as_str(), edge_list(py, &nuclide.decays))
                .unwrap();
        }
        dict.into()
    }

    /// Decay photon sources of each nuclide that has them (D1S data).
    ///
    /// Returns:
    ///     dict[str, list[tuple[list[float], list[float]]]]: nuclide name ->
    ///     list of (energies, intensities) for each photon source.
    #[getter]
    pub fn photon_sources(&self, py: Python) -> Py<PyAny> {
        let dict = PyDict::new(py);
        for (name, nuclide) in self.inner.iter() {
            if nuclide.sources.is_empty() {
                continue;
            }
            let sources = PyList::empty(py);
            for src in &nuclide.sources {
                if src.particle == "photon" {
                    match &src.distribution {
                        DecaySourceDistribution::Discrete {
                            energies,
                            intensities,
                        } => {
                            sources
                                .append((energies.clone(), intensities.clone()))
                                .unwrap();
                        }
                    }
                }
            }
            if !sources.is_empty() {
                dict.set_item(name.as_str(), sources).unwrap();
            }
        }
        dict.into()
    }
}

/// One chain edge per entry, as `(kind, target, branching)`.
///
/// `reactions` and `decays` are both `Vec<ChainReaction>` and both convert the
/// same way, so a route walked across the two is walked over one tuple shape.
/// `target` is genuinely `None` rather than the string `"None"`: the type these
/// getters advertise is `str | None`, and a `"None"` string is a nuclide name
/// that never matches anything in the chain.
fn edge_list<'py>(py: Python<'py>, edges: &[yani::ChainReaction]) -> Bound<'py, PyList> {
    let list = PyList::empty(py);
    for edge in edges {
        list.append((edge.kind.as_str(), edge.target.as_deref(), edge.branching))
            .unwrap();
    }
    list
}

/// Read the `library` field from a chain directory's `manifest.json`, if present.
/// Returns `None` when the file or field is missing, or recorded as "unknown".
fn read_chain_library(path: &str) -> Option<String> {
    let version_path = std::path::Path::new(path).join("manifest.json");
    let text = std::fs::read_to_string(version_path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let lib = value.get("library")?.as_str()?;
    if lib.is_empty() || lib == "unknown" {
        None
    } else {
        Some(lib.to_string())
    }
}
