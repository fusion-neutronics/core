use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use std::collections::HashMap;
use std::sync::Arc;
use yamc_nuclide::buffer::F64Buffer;
use yamc_nuclide::nuclide::{FissionNuData, Nuclide};
use yamc_nuclide::reaction::Reaction;

/// Nuclide data container exposed to Python.
///
/// Create a new (optionally named) nuclide instance.
///
/// Args:
///     name (Optional[str]): Optional nuclide identifier (e.g. "Li6", "Fe56"). If not
///         supplied you must pass `path` to `load` later.
///
/// Notes:
///     Individual fields (e.g. `name`, `atomic_number`, `available_temperatures`,
///     `loaded_temperatures`) are exposed as read-only attributes via PyO3 getters.
///     Detailed descriptions appear once each in the generated documentation--this
///     summary omits a full per-attribute list to avoid duplication.
#[gen_stub_pyclass]
#[pyclass(name = "Nuclide", from_py_object)]
#[derive(Clone, Default)]
pub struct PyNuclide {
    pub name: Option<String>,
    pub element: Option<String>,
    pub atomic_symbol: Option<String>,
    pub atomic_number: Option<u32>,
    pub neutron_number: Option<u32>,
    pub mass_number: Option<u32>,
    pub atomic_weight_ratio: Option<f64>,
    pub library: Option<String>,
    /// Per-temperature energy grids. Shares its buffers with the loaded
    /// `Nuclide` rather than copying them, so `Nuclide <-> PyNuclide` round
    /// trips are O(1) in the grid size; the Python getters below still hand out
    /// lists.
    pub energy: Option<HashMap<String, F64Buffer>>,
    /// Reactions indexed by temperature index (use loaded_temperatures for keys)
    pub reactions: Vec<HashMap<i32, Arc<Reaction>>>,
    pub fissionable: bool,
    pub available_temperatures: Vec<String>,
    pub loaded_temperatures: Vec<String>,
    pub data_path: Option<String>,
    pub fission_nu: Option<FissionNuData>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNuclide {
    pub fn __repr__(&self) -> String {
        let name = self.name.as_deref().unwrap_or("unnamed");
        let n_temps = self.loaded_temperatures.len();
        format!("Nuclide(name={name}, loaded_temperatures={n_temps})")
    }

    /// Name / identifier for the nuclide (e.g. "Li6", "Fe56").
    ///
    /// Returns:
    ///     Optional[str]: Nuclide name or None if not yet set.
    #[getter]
    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }

    /// Chemical element symbol (e.g. "Fe").
    ///
    /// Returns:
    ///     Optional[str]: Element symbol or None if data not loaded.
    #[getter]
    pub fn element(&mut self) -> Option<String> {
        let mut rust_nuclide = Nuclide::from(self.clone());
        let result = rust_nuclide.get_element();
        // Update self with any auto-loaded data
        *self = PyNuclide::from(rust_nuclide);
        result
    }

    /// Atomic symbol (currently same as element symbol).
    ///
    /// Returns:
    ///     Optional[str]: Atomic symbol string.
    #[getter]
    pub fn atomic_symbol(&self) -> Option<String> {
        self.atomic_symbol.clone()
    }

    /// Proton number Z.
    ///
    /// Returns:
    ///     Optional[int]: Atomic number.
    #[getter]
    pub fn atomic_number(&mut self) -> Option<u32> {
        let mut rust_nuclide = Nuclide::from(self.clone());
        let result = rust_nuclide.get_atomic_number();
        // Update self with any auto-loaded data
        *self = PyNuclide::from(rust_nuclide);
        result
    }

    /// Neutron number N.
    ///
    /// Returns:
    ///     Optional[int]: Neutron count.
    #[getter]
    pub fn neutron_number(&self) -> Option<u32> {
        self.neutron_number
    }

    /// Mass number A = Z + N.
    ///
    /// Returns:
    ///     Optional[int]: Mass number.
    #[getter]
    pub fn mass_number(&mut self) -> Option<u32> {
        let mut rust_nuclide = Nuclide::from(self.clone());
        let result = rust_nuclide.get_mass_number();
        // Update self with any auto-loaded data
        *self = PyNuclide::from(rust_nuclide);
        result
    }

    /// Atomic weight ratio (target mass / neutron mass) from nuclear data file.
    ///
    /// Returns:
    ///     Optional[float]: Atomic weight ratio.
    #[getter]
    pub fn atomic_weight_ratio(&self) -> Option<f64> {
        self.atomic_weight_ratio
    }

    /// Originating nuclear data library identifier.
    ///
    /// Returns:
    ///     Optional[str]: Library name/code.
    #[getter]
    pub fn library(&self) -> Option<String> {
        self.library.clone()
    }

    /// Whether the nuclide is fissionable.
    ///
    /// Returns:
    ///     bool: True if fissionable.
    #[getter]
    pub fn fissionable(&self) -> bool {
        self.fissionable
    }

    /// All temperatures present in the source data file.
    ///
    /// Returns:
    ///     List[str]: Temperature labels in Kelvin (e.g. ["293"]).
    #[getter]
    pub fn available_temperatures(&mut self) -> Vec<String> {
        let mut rust_nuclide = Nuclide::from(self.clone());
        let result = rust_nuclide.get_available_temperatures();
        // Update self with any auto-loaded data
        *self = PyNuclide::from(rust_nuclide);
        result
    }

    /// Temperatures actually loaded into memory (subset of available_temperatures).
    ///
    /// Returns:
    ///     List[str]: Loaded temperatures.
    #[getter]
    pub fn loaded_temperatures(&self) -> Vec<String> {
        self.loaded_temperatures.clone()
    }

    /// Path to the data file used to populate this nuclide (if known).
    ///
    /// Returns:
    ///     Optional[str]: Filesystem path or None.
    #[getter]
    pub fn data_path(&self) -> Option<String> {
        self.data_path.clone()
    }
    /// Create a new (optionally named) nuclide.
    ///
    /// Args:
    ///     name (Optional[str]): Optional nuclide identifier (e.g. "Li6", "Fe56"). If not
    ///         supplied you must pass `path` to `load` later.
    ///
    /// Returns:
    ///     Nuclide: A nuclide object with no data loaded yet.
    #[new]
    #[pyo3(signature = (name=None))]
    pub fn new(name: Option<String>) -> Self {
        PyNuclide {
            name,
            element: None,
            atomic_symbol: None,
            atomic_number: None,
            neutron_number: None,
            mass_number: None,
            atomic_weight_ratio: None,
            library: None,
            energy: None,
            reactions: Vec::new(),
            fissionable: false,
            available_temperatures: Vec::new(),
            loaded_temperatures: Vec::new(),
            data_path: None,
            fission_nu: None,
        }
    }

    /// Read nuclear data from a file, keyword, or directory.
    ///
    /// You can either provide a source explicitly via `path` or rely on
    /// the `name` given at construction and the global configuration to resolve it.
    /// Providing an explicit `path` will override any global configuration.
    ///
    /// The `path` argument accepts:
    ///   - A file path (e.g. "tests/Li6.arrow")
    ///   - A keyword (e.g. "tendl-2025", "fendl-3.2d", "endf-b8.1")
    ///   - A directory path (e.g. "/path/to/neutron/") -- resolves to `<dir>/<name>.arrow`
    ///   - None -- looks up the path from yamc.cross_section_data
    ///
    /// When `temperatures` is provided only those temperatures are loaded while
    /// `available_temperatures` always lists every temperature present in the
    /// file. The subset actually loaded is stored in `loaded_temperatures`.
    ///
    /// Args:
    ///     path (Optional[str]): Optional path to the nuclide data file, keyword,
    ///         or directory. If omitted, the constructor `name` is used to look up
    ///         the source from yamc.cross_section_data.
    ///     temperatures (Optional[List[str]]): Temperature strings in Kelvin (e.g. ["293"]).
    ///         If given only these temperatures are loaded.
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     ValueError: If neither `path` nor `name` is available, if the nuclide
    ///         name is not found in global configuration (when path not provided),
    ///         or if the data cannot be read / parsed.
    ///
    /// Examples:
    ///     Compare the same nuclide from different data sources:
    ///
    ///     >>> # Set global default
    ///     >>> yamc.cross_section_data = "tendl-2025"
    ///     >>>
    ///     >>> # Load from global config (will use TENDL)
    ///     >>> li6_tendl = yamc.Nuclide("Li6")
    ///     >>> li6_tendl.read_nuclear_data()
    ///     >>>
    ///     >>> # Override to use FENDL for comparison
    ///     >>> li6_fendl = yamc.Nuclide("Li6")
    ///     >>> li6_fendl.read_nuclear_data("fendl-3.2d")
    ///     >>>
    ///     >>> # Use a directory
    ///     >>> li6_dir = yamc.Nuclide("Li6")
    ///     >>> li6_dir.read_nuclear_data("/path/to/neutron/")
    ///     >>>
    ///     >>> # Use custom local file
    ///     >>> li6_custom = yamc.Nuclide("Li6")
    ///     >>> li6_custom.read_nuclear_data("path/to/custom_Li6.arrow")
    #[pyo3(signature = (path=None, temperatures=None), text_signature = "(self, path=None, temperatures=None)")]
    pub fn read_nuclear_data(
        &mut self,
        path: Option<String>,
        temperatures: Option<Vec<String>>,
    ) -> PyResult<()> {
        use std::collections::HashSet;

        let temps_set: Option<HashSet<String>> = temperatures.map(|v| v.into_iter().collect());

        // Use the new Rust backend method that handles all the complex logic
        let nuclide = yamc_nuclide::nuclide::load_nuclide_for_python(
            path.as_deref(),
            self.name.as_deref(),
            temps_set.as_ref(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Simple field assignment from the loaded nuclide
        *self = PyNuclide::from(nuclide);
        Ok(())
    }

    /// Mapping of temperature -> MT number -> reaction data.
    ///
    /// Returns:
    ///     Dict[str, Dict[int, Dict[str, Any]]]: Nested dictionary. The innermost
    ///     dictionary has these keys:
    ///
    ///         - cross_section (List[float])
    ///         - threshold_idx (int)
    ///         - interpolation (List[int])
    ///         - energy (Optional[List[float]]): Present when reaction has its own grid
    #[getter]
    pub fn reactions(&self, py: Python) -> PyResult<Py<PyAny>> {
        let py_dict = PyDict::new(py);
        // Iterate over reactions with temperature keys from loaded_temperatures
        for (idx, mt_map) in self.reactions.iter().enumerate() {
            let temp = self
                .loaded_temperatures
                .get(idx)
                .cloned()
                .unwrap_or_else(|| idx.to_string());
            let mt_dict = PyDict::new(py);
            for (mt, reaction) in mt_map {
                let py_reaction = crate::material::PyReaction::from_reaction(reaction, py)?;
                let py_reaction_obj = Py::new(py, py_reaction)?;
                mt_dict.set_item(mt, py_reaction_obj)?;
            }
            py_dict.set_item(temp, mt_dict)?;
        }
        Ok(py_dict.into())
    }

    /// List of MT numbers available for the (first) loaded temperature.
    ///
    /// Returns:
    ///     Optional[List[int]]: List of MT identifiers or None if no data.
    #[getter]
    pub fn reaction_mts(&self) -> Option<Vec<i32>> {
        Nuclide::from(self.clone()).reaction_mts()
    }

    /// Energy grids by temperature.
    ///
    /// Returns:
    ///     Optional[Dict[str, List[float]]]: Map of temperature key to energy grid
    ///     or None if no energy data loaded.
    #[getter]
    pub fn energy(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if let Some(energy_map) = &self.energy {
            let py_dict = PyDict::new(py);
            for (temp_key, energy_grid) in energy_map.iter() {
                py_dict.set_item(temp_key, energy_grid.as_slice())?;
            }
            Ok(Some(py_dict.into()))
        } else {
            Ok(None)
        }
    }

    /// Get the energy grid for a specific temperature.
    ///
    /// Args:
    ///     temperature (str): Temperature key in Kelvin (e.g. "293").
    ///
    /// Returns:
    ///     Optional[List[float]]: The energy grid or None if not present.
    pub fn energy_grid(&self, temperature: &str) -> Option<Vec<f64>> {
        let nuclide = Nuclide::from(self.clone());
        nuclide.energy_grid(temperature).map(|grid| grid.to_vec())
    }

    /// Get energy grid for a specific temperature and MT number.
    ///
    /// Args:
    ///     temperature (str): Temperature to use for reaction data.
    ///     mt (int): ENDF/MT number for the reaction channel.
    ///
    /// Returns:
    ///     Optional[List[float]]: Reaction energy grid if present.
    pub fn get_reaction_energy_grid(&self, temperature: &str, mt: i32) -> Option<Vec<f64>> {
        // Get temperature index
        let temp_idx = self
            .loaded_temperatures
            .iter()
            .position(|t| t == temperature)?;
        if let Some(temp_reactions) = self.reactions.get(temp_idx) {
            if let Some(reaction) = temp_reactions.get(&mt) {
                if !reaction.energy.is_empty() {
                    return Some(reaction.energy.to_vec());
                }
            }
        }
        None
    }

    /// Get microscopic cross section data for a specific reaction and temperature.
    ///
    /// Args:
    ///     reaction (Union[int, str]): Either an ENDF/MT number (int) or reaction name (str)
    ///         like "(n,gamma)", "(n,elastic)", "fission", etc.
    ///     temperature (Optional[str]): Temperature to use. If None, uses the single
    ///         loaded temperature if only one is available.
    ///
    /// Returns:
    ///     Tuple[List[float], List[float]]: A tuple of (cross_section_values, energy_grid).
    ///
    /// Raises:
    ///     Exception: If temperature not found, reaction not found, multiple temperatures loaded
    ///         without specifying one, or no data available.
    #[pyo3(signature = (reaction, temperature=None))]
    pub fn microscopic_cross_section(
        &self,
        #[gen_stub(override_type(type_repr = "builtins.int | builtins.str"))] reaction: &Bound<
            '_,
            PyAny,
        >,
        temperature: Option<&str>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let mut nuclide: Nuclide = self.clone().into();

        // Handle both integer and string inputs
        let result = if let Ok(mt_num) = reaction.extract::<i32>() {
            nuclide.microscopic_cross_section(mt_num, temperature, true)
        } else if let Ok(reaction_name) = reaction.extract::<String>() {
            nuclide.microscopic_cross_section(reaction_name, temperature, true)
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "reaction must be either an integer (MT number) or string (reaction name)",
            ));
        };

        match result {
            Ok((cross_section, energy)) => Ok((cross_section, energy)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                e.to_string(),
            )),
        }
    }

    /// Sample a reaction based on cross sections at a given energy and temperature.
    ///
    /// This method randomly selects a nuclear reaction channel based on the relative
    /// cross sections at the specified neutron energy. It uses Monte Carlo sampling
    /// to select between absorption, elastic scattering, fission (if fissionable),
    /// and non-elastic reactions according to their probabilities.
    ///
    /// Args:
    ///     energy (float): Neutron energy in eV.
    ///     temperature (str): Temperature to use for reaction data in Kelvin (e.g. "294", "300").
    ///     seed (Optional[int]): Random seed for reproducible sampling. If None,
    ///         uses system random state.
    ///
    /// Returns:
    ///     Optional[Dict[str, Any]]: Dictionary containing the sampled reaction data:
    ///         - mt_number (int): ENDF/MT number of the sampled reaction
    ///         - cross_section (List[float]): Cross section values in barns
    ///         - threshold_idx (int): Index where reaction becomes active
    ///         - interpolation (List[int]): Interpolation flags
    ///         - energy (List[float]): Reaction energy grid
    ///
    ///         Returns None if no reaction could be sampled (e.g., zero total cross section).
    ///
    /// Raises:
    ///     ValueError: If temperature not found or no reaction data available.
    ///
    /// Examples:
    ///     >>> nuclide = Nuclide("Li6")
    ///     >>> nuclide.read_nuclear_data()
    ///     >>> reaction = nuclide.sample_reaction(1e-3, "294", seed=42)
    ///     >>> if reaction:
    ///     ...     print(f"Sampled MT {reaction['mt_number']}")
    #[pyo3(signature = (energy, temperature, seed=None), text_signature = "(self, energy, temperature, seed=None)")]
    pub fn sample_reaction(
        &mut self,
        energy: f64,
        temperature: &str,
        seed: Option<u64>,
    ) -> PyResult<Option<Py<PyAny>>> {
        use pyo3::types::PyDict;
        use pyo3::Python;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut nuclide: Nuclide = self.clone().into();

        // Create random number generator with optional seed
        let mut rng = if let Some(seed_val) = seed {
            StdRng::seed_from_u64(seed_val)
        } else {
            StdRng::from_rng(&mut rand::rng())
        };

        // Sample the reaction (now with auto-loading)
        let sampled_reaction = nuclide.sample_reaction(energy, temperature, &mut rng);

        // Extract reaction data before moving nuclide
        let reaction_data = sampled_reaction.map(|r| {
            (
                r.mt_number,
                r.cross_section.clone(),
                r.threshold_idx,
                r.energy.clone(),
            )
        });

        // Update self with any auto-loaded data
        *self = PyNuclide::from(nuclide);

        if let Some((mt_number, cross_section, threshold_idx, energy)) = reaction_data {
            // Convert the reaction to a Python dictionary
            Python::attach(|py| {
                let reaction_dict = PyDict::new(py);
                reaction_dict.set_item("mt_number", mt_number)?;
                reaction_dict.set_item("cross_section", cross_section.as_slice())?;
                reaction_dict.set_item("threshold_idx", threshold_idx)?;
                reaction_dict.set_item("energy", energy.as_slice())?;
                Ok(Some(reaction_dict.into()))
            })
        } else {
            Ok(None)
        }
    }
}

impl From<Nuclide> for PyNuclide {
    fn from(n: Nuclide) -> Self {
        PyNuclide {
            name: n.name,
            element: n.element,
            atomic_symbol: n.atomic_symbol,
            atomic_number: n.atomic_number,
            neutron_number: n.neutron_number,
            mass_number: n.mass_number,
            atomic_weight_ratio: n.atomic_weight_ratio,
            library: n.library,
            energy: n.energy,
            reactions: n.reactions,
            fissionable: n.fissionable,
            available_temperatures: n.available_temperatures,
            loaded_temperatures: n.loaded_temperatures,
            data_path: n.data_path,
            fission_nu: n.fission_nu,
        }
    }
}

impl From<PyNuclide> for Nuclide {
    fn from(py: PyNuclide) -> Self {
        Nuclide {
            name: py.name,
            element: py.element,
            atomic_symbol: py.atomic_symbol,
            atomic_number: py.atomic_number,
            neutron_number: py.neutron_number,
            mass_number: py.mass_number,
            atomic_weight_ratio: py.atomic_weight_ratio,
            library: py.library,
            energy: py.energy,
            reactions: py.reactions,
            fissionable: py.fissionable,
            available_temperatures: py.available_temperatures,
            loaded_temperatures: py.loaded_temperatures,
            data_path: py.data_path,
            fission_nu: py.fission_nu,
            fast_xs: Vec::new(),
            urr_data: Vec::new(),
            urr_present: false,
            fission_photon_release: None,
            covariance: None,
            elastic_flat_cache: Default::default(),
            fission_chi_flat_cache: Default::default(),
            delayed_neutron_cache: Default::default(),
            inelastic_angle_flat_cache: Default::default(),
            load_scope: Default::default(),
        }
    }
}

#[gen_stub_pyfunction]
#[pyfunction]
/// Clear any internally cached nuclide data.
///
/// This forces subsequent reads to re-parse JSON files.
///
/// Returns:
///     None
#[pyo3(text_signature = "()")]
pub fn clear_nuclide_cache() {
    yamc_nuclide::nuclide::clear_nuclide_cache();
}
