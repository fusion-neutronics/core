use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::HashMap;
use yamc_materials::material::Material;

/// A homogeneous mixture of nuclides at a fixed density and temperature.
///
/// A Material is built from a composition dictionary whose keys may be
/// nuclide names (``"Fe56"``), element symbols (``"Fe"``), chemical formulas
/// (``"H2O"``), or :class:`Nuclide` objects, and whose values are atom or
/// weight fractions (or :func:`enriched` specifications). Density is always a
/// float; ``units`` sets its meaning (``g/cm3`` by default, or
/// ``atom/barn-cm`` for a total atom density). For absolute per-nuclide atom
/// densities use :meth:`Material.from_atom_densities`.
///
/// Materials may also carry a name, ID, volume (cm³), temperature (K), and a
/// ``transmutable`` flag that marks them for depletion by
/// ``Model.simulate_transmutation`` (not needed for ``Material.transmute()``).
///
/// Examples:
///     >>> import yamc
///     >>> water = yamc.Material(composition={"H2O": 1.0}, density=1.0)
///     >>> iron = yamc.Material(
///     ...     composition={"Fe54": 0.0585, "Fe56": 0.9175,
///     ...                  "Fe57": 0.0212, "Fe58": 0.0028},
///     ...     density=7.874, name="iron")
#[gen_stub_pyclass]
#[pyclass(name = "Material", from_py_object)]
#[derive(Clone)]
pub struct PyMaterial {
    pub internal: Material,
}

/// Check if a string matches the nuclide naming pattern (e.g. Fe56, U235, Am241m).
///
/// Pattern: 1-2 letters (uppercase + optional lowercase) + 1+ digits + optional 'm'.
fn looks_like_nuclide(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() || !chars[0].is_ascii_uppercase() {
        return false;
    }
    let mut i = 1;
    if i < chars.len() && chars[i].is_ascii_lowercase() {
        i += 1;
    }
    let digit_start = i;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i == digit_start {
        return false;
    }
    if i < chars.len() && chars[i] == 'm' {
        i += 1;
    }
    i == chars.len()
}

/// Process a single composition entry, classifying the key as element/nuclide/formula.
/// Returns a map of nuclides to add to the composition.
fn expand_composition_entry(
    key: &str,
    fraction: f64,
    fraction_type: &str,
    enrichment: Option<(f64, &str, &str)>, // (percent, target, enrich_type)
) -> PyResult<HashMap<String, f64>> {
    use yamc_nuclide::composition;

    if key.chars().all(|c| c.is_alphabetic()) {
        // Element (symbol like "Fe" or name like "lithium")
        match enrichment {
            Some((percent, target, etype)) => composition::expand_element_enriched(
                key,
                fraction,
                percent,
                target,
                etype,
                fraction_type,
            )
            .map_err(PyValueError::new_err),
            None => composition::expand_element(key, fraction, fraction_type)
                .map_err(PyValueError::new_err),
        }
    } else if looks_like_nuclide(key) {
        // Nuclide (e.g. Fe56, U235, Am241m)
        if enrichment.is_some() {
            return Err(PyValueError::new_err(format!(
                "enrichment is not supported for nuclide '{key}' -- enrich the parent element instead"
            )));
        }
        composition::validate_nuclide_name(key).map_err(PyValueError::new_err)?;
        Ok(HashMap::from([(key.to_string(), fraction)]))
    } else {
        // Formula (e.g. H2O, Li4SiO4) -- expand_formula normalizes to sum=1.0,
        // so we scale the resulting nuclides by `fraction` afterwards.
        let nuclides = match enrichment {
            Some((percent, target, etype)) => composition::expand_formula(
                key,
                fraction_type,
                Some(percent),
                Some(target),
                Some(etype),
            )
            .map_err(PyValueError::new_err)?,
            None => composition::expand_formula(key, fraction_type, None, None, None)
                .map_err(PyValueError::new_err)?,
        };
        // Scale by the user's fraction
        Ok(nuclides
            .into_iter()
            .map(|(nuc, frac)| (nuc, frac * fraction))
            .collect())
    }
}

impl PyMaterial {
    /// The factor that turns a per-cm3 quantity into the one `per` asks for.
    ///
    /// Activity, decay heat and the photon spectrum all count atoms, and the
    /// solve holds atom DENSITIES, so the intensive answer is the one it
    /// natively has and `volume` is only ever the multiplier that makes it
    /// extensive (`atoms = density * BARN_PER_CM_SQ * volume`). Every one of
    /// them is therefore linear in this factor, which is what lets one helper
    /// serve all three (issue #567).
    ///
    /// * `None` (the default) asks for the total, so the factor is the volume
    ///   and the error when it is unset is the same one as before.
    /// * `"cm3"` asks for the intensive form the solve already has, so the
    ///   factor is 1 and nothing is required.
    /// * `"g"` divides by the mass density, which needs only `density`, and
    ///   `density` is already mandatory.
    ///
    /// The unit is therefore a function of what the caller wrote, never of what
    /// happens to be set on the material: a caller who does not pass `per`
    /// cannot tell this exists.
    fn scale_for(&self, per: Option<&str>, quantity: &str) -> PyResult<f64> {
        match per {
            None => self.internal.volume.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{quantity} requires material.volume in cm^3 for a total; \
                     pass per='cm3' or per='g' for a specific quantity, which needs no volume"
                ))
            }),
            Some("cm3") => Ok(1.0),
            Some("g") => {
                let density = self
                    .internal
                    .get_mass_density()
                    .map_err(PyValueError::new_err)?;
                if density.is_nan() || density <= 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "{quantity} with per='g' needs a positive mass density, got {density}"
                    )));
                }
                Ok(1.0 / density)
            }
            Some(other) => Err(PyValueError::new_err(format!(
                "unknown per={other:?} for {quantity}; expected None for a total, \
                 'cm3' for a per-volume quantity, or 'g' for a per-mass one"
            ))),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMaterial {
    /// Get the name of the material
    #[getter]
    fn name(&self) -> Option<String> {
        self.internal.name.as_deref().map(|s| s.to_string())
    }

    /// Set the name of the material
    #[setter]
    fn set_name(&mut self, name: String) {
        self.internal.name = Some(name);
    }

    /// Get the material ID
    #[getter]
    fn id(&self) -> Option<u32> {
        self.internal.material_id
    }

    /// Set the material ID
    #[setter(id)]
    fn set_id(&mut self, id: u32) {
        self.internal.material_id = Some(id);
    }

    /// Whether this material is marked for depletion by ``Model.simulate_transmutation``.
    ///
    /// Only the coupled model driver reads this: in a geometry of many materials
    /// it selects which ones to deplete. ``Material.transmute()`` ignores it, so
    /// you do not need to set it for a standalone transmutation.
    #[getter]
    fn transmutable(&self) -> bool {
        self.internal.transmutable
    }

    /// Mark this material for depletion by ``Model.simulate_transmutation``.
    ///
    /// Not needed for ``Material.transmute()``.
    #[setter]
    fn set_transmutable(&mut self, transmutable: bool) {
        self.internal.transmutable = transmutable;
    }

    /// Sample a distance to the next neutron collision.
    ///
    /// Uses the macroscopic total cross section (calculating it first if missing)
    /// and an exponential distribution to sample a path length. A deterministic
    /// RNG seed can be supplied for reproducibility.
    ///
    /// Args:
    ///     energy (float): Neutron energy in eV.
    ///     seed (Optional[int]): RNG seed; if omitted a fixed internal seed is used.
    ///
    /// Returns:
    ///     Optional[float]: Sampled distance in cm, or None if total XS unavailable.
    fn sample_distance_to_collision(&self, energy: f64, seed: Option<u64>) -> Option<f64> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(12345),
        };
        self.internal.sample_distance_to_collision(energy, &mut rng)
    }

    /// Create a new material with a declarative composition and density.
    ///
    /// Args:
    ///     composition (dict[str | Nuclide, float | Enriched]): Material composition.
    ///         Keys are element symbols ("Fe"), nuclide names ("Fe56"),
    ///         chemical formulas ("H2O"), or ``yamc.Nuclide`` objects.
    ///         Values are fractions (float) or enriched() objects.
    ///     density (float): Density value, interpreted per ``units``. For
    ///         ``units="atom/barn-cm"`` it is the total atom density and the
    ///         composition values are relative atom fractions.
    ///     units (str): Density units: "g/cm3", "g/cc", "kg/m3", or
    ///         "atom/barn-cm" (total atom density; requires fraction_type="atom").
    ///     name (Optional[str]): Name for the material.
    ///     id (Optional[int]): ID for the material.
    ///     fraction_type (str): "atom" for atom fractions or "mass" for mass fractions.
    ///     transmutable (bool): Mark this material for depletion by
    ///         ``Model.simulate_transmutation``. Not needed for
    ///         ``Material.transmute()``, which depletes whatever it is called on.
    ///     volume (Optional[float]): Material volume in cm³.
    ///     temperature (Optional[float]): Temperature in Kelvin.
    ///
    /// Returns:
    ///     Material: A new material.
    #[new]
    #[pyo3(
        signature = (composition, density, *, units="g/cm3".to_string(), name=None, id=None, fraction_type="atom".to_string(), transmutable=false, volume=None, temperature=None),
        text_signature = "(composition, density, *, units='g/cm3', name=None, id=None, fraction_type='atom', transmutable=False, volume=None, temperature=None)"
    )]
    fn new(
        // `Enriched` is a pyclass in this crate, so it needs no package
        // qualifier and no cross-package import: naming one wheel here made the
        // other wheel's stub depend on a package it does not install.
        #[gen_stub(override_type(type_repr = "dict[str | Nuclide, float | Enriched]"))]
        composition: &Bound<'_, pyo3::types::PyDict>,
        density: f64,
        units: String,
        name: Option<String>,
        id: Option<u32>,
        fraction_type: String,
        transmutable: bool,
        volume: Option<f64>,
        temperature: Option<f64>,
    ) -> PyResult<Self> {
        // Build nuclides map from composition dict using composition module.
        // Also track nuclide insertion order: each user-input key contributes
        // its expanded nuclides (sorted alphabetically within the key) to a
        // Vec, in the order the user listed the keys. Nuclides already seen
        // earlier are skipped to keep names unique.
        let mut nuclides: HashMap<String, f64> = HashMap::new();
        let mut input_order: Vec<String> = Vec::new();

        for (key_obj, value_obj) in composition.iter() {
            let key: String = if let Ok(s) = key_obj.extract::<String>() {
                s
            } else if let Ok(nuc) = key_obj.extract::<pyo3::PyRef<'_, crate::material::PyNuclide>>()
            {
                nuc.name.clone().ok_or_else(|| {
                    PyValueError::new_err("Nuclide used as composition key has no name set")
                })?
            } else {
                return Err(PyValueError::new_err(
                    "composition keys must be str (nuclide/element/formula) or yamc.Nuclide",
                ));
            };

            // A plain fraction, or an Enriched. The latter used to be read by
            // `getattr`, which accepted any object carrying those four
            // attributes and reported a missing one as an AttributeError from
            // inside the constructor.
            let entry_nuclides = if let Ok(frac_val) = value_obj.extract::<f64>() {
                expand_composition_entry(&key, frac_val, &fraction_type, None)?
            } else {
                let spec = value_obj.extract::<super::PyEnriched>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "composition value for '{key}' must be a float or enriched() object, got {}",
                        value_obj.get_type().name().map(|n| n.to_string())
                            .unwrap_or_else(|_| "an unknown type".to_string())
                    ))
                })?;
                let (frac_val, percent, target, etype) = spec.as_entry();

                expand_composition_entry(
                    &key,
                    frac_val,
                    &fraction_type,
                    Some((percent, target, etype)),
                )?
            };

            // Append this entry's nuclides (sorted alphabetically) to the
            // input-order vec, skipping any already-seen names.
            let mut entry_keys: Vec<&String> = entry_nuclides.keys().collect();
            entry_keys.sort();
            for n in entry_keys {
                if !input_order.contains(n) {
                    input_order.push(n.clone());
                }
            }

            yamc_nuclide::composition::merge_nuclides(&mut nuclides, &entry_nuclides);
        }

        // `density` is always a numeric value; its meaning is set by `units`.
        // (The old `density="sum"` sentinel is gone -- pass absolute per-nuclide
        // atom densities via Material.from_atom_densities instead.)
        let mut internal = Material::new(nuclides, &fraction_type, &units, Some(density))
            .map_err(PyValueError::new_err)?;
        internal.nuclide_input_order = Some(input_order);

        // Set optional fields
        if let Some(n) = name {
            internal.name = Some(n);
        }
        if let Some(id) = id {
            internal.material_id = Some(id);
        }
        internal.transmutable = transmutable;
        if let Some(v) = volume {
            internal.volume(Some(v)).map_err(PyValueError::new_err)?;
        }
        if let Some(t) = temperature {
            if !t.is_finite() {
                return Err(PyValueError::new_err("temperature must be a finite number"));
            }
            internal.set_temperature(format!("{}", t));
        }

        Ok(PyMaterial { internal })
    }

    /// Build a material directly from absolute per-nuclide atom densities.
    ///
    /// Stores the given atoms/barn-cm values verbatim (no normalize-then-recombine
    /// round-trip), so callers that already hold raw number densities -- e.g. the
    /// transmutation stepper -- keep full precision. ``density`` then reports the
    /// sum of the values. This replaces the old ``Material(comp, density="sum")``.
    ///
    /// Args:
    ///     atom_densities (dict[str, float]): nuclide name -> atoms/barn-cm.
    ///     name (Optional[str]): Name for the material.
    ///     id (Optional[int]): ID for the material.
    ///     transmutable (bool): Mark this material for depletion by
    ///         ``Model.simulate_transmutation``. Not needed for
    ///         ``Material.transmute()``, which depletes whatever it is called on.
    ///     volume (Optional[float]): Material volume in cm³.
    ///     temperature (Optional[float]): Temperature in Kelvin.
    ///
    /// Returns:
    ///     Material: A material whose composition is absolute atom densities.
    #[staticmethod]
    #[pyo3(
        signature = (atom_densities, *, name=None, id=None, transmutable=false, volume=None, temperature=None),
        text_signature = "(atom_densities, *, name=None, id=None, transmutable=False, volume=None, temperature=None)"
    )]
    fn from_atom_densities(
        atom_densities: &Bound<'_, pyo3::types::PyDict>,
        name: Option<String>,
        id: Option<u32>,
        transmutable: bool,
        volume: Option<f64>,
        temperature: Option<f64>,
    ) -> PyResult<Self> {
        let mut densities: HashMap<String, f64> = HashMap::new();
        let mut input_order: Vec<String> = Vec::new();
        for (key_obj, value_obj) in atom_densities.iter() {
            let nuclide: String = key_obj.extract().map_err(|_| {
                PyValueError::new_err("from_atom_densities keys must be nuclide-name strings")
            })?;
            let value: f64 = value_obj.extract().map_err(|_| {
                PyValueError::new_err(format!("atom density for '{nuclide}' must be a float"))
            })?;
            if !input_order.contains(&nuclide) {
                input_order.push(nuclide.clone());
            }
            densities.insert(nuclide, value);
        }
        if densities.is_empty() {
            return Err(PyValueError::new_err(
                "from_atom_densities requires at least one nuclide",
            ));
        }
        // "sum" mode stores composition values as absolute atom densities.
        let mut internal =
            Material::new(densities, "atom", "sum", None).map_err(PyValueError::new_err)?;
        internal.nuclide_input_order = Some(input_order);
        if let Some(n) = name {
            internal.name = Some(n);
        }
        if let Some(id) = id {
            internal.material_id = Some(id);
        }
        internal.transmutable = transmutable;
        if let Some(v) = volume {
            internal.volume(Some(v)).map_err(PyValueError::new_err)?;
        }
        if let Some(t) = temperature {
            if !t.is_finite() {
                return Err(PyValueError::new_err("temperature must be a finite number"));
            }
            internal.set_temperature(format!("{}", t));
        }
        Ok(PyMaterial { internal })
    }

    /// Get the percent type ("atom" or "mass") for this material's nuclide fractions.
    #[getter]
    fn fraction_type(&self) -> String {
        self.internal.fraction_type.as_str().to_string()
    }

    /// Get the material nuclides as a tuple of (name, fraction) pairs.
    ///
    /// When the Material was constructed via the Python API, the order matches
    /// the user's composition dict (with each input key's expanded nuclides
    /// sorted alphabetically). Nuclides added later (e.g. by transmutation) appear
    /// alphabetically at the end. When no input order is tracked, all nuclides
    /// are alphabetical.
    #[getter]
    fn nuclides(&self) -> Vec<(String, f64)> {
        if let Some(order) = &self.internal.nuclide_input_order {
            let mut out: Vec<(String, f64)> = Vec::with_capacity(self.internal.nuclides.len());
            // First, listed nuclides in user-input order
            for n in order {
                if let Some(v) = self.internal.nuclides.get(n) {
                    out.push((n.clone(), *v));
                }
            }
            // Then any nuclides not in input_order (e.g. from transmutation), alphabetical
            let mut extras: Vec<(String, f64)> = self
                .internal
                .nuclides
                .iter()
                .filter(|(k, _)| !order.contains(k))
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            extras.sort_by(|a, b| a.0.cmp(&b.0));
            out.extend(extras);
            return out;
        }

        // No input order tracked: alphabetical
        let mut nuclide_vec: Vec<(String, f64)> = self
            .internal
            .nuclides
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        nuclide_vec.sort_by(|a, b| a.0.cmp(&b.0));

        nuclide_vec
    }

    /// Material volume in cm^3, if set.
    #[getter]
    fn volume(&self) -> Option<f64> {
        self.internal.volume
    }

    /// Set the material volume (cm^3).
    #[setter]
    fn set_volume(&mut self, value: f64) -> PyResult<()> {
        self.internal
            .volume(Some(value))
            .map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Return a list of nuclide names currently present in the material.
    #[pyo3(text_signature = "(self)")]
    fn get_nuclide_names(&self) -> Vec<String> {
        self.internal.get_nuclides()
    }

    /// String representation of the Material
    fn __str__(&self) -> PyResult<String> {
        let mut result = String::from("Material:\n");

        // Add density information
        if let Some(density) = self.internal.density {
            result.push_str(&format!(
                "  Density: {} {}\n",
                density,
                self.internal.density_units.as_str()
            ));
        } else {
            result.push_str("  Density: not set\n");
        }

        // Add volume information
        if let Some(volume) = self.internal.volume {
            result.push_str(&format!("  Volume: {} cm³\n", volume));
        } else {
            result.push_str("  Volume: not set\n");
        }

        // Add nuclide information
        result.push_str("  Composition:\n");
        for (nuclide, fraction) in &self.internal.nuclides {
            result.push_str(&format!("    {}: {}\n", nuclide, fraction));
        }

        Ok(result)
    }

    /// Return the same string as __str__
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }

    /// Rich Jupyter display: material metadata (id, density, temperature,
    /// volume) beside the per-nuclide composition table.
    fn _repr_html_(&self) -> String {
        use crate::html_repr::{card, esc, kv, num, table};
        let name = self
            .internal
            .name
            .clone()
            .unwrap_or_else(|| "Material".to_string());

        let mut meta: Vec<(&str, String)> = Vec::new();
        if let Some(id) = self.internal.material_id {
            meta.push(("id", format!("{id}")));
        }
        match self.internal.density {
            Some(d) => meta.push((
                "density",
                format!("{} {}", num(d), esc(self.internal.density_units.as_str())),
            )),
            None => meta.push((
                "density",
                "<em style=\"color:#656d76;\">not set</em>".to_string(),
            )),
        }
        let temp = self.internal.temperature().to_string();
        if !temp.is_empty() {
            meta.push(("temperature", format!("{} K", esc(&temp))));
        }
        if let Some(v) = self.internal.volume {
            meta.push(("volume", format!("{} cm³", num(v))));
        }

        let mut comp_rows: Vec<Vec<String>> = Vec::new();
        for (nuclide, fraction) in &self.internal.nuclides {
            comp_rows.push(vec![esc(&nuclide.to_string()), format!("{fraction}")]);
        }
        let comp = if comp_rows.is_empty() {
            "<em style=\"color:#656d76;\">no composition</em>".to_string()
        } else {
            table(&["nuclide", "fraction"], &comp_rows, 1)
        };

        let body = format!(
            "<div style=\"display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start;\">\
             <div>{}</div><div>{comp}</div></div>",
            kv(&meta)
        );
        card(&format!("Material: {name}"), "", &body)
    }

    /// Density value, in the material's ``units``. Always a float: for
    /// ``"atom/barn-cm"`` (and transmuted materials) it is the total atom
    /// density, otherwise the mass density.
    #[getter]
    fn density(&self) -> PyResult<f64> {
        match self.internal.density {
            Some(d) => Ok(d),
            // Raw per-nuclide ("sum") materials store no scalar density; report
            // the total atom density (the sum of the per-nuclide values).
            None => self
                .internal
                .total_atom_density()
                .map_err(PyValueError::new_err),
        }
    }

    /// Density units string ("g/cm3", "kg/m3", or "atom/barn-cm").
    #[getter]
    fn density_units(&self) -> String {
        match self.internal.density_units {
            // The internal raw-per-nuclide representation (from transmutation)
            // is reported with the public atom-density unit, never bare "sum".
            yamc_materials::DensityUnits::Sum => "atom/barn-cm".to_string(),
            other => other.as_str().to_string(),
        }
    }

    #[pyo3(name = "read_nuclear_data")]
    /// Bulk load nuclear data from a mapping of nuclide -> file path or a keyword string.
    ///
    /// Args:
    ///     nuclide_path_map (Optional[Dict[str,str] | str]): Mapping of nuclide names to file paths or a keyword string.
    ///     photon_data (Optional[Dict[str,str]]): Mapping of element symbols to photon Arrow data paths.
    ///
    /// Raises:
    ///     ValueError: If any file cannot be read / parsed.
    #[pyo3(signature = (nuclide_path_map=None, photon_data=None))]
    fn read_nuclear_data(
        &mut self,
        _py: Python,
        nuclide_path_map: Option<&Bound<'_, pyo3::types::PyAny>>,
        photon_data: Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        // Extract Python data to Rust types
        let (dict_data, keyword_data) = if let Some(obj) = nuclide_path_map {
            if obj.is_instance_of::<pyo3::types::PyDict>() {
                let d = obj.cast::<pyo3::types::PyDict>()?;
                let mut rust_map = HashMap::new();
                for (k, v) in d.iter() {
                    rust_map.insert(k.extract::<String>()?, v.extract::<String>()?);
                }
                (Some(rust_map), None)
            } else if obj.is_instance_of::<pyo3::types::PyString>() {
                let keyword: String = obj.extract()?;
                (None, Some(keyword))
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "nuclide_path_map must be a dict or a str keyword",
                ));
            }
        } else {
            (None, None)
        };

        // Call pure Rust function
        self.internal
            .load_nuclear_data_from_input(dict_data, keyword_data, photon_data)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Return raw pointer address of an internal shared Nuclide (debug only).
    ///
    /// Args:
    ///     nuclide (str): Nuclide name.
    ///
    /// Returns:
    ///     Optional[int]: Address value (process-local) or None if not present.
    fn nuclide_ptr_addr(&self, nuclide: &str) -> Option<usize> {
        self.internal.nuclide_data.get(nuclide).map(|arc| {
            let ptr: *const yamc_nuclide::nuclide::Nuclide = std::sync::Arc::as_ptr(arc);
            ptr as usize
        })
    }

    /// Temperature label in Kelvin (e.g. "293").
    #[getter]
    fn temperature(&self) -> String {
        self.internal.temperature().to_string()
    }

    /// Set current temperature label.
    ///
    /// Takes a number or a label. The label form matters because the getter
    /// returns one: without it ``m.temperature = m.temperature`` raised
    /// ``ValueError``, which is a defect on its own and becomes a sharper one
    /// now that a non-integer temperature is legitimate input. Both spellings
    /// name one temperature, since the core normalises the ``K`` suffix away.
    #[setter]
    fn set_temperature(&mut self, temperature: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        if let Ok(value) = temperature.extract::<i64>() {
            self.internal.set_temperature(value.to_string());
            return Ok(());
        }
        if let Ok(value) = temperature.extract::<f64>() {
            if !value.is_finite() {
                return Err(PyValueError::new_err("temperature must be a finite number"));
            }
            self.internal.set_temperature(format!("{}", value));
            return Ok(());
        }
        // After the numeric arms, so a Python float still takes the float path
        // and is formatted the way the constructor formats it.
        if let Ok(label) = temperature.extract::<String>() {
            self.internal.set_temperature(label);
            return Ok(());
        }

        Err(PyValueError::new_err(
            "temperature must be a number or a temperature label such as '294' or '294K'",
        ))
    }

    /// Return (and build if needed) the unified neutron energy grid.
    ///
    /// Returns:
    ///     List[float]: Energy grid in eV.
    #[pyo3(text_signature = "(self)")]
    fn unified_energy_grid_neutron(&mut self) -> Vec<f64> {
        self.internal.unified_energy_grid_neutron()
    }

    /// Calculate microscopic neutron cross sections on the unified energy grid.
    ///
    /// Args:
    ///     mt_filter (Optional[List[int]]): Restrict to these MT numbers.
    ///
    /// Returns:
    ///     Dict[str, Dict[int, List[float]]]: nuclide -> MT -> xs array
    #[pyo3(signature = (mt_filter=None))]
    fn calculate_microscopic_xs_neutron(
        &mut self,
        mt_filter: Option<Vec<i32>>,
    ) -> HashMap<String, HashMap<i32, Vec<f64>>> {
        self.internal
            .calculate_microscopic_xs_neutron(mt_filter.as_ref())
    }

    /// Number density (atoms / barn-cm) per nuclide.
    ///
    /// Returns:
    ///     Dict[str, float]: nuclide -> atoms / b-cm
    fn get_atoms_per_barn_cm(&self) -> PyResult<HashMap<String, f64>> {
        self.internal
            .get_atoms_per_barn_cm()
            .map_err(PyValueError::new_err)
    }

    /// Calculate the radioactive activity of the current material inventory.
    ///
    /// The material must have a ``volume`` in cm³. Atom densities are converted
    /// to total atoms using ``atoms/barn-cm * 1e24 * volume``. The supplied
    /// transmutation chain provides half-lives; each nuclide's activity is
    /// ``N * ln(2) / half_life``. Stable nuclides are omitted.
    ///
    /// Args:
    ///     by_nuclide (bool): Return a ``dict[str, float]`` by nuclide instead
    ///         of the total.
    ///     per (str | None): ``None`` (default) for the total, which needs
    ///         ``volume``; ``"cm3"`` for Bq/cm³, which needs nothing; ``"g"``
    ///         for Bq/g, which needs only ``density``.
    ///
    /// Returns:
    ///     float | dict[str, float]: Activity, in Bq when ``per`` is ``None``,
    ///     Bq/cm³ when it is ``"cm3"`` and Bq/g when it is ``"g"``. The unit of
    ///     a ``by_nuclide`` dict's values is the same.
    #[pyo3(signature = (*, by_nuclide=false, per=None))]
    fn activity(&self, py: Python<'_>, by_nuclide: bool, per: Option<&str>) -> PyResult<Py<PyAny>> {
        let volume = self.scale_for(per, "activity")?;
        let atom_densities = self
            .internal
            .get_atoms_per_barn_cm()
            .map_err(PyValueError::new_err)?;
        let chain = crate::distribution::resolve_chain()?.chain;

        let activities = yani_decay::activity_by_nuclide(&atom_densities, volume, &chain);

        if by_nuclide {
            let breakdown = PyDict::new(py);
            for (nuclide, activity) in &activities {
                breakdown.set_item(nuclide, activity)?;
            }
            Ok(breakdown.into())
        } else {
            let total = yani_decay::total(&activities);
            Ok(total.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Calculate decay heat from the current material inventory.
    ///
    /// The material must have a ``volume`` in cm³. Atom densities are converted
    /// to total atoms using ``atoms/barn-cm * 1e24 * volume``. The supplied
    /// transmutation chain provides half-lives and mean decay energies.
    ///
    /// Args:
    ///     by_nuclide (bool): Return a ``dict[str, float]`` by nuclide instead
    ///         of the total.
    ///     per (str | None): ``None`` (default) for the total, which needs
    ///         ``volume``; ``"cm3"`` for W/cm³, which needs nothing; ``"g"``
    ///         for W/g, which needs only ``density``.
    ///
    /// Returns:
    ///     float | dict[str, float]: Decay heat, in W when ``per`` is ``None``,
    ///     W/cm³ when it is ``"cm3"`` and W/g when it is ``"g"``. The unit of a
    ///     ``by_nuclide`` dict's values is the same.
    #[pyo3(signature = (*, by_nuclide=false, per=None))]
    fn decay_heat(
        &self,
        py: Python<'_>,
        by_nuclide: bool,
        per: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let volume = self.scale_for(per, "decay_heat")?;
        let atom_densities = self
            .internal
            .get_atoms_per_barn_cm()
            .map_err(PyValueError::new_err)?;
        let chain = crate::distribution::resolve_chain()?.chain;

        let heats = yani_decay::decay_heat_by_nuclide(&atom_densities, volume, &chain);

        if by_nuclide {
            let breakdown = PyDict::new(py);
            for (nuclide, heat) in &heats {
                breakdown.set_item(nuclide, heat)?;
            }
            Ok(breakdown.into())
        } else {
            let total = yani_decay::total(&heats);
            Ok(total.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// The decay photon line spectrum of the current inventory.
    ///
    /// The material must have a ``volume`` in cm³. Each nuclide's atom count
    /// multiplies the line intensities the transmutation chain records for it,
    /// which are per atom per second (the emission probability already times
    /// the decay constant), so the values are photons per second rather than
    /// probabilities. Lines at the same energy are summed and the result is
    /// ascending in energy. Stable nuclides, nuclides the chain does not know
    /// and non-photon sources contribute nothing.
    ///
    /// The two lists are the ``(x, p)`` pair the source distributions take, so
    /// the spectrum round-trips straight into a photon transport run:
    ///
    ///     >>> energies, rates = activated.decay_photon_spectrum()
    ///     >>> source = PhotonSource(energy=sources.Discrete(energies, rates))
    ///
    /// ``Discrete`` normalizes the weights, so the shape is what transport
    /// samples; keep ``sum(rates)`` yourself for the absolute emission rate
    /// (photons/s) that scales the tallies.
    ///
    /// Returns:
    ///     tuple[list[float], list[float]]: Line energies (eV) and their
    ///     emission rates (photons/s).
    #[pyo3(signature = (*, per=None))]
    fn decay_photon_spectrum(&self, per: Option<&str>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let volume = self.scale_for(per, "decay_photon_spectrum")?;
        let atom_densities = self
            .internal
            .get_atoms_per_barn_cm()
            .map_err(PyValueError::new_err)?;
        let chain = crate::distribution::resolve_chain()?.chain;
        let lines = yani_decay::decay_photon_lines(&atom_densities, volume, &chain);
        Ok(lines.into_iter().unzip())
    }

    /// Contact dose rate from the material's own decay photons.
    ///
    /// The dose someone receives with a hand on the material. It is a slab
    /// estimate rather than a transport result: the material is taken to be a
    /// half-space, so half the photons emitted at any depth head for the
    /// surface and the ones that arrive are the ones the material did not
    /// attenuate itself. That leaves no distance and no volume in the answer --
    /// unlike ``activity`` and ``decay_heat``, this needs no ``material.volume``
    /// and a bigger lump of the same material reads the same.
    ///
    /// Per photon line of energy ``E`` and per-atom emission rate ``S`` the
    /// estimate is ``(build_up / 2) * (response(E) / mu_material(E)) * S * E``
    /// for the absorbed dose in air, and the same without the trailing ``E``
    /// for the effective dose. ``mu_material`` is the material's own linear
    /// attenuation coefficient, built from the NIST XCOM mass attenuation
    /// coefficients of the elements present; the response is the NIST-126 mass
    /// energy-absorption coefficient of air, or the ICRP-116 photon
    /// effective-dose coefficient for anterior-posterior irradiation.
    ///
    /// Follows the FISPACT-II manual (UKAEA-CCFE-RE(21)02, Appendix C.7.1) for
    /// the absorbed-air quantity and agrees with OpenMC's
    /// ``Material.get_photon_contact_dose_rate``.
    ///
    /// Bremsstrahlung from decay electrons is not modelled, and nuclides whose
    /// radiation the chain file does not describe contribute nothing. Photon
    /// lines outside the tabulated range (1 keV to 20 MeV for the absorbed-air
    /// quantity, 10 keV to 20 MeV for the effective dose) are dropped.
    ///
    /// Args:
    ///     dose_quantity (str): ``'absorbed-air'`` for the absorbed dose in air
    ///         in Gy/h (the default, following FISPACT-II), or ``'effective'``
    ///         for the ICRP-116 effective dose in Sv/h.
    ///     build_up (float): Build-up factor standing in for the photons that
    ///         scatter in the slab and still arrive. Default 2.0, as the
    ///         FISPACT-II manual suggests.
    ///     by_nuclide (bool): Return a ``dict[str, float]`` by nuclide instead
    ///         of the total. Nuclides that contribute nothing are left out.
    ///
    /// Returns:
    ///     float | dict[str, float]: Contact dose rate in Gy/h
    ///     (``'absorbed-air'``) or Sv/h (``'effective'``).
    ///
    /// Examples:
    ///     >>> activated.contact_dose()
    ///     380.088...
    ///     >>> activated.contact_dose(by_nuclide=True)
    ///     {'Co60': 380.088...}
    ///     >>> activated.contact_dose(dose_quantity='effective')
    ///     380.728...
    #[pyo3(signature = (*, dose_quantity="absorbed-air", build_up=2.0, by_nuclide=false))]
    fn contact_dose(
        &self,
        py: Python<'_>,
        dose_quantity: &str,
        build_up: f64,
        by_nuclide: bool,
    ) -> PyResult<Py<PyAny>> {
        let quantity = match dose_quantity {
            "absorbed-air" => yani_decay::DoseQuantity::AbsorbedAir,
            "effective" => yani_decay::DoseQuantity::Effective,
            other => {
                return Err(PyValueError::new_err(format!(
                    "dose_quantity must be 'absorbed-air' or 'effective', got '{other}'"
                )))
            }
        };
        let atom_densities = self
            .internal
            .get_atoms_per_barn_cm()
            .map_err(PyValueError::new_err)?;
        let chain = crate::distribution::resolve_chain()?.chain;

        if by_nuclide {
            let doses =
                yani_decay::contact_dose_by_nuclide(&atom_densities, &chain, quantity, build_up)
                    .map_err(PyValueError::new_err)?;
            let breakdown = PyDict::new(py);
            for (nuclide, dose) in &doses {
                breakdown.set_item(nuclide, dose)?;
            }
            Ok(breakdown.into())
        } else {
            let total = yani_decay::contact_dose_total(&atom_densities, &chain, quantity, build_up)
                .map_err(PyValueError::new_err)?;
            Ok(total.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Compute neutron mean free path at a given energy.
    ///
    /// Args:
    ///     energy (float): Neutron energy in eV.
    ///
    /// Returns:
    ///     Optional[float]: Mean free path (cm) or None if total XS unavailable.
    fn mean_free_path_neutron(&mut self, energy: f64) -> Option<f64> {
        self.internal.mean_free_path_neutron(energy)
    }

    /// Sorted list of all unique MT reaction numbers present.
    #[getter]
    fn reaction_mts(&mut self) -> PyResult<Vec<i32>> {
        self.internal
            .reaction_mts()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Sample which nuclide undergoes an interaction at a given energy.
    ///
    /// Uses per-nuclide macroscopic total cross sections as weights.
    ///
    /// Args:
    ///     energy (float): Neutron energy in eV.
    ///     seed (Optional[int]): RNG seed for reproducibility.
    ///
    /// Returns:
    ///     str: Selected nuclide name.
    #[pyo3(signature = (energy, seed=None))]
    fn sample_interacting_nuclide(&self, energy: f64, seed: Option<u64>) -> PyResult<String> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(12345),
        };
        Ok(self.internal.sample_interacting_nuclide(energy, &mut rng))
    }

    /// Calculate macroscopic cross section for a specific reaction.
    ///
    /// This method accepts either an integer MT number or a string reaction name
    /// (like "(n,gamma)", "fission", etc.) and returns the macroscopic cross section
    /// for that reaction.
    ///
    /// Args:
    ///     reaction (Union[int, str]): Either an MT number or reaction name.
    ///     temperature (Optional[str]): Temperature key (e.g., "294"). If not provided,
    ///         uses material.temperature, or the only available temperature if there's just one.
    ///
    /// Returns:
    ///     Tuple[List[float], List[float]]: (cross_section_values, energy_grid)
    #[pyo3(text_signature = "(self, reaction, temperature=None)")]
    #[pyo3(signature = (reaction, temperature=None))]
    fn macroscopic_cross_section(
        &mut self,
        #[gen_stub(override_type(type_repr = "builtins.int | builtins.str"))] reaction: &Bound<
            '_,
            pyo3::PyAny,
        >,
        temperature: Option<String>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let temp_ref = temperature.as_deref();
        // Handle both int and str inputs
        if let Ok(mt_number) = reaction.extract::<i32>() {
            // Integer MT number
            Ok(self.internal.macroscopic_cross_section(mt_number, temp_ref))
        } else if let Ok(reaction_name) = reaction.extract::<String>() {
            // String reaction name
            Ok(self
                .internal
                .macroscopic_cross_section(reaction_name, temp_ref))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "reaction must be an integer MT number or a string reaction name",
            ))
        }
    }

    /// Transmute this material over an irradiation/cooling schedule, without
    /// re-running transport.
    ///
    /// Each irradiation ``Pulse`` carries a ``NeutronSource`` whose energy is a
    /// ``Histogram`` (the multigroup spectrum); the pulse ``rate`` is the total
    /// flux magnitude [n/cm²/s]. The spectrum collapses cross sections into
    /// one-group reaction rates, scaled by ``rate``. A ``Cooldown`` is a
    /// decay-only step. Different pulses may carry different spectra (e.g. a DD
    /// and a DT campaign).
    ///
    /// Args:
    ///     schedule (PulseSchedule): The irradiation/cooling timeline. Every
    ///         irradiation ``Pulse`` must carry a ``NeutronSource`` whose energy
    ///         is a ``Histogram``; its ``rate`` is the flux magnitude [n/cm²/s].
    ///         The source's position/direction are ignored (no transport runs).
    ///         Use ``Cooldown`` for decay-only steps.
    ///
    /// The transmutation network is assembled from the configured per-subsection
    /// sources (``yamc.transmutation_decay_data`` / ``_reactions`` /
    /// ``_fission_yields``); decay-only schedules need only the decay source.
    ///
    /// The material does **not** need ``transmutable=True``. That flag exists so
    /// ``Model.simulate_transmutation`` can pick which materials in a geometry to
    /// deplete; here you have already named the one you mean.
    ///
    /// The cross sections this needs are loaded **into the material** and kept,
    /// so calling it again on the same material does no file reading at all --
    /// on a steel against ENDF/B-8.1 that is 1.8 s of a 2.3 s call. The
    /// composition is not modified. A sweep over thousands of *distinct*
    /// compositions should call ``release_nuclear_data()`` on each material when
    /// it is done with it, since otherwise every one of them holds a few hundred
    /// nuclides' data for as long as it lives.
    ///
    /// Returns:
    ///     TransmutationResults: The same object ``Model.simulate_transmutation``
    ///         returns, keyed by this material's ``id`` (0 if it has none).
    ///         ``get_material(id, step)`` takes step 0 as the initial
    ///         composition and step ``i`` as the state after schedule step
    ///         ``i - 1``; ``get_reaction_rates(id, step)`` indexes the schedule
    ///         steps directly, so its step 0 is the first step.
    ///
    ///     data_uncertainty (DataUncertainty, optional): Ask for nuclear-data
    ///         uncertainty on the result. The activation cross sections are
    ///         sampled from their ENDF MF=33 covariance, folded against this
    ///         material's own spectrum, and the schedule is re-solved until the
    ///         reported standard deviations settle. Omit it (the default) and
    ///         nothing is read, folded or sampled: the inventories are
    ///         bit-identical either way. Read the sigmas with
    ///         ``get_nuclide_uncertainty``, and what was and was not covered
    ///         with ``data_uncertainty_info``.
    ///
    ///     self_shielding_chord (float, optional): Mean chord length ``4V/S`` of
    ///         this material's lump, in cm, which is twice the thickness for a
    ///         thin slab and ``4R/3`` for a sphere. Given one, the flux inside
    ///         each energy group is depressed where the total cross section is
    ///         large, and the reaction rates come back shielded. Omit it (the
    ///         default) and nothing is shielded: the rates are bit-identical to
    ///         a run without it, and no shape is inferred from a geometry this
    ///         material does not have. Read what was done with
    ///         ``self_shielding_info``.
    ///
    ///         The flux inside the lump comes from a slowing-down solve, which
    ///         assumes nothing about resonances being narrow. The cheaper
    ///         narrow-resonance approximation is deliberately not offered: it
    ///         over-shields strong elastic scatterers badly enough to be worse
    ///         than applying no correction at all (W186(n,gamma) in FNG-tung
    ///         measures 1.29 b, where this gives C/E 0.80, no correction gives
    ///         0.86 and narrow resonance gives 0.46).
    ///
    ///     self_shielding_shape (SphereLump | CubeLump | FoilLump | CylinderLump | WireLump, optional):
    ///         The lump's form, from ``shapes``, as an alternative to stating the
    ///         chord yourself. The chord is ``4V/S``, so the shape decides how
    ///         much surface a volume hides behind: ``SphereLump()`` and ``CubeLump()``
    ///         are fixed by ``Material(volume=...)`` and take no arguments,
    ///         while ``FoilLump(thickness=)``, ``CylinderLump(radius=)``
    ///         and ``WireLump(radius=)`` carry the one dimension a volume cannot imply.
    ///         Give this or ``self_shielding_chord``, not both.
    ///
    ///         A sphere has the least surface for its volume, so it has the
    ///         longest chord and shields more than any other shape of the same
    ///         size. It is an upper bound rather than a safe default, which is
    ///         why there is no default here at all.
    ///
    /// Raises:
    ///     ValueError: If an irradiation pulse lacks a NeutronSource, its energy
    ///         is not a Histogram, or on other invalid inputs.
    ///
    /// Examples:
    ///     >>> spectrum = yamc.NeutronSource(energy=yamc.sources.Histogram("CCFE-709", flux_709))
    ///     >>> sched = yamc.PulseSchedule([
    ///     ...     yamc.Pulse(rate=1e14, duration=(1, "h"), source=spectrum),
    ///     ...     yamc.Cooldown(duration=(7, "d")),
    ///     ... ])
    ///     >>> results = iron.transmute(schedule=sched)
    ///     >>> final = results.get_final_material(iron.id or 0)
    ///     >>> # the isomeric split of the dominant channel, as two edges of one
    ///     >>> # reaction: the sum is sigma * phi * N and the ratio is f_m
    ///     >>> results.get_reaction_rates(iron.id or 0, 0)["W186"]["(n,2n)"]
    ///     [('W185', 1.1e-08), ('W185_m1', 4.6e-07)]
    #[pyo3(signature = (schedule, data_uncertainty = None, self_shielding_chord = None, self_shielding_shape = None))]
    fn transmute(
        &mut self,
        py: Python<'_>,
        schedule: &Bound<'_, pyo3::types::PyAny>,
        data_uncertainty: Option<crate::data_uncertainty::PyDataUncertainty>,
        self_shielding_chord: Option<f64>,
        self_shielding_shape: Option<Bound<'_, PyAny>>,
    ) -> PyResult<crate::transmutation_results::PyTransmutationResults> {
        let sched = schedule
            .cast::<crate::distribution::PyPulseSchedule>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("schedule must be a PulseSchedule")
            })?;
        let sched = sched.borrow();
        // Each irradiation pulse carries its spectrum (a Histogram-energy
        // NeutronSource) and a magnitude (rate); cooldowns are decay-only.
        let (spectra, steps) = sched.transmute_plan(py)?;

        let loaded = crate::distribution::resolve_chain()?;

        // A chord or a shape, never both: they would be two statements of the
        // same length, and nothing good comes of deciding which one wins.
        let chord = match (self_shielding_chord, self_shielding_shape.as_ref()) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "give self_shielding_chord or self_shielding_shape, not both: a shape \
                     already determines the chord",
                ))
            }
            (Some(chord), None) => Some(chord),
            (None, Some(shape)) => {
                let shape = crate::shapes::shape_of(shape)?;
                Some(
                    shape
                        .chord_cm(self.internal.volume)
                        .map_err(PyValueError::new_err)?,
                )
            }
            (None, None) => None,
        };
        let shielding = match chord {
            Some(chord) => {
                Some(yani_transmute::Shielding::new(chord).map_err(PyValueError::new_err)?)
            }
            None => None,
        };

        // Everything the solve needs, owned and free of the GIL, before it is
        // released. `sched` is a `PyRef` and must go first; the chain is three
        // `Arc`s, and the uncertainty request is a plain struct (issue #576,
        // finding 7).
        //
        // `Model.transmute` has released the GIL for as long as it has existed
        // and this was the outlier: one `Material.transmute` froze the whole
        // interpreter for the length of the solve, so a Python-level
        // `ThreadPoolExecutor` over cases could not overlap them at all.
        drop(sched);
        let uncertainty = data_uncertainty.map(|u| u.inner);
        let material = &mut self.internal;
        let results = py.detach(move || {
            yani_transmute::transmute_material_shielded(
                material,
                &spectra,
                &steps,
                loaded.chain,
                &loaded.branch,
                loaded.parts,
                uncertainty.as_ref(),
                shielding.as_ref(),
            )
            // `Box<dyn Error>` is not `Send`, so it cannot come back out
            // through `detach`; the message is what the caller sees anyway.
            .map_err(|e| e.to_string())
        });
        let results = results.map_err(PyValueError::new_err)?;

        Ok(crate::transmutation_results::PyTransmutationResults { inner: results })
    }

    /// Drop the nuclear data this material has loaded.
    ///
    /// ``transmute()`` loads the cross sections it needs into the material and
    /// leaves them there, so calling it again on the same material does no file
    /// reading at all. On an ENDF/B-8.1 chain that is a few hundred nuclides
    /// held for as long as the material is.
    ///
    /// That is the right trade for a material you transmute more than once, and
    /// the wrong one for a sweep over thousands of *distinct* compositions --
    /// so releasing is explicit rather than automatic. The material stays
    /// usable: the next call that needs the data loads it again.
    ///
    /// Examples:
    ///     >>> for composition in many_compositions:
    ///     ...     mat = yamc.Material(composition=composition, density=7.9)
    ///     ...     results = mat.transmute(schedule=sched)
    ///     ...     mat.release_nuclear_data()
    #[pyo3(text_signature = "(self)")]
    fn release_nuclear_data(&mut self) {
        self.internal.release_nuclear_data();
    }

    /// Return the average molar mass of this material (g/mol).
    #[pyo3(text_signature = "(self)")]
    fn average_molar_mass(&self) -> PyResult<f64> {
        self.internal
            .average_molar_mass()
            .map_err(PyValueError::new_err)
    }

    /// Return the mass density in g/cm³ regardless of stored units.
    #[pyo3(text_signature = "(self)")]
    fn get_mass_density(&self) -> PyResult<f64> {
        self.internal
            .get_mass_density()
            .map_err(PyValueError::new_err)
    }

    /// Create a new material by mixing multiple materials.
    ///
    /// Args:
    ///     materials (List[Material]): Materials to mix.
    ///     fractions (List[float]): Mixing fractions (one per material).
    ///     fraction_type (str): "atom", "mass", or "volume".
    ///     name (Optional[str]): Name for the new material.
    ///     id (Optional[int]): ID for the new material.
    ///
    /// Returns:
    ///     Material: A new material with the mixed composition and density.
    #[classmethod]
    #[pyo3(
        signature = (materials, fractions, fraction_type="atom".to_string(), *, name=None, id=None),
        text_signature = "(materials, fractions, fraction_type='atom', *, name=None, id=None)"
    )]
    fn mix_materials(
        _cls: &Bound<'_, PyType>,
        materials: Vec<PyRef<'_, PyMaterial>>,
        fractions: Vec<f64>,
        fraction_type: String,
        name: Option<String>,
        id: Option<u32>,
    ) -> PyResult<Self> {
        let mat_refs: Vec<&Material> = materials.iter().map(|m| &m.internal).collect();
        let result = Material::mix_materials(&mat_refs, &fractions, &fraction_type, name, id)
            .map_err(PyValueError::new_err)?;
        Ok(PyMaterial { internal: result })
    }
}
