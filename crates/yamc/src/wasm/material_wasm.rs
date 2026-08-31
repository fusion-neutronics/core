use js_sys::{Array, Float64Array, Map};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use yamc_materials::material::{DensityUnits, Material};

#[wasm_bindgen]
pub struct WasmMaterial {
    inner: Option<Material>,
    // Accumulation state for lazy construction
    nuclides: HashMap<String, f64>,
    fraction_type: String,
    density_unit: String,
    density_value: Option<f64>,
}

impl WasmMaterial {
    /// Ensure the Material is built from accumulated state.
    fn ensure_built(&mut self) -> Result<(), JsValue> {
        if self.inner.is_some() {
            return Ok(());
        }
        let mat = Material::new(
            self.nuclides.clone(),
            &self.fraction_type,
            &self.density_unit,
            self.density_value,
        )
        .map_err(|e| JsValue::from_str(&e))?;
        self.inner = Some(mat);
        Ok(())
    }

    fn inner_ref(&self) -> Result<&Material, JsValue> {
        self.inner
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Material not yet built -- call set_density first"))
    }

    fn inner_mut(&mut self) -> Result<&mut Material, JsValue> {
        self.inner
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Material not yet built -- call set_density first"))
    }

    pub fn ensure_nuclides_loaded(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

#[wasm_bindgen]
impl WasmMaterial {
    // No Default impl to go with this. `new` here is the JS `new WasmMaterial()`
    // constructor, and JS has no way to reach a Rust Default, so the impl would
    // be unreachable code added only to quiet the lint.
    #[allow(clippy::new_without_default)]
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        WasmMaterial {
            inner: None,
            nuclides: HashMap::new(),
            fraction_type: "atom".to_string(),
            density_unit: "g/cm3".to_string(),
            density_value: None,
        }
    }

    #[wasm_bindgen]
    pub fn add_nuclide(
        &mut self,
        nuclide: &str,
        fraction: f64,
        fraction_type: Option<String>,
    ) -> Result<(), JsValue> {
        if let Some(ref ft) = fraction_type {
            self.fraction_type = ft.clone();
        }
        yamc_nuclide::composition::validate_nuclide_name(nuclide)
            .map_err(|e| JsValue::from_str(&e))?;
        self.nuclides.insert(nuclide.to_string(), fraction);
        // Invalidate built material so it gets rebuilt
        self.inner = None;
        Ok(())
    }

    #[wasm_bindgen]
    pub fn add_element(
        &mut self,
        element: &str,
        fraction: f64,
        fraction_type: Option<String>,
        enrichment: Option<f64>,
        enrichment_target: Option<String>,
        enrichment_type: Option<String>,
    ) -> Result<(), JsValue> {
        if let Some(ref ft) = fraction_type {
            self.fraction_type = ft.clone();
        }
        let new_nuclides = match (enrichment, enrichment_target.as_deref()) {
            (Some(e), Some(t)) => {
                let et = enrichment_type.as_deref().unwrap_or("atom");
                yamc_nuclide::composition::expand_element_enriched(
                    element,
                    fraction,
                    e,
                    t,
                    et,
                    &self.fraction_type,
                )
                .map_err(|e| JsValue::from_str(&e))?
            }
            (None, None) => {
                yamc_nuclide::composition::expand_element(element, fraction, &self.fraction_type)
                    .map_err(|e| JsValue::from_str(&e))?
            }
            _ => {
                return Err(JsValue::from_str(
                    "enrichment and enrichment_target must both be provided or both omitted",
                ))
            }
        };
        yamc_nuclide::composition::merge_nuclides(&mut self.nuclides, &new_nuclides);
        self.inner = None;
        Ok(())
    }

    #[wasm_bindgen]
    pub fn set_density(&mut self, unit: &str, value: Option<f64>) -> Result<(), JsValue> {
        self.density_unit = unit.to_string();
        self.density_value = value;
        // Force rebuild with new density
        self.inner = None;
        self.ensure_built()
    }

    #[wasm_bindgen]
    pub fn set_volume(&mut self, value: f64) -> Result<(), JsValue> {
        self.ensure_built()?;
        self.inner_mut()?
            .volume(Some(value))
            .map_err(|e| JsValue::from_str(&e))
            .map(|_| ())
    }

    #[wasm_bindgen]
    pub fn set_temperature(&mut self, temperature: &str) {
        if let Some(ref mut mat) = self.inner {
            mat.set_temperature(temperature);
        }
    }

    #[wasm_bindgen]
    pub fn get_nuclides(&self) -> Array {
        let nuclides: Vec<String> = if let Some(ref mat) = self.inner {
            mat.get_nuclides()
        } else {
            let mut keys: Vec<String> = self.nuclides.keys().cloned().collect();
            keys.sort();
            keys
        };
        nuclides
            .into_iter()
            .map(|n| JsValue::from_str(&n))
            .collect::<Array>()
    }

    #[wasm_bindgen]
    pub fn get_atoms_per_barn_cm(&mut self) -> Result<JsValue, JsValue> {
        self.ensure_built()?;
        let inner = self.inner_ref()?;

        if inner.density.is_none() && inner.density_units != DensityUnits::Sum {
            return Err(JsValue::from_str(
                "Cannot calculate atoms per cc: Material has no density defined",
            ));
        }

        if inner.nuclides.is_empty() {
            return Err(JsValue::from_str(
                "Cannot calculate atoms per cc: Material has no nuclides defined",
            ));
        }

        let atoms_per_barn_cm = inner
            .get_atoms_per_barn_cm()
            .map_err(|e| JsValue::from_str(&e))?;

        let map = Map::new();
        for (nuclide, density) in atoms_per_barn_cm {
            map.set(&JsValue::from_str(&nuclide), &JsValue::from_f64(density));
        }
        Ok(map.into())
    }

    #[wasm_bindgen(js_name = macroscopicCrossSection)]
    pub fn macroscopic_cross_section(
        &mut self,
        reaction: i32,
        temperature: Option<String>,
    ) -> Result<Array, JsValue> {
        self.ensure_built()?;
        let inner = self.inner_mut()?;

        if inner.density.is_none() && inner.density_units != DensityUnits::Sum {
            return Err(JsValue::from_str(
                "Cannot calculate macroscopic cross sections: Material has no density defined",
            ));
        }
        if inner.nuclides.is_empty() {
            return Err(JsValue::from_str(
                "Cannot calculate macroscopic cross sections: Material has no nuclides defined",
            ));
        }
        let temp_ref = temperature.as_deref();
        let (xs_values, energy_grid) = inner.macroscopic_cross_section(reaction, temp_ref);
        let result = Array::new();
        result.push(&Float64Array::from(xs_values.as_slice()));
        result.push(&Float64Array::from(energy_grid.as_slice()));
        Ok(result)
    }

    #[wasm_bindgen]
    pub fn reaction_mts(&mut self) -> Result<Array, JsValue> {
        self.ensure_built()?;
        match self.inner_mut()?.reaction_mts() {
            Ok(mts) => Ok(mts.into_iter().map(JsValue::from).collect::<Array>()),
            Err(e) => Err(JsValue::from_str(&format!(
                "Failed to get MT numbers: {}",
                e
            ))),
        }
    }

    #[wasm_bindgen]
    pub fn mean_free_path_neutron(&mut self, energy: f64) -> Option<f64> {
        self.ensure_built().ok()?;
        self.inner_mut().ok()?.mean_free_path_neutron(energy)
    }

    #[wasm_bindgen]
    pub fn sample_distance_to_collision(&mut self, energy: f64) -> Option<f64> {
        self.ensure_built().ok()?;
        let mut rng = rand::rng();
        self.inner_ref()
            .ok()?
            .sample_distance_to_collision(energy, &mut rng)
    }

    // Inherent rather than a Display impl, deliberately: this is what JS calls
    // as `material.toString()`, and `#[wasm_bindgen]` cannot export a trait impl
    // ("trait impls are not supported"), so taking clippy's suggestion here does
    // not compile.
    #[allow(clippy::inherent_to_string)]
    #[wasm_bindgen]
    pub fn to_string(&self) -> String {
        match &self.inner {
            Some(mat) => format!("{:?}", mat),
            None => format!("WasmMaterial(pending, {} nuclides)", self.nuclides.len()),
        }
    }

    #[wasm_bindgen(js_name = sampleInteractingNuclide)]
    pub fn sample_interacting_nuclide_wasm(
        &self,
        energy: f64,
        seed: Option<u64>,
    ) -> Result<String, JsValue> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let inner = self.inner_ref()?;
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(12345),
        };
        Ok(inner.sample_interacting_nuclide(energy, &mut rng))
    }
}
