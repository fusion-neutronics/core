use js_sys::Array;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use wasm_bindgen::prelude::*;
use yamc_nuclide::nuclide::Nuclide;

// Global cache for WASM nuclides to avoid reloading
static WASM_NUCLIDE_CACHE: Lazy<Mutex<HashMap<String, Arc<Nuclide>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// Get a nuclide from the WASM cache.
// Note: In WASM environments, loading from files is not directly supported.
// A different data format (e.g., serialized JSON) would need to be implemented.
pub fn get_or_load_nuclide_wasm(
    nuclide_name: &str,
    _path_map: &HashMap<String, String>,
) -> Result<Arc<Nuclide>, Box<dyn std::error::Error>> {
    // Try to get from cache first
    {
        let cache = WASM_NUCLIDE_CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(nuclide) = cache.get(nuclide_name) {
            return Ok(Arc::clone(nuclide));
        }
    }

    // WASM file loading not currently supported - nuclide data must be pre-loaded
    Err(format!(
        "Nuclide '{}' not found in WASM cache. File loading is not supported in WASM environments.",
        nuclide_name
    )
    .into())
}

#[wasm_bindgen]
pub struct WasmNuclide {
    inner: Arc<Nuclide>,
}

#[wasm_bindgen]
impl WasmNuclide {
    #[wasm_bindgen]
    pub fn load(_name: &str, _path: &str) -> Result<WasmNuclide, JsValue> {
        // WASM file loading not currently supported
        // A serializable data format would need to be implemented
        Err(JsValue::from_str(
            "File loading is not supported in WASM environments. Use a serialized data format.",
        ))
    }

    #[wasm_bindgen]
    pub fn get_name(&self) -> String {
        self.inner
            .name
            .clone()
            .unwrap_or_else(|| "Unknown".to_string())
    }

    #[wasm_bindgen]
    pub fn get_available_temperatures(&self) -> Array {
        self.inner
            .available_temperatures
            .iter()
            .map(|t| JsValue::from_str(t))
            .collect::<Array>()
    }

    #[wasm_bindgen]
    pub fn get_available_reactions(&self, temperature: &str) -> Result<Array, JsValue> {
        // Find the temperature index
        let temp_idx = self
            .inner
            .loaded_temperatures
            .iter()
            .position(|t| t == temperature);

        match temp_idx.and_then(|idx| self.inner.reactions.get(idx)) {
            Some(reactions) => {
                let mt_numbers = reactions.keys().cloned().collect::<Vec<i32>>();
                Ok(mt_numbers.into_iter().map(JsValue::from).collect::<Array>())
            }
            None => Err(JsValue::from_str(&format!(
                "Temperature {} not found",
                temperature
            ))),
        }
    }
}

#[wasm_bindgen]
pub fn wasm_load_nuclide(name: &str, path: &str) -> Result<WasmNuclide, JsValue> {
    WasmNuclide::load(name, path)
}
