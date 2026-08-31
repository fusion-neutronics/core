use js_sys::{Array, JSON};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use yamc_nuclide::buffer::F64Buffer;
use yamc_nuclide::reaction::Reaction;

#[wasm_bindgen]
pub struct WasmReaction {
    inner: Reaction,
}

#[derive(Serialize, Deserialize)]
struct ReactionData {
    threshold_idx: usize,
    cross_section: F64Buffer,
    energy: F64Buffer,
}

#[wasm_bindgen]
impl WasmReaction {
    #[wasm_bindgen(constructor)]
    pub fn new(threshold_idx: usize) -> Self {
        WasmReaction {
            inner: Reaction {
                threshold_idx,
                cross_section: F64Buffer::default(),
                energy: F64Buffer::default(),
                mt_number: 0,
                products: Vec::new(),
                q_value: 0.0,
                redundant: false,
                scatter_in_cm: false,
            },
        }
    }

    #[wasm_bindgen]
    pub fn set_cross_section(&mut self, cross_section: Vec<f64>) {
        // `Vec` in, no copy out: `F64Buffer::from` takes the allocation over.
        self.inner.cross_section = cross_section.into();
    }

    #[wasm_bindgen]
    pub fn set_energy(&mut self, energy: Vec<f64>) {
        self.inner.energy = energy.into();
    }

    #[wasm_bindgen]
    pub fn get_threshold_idx(&self) -> usize {
        self.inner.threshold_idx
    }

    #[wasm_bindgen]
    pub fn get_cross_section(&self) -> Array {
        self.inner
            .cross_section
            .iter()
            .map(|&x| JsValue::from_f64(x))
            .collect::<Array>()
    }

    #[wasm_bindgen]
    pub fn get_energy(&self) -> Array {
        self.inner
            .energy
            .iter()
            .map(|&x| JsValue::from_f64(x))
            .collect::<Array>()
    }

    #[wasm_bindgen]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        let data = ReactionData {
            threshold_idx: self.inner.threshold_idx,
            cross_section: self.inner.cross_section.clone(),
            energy: self.inner.energy.clone(),
        };

        let serialized = serde_json::to_string(&data)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))?;

        JSON::parse(&serialized)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {:?}", e)))
    }
}
