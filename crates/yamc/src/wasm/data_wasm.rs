use serde::Serialize;
use serde_wasm_bindgen::to_value;
use wasm_bindgen::prelude::*;
use yamc_nuclide::data::{
    RepresentativeAbundance, ELEMENT_NAMES, ELEMENT_NUCLIDES, NATURAL_ABUNDANCE,
    NATURAL_ABUNDANCE_RECORDS,
};

#[wasm_bindgen]
pub fn natural_abundance() -> JsValue {
    let map: std::collections::HashMap<String, f64> = NATURAL_ABUNDANCE
        .iter()
        .map(|(k, v)| ((*k).to_string(), *v))
        .collect();
    to_value(&map).unwrap()
}

/// One TICE 2013 row, with the same keys as the Python
/// `natural_abundance_records()` so the two front ends read alike.
#[derive(Serialize)]
struct AbundanceRecordJs {
    representative_value: Option<f64>,
    representative_uncertainty: Option<f64>,
    representative_interval: Option<(f64, f64)>,
    observed_interval: Option<(f64, f64)>,
    best_measurement: f64,
    best_measurement_uncertainty: Option<f64>,
    best_measurement_coverage: Option<&'static str>,
    best_measurement_calibration: Option<char>,
    annotations: Option<&'static str>,
}

/// The TICE 2013 record behind each `natural_abundance()` entry. Reference
/// data only: nothing samples or propagates these uncertainties. An absent
/// field is one the table leaves empty ("not stated", never zero).
#[wasm_bindgen]
pub fn natural_abundance_records() -> JsValue {
    let map: std::collections::HashMap<String, AbundanceRecordJs> = NATURAL_ABUNDANCE_RECORDS
        .iter()
        .map(|(nuclide, record)| {
            let (value, uncertainty, interval) = match record.representative {
                RepresentativeAbundance::Value { value, uncertainty } => {
                    (Some(value), uncertainty, None)
                }
                RepresentativeAbundance::Interval { low, high } => (None, None, Some((low, high))),
            };
            let row = AbundanceRecordJs {
                representative_value: value,
                representative_uncertainty: uncertainty,
                representative_interval: interval,
                observed_interval: record.observed_interval,
                best_measurement: record.best_measurement,
                best_measurement_uncertainty: record.best_measurement_uncertainty,
                best_measurement_coverage: record.best_measurement_coverage,
                best_measurement_calibration: record.best_measurement_calibration,
                annotations: record.annotations,
            };
            ((*nuclide).to_string(), row)
        })
        .collect();
    to_value(&map).unwrap()
}

#[wasm_bindgen]
pub fn element_nuclides() -> JsValue {
    let mut map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for (element, nuclides) in ELEMENT_NUCLIDES.iter() {
        let mut sorted_nuclides: Vec<String> = nuclides.iter().map(|n| n.to_string()).collect();
        sorted_nuclides.sort();
        map.insert((*element).to_string(), sorted_nuclides);
    }
    to_value(&map).unwrap()
}

#[wasm_bindgen]
pub fn element_names() -> JsValue {
    let map: std::collections::HashMap<String, String> = ELEMENT_NAMES
        .iter()
        .map(|(symbol, name)| ((*symbol).to_string(), (*name).to_string()))
        .collect();
    to_value(&map).unwrap()
}
