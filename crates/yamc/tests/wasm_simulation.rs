//! End-to-end check of the wasm-bindgen `WasmSimulation` path on a
//! native target.
//!
//! Builds a Li-6 sphere `Model` in Rust, serializes it to JSON, then
//! drives the wasm-bindgen entry points exactly as JS would:
//!
//!     sim.load_model_json(model_json)
//!     sim.add_file("/Li6.arrow/<file>", bytes)   // per file
//!     sim.simulate_transport(particles, batches, seed)
//!
//! Runs as part of `cargo test -p yamc --features wasm-test`. The test
//! installs a fresh `InMemoryStorage` as the process-global; don't add
//! anything to this file that expects `NativeStorage`-style filesystem
//! reads -- they'd be redirected.

#![cfg(feature = "wasm")]

#[path = "common/li6_model.rs"]
mod li6_model;

use yamc::wasm::simulation_wasm::WasmSimulation;

fn load_fixture_into(sim: &WasmSimulation, nuclide: &str) {
    let dir = std::path::PathBuf::from(format!("tests/{nuclide}.arrow"));
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {dir:?}: {e}")) {
        let entry = entry.unwrap();
        let bytes = std::fs::read(entry.path()).unwrap();
        let virtual_path = format!("/{nuclide}.arrow/{}", entry.file_name().to_string_lossy());
        sim.add_file(virtual_path, bytes);
    }
}

#[test]
fn li6_sphere_runs_via_loaded_model_json() {
    // Build the model in Rust, serialize to JSON -- this is the same
    // step `model.save()` / `model.export()` will do in Python.
    let model_json = li6_model::li6_sphere_model_json("294");

    // Drive WasmSimulation exactly as the JS host would.
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).expect("load_model_json");

    // `model_required_nuclides` is the JS-side hook that tells the host
    // which `<Nuclide>.arrow/` section sets to fetch; `model_missing_nuclides`
    // is what gates Simulate until the store holds them.
    assert_eq!(sim.model_required_nuclides(), "Li6");
    assert_eq!(sim.model_missing_nuclides(), "Li6");
    load_fixture_into(&sim, "Li6");
    assert_eq!(sim.model_missing_nuclides(), "");
    assert_eq!(sim.model_missing_elements(), "");

    let result_json = sim.simulate_transport(200, 5, 42);
    eprintln!("loaded-model result: {result_json}");
    li6_model::assert_tritium_result(&result_json);
}
