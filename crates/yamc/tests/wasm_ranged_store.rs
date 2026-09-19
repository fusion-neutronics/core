//! The in-memory store filled the way the browser fetcher fills it, run on the
//! native target.
//!
//! `WasmSimulation.fetchNuclearData` fetches `reactions.arrow` and
//! `energy.arrow` as the byte ranges of the material's temperature (or of the
//! two that bracket it) and splices them into an Arrow IPC stream; every other
//! section arrives whole. This does the same from the on-disk Li6 fixture,
//! slicing the spans `fetch_plan` names out of the whole files, and then runs
//! transport against the result. It is the check that what the fetcher brings
//! is what the loader reads: a plan that named the wrong batches would load a
//! nuclide with no reactions at the material's temperature and fail here.
//!
//! The network half of the same path is `tests/wasm_fetch_browser.rs`, which
//! runs only in a browser. Together they cover the fetcher without a browser
//! test having to be the only thing standing between a CDN layout change and a
//! broken exported page.
//!
//! Runs as part of `cargo test -p yamc --features wasm-test`. Each test
//! installs a fresh `InMemoryStorage` as the process-global, so the tests are
//! serialised, and the global nuclide cache is cleared so that neither can be
//! served the other's load.

#![cfg(feature = "wasm")]

#[path = "common/li6_model.rs"]
mod li6_model;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use nuclear_data_schema::reaction_ranges::splice_spans;
use yamc::wasm::fetch_plan::{plan_neutron_sections, store_dir, wanted_labels};
use yamc::wasm::simulation_wasm::WasmSimulation;
use yamc_nuclide::nuclide_arrow::read_available_temperatures;

static SERIAL: Mutex<()> = Mutex::new(());

fn fixture_dir() -> Option<PathBuf> {
    let dir = PathBuf::from("tests/Li6.arrow");
    if dir.join("version.json").is_file() {
        Some(dir)
    } else {
        eprintln!("skipping: no Li6 fixture at {}", dir.display());
        None
    }
}

/// Bytes registered by the ranged fill against the size of the whole files it
/// ranged into.
struct Filled {
    ranged_bytes: usize,
    whole_bytes: usize,
}

/// Register the fixture in the store as the browser fetcher would for a
/// material at `temperature`: `version.json` and `nuclide.arrow` whole, then
/// the plan. Absent optional sections are simply not registered, as a 404 in
/// the browser is not; the `.absent` markers on disk are the native cache's
/// convention and never reach the store.
fn fill_like_the_browser(sim: &WasmSimulation, dir: &Path, temperature: &str) -> Filled {
    let store = store_dir("Li6");
    for name in ["version.json", "nuclide.arrow"] {
        sim.add_file(
            format!("{store}/{name}"),
            std::fs::read(dir.join(name)).unwrap(),
        );
    }
    let version: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("version.json")).unwrap()).unwrap();
    assert!(
        version.get("reaction_ranges").is_some() && version.get("energy_ranges").is_some(),
        "the Li6 fixture carries no byte-range index; refetch it (format_version 2)"
    );

    // Through the store, as the fetcher reads it.
    let available = read_available_temperatures(Path::new(&store)).unwrap();
    let requested: BTreeSet<String> = [temperature.to_string()].into_iter().collect();
    let wanted = wanted_labels(&available, &requested).unwrap();
    let plan = plan_neutron_sections(&version, Some(&wanted)).unwrap();

    let mut filled = Filled {
        ranged_bytes: 0,
        whole_bytes: 0,
    };
    for section in plan {
        let path = dir.join(section.name);
        if !path.is_file() {
            assert!(
                !section.required,
                "fixture lacks required section {}",
                section.name
            );
            continue;
        }
        let whole = std::fs::read(&path).unwrap();
        let bytes = match &section.spans {
            None => whole.clone(),
            Some(spans) => {
                let bodies: Vec<Vec<u8>> = spans
                    .iter()
                    .map(|&(offset, length)| {
                        whole[offset as usize..(offset + length) as usize].to_vec()
                    })
                    .collect();
                filled.ranged_bytes += bodies.iter().map(Vec::len).sum::<usize>();
                filled.whole_bytes += whole.len();
                splice_spans(&bodies)
            }
        };
        sim.add_file(format!("{store}/{}", section.name), bytes);
    }
    filled
}

#[test]
fn one_temperature_of_the_store_runs_transport() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dir) = fixture_dir() else { return };
    yamc_nuclide::nuclide::clear_nuclide_cache();

    let sim = WasmSimulation::new();
    sim.load_model_json(li6_model::li6_sphere_model_json("294"))
        .unwrap();
    let filled = fill_like_the_browser(&sim, &dir, "294");

    // The point of ranging: one of six temperatures is well under half of the
    // two files it ranges into.
    assert!(filled.whole_bytes > 0, "nothing was ranged");
    assert!(
        filled.ranged_bytes * 2 < filled.whole_bytes,
        "ranged {} bytes of {}",
        filled.ranged_bytes,
        filled.whole_bytes
    );
    assert_eq!(sim.model_missing_nuclides(), "");

    let result = sim.simulate_transport(200, 5, 42);
    li6_model::assert_tritium_result(&result);
}

#[test]
fn a_bracketing_pair_serves_a_blended_temperature() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dir) = fixture_dir() else { return };
    yamc_nuclide::nuclide::clear_nuclide_cache();

    // 450 K is published at no temperature; the loader blends 294 K and 600 K,
    // so those two are what the fetcher has to bring.
    let sim = WasmSimulation::new();
    sim.load_model_json(li6_model::li6_sphere_model_json("450"))
        .unwrap();
    let filled = fill_like_the_browser(&sim, &dir, "450");
    assert!(
        filled.ranged_bytes * 2 < filled.whole_bytes,
        "ranged {} bytes of {}",
        filled.ranged_bytes,
        filled.whole_bytes
    );

    let result = sim.simulate_transport(200, 5, 42);
    li6_model::assert_tritium_result(&result);
}
