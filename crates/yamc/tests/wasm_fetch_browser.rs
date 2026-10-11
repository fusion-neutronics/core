//! The browser fetcher against the real CDN, in a real browser.
//!
//! `WasmSimulation.fetchNuclearData` is the one piece of the exported page
//! that talks to the network, and the shape of what it talks to has changed
//! under it before: the page shipped fetching `.arrow.tar` bundles for a while
//! after the CDN stopped publishing them, and nothing in CI noticed, because
//! the wasm lane compiled the binding and ran no test that fetched. This is
//! that test. It runs in headless Firefox under
//!
//!     wasm-pack test --headless --firefox --features wasm-test
//!
//! from `crates/yamc` (the `ci-wasm.yml` lane), fetches Li6 from the published
//! ENDF/B-VIII.1 objects and runs transport on what arrived. It needs the
//! network, which the same lane already needs to fetch its fixtures.
//!
//! The arithmetic of the fetch (which spans of which sections) is tested
//! natively in `fetch_plan` and `tests/wasm_ranged_store.rs`; this covers the
//! part they cannot: the published layout, CORS, range requests, and the
//! wasm-bindgen Promise plumbing.

#![cfg(all(feature = "wasm", target_arch = "wasm32"))]

#[path = "common/li6_model.rs"]
mod li6_model;

use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::*;
use yamc::wasm::simulation_wasm::{default_library_url, WasmSimulation};

wasm_bindgen_test_configure!(run_in_browser);

/// The fetch summary, or `None` when the origin has not published the library
/// in the release layout yet. That is the transition state until the next data
/// publish, and it is reported and skipped rather than failed; every other
/// rejection fails the test.
async fn fetch(sim: &WasmSimulation) -> Option<serde_json::Value> {
    let summary = match JsFuture::from(sim.fetch_nuclear_data(default_library_url(), None)).await {
        Ok(summary) => summary,
        Err(e) => {
            let message = e.as_string().unwrap_or_else(|| format!("{e:?}"));
            if message.contains("has not been published in the release layout") {
                console_log!("skipped: {message}");
                return None;
            }
            panic!("fetchNuclearData rejected: {message}");
        }
    };
    Some(serde_json::from_str(&summary.as_string().expect("a JSON string")).expect("summary JSON"))
}

#[wasm_bindgen_test]
async fn fetches_the_model_nuclides_from_the_cdn_and_runs() {
    let sim = WasmSimulation::new();
    sim.load_model_json(li6_model::li6_sphere_model_json("294"))
        .unwrap();
    assert_eq!(sim.model_missing_nuclides(), "Li6");
    assert_eq!(sim.model_missing_elements(), "");

    let Some(summary) = fetch(&sim).await else {
        return;
    };
    assert_eq!(summary["status"], "ok", "{summary}");
    assert_eq!(summary["library"], "endf-b8.1", "{summary}");
    assert!(summary["release"].is_string(), "{summary}");
    assert_eq!(summary["nuclides"], serde_json::json!(["Li6"]), "{summary}");
    // The published version.json carries the index, so the reactions came
    // as the 294 K byte ranges rather than the whole file.
    assert_eq!(summary["ranged"], serde_json::json!(["Li6"]), "{summary}");
    assert!(
        summary["requests"].as_u64().unwrap() > 2,
        "a ranged fetch is several requests: {summary}"
    );
    // Whole Li6 is about 1.4 MB; one temperature of its two ranged sections
    // plus the whole of the rest is well under that.
    let bytes = summary["bytes"].as_u64().unwrap();
    assert!(
        (200_000..1_200_000).contains(&bytes),
        "unexpected byte count: {summary}"
    );
    assert_eq!(sim.model_missing_nuclides(), "");
    assert!(sim.file_count() >= 6, "files: {}", sim.file_count());

    let result = sim.simulate_transport(200, 1, 42);
    li6_model::assert_tritium_result(&result);

    // Everything is held now, so a second call has nothing to fetch.
    let again = fetch(&sim)
        .await
        .expect("nothing to fetch, so nothing to refuse");
    assert_eq!(again["nuclides"], serde_json::json!([]), "{again}");
    assert_eq!(again["requests"], 0, "{again}");
}
