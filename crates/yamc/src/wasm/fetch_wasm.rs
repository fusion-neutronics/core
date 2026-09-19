//! `WasmSimulation.fetchNuclearData`: the browser-side nuclear-data downloader.
//!
//! The native build resolves a material's nuclides through
//! `yamc_nuclide::url_cache`, which has reqwest and a filesystem. Neither exists
//! on wasm32, so the browser build fetches the same published section objects
//! through the page's own `fetch` and registers the bytes in the in-memory
//! [`InMemoryStorage`] the loader reads from. What to fetch, and which byte
//! spans of it, is decided by [`super::fetch_plan`]; this module only executes
//! that plan and is the one part of the path that has to run in a browser.
//!
//! One nuclide at a time, every section of it in flight together: the requests
//! for one nuclide are issued before any is awaited, so the browser overlaps
//! them, and the next nuclide starts once this one is registered. A progress
//! callback, when the host passes one, hears about each nuclide as it starts.
//!
//! The origin answers a `Range` request with 206 and the bytes asked for, or,
//! if some proxy on the way dropped the header, with 200 and the whole object.
//! The second is not an error: what arrived is the entire file and is
//! registered as such, and the other spans of that section are left to finish
//! on their own. Spliced after a schema message, a whole file's `ARROW1` magic
//! would be read as a record batch, which is why the two cases are told apart
//! by status rather than by length.

use std::collections::BTreeSet;
use std::path::Path;

use js_sys::{Function, Object, Promise, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{future_to_promise, JsFuture};
use web_sys::{RequestInit, Response};

use nuclear_data_schema::reaction_ranges::{splice_spans, Range};
use yamc_nuclide::in_memory_storage::InMemoryStorage;
use yamc_nuclide::nuclide_arrow::read_available_temperatures;
use yamc_nuclide::storage::Storage;

use super::fetch_plan::{
    neutron_url, photon_url, plan_neutron_sections, plan_photon_sections, required_temperatures,
    store_dir, wanted_labels, SectionPlan,
};
use super::simulation_wasm::WasmSimulation;

#[wasm_bindgen]
extern "C" {
    /// The global `fetch`, bound directly rather than through `Window` so the
    /// same binding serves a page and a worker.
    #[wasm_bindgen(js_name = fetch)]
    fn js_fetch(url: &str, init: &RequestInit) -> Promise;
}

/// Transient failures (a dropped connection, a 5xx) are retried this many
/// times before the whole fetch fails. A 404 is definitive and never retried.
const ATTEMPTS: usize = 3;

/// A readable description of a rejected Promise's value.
fn describe(error: JsValue) -> String {
    if let Some(text) = error.as_string() {
        return text;
    }
    if let Some(error) = error.dyn_ref::<js_sys::Error>() {
        return String::from(error.message());
    }
    format!("{error:?}")
}

/// One issued GET.
struct Request {
    url: String,
    span: Option<Range>,
    promise: Promise,
}

/// What a GET came back with.
enum Fetched {
    /// The bytes asked for. `partial` records whether the origin honoured the
    /// `Range` header (206) or ignored it and sent the whole object (200).
    Body { bytes: Vec<u8>, partial: bool },
    /// A definitive 404. The section is absent upstream.
    Absent,
}

/// Issue a GET, for one byte span when `span` is given. The request is in
/// flight as soon as this returns; [`finish`] collects the answer.
fn start(url: &str, span: Option<Range>) -> Request {
    let init = RequestInit::new();
    if let Some((offset, length)) = span {
        let headers = Object::new();
        // A single byte range is a CORS-safelisted request header, so this
        // costs no preflight. Several ranges in one header would.
        let range = format!("bytes={}-{}", offset, offset + length - 1);
        // `Reflect::set` on a fresh plain object cannot fail.
        let _ = Reflect::set(
            &headers,
            &JsValue::from_str("Range"),
            &JsValue::from_str(&range),
        );
        init.set_headers(&headers);
    }
    Request {
        url: url.to_string(),
        span,
        promise: js_fetch(url, &init),
    }
}

async fn finish(request: Request) -> Result<Fetched, String> {
    let Request { url, span, promise } = request;
    let response: Response = JsFuture::from(promise)
        .await
        .map_err(|e| format!("{url}: {}", describe(e)))?
        .dyn_into()
        .map_err(|_| format!("{url}: fetch resolved to something other than a Response"))?;
    match response.status() {
        200 | 206 => {
            let partial = response.status() == 206;
            let buffer = JsFuture::from(
                response
                    .array_buffer()
                    .map_err(|e| format!("{url}: {}", describe(e)))?,
            )
            .await
            .map_err(|e| format!("{url}: reading the body: {}", describe(e)))?;
            let bytes = Uint8Array::new(&buffer).to_vec();
            if let (true, Some((offset, length))) = (partial, span) {
                if bytes.len() as u64 != length {
                    // A short span cannot be spliced: the framing still walks,
                    // and the message header at the cut is read as whatever the
                    // next bytes happen to be.
                    return Err(format!(
                        "{url}: asked for {length} bytes at {offset} and got {}",
                        bytes.len()
                    ));
                }
            }
            Ok(Fetched::Body { bytes, partial })
        }
        404 => Ok(Fetched::Absent),
        status => Err(format!("{url}: HTTP {status}")),
    }
}

/// [`finish`], reissuing the request on a transient failure.
async fn finish_retrying(request: Request) -> Result<Fetched, String> {
    let (url, span) = (request.url.clone(), request.span);
    let mut request = request;
    let mut attempt = 1;
    loop {
        match finish(request).await {
            Ok(fetched) => return Ok(fetched),
            Err(e) if attempt < ATTEMPTS => {
                attempt += 1;
                request = start(&url, span);
                let _ = e;
            }
            Err(e) => return Err(format!("{e} (after {attempt} attempts)")),
        }
    }
}

/// Every request one section needs, already in flight.
struct SectionFetch {
    plan: SectionPlan,
    url: String,
    requests: Vec<Request>,
}

fn start_section(base_url: &str, plan: SectionPlan) -> SectionFetch {
    let url = format!("{base_url}/{}", plan.name);
    let requests = match &plan.spans {
        None => vec![start(&url, None)],
        Some(spans) => spans.iter().map(|span| start(&url, Some(*span))).collect(),
    };
    SectionFetch {
        plan,
        url,
        requests,
    }
}

/// Running totals for the summary the host gets back.
#[derive(Default)]
struct Stats {
    requests: usize,
    bytes: usize,
}

/// Collect a section. `Ok(Some(bytes))` is the object, whole or spliced from
/// its spans; `Ok(None)` is an absent optional section.
async fn finish_section(fetch: SectionFetch, stats: &mut Stats) -> Result<Option<Vec<u8>>, String> {
    let SectionFetch {
        plan,
        url,
        requests,
    } = fetch;
    stats.requests += requests.len();
    if plan.spans.is_none() {
        let request = requests
            .into_iter()
            .next()
            .expect("a whole-object fetch issues one request");
        return match finish_retrying(request).await? {
            Fetched::Body { bytes, .. } => {
                stats.bytes += bytes.len();
                Ok(Some(bytes))
            }
            Fetched::Absent if plan.required => {
                Err(format!("{url}: 404, and this section is required"))
            }
            Fetched::Absent => Ok(None),
        };
    }
    let mut bodies = Vec::with_capacity(requests.len());
    for request in requests {
        match finish_retrying(request).await? {
            Fetched::Body {
                bytes,
                partial: true,
            } => {
                stats.bytes += bytes.len();
                bodies.push(bytes);
            }
            // Range ignored somewhere on the way: what is in hand IS the whole
            // object, so register it as one rather than splicing it as a slice.
            Fetched::Body {
                bytes,
                partial: false,
            } => {
                stats.bytes += bytes.len();
                return Ok(Some(bytes));
            }
            // The nuclide has a version.json naming ranges but no object to
            // range into, which is a broken publish rather than an absent
            // optional section.
            Fetched::Absent => {
                return Err(format!(
                    "{url}: 404, but its version.json names byte ranges"
                ))
            }
        }
    }
    Ok(Some(splice_spans(&bodies)))
}

/// The host's progress callback, if any: `(message, done, total)`.
struct Progress(Option<Function>);

impl Progress {
    fn report(&self, message: &str, done: usize, total: usize) {
        if let Some(callback) = &self.0 {
            // A throwing callback is the host's bug and not a reason to abandon
            // the download.
            let _ = callback.call3(
                &JsValue::NULL,
                &JsValue::from_str(message),
                &JsValue::from_f64(done as f64),
                &JsValue::from_f64(total as f64),
            );
        }
    }
}

/// What the loaded model needs and does not yet hold.
struct Targets {
    /// Nuclide name and the material temperatures it is wanted at (`None` for
    /// every temperature).
    nuclides: Vec<(String, Option<BTreeSet<String>>)>,
    /// Element symbols photon transport needs.
    elements: Vec<String>,
}

/// Fetch and register one nuclide. Returns whether its reactions were ranged.
async fn fetch_nuclide(
    storage: &InMemoryStorage,
    library_url: &str,
    name: &str,
    temperatures: Option<&BTreeSet<String>>,
    stats: &mut Stats,
) -> Result<bool, String> {
    let base_url = neutron_url(library_url, name);
    let dir = store_dir(name);

    // The stamp and the temperature list first: the plan for everything else
    // needs the byte-range index out of one and the published temperatures
    // out of the other.
    let mut first = Vec::with_capacity(2);
    for name in ["version.json", "nuclide.arrow"] {
        first.push(start_section(
            &base_url,
            SectionPlan {
                name,
                required: true,
                spans: None,
            },
        ));
    }
    let mut first_bytes = Vec::with_capacity(2);
    for fetch in first {
        let section = fetch.plan.name;
        let bytes = finish_section(fetch, stats)
            .await?
            .expect("a required section is bytes or an error");
        storage.add_file(format!("{dir}/{section}"), bytes.clone());
        first_bytes.push(bytes);
    }
    let version: serde_json::Value = serde_json::from_slice(&first_bytes[0])
        .map_err(|e| format!("{base_url}/version.json: not JSON: {e}"))?;

    let wanted = match temperatures {
        None => None,
        Some(requested) => {
            let available =
                read_available_temperatures(Path::new(&dir)).map_err(|e| format!("{name}: {e}"))?;
            Some(wanted_labels(&available, requested).map_err(|e| format!("{name}: {e}"))?)
        }
    };
    let plan =
        plan_neutron_sections(&version, wanted.as_ref()).map_err(|e| format!("{name}: {e}"))?;
    let ranged = plan
        .iter()
        .any(|section| section.name == "reactions.arrow" && section.spans.is_some());

    let fetches: Vec<SectionFetch> = plan
        .into_iter()
        .map(|section| start_section(&base_url, section))
        .collect();
    for fetch in fetches {
        let section = fetch.plan.name;
        if let Some(bytes) = finish_section(fetch, stats).await? {
            storage.add_file(format!("{dir}/{section}"), bytes);
        }
    }
    Ok(ranged)
}

/// Fetch and register one element's photon data.
async fn fetch_element(
    storage: &InMemoryStorage,
    library_url: &str,
    element: &str,
    stats: &mut Stats,
) -> Result<(), String> {
    let base_url = photon_url(library_url, element);
    let dir = store_dir(element);
    let fetches: Vec<SectionFetch> = plan_photon_sections()
        .into_iter()
        .map(|section| start_section(&base_url, section))
        .collect();
    for fetch in fetches {
        let section = fetch.plan.name;
        if let Some(bytes) = finish_section(fetch, stats).await? {
            storage.add_file(format!("{dir}/{section}"), bytes);
        }
    }
    Ok(())
}

async fn fetch_all(
    storage: InMemoryStorage,
    targets: Targets,
    library_url: String,
    progress: Progress,
) -> Result<String, String> {
    let total = targets.nuclides.len() + targets.elements.len();
    let mut stats = Stats::default();
    let mut nuclides = Vec::new();
    let mut ranged = Vec::new();
    let mut elements = Vec::new();
    let mut done = 0;

    for (name, temperatures) in &targets.nuclides {
        progress.report(
            &format!("Fetching {name} ({}/{total})", done + 1),
            done,
            total,
        );
        if fetch_nuclide(
            &storage,
            &library_url,
            name,
            temperatures.as_ref(),
            &mut stats,
        )
        .await?
        {
            ranged.push(name.clone());
        }
        nuclides.push(name.clone());
        done += 1;
    }
    for element in &targets.elements {
        progress.report(
            &format!("Fetching photon data for {element} ({}/{total})", done + 1),
            done,
            total,
        );
        fetch_element(&storage, &library_url, element, &mut stats).await?;
        elements.push(element.clone());
        done += 1;
    }
    progress.report("Done", total, total);

    Ok(serde_json::json!({
        "status": "ok",
        "nuclides": nuclides,
        "ranged": ranged,
        "elements": elements,
        "requests": stats.requests,
        "bytes": stats.bytes,
    })
    .to_string())
}

#[wasm_bindgen]
impl WasmSimulation {
    /// Fetch the nuclear data the loaded model needs and does not yet hold
    /// into the in-memory store, from `library_url` (see
    /// [`default_library_url`](super::simulation_wasm::default_library_url)).
    ///
    /// Each required nuclide is fetched as its published section objects. Where
    /// `version.json` carries the byte-range index, `reactions.arrow` and
    /// `energy.arrow` come as the ranges of the temperature each material is at
    /// (or the two that bracket it), which is most of the saving: on
    /// ENDF/B-VIII.1 one temperature of a nuclide's reactions is about a sixth
    /// of them. Per-element photon data follows when the model has photons in
    /// flight. A nuclide or element whose data is already in the store (fetched
    /// earlier, or embedded by the exporting Python) is skipped.
    ///
    /// `on_progress`, when given, is called as `(message, done, total)` as each
    /// nuclide or element starts.
    ///
    /// Resolves to a JSON string,
    /// `{"status":"ok","nuclides":[...],"ranged":[...],"elements":[...],"requests":n,"bytes":n}`,
    /// where `ranged` lists the nuclides whose reactions were fetched by byte
    /// range. Rejects with a message naming the URL that failed. A material
    /// temperature the published data does not cover is refused before any
    /// bytes are fetched for that nuclide, with the same message the loader
    /// would give.
    #[wasm_bindgen(js_name = fetchNuclearData)]
    pub fn fetch_nuclear_data(
        &self,
        library_url: String,
        on_progress: Option<Function>,
    ) -> Promise {
        let storage = self.storage().clone();
        let targets = self.fetch_targets();
        future_to_promise(async move {
            let targets = targets.map_err(|e| JsValue::from_str(&e))?;
            let summary = fetch_all(storage, targets, library_url, Progress(on_progress))
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            Ok(JsValue::from_str(&summary))
        })
    }
}

impl WasmSimulation {
    /// The nuclides and elements the loaded model needs that the store does not
    /// hold yet, with the temperatures the materials want each nuclide at.
    fn fetch_targets(&self) -> Result<Targets, String> {
        let storage = self.storage();
        self.with_model(|model| {
            let nuclides = required_temperatures(model)
                .into_iter()
                .filter(|(name, _)| {
                    !storage.exists(Path::new(&format!("{}/nuclide.arrow", store_dir(name))))
                })
                .collect();
            let elements = model
                .required_elements()
                .into_iter()
                .filter(|element| {
                    !storage.exists(Path::new(&format!("{}/element.arrow", store_dir(element))))
                })
                .collect();
            Targets { nuclides, elements }
        })
    }
}
