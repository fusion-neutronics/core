//! What a browser has to fetch to run the loaded model, and which bytes of it.
//!
//! A nuclide is published as a directory of section objects
//! (`<library>/neutron/<Nuclide>.arrow/{version.json, nuclide.arrow,
//! energy.arrow, reactions.arrow, ...}`), the same layout the native downloader
//! in `yamc_nuclide::url_cache` reads. Two of those sections carry nearly all
//! of the bytes and are written one Arrow record batch per unit a reader might
//! want on its own: `reactions.arrow` one batch per (MT, temperature) and
//! `energy.arrow` one batch per temperature, with a byte-range index for both
//! in `version.json`. Transport at one material temperature needs one
//! temperature's batches (or the two that bracket it), so a browser can ask for
//! those alone over HTTP range requests and skip the other five or six
//! temperatures, which on ENDF/B-VIII.1 is most of every file.
//!
//! This module is the arithmetic of that: which nuclides at which temperatures,
//! which labels those resolve to against what the nuclide publishes, and which
//! byte spans of which sections to request. It touches neither the network nor
//! the DOM, so it compiles and is tested on the native target; the wasm32-only
//! fetcher in `fetch_wasm` executes what it plans.
//!
//! The temperature resolution goes through `yamc_nuclide::temperature::resolve`,
//! the same function the loader applies when the material later asks for its
//! temperature. That is the point of doing it here rather than in JavaScript:
//! the bytes fetched are exactly the bytes the loader will read, by
//! construction rather than by keeping two implementations in step.

use std::collections::{BTreeMap, BTreeSet};

use nuclear_data_schema::energy_ranges::EnergyRanges;
use nuclear_data_schema::reaction_ranges::{Range, ReactionRanges};
use yamc_nuclide::temperature::{resolve, strip_k, TemperatureSource};
use yamc_nuclide::url_cache::{NEUTRON_SECTIONS, PHOTON_SECTIONS};

use crate::model::Model;

/// The published library an exported page fetches from unless the host names
/// another. Per-nuclide neutron data lives under `neutron/`, per-element photon
/// data under `photon/`.
pub const DEFAULT_LIBRARY_URL: &str = "https://yamc-data.xsplot.com/endf-b8.1";

/// The two sections a temperature-aware fetch can range into.
const REACTIONS: &str = "reactions.arrow";
const ENERGY: &str = "energy.arrow";

/// Base URL of one nuclide's neutron section objects.
pub fn neutron_url(library_url: &str, nuclide: &str) -> String {
    format!(
        "{}/neutron/{nuclide}.arrow",
        library_url.trim_end_matches('/')
    )
}

/// Base URL of one element's photon section objects.
pub fn photon_url(library_url: &str, element: &str) -> String {
    format!(
        "{}/photon/{element}.arrow",
        library_url.trim_end_matches('/')
    )
}

/// The virtual directory a nuclide's or element's sections are registered under
/// in the in-memory store. Matches the `/<Name>.arrow` the wasm transport path
/// points each material at.
pub fn store_dir(name: &str) -> String {
    format!("/{name}.arrow")
}

/// Which temperatures each nuclide is needed at, from the model's materials.
///
/// Labels are stripped of any `K`, as materials spell them. `None` means every
/// temperature: some material holding the nuclide has no temperature set, and
/// the loader then materialises every temperature the file carries, so the
/// fetch has to bring them all. A nuclide two materials hold at two
/// temperatures wants both.
pub fn required_temperatures(model: &Model) -> BTreeMap<String, Option<BTreeSet<String>>> {
    let mut wanted: BTreeMap<String, Option<BTreeSet<String>>> = BTreeMap::new();
    for material in model.geometry.materials() {
        let temperature = strip_k(material.temperature());
        for nuclide in material.nuclides.keys() {
            let entry = wanted
                .entry(nuclide.clone())
                .or_insert_with(|| Some(BTreeSet::new()));
            match entry {
                None => {}
                Some(set) if temperature.is_empty() => {
                    set.clear();
                    *entry = None;
                }
                Some(set) => {
                    set.insert(temperature.to_string());
                }
            }
        }
    }
    wanted
}

/// The published labels a set of requested temperatures needs.
///
/// `available` as `nuclide.arrow` spells them (`"294K"`), which is also how the
/// byte-range index keys its batches; the answer keeps that spelling. A
/// request the data carries resolves to itself, one it brackets resolves to
/// both neighbours (the loader blends them), and one outside the range is the
/// same error the loader would raise, so the page reports it before fetching a
/// byte rather than after fetching everything.
pub fn wanted_labels(
    available: &[String],
    requested: &BTreeSet<String>,
) -> Result<BTreeSet<String>, String> {
    let mut labels = BTreeSet::new();
    for label in requested {
        match resolve(label, available).map_err(|e| e.to_string())? {
            TemperatureSource::Exact { idx } => {
                labels.insert(available[idx].clone());
            }
            TemperatureSource::Blend { lo_idx, hi_idx, .. } => {
                labels.insert(available[lo_idx].clone());
                labels.insert(available[hi_idx].clone());
            }
        }
    }
    Ok(labels)
}

/// One section object to fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionPlan {
    /// Filename under the nuclide's or element's directory.
    pub name: &'static str,
    /// Whether a 404 is an error. Optional sections (URR tables, fission nu,
    /// Compton profiles, ...) are absent for most nuclides and elements.
    pub required: bool,
    /// The byte spans to request, schema message first, when only part of the
    /// object is wanted. `None` fetches it whole. The responses to `Some`
    /// spans concatenate, with the end-of-stream marker appended, into an
    /// Arrow IPC stream the loader reads like the whole file.
    pub spans: Option<Vec<Range>>,
}

/// Whether a wanted label matches an index label. Both come from the files, so
/// they agree on spelling already; stripping the `K` keeps that from mattering.
fn same_label(wanted: &BTreeSet<String>, label: &str) -> bool {
    let bare = strip_k(label);
    wanted.iter().any(|w| strip_k(w) == bare)
}

/// Plan every neutron section after `version.json` and `nuclide.arrow`, which
/// the fetcher needs first (for the index and the temperature list) and so has
/// already fetched whole.
///
/// With `wanted` labels and an index in `version.json`, `reactions.arrow` and
/// `energy.arrow` come as the spans of those labels' batches; every other
/// section is fetched whole, since none of them has a temperature axis. Without
/// an index, or with no `wanted` (every temperature), both are fetched whole,
/// which is what data published before the index existed looks like and is not
/// an error.
///
/// An error is a request the index cannot serve: labels the nuclide's
/// temperature list carries but no batch is published at, which is a broken
/// publish. Fetching the schema alone would give the loader a stream with no
/// reactions and nothing to say about it.
pub fn plan_neutron_sections(
    version: &serde_json::Value,
    wanted: Option<&BTreeSet<String>>,
) -> Result<Vec<SectionPlan>, String> {
    let mut plan = Vec::new();
    for &(name, required) in NEUTRON_SECTIONS {
        if name == "version.json" || name == "nuclide.arrow" {
            continue;
        }
        let spans = match (name, wanted) {
            (REACTIONS, Some(wanted)) => reaction_spans(version, wanted)?,
            (ENERGY, Some(wanted)) => energy_spans(version, wanted)?,
            _ => None,
        };
        plan.push(SectionPlan {
            name,
            required,
            spans,
        });
    }
    Ok(plan)
}

/// Every photon section, whole. Photon data has no temperature axis and no
/// byte-range index.
pub fn plan_photon_sections() -> Vec<SectionPlan> {
    PHOTON_SECTIONS
        .iter()
        .map(|&(name, required)| SectionPlan {
            name,
            required,
            spans: None,
        })
        .collect()
}

fn reaction_spans(
    version: &serde_json::Value,
    wanted: &BTreeSet<String>,
) -> Result<Option<Vec<Range>>, String> {
    let Some(index) = ReactionRanges::from_version_json(version) else {
        return Ok(None);
    };
    let published: BTreeSet<&str> = index
        .mts
        .values()
        .flat_map(|by_temperature| by_temperature.keys().map(String::as_str))
        .collect();
    if published.iter().all(|label| same_label(wanted, label)) {
        // Every batch is wanted: one plain GET beats a span per MT.
        return Ok(None);
    }
    let spans = index.spans_where(|_, label| same_label(wanted, label));
    if spans.len() == 1 {
        return Err(format!(
            "version.json indexes {REACTIONS} but publishes no batch at {:?}; the index names {:?}",
            wanted, published
        ));
    }
    Ok(Some(spans))
}

fn energy_spans(
    version: &serde_json::Value,
    wanted: &BTreeSet<String>,
) -> Result<Option<Vec<Range>>, String> {
    let Some(index) = EnergyRanges::from_version_json(version) else {
        return Ok(None);
    };
    if index
        .temperatures
        .keys()
        .all(|label| same_label(wanted, label))
    {
        return Ok(None);
    }
    let spans = index.spans_for(|label| same_label(wanted, label));
    if spans.len() == 1 {
        return Err(format!(
            "version.json indexes {ENERGY} but publishes no grid at {:?}; the index names {:?}",
            wanted,
            index.temperatures.keys().collect::<Vec<_>>()
        ));
    }
    Ok(Some(spans))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ladder() -> Vec<String> {
        ["250K", "294K", "600K", "900K", "1200K", "2500K"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn set(labels: &[&str]) -> BTreeSet<String> {
        labels.iter().map(|s| s.to_string()).collect()
    }

    /// An index in the shape the converter writes: one batch per (MT,
    /// temperature), the temperatures of one MT back to back.
    fn version() -> serde_json::Value {
        let mut mts = serde_json::Map::new();
        let mut off = 768u64;
        for mt in [1, 2, 102] {
            let mut by_t = serde_json::Map::new();
            for t in ["250K", "294K", "600K"] {
                by_t.insert(t.to_string(), serde_json::json!([off, 100]));
                off += 100;
            }
            mts.insert(mt.to_string(), serde_json::Value::Object(by_t));
        }
        serde_json::json!({
            "reaction_ranges": {"schema": [64, 704], "mts": mts},
            "energy_ranges": {
                "schema": [64, 384],
                "temperatures": {"0K": [448, 50], "250K": [498, 50], "294K": [548, 50], "600K": [598, 50]}
            }
        })
    }

    #[test]
    fn an_exact_temperature_wants_its_own_label_in_file_spelling() {
        assert_eq!(
            wanted_labels(&ladder(), &set(&["294"])).unwrap(),
            set(&["294K"])
        );
    }

    #[test]
    fn a_bracketed_temperature_wants_both_neighbours() {
        assert_eq!(
            wanted_labels(&ladder(), &set(&["450"])).unwrap(),
            set(&["294K", "600K"])
        );
    }

    #[test]
    fn two_materials_sharing_a_bracket_rung_want_it_once() {
        assert_eq!(
            wanted_labels(&ladder(), &set(&["294", "450"])).unwrap(),
            set(&["294K", "600K"])
        );
    }

    #[test]
    fn a_temperature_outside_the_data_is_refused_before_any_fetch() {
        let err = wanted_labels(&ladder(), &set(&["3000"])).unwrap_err();
        assert!(err.contains("3000"), "{err}");
        assert!(err.contains("2500"), "{err}");
    }

    #[test]
    fn one_temperature_ranges_reactions_and_energy_and_fetches_the_rest_whole() {
        let plan = plan_neutron_sections(&version(), Some(&set(&["294K"]))).unwrap();
        let names: Vec<&str> = plan.iter().map(|s| s.name).collect();
        assert!(!names.contains(&"version.json"));
        assert!(!names.contains(&"nuclide.arrow"));
        assert!(names.contains(&"products.arrow"));
        assert!(names.contains(&"distributions.arrow"));
        assert!(
            !names.contains(&"fast_xs.arrow"),
            "the transport lookup is built, not fetched"
        );

        let reactions = plan.iter().find(|s| s.name == REACTIONS).unwrap();
        // The schema, then the 294K batch of each of the three MTs. They are
        // separated by the other temperatures, so nothing coalesces.
        assert_eq!(
            reactions.spans,
            Some(vec![(64, 704), (868, 100), (1168, 100), (1468, 100)])
        );
        let energy = plan.iter().find(|s| s.name == ENERGY).unwrap();
        assert_eq!(energy.spans, Some(vec![(64, 384), (548, 50)]));
        for other in plan
            .iter()
            .filter(|s| s.name != REACTIONS && s.name != ENERGY)
        {
            assert_eq!(other.spans, None, "{}", other.name);
        }
    }

    #[test]
    fn a_bracket_pair_ranges_adjacent_batches_into_one_span() {
        let plan = plan_neutron_sections(&version(), Some(&set(&["294K", "600K"]))).unwrap();
        let reactions = plan.iter().find(|s| s.name == REACTIONS).unwrap();
        assert_eq!(
            reactions.spans,
            Some(vec![(64, 704), (868, 200), (1168, 200), (1468, 200)])
        );
        let energy = plan.iter().find(|s| s.name == ENERGY).unwrap();
        assert_eq!(energy.spans, Some(vec![(64, 384), (548, 100)]));
    }

    #[test]
    fn every_temperature_fetches_whole_files() {
        let plan = plan_neutron_sections(&version(), None).unwrap();
        assert!(plan.iter().all(|s| s.spans.is_none()));
        // Wanting every published label is the same as wanting the whole file,
        // and one GET is cheaper than a span per MT.
        let plan =
            plan_neutron_sections(&version(), Some(&set(&["250K", "294K", "600K", "0K"]))).unwrap();
        assert!(plan.iter().all(|s| s.spans.is_none()));
    }

    #[test]
    fn a_version_without_an_index_fetches_whole_files() {
        let version = serde_json::json!({"data_version": "1"});
        let plan = plan_neutron_sections(&version, Some(&set(&["294K"]))).unwrap();
        assert!(plan.iter().all(|s| s.spans.is_none()));
    }

    #[test]
    fn a_label_the_index_lacks_is_an_error_not_an_empty_stream() {
        let err = plan_neutron_sections(&version(), Some(&set(&["900K"]))).unwrap_err();
        assert!(err.contains("900K"), "{err}");
    }

    #[test]
    fn photon_sections_are_whole_and_start_with_the_stamp() {
        let plan = plan_photon_sections();
        assert_eq!(plan[0].name, "version.json");
        assert!(plan.iter().any(|s| s.name == "element.arrow" && s.required));
        assert!(plan.iter().all(|s| s.spans.is_none()));
    }

    #[test]
    fn urls_and_store_paths_follow_the_published_layout() {
        assert_eq!(
            neutron_url("https://x.test/lib/", "Fe56"),
            "https://x.test/lib/neutron/Fe56.arrow"
        );
        assert_eq!(
            photon_url("https://x.test/lib", "Fe"),
            "https://x.test/lib/photon/Fe.arrow"
        );
        assert_eq!(store_dir("Fe56"), "/Fe56.arrow");
    }
}
