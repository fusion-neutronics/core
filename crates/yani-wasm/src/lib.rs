//! WebAssembly bindings for yani: build a material, build an irradiation
//! schedule, transmute, read the inventory back.
//!
//! # The shape of the boundary
//!
//! Everything here is synchronous. `std::fs` and `reqwest` do not work on
//! `wasm32-unknown-unknown`, so the host fetches bytes in JS -- where `await`
//! belongs -- and hands them over with [`YaniSession::add_chain_section`] and
//! [`YaniSession::add_nuclide_data`]. Nothing in Rust does I/O.
//!
//! This mirrors `yamc`'s `WasmSimulation`, including its single-instance
//! caveat: `yamc_nuclide::storage::set_storage` swaps a *process-global*
//! backend and the nuclear-data config is global too, so a second
//! `YaniSession` takes the first one's data with it. One per page.
//!
//! # Why a session rather than free functions
//!
//! A transmutation needs a chain, cross sections for every nuclide the chain
//! reaches, and a material, and the expensive part is the data. Holding them on
//! one object lets the host load once and run many schedules against it, which
//! is what makes the UI feel interactive after the first fetch.

use std::collections::HashMap;
use std::sync::Arc;

use wasm_bindgen::prelude::*;

use yamc_materials::Material;
use yamc_nuclide::storage::in_memory_storage::InMemoryStorage;
use yani::{BranchTable, ChainNuclide, ChainSections};
use yani_transmute::{MultigroupSpectrum, TransmuteStep};

/// The build-up factor the contact dose is reported with.
///
/// 2.0, as the FISPACT-II manual suggests and as `Material.contact_dose`
/// defaults to in the wheels, so a browser run and a Python run of the same
/// inventory agree. It is not a session setting: the page reports one dose
/// series, and a knob whose only effect is to scale it linearly would be a
/// boundary crossing to multiply by a constant.
const CONTACT_DOSE_BUILD_UP: f64 = 2.0;

#[wasm_bindgen(start)]
pub fn wasm_start() {
    console_error_panic_hook::set_once();
}

/// A parsed chain plus its branching overlay.
struct LoadedChain {
    chain: Arc<HashMap<String, ChainNuclide>>,
    branch: BranchTable,
    /// Which optional subsections the host actually uploaded.
    parts: yani::ChainParts,
}

/// The session a page holds for its lifetime.
#[wasm_bindgen]
pub struct YaniSession {
    storage: InMemoryStorage,
    chain_sections: ChainSections,
    loaded: Option<LoadedChain>,
    material: Option<Material>,
}

impl Default for YaniSession {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl YaniSession {
    /// Install the in-memory storage backend and start empty.
    ///
    /// Installing here rather than lazily means every later read goes through
    /// it: the wasm default backend errors on every call, which is the correct
    /// behaviour for a build with no filesystem but a confusing one to hit
    /// halfway through a run.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let storage = InMemoryStorage::new();
        yamc_nuclide::storage::set_storage(Box::new(storage.clone()));
        Self {
            storage,
            chain_sections: ChainSections::default(),
            loaded: None,
            material: None,
        }
    }

    /// Hand over one chain section, e.g. `("decay", "nuclides.arrow", bytes)`.
    ///
    /// Subsections are `decay`, `reactions`, `fission_yields` and `branching`;
    /// an unknown one is an error rather than a silent drop, because a chain
    /// missing a section loads as a transmutation that produces nothing.
    #[wasm_bindgen(js_name = addChainSection)]
    pub fn add_chain_section(
        &mut self,
        subsection: &str,
        file: &str,
        bytes: Vec<u8>,
    ) -> Result<(), String> {
        self.chain_sections
            .insert(subsection, file, bytes)
            .map_err(|e| e.to_string())?;
        // Any previously parsed chain no longer reflects what is loaded.
        self.loaded = None;
        Ok(())
    }

    /// Parse the chain from the sections handed over so far.
    ///
    /// Returns the number of nuclides in it, which is the host's confirmation
    /// that it supplied a real chain rather than an empty one.
    #[wasm_bindgen(js_name = loadChain)]
    pub fn load_chain(&mut self) -> Result<usize, String> {
        let (chain, branch) =
            yani::parse_chain_parts_from_bytes(&self.chain_sections).map_err(|e| e.to_string())?;
        let n = chain.len();
        self.loaded = Some(LoadedChain {
            chain: Arc::new(chain),
            branch,
            // Taken from the bytes rather than assumed complete: a page is free
            // to upload decay and reactions only, and should be told its
            // fission products are being dropped rather than shown a solve
            // that quietly never made them.
            parts: self.chain_sections.parts(),
        });
        Ok(n)
    }

    /// Hand over one section of a nuclide's cross-section data and register it.
    ///
    /// `file` is `nuclide.arrow`, `reactions.arrow` or `version.json`, matching
    /// `NEUTRON_XS_ONLY_SECTIONS`. The bytes land at `/{nuclide}.arrow/{file}`
    /// in the virtual filesystem and the nuclear-data config is pointed at that
    /// directory, which is what `transmute` resolves against.
    #[wasm_bindgen(js_name = addNuclideData)]
    pub fn add_nuclide_data(&self, nuclide: &str, file: &str, bytes: Vec<u8>) {
        self.storage
            .add_file(format!("/{nuclide}.arrow/{file}"), bytes);
        let mut cfg = yamc_nuclide::config::CONFIG
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // Written straight into the map rather than through `set_cross_section`,
        // which validates the path with `std::path::Path::exists` and *panics*
        // when it is absent. Here the path is always absent: it names a file in
        // the virtual filesystem, which the real one knows nothing about. A
        // panic crossing the wasm boundary aborts the module, so the check has
        // to be avoided rather than caught.
        cfg.cross_sections
            .insert(nuclide.to_string(), format!("/{nuclide}.arrow"));
    }

    /// How many files the virtual filesystem currently holds. Diagnostics.
    #[wasm_bindgen(js_name = fileCount)]
    pub fn file_count(&self) -> usize {
        self.storage.len()
    }

    /// Drop the raw cross-section bytes, keeping the parsed nuclides.
    ///
    /// `get_or_load_nuclide` caches the parsed form globally, so once a run has
    /// happened the bytes in the virtual filesystem are dead weight -- and on
    /// wasm32 they are dead weight inside a 4 GB address space shared with
    /// everything else. Calling this after a run roughly halves what a session
    /// holds.
    ///
    /// The nuclear-data config still points at the same paths, so anything not
    /// already parsed will fail to load until the host supplies it again.
    #[wasm_bindgen(js_name = clearNuclideData)]
    pub fn clear_nuclide_data(&mut self) {
        self.storage.clear();
    }

    /// The nuclides a transmutation of the current material would need cross
    /// sections for, as a JSON array the host fetches.
    ///
    /// This is the whole download budget, and it is knowable before any of it
    /// is fetched -- which is what lets the UI show a byte count up front
    /// rather than an open-ended spinner.
    #[wasm_bindgen(js_name = requiredNuclides)]
    pub fn required_nuclides(&self) -> Result<String, String> {
        let loaded = self.chain()?;
        let material = self
            .material
            .as_ref()
            .ok_or("no material set; call pnnlMaterial or buildMaterial first")?;
        let seeds: Vec<String> = material.nuclides.keys().cloned().collect();
        let seed_refs: Vec<&str> = seeds.iter().map(String::as_str).collect();
        let reachable = yani::reachable_nuclides(&loaded.chain, &seed_refs);
        let mut names: Vec<&String> = loaded
            .chain
            .iter()
            .filter(|(name, cn)| !cn.reactions.is_empty() && reachable.contains(*name))
            .map(|(name, _)| name)
            .collect();
        names.sort();
        Ok(
            serde_json::Value::from(names.into_iter().cloned().collect::<Vec<String>>())
                .to_string(),
        )
    }

    /// The MT numbers the loaded chain names, as a sorted JSON array.
    ///
    /// Every reaction a run will ask a cross section for, and nothing else. The
    /// host uses it to fetch only those MTs out of each `reactions.arrow`,
    /// through the byte ranges `version.json` publishes: the full-grid transport
    /// MTs an activation run never reads are most of every file.
    ///
    /// Answered here rather than computed in JS so the page needs no second copy
    /// of the chain-kind-to-MT table. An MT a copy of it missed would not 404,
    /// it would be a reaction rate quietly collapsing to zero.
    ///
    /// A chain-wide union rather than a set per nuclide, which is what
    /// `transmute` itself scopes the load to. It costs no extra bytes: an MT the
    /// union adds for one nuclide simply has no batch in another's file.
    #[wasm_bindgen(js_name = requiredMts)]
    pub fn required_mts(&self) -> Result<String, String> {
        let loaded = self.chain()?;
        let mut mts: Vec<i32> = yani_transmute::activation_mts(&loaded.chain, &loaded.branch)
            .into_iter()
            .collect();
        // Sorted so the answer is stable across runs: it is a HashSet upstream,
        // and a host that diffs or caches it should not see it reshuffle.
        mts.sort_unstable();
        Ok(serde_json::Value::from(mts).to_string())
    }

    /// The energy boundaries of a named group structure, as a JSON array in eV.
    ///
    /// Any of the twelve names
    /// [`yamc_nuclide::group_structures::available_group_structures`] lists, ten
    /// neutron and two photon. Read back from here rather than duplicated in JS
    /// so a name means the same thing in the browser as in the wheel: a second
    /// copy of 710 boundaries is a second copy to drift.
    ///
    /// The tables are compiled in, so this needs no data.
    #[wasm_bindgen(js_name = groupStructure)]
    pub fn group_structure(&self, name: &str) -> Result<String, String> {
        let boundaries = yamc_nuclide::group_structures::get_group_structure(name)?;
        Ok(serde_json::Value::from(boundaries.to_vec()).to_string())
    }

    // ---- materials -------------------------------------------------------

    /// Every material name in the PNNL compendium, as a JSON array.
    ///
    /// The compendium is compiled in, so this needs no data and works offline.
    ///
    /// JSON rather than a joined string because the report's own names contain
    /// commas -- "Steel, Stainless 316" is one name, not two -- so any
    /// separator cheap enough to be worth using is already taken.
    #[wasm_bindgen(js_name = pnnlNames)]
    pub fn pnnl_names() -> String {
        serde_json::Value::from(yamc_materials::collections::pnnl::names()).to_string()
    }

    /// PNNL names matching `query`, case-insensitively, as a JSON array.
    #[wasm_bindgen(js_name = pnnlSearch)]
    pub fn pnnl_search(query: &str) -> String {
        serde_json::Value::from(yamc_materials::collections::pnnl::search(query)).to_string()
    }

    /// Set the current material from the PNNL compendium.
    #[wasm_bindgen(js_name = pnnlMaterial)]
    pub fn pnnl_material(&mut self, name: &str, volume: f64) -> Result<(), String> {
        let mut material = yamc_materials::collections::pnnl::material(name)?;
        material.volume(Some(volume))?;
        self.material = Some(material);
        Ok(())
    }

    /// Set the current material from an explicit composition.
    ///
    /// `composition_json` maps element symbols, nuclide names or chemical
    /// formulas to fractions, e.g. `{"Fe": 0.7, "Cr52": 0.2, "H2O": 0.1}`.
    /// Element symbols expand over natural abundance, nuclide names are taken
    /// literally, formulas are parsed -- the same rules the Python API uses.
    #[wasm_bindgen(js_name = buildMaterial)]
    pub fn build_material(
        &mut self,
        composition_json: &str,
        density: f64,
        density_units: &str,
        fraction_type: &str,
        volume: f64,
    ) -> Result<(), String> {
        let parsed: serde_json::Value =
            serde_json::from_str(composition_json).map_err(|e| format!("composition: {e}"))?;
        let entries = parsed
            .as_object()
            .ok_or("composition must be a JSON object of name -> fraction")?;

        let mut nuclides: HashMap<String, f64> = HashMap::new();
        for (key, value) in entries {
            let fraction = value
                .as_f64()
                .ok_or_else(|| format!("composition['{key}'] is not a number"))?;
            // A nuclide name is taken literally; anything else is an element
            // or a formula and expands.
            //
            // The nuclide check has to come first, because the formula parser
            // accepts nuclide names and means something else by them: "Fe56"
            // parses as Fe with a subscript of 56, which expands over natural
            // abundance and quietly hands back iron rather than the isotope
            // that was asked for. That is an 8.8% error in the atom density of
            // Fe56 alone, and it silently introduces Fe54 and Fe58 into a
            // single-nuclide material.
            //
            // The mass table is what decides, because it is the one check that
            // is strict enough: `validate_nuclide_name` accepts "H2O". The
            // table knows "Fe56" and not "H2O" or "Fe", so a formula and an
            // element still reach the parsers that should have them.
            let expanded = yamc_nuclide::composition::atomic_mass(key)
                .map(|_| HashMap::from([(key.to_string(), fraction)]))
                .or_else(|_| {
                    yamc_nuclide::composition::expand_formula(key, fraction_type, None, None, None)
                        .map(|m| {
                            m.into_iter()
                                .map(|(n, f)| (n, f * fraction))
                                .collect::<HashMap<String, f64>>()
                        })
                })
                .or_else(|_| {
                    yamc_nuclide::composition::expand_element(key, fraction, fraction_type)
                })
                .map_err(|e| format!("composition['{key}']: {e}"))?;
            yamc_nuclide::composition::merge_nuclides(&mut nuclides, &expanded);
        }

        let mut material = Material::new(nuclides, fraction_type, density_units, Some(density))?;
        material.volume(Some(volume))?;
        self.material = Some(material);
        Ok(())
    }

    /// The current material's nuclides and atom densities, as JSON.
    #[wasm_bindgen(js_name = materialJson)]
    pub fn material_json(&self) -> Result<String, String> {
        let material = self.material.as_ref().ok_or("no material set")?;
        let densities = material.get_atoms_per_barn_cm()?;
        Ok(serde_json::json!({
            "name": material.name,
            "density": material.density,
            "density_units": material.density_units.as_str(),
            "volume": material.volume,
            "atoms_per_barn_cm": densities,
        })
        .to_string())
    }

    // ---- running ---------------------------------------------------------

    /// Transmute the current material over a schedule.
    ///
    /// `spectra_json` is an array of `{"boundaries": [...], "values": [...]}`,
    /// energies ascending in eV with one more boundary than value. The values
    /// are a shape and are normalized here, so a multigroup flux can be passed
    /// with its own magnitudes and the total given per step as `rate`.
    ///
    /// `schedule_json` is an array of `{"dt": seconds, "rate": n/cm^2/s,
    /// "spectrum": index}`, where a step with no `spectrum` is decay-only.
    ///
    /// Returns one entry per step: cumulative time, total activity, decay heat
    /// and contact dose rate, each also broken down by nuclide, the atom
    /// densities, and the decay photon line spectrum. Everything the plots
    /// need, in one call, so the host does not pay a boundary crossing per
    /// series.
    pub fn run(&mut self, spectra_json: &str, schedule_json: &str) -> Result<String, String> {
        // Taken apart before `material` below borrows the same `self` mutably.
        // The chain itself is an `Arc`, so only the branching overlay is copied.
        let (chain, branch, parts) = {
            let loaded = self.chain()?;
            (
                Arc::clone(&loaded.chain),
                loaded.branch.clone(),
                loaded.parts,
            )
        };
        // `&mut`, because the solve loads the cross sections it needs into the
        // material and keeps them there, so a second run does no fetching
        // (issue #576, finding 3). In the browser that matters more than
        // anywhere: the data comes over the network.
        let material = self.material.as_mut().ok_or("no material set")?;
        let volume = material
            .volume
            .ok_or("material has no volume; activity and decay heat need cm^3")?;

        let spectra = parse_spectra(spectra_json)?;
        let steps = parse_steps(schedule_json, spectra.len())?;

        let results = yani_transmute::transmute_material(
            material,
            &spectra,
            &steps,
            Arc::clone(&chain),
            &branch,
            parts,
            // No uncertainty in the browser: the covariance tables are large
            // and the wasm build fetches its data over the network.
            None,
        )
        .map_err(|e| e.to_string())?;

        // One entry per schedule step: `results` also carries the initial
        // composition at index 0, which is not a step.
        let step_materials = results.step_materials(material.material_id.unwrap_or(0));
        let mut out = Vec::with_capacity(step_materials.len());
        let mut elapsed = 0.0;
        for (step, mat) in steps.iter().zip(step_materials) {
            elapsed += step.dt;
            let densities = mat.get_atoms_per_barn_cm()?;
            let activity = yani_decay::activity_by_nuclide(&densities, volume, &chain);
            let heat = yani_decay::decay_heat_by_nuclide(&densities, volume, &chain);
            let lines = yani_decay::decay_photon_lines(&densities, volume, &chain);
            // Unlike the three above, this one takes no volume: the slab
            // estimate is intensive. It can fail -- on an element with no
            // attenuation data -- and that propagates rather than reading as
            // zero, because a missing element understates the self-shielding
            // and so overstates the dose.
            let contact_dose = yani_decay::contact_dose_by_nuclide(
                &densities,
                &chain,
                yani_decay::DoseQuantity::AbsorbedAir,
                CONTACT_DOSE_BUILD_UP,
            )?;
            out.push(serde_json::json!({
                "time": elapsed,
                "irradiated": step.irradiation.is_some(),
                // Through `yani_decay::total`, which sorts, rather than
                // summing the map directly: `HashMap` order is a property of
                // the instance, so a total taken in it moves in the last bit
                // between runs. The Python bindings have always gone through
                // it; this session was never updated (issue #558).
                "activity": yani_decay::total(&activity),
                "decay_heat": yani_decay::total(&heat),
                "activity_by_nuclide": activity,
                "decay_heat_by_nuclide": heat,
                "atoms_per_barn_cm": densities,
                "photon_energy": lines.iter().map(|(e, _)| *e).collect::<Vec<f64>>(),
                "photon_intensity": lines.iter().map(|(_, i)| *i).collect::<Vec<f64>>(),
                "contact_dose": yani_decay::total(&contact_dose),
                "contact_dose_by_nuclide": contact_dose,
            }));
        }
        Ok(serde_json::Value::Array(out).to_string())
    }
}

impl YaniSession {
    fn chain(&self) -> Result<&LoadedChain, String> {
        self.loaded
            .as_ref()
            .ok_or_else(|| "no chain loaded; call loadChain first".to_string())
    }
}

/// Seconds for a `(value, unit)` duration, using the same spellings the Python
/// API accepts (`s`, `min`, `h`, `d`, `a`, and their long forms).
#[wasm_bindgen(js_name = durationToSeconds)]
pub fn duration_to_seconds(value: f64, unit: &str) -> Result<f64, String> {
    yani_transmute::duration_to_seconds(value, unit)
}

fn parse_spectra(json: &str) -> Result<Vec<MultigroupSpectrum>, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("spectra: {e}"))?;
    let array = parsed.as_array().ok_or("spectra must be a JSON array")?;
    let mut out = Vec::with_capacity(array.len());
    for (i, entry) in array.iter().enumerate() {
        let floats = |key: &str| -> Result<Vec<f64>, String> {
            entry
                .get(key)
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("spectrum {i} has no '{key}' array"))?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| format!("spectrum {i}: '{key}' holds a non-number"))
                })
                .collect()
        };
        let boundaries = floats("boundaries")?;
        let values = floats("values")?;
        // Normalizing here is what lets a caller pass a flux with its own
        // magnitudes and give the total separately as the step's rate.
        let total: f64 = values.iter().sum();
        if total <= 0.0 {
            return Err(format!(
                "spectrum {i}: values sum to {total}, must be positive"
            ));
        }
        out.push(MultigroupSpectrum {
            boundaries,
            masses: values.iter().map(|v| v / total).collect(),
            relative_std_dev: None,
        });
    }
    Ok(out)
}

fn parse_steps(json: &str, spectra: usize) -> Result<Vec<TransmuteStep>, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("schedule: {e}"))?;
    let array = parsed.as_array().ok_or("schedule must be a JSON array")?;
    if array.is_empty() {
        return Err("schedule has no steps".to_string());
    }
    let mut out = Vec::with_capacity(array.len());
    for (i, entry) in array.iter().enumerate() {
        let dt = entry
            .get("dt")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| format!("step {i} has no numeric 'dt'"))?;
        let irradiation = match entry.get("spectrum").and_then(serde_json::Value::as_u64) {
            Some(index) => {
                let index = index as usize;
                if index >= spectra {
                    return Err(format!(
                        "step {i}: spectrum index {index} out of range ({spectra} spectra)"
                    ));
                }
                let rate = entry
                    .get("rate")
                    .and_then(serde_json::Value::as_f64)
                    .ok_or_else(|| format!("step {i} names a spectrum but has no 'rate'"))?;
                Some((index, rate))
            }
            None => None,
        };
        out.push(TransmuteStep { dt, irradiation });
    }
    Ok(out)
}
