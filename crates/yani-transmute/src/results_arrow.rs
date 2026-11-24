//! Arrow IPC save/load for [`TransmutationResults`].
//!
//! File format (version 1, lossless):
//!
//! - One Arrow IPC file per result; plain IPC, readable with pyarrow / polars.
//!   Every file has the same schema, so a glob of them scans as one table.
//! - One long table, one row per stored number:
//!   `material_id`, `material_name`, `step`, `quantity`, `nuclide`,
//!   `reaction`, `product`, `value`. `quantity` is `density` (a step's
//!   inventory, atoms per barn-cm), `reaction_rate` (a step's per-edge rate:
//!   parent `nuclide`, `reaction` kind, `product` or null) or `source_rate`
//!   (a step's flux magnitude). Rows go by material id, then step, then
//!   nuclide name, so the same result always writes the same file; a parent's
//!   products keep the order the solve built them in.
//! - Schema-level metadata: the format version, the writing build, and the
//!   rest of the result as JSON (see [`Meta`]), objects sorted and floats
//!   written so they parse back to the same bits. A non-finite float that
//!   JSON would turn into `null` is refused on save rather than lost.
//!
//! The round trip is lossless for everything the result stores. Two things it
//! does not store are data the solve read rather than results: the
//! transmutation chain and the nuclear data loaded on each material's initial
//! state. Those are what [`TransmutationResults::get_production_routes`],
//! [`TransmutationResults::get_isomeric_branching`] and
//! [`TransmutationResults::get_reaction_rate_spectrum`] re-derive from, and a
//! loaded result answers them once the same data is reattached. The file pins
//! that data rather than storing it: the chain by a sha256 of its contents,
//! checked by [`TransmutationResults::reattach_chain`], and each initial-state
//! nuclide by the file it was read from and what of it was read, which
//! [`TransmutationResults::reattach_nuclide_data`] reads again (a library
//! keyword's files sit in a folder of their release, so this is that
//! release). Files from another format version are refused (no backwards
//! compatibility).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::{Array, Float64Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use yamc_materials::material::Material;

use yamc_nuclide::load_scope::{LoadScope, SectionScope};
use yamc_nuclide::nuclide::Nuclide;

use crate::results::{CollapseInputs, CollapseReuse, TransmutationResults};

const COL_MATERIAL_ID: &str = "material_id";
const COL_MATERIAL_NAME: &str = "material_name";
const COL_STEP: &str = "step";
const COL_QUANTITY: &str = "quantity";
const COL_NUCLIDE: &str = "nuclide";
const COL_REACTION: &str = "reaction";
const COL_PRODUCT: &str = "product";
const COL_VALUE: &str = "value";

const QUANTITY_DENSITY: &str = "density";
const QUANTITY_REACTION_RATE: &str = "reaction_rate";
const QUANTITY_SOURCE_RATE: &str = "source_rate";

const META_FORMAT: &str = "yani_transmutation_results_format";
const META_VERSION: &str = "yani_version";
const META_RESULT: &str = "result";

/// Current on-disk format version. Readers refuse anything else.
const FORMAT_VERSION: &str = "1";

/// Everything the result holds apart from the table's numbers.
#[derive(serde::Serialize, serde::Deserialize)]
struct Meta {
    times: Vec<f64>,
    timesteps: Vec<f64>,
    /// Each material's states without their nuclides (the table holds those),
    /// in step order.
    states: BTreeMap<u32, Vec<Material>>,
    uncertainty: BTreeMap<u32, crate::uncertainty::Ensemble>,
    uncertainty_info: BTreeMap<u32, crate::uncertainty::Info>,
    shielding_info: BTreeMap<u32, crate::self_shielding::ShieldingInfo>,
    collapse: BTreeMap<u32, CollapseInputs>,
    collapse_reuse: Option<CollapseReuse>,
    rate_covariance: BTreeMap<u32, crate::history_statistics::RateCovariance>,
    branching_report: BTreeMap<u32, Vec<Arc<crate::branching_rule::BranchingReport>>>,
    data_releases: BTreeMap<String, yamc_nuclide::storage::release::DataRelease>,
    displacement_damage: BTreeMap<u32, crate::damage::DisplacementDamage>,
    product_screening: BTreeMap<u32, crate::screening::ProductScreening>,
    data_gaps: BTreeMap<u32, crate::data_gaps::DataGaps>,
    chain_gaps: Option<Arc<crate::data_gaps::ChainGaps>>,
    decay_library: Option<String>,
    /// How many per-step entries each material's reaction rates and source
    /// rates hold. A decay-only step's rate map is empty and writes no rows,
    /// so the lengths are kept here rather than read off the table.
    reaction_rate_steps: BTreeMap<u32, usize>,
    source_rate_steps: BTreeMap<u32, usize>,
    /// The sha256 of the run's chain ([`chain_sha256`]), `None` when it had
    /// none.
    chain_sha256: Option<String>,
    /// Where each material's initial-state nuclear data was read from; only
    /// materials that carried some.
    nuclide_sources: BTreeMap<u32, Vec<NuclideSource>>,
}

/// What a result read back from a file lacks of what its run held, and what
/// pins it.
///
/// [`TransmutationResults::get_production_routes`],
/// [`TransmutationResults::get_isomeric_branching`] (the chain) and
/// [`TransmutationResults::get_reaction_rate_spectrum`] (the initial state's
/// nuclear data) answer once it is reattached.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Detached {
    /// The run's chain's sha256 while the chain is not attached.
    chain_sha256: Option<String>,
    /// The materials whose initial-state data is not attached.
    nuclide_sources: BTreeMap<u32, Vec<NuclideSource>>,
    /// Every material's record as the file had it, kept once reattached: the
    /// data read again can be a wider read of the same file (the nuclide
    /// cache hands back what it holds when that covers the request), and
    /// saving again should record what the run read, not that.
    recorded: BTreeMap<u32, Vec<NuclideSource>>,
}

impl Detached {
    /// Whether the run's chain is not attached.
    pub fn chain(&self) -> bool {
        self.chain_sha256.is_some()
    }

    /// The materials whose initial-state nuclear data is not attached.
    pub fn nuclide_data(&self) -> Vec<u32> {
        self.nuclide_sources.keys().copied().collect()
    }

    fn is_empty(&self) -> bool {
        self.chain_sha256.is_none() && self.nuclide_sources.is_empty()
    }
}

/// One initial-state nuclide's data: the file it was read from and the part
/// of it that was read, enough to read the same again.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub(crate) struct NuclideSource {
    name: String,
    /// `None` for data that was not read from a file.
    path: Option<String>,
    full: bool,
    mts: Option<Vec<i32>>,
    temperatures: Option<Vec<String>>,
    covariance: bool,
    angular_covariance: bool,
    fission_covariance: bool,
    resonance_parameters: bool,
}

impl NuclideSource {
    fn of(name: &str, nuclide: &Nuclide) -> Self {
        // Destructured whole, so a field added to `LoadScope` fails to compile
        // here rather than being left out of the record.
        let LoadScope {
            sections,
            mts,
            temperatures,
            covariance,
            angular_covariance,
            fission_covariance,
            resonance_parameters,
        } = &nuclide.load_scope;
        fn sorted<T: Ord + Clone>(set: &std::collections::HashSet<T>) -> Vec<T> {
            let mut v: Vec<T> = set.iter().cloned().collect();
            v.sort();
            v
        }
        Self {
            name: name.to_string(),
            path: nuclide.data_path.clone(),
            full: *sections == SectionScope::Full,
            mts: mts.as_ref().map(sorted),
            temperatures: temperatures.as_ref().map(sorted),
            covariance: *covariance,
            angular_covariance: *angular_covariance,
            fission_covariance: *fission_covariance,
            resonance_parameters: *resonance_parameters,
        }
    }

    fn scope(&self) -> LoadScope {
        LoadScope {
            sections: if self.full {
                SectionScope::Full
            } else {
                SectionScope::XsOnly
            },
            mts: self.mts.as_ref().map(|v| v.iter().copied().collect()),
            temperatures: self
                .temperatures
                .as_ref()
                .map(|v| v.iter().cloned().collect()),
            covariance: self.covariance,
            angular_covariance: self.angular_covariance,
            fission_covariance: self.fission_covariance,
            resonance_parameters: self.resonance_parameters,
        }
    }

    fn load(&self) -> Result<Arc<Nuclide>, String> {
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| format!("{} was not read from a file", self.name))?;
        if !std::path::Path::new(path).exists() {
            return Err(format!("{}: {path} is no longer there", self.name));
        }
        let paths = HashMap::from([(self.name.clone(), path.clone())]);
        yamc_nuclide::nuclide::get_or_load_nuclide(&self.name, &paths, &self.scope()).map_err(|e| {
            format!(
                "{}: {}",
                self.name,
                e.to_string().lines().next().unwrap_or("")
            )
        })
    }
}

/// A sha256 of a chain's contents, hex: every nuclide's full record, in name
/// order, so it is the same for the same data whatever order it was read in.
pub fn chain_sha256(chain: &HashMap<String, yani::ChainNuclide>) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    struct Hasher(Sha256);
    impl Write for Hasher {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0.update(s.as_bytes());
            Ok(())
        }
    }
    let mut names: Vec<&String> = chain.keys().collect();
    names.sort_unstable();
    let mut hasher = Hasher(Sha256::new());
    for name in names {
        // `Debug` writes every field, floats to the digits that parse back to
        // the same bits, and the chain holds no hash-ordered collection.
        let _ = writeln!(hasher, "{:?}", chain[name]);
    }
    hasher
        .0
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn schema(metadata: HashMap<String, String>) -> Schema {
    Schema::new(vec![
        Field::new(COL_MATERIAL_ID, DataType::UInt32, false),
        Field::new(COL_MATERIAL_NAME, DataType::Utf8, true),
        Field::new(COL_STEP, DataType::UInt32, false),
        Field::new(COL_QUANTITY, DataType::Utf8, false),
        Field::new(COL_NUCLIDE, DataType::Utf8, true),
        Field::new(COL_REACTION, DataType::Utf8, true),
        Field::new(COL_PRODUCT, DataType::Utf8, true),
        Field::new(COL_VALUE, DataType::Float64, false),
    ])
    .with_metadata(metadata)
}

/// The table's columns, filled row by row.
#[derive(Default)]
struct Rows {
    material_id: Vec<u32>,
    material_name: Vec<Option<String>>,
    step: Vec<u32>,
    quantity: Vec<&'static str>,
    nuclide: Vec<Option<String>>,
    reaction: Vec<Option<String>>,
    product: Vec<Option<String>>,
    value: Vec<f64>,
}

impl Rows {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        id: u32,
        name: &Option<String>,
        step: usize,
        quantity: &'static str,
        nuclide: Option<&str>,
        reaction: Option<&str>,
        product: Option<&str>,
        value: f64,
    ) -> Result<(), String> {
        self.material_id.push(id);
        self.material_name.push(name.clone());
        self.step
            .push(u32::try_from(step).map_err(|_| format!("step {step} does not fit a u32"))?);
        self.quantity.push(quantity);
        self.nuclide.push(nuclide.map(str::to_string));
        self.reaction.push(reaction.map(str::to_string));
        self.product.push(product.map(str::to_string));
        self.value.push(value);
        Ok(())
    }
}

/// A material's ids in a deterministic order: the order the result was built
/// in is not kept by its `HashMap`s, so the table is written by ascending id.
fn sorted_ids(results: &TransmutationResults) -> Vec<u32> {
    let mut ids: Vec<u32> = results.materials.keys().copied().collect();
    ids.sort_unstable();
    ids
}

/// A nuclide map's entries by name, so the table does not depend on hash
/// order.
fn by_name<V>(map: &HashMap<String, V>) -> Vec<(&String, &V)> {
    let mut entries: Vec<(&String, &V)> = map.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
}

impl TransmutationResults {
    /// The result as the bytes of an Arrow IPC file. See the module
    /// documentation for the format.
    pub fn to_arrow_bytes(&self) -> Result<Vec<u8>, String> {
        let ids = sorted_ids(self);
        let mut rows = Rows::default();
        let mut states: BTreeMap<u32, Vec<Material>> = BTreeMap::new();
        let mut nuclide_sources: BTreeMap<u32, Vec<NuclideSource>> = BTreeMap::new();
        for &id in &ids {
            let steps = &self.materials[&id];
            let mut definitions = Vec::with_capacity(steps.len());
            for (step, material) in steps.iter().enumerate() {
                for (nuclide, density) in by_name(&material.nuclides) {
                    rows.push(
                        id,
                        &material.name,
                        step,
                        QUANTITY_DENSITY,
                        Some(nuclide),
                        None,
                        None,
                        *density,
                    )?;
                }
                let mut definition = material.clone();
                definition.nuclides.clear();
                definition.nuclide_data.clear();
                definitions.push(definition);
            }
            let recorded = self.detached.as_ref().and_then(|d| d.recorded.get(&id));
            if let Some(sources) = recorded {
                nuclide_sources.insert(id, sources.clone());
            } else if let Some(initial) = steps.first().filter(|m| !m.nuclide_data.is_empty()) {
                let sources = by_name(&initial.nuclide_data)
                    .into_iter()
                    .map(|(name, nuclide)| NuclideSource::of(name, nuclide))
                    .collect();
                nuclide_sources.insert(id, sources);
            }
            states.insert(id, definitions);
            let name = &steps.first().and_then(|m| m.name.clone());
            if let Some(rates) = self.reaction_rates.get(&id) {
                for (step, edges) in rates.iter().enumerate() {
                    for (parent, kinds) in by_name(edges) {
                        for (kind, products) in by_name(kinds) {
                            for (product, rate) in products {
                                rows.push(
                                    id,
                                    name,
                                    step,
                                    QUANTITY_REACTION_RATE,
                                    Some(parent),
                                    Some(kind),
                                    product.as_deref(),
                                    *rate,
                                )?;
                            }
                        }
                    }
                }
            }
            if let Some(source) = self.source_rates.get(&id) {
                for (step, rate) in source.iter().enumerate() {
                    rows.push(
                        id,
                        name,
                        step,
                        QUANTITY_SOURCE_RATE,
                        None,
                        None,
                        None,
                        *rate,
                    )?;
                }
            }
        }
        let sorted = |m: &HashMap<u32, _>| -> BTreeMap<u32, _> { m.clone().into_iter().collect() };
        let meta = Meta {
            times: self.times.clone(),
            timesteps: self.timesteps.clone(),
            states,
            uncertainty: self.uncertainty.clone().into_iter().collect(),
            uncertainty_info: self.uncertainty_info.clone().into_iter().collect(),
            shielding_info: sorted(&self.shielding_info),
            collapse: self.collapse.clone().into_iter().collect(),
            collapse_reuse: self.collapse_reuse,
            rate_covariance: self.rate_covariance.clone().into_iter().collect(),
            branching_report: self.branching_report.clone().into_iter().collect(),
            data_releases: self.data_releases.clone(),
            displacement_damage: self.displacement_damage.clone().into_iter().collect(),
            product_screening: self.product_screening.clone().into_iter().collect(),
            data_gaps: self.data_gaps.clone().into_iter().collect(),
            chain_gaps: self.chain_gaps().cloned(),
            decay_library: self.decay_library().map(str::to_string),
            reaction_rate_steps: self
                .reaction_rates
                .iter()
                .map(|(id, r)| (*id, r.len()))
                .collect(),
            source_rate_steps: self
                .source_rates
                .iter()
                .map(|(id, r)| (*id, r.len()))
                .collect(),
            // A loaded result keeps the record of what it was never
            // reattached to, so saving it again loses nothing.
            chain_sha256: match &self.chain {
                Some(chain) => Some(chain_sha256(chain)),
                None => self.detached.as_ref().and_then(|d| d.chain_sha256.clone()),
            },
            nuclide_sources,
        };
        let non_finite = crate::serde_remote::non_finite(&meta);
        if !non_finite.is_empty() {
            return Err(format!(
                "the result holds {} non-finite number(s), which the file cannot hold: {}",
                non_finite.len(),
                non_finite.join(", ")
            ));
        }
        // Through a `Value` with its objects sorted: the result keeps much of
        // itself in `HashMap`s, and written straight their keys would come out
        // in hash order, so the same result would not write the same bytes.
        let mut json =
            serde_json::to_value(&meta).map_err(|e| format!("failed to encode the result: {e}"))?;
        json.sort_all_objects();
        let json = json.to_string();
        let metadata = HashMap::from([
            (META_FORMAT.to_string(), FORMAT_VERSION.to_string()),
            (
                META_VERSION.to_string(),
                env!("CARGO_PKG_VERSION").to_string(),
            ),
            (META_RESULT.to_string(), json),
        ]);
        let schema = Arc::new(schema(metadata));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(UInt32Array::from(rows.material_id)),
                Arc::new(StringArray::from(rows.material_name)),
                Arc::new(UInt32Array::from(rows.step)),
                Arc::new(StringArray::from(rows.quantity)),
                Arc::new(StringArray::from(rows.nuclide)),
                Arc::new(StringArray::from(rows.reaction)),
                Arc::new(StringArray::from(rows.product)),
                Arc::new(Float64Array::from(rows.value)),
            ],
        )
        .map_err(|e| format!("failed to build the result table: {e}"))?;
        let mut bytes = Vec::new();
        {
            let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut bytes, &schema)
                .map_err(|e| format!("failed to start the Arrow file: {e}"))?;
            writer
                .write(&batch)
                .map_err(|e| format!("failed to write the result table: {e}"))?;
            writer
                .finish()
                .map_err(|e| format!("failed to finish the Arrow file: {e}"))?;
        }
        Ok(bytes)
    }

    /// Write the result to an Arrow IPC file at `path`.
    pub fn to_arrow(&self, path: &std::path::Path) -> Result<(), String> {
        let bytes = self.to_arrow_bytes()?;
        std::fs::write(path, bytes).map_err(|e| format!("failed to write {}: {e}", path.display()))
    }

    /// A result back from the bytes [`Self::to_arrow_bytes`] wrote.
    ///
    /// The chain and each initial state's nuclear data are not in the file;
    /// see [`Self::reattach`].
    pub fn from_arrow_bytes(bytes: &[u8]) -> Result<Self, String> {
        let reader = arrow_ipc::reader::FileReader::try_new(std::io::Cursor::new(bytes), None)
            .map_err(|e| format!("not a yani transmutation results file: {e}"))?;
        let schema = reader.schema();
        let metadata = schema.metadata();
        match metadata.get(META_FORMAT).map(String::as_str) {
            Some(FORMAT_VERSION) => {}
            Some(other) => {
                return Err(format!(
                    "transmutation results file format {other} is not read by this build \
                     (it reads format {FORMAT_VERSION})"
                ))
            }
            None => return Err("not a yani transmutation results file: no format version".into()),
        }
        let json = metadata
            .get(META_RESULT)
            .ok_or("transmutation results file has no result metadata")?;
        let meta: Meta =
            serde_json::from_str(json).map_err(|e| format!("failed to decode the result: {e}"))?;

        let mut results = TransmutationResults::new(meta.timesteps.clone());
        results.times = meta.times;
        let mut materials: HashMap<u32, Vec<Material>> = meta.states.into_iter().collect();
        let mut reaction_rates: HashMap<u32, Vec<yani::EdgeRates>> = meta
            .reaction_rate_steps
            .iter()
            .map(|(id, n)| (*id, vec![yani::EdgeRates::default(); *n]))
            .collect();
        let mut source_rates: HashMap<u32, Vec<f64>> = meta
            .source_rate_steps
            .iter()
            .map(|(id, n)| (*id, vec![0.0; *n]))
            .collect();
        for batch in reader {
            let batch = batch.map_err(|e| format!("failed to read the result table: {e}"))?;
            let column = |name: &str| {
                batch
                    .column_by_name(name)
                    .ok_or_else(|| format!("result table has no column {name}"))
            };
            let ids = column(COL_MATERIAL_ID)?
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or("material_id is not u32")?;
            let steps = column(COL_STEP)?
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or("step is not u32")?;
            let strings = |name: &str| -> Result<&StringArray, String> {
                column(name)?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| format!("{name} is not a string column"))
            };
            let quantity = strings(COL_QUANTITY)?;
            let nuclide = strings(COL_NUCLIDE)?;
            let reaction = strings(COL_REACTION)?;
            let product = strings(COL_PRODUCT)?;
            let value = column(COL_VALUE)?
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or("value is not f64")?;
            let text = |a: &StringArray, i: usize| (!a.is_null(i)).then(|| a.value(i).to_string());
            for i in 0..batch.num_rows() {
                let id = ids.value(i);
                let step = steps.value(i) as usize;
                let v = value.value(i);
                match quantity.value(i) {
                    QUANTITY_DENSITY => {
                        let state = materials
                            .get_mut(&id)
                            .and_then(|s| s.get_mut(step))
                            .ok_or_else(|| format!("density row for unknown state {id}/{step}"))?;
                        let name = text(nuclide, i).ok_or("density row without a nuclide")?;
                        state.nuclides.insert(name, v);
                    }
                    QUANTITY_REACTION_RATE => {
                        let parent =
                            text(nuclide, i).ok_or("reaction rate row without a parent")?;
                        let kind =
                            text(reaction, i).ok_or("reaction rate row without a reaction")?;
                        reaction_rates
                            .get_mut(&id)
                            .and_then(|s| s.get_mut(step))
                            .ok_or_else(|| {
                                format!("reaction rate row for unknown step {id}/{step}")
                            })?
                            .entry(parent)
                            .or_default()
                            .entry(kind)
                            .or_default()
                            .push((text(product, i), v));
                    }
                    QUANTITY_SOURCE_RATE => {
                        *source_rates
                            .get_mut(&id)
                            .and_then(|s| s.get_mut(step))
                            .ok_or_else(|| {
                                format!("source rate row for unknown step {id}/{step}")
                            })? = v;
                    }
                    other => return Err(format!("unknown quantity {other:?} in the result table")),
                }
            }
        }
        results.materials = std::mem::take(&mut materials);
        results.reaction_rates = reaction_rates;
        results.source_rates = source_rates;
        results.uncertainty = meta.uncertainty.into_iter().collect();
        results.uncertainty_info = meta.uncertainty_info.into_iter().collect();
        results.shielding_info = meta.shielding_info.into_iter().collect();
        results.collapse = meta.collapse.into_iter().collect();
        results.collapse_reuse = meta.collapse_reuse;
        results.rate_covariance = meta.rate_covariance.into_iter().collect();
        results.branching_report = meta.branching_report.into_iter().collect();
        results.data_releases = meta.data_releases;
        results.displacement_damage = meta.displacement_damage.into_iter().collect();
        results.product_screening = meta.product_screening.into_iter().collect();
        results.data_gaps = meta.data_gaps.into_iter().collect();
        results.restore_chain_record(meta.chain_gaps, meta.decay_library);
        results.detached = Some(Detached {
            chain_sha256: meta.chain_sha256,
            recorded: meta.nuclide_sources.clone(),
            nuclide_sources: meta.nuclide_sources,
        });
        Ok(results)
    }

    /// On a result read back from a file, what it lacks of what its run held;
    /// `None` on a result a run returned, and once everything is reattached.
    pub fn detached(&self) -> Option<&Detached> {
        self.detached.as_ref().filter(|d| !d.is_empty())
    }

    /// Give a result read back from a file the chain its run read, so
    /// [`Self::get_production_routes`] and [`Self::get_isomeric_branching`]
    /// answer again. Refused, and nothing changed, unless `chain` is the same
    /// data: its [`chain_sha256`] must be the one the file recorded.
    pub fn reattach_chain(
        &mut self,
        chain: Arc<HashMap<String, yani::ChainNuclide>>,
    ) -> Result<(), String> {
        let Some(expected) = self.detached.as_ref().and_then(|d| d.chain_sha256.as_ref()) else {
            return Err("this result has its chain".into());
        };
        let found = chain_sha256(&chain);
        if &found != expected {
            return Err(format!(
                "the configured transmutation data is not the chain this result was \
                 computed with (sha256 {found}, the run's {expected})"
            ));
        }
        if let Some(d) = self.detached.as_mut() {
            d.chain_sha256 = None;
        }
        self.chain = Some(chain);
        Ok(())
    }

    /// Give a result read back from a file the data its run read, as far
    /// as it can be found: the chain assembled from the configured
    /// transmutation sources ([`Self::reattach_chain`], so only if it is the
    /// same data) and each initial state's nuclear data read again from its
    /// files ([`Self::reattach_nuclide_data`]). Returns what could not be
    /// reattached and why; empty when everything was.
    pub fn reattach(&mut self) -> Vec<String> {
        let mut problems = Vec::new();
        if self.detached().is_some_and(|d| d.chain()) {
            // The run read the reactions only if it irradiated, and then it
            // kept rates; a decay-only run reads the decay data alone.
            let irradiated = self
                .reaction_rates
                .values()
                .any(|steps| steps.iter().any(|edges| !edges.is_empty()));
            let chain_use = yamc_nuclide::config::ChainUse::transmutation(irradiated);
            // The guard is dropped before the load, which reads the config too.
            let sources = yamc_nuclide::config::Config::global().transmutation_sources(chain_use);
            let loaded =
                sources.and_then(|s| crate::load_chain_sources(&s).map_err(|e| e.to_string()));
            match loaded.and_then(|loaded| self.reattach_chain(loaded.chain)) {
                Ok(()) => {}
                Err(e) => problems.push(format!("chain: {e}")),
            }
        }
        problems.extend(self.reattach_nuclide_data());
        problems
    }

    /// Read each detached material's initial-state nuclear data again from
    /// the files its run read it from, so [`Self::get_reaction_rate_spectrum`]
    /// answers again. A material whose files cannot all be read is left
    /// detached; the returned messages say which and why.
    pub fn reattach_nuclide_data(&mut self) -> Vec<String> {
        let Some(detached) = self.detached.as_mut() else {
            return Vec::new();
        };
        let mut problems = Vec::new();
        let mut done = Vec::new();
        for (id, sources) in &detached.nuclide_sources {
            let Some(initial) = self.materials.get_mut(id).and_then(|s| s.first_mut()) else {
                continue;
            };
            let loaded: Result<HashMap<String, Arc<Nuclide>>, String> = sources
                .iter()
                .map(|source| Ok((source.name.clone(), source.load()?)))
                .collect();
            let outcome = loaded.and_then(|data| {
                initial.nuclide_data = data;
                // A temperature the files only bracket is built by blending,
                // as the run built it, rather than read.
                let temperature = initial.temperature().to_string();
                initial
                    .ensure_temperature_loaded(&temperature)
                    .map_err(|e| e.to_string())
            });
            match outcome {
                Ok(()) => done.push(*id),
                Err(e) => {
                    initial.nuclide_data.clear();
                    problems.push(format!("material {id}: {e}"));
                }
            }
        }
        for id in done {
            detached.nuclide_sources.remove(&id);
        }
        problems
    }

    /// A result back from the Arrow IPC file at `path`.
    pub fn from_arrow(path: &std::path::Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        Self::from_arrow_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yani::EdgeRates;

    fn chain(branching: f64) -> Arc<HashMap<String, yani::ChainNuclide>> {
        let nuclide =
            |name: &str, reactions: Vec<yani::ChainReaction>, decays: Vec<yani::ChainReaction>| {
                yani::ChainNuclide {
                    name: name.to_string(),
                    half_life: None,
                    half_life_uncertainty: None,
                    decay_energy: 0.0,
                    decay_energy_uncertainty: None,
                    decay_evaluated: true,
                    excitation_energy: None,
                    level_index: None,
                    decay_energy_components: Default::default(),
                    reactions: reactions.into(),
                    decays,
                    fission_yields: None,
                    sources: Default::default(),
                }
            };
        let rx = |kind: &str, target: &str| yani::ChainReaction {
            kind: kind.to_string(),
            target: Some(target.to_string()),
            branching,
            q_value: Some(0.0),
            branching_uncertainty: None,
            evaluated_branching: None,
            multiplicity: None,
        };
        Arc::new(HashMap::from([
            (
                "W186".to_string(),
                nuclide(
                    "W186",
                    vec![rx("(n,2n)", "W185"), rx("(n,2n)", "W185_m1")],
                    vec![],
                ),
            ),
            (
                "W185_m1".to_string(),
                nuclide("W185_m1", vec![], vec![rx("it", "W185")]),
            ),
            ("W185".to_string(), nuclide("W185", vec![], vec![])),
        ]))
    }

    fn material(nuclides: &[(&str, f64)]) -> Material {
        let mut m = Material::new(
            nuclides.iter().map(|(n, d)| (n.to_string(), *d)).collect(),
            "atom",
            "g/cm3",
            Some(19.3),
        )
        .expect("material");
        m.name = Some("foil".to_string());
        m
    }

    /// Two materials, an irradiation then a cooldown, and the chain set.
    fn result() -> TransmutationResults {
        let mut results = TransmutationResults::new(vec![300.0, 3600.0]);
        results.set_chain(chain(1.0), Some("endf-b8.1".to_string()));
        for id in [7, 3] {
            results.add_initial(
                id,
                material(&[("W186", 1.0e-3), ("W184", 2.0e-3)]),
                vec![1.0e10, 0.0],
            );
            let mut rates = EdgeRates::new();
            rates.insert(
                "W186".to_string(),
                HashMap::from([(
                    "(n,2n)".to_string(),
                    // Not in name order: the order the solve built is kept.
                    vec![
                        (Some("W185_m1".to_string()), 3.0e-12),
                        (Some("W185".to_string()), 0.1 + 0.2),
                    ],
                )]),
            );
            results.add_step_rates(id, rates);
            results.add_step(
                id,
                material(&[("W186", 9.9e-4), ("W185", 1.0e-9), ("W185_m1", 1.0 / 3.0)]),
            );
            results.add_step_rates(id, EdgeRates::new());
            results.add_step(id, material(&[("W186", 9.9e-4), ("W185", 1.2e-9)]));
        }
        results
    }

    fn routes(r: &TransmutationResults) -> Option<Vec<String>> {
        r.get_production_routes(7, "W185", 0, 1, 3).map(|routes| {
            routes
                .iter()
                .map(|x| format!("{} {:?}", x.text(), x.share))
                .collect()
        })
    }

    #[test]
    fn a_saved_result_loads_back_the_same() {
        let original = result();
        let bytes = original.to_arrow_bytes().expect("save");
        let loaded = TransmutationResults::from_arrow_bytes(&bytes).expect("load");

        assert_eq!(loaded.times, original.times);
        assert_eq!(loaded.timesteps, original.timesteps);
        assert_eq!(loaded.source_rates, original.source_rates);
        // Bit for bit, including a value with no short decimal form.
        assert_eq!(loaded.reaction_rates, original.reaction_rates);
        for (id, states) in &original.materials {
            let back = &loaded.materials[id];
            assert_eq!(back.len(), states.len());
            for (a, b) in states.iter().zip(back) {
                assert_eq!(a.nuclides, b.nuclides);
                assert_eq!(a.name, b.name);
                assert_eq!(a.density, b.density);
            }
        }
        assert_eq!(loaded.decay_library(), original.decay_library());
        // And it saves to the same bytes, so a file is a fixed point.
        assert_eq!(loaded.to_arrow_bytes().expect("save again"), bytes);
        assert_eq!(original.to_arrow_bytes().expect("save twice"), bytes);
    }

    #[test]
    fn the_chain_is_pinned_not_stored() {
        let original = result();
        let bytes = original.to_arrow_bytes().expect("save");
        let mut loaded = TransmutationResults::from_arrow_bytes(&bytes).expect("load");
        assert!(loaded.detached().is_some_and(|d| d.chain()));
        assert_eq!(routes(&loaded), None, "no chain, no routes");

        // Other data is refused and changes nothing.
        let error = loaded
            .reattach_chain(chain(0.5))
            .expect_err("a different chain");
        assert!(error.contains("not the chain"), "{error}");
        assert!(loaded.detached().is_some_and(|d| d.chain()));

        loaded.reattach_chain(chain(1.0)).expect("the same chain");
        assert!(loaded.detached().is_none());
        assert_eq!(routes(&loaded), routes(&original));
        assert!(routes(&original).is_some());
        assert_eq!(loaded.to_arrow_bytes().expect("save again"), bytes);
    }

    #[test]
    fn the_chain_fingerprint_ignores_read_order() {
        let a = chain(1.0);
        // Built in the opposite name order, so its hash order differs too.
        let mut names: Vec<&String> = a.keys().collect();
        names.sort_unstable_by(|x, y| y.cmp(x));
        let b: HashMap<_, _> = names
            .into_iter()
            .map(|k| (k.clone(), a[k].clone()))
            .collect();
        assert_eq!(chain_sha256(&a), chain_sha256(&b));
        assert_ne!(chain_sha256(&a), chain_sha256(&chain(0.5)));
    }

    #[test]
    fn a_non_finite_number_is_refused_rather_than_lost() {
        let mut r = result();
        r.timesteps[1] = f64::NAN;
        let error = r.to_arrow_bytes().expect_err("NaN");
        assert!(error.contains("timesteps.1"), "{error}");
    }

    #[test]
    fn an_infinite_ratio_round_trips() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct S(#[serde(with = "crate::serde_remote::non_finite_pair")] Option<(f64, f64)>);
        for v in [
            Some((1.0e6, f64::INFINITY)),
            Some((0.1, f64::NEG_INFINITY)),
            Some((2.0, 0.3)),
            None,
        ] {
            let json = serde_json::to_string(&S(v)).unwrap();
            assert_eq!(serde_json::from_str::<S>(&json).unwrap(), S(v), "{json}");
            assert!(crate::serde_remote::non_finite(&S(v)).is_empty());
        }
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct N(#[serde(with = "crate::serde_remote::non_finite_pair")] Option<(f64, f64)>);
        let json = serde_json::to_string(&N(Some((f64::NAN, 1.0)))).unwrap();
        assert!(serde_json::from_str::<N>(&json)
            .unwrap()
            .0
            .unwrap()
            .0
            .is_nan());
    }

    #[test]
    fn tuple_keys_round_trip() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct S(#[serde(with = "crate::serde_remote::pairs")] BTreeMap<(String, i32), f64>);
        let s = S(BTreeMap::from([
            (("Fe56".into(), 102), 0.5),
            (("Fe56".into(), 16), 1.5),
        ]));
        assert_eq!(
            serde_json::from_str::<S>(&serde_json::to_string(&s).unwrap()).unwrap(),
            s
        );
    }

    #[test]
    fn other_files_are_refused() {
        assert!(TransmutationResults::from_arrow_bytes(b"not arrow").is_err());
        // An Arrow file that is not a result.
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        )
        .unwrap();
        let mut bytes = Vec::new();
        let mut w = arrow_ipc::writer::FileWriter::try_new(&mut bytes, &schema).unwrap();
        w.write(&batch).unwrap();
        w.finish().unwrap();
        drop(w);
        let error = TransmutationResults::from_arrow_bytes(&bytes).expect_err("not a result");
        assert!(error.contains("no format version"), "{error}");
    }
}
