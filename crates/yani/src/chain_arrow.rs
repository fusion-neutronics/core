//! Arrow IPC reader/writer for transmutation chain directories.
//!
//! Reads/writes a `.chain.arrow/` directory whose layout matches the
//! `nuclear_data_to_yamc_format` schema (nuclides/decays/reactions/
//! sources/fission_yields + version.json).

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, Int32Builder, ListBuilder, StringBuilder};
use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, ListArray, RecordBatch, StringArray};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};

/// The declared schema for a section, from `nuclear-data-schema`.
///
/// Panics on an unknown path, which can only be a typo: the crate is the whole
/// set of sections this format has.
fn section_schema(path: &str) -> Schema {
    nuclear_data_schema::section(path)
        .unwrap_or_else(|| panic!("no declared schema for section {path}"))
}

use crate::chain::{
    BranchCurve, BranchQuantity, BranchTable, ChainNuclide, ChainParts, ChainReaction, DecaySource,
    DecaySourceDistribution, EvaluatedYields, FissionYield, FissionYieldSet,
};

fn read_arrow_bytes(bytes: &[u8]) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
    let reader = FileReader::try_new(std::io::Cursor::new(bytes), None)?;
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch?);
    }
    Ok(batches)
}

fn read_arrow_file(path: &Path) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
    read_arrow_bytes(&std::fs::read(path)?)
}

/// Read a split-layout section, checking it against its declared schema first.
///
/// The flat layout keeps using [`read_arrow_file`]: its sections are not
/// declared, and its file names collide with both the neutron sections and the
/// split ones, so the section has to be stated rather than inferred.
///
/// `origin` names the bytes in an error message: a path when they were read
/// from one, the section name when a host handed them over directly.
fn read_section_bytes(
    bytes: &[u8],
    section: &str,
    origin: &str,
) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
    let reader = FileReader::try_new(std::io::Cursor::new(bytes), None)?;
    nuclear_data_schema::check_batch(section, reader.schema().as_ref())
        .map_err(|e| format!("{origin}: {e}"))?;
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch?);
    }
    Ok(batches)
}

fn col<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T, Box<dyn Error>> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| format!("column '{name}' not found"))?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| format!("column '{name}' has unexpected array type").into())
}

fn list_f64(list: &ListArray, i: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let values = list.value(i);
    let floats = values
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or("list<f64> inner type mismatch")?;
    Ok((0..floats.len()).map(|j| floats.value(j)).collect())
}

/// A list of floats whose items may be null, as `None`.
fn list_opt_f64(list: &ListArray, i: usize) -> Result<Vec<Option<f64>>, Box<dyn Error>> {
    let values = list.value(i);
    let floats = values
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or("list<f64> inner type mismatch")?;
    Ok((0..floats.len())
        .map(|j| (!floats.is_null(j)).then(|| floats.value(j)))
        .collect())
}

fn list_str(list: &ListArray, i: usize) -> Result<Vec<String>, Box<dyn Error>> {
    let values = list.value(i);
    let strs = values
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("list<str> inner type mismatch")?;
    Ok((0..strs.len()).map(|j| strs.value(j).to_string()).collect())
}

/// Whether a branching row `(kind, target)` should be grafted onto `parent`.
///
/// A graft adds a metastable-production channel the base three-part chain lacks
/// so reduction routes it and the rate-time fold can populate its branching. It
/// is allowed only when `target` is itself a chain nuclide (has decay data) and
/// either the reaction is `(n,n')` (self-inelastic; the base chains omit it
/// entirely and the fold injects its rate) or the base chain already carries
/// this reaction's ground residual (`kind_present`), so the fold re-partitions a
/// nonzero mass and no parent atoms leak into an unproduced target. The `(n,n')`
/// ground self-loop (`target == parent`) is a depletion no-op and is skipped.
fn should_graft(
    kind: &str,
    target: &str,
    parent: &str,
    target_is_nuclide: bool,
    kind_present: bool,
) -> bool {
    target != parent && target_is_nuclide && (kind == "(n,n')" || kind_present)
}

/// The nuclide a decay mode of `parent` leaves in the chain, or `None` where
/// the chain models no product.
///
/// Only the product is dropped. The branching ratio is evaluated data and is
/// kept, so the parent is still removed at its full decay constant and a
/// branch with no product takes its share of the atoms out of the chain.
///
/// A mode ending in spontaneous fission leaves fission fragments. The chain
/// carries no spontaneous-fission product yields (out of scope here), so
/// nothing it makes is modelled. What the file stores as its target is not a
/// daughter either: `sf` moves neither Z nor A in the chain build's mode table,
/// so the target is the parent itself (Cf252 sf -> Cf252, 115 parents in
/// ENDF/B-VIII.1), the ground state of an isomer (Am242_m2 sf -> Am242), the
/// intermediate of a delayed fission (Tl180 ec/beta+,sf -> Hg180), or wherever
/// `replace_missing` walked to from an absent one (Ds279_m1 sf -> Fm259). Read
/// as an edge, the first put the branch back on the parent's own diagonal, so
/// Cf252 was removed at 0.969 of its decay constant; the rest made atoms that
/// fission does not make. OpenMC's matrix skips every mode containing `sf`.
///
/// Any other mode naming its own parent is a `replace_missing` artifact too:
/// Es258's EC daughter Cf258 has no decay data, and the walk from it by beta-
/// lands back on Es258. A decay cannot return a nucleus to itself, so that edge
/// goes the same way, where OpenMC keeps it.
fn modelled_decay_target(parent: &str, kind: &str, target: Option<&str>) -> Option<String> {
    if kind.split(',').any(|mode| mode == "sf") {
        return None;
    }
    target.filter(|t| *t != parent).map(str::to_string)
}

/// Parse a transmutation chain from a `.chain.arrow/` directory.
///
/// The flat layout has no evaluated yields, so every [`FissionYield`] comes
/// back with `independent` and `cumulative` as `None`. The split layout read
/// by [`parse_chain_parts`] carries them.
pub fn parse_chain_arrow<P: AsRef<Path>>(
    dir: P,
) -> Result<HashMap<String, ChainNuclide>, Box<dyn Error>> {
    let dir = dir.as_ref();
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir.display()).into());
    }

    let mut chain: HashMap<String, ChainNuclide> = HashMap::new();
    let mut fy_parent_refs: Vec<(String, String)> = Vec::new();

    // nuclides.arrow -- required
    for batch in read_arrow_file(&dir.join("nuclides.arrow"))? {
        let names = col::<StringArray>(&batch, "name")?;
        let half_lives = col::<Float64Array>(&batch, "half_life")?;
        let decay_energies = batch
            .schema()
            .index_of("decay_energy")
            .ok()
            .and_then(|idx| batch.column(idx).as_any().downcast_ref::<Float64Array>());
        // Optional: a file written before this column existed reads exactly as
        // it did, with every half-life carrying no stated uncertainty.
        let half_life_sigmas = batch
            .schema()
            .index_of("half_life_uncertainty")
            .ok()
            .and_then(|idx| batch.column(idx).as_any().downcast_ref::<Float64Array>());
        let decay_energy_sigmas = batch
            .schema()
            .index_of("decay_energy_uncertainty")
            .ok()
            .and_then(|idx| batch.column(idx).as_any().downcast_ref::<Float64Array>());
        let parents = col::<StringArray>(&batch, "fission_yield_parent")?;
        for i in 0..batch.num_rows() {
            let name = names.value(i).to_string();
            let half_life = if half_lives.is_null(i) {
                None
            } else {
                Some(half_lives.value(i))
            };
            let half_life_uncertainty = half_life_sigmas
                .filter(|column| !column.is_null(i))
                .map(|column| column.value(i));
            let decay_energy_uncertainty = decay_energy_sigmas
                .filter(|column| !column.is_null(i))
                .map(|column| column.value(i));
            let decay_energy = decay_energies
                .and_then(|values| {
                    if values.is_null(i) {
                        None
                    } else {
                        Some(values.value(i))
                    }
                })
                .unwrap_or(0.0);
            if !parents.is_null(i) {
                fy_parent_refs.push((name.clone(), parents.value(i).to_string()));
            }
            chain.insert(
                name.clone(),
                ChainNuclide {
                    name,
                    half_life,
                    half_life_uncertainty,
                    decay_energy,
                    decay_energy_uncertainty,
                    decay_energy_components: Default::default(),
                    reactions: Vec::new(),
                    decays: Vec::new(),
                    fission_yields: None,
                    sources: Vec::new(),
                },
            );
        }
    }

    // decays.arrow -- optional
    let decays_path = dir.join("decays.arrow");
    if decays_path.exists() {
        for batch in read_arrow_file(&decays_path)? {
            let nuclides = col::<StringArray>(&batch, "nuclide")?;
            let types = col::<StringArray>(&batch, "type")?;
            let targets = col::<StringArray>(&batch, "target")?;
            let branching = col::<Float64Array>(&batch, "branching_ratio")?;
            for i in 0..batch.num_rows() {
                let parent = nuclides.value(i);
                if let Some(nuc) = chain.get_mut(parent) {
                    let kind = types.value(i);
                    let target = (!targets.is_null(i)).then(|| targets.value(i));
                    nuc.decays.push(ChainReaction {
                        kind: kind.to_string(),
                        target: modelled_decay_target(parent, kind, target),
                        branching: branching.value(i),
                        q_value: None,
                        branching_uncertainty: None,
                    });
                }
            }
        }
    }

    // reactions.arrow -- optional
    let reactions_path = dir.join("reactions.arrow");
    if reactions_path.exists() {
        for batch in read_arrow_file(&reactions_path)? {
            let nuclides = col::<StringArray>(&batch, "nuclide")?;
            let types = col::<StringArray>(&batch, "type")?;
            let targets = col::<StringArray>(&batch, "target")?;
            let branching = col::<Float64Array>(&batch, "branching_ratio")?;
            // Optional on read: files written before Q was carried have no such
            // column, and the same reader takes both.
            let q_values = col::<Float64Array>(&batch, "Q").ok();
            for i in 0..batch.num_rows() {
                if let Some(nuc) = chain.get_mut(nuclides.value(i)) {
                    nuc.reactions.push(ChainReaction {
                        kind: types.value(i).to_string(),
                        target: if targets.is_null(i) {
                            None
                        } else {
                            Some(targets.value(i).to_string())
                        },
                        branching: branching.value(i),
                        q_value: q_values.as_ref().map(|q| q.value(i)),
                        branching_uncertainty: None,
                    });
                }
            }
        }
    }

    // sources.arrow -- optional. Both "discrete" and "tabular" types map to
    // DecaySourceDistribution::Discrete; the Arrow chain format stores them the
    // same way and does not distinguish between them.
    let sources_path = dir.join("sources.arrow");
    if sources_path.exists() {
        for batch in read_arrow_file(&sources_path)? {
            let nuclides = col::<StringArray>(&batch, "nuclide")?;
            let particles = col::<StringArray>(&batch, "particle")?;
            let energies_col = col::<ListArray>(&batch, "energies")?;
            let intensities_col = col::<ListArray>(&batch, "intensities")?;
            for i in 0..batch.num_rows() {
                if let Some(nuc) = chain.get_mut(nuclides.value(i)) {
                    let energies = list_f64(energies_col, i)?;
                    let intensities = list_f64(intensities_col, i)?;
                    // Skip sources with no data.
                    if energies.is_empty() || energies.len() != intensities.len() {
                        continue;
                    }
                    nuc.sources.push(DecaySource {
                        particle: particles.value(i).to_string(),
                        distribution: DecaySourceDistribution::Discrete {
                            energies,
                            intensities,
                        },
                    });
                }
            }
        }
    }

    // fission_yields.arrow -- optional. Collect per-nuclide yields, then attach.
    let fy_path = dir.join("fission_yields.arrow");
    if fy_path.exists() {
        let mut by_nuclide: HashMap<String, Vec<FissionYield>> = HashMap::new();
        for batch in read_arrow_file(&fy_path)? {
            let nuclides = col::<StringArray>(&batch, "nuclide")?;
            let energies = col::<Float64Array>(&batch, "energy")?;
            let products_col = col::<ListArray>(&batch, "products")?;
            let yields_col = col::<ListArray>(&batch, "yields")?;
            for i in 0..batch.num_rows() {
                let products = list_str(products_col, i)?;
                let yields = list_f64(yields_col, i)?;
                let pairs: Vec<(String, f64)> = products.into_iter().zip(yields).collect();
                by_nuclide
                    .entry(nuclides.value(i).to_string())
                    .or_default()
                    .push(FissionYield {
                        energy: energies.value(i),
                        products: pairs,
                        independent: None,
                        cumulative: None,
                    });
            }
        }
        for (name, yields) in by_nuclide {
            if let Some(nuc) = chain.get_mut(&name) {
                // Sorted on construction; Arrow row order is not a contract.
                nuc.fission_yields = Some(Arc::new(FissionYieldSet::new(yields)));
            }
        }
    }

    // Resolve parent references (e.g. U236 inherits U235 yields).
    for (nuclide_name, parent_name) in fy_parent_refs {
        let parent_yields = chain
            .get(&parent_name)
            .and_then(|p| p.fission_yields.clone());
        if let Some(yields) = parent_yields {
            if let Some(nuc) = chain.get_mut(&nuclide_name) {
                nuc.fission_yields = Some(yields);
            }
        }
    }

    Ok(chain)
}

/// Get-or-create a bare nuclide entry in the chain map.
fn ensure_nuclide<'a>(
    chain: &'a mut HashMap<String, ChainNuclide>,
    name: &str,
) -> &'a mut ChainNuclide {
    chain
        .entry(name.to_string())
        .or_insert_with(|| ChainNuclide {
            name: name.to_string(),
            half_life: None,
            decay_energy: 0.0,
            reactions: Vec::new(),
            decays: Vec::new(),
            half_life_uncertainty: None,
            fission_yields: None,
            sources: Vec::new(),
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
        })
}

/// Parse a transmutation chain from separate v2 subsection directories.
///
/// Each argument is a directory whose top level holds that subsection's arrow
/// files (as produced by the split converter and unpacked from the per-
/// subsection tarballs):
/// - `decay_dir`: `nuclides.arrow` (name, half_life, decay_energy) plus optional
///   `decay_modes.arrow` and `sources.arrow`.
/// - `reactions_dir`: optional `reactions.arrow`.
/// - `fpy_dir`: optional `fission_yields.arrow`, `aliases.arrow` and
///   `evaluated_yields.arrow`.
///
/// The three directories may come from different libraries. Nuclides that
/// appear only in the reactions / fission-yields parts are created with no
/// decay data so the parts union correctly.
/// One chain subsection's section files, keyed by file name
/// (`"nuclides.arrow"`, `"decay_modes.arrow"`, ...).
pub type SectionFiles = HashMap<String, Vec<u8>>;

/// The bytes of a v2 split chain, subsection by subsection.
///
/// What [`parse_chain_parts`] reads off disk, and what a host with no
/// filesystem supplies instead. A browser build is the case that forces the
/// distinction: `std::fs` compiles for `wasm32-unknown-unknown` but every call
/// fails at runtime, so a path-only loader cannot serve it at all.
///
/// Absent files mean an absent section. Only `decay/nuclides.arrow` is
/// required; see [`parse_chain_parts_from_bytes`].
#[derive(Debug, Default, Clone)]
pub struct ChainSections {
    pub decay: SectionFiles,
    pub reactions: SectionFiles,
    pub fission_yields: SectionFiles,
    pub branching: SectionFiles,
}

impl ChainSections {
    /// Which optional subsections these bytes actually carry.
    ///
    /// A bytes-fed host (the browser binding) decides what to upload, so what
    /// is here is what the chain will be built from. Deriving the parts rather
    /// than assuming a complete chain is what lets a page that uploaded only
    /// decay and reactions be told its fission products are being dropped,
    /// instead of solving as though they were never made.
    pub fn parts(&self) -> ChainParts {
        ChainParts {
            reactions: !self.reactions.is_empty(),
            fission_yields: !self.fission_yields.is_empty(),
        }
    }

    /// Add one section file, e.g. `("decay", "nuclides.arrow", bytes)`.
    ///
    /// Errors on an unknown subsection rather than dropping the bytes: a
    /// silently ignored section loads as an empty chain, which produces a
    /// transmutation of nothing rather than a complaint.
    pub fn insert(
        &mut self,
        subsection: &str,
        file: &str,
        bytes: Vec<u8>,
    ) -> Result<(), Box<dyn Error>> {
        let target = match subsection {
            "decay" => &mut self.decay,
            "reactions" => &mut self.reactions,
            "fission_yields" => &mut self.fission_yields,
            "branching" => &mut self.branching,
            other => {
                return Err(format!(
                    "unknown chain subsection '{other}' \
                     (expected decay, reactions, fission_yields or branching)"
                )
                .into())
            }
        };
        target.insert(file.to_string(), bytes);
        Ok(())
    }
}

/// Attach `fission_yields/evaluated_yields.arrow` to the nominal yields read
/// from `fission_yields.arrow`, by nuclide and incident energy.
///
/// Every row has to land on a nominal one. The evaluated yields ride on
/// [`FissionYield`], so a row with nowhere to go is either a file written
/// against different nominal yields or a damaged one, and dropping it would
/// lose the data the file exists to keep.
fn attach_evaluated_yields(
    by_nuclide: &mut HashMap<String, Vec<FissionYield>>,
    bytes: &[u8],
) -> Result<(), Box<dyn Error>> {
    const SECTION: &str = "fission_yields/evaluated_yields.arrow";
    for batch in read_section_bytes(bytes, SECTION, SECTION)? {
        let nuclides = col::<StringArray>(&batch, "nuclide")?;
        let energies = col::<Float64Array>(&batch, "energy")?;
        let kinds = col::<StringArray>(&batch, "kind")?;
        let interpolation = col::<Int32Array>(&batch, "interpolation")?;
        let products_col = col::<ListArray>(&batch, "products")?;
        let yields_col = col::<ListArray>(&batch, "yields")?;
        let sigmas_col = col::<ListArray>(&batch, "yield_uncertainties")?;
        for i in 0..batch.num_rows() {
            let (nuclide, energy, kind) = (nuclides.value(i), energies.value(i), kinds.value(i));
            let products = list_str(products_col, i)?;
            // Read the items as optional so a null yield is refused rather
            // than loaded as a stated 0.0, which list_f64 would do.
            let yields = list_opt_f64(yields_col, i)?
                .into_iter()
                .collect::<Option<Vec<f64>>>()
                .ok_or_else(|| {
                    format!("{SECTION}: {nuclide} {kind} at {energy} eV has a null yield")
                })?;
            let uncertainties = if sigmas_col.is_null(i) {
                vec![None; products.len()]
            } else {
                list_opt_f64(sigmas_col, i)?
            };
            if yields.len() != products.len() || uncertainties.len() != products.len() {
                return Err(format!(
                    "{SECTION}: {nuclide} {kind} at {energy} eV has {} products, {} yields \
                     and {} uncertainties",
                    products.len(),
                    yields.len(),
                    uncertainties.len()
                )
                .into());
            }
            let entry = by_nuclide
                .get_mut(nuclide)
                .and_then(|sets| sets.iter_mut().find(|y| y.energy == energy))
                .ok_or_else(|| {
                    format!(
                        "{SECTION}: {nuclide} at {energy} eV has no row in \
                         fission_yields/fission_yields.arrow to belong to"
                    )
                })?;
            let slot = match kind {
                "independent" => &mut entry.independent,
                "cumulative" => &mut entry.cumulative,
                other => {
                    return Err(format!(
                        "{SECTION}: unknown kind {other:?} for {nuclide} at {energy} eV \
                         (expected independent or cumulative)"
                    )
                    .into())
                }
            };
            if slot.is_some() {
                return Err(
                    format!("{SECTION}: {nuclide} has two {kind} rows at {energy} eV").into(),
                );
            }
            *slot = Some(EvaluatedYields {
                products,
                yields,
                uncertainties,
                interpolation: (!interpolation.is_null(i)).then(|| interpolation.value(i)),
            });
        }
    }
    Ok(())
}

/// Parse a v2 split chain from bytes, with no filesystem involved.
///
/// The filesystem entry point is [`parse_chain_parts`], which reads the same
/// eight files and delegates here.
pub fn parse_chain_parts_from_bytes(
    parts: &ChainSections,
) -> Result<(HashMap<String, ChainNuclide>, BranchTable), Box<dyn Error>> {
    let mut chain: HashMap<String, ChainNuclide> = HashMap::new();

    // decay/nuclides.arrow -- the decay index (name, half_life, decay_energy).
    // Required: every other section here is optional, so without this guard a
    // chain missing it loads as EMPTY and every transmutation silently
    // produces nothing.
    let Some(nuclides_bytes) = parts.decay.get("nuclides.arrow") else {
        return Err("chain has no decay/nuclides.arrow \
                    (expected a v2 split chain: decay, reactions, fission_yields)"
            .into());
    };
    {
        for batch in read_section_bytes(
            nuclides_bytes,
            "decay/nuclides.arrow",
            "decay/nuclides.arrow",
        )? {
            let names = col::<StringArray>(&batch, "name")?;
            let half_lives = col::<Float64Array>(&batch, "half_life")?;
            let decay_energies = col::<Float64Array>(&batch, "decay_energy")?;
            let half_life_sigmas = batch
                .schema()
                .index_of("half_life_uncertainty")
                .ok()
                .and_then(|idx| batch.column(idx).as_any().downcast_ref::<Float64Array>());
            let decay_energy_sigmas = batch
                .schema()
                .index_of("decay_energy_uncertainty")
                .ok()
                .and_then(|idx| batch.column(idx).as_any().downcast_ref::<Float64Array>());
            // The component split, optional: a file written before it reads
            // with every component absent.
            let optional = |name: &str| {
                batch
                    .schema()
                    .index_of(name)
                    .ok()
                    .and_then(|idx| batch.column(idx).as_any().downcast_ref::<Float64Array>())
            };
            let components: Vec<(Option<&Float64Array>, Option<&Float64Array>)> =
                crate::chain::DECAY_ENERGY_COMPONENTS
                    .iter()
                    .map(|c| {
                        (
                            optional(&format!("decay_energy_{c}")),
                            optional(&format!("decay_energy_{c}_uncertainty")),
                        )
                    })
                    .collect();
            for i in 0..batch.num_rows() {
                let name = names.value(i).to_string();
                let nuc = ensure_nuclide(&mut chain, &name);
                for (c, (energy, sigma)) in components.iter().enumerate() {
                    nuc.decay_energy_components[c] = energy
                        .filter(|column| !column.is_null(i))
                        .map(|column| crate::chain::DecayEnergyComponent {
                            energy: column.value(i),
                            uncertainty: sigma
                                .filter(|column| !column.is_null(i))
                                .map(|column| column.value(i)),
                        });
                }
                nuc.half_life_uncertainty = half_life_sigmas
                    .filter(|column| !column.is_null(i))
                    .map(|column| column.value(i));
                nuc.decay_energy_uncertainty = decay_energy_sigmas
                    .filter(|column| !column.is_null(i))
                    .map(|column| column.value(i));
                nuc.half_life = if half_lives.is_null(i) {
                    None
                } else {
                    Some(half_lives.value(i))
                };
                nuc.decay_energy = if decay_energies.is_null(i) {
                    0.0
                } else {
                    decay_energies.value(i)
                };
            }
        }
    }

    // decay/decay_modes.arrow -- optional.
    if let Some(bytes) = parts.decay.get("decay_modes.arrow") {
        for batch in
            read_section_bytes(bytes, "decay/decay_modes.arrow", "decay/decay_modes.arrow")?
        {
            let nuclides = col::<StringArray>(&batch, "nuclide")?;
            let types = col::<StringArray>(&batch, "type")?;
            let targets = col::<StringArray>(&batch, "target")?;
            let branching = col::<Float64Array>(&batch, "branching_ratio")?;
            // Optional: a file written before the column reads with no mode
            // carrying a stated sigma. Taken as stored, 0.0 included.
            let branching_sigmas = batch
                .schema()
                .index_of("branching_ratio_uncertainty")
                .ok()
                .and_then(|idx| batch.column(idx).as_any().downcast_ref::<Float64Array>());
            for i in 0..batch.num_rows() {
                let parent = nuclides.value(i);
                let kind = types.value(i);
                let target = (!targets.is_null(i)).then(|| targets.value(i));
                let nuc = ensure_nuclide(&mut chain, parent);
                nuc.decays.push(ChainReaction {
                    kind: kind.to_string(),
                    target: modelled_decay_target(parent, kind, target),
                    branching: branching.value(i),
                    q_value: None,
                    branching_uncertainty: branching_sigmas
                        .filter(|column| !column.is_null(i))
                        .map(|column| column.value(i)),
                });
            }
        }
    }

    // decay/sources.arrow -- optional.
    if let Some(bytes) = parts.decay.get("sources.arrow") {
        for batch in read_section_bytes(bytes, "decay/sources.arrow", "decay/sources.arrow")? {
            let nuclides = col::<StringArray>(&batch, "nuclide")?;
            let particles = col::<StringArray>(&batch, "particle")?;
            let energies_col = col::<ListArray>(&batch, "energies")?;
            let intensities_col = col::<ListArray>(&batch, "intensities")?;
            for i in 0..batch.num_rows() {
                let energies = list_f64(energies_col, i)?;
                let intensities = list_f64(intensities_col, i)?;
                if energies.is_empty() || energies.len() != intensities.len() {
                    continue;
                }
                let nuc = ensure_nuclide(&mut chain, nuclides.value(i));
                nuc.sources.push(DecaySource {
                    particle: particles.value(i).to_string(),
                    distribution: DecaySourceDistribution::Discrete {
                        energies,
                        intensities,
                    },
                });
            }
        }
    }

    // reactions/reactions.arrow -- optional.
    if let Some(bytes) = parts.reactions.get("reactions.arrow") {
        for batch in read_section_bytes(
            bytes,
            "reactions/reactions.arrow",
            "reactions/reactions.arrow",
        )? {
            let nuclides = col::<StringArray>(&batch, "nuclide")?;
            let types = col::<StringArray>(&batch, "type")?;
            let targets = col::<StringArray>(&batch, "target")?;
            let branching = col::<Float64Array>(&batch, "branching_ratio")?;
            let q_values = col::<Float64Array>(&batch, "Q").ok();
            for i in 0..batch.num_rows() {
                let nuc = ensure_nuclide(&mut chain, nuclides.value(i));
                nuc.reactions.push(ChainReaction {
                    kind: types.value(i).to_string(),
                    target: if targets.is_null(i) {
                        None
                    } else {
                        Some(targets.value(i).to_string())
                    },
                    branching: branching.value(i),
                    q_value: q_values.as_ref().map(|q| q.value(i)),
                    branching_uncertainty: None,
                });
            }
        }
    }

    // fission_yields/fission_yields.arrow -- optional.
    if let Some(bytes) = parts.fission_yields.get("fission_yields.arrow") {
        let mut by_nuclide: HashMap<String, Vec<FissionYield>> = HashMap::new();
        for batch in read_section_bytes(
            bytes,
            "fission_yields/fission_yields.arrow",
            "fission_yields/fission_yields.arrow",
        )? {
            let nuclides = col::<StringArray>(&batch, "nuclide")?;
            let energies = col::<Float64Array>(&batch, "energy")?;
            let products_col = col::<ListArray>(&batch, "products")?;
            let yields_col = col::<ListArray>(&batch, "yields")?;
            for i in 0..batch.num_rows() {
                let products = list_str(products_col, i)?;
                let yields = list_f64(yields_col, i)?;
                let pairs: Vec<(String, f64)> = products.into_iter().zip(yields).collect();
                by_nuclide
                    .entry(nuclides.value(i).to_string())
                    .or_default()
                    .push(FissionYield {
                        energy: energies.value(i),
                        products: pairs,
                        independent: None,
                        cumulative: None,
                    });
            }
        }
        if let Some(bytes) = parts.fission_yields.get("evaluated_yields.arrow") {
            attach_evaluated_yields(&mut by_nuclide, bytes)?;
        }
        for (name, yields) in by_nuclide {
            // `FissionYieldSet::new` sorts: the rows accumulate in Arrow row
            // order, which nothing in the format guarantees is ascending.
            ensure_nuclide(&mut chain, &name).fission_yields =
                Some(Arc::new(FissionYieldSet::new(yields)));
        }
    }

    if parts.fission_yields.contains_key("evaluated_yields.arrow")
        && !parts.fission_yields.contains_key("fission_yields.arrow")
    {
        return Err(
            "fission_yields/evaluated_yields.arrow was supplied without \
                    fission_yields/fission_yields.arrow, whose rows it belongs to"
                .into(),
        );
    }

    // fission_yields/aliases.arrow -- inheritors copy a parent's yields.
    if let Some(bytes) = parts.fission_yields.get("aliases.arrow") {
        let mut refs: Vec<(String, String)> = Vec::new();
        for batch in read_section_bytes(
            bytes,
            "fission_yields/aliases.arrow",
            "fission_yields/aliases.arrow",
        )? {
            let nuclides = col::<StringArray>(&batch, "nuclide")?;
            let parents = col::<StringArray>(&batch, "fission_yield_parent")?;
            for i in 0..batch.num_rows() {
                refs.push((nuclides.value(i).to_string(), parents.value(i).to_string()));
            }
        }
        for (name, parent) in refs {
            let parent_yields = chain.get(&parent).and_then(|p| p.fission_yields.clone());
            if let Some(yields) = parent_yields {
                ensure_nuclide(&mut chain, &name).fission_yields = Some(yields);
            }
        }
    }

    // branching/branching.arrow -- optional isomeric-branching overlay.
    //
    // Each row is a verbatim energy-dependent curve for one
    // (parent, reaction, final-state) triple. We only keep curves whose parent
    // is present in the (possibly reduced) chain.
    //
    // We also graft any production channel the base chain lacks, so the matrix
    // and reduction BFS can route the metastable and the rate-time fold can set
    // its flux-weighted branching. The base three-part chains carry the ground
    // residual for nuclide-changing reactions (e.g. `(n,2n) -> Ag106`) but
    // usually omit the metastable partials (`Ag106_m1`), and omit `(n,n')`
    // entirely. A graft is added when the target is itself a chain nuclide (has
    // decay data) and either:
    //   * the row is a self-inelastic `(n,n')` metastable (target != parent;
    //     the fold injects its rate directly), or
    //   * the base chain already carries this reaction's ground residual, so
    //     the fold re-partitions a nonzero mass and no parent atoms leak into an
    //     unproduced target.
    // The grafted branching is a 0.0 placeholder overwritten by the rate-time
    // fold; reduction follows targets regardless of branching value, and
    // `populated_nuclides` bounds the edge from the curves rather than the
    // placeholder. The ground/self `(n,n')` row (target == parent) is a
    // depletion no-op and is skipped.
    let mut branch_table: BranchTable = BranchTable::new();
    {
        if let Some(bytes) = parts.branching.get("branching.arrow") {
            for batch in read_arrow_bytes(bytes)? {
                let nuclides = col::<StringArray>(&batch, "nuclide")?;
                let reactions = col::<StringArray>(&batch, "reaction")?;
                let targets = col::<StringArray>(&batch, "target")?;
                let quantities = col::<StringArray>(&batch, "quantity")?;
                let energy_col = col::<ListArray>(&batch, "energy")?;
                let values_col = col::<ListArray>(&batch, "values")?;
                for i in 0..batch.num_rows() {
                    let parent = nuclides.value(i);
                    if !chain.contains_key(parent) {
                        continue;
                    }
                    let kind = reactions.value(i).to_string();
                    let target = targets.value(i).to_string();
                    let quantity = match quantities.value(i) {
                        "yield" => BranchQuantity::Yield,
                        "cross_section" => BranchQuantity::CrossSection,
                        _ => continue,
                    };
                    let energy = list_f64(energy_col, i)?;
                    let values = list_f64(values_col, i)?;
                    if energy.is_empty() || energy.len() != values.len() {
                        continue;
                    }

                    // Graft the metastable-production channel when the base
                    // chain lacks it (see the block comment above and
                    // `should_graft` for the guard rationale).
                    let target_is_nuclide = chain.contains_key(&target);
                    let nuc = chain.get_mut(parent).expect("parent present");
                    let kind_present = nuc.reactions.iter().any(|r| r.kind == kind);
                    if should_graft(&kind, &target, parent, target_is_nuclide, kind_present) {
                        let exists = nuc
                            .reactions
                            .iter()
                            .any(|r| r.kind == kind && r.target.as_deref() == Some(&target));
                        if !exists {
                            nuc.reactions.push(ChainReaction {
                                kind: kind.clone(),
                                target: Some(target.clone()),
                                branching: 0.0,
                                q_value: None,
                                branching_uncertainty: None,
                            });
                        }
                    }

                    branch_table
                        .entry(parent.to_string())
                        .or_default()
                        .entry(kind)
                        .or_default()
                        .push(BranchCurve {
                            target,
                            quantity,
                            energy,
                            values,
                        });
                }
            }
        }
    }

    Ok((chain, branch_table))
}

/// Read one optional section file. A missing file is an absent section, which
/// every caller but `decay/nuclides.arrow` treats as "nothing to add".
fn load_optional(dir: &Path, file: &str, into: &mut SectionFiles) -> Result<(), Box<dyn Error>> {
    let path = dir.join(file);
    if path.exists() {
        into.insert(file.to_string(), std::fs::read(&path)?);
    }
    Ok(())
}

/// Parse a v2 split chain from directories on disk.
///
/// Reads the eight section files into [`ChainSections`] and hands them to
/// [`parse_chain_parts_from_bytes`], which holds the actual parsing. The split
/// exists so a host without a filesystem can reach the same parser.
pub fn parse_chain_parts(
    decay_dir: &Path,
    reactions_dir: Option<&Path>,
    fpy_dir: Option<&Path>,
    branch_dir: Option<&Path>,
) -> Result<(HashMap<String, ChainNuclide>, BranchTable), Box<dyn Error>> {
    let mut parts = ChainSections::default();

    // Required, and diagnosed here rather than in the parser: only a filesystem
    // caller can see the retired flat layout (nuclides.arrow / decays.arrow at
    // the root, no subsection dirs), which otherwise loads as an EMPTY chain
    // and makes every transmutation silently produce nothing.
    let nuclides_path = decay_dir.join("nuclides.arrow");
    if !nuclides_path.exists() {
        let flat_layout = decay_dir
            .parent()
            .is_some_and(|root| root.join("nuclides.arrow").exists());
        return Err(format!(
            "chain subsection '{}' has no nuclides.arrow{}",
            decay_dir.display(),
            if flat_layout {
                ": the directory above is in the retired flat chain layout \
                 (nuclides.arrow / decays.arrow at its root). Re-export it with \
                 `TransmutationChain.export_to_arrow`, or delete it and let yamc \
                 re-download the split layout"
            } else {
                " (expected a v2 split chain: decay/, reactions/, fission_yields/)"
            }
        )
        .into());
    }
    parts
        .decay
        .insert("nuclides.arrow".to_string(), std::fs::read(&nuclides_path)?);

    load_optional(decay_dir, "decay_modes.arrow", &mut parts.decay)?;
    load_optional(decay_dir, "sources.arrow", &mut parts.decay)?;
    if let Some(reactions_dir) = reactions_dir {
        load_optional(reactions_dir, "reactions.arrow", &mut parts.reactions)?;
    }
    if let Some(fpy_dir) = fpy_dir {
        load_optional(fpy_dir, "fission_yields.arrow", &mut parts.fission_yields)?;
        load_optional(fpy_dir, "aliases.arrow", &mut parts.fission_yields)?;
        load_optional(fpy_dir, "evaluated_yields.arrow", &mut parts.fission_yields)?;
    }
    if let Some(branch_dir) = branch_dir {
        load_optional(branch_dir, "branching.arrow", &mut parts.branching)?;
    }

    parse_chain_parts_from_bytes(&parts)
}

/// Write a transmutation chain to the v2 split-subsection layout under `dir`:
/// `decay/` (nuclides + decay_modes + sources), `reactions/`, and
/// `fission_yields/`, plus a `manifest.json`. Reading it back with
/// [`parse_chain_parts`] gives the same chain.
///
/// It is not a copy of the file the chain was read from, though. The reader
/// drops the target of every decay mode the chain models no product for (any
/// mode involving spontaneous fission, and one whose stored target is its own
/// parent: see `modelled_decay_target`), so those rows are written with a null
/// target where the source file names the parent, a ground state or a
/// `replace_missing` stand-in. Their branching ratios, and the sigmas on
/// them, are written unchanged.
pub fn export_chain_parts<P: AsRef<Path>>(
    chain: &HashMap<String, ChainNuclide>,
    dir: P,
    library: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let dir = dir.as_ref();
    let decay_dir = dir.join("decay");
    let reactions_dir = dir.join("reactions");
    let fy_dir = dir.join("fission_yields");
    std::fs::create_dir_all(&decay_dir)?;
    std::fs::create_dir_all(&reactions_dir)?;
    std::fs::create_dir_all(&fy_dir)?;

    let mut names: Vec<&String> = chain.keys().collect();
    names.sort();

    // decay/nuclides.arrow
    {
        let mut name_b = StringBuilder::new();
        let mut hl_b = Float64Builder::new();
        let mut de_b = Float64Builder::new();
        let mut hl_sigma_b = Float64Builder::new();
        let mut de_sigma_b = Float64Builder::new();
        let mut component_b: [(Float64Builder, Float64Builder); 3] = Default::default();
        for name in &names {
            let nuc = &chain[*name];
            name_b.append_value(&nuc.name);
            for (part, (energy_b, sigma_b)) in nuc
                .decay_energy_components
                .iter()
                .zip(component_b.iter_mut())
            {
                energy_b.append_option(part.map(|p| p.energy));
                sigma_b.append_option(part.and_then(|p| p.uncertainty));
            }
            match nuc.half_life {
                Some(h) => hl_b.append_value(h),
                None => hl_b.append_null(),
            }
            de_b.append_value(nuc.decay_energy);
            // Written back as read: null stays null and a stored 0.0 stays
            // 0.0, so a round trip changes nothing a reader could tell apart.
            match nuc.half_life_uncertainty {
                Some(sigma) => hl_sigma_b.append_value(sigma),
                None => hl_sigma_b.append_null(),
            }
            match nuc.decay_energy_uncertainty {
                Some(sigma) => de_sigma_b.append_value(sigma),
                None => de_sigma_b.append_null(),
            }
        }
        let schema = Arc::new(section_schema("decay/nuclides.arrow"));
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(name_b.finish()),
            Arc::new(hl_b.finish()),
            Arc::new(de_b.finish()),
            Arc::new(hl_sigma_b.finish()),
            Arc::new(de_sigma_b.finish()),
        ];
        for (energy_b, sigma_b) in component_b.iter_mut() {
            columns.push(Arc::new(energy_b.finish()));
            columns.push(Arc::new(sigma_b.finish()));
        }
        let batch = RecordBatch::try_new(schema.clone(), columns)?;
        write_arrow_file(&decay_dir.join("nuclides.arrow"), schema, batch)?;
    }

    // decay/decay_modes.arrow and reactions/reactions.arrow share a shape:
    // (nuclide, type, target, branching_ratio), each with one column of its
    // own.
    let write_reactions = |path: &Path,
                           section: &str,
                           pick: &dyn Fn(&ChainNuclide) -> &Vec<ChainReaction>|
     -> Result<(), Box<dyn Error>> {
        let mut nuc_b = StringBuilder::new();
        let mut type_b = StringBuilder::new();
        let mut target_b = StringBuilder::new();
        let mut q_b = Float64Builder::new();
        let mut br_b = Float64Builder::new();
        let mut br_sigma_b = Float64Builder::new();
        for name in &names {
            let nuc = &chain[*name];
            for r in pick(nuc) {
                nuc_b.append_value(&nuc.name);
                type_b.append_value(&r.kind);
                match &r.target {
                    Some(t) => target_b.append_value(t),
                    None => target_b.append_null(),
                }
                // Q is declared non-nullable, and a decay mode has none, so
                // the column is only built for the section that declares it.
                q_b.append_value(r.q_value.unwrap_or(0.0));
                br_b.append_value(r.branching);
                br_sigma_b.append_option(r.branching_uncertainty);
            }
        }
        // The two files do NOT share a schema, though they share a shape.
        // `reactions/reactions.arrow` declares a fifth column, Q, and its own
        // metadata. Writing both with the decay_modes schema produced a
        // reactions file with no Q at all, and nothing caught it: check_batch
        // is deliberately lenient toward a declared column the batch omits, and
        // the schema test covered only the decay sections.
        let schema = Arc::new(section_schema(section));
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(nuc_b.finish()),
            Arc::new(type_b.finish()),
            Arc::new(target_b.finish()),
        ];
        if section == "reactions/reactions.arrow" {
            columns.push(Arc::new(q_b.finish()));
        }
        columns.push(Arc::new(br_b.finish()));
        // The mirror image: only decay_modes declares a branching sigma, and
        // check_batch refuses a reactions file that carries one.
        if section == "decay/decay_modes.arrow" {
            columns.push(Arc::new(br_sigma_b.finish()));
        }
        let batch = RecordBatch::try_new(schema.clone(), columns)?;
        write_arrow_file(path, schema, batch)
    };
    write_reactions(
        &decay_dir.join("decay_modes.arrow"),
        "decay/decay_modes.arrow",
        &|n| &n.decays,
    )?;
    write_reactions(
        &reactions_dir.join("reactions.arrow"),
        "reactions/reactions.arrow",
        &|n| &n.reactions,
    )?;

    // decay/sources.arrow
    {
        let mut nuc_b = StringBuilder::new();
        let mut particle_b = StringBuilder::new();
        let mut energies_b = ListBuilder::new(Float64Builder::new());
        let mut intensities_b = ListBuilder::new(Float64Builder::new());
        // DecaySourceDistribution has one variant, so the type is fixed. It is
        // still written, because the section declares the column and a reader
        // is entitled to expect it.
        let mut type_b = StringBuilder::new();
        for name in &names {
            let nuc = &chain[*name];
            for s in &nuc.sources {
                nuc_b.append_value(&nuc.name);
                particle_b.append_value(&s.particle);
                type_b.append_value("discrete");
                let DecaySourceDistribution::Discrete {
                    energies,
                    intensities,
                } = &s.distribution;
                for e in energies {
                    energies_b.values().append_value(*e);
                }
                energies_b.append(true);
                for i in intensities {
                    intensities_b.values().append_value(*i);
                }
                intensities_b.append(true);
            }
        }
        let schema = Arc::new(section_schema("decay/sources.arrow"));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(nuc_b.finish()),
                Arc::new(particle_b.finish()),
                Arc::new(type_b.finish()),
                Arc::new(energies_b.finish()),
                Arc::new(intensities_b.finish()),
            ],
        )?;
        write_arrow_file(&decay_dir.join("sources.arrow"), schema, batch)?;
    }

    // fission_yields/fission_yields.arrow
    {
        let mut nuc_b = StringBuilder::new();
        let mut energy_b = Float64Builder::new();
        let mut products_b = ListBuilder::new(StringBuilder::new());
        let mut yields_b = ListBuilder::new(Float64Builder::new());
        for name in &names {
            let nuc = &chain[*name];
            if let Some(fy) = &nuc.fission_yields {
                for entry in &fy.yields {
                    nuc_b.append_value(&nuc.name);
                    energy_b.append_value(entry.energy);
                    for (p, y) in &entry.products {
                        products_b.values().append_value(p);
                        yields_b.values().append_value(*y);
                    }
                    products_b.append(true);
                    yields_b.append(true);
                }
            }
        }
        let schema = Arc::new(section_schema("fission_yields/fission_yields.arrow"));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(nuc_b.finish()),
                Arc::new(energy_b.finish()),
                Arc::new(products_b.finish()),
                Arc::new(yields_b.finish()),
            ],
        )?;
        write_arrow_file(&fy_dir.join("fission_yields.arrow"), schema, batch)?;
    }

    // fission_yields/evaluated_yields.arrow, written only when the chain
    // carries evaluated yields, as the converter does. Without it a round trip
    // would drop them quietly, since a missing optional file reads as "none".
    {
        let mut nuc_b = StringBuilder::new();
        let mut energy_b = Float64Builder::new();
        let mut kind_b = StringBuilder::new();
        let mut interpolation_b = Int32Builder::new();
        let mut products_b = ListBuilder::new(StringBuilder::new());
        let mut yields_b = ListBuilder::new(Float64Builder::new());
        let mut sigmas_b = ListBuilder::new(Float64Builder::new());
        let mut rows = 0usize;
        for name in &names {
            let nuc = &chain[*name];
            let Some(fy) = &nuc.fission_yields else {
                continue;
            };
            for entry in &fy.yields {
                for (kind, evaluated) in [
                    ("independent", &entry.independent),
                    ("cumulative", &entry.cumulative),
                ] {
                    let Some(evaluated) = evaluated else {
                        continue;
                    };
                    nuc_b.append_value(&nuc.name);
                    energy_b.append_value(entry.energy);
                    kind_b.append_value(kind);
                    interpolation_b.append_option(evaluated.interpolation);
                    for p in &evaluated.products {
                        products_b.values().append_value(p);
                    }
                    products_b.append(true);
                    yields_b.values().append_slice(&evaluated.yields);
                    yields_b.append(true);
                    for sigma in &evaluated.uncertainties {
                        sigmas_b.values().append_option(*sigma);
                    }
                    sigmas_b.append(true);
                    rows += 1;
                }
            }
        }
        if rows > 0 {
            let schema = Arc::new(section_schema("fission_yields/evaluated_yields.arrow"));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(nuc_b.finish()),
                    Arc::new(energy_b.finish()),
                    Arc::new(kind_b.finish()),
                    Arc::new(interpolation_b.finish()),
                    Arc::new(products_b.finish()),
                    Arc::new(yields_b.finish()),
                    Arc::new(sigmas_b.finish()),
                ],
            )?;
            write_arrow_file(&fy_dir.join("evaluated_yields.arrow"), schema, batch)?;
        } else {
            remove_stale(&fy_dir.join("evaluated_yields.arrow"))?;
        }
    }
    // The yields above are written per nuclide, inheritors included, so no
    // aliases.arrow belongs beside them. One left by a converter run into the
    // same directory would be read over them.
    remove_stale(&fy_dir.join("aliases.arrow"))?;

    std::fs::write(
        dir.join("manifest.json"),
        format!(
            r#"{{"format_version": 2, "library": "{}", "converter_version": "yani {}", "subsections": {{"decay": {{"path": "decay"}}, "reactions": {{"path": "reactions"}}, "fission_yields": {{"path": "fission_yields"}}}}}}"#,
            library.unwrap_or("unknown"),
            env!("CARGO_PKG_VERSION"),
        ),
    )?;

    Ok(())
}

/// Remove an optional file an export has nothing to write into, if an earlier
/// run left one. A reader loads whatever is present, so a stale file would be
/// attached to a chain it was not written from.
fn remove_stale(path: &Path) -> Result<(), Box<dyn Error>> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            Err(format!("removing stale {}: {e}", path.display()).into())
        }
        _ => Ok(()),
    }
}

fn write_arrow_file(
    path: &Path,
    schema: Arc<Schema>,
    batch: RecordBatch,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = FileWriter::try_new(file, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(())
}

/// Write a transmutation chain to a `.chain.arrow/` directory.
///
/// Produces the 5 arrow IPC files (nuclides, decays, reactions, sources,
/// fission_yields) plus a `version.json`, matching the layout consumed by
/// [`parse_chain_arrow`]. Nuclides are emitted in sorted-name order so output
/// is deterministic.
///
/// `library` is the source data library identifier (e.g. `"endf-b8.1"`) written
/// into `version.json`; pass `None` if unknown (recorded as `"unknown"`).
///
/// The flat layout has no place for evaluated yields: the `independent` and
/// `cumulative` fields of each [`FissionYield`] are not written, and a round
/// trip through this layout drops them. [`export_chain_parts`] is the lossless
/// writer.
pub fn export_chain_arrow<P: AsRef<Path>>(
    chain: &HashMap<String, ChainNuclide>,
    dir: P,
    library: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;

    let mut names: Vec<&String> = chain.keys().collect();
    names.sort();

    // nuclides.arrow
    {
        let mut name_b = StringBuilder::new();
        let mut hl_b = Float64Builder::new();
        let mut decay_energy_b = Float64Builder::new();
        let mut parent_b = StringBuilder::new();
        for name in &names {
            let nuc = &chain[*name];
            name_b.append_value(&nuc.name);
            match nuc.half_life {
                Some(h) => hl_b.append_value(h),
                None => hl_b.append_null(),
            }
            decay_energy_b.append_value(nuc.decay_energy);
            // We don't track parent refs on the in-memory chain; always null.
            parent_b.append_null();
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("half_life", DataType::Float64, true),
            Field::new("decay_energy", DataType::Float64, false),
            Field::new("fission_yield_parent", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(name_b.finish()),
                Arc::new(hl_b.finish()),
                Arc::new(decay_energy_b.finish()),
                Arc::new(parent_b.finish()),
            ],
        )?;
        write_arrow_file(&dir.join("nuclides.arrow"), schema, batch)?;
    }

    // decays.arrow
    {
        let mut nuc_b = StringBuilder::new();
        let mut type_b = StringBuilder::new();
        let mut target_b = StringBuilder::new();
        let mut br_b = Float64Builder::new();
        for name in &names {
            let nuc = &chain[*name];
            for d in &nuc.decays {
                nuc_b.append_value(&nuc.name);
                type_b.append_value(&d.kind);
                match &d.target {
                    Some(t) => target_b.append_value(t),
                    None => target_b.append_null(),
                }
                br_b.append_value(d.branching);
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("nuclide", DataType::Utf8, false),
            Field::new("type", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, true),
            Field::new("branching_ratio", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(nuc_b.finish()),
                Arc::new(type_b.finish()),
                Arc::new(target_b.finish()),
                Arc::new(br_b.finish()),
            ],
        )?;
        write_arrow_file(&dir.join("decays.arrow"), schema, batch)?;
    }

    // reactions.arrow
    {
        let mut nuc_b = StringBuilder::new();
        let mut type_b = StringBuilder::new();
        let mut target_b = StringBuilder::new();
        let mut br_b = Float64Builder::new();
        for name in &names {
            let nuc = &chain[*name];
            for r in &nuc.reactions {
                nuc_b.append_value(&nuc.name);
                type_b.append_value(&r.kind);
                match &r.target {
                    Some(t) => target_b.append_value(t),
                    None => target_b.append_null(),
                }
                br_b.append_value(r.branching);
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("nuclide", DataType::Utf8, false),
            Field::new("type", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, true),
            Field::new("branching_ratio", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(nuc_b.finish()),
                Arc::new(type_b.finish()),
                Arc::new(target_b.finish()),
                Arc::new(br_b.finish()),
            ],
        )?;
        write_arrow_file(&dir.join("reactions.arrow"), schema, batch)?;
    }

    // sources.arrow
    {
        let mut nuc_b = StringBuilder::new();
        let mut particle_b = StringBuilder::new();
        let mut energies_b = ListBuilder::new(Float64Builder::new());
        let mut intensities_b = ListBuilder::new(Float64Builder::new());
        for name in &names {
            let nuc = &chain[*name];
            for s in &nuc.sources {
                nuc_b.append_value(&nuc.name);
                particle_b.append_value(&s.particle);
                match &s.distribution {
                    DecaySourceDistribution::Discrete {
                        energies,
                        intensities,
                    } => {
                        for e in energies {
                            energies_b.values().append_value(*e);
                        }
                        energies_b.append(true);
                        for i in intensities {
                            intensities_b.values().append_value(*i);
                        }
                        intensities_b.append(true);
                    }
                }
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("nuclide", DataType::Utf8, false),
            Field::new("particle", DataType::Utf8, false),
            Field::new(
                "energies",
                DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                false,
            ),
            Field::new(
                "intensities",
                DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(nuc_b.finish()),
                Arc::new(particle_b.finish()),
                Arc::new(energies_b.finish()),
                Arc::new(intensities_b.finish()),
            ],
        )?;
        write_arrow_file(&dir.join("sources.arrow"), schema, batch)?;
    }

    // fission_yields.arrow
    {
        let mut nuc_b = StringBuilder::new();
        let mut energy_b = Float64Builder::new();
        let mut products_b = ListBuilder::new(StringBuilder::new());
        let mut yields_b = ListBuilder::new(Float64Builder::new());
        for name in &names {
            let nuc = &chain[*name];
            if let Some(fy) = &nuc.fission_yields {
                for entry in &fy.yields {
                    nuc_b.append_value(&nuc.name);
                    energy_b.append_value(entry.energy);
                    for (p, y) in &entry.products {
                        products_b.values().append_value(p);
                        yields_b.values().append_value(*y);
                    }
                    products_b.append(true);
                    yields_b.append(true);
                }
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("nuclide", DataType::Utf8, false),
            Field::new("energy", DataType::Float64, false),
            Field::new(
                "products",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new(
                "yields",
                DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(nuc_b.finish()),
                Arc::new(energy_b.finish()),
                Arc::new(products_b.finish()),
                Arc::new(yields_b.finish()),
            ],
        )?;
        write_arrow_file(&dir.join("fission_yields.arrow"), schema, batch)?;
    }

    // Library names are simple keywords (e.g. "endf-b8.1") with no JSON-special
    // characters, so a direct format is safe and avoids a serde_json dependency.
    std::fs::write(
        dir.join("version.json"),
        format!(
            r#"{{"format_version": 1, "library": "{}", "converter_version": "yani {}"}}"#,
            library.unwrap_or("unknown"),
            env!("CARGO_PKG_VERSION"),
        ),
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_chain_parts, should_graft};

    /// Everything export_chain_parts writes must match the declared schema, and
    /// read back through parse_chain_parts.
    ///
    /// This writer and the Python converter both produce these files, so a
    /// column present in one and not the other is a format split under a single
    /// filename. decay/sources.arrow was exactly that: written here without the
    /// `type` column the section declares.
    #[test]
    fn a_half_life_uncertainty_survives_a_round_trip_and_absence_is_not_zero() {
        // The two claims this column exists to keep apart: a stated sigma comes
        // back as itself, and an unstated one comes back as None rather than
        // 0.0, which would read downstream as "measured to be exact".
        let dir = std::env::temp_dir().join(format!("yani-hl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut chain: std::collections::HashMap<String, crate::ChainNuclide> =
            std::collections::HashMap::new();
        chain.insert(
            "Co60".to_string(),
            crate::ChainNuclide {
                name: "Co60".to_string(),
                half_life: Some(1.66344e8),
                half_life_uncertainty: Some(1.21e4),
                decay_energy: 2.5e6,
                decay_energy_uncertainty: Some(3.4e3),
                decay_energy_components: Default::default(),
                reactions: Vec::new(),
                decays: Vec::new(),
                fission_yields: None,
                sources: Vec::new(),
            },
        );
        chain.insert(
            "Xx999".to_string(),
            crate::ChainNuclide {
                name: "Xx999".to_string(),
                half_life: Some(1.0),
                half_life_uncertainty: None,
                decay_energy: 0.0,
                reactions: Vec::new(),
                decays: Vec::new(),
                fission_yields: None,
                sources: Vec::new(),
                decay_energy_uncertainty: None,
                decay_energy_components: Default::default(),
            },
        );

        super::export_chain_parts(&chain, &dir, Some("test")).expect("export succeeds");
        let (back, _branch) =
            super::parse_chain_parts(&dir.join("decay"), None, None, None).expect("load succeeds");
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(back["Co60"].half_life_uncertainty, Some(1.21e4));
        assert_eq!(back["Co60"].decay_energy_uncertainty, Some(3.4e3));
        assert_eq!(
            back["Xx999"].half_life_uncertainty, None,
            "an unstated uncertainty must not come back as zero"
        );
        assert_eq!(
            back["Xx999"].decay_energy_uncertainty, None,
            "an unstated decay-energy uncertainty must not come back as zero"
        );
    }

    /// A chain of three decay modes: a sigma, a stored 0.0 and none at all.
    fn chain_with_branching_sigmas() -> std::collections::HashMap<String, crate::ChainNuclide> {
        use crate::chain::{ChainNuclide, ChainReaction, DecayEnergyComponent};
        let mode = |kind: &str, target: &str, branching: f64, sigma: Option<f64>| ChainReaction {
            kind: kind.to_string(),
            target: Some(target.to_string()),
            branching,
            q_value: None,
            branching_uncertainty: sigma,
        };
        let nuclide = |name: &str, decays: Vec<ChainReaction>| ChainNuclide {
            name: name.to_string(),
            half_life: Some(3.6e3),
            // What MT=457 writes when it states no sigma, stored as such.
            half_life_uncertainty: Some(0.0),
            decay_energy: 2.0e6,
            decay_energy_uncertainty: Some(0.0),
            decay_energy_components: [
                Some(DecayEnergyComponent {
                    energy: 2.0e6,
                    uncertainty: Some(0.0),
                }),
                None,
                None,
            ],
            reactions: Vec::new(),
            decays,
            fission_yields: None,
            sources: Vec::new(),
        };
        std::collections::HashMap::from([
            (
                "Bi212".to_string(),
                nuclide(
                    "Bi212",
                    vec![
                        mode("alpha", "Tl208", 0.3594, Some(6.0e-4)),
                        mode("beta-", "Po212", 0.6406, Some(6.0e-4)),
                    ],
                ),
            ),
            (
                "In116_m1".to_string(),
                nuclide("In116_m1", vec![mode("beta-", "Sn116", 1.0, Some(0.0))]),
            ),
            (
                "Xx999".to_string(),
                nuclide("Xx999", vec![mode("beta-", "Yy999", 1.0, None)]),
            ),
        ])
    }

    #[test]
    fn stored_sigmas_come_back_as_stored_zeros_included() {
        // The file holds the tape's numbers, and 0.0 is one of them (MT=457's
        // "not stated"). Export has to write back exactly what it read, so a
        // 0.0 stays a 0.0 and a null stays a null: turning one into the other
        // is a claim about the data the data never made.
        let chain = chain_with_branching_sigmas();
        let dir = std::env::temp_dir().join(format!("yani-sigmas-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        super::export_chain_parts(&chain, &dir, Some("test")).expect("export succeeds");
        let (back, _branch) =
            super::parse_chain_parts(&dir.join("decay"), None, None, None).expect("load succeeds");
        let _ = std::fs::remove_dir_all(&dir);

        for (name, written) in &chain {
            let read = &back[name];
            let sigmas = |n: &crate::ChainNuclide| -> Vec<Option<f64>> {
                n.decays.iter().map(|d| d.branching_uncertainty).collect()
            };
            assert_eq!(sigmas(read), sigmas(written), "{name} dBR");
            assert_eq!(read.half_life_uncertainty, Some(0.0), "{name}");
            assert_eq!(read.decay_energy_uncertainty, Some(0.0), "{name}");
            assert_eq!(
                read.decay_energy_components, written.decay_energy_components,
                "{name}"
            );
        }
        assert_eq!(
            back["In116_m1"].decays[0].branching_uncertainty,
            Some(0.0),
            "a stored 0.0 must not come back as null"
        );
        assert_eq!(
            back["Xx999"].decays[0].branching_uncertainty, None,
            "a null must not come back as 0.0"
        );
    }

    #[test]
    fn decay_modes_written_before_the_sigma_column_read_as_unstated() {
        // The published files up to now have four columns. Nullable and last
        // is what lets them keep loading, with every mode carrying no sigma.
        let chain = chain_with_branching_sigmas();
        let dir = std::env::temp_dir().join(format!("yani-old-modes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        super::export_chain_parts(&chain, &dir, Some("test")).expect("export succeeds");

        let mut names: Vec<&String> = chain.keys().collect();
        names.sort();
        let (mut nuclide, mut kind, mut target, mut branching) = (
            super::StringBuilder::new(),
            super::StringBuilder::new(),
            super::StringBuilder::new(),
            super::Float64Builder::new(),
        );
        for name in names {
            for d in &chain[name].decays {
                nuclide.append_value(name);
                kind.append_value(&d.kind);
                target.append_option(d.target.as_deref());
                branching.append_value(d.branching);
            }
        }
        let declared = nuclear_data_schema::decay_decay_modes();
        let old = std::sync::Arc::new(super::Schema::new(declared.fields()[..4].to_vec()));
        assert_eq!(
            declared.fields().len(),
            5,
            "this test writes the layout from before the fifth column"
        );
        let batch = super::RecordBatch::try_new(
            old.clone(),
            vec![
                std::sync::Arc::new(nuclide.finish()),
                std::sync::Arc::new(kind.finish()),
                std::sync::Arc::new(target.finish()),
                std::sync::Arc::new(branching.finish()),
            ],
        )
        .expect("a four-column batch");
        super::write_arrow_file(&dir.join("decay/decay_modes.arrow"), old, batch)
            .expect("overwrite decay_modes.arrow");

        let (back, _branch) = super::parse_chain_parts(&dir.join("decay"), None, None, None)
            .expect("a file without the column still loads");
        let _ = std::fs::remove_dir_all(&dir);
        let decays: Vec<_> = back.values().flat_map(|n| &n.decays).collect();
        assert_eq!(decays.len(), 4, "every mode still loads");
        for d in decays {
            assert_eq!(
                d.branching_uncertainty, None,
                "a column the file does not have must read as unstated, not zero"
            );
        }
    }

    #[test]
    fn exported_parts_match_the_declared_schemas_and_round_trip() {
        use crate::chain::{ChainNuclide, ChainReaction, DecaySource, DecaySourceDistribution};
        use std::collections::HashMap;

        let mut chain: HashMap<String, ChainNuclide> = HashMap::new();
        chain.insert(
            "Co60".to_string(),
            ChainNuclide {
                name: "Co60".to_string(),
                half_life: Some(1.66e8),
                decay_energy: 2.5e6,
                // Not empty: a section written with no rows still carries its
                // schema, but an empty chain also cannot show that a VALUE
                // survives, and Q was being dropped rather than mistyped.
                reactions: vec![ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("Co61".to_string()),
                    branching: 1.0,
                    q_value: Some(7.492e6),
                    branching_uncertainty: None,
                }],
                decays: vec![ChainReaction {
                    kind: "beta-".to_string(),
                    target: Some("Ni60".to_string()),
                    branching: 1.0,
                    q_value: None,
                    branching_uncertainty: None,
                }],
                fission_yields: None,
                sources: vec![DecaySource {
                    particle: "photon".to_string(),
                    distribution: DecaySourceDistribution::Discrete {
                        energies: vec![1.17e6, 1.33e6],
                        intensities: vec![1.0, 1.0],
                    },
                }],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
                decay_energy_components: Default::default(),
            },
        );

        let dir = std::env::temp_dir().join(format!("yani-parts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        super::export_chain_parts(&chain, &dir, Some("test")).expect("export succeeds");

        // Every section export_chain_parts writes, not a subset. The two it
        // used to skip are where the bug was: reactions/reactions.arrow was
        // written with the decay_modes schema, so it had no Q column at all.
        // And `if !path.exists() { continue }` would have hidden it anyway,
        // which is why a missing file is now a failure: this writer emits all
        // four unconditionally.
        for section in [
            "decay/nuclides.arrow",
            "decay/decay_modes.arrow",
            "decay/sources.arrow",
            "reactions/reactions.arrow",
        ] {
            let path = dir.join(section);
            assert!(
                path.exists(),
                "{section} was not written; export_chain_parts writes it unconditionally"
            );
            let batches = super::read_arrow_file(&path).expect("readable");
            let declared = nuclear_data_schema::section(section).expect("declared");
            let batch_schema = batches[0].schema();
            let written: Vec<String> = batch_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            let expected: Vec<String> =
                declared.fields().iter().map(|f| f.name().clone()).collect();
            assert_eq!(
                written, expected,
                "{section} written with the wrong columns"
            );
        }

        let (back, _branch) = parse_chain_parts(
            &dir.join("decay"),
            Some(&dir.join("reactions")),
            Some(&dir.join("fission_yields")),
            None,
        )
        .expect("round trips");
        let co60 = back.get("Co60").expect("Co60 survives the round trip");
        assert_eq!(co60.sources.len(), 1);
        assert_eq!(co60.sources[0].particle, "photon");

        // Q has to come back, not just be declared. Dropping it here is
        // invisible to every other assertion: the file still parses, the chain
        // still transmutes, and the number is simply gone.
        assert_eq!(co60.reactions.len(), 1);
        assert_eq!(co60.reactions[0].q_value, Some(7.492e6));
        assert_eq!(co60.decays.len(), 1);
        assert_eq!(
            co60.decays[0].q_value, None,
            "decay/decay_modes.arrow declares no Q column"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A fissioning nuclide with the given nominal yield energies, the first
    /// of them carrying evaluated yields when `evaluated` is set.
    fn fissioning(name: &str, energies: &[f64], evaluated: bool) -> crate::chain::ChainNuclide {
        use crate::chain::{EvaluatedYields, FissionYield, FissionYieldSet};
        let yields = energies
            .iter()
            .enumerate()
            .map(|(i, &energy)| FissionYield {
                energy,
                // A merged entry: the tape's I135 and I135_m1 summed.
                products: vec![("I135".to_string(), 0.07), ("Cs137".to_string(), 0.06)],
                independent: (evaluated && i == 0).then(|| EvaluatedYields {
                    products: vec!["I135".into(), "I135_m1".into(), "Cs137".into()],
                    yields: vec![0.05, 0.02, 0.06],
                    // An evaluator's 0.0 and a null must both come back as
                    // written, not one turned into the other.
                    uncertainties: vec![Some(0.001), Some(0.0), None],
                    interpolation: None,
                }),
                cumulative: (evaluated && i == 0).then(|| EvaluatedYields {
                    products: vec!["Cs137".into()],
                    yields: vec![0.0619],
                    uncertainties: vec![Some(3.1e-4)],
                    interpolation: Some(2),
                }),
            })
            .collect();
        crate::chain::ChainNuclide {
            name: name.to_string(),
            half_life: Some(2.2e16),
            half_life_uncertainty: None,
            decay_energy: 0.0,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
            reactions: Vec::new(),
            decays: Vec::new(),
            fission_yields: Some(std::sync::Arc::new(FissionYieldSet::new(yields))),
            sources: Vec::new(),
        }
    }

    #[test]
    fn evaluated_yields_round_trip_and_absence_is_none() {
        let dir = std::env::temp_dir().join(format!("yani-evaluated-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut chain = std::collections::HashMap::new();
        chain.insert(
            "U235".to_string(),
            fissioning("U235", &[0.0253, 5.0e5], true),
        );
        chain.insert("Pu239".to_string(), fissioning("Pu239", &[0.0253], false));
        super::export_chain_parts(&chain, &dir, Some("test")).expect("export succeeds");

        let path = dir.join("fission_yields/evaluated_yields.arrow");
        let batches = super::read_arrow_file(&path).expect("evaluated yields written");
        let declared = nuclear_data_schema::section("fission_yields/evaluated_yields.arrow")
            .expect("declared");
        assert_eq!(batches[0].schema().fields(), declared.fields());
        // One independent and one cumulative row, and nothing for the energy
        // or the nuclide that carries none.
        assert_eq!(batches[0].num_rows(), 2);

        let read = |dir: &std::path::Path| {
            parse_chain_parts(
                &dir.join("decay"),
                Some(&dir.join("reactions")),
                Some(&dir.join("fission_yields")),
                None,
            )
            .map(|(chain, _)| chain)
        };
        let back = read(&dir).expect("round trips");
        for name in ["U235", "Pu239"] {
            let (written, read) = (
                &chain[name].fission_yields.as_ref().unwrap().yields,
                &back[name].fission_yields.as_ref().unwrap().yields,
            );
            assert_eq!(written.len(), read.len());
            for (w, r) in written.iter().zip(read) {
                assert_eq!(w.energy, r.energy);
                assert_eq!(w.products, r.products, "{name}: the nominal yields moved");
                assert_eq!(w.independent, r.independent, "{name} at {} eV", w.energy);
                assert_eq!(w.cumulative, r.cumulative, "{name} at {} eV", w.energy);
            }
        }

        // Without the file the nominal yields read the same and nothing is
        // evaluated, which is how a library published before it loads.
        std::fs::remove_file(&path).unwrap();
        let bare = read(&dir).expect("loads without evaluated yields");
        let u235 = &bare["U235"].fission_yields.as_ref().unwrap().yields;
        assert!(u235
            .iter()
            .all(|y| y.independent.is_none() && y.cumulative.is_none()));
        assert_eq!(
            u235[0].products,
            back["U235"].fission_yields.as_ref().unwrap().yields[0].products
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Exporting a chain with no evaluated yields over a directory that holds
    /// some from an earlier export removes them. Left in place they would
    /// attach, by nuclide and energy, to yields they were not written from.
    #[test]
    fn an_export_without_evaluated_yields_removes_stale_ones() {
        let dir = std::env::temp_dir().join(format!("yani-stale-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let chain = |evaluated| {
            std::collections::HashMap::from([(
                "U235".to_string(),
                fissioning("U235", &[0.0253], evaluated),
            )])
        };
        super::export_chain_parts(&chain(true), &dir, None).unwrap();
        let path = dir.join("fission_yields/evaluated_yields.arrow");
        assert!(path.exists());
        // A converter's aliases.arrow has no place beside an export either,
        // which writes every nuclide's yields in full.
        std::fs::write(dir.join("fission_yields/aliases.arrow"), b"stale").unwrap();

        super::export_chain_parts(&chain(false), &dir, None).unwrap();
        assert!(!path.exists(), "a stale evaluated_yields.arrow was left");
        assert!(!dir.join("fission_yields/aliases.arrow").exists());
        let (back, _) = parse_chain_parts(
            &dir.join("decay"),
            None,
            Some(&dir.join("fission_yields")),
            None,
        )
        .expect("reloads");
        let yields = &back["U235"].fission_yields.as_ref().unwrap().yields;
        assert!(yields[0].independent.is_none() && yields[0].cumulative.is_none());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// An evaluated row with no nominal row to belong to is refused, not
    /// dropped: it is the data the file exists to keep.
    #[test]
    fn an_evaluated_yield_with_no_nominal_row_is_refused() {
        let root = std::env::temp_dir().join(format!("yani-orphan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let with = std::collections::HashMap::from([(
            "U235".to_string(),
            fissioning("U235", &[0.0253], true),
        )]);
        let without = std::collections::HashMap::from([(
            "U235".to_string(),
            fissioning("U235", &[5.0e5], false),
        )]);
        super::export_chain_parts(&with, root.join("a"), None).unwrap();
        super::export_chain_parts(&without, root.join("b"), None).unwrap();
        std::fs::copy(
            root.join("a/fission_yields/evaluated_yields.arrow"),
            root.join("b/fission_yields/evaluated_yields.arrow"),
        )
        .unwrap();

        let err = parse_chain_parts(
            &root.join("b/decay"),
            None,
            Some(&root.join("b/fission_yields")),
            None,
        )
        .expect_err("an orphaned evaluated row must not load")
        .to_string();
        assert!(err.contains("U235 at 0.0253 eV has no row"), "got: {err}");

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The sections of a one-nuclide chain whose first energy carries an
    /// independent and a cumulative row, with `evaluated_yields.arrow` passed
    /// through `tamper` before it is handed back as bytes.
    fn tampered_sections(
        tamper: impl FnOnce(arrow_array::RecordBatch) -> Vec<arrow_array::RecordBatch>,
    ) -> super::ChainSections {
        let dir = std::env::temp_dir().join(format!(
            "yani-tamper-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let chain = std::collections::HashMap::from([(
            "U235".to_string(),
            fissioning("U235", &[0.0253, 5.0e5], true),
        )]);
        super::export_chain_parts(&chain, &dir, None).unwrap();
        let mut parts = super::ChainSections::default();
        for (subsection, file) in [
            ("decay", "nuclides.arrow"),
            ("fission_yields", "fission_yields.arrow"),
        ] {
            let bytes = std::fs::read(dir.join(subsection).join(file)).unwrap();
            parts.insert(subsection, file, bytes).unwrap();
        }
        let path = dir.join("fission_yields/evaluated_yields.arrow");
        let batch = super::read_arrow_file(&path).unwrap().remove(0);
        let schema = batch.schema();
        let mut bytes = Vec::new();
        let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut bytes, &schema).unwrap();
        for batch in tamper(batch) {
            writer.write(&batch).unwrap();
        }
        writer.finish().unwrap();
        drop(writer);
        parts
            .insert("fission_yields", "evaluated_yields.arrow", bytes)
            .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        parts
    }

    /// `batch` with the column `name` replaced by `values`.
    fn replace_column(
        batch: arrow_array::RecordBatch,
        name: &str,
        values: arrow_array::ArrayRef,
    ) -> arrow_array::RecordBatch {
        let index = batch.schema().index_of(name).unwrap();
        let mut columns = batch.columns().to_vec();
        columns[index] = values;
        arrow_array::RecordBatch::try_new(batch.schema(), columns).unwrap()
    }

    fn refusal(parts: &super::ChainSections) -> String {
        super::parse_chain_parts_from_bytes(parts)
            .expect_err("a damaged evaluated_yields.arrow must not load")
            .to_string()
    }

    /// A kind the reader does not know would otherwise have nowhere to go, and
    /// guessing which slot it belongs in would mislabel the data.
    #[test]
    fn an_evaluated_yield_of_unknown_kind_is_refused() {
        let parts = tampered_sections(|batch| {
            let kinds = arrow_array::StringArray::from(vec!["independent", "fission"]);
            vec![replace_column(batch, "kind", std::sync::Arc::new(kinds))]
        });
        let err = refusal(&parts);
        assert!(err.contains("unknown kind \"fission\""), "got: {err}");
    }

    /// Two rows for one slot cannot both be kept, and keeping either would
    /// silently drop the other.
    #[test]
    fn a_duplicated_evaluated_yield_row_is_refused() {
        let parts = tampered_sections(|batch| vec![batch.clone(), batch]);
        let err = refusal(&parts);
        assert!(
            err.contains("U235 has two independent rows at 0.0253 eV"),
            "got: {err}"
        );
    }

    /// A DY stays aligned with its Y only if the lists are the same length.
    #[test]
    fn an_evaluated_yield_with_misaligned_lists_is_refused() {
        let parts = tampered_sections(|batch| {
            let mut yields =
                arrow_array::builder::ListBuilder::new(arrow_array::builder::Float64Builder::new());
            // The independent row loses its last yield.
            yields.values().append_slice(&[0.05, 0.02]);
            yields.append(true);
            yields.values().append_slice(&[0.0619]);
            yields.append(true);
            vec![replace_column(
                batch,
                "yields",
                std::sync::Arc::new(yields.finish()),
            )]
        });
        let err = refusal(&parts);
        assert!(
            err.contains("U235 independent at 0.0253 eV has 3 products, 2 yields"),
            "got: {err}"
        );
    }

    /// A null yield is not a stated one, so it is refused rather than read as
    /// an exact 0.0.
    #[test]
    fn an_evaluated_yield_with_a_null_item_is_refused() {
        let parts = tampered_sections(|batch| {
            let mut yields =
                arrow_array::builder::ListBuilder::new(arrow_array::builder::Float64Builder::new());
            yields.values().append_value(0.05);
            yields.values().append_null();
            yields.values().append_value(0.01);
            yields.append(true);
            yields.values().append_slice(&[0.0619]);
            yields.append(true);
            vec![replace_column(
                batch,
                "yields",
                std::sync::Arc::new(yields.finish()),
            )]
        });
        let err = refusal(&parts);
        assert!(
            err.contains("U235 independent at 0.0253 eV has a null yield"),
            "got: {err}"
        );
    }

    /// With no nominal yields supplied at all, every evaluated row is an
    /// orphan, and the reader says which file is missing.
    #[test]
    fn evaluated_yields_without_nominal_yields_are_refused() {
        let mut parts = tampered_sections(|batch| vec![batch]);
        parts.fission_yields.remove("fission_yields.arrow");
        let err = refusal(&parts);
        assert!(
            err.contains("supplied without fission_yields/fission_yields.arrow"),
            "got: {err}"
        );
    }

    /// A chain directory in the retired flat layout must be refused, not read
    /// as an empty chain: every section below the decay index is optional, so
    /// without the guard a stale cache directory transmutes nothing at all and
    /// says nothing about it.
    #[test]
    fn flat_layout_is_refused_not_silently_empty() {
        let root = std::env::temp_dir().join(format!("yani_flat_chain_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        // The v1 marker: the index sits at the root, with no subsection dirs.
        std::fs::write(root.join("nuclides.arrow"), b"").unwrap();

        let err = parse_chain_parts(
            &root.join("decay"),
            Some(&root.join("reactions")),
            Some(&root.join("fission_yields")),
            None,
        )
        .expect_err("a flat-layout directory must not load as an empty chain")
        .to_string();
        assert!(
            err.contains("retired flat chain layout"),
            "message should name the cause, got: {err}"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A directory that is neither layout gets the generic message rather than
    /// the flat-layout diagnosis.
    #[test]
    fn missing_decay_index_is_refused() {
        let root = std::env::temp_dir().join(format!("yani_empty_chain_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let err = parse_chain_parts(
            &root.join("decay"),
            Some(&root.join("reactions")),
            Some(&root.join("fission_yields")),
            None,
        )
        .expect_err("a directory with no decay index must not load")
        .to_string();
        assert!(err.contains("v2 split chain"), "got: {err}");

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn graft_policy() {
        // (n,n') metastable: base chains carry no (n,n') at all, still graft.
        assert!(should_graft("(n,n')", "Pb204_m1", "Pb204", true, false));
        // (n,n') ground self-loop (target == parent) is a no-op, never grafted.
        assert!(!should_graft("(n,n')", "Pb204", "Pb204", true, false));
        // (n,2n) metastable: graft only when the base carries the (n,2n) ground,
        // otherwise the parent would deplete into an unproduced target.
        assert!(should_graft("(n,2n)", "Ag106_m1", "Ag107", true, true));
        assert!(!should_graft("(n,2n)", "Rh102_m1", "Ag107", true, false));
        // A target with no decay node is not a chain nuclide: never graft.
        assert!(!should_graft("(n,2n)", "Zz999_m1", "Ag107", false, true));
    }

    #[test]
    fn decay_target_policy() {
        use super::modelled_decay_target as target;
        // An ordinary mode keeps its daughter, and one with none stays without.
        assert_eq!(
            target("Cf252", "alpha", Some("Cm248")).as_deref(),
            Some("Cm248")
        );
        assert_eq!(target("He5", "alpha", None), None);
        // An isomer's transition to its own ground state is not a self-edge.
        assert_eq!(
            target("W185_m1", "IT", Some("W185")).as_deref(),
            Some("W185")
        );
        // Spontaneous fission names no product, however its target was
        // written: the parent, an isomer's ground state, a delayed fission's
        // intermediate, or a replace_missing substitute.
        assert_eq!(target("Cf252", "sf", Some("Cf252")), None);
        assert_eq!(target("Am242_m2", "sf", Some("Am242")), None);
        assert_eq!(target("Tl180", "ec/beta+,sf", Some("Hg180")), None);
        assert_eq!(target("Ds279_m1", "sf", Some("Fm259")), None);
        // Any other mode back into its own parent is a replace_missing loop.
        assert_eq!(target("Es258", "ec/beta+", Some("Es258")), None);
    }
}
