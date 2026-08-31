//! Build the transmutation Arrow format from ENDF evaluations.
//!
//! One half of the answer to "yamc and yani should each convert ENDF into the
//! format they need". This is the yani half: decay data, neutron reaction
//! targets and fission yields, which is everything an activation calculation
//! reads and nothing more. It needs no NJOY, so unlike the transport side it
//! can run from a bare `pip install`.
//!
//! Written against [`endf::Chain`] rather than [`yani`]'s own chain types, for
//! two reasons that are not stylistic:
//!
//! * `endf::chain::ReactionPath` carries `q_value` and
//!   `endf::chain::Nuclide` carries `borrowed_yields_from`. The yani types
//!   carry neither, so a converter built on them could not write the `Q` column
//!   `reactions/reactions.arrow` declares, nor `fission_yields/aliases.arrow`
//!   at all: it would have to expand every alias into a full copy of its
//!   parent's yields and lose the relationship.
//! * `yani::export_chain_parts` exists to round-trip a chain yani already
//!   holds. That is a different job from converting an evaluation, and reusing
//!   it would have meant widening the in-memory type to carry fields the solver
//!   never reads.
//!
//! Decay source spectra are the one thing [`endf::Chain`] does not carry, so
//! they are read from the same decay evaluations separately and joined by
//! nuclide. See [`decay_sources`].

pub mod branching;

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_ipc::writer::{FileWriter, IpcWriteOptions};
use arrow_ipc::CompressionType;
use arrow_schema::{ArrowError, Schema};

use endf::chain::Chain;
use endf::{Decay, Material};

/// The declared schema for a section, by its path in the format.
///
/// Panics on an unknown path, which can only be a typo: the schema crate is the
/// whole set of sections this format has.
fn section_schema(path: &str) -> Schema {
    nuclear_data_schema::section(path)
        .unwrap_or_else(|| panic!("no declared schema for section {path}"))
}

/// LZ4 for every subsection, matching the transport sections and the published
/// libraries. See the note on `yamc_convert::sections`: uncompressed is the same
/// data at roughly 2.4x the size, and the generation scripts' integrity gate
/// rejects it.
fn section_write_options() -> Result<IpcWriteOptions, ArrowError> {
    IpcWriteOptions::default().try_with_compression(Some(CompressionType::LZ4_FRAME))
}

pub(crate) fn write_section(
    path: &Path,
    section: &str,
    columns: Vec<ArrayRef>,
) -> Result<(), Box<dyn Error>> {
    let schema = Arc::new(section_schema(section));
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let mut writer =
        FileWriter::try_new_with_options(File::create(path)?, &schema, section_write_options()?)?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(())
}

/// The decay photon, electron and other particle spectra, by nuclide.
///
/// [`endf::Chain`] describes who decays into what, not what comes out, so this
/// reads the same decay evaluations a second time. [`endf::Decay::sources`]
/// returns intensities already multiplied by the decay constant and the
/// spectrum normalisation, which is the per atom per second convention the
/// format stores. Reading `Decay::spectra` instead would be low by both
/// factors, silently.
pub fn decay_sources(
    decay: &[Material],
) -> Result<BTreeMap<String, Vec<SourceRow>>, Box<dyn Error>> {
    let mut out: BTreeMap<String, Vec<SourceRow>> = BTreeMap::new();
    for material in decay {
        let Ok(d) = Decay::from_material(material) else {
            continue;
        };
        // The neutron's own decay evaluation is not a chain nuclide, and
        // Chain::from_endf skips it. Emitting its spectrum anyway leaves a
        // sources row pointing at a nuclide the chain does not contain, which
        // the reader turns into a phantom.
        if d.nuclide.atomic_number == 0 {
            continue;
        }
        let name = d.nuclide.name.clone();
        for (particle, dist) in d.sources()? {
            for row in flatten(particle, &dist) {
                out.entry(name.clone()).or_default().push(row);
            }
        }
    }
    Ok(out)
}

/// One row of `decay/sources.arrow`.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceRow {
    pub particle: String,
    /// `"discrete"` or `"tabular"`.
    pub kind: String,
    pub energies: Vec<f64>,
    pub intensities: Vec<f64>,
}

/// Flatten one distribution into rows.
///
/// A mixture becomes one row per component with the component's probability
/// multiplied into its intensities, which is what keeps the total emission rate
/// right without the format needing a mixture concept.
fn flatten(particle: &str, dist: &endf::univariate::Univariate) -> Vec<SourceRow> {
    use endf::univariate::Univariate;
    match dist {
        Univariate::Discrete(d) => vec![SourceRow {
            particle: particle.to_string(),
            kind: "discrete".to_string(),
            energies: d.x.clone(),
            intensities: d.p.clone(),
        }],
        Univariate::Tabular(t) => vec![SourceRow {
            particle: particle.to_string(),
            kind: "tabular".to_string(),
            energies: t.x.clone(),
            intensities: t.p.clone(),
        }],
        Univariate::Mixture(m) => m
            .probability
            .iter()
            .zip(m.distribution.iter())
            .flat_map(|(w, d)| {
                flatten(particle, d).into_iter().map(move |mut row| {
                    for value in &mut row.intensities {
                        *value *= w;
                    }
                    row
                })
            })
            .collect(),
        _ => Vec::new(),
    }
}

pub(crate) fn list_of(values: &[Vec<f64>]) -> ArrayRef {
    let mut b = ListBuilder::new(Float64Builder::new());
    for row in values {
        b.values().append_slice(row);
        b.append(true);
    }
    Arc::new(b.finish())
}

pub(crate) fn strings(values: &[String]) -> ArrayRef {
    let mut b = StringBuilder::new();
    for v in values {
        b.append_value(v);
    }
    Arc::new(b.finish())
}

pub(crate) fn opt_strings(values: &[Option<String>]) -> ArrayRef {
    let mut b = StringBuilder::new();
    for v in values {
        match v {
            Some(s) => b.append_value(s),
            None => b.append_null(),
        }
    }
    Arc::new(b.finish())
}

pub(crate) fn floats(values: &[f64]) -> ArrayRef {
    let mut b = Float64Builder::new();
    b.append_slice(values);
    Arc::new(b.finish())
}

pub(crate) fn opt_floats(values: &[Option<f64>]) -> ArrayRef {
    let mut b = Float64Builder::new();
    for v in values {
        match v {
            Some(x) => b.append_value(*x),
            None => b.append_null(),
        }
    }
    Arc::new(b.finish())
}

/// Write the `decay/` subsection.
pub fn write_decay(
    chain: &Chain,
    sources: &BTreeMap<String, Vec<SourceRow>>,
    dir: &Path,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(dir)?;

    let mut names = Vec::new();
    let mut half_lives = Vec::new();
    let mut decay_energies = Vec::new();
    let mut half_life_sigmas = Vec::new();
    let mut decay_energy_sigmas = Vec::new();
    for n in &chain.nuclides {
        names.push(n.name.clone());
        half_lives.push(n.half_life);
        decay_energies.push(n.decay_energy);
        half_life_sigmas.push(n.half_life_uncertainty);
        decay_energy_sigmas.push(n.decay_energy_uncertainty);
    }
    write_section(
        &dir.join("nuclides.arrow"),
        "decay/nuclides.arrow",
        vec![
            strings(&names),
            opt_floats(&half_lives),
            floats(&decay_energies),
            opt_floats(&half_life_sigmas),
            opt_floats(&decay_energy_sigmas),
        ],
    )?;

    let mut nuc = Vec::new();
    let mut kind = Vec::new();
    let mut target = Vec::new();
    let mut branching = Vec::new();
    for n in &chain.nuclides {
        for d in &n.decay_modes {
            nuc.push(n.name.clone());
            kind.push(d.kind.clone());
            target.push(d.target.clone());
            branching.push(d.branching_ratio);
        }
    }
    write_section(
        &dir.join("decay_modes.arrow"),
        "decay/decay_modes.arrow",
        vec![
            strings(&nuc),
            strings(&kind),
            opt_strings(&target),
            floats(&branching),
        ],
    )?;

    let mut nuc = Vec::new();
    let mut particle = Vec::new();
    let mut kind = Vec::new();
    let mut energies = Vec::new();
    let mut intensities = Vec::new();
    for (name, rows) in sources {
        for row in rows {
            nuc.push(name.clone());
            particle.push(row.particle.clone());
            kind.push(row.kind.clone());
            energies.push(row.energies.clone());
            intensities.push(row.intensities.clone());
        }
    }
    write_section(
        &dir.join("sources.arrow"),
        "decay/sources.arrow",
        vec![
            strings(&nuc),
            strings(&particle),
            strings(&kind),
            list_of(&energies),
            list_of(&intensities),
        ],
    )?;
    Ok(())
}

/// Write the `reactions/` subsection.
///
/// The `Q` column is why this is written from `endf::Chain`: it is declared
/// non-nullable and the yani chain types have nowhere to hold it.
pub fn write_reactions(chain: &Chain, dir: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(dir)?;
    let mut nuc = Vec::new();
    let mut kind = Vec::new();
    let mut target = Vec::new();
    let mut q = Vec::new();
    let mut branching = Vec::new();
    for n in &chain.nuclides {
        for r in &n.reactions {
            nuc.push(n.name.clone());
            kind.push(r.kind.clone());
            target.push(r.target.clone());
            q.push(r.q_value);
            branching.push(r.branching_ratio);
        }
    }
    write_section(
        &dir.join("reactions.arrow"),
        "reactions/reactions.arrow",
        vec![
            strings(&nuc),
            strings(&kind),
            opt_strings(&target),
            floats(&q),
            floats(&branching),
        ],
    )
}

/// Write the `fission_yields/` subsection.
///
/// A nuclide with no yields of its own borrows another's, and that is recorded
/// as an alias rather than by copying the parent's product and yield vectors
/// into every inheritor. The copy is what the format is trying to avoid: it
/// destroys the relationship and multiplies the file size by the number of
/// inheritors.
pub fn write_fission_yields(chain: &Chain, dir: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(dir)?;

    let mut nuc = Vec::new();
    let mut energy = Vec::new();
    let mut products: Vec<Vec<String>> = Vec::new();
    let mut yields: Vec<Vec<f64>> = Vec::new();
    let mut alias_nuc = Vec::new();
    let mut alias_parent = Vec::new();

    for n in &chain.nuclides {
        if let Some(parent) = &n.borrowed_yields_from {
            alias_nuc.push(n.name.clone());
            alias_parent.push(parent.clone());
            continue;
        }
        // Every energy carries the same product list, padded with zeros where
        // the evaluation lists a product at one incident energy and not
        // another. The values are unchanged either way, since a padded entry is
        // exactly 0.0, but a uniform list means a consumer can index products
        // positionally across energies without checking. It is also what the
        // Python converter writes, and a difference here would be a difference
        // in published data for no reason.
        let union: std::collections::BTreeSet<&String> =
            n.yield_data.values().flat_map(|m| m.keys()).collect();
        for (e, by_product) in &n.yield_data {
            nuc.push(n.name.clone());
            energy.push(e.parse::<f64>().unwrap_or(0.0));
            products.push(union.iter().map(|p| (*p).clone()).collect());
            yields.push(
                union
                    .iter()
                    .map(|p| by_product.get(*p).copied().unwrap_or(0.0))
                    .collect(),
            );
        }
    }

    let mut product_lists = ListBuilder::new(StringBuilder::new());
    for row in &products {
        for p in row {
            product_lists.values().append_value(p);
        }
        product_lists.append(true);
    }
    write_section(
        &dir.join("fission_yields.arrow"),
        "fission_yields/fission_yields.arrow",
        vec![
            strings(&nuc),
            floats(&energy),
            Arc::new(product_lists.finish()),
            list_of(&yields),
        ],
    )?;

    // Written only when there is something to say, matching the Python
    // converter: a library with no borrowed yields leaves no aliases file.
    if !alias_nuc.is_empty() {
        write_section(
            &dir.join("aliases.arrow"),
            "fission_yields/aliases.arrow",
            vec![strings(&alias_nuc), strings(&alias_parent)],
        )?;
    }
    Ok(())
}

/// Provenance and the chain manifest, matching what the Python converter
/// writes so a consumer cannot tell which produced a directory.
///
/// `data_version` identifies the published release rather than the code, and is
/// what yamc compares a cached copy against (issue #366). It is supplied by the
/// build: only the build knows whether a run is a new release or a resumed one.
fn write_provenance(
    dir: &Path,
    subsection: &str,
    library: &str,
    decay_library: &str,
    data_version: &str,
    created_utc: &str,
) -> Result<(), Box<dyn Error>> {
    let body = serde_json::json!({
        "subsection": subsection,
        "library": library,
        // Always present, empty when the caller did not say, so a reader can
        // tell "not recorded" from "same as library" rather than guessing.
        "decay_library": decay_library,
        "data_version": data_version,
        "source": "endf",
        "converter_version": concat!("yani-convert ", env!("CARGO_PKG_VERSION")),
        "created_utc": created_utc,
    });
    std::fs::write(
        dir.join("provenance.json"),
        serde_json::to_string_pretty(&body)?,
    )?;
    Ok(())
}

/// Record `written` in the chain's manifest, keeping what is already there.
///
/// Merged rather than overwritten because a chain is assembled by more than one
/// call and no call sees all of it. Overwriting is how a library ends up
/// advertising one subsection while shipping four. An entry whose directory has
/// gone is dropped rather than left advertising something removed, and a
/// directory rebuilt for a different library starts over.
/// `parents` is the set of nuclides the written subsections carry reactions
/// for, recorded per subsection so a reader can tell what the chain can answer
/// for before it solves anything. A chain built for one foil's isotopes solves
/// a different foil to an inventory of nothing, silently, because every
/// reaction it holds has the wrong parent; without this the only way to find
/// that out is to notice the decay heat came back as zero.
fn merge_manifest(
    out: &Path,
    provenance: &Provenance,
    written: &[&str],
    parents: &[String],
) -> Result<(), Box<dyn Error>> {
    let manifest_path = out.join("manifest.json");
    let mut subsections = serde_json::Map::new();
    if let Ok(text) = std::fs::read_to_string(&manifest_path) {
        if let Ok(existing) = serde_json::from_str::<serde_json::Value>(&text) {
            if existing.get("library").and_then(|v| v.as_str()) == Some(provenance.library.as_str())
            {
                if let Some(map) = existing.get("subsections").and_then(|v| v.as_object()) {
                    subsections = map.clone();
                }
            }
        }
    }
    for subsection in written {
        // Replaced rather than merged with what was there. The subsection's
        // data was just overwritten, so its scope is whatever this call wrote;
        // carrying over a previous call's parents would advertise reactions the
        // directory no longer holds, which is the failure this exists to catch.
        subsections.insert(
            (*subsection).to_string(),
            serde_json::json!({ "path": subsection, "parents": parents }),
        );
    }
    subsections.retain(|_, entry| {
        entry
            .get("path")
            .and_then(|p| p.as_str())
            .is_some_and(|p| out.join(p).is_dir())
    });
    let manifest = serde_json::json!({
        "format_version": 2,
        "library": provenance.library,
        "data_version": provenance.data_version,
        "converter_version": concat!("yani-convert ", env!("CARGO_PKG_VERSION")),
        "created_utc": provenance.created_utc,
        "subsections": subsections,
    });
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}

/// What a converted directory records about where it came from.
#[derive(Debug, Clone, Default)]
pub struct Provenance {
    /// Library name, e.g. `"endf-b8.1"`.
    pub library: String,
    /// Which library the DECAY evaluations came from, when that is not
    /// `library`. TENDL publishes no decay data of its own and borrows
    /// ENDF/B-VIII.1's, so without this a TENDL chain reads as though its decay
    /// half were TENDL too. Empty when the caller did not say, which is what
    /// every existing output records.
    pub decay_library: String,
    /// The published release this output belongs to. Identifies the DATA, not
    /// the code: two rebuilds from one converter are different data and must
    /// invalidate a cache (issue #366).
    pub data_version: String,
    /// Supplied rather than read from the clock, so a caller that wants a
    /// reproducible directory can fix it.
    pub created_utc: String,
}

/// The evaluations a chain is built from.
///
/// Grouped because they always travel together and because which of them is
/// required depends on the subsections being written, which is easier to say
/// about one value than about three parameters.
#[derive(Debug, Clone, Default)]
pub struct Inputs<'a> {
    pub decay: &'a [Material],
    pub fpy: &'a [Material],
    pub neutron: &'a [Material],
}

/// Convert decay, fission yield and neutron evaluations into a transmutation
/// directory.
///
/// The whole yani half of "convert ENDF into the format we need", in one call
/// and with no external tools: no NJOY, no HDF5, no Python.
///
/// `created_utc` is passed in rather than read from the clock, so a caller that
/// needs a reproducible directory can supply a fixed stamp and get byte-stable
/// output.
pub fn convert_transmutation(
    inputs: &Inputs,
    reactions: &[&str],
    branch_ratios: Option<&Path>,
    subsections: &[&str],
    out: &Path,
    provenance: &Provenance,
) -> Result<Chain, Box<dyn Error>> {
    let (decay, fpy, neutron) = (inputs.decay, inputs.fpy, inputs.neutron);
    let Provenance {
        library,
        decay_library,
        data_version,
        created_utc,
    } = provenance;
    let mut chain = Chain::from_endf(decay, fpy, neutron, reactions)?;
    if let Some(path) = branch_ratios {
        apply_branch_ratios(&mut chain, path)?;
    }
    let chain = chain;
    let sources = decay_sources(decay)?;

    std::fs::create_dir_all(out)?;
    let wants = |name: &str| subsections.contains(&name);
    if wants("decay") {
        write_decay(&chain, &sources, &out.join("decay"))?;
    }
    if wants("reactions") {
        write_reactions(&chain, &out.join("reactions"))?;
    }
    if wants("fission_yields") {
        write_fission_yields(&chain, &out.join("fission_yields"))?;
    }

    for subsection in subsections {
        write_provenance(
            &out.join(subsection),
            subsection,
            library,
            decay_library,
            data_version,
            created_utc,
        )?;
    }

    // The nuclides this chain can actually be driven from. A nuclide with no
    // reactions is reachable as a product but cannot start anything, so it is
    // not a parent and listing it would let a material through that the chain
    // has nothing to say about.
    let parents: Vec<String> = chain
        .nuclides
        .iter()
        .filter(|n| !n.reactions.is_empty())
        .map(|n| n.name.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    merge_manifest(out, provenance, subsections, &parents)?;

    Ok(chain)
}

/// Split a reaction between a ground state and its metastable partners.
///
/// Without this every `(n,gamma)` gives its whole rate to the ground state, and
/// the metastable product simply is not in the network: 101 of the 5175
/// reaction rows in ENDF/B-VIII.1 are metastable targets that exist only
/// because a branching table put them there. The file is the `openmc_data`
/// shape, `{reaction: {parent: {target: fraction}}}`.
fn apply_branch_ratios(chain: &mut Chain, path: &Path) -> Result<(), Box<dyn Error>> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let all: BTreeMap<String, BTreeMap<String, BTreeMap<String, f64>>> =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    for (reaction, ratios) in &all {
        // Not strict: a table is written for a whole library and routinely
        // names parents a given chain does not contain. The Python converter
        // passes strict=False for the same reason.
        let known: BTreeMap<String, BTreeMap<String, f64>> = ratios
            .iter()
            .filter(|(parent, _)| chain.nuclides.iter().any(|n| &n.name == *parent))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if known.is_empty() {
            continue;
        }
        chain.set_branch_ratios(&known, reaction)?;
    }
    Ok(())
}

/// Convert straight from file paths.
///
/// The entry point a binding wants: it takes strings and returns a count, so
/// nothing above this crate has to name [`endf`] at all. That is the point of
/// the layering rather than a convenience. A yani user never types
/// `import endf`, so `yani-python` should never `use endf` either.
///
/// `reactions` defaults to every reaction the chain builder knows rather than
/// the short default set, because a network quietly missing channels is worse
/// than a slower conversion.
#[allow(clippy::too_many_arguments)]
pub fn convert_transmutation_files(
    decay_files: &[String],
    fpy_files: &[String],
    neutron_files: &[String],
    reactions: Option<&[String]>,
    branch_ratios: Option<&Path>,
    subsections: Option<&[String]>,
    out: &Path,
    provenance: &Provenance,
) -> Result<usize, Box<dyn Error>> {
    let read = |paths: &[String]| -> Result<Vec<Material>, Box<dyn Error>> {
        paths
            .iter()
            .map(|p| Material::from_file(p).map_err(|e| format!("{p}: {e}").into()))
            .collect()
    };
    let decay = read(decay_files)?;
    let fpy = read(fpy_files)?;
    let neutron = read(neutron_files)?;

    // Every reaction the chain builder knows, not endf::chain::DEFAULT_REACTIONS.
    // That short list is six names, and defaulting to it silently drops 2554 of
    // the 5175 reaction rows ENDF/B-VIII.1 produces: every (n,na), (n,np),
    // (n,2na) and the rest simply would not be in the network. The Python
    // converter passes the full set for the same reason.
    let owned: Vec<String> = match reactions {
        Some(names) => names.to_vec(),
        None => endf::chain::REACTIONS
            .iter()
            .map(|r| r.name.to_string())
            .collect(),
    };
    let names: Vec<&str> = owned.iter().map(String::as_str).collect();

    let default_subsections = [
        "decay".to_string(),
        "reactions".to_string(),
        "fission_yields".to_string(),
    ];
    let wanted: Vec<&str> = subsections
        .unwrap_or(&default_subsections)
        .iter()
        .map(String::as_str)
        .collect();
    for s in &wanted {
        if !["decay", "reactions", "fission_yields"].contains(s) {
            return Err(format!(
                "unknown subsection {s:?}; expected decay, reactions or fission_yields. \
                 Branching is written by convert_branching, not here"
            )
            .into());
        }
    }

    let chain = convert_transmutation(
        &Inputs {
            decay: &decay,
            fpy: &fpy,
            neutron: &neutron,
        },
        &names,
        branch_ratios,
        &wanted,
        out,
        provenance,
    )?;
    Ok(chain.nuclides.len())
}

/// Convert the isomeric branching subsection.
///
/// Separate from [`convert_transmutation`] on purpose, mirroring the Python
/// converter: the branching curves and the decay data routinely come from
/// different libraries (TENDL branching beside ENDF/B decay is the usual
/// combination), so they are written by separate calls with their own
/// provenance and merged into the same manifest.
pub fn convert_branching_files(
    neutron_files: &[String],
    decay_files: &[String],
    out: &Path,
    provenance: &Provenance,
    tol_ev: f64,
    linearize_tol: f64,
) -> Result<branching::BranchingStats, Box<dyn Error>> {
    let read = |paths: &[String]| -> Result<Vec<Material>, Box<dyn Error>> {
        paths
            .iter()
            .map(|p| Material::from_file(p).map_err(|e| format!("{p}: {e}").into()))
            .collect()
    };
    let neutron = read(neutron_files)?;
    let decay = read(decay_files)?;

    let (rows, stats) = branching::extract_branching(&neutron, &decay, tol_ev, linearize_tol)?;
    let dir = out.join("branching");
    branching::write_branching(&rows, &dir)?;
    write_provenance(
        &dir,
        "branching",
        &provenance.library,
        &provenance.decay_library,
        &provenance.data_version,
        &provenance.created_utc,
    )?;
    let parents: Vec<String> = rows
        .iter()
        .map(|r| r.nuclide.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    merge_manifest(out, provenance, &["branching"], &parents)?;
    Ok(stats)
}
