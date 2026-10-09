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
//!   holds. That is a different job from converting an evaluation, and the
//!   fields above are exactly the ones such a round trip has no source for.
//!
//! Decay source spectra are the one thing [`endf::Chain`] does not carry, so
//! they are read from the same decay evaluations separately and joined by
//! nuclide. See [`decay_sources`].

pub mod branching;
pub mod production;

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, Int32Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, Int32Array, RecordBatch};
use arrow_ipc::writer::{FileWriter, IpcWriteOptions};
use arrow_ipc::CompressionType;
use arrow_schema::{ArrowError, Schema};

use endf::chain::{collect_q_values, Chain, QValues};
use endf::decay::DecayInconsistency;
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
/// reads the same decay evaluations a second time. [`endf::Decay::spectrum_sources`]
/// returns intensities already multiplied by the decay constant and the
/// spectrum normalisation, which is the per atom per second convention the
/// format stores. Reading `Decay::spectra` instead would be low by both
/// factors, silently.
///
/// One row per spectrum's lines and one per its continuum, never a merge:
/// the gamma and x-ray spectra have their own normalisation and its sigma is
/// common to their lines only, so merging them loses which sigma is whose.
pub fn decay_sources(
    decay: &[Material],
) -> Result<BTreeMap<String, Vec<SourceRow>>, Box<dyn Error>> {
    use endf::univariate::Univariate;
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
        for source in d.spectrum_sources()? {
            let (kind, energies, intensities, interpolation) = match source.distribution {
                Univariate::Discrete(lines) => ("discrete", lines.x, lines.p, None),
                Univariate::Tabular(t) => ("tabular", t.x, t.p, Some(t.interpolation.endf_code())),
                _ => unreachable!("a spectrum source is lines or a continuum"),
            };
            out.entry(name.clone()).or_default().push(SourceRow {
                particle: source.particle.to_string(),
                kind: kind.to_string(),
                energies,
                intensities,
                interpolation,
                radiation: source.radiation.to_string(),
                normalization: source.normalization,
                intensity_uncertainties: source.intensity_uncertainties,
                energy_uncertainties: source.energy_uncertainties,
                covariance: source.covariance,
            });
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
    /// Per atom per second: a line's emission rate on a `discrete` row, the
    /// emission-rate density per eV at each energy on a `tabular` one.
    pub intensities: Vec<f64>,
    /// The ENDF interpolation code a `tabular` row is read with, `None` on a
    /// `discrete` one. Without it the continuum has no integral: the same
    /// points read as a histogram and as linear-linear give different totals.
    pub interpolation: Option<i32>,
    /// The spectrum the row was read from, e.g. `"gamma"` or `"xray"`.
    pub radiation: String,
    /// FD on a `discrete` row, FC on a `tabular` one, with its sigma, as the
    /// tape writes them.
    pub normalization: (f64, f64),
    /// Per line, in the units of `intensities`. `None` on a `tabular` row.
    pub intensity_uncertainties: Option<Vec<f64>>,
    /// Per line dER [eV]. `None` on a `tabular` row.
    pub energy_uncertainties: Option<Vec<f64>>,
    /// The spectrum's stated covariance, where it has one.
    pub covariance: Option<endf::SpectrumCovariance>,
}

pub(crate) fn list_of(values: &[Vec<f64>]) -> ArrayRef {
    let mut b = ListBuilder::new(Float64Builder::new());
    for row in values {
        b.values().append_slice(row);
        b.append(true);
    }
    Arc::new(b.finish())
}

/// A nullable `list<double>` column, `None` written as a null rather than as
/// an empty list, so a reader can tell "not given" from "given and empty".
pub(crate) fn opt_list_of(values: &[Option<Vec<f64>>]) -> ArrayRef {
    let mut b = ListBuilder::new(Float64Builder::new());
    for row in values {
        match row {
            Some(row) => {
                b.values().append_slice(row);
                b.append(true);
            }
            None => b.append_null(),
        }
    }
    Arc::new(b.finish())
}

pub(crate) fn string_lists(values: &[Vec<String>]) -> ArrayRef {
    let mut b = ListBuilder::new(StringBuilder::new());
    for row in values {
        for v in row {
            b.values().append_value(v);
        }
        b.append(true);
    }
    Arc::new(b.finish())
}

pub(crate) fn opt_ints(values: &[Option<i32>]) -> ArrayRef {
    let mut b = Int32Builder::new();
    for v in values {
        b.append_option(*v);
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
    // [component][nuclide], (energy, sigma).
    let mut component_energies: [Vec<Option<f64>>; 3] = Default::default();
    let mut component_sigmas: [Vec<Option<f64>>; 3] = Default::default();
    for n in &chain.nuclides {
        names.push(n.name.clone());
        half_lives.push(n.half_life);
        decay_energies.push(n.decay_energy);
        half_life_sigmas.push(n.half_life_uncertainty);
        decay_energy_sigmas.push(n.decay_energy_uncertainty);
        for (c, part) in n.decay_energy_components.iter().enumerate() {
            component_energies[c].push(part.map(|(e, _)| e));
            component_sigmas[c].push(part.map(|(_, s)| s));
        }
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
            opt_floats(&component_energies[0]),
            opt_floats(&component_sigmas[0]),
            opt_floats(&component_energies[1]),
            opt_floats(&component_sigmas[1]),
            opt_floats(&component_energies[2]),
            opt_floats(&component_sigmas[2]),
        ],
    )?;

    let mut nuc = Vec::new();
    let mut kind = Vec::new();
    let mut target = Vec::new();
    let mut branching = Vec::new();
    let mut branching_sigmas = Vec::new();
    let mut evaluated = Vec::new();
    for n in &chain.nuclides {
        for d in &n.decay_modes {
            nuc.push(n.name.clone());
            kind.push(d.kind.clone());
            target.push(d.target.clone());
            branching.push(d.branching_ratio);
            branching_sigmas.push(d.branching_ratio_uncertainty);
            evaluated.push(d.evaluated_branching_ratio);
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
            floats(&branching_sigmas),
            floats(&evaluated),
        ],
    )?;

    let rows: Vec<(&String, &SourceRow)> = sources
        .iter()
        .flat_map(|(name, rows)| rows.iter().map(move |row| (name, row)))
        .collect();
    let text = |f: fn(&SourceRow) -> &String| -> Vec<String> {
        rows.iter().map(|(_, row)| f(row).clone()).collect()
    };
    let lists = |f: fn(&SourceRow) -> Option<&Vec<f64>>| -> ArrayRef {
        opt_list_of(
            &rows
                .iter()
                .map(|(_, row)| f(row).cloned())
                .collect::<Vec<_>>(),
        )
    };
    let ints = |f: fn(&SourceRow) -> Option<i32>| -> ArrayRef {
        let mut b = Int32Builder::new();
        for (_, row) in &rows {
            b.append_option(f(row));
        }
        Arc::new(b.finish())
    };
    let numbers = |f: fn(&SourceRow) -> f64| -> ArrayRef {
        floats(&rows.iter().map(|(_, row)| f(row)).collect::<Vec<_>>())
    };
    let nuclides: Vec<String> = rows.iter().map(|(name, _)| (*name).clone()).collect();
    // LS and LB are flags of a few values, but they are I11 fields on the
    // tape, so one past i32 is refused rather than wrapped.
    let flag = |name: &str, what: &str, v: i64| -> Result<i32, Box<dyn Error>> {
        i32::try_from(v)
            .map_err(|_| format!("{name}: decay covariance {what} {v} does not fit in i32").into())
    };
    let mut ls = Vec::with_capacity(rows.len());
    let mut lb = Vec::with_capacity(rows.len());
    for (name, row) in &rows {
        let c = row.covariance.as_ref();
        ls.push(
            c.and_then(|c| c.ls)
                .map(|v| flag(name, "LS", v))
                .transpose()?,
        );
        lb.push(c.map(|c| flag(name, "LB", c.lb)).transpose()?);
    }
    write_section(
        &dir.join("sources.arrow"),
        "decay/sources.arrow",
        vec![
            strings(&nuclides),
            strings(&text(|r| &r.particle)),
            strings(&text(|r| &r.kind)),
            lists(|r| Some(&r.energies)),
            lists(|r| Some(&r.intensities)),
            ints(|r| r.interpolation),
            strings(&text(|r| &r.radiation)),
            numbers(|r| r.normalization.0),
            numbers(|r| r.normalization.1),
            lists(|r| r.intensity_uncertainties.as_ref()),
            lists(|r| r.energy_uncertainties.as_ref()),
            Arc::new(Int32Array::from(ls)),
            Arc::new(Int32Array::from(lb)),
            lists(|r| r.covariance.as_ref().map(|c| &c.energies)),
            lists(|r| r.covariance.as_ref().map(|c| &c.values)),
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
    let mut laws: Vec<Option<i32>> = Vec::new();
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
            let e = e.parse::<f64>().unwrap_or(0.0);
            nuc.push(n.name.clone());
            energy.push(e);
            laws.push(nominal_yield_law(n, e)?);
            products.push(union.iter().map(|p| (*p).clone()).collect());
            yields.push(
                union
                    .iter()
                    .map(|p| by_product.get(*p).copied().unwrap_or(0.0))
                    .collect(),
            );
        }
    }

    // Built before anything is written, since it can refuse the chain: a
    // refusal after fission_yields.arrow is on disk would leave a nominal file
    // with nothing beside it, which reads as a library published before the
    // evaluated yields existed.
    let evaluated = evaluated_yields_columns(chain)?;
    write_section(
        &dir.join("fission_yields.arrow"),
        "fission_yields/fission_yields.arrow",
        vec![
            strings(&nuc),
            floats(&energy),
            string_lists(&products),
            list_of(&yields),
            opt_ints(&laws),
        ],
    )?;
    // Both optional files are written only when there is something to say,
    // matching the Python converter. When there is not, one left by an earlier
    // run into the same directory is removed: a reader loads whatever is
    // present and would attach it to yields it was not derived from.
    match evaluated {
        Some(columns) => write_section(
            &dir.join("evaluated_yields.arrow"),
            "fission_yields/evaluated_yields.arrow",
            columns,
        )?,
        None => remove_stale(&dir.join("evaluated_yields.arrow"))?,
    }
    if alias_nuc.is_empty() {
        remove_stale(&dir.join("aliases.arrow"))?;
    } else {
        write_section(
            &dir.join("aliases.arrow"),
            "fission_yields/aliases.arrow",
            vec![strings(&alias_nuc), strings(&alias_parent)],
        )?;
    }
    Ok(())
}

/// The law the nominal yields at `energy` are reached with from the energy
/// below: the I of that energy's MT=454 LIST, which the nominal yields are
/// built from. `None` at the lowest energy, which states LE there instead,
/// and for yields with no evaluation behind them.
///
/// Written as the tape states it. Which laws a reader accepts is the
/// reader's to decide, as it is for the evaluated yields' copy of the same
/// code.
fn nominal_yield_law(n: &endf::chain::Nuclide, energy: f64) -> Result<Option<i32>, Box<dyn Error>> {
    let Some(evaluation) = &n.yield_evaluation else {
        return Ok(None);
    };
    let law = evaluation
        .energies
        .iter()
        .position(|e| *e == energy)
        .and_then(|i| {
            evaluation
                .independent_interpolation
                .get(i)
                .copied()
                .flatten()
        });
    Ok(law.map(i32::try_from).transpose()?)
}

/// Remove an optional file this run has nothing to write into, if an earlier
/// run left one.
fn remove_stale(path: &Path) -> Result<(), Box<dyn Error>> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            Err(format!("removing stale {}: {e}", path.display()).into())
        }
        _ => Ok(()),
    }
}

/// The columns of `fission_yields/evaluated_yields.arrow`: every yield
/// evaluation exactly as its tape gives it, both MT=454 and MT=459, with DY.
/// `None` when no nuclide has an evaluation, so no file is written.
///
/// Nothing here is derived. Where `fission_yields.arrow` maps products onto
/// the decay library and sums the ones that meet, this keeps the tape's
/// products, values and uncertainties as they are, so an evaluator's 0.0 DY is
/// written as 0.0. Only owners get rows: a nuclide that borrows its yields is
/// in `aliases.arrow`, as it is for the nominal yields.
///
/// Every row is at an energy the owner has a nominal row for, since a reader
/// attaches it there. An evaluation that breaks that (cumulative yields with
/// no independent ones beside them) is refused rather than written as a file
/// no reader could load.
fn evaluated_yields_columns(chain: &Chain) -> Result<Option<Vec<ArrayRef>>, Box<dyn Error>> {
    let mut nuc = Vec::new();
    let mut energy = Vec::new();
    let mut kind = Vec::new();
    let mut interpolation = Vec::new();
    let mut products: Vec<Vec<String>> = Vec::new();
    let mut yields: Vec<Vec<f64>> = Vec::new();
    let mut sigmas: Vec<Vec<f64>> = Vec::new();

    for n in &chain.nuclides {
        if n.borrowed_yields_from.is_some() {
            continue;
        }
        let Some(evaluation) = &n.yield_evaluation else {
            continue;
        };
        let nominal = n.yield_energies();
        for (label, sets, laws) in [
            (
                "independent",
                &evaluation.independent,
                &evaluation.independent_interpolation,
            ),
            (
                "cumulative",
                &evaluation.cumulative,
                &evaluation.cumulative_interpolation,
            ),
        ] {
            for ((e, set), law) in evaluation.energies.iter().zip(sets).zip(laws) {
                if !nominal.contains(e) {
                    return Err(format!(
                        "{}: {label} yields at {e} eV have no independent yields at \
                         that energy for the chain to be built from. The fission \
                         yield tape for {} gives MT={} at an energy the chain has no \
                         MT=454 yields for; leave that tape out of the fission yield \
                         inputs, or extend evaluated_yields.arrow so rows need not sit \
                         on a nominal one",
                        n.name,
                        n.name,
                        if label == "independent" { 454 } else { 459 }
                    )
                    .into());
                }
                nuc.push(n.name.clone());
                energy.push(*e);
                kind.push(label.to_string());
                interpolation.push(law.map(i32::try_from).transpose()?);
                products.push(set.iter().map(|p| p.name.clone()).collect());
                yields.push(set.iter().map(|p| p.yield_.0).collect());
                sigmas.push(set.iter().map(|p| p.yield_.1).collect());
            }
        }
    }

    if nuc.is_empty() {
        return Ok(None);
    }
    Ok(Some(vec![
        strings(&nuc),
        floats(&energy),
        strings(&kind),
        opt_ints(&interpolation),
        string_lists(&products),
        list_of(&yields),
        list_of(&sigmas),
    ]))
}

/// Provenance and the chain manifest, matching what the Python converter
/// writes so a consumer cannot tell which produced a directory.
///
/// `data_version` identifies the published release rather than the code, and is
/// what yamc compares a cached copy against. It is supplied by the
/// build: only the build knows whether a run is a new release or a resumed one.
///
/// `extra` is merged into the record: what one subsection has to say about
/// itself beyond the common fields, such as which decay energies are
/// placeholders.
fn write_provenance(
    dir: &Path,
    subsection: &str,
    library: &str,
    decay_library: &str,
    data_version: &str,
    created_utc: &str,
    extra: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<(), Box<dyn Error>> {
    let mut body = serde_json::json!({
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
    if let (Some(extra), Some(record)) = (extra, body.as_object_mut()) {
        for (key, value) in extra {
            record.insert(key.clone(), value.clone());
        }
    }
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
    /// invalidate a cache.
    pub data_version: String,
    /// Supplied rather than read from the clock, so a caller that wants a
    /// reproducible directory can fix it.
    pub created_utc: String,
}

/// What a chain is built from: two sublibraries of evaluations, and the Q
/// values read out of a third.
///
/// Grouped because they always travel together and because which of them is
/// required depends on the subsections being written, which is easier to say
/// about one value than about three parameters.
#[derive(Debug, Clone)]
pub struct Inputs<'a> {
    pub decay: &'a [Material],
    pub fpy: &'a [Material],
    /// The neutron set as Q values rather than as evaluations, because that is
    /// all a chain reads of it and because a caller whose neutron sublibrary
    /// does not fit in memory can fill the map one file at a time.
    /// [`convert_transmutation_files`] does exactly that.
    pub q_values: &'a QValues,
    /// Decay evaluations from a second library, read only to replace the
    /// placeholder average decay energies in `decay` (see
    /// `endf::chain::Chain::fill_placeholder_decay_energies`). Empty for no
    /// fill, which leaves every number as `decay` gave it.
    pub decay_fill: &'a [Material],
    /// The library `decay_fill` came from, recorded per replaced nuclide.
    pub decay_fill_library: &'a str,
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
    let (decay, fpy, q_values) = (inputs.decay, inputs.fpy, inputs.q_values);
    let Provenance {
        library,
        decay_library,
        data_version,
        created_utc,
    } = provenance;
    let mut chain = Chain::from_endf(decay, fpy, q_values, reactions)?;
    if let Some(path) = branch_ratios {
        apply_branch_ratios(&mut chain, path)?;
    }
    let fill = if inputs.decay_fill.is_empty() {
        None
    } else {
        Some(chain.fill_placeholder_decay_energies(inputs.decay_fill, inputs.decay_fill_library)?)
    };
    let chain = chain;
    let mut decay_record = decay_energy_record(&chain, fill.as_ref());
    decay_record.insert(
        "decay_inconsistencies".to_string(),
        decay_inconsistency_record(decay),
    );
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
            (*subsection == "decay").then_some(&decay_record),
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

/// What the decay subsection's provenance says about its average energies.
///
/// The placeholder records are listed by name rather than counted, because
/// the question a reader asks is whether a nuclide carrying heat in their
/// inventory is one of them, and a count cannot answer it. The list is bounded
/// by the library (about a thousand names for ENDF/B-VIII.1), which a JSON
/// sidecar carries without trouble.
fn decay_energy_record(
    chain: &Chain,
    fill: Option<&endf::chain::DecayEnergyFill>,
) -> serde_json::Map<String, serde_json::Value> {
    let placeholders: Vec<&str> = chain
        .nuclides
        .iter()
        .filter(|n| n.decay_energy_source.as_deref() == Some(endf::chain::DECAY_ENERGY_PLACEHOLDER))
        .map(|n| n.name.as_str())
        .collect();
    let mut record = serde_json::Map::new();
    record.insert(
        "decay_energy_placeholders".to_string(),
        serde_json::json!({
            "rule": "MT=457 light and electromagnetic average energies equal (Q/3 each), no evaluated decay scheme",
            "count": placeholders.len(),
            "nuclides": placeholders,
        }),
    );
    if let Some(fill) = fill {
        record.insert(
            "decay_energy_fill".to_string(),
            serde_json::json!({
                "library": fill.library,
                "replaced": fill.replaced.iter().map(|(name, before, after)| {
                    serde_json::json!({"nuclide": name, "before_eV": before, "after_eV": after})
                }).collect::<Vec<_>>(),
                "half_life_mismatch": fill.half_life_mismatch,
                "unfilled": fill.unfilled,
            }),
        );
    }
    record
}

/// What the decay records say that cannot be right, for the decay
/// subsection's provenance.
///
/// One list per kind of [`endf::decay::DecayInconsistency`], every kind
/// present so a reader indexes without guessing, and the records listed by
/// name for the same reason the placeholders are: the question is whether one
/// of the heat carriers in an inventory is on it. Nothing is corrected; the
/// numbers written are the library's.
fn decay_inconsistency_record(decay: &[Material]) -> serde_json::Value {
    let mut by_kind: BTreeMap<&'static str, Vec<serde_json::Value>> = DecayInconsistency::LABELS
        .iter()
        .map(|&label| (label, Vec::new()))
        .collect();
    let mut count = 0;
    for material in decay {
        let Ok(d) = Decay::from_material(material) else {
            continue;
        };
        if d.nuclide.atomic_number == 0 {
            continue;
        }
        for finding in d.inconsistencies() {
            let entry = match &finding {
                DecayInconsistency::ZeroHalfLife => {
                    serde_json::json!({"nuclide": d.nuclide.name})
                }
                DecayInconsistency::BranchingRatioSum { sum } => {
                    serde_json::json!({"nuclide": d.nuclide.name, "sum": sum})
                }
                DecayInconsistency::IsomericTransitionEnergy { q, recoverable } => {
                    serde_json::json!({
                        "nuclide": d.nuclide.name,
                        "q_eV": q,
                        "recoverable_eV": recoverable,
                    })
                }
            };
            by_kind.entry(finding.label()).or_default().push(entry);
            count += 1;
        }
    }
    let mut record = serde_json::Map::new();
    record.insert("count".to_string(), serde_json::json!(count));
    for (kind, entries) in by_kind {
        record.insert(kind.to_string(), serde_json::Value::Array(entries));
    }
    serde_json::Value::Object(record)
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
///
/// `decay_fill_files` and `decay_fill_library` are the second decay library
/// that replaces placeholder average energies; both empty for no fill.
#[allow(clippy::too_many_arguments)]
pub fn convert_transmutation_files(
    decay_files: &[String],
    fpy_files: &[String],
    neutron_files: &[String],
    decay_fill_files: &[String],
    decay_fill_library: &str,
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
    if !decay_fill_files.is_empty() && decay_fill_library.is_empty() {
        return Err(
            "decay_fill_files were given without decay_fill_library; the \
                    replacements are recorded per nuclide under that name, so it \
                    cannot be left blank"
                .into(),
        );
    }
    let decay = read(decay_files)?;
    let fpy = read(fpy_files)?;
    let decay_fill = read(decay_fill_files)?;

    // Read and dropped one at a time, rather than collected like the others. A
    // chain wants nothing from a neutron evaluation but its channels' Q values,
    // and holding the parsed set to get them peaked at 39 GB over TENDL's 2848
    // files, which is more than an ordinary machine has: it was killed three
    // times on a 45 GB one.
    //
    // One map per file, in parallel, merged in file order afterwards. The files
    // are independent, and merging in order leaves the result identical to a
    // sequential read, including which of two evaluations of the same nuclide
    // wins. Only the evaluations in flight are held, so the peak is set by the
    // core count rather than by the size of the sublibrary, and parsing is what
    // the pass spends effectively all of its time on: 41 s of the 41.4 s a
    // TENDL-2017 reactions build took on one core.
    let read_q = |path: &String| -> Result<QValues, String> {
        let material = Material::from_file(path).map_err(|e| format!("{path}: {e}"))?;
        let mut out = QValues::new();
        collect_q_values(&material, &mut out);
        Ok(out)
    };
    #[cfg(not(target_arch = "wasm32"))]
    let per_file: Vec<QValues> = {
        use rayon::prelude::*;
        neutron_files
            .par_iter()
            .map(read_q)
            .collect::<Result<Vec<_>, _>>()?
    };
    #[cfg(target_arch = "wasm32")]
    let per_file: Vec<QValues> = neutron_files
        .iter()
        .map(read_q)
        .collect::<Result<Vec<_>, _>>()?;

    let mut q_values = QValues::new();
    for map in per_file {
        for (nuclide, channels) in map {
            q_values.entry(nuclide).or_default().extend(channels);
        }
    }

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
            q_values: &q_values,
            decay_fill: &decay_fill,
            decay_fill_library,
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
    let decay = read(decay_files)?;

    // Streamed, like the Q values above and for the same reason.
    // The branching pass gets away with holding its neutron set today only
    // because the driver scopes the call to the parents of a reactions
    // subsection, a few hundred rather than a few thousand evaluations, which
    // is a convention rather than a promise.
    //
    // In parallel, one evaluation at a time per worker, absorbed afterwards in
    // file order. Each evaluation's rows depend on nothing but that evaluation
    // and the isomer table, and absorbing in order leaves the rows, the flagged
    // levels and the partial-sum lines exactly as reading the files one at a
    // time left them.
    let extractor = branching::BranchingExtractor::new(&decay, tol_ev, linearize_tol);
    let extract = |path: &String| -> Result<branching::BranchingPartial, String> {
        let material = Material::from_file(path).map_err(|e| format!("{path}: {e}"))?;
        Ok(extractor.extract_one(&material))
    };
    #[cfg(not(target_arch = "wasm32"))]
    let partials: Vec<branching::BranchingPartial> = {
        use rayon::prelude::*;
        neutron_files
            .par_iter()
            .map(extract)
            .collect::<Result<Vec<_>, _>>()?
    };
    #[cfg(target_arch = "wasm32")]
    let partials: Vec<branching::BranchingPartial> = neutron_files
        .iter()
        .map(extract)
        .collect::<Result<Vec<_>, _>>()?;

    let mut extractor = extractor;
    for partial in partials {
        extractor.absorb(partial);
    }
    let branching::Extracted {
        rows,
        covariance,
        stats,
    } = extractor.finish();
    let dir = out.join("branching");
    branching::write_branching(&rows, &dir)?;
    branching::write_branching_covariance(&covariance, &dir)?;
    // What of MF=40 the covariance file does not show, beside the data rather
    // than only in the returned statistics: the parts that hold no block and
    // so have no row, the keys the converter left null with the reason, and
    // the targets it gave by excitation where the two files' LFS disagree.
    let mut mf40_gaps = serde_json::Map::new();
    for (key, lines) in [
        ("mf40_without_blocks", &stats.mf40_without_blocks),
        ("mf40_unmatched_states", &stats.mf40_unmatched_states),
        ("mf40_partner_unresolved", &stats.mf40_partner_unresolved),
        (
            "mf40_states_placed_by_excitation",
            &stats.mf40_states_placed_by_excitation,
        ),
    ] {
        mf40_gaps.insert(key.to_string(), serde_json::json!(lines));
    }
    write_provenance(
        &dir,
        "branching",
        &provenance.library,
        &provenance.decay_library,
        &provenance.data_version,
        &provenance.created_utc,
        Some(&mf40_gaps),
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
