//! The Arrow schema of the simulation-ready nuclear data format, declared once.
//!
//! Before this crate the format was declared in at least three places: the
//! retired Python converter's `schemas.py`, and `yani`'s `chain_arrow.rs`,
//! which writes the transmutation subsections itself. The readers named the
//! same columns a fourth time as string literals scattered through their
//! parsing code. Nothing connected them, and the failure mode is quiet: issue
//! #126 was writer and reader disagreeing about a name, and it reached
//! published data.
//!
//! Every consumer builds against these declarations, so a rename is a compile
//! error rather than a runtime surprise. There is no rendered copy any more:
//! the JSON manifest and the `emit-schema-manifest` binary that produced it
//! existed for the Python converter, which read the declarations without
//! being able to compile against them. Now that every writer and reader is
//! Rust, the declarations below are the only statement of the format.
//!
//! # Layout
//!
//! - [`neutron`] -- `{nuclide}.arrow/`
//! - [`photon`] -- `{element}.arrow/`
//! - [`transmutation`] -- `transmutation_{library}.arrow/<subsection>/`

use arrow_schema::{DataType, Field, Metadata, Schema};
use std::sync::Arc;

/// Where each (MT, temperature) lives inside a published `reactions.arrow`, so
/// a reader can fetch only the cross sections it needs. Part of the published
/// format, and shared by the converter that writes the index and the loader
/// that reads it.
pub mod energy_ranges;
pub mod reaction_ranges;

/// Every section, keyed by its path within the output directory.
///
/// The key is exactly the relative path the file is written to, so a consumer
/// can look up the schema it is about to read or write without a second map.
pub fn all_sections() -> Vec<(&'static str, Schema)> {
    vec![
        ("branching/branching.arrow", branching_branching()),
        (
            "branching/branching_covariance.arrow",
            branching_branching_covariance(),
        ),
        ("bremsstrahlung.arrow", bremsstrahlung()),
        ("compton.arrow", compton()),
        ("covariance.arrow", covariance()),
        ("decay/decay_modes.arrow", decay_decay_modes()),
        ("decay/nuclides.arrow", decay_nuclides()),
        ("decay/sources.arrow", decay_sources()),
        ("distributions.arrow", distributions()),
        ("element.arrow", element()),
        ("energy.arrow", energy()),
        ("fission_photon.arrow", fission_photon()),
        ("fission_yields/aliases.arrow", fission_yields_aliases()),
        (
            "fission_yields/evaluated_yields.arrow",
            fission_yields_evaluated_yields(),
        ),
        (
            "fission_yields/fission_yields.arrow",
            fission_yields_fission_yields(),
        ),
        ("nuclide.arrow", nuclide()),
        ("products.arrow", products()),
        ("reactions.arrow", reactions()),
        ("reactions/reactions.arrow", reactions_reactions()),
        ("subshells.arrow", subshells()),
        ("total_nu.arrow", total_nu()),
        ("urr.arrow", urr()),
    ]
}

/// Look up a section by its path, e.g. `"reactions.arrow"`.
pub fn section(path: &str) -> Option<Schema> {
    all_sections()
        .into_iter()
        .find(|(p, _)| *p == path)
        .map(|(_, s)| s)
}

/// The section key for a file inside a `{Nuclide}.arrow/` or `{Element}.arrow/`
/// directory, where the file name alone identifies the section.
///
/// Returns `None` for anything not declared as a flat section, including every
/// transmutation path: those live one directory down and reuse names the
/// neutron sections also use, so their readers pass the section explicitly
/// rather than having it guessed from the path. Guessing was tried and is
/// unsound, since a temporary directory can be named anything.
pub fn flat_section_for_path(path: &std::path::Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|p| p.to_str())?;
    // Transmutation subsection directories are never a flat neutron/photon
    // section, whatever the file is called.
    if matches!(
        parent,
        "decay" | "reactions" | "fission_yields" | "branching"
    ) {
        return None;
    }
    let declared = section(name).is_some() && !name.contains('/');
    declared.then(|| name.to_string())
}

/// Check a batch's columns against the declaration for `section`.
///
/// Catches the failure the per-file declarations exist to prevent: a reader
/// pointed at the wrong section, or at data written to a schema this build does
/// not know. Issue #126 was writer and reader disagreeing about a name, and it
/// reached published data before anyone noticed.
///
/// Deliberately lenient in one direction: a column the declaration has and the
/// batch does not is allowed, because several sections are written with
/// optional columns omitted and the readers already handle that per field. A
/// column the batch has and the declaration does not is an error, since it
/// means this is not the section it claims to be.
///
/// `Ok(())` for an unknown section, so a caller can validate opportunistically
/// without knowing whether a path is declared.
pub fn check_batch(section: &str, batch: &Schema) -> Result<(), String> {
    let Some(declared) = self::section(section) else {
        return Ok(());
    };
    let mut undeclared: Vec<&str> = batch
        .fields()
        .iter()
        .filter(|f| declared.field_with_name(f.name()).is_err())
        .map(|f| f.name().as_str())
        .collect();
    undeclared.sort();
    if !undeclared.is_empty() {
        return Err(format!(
            "{section} has columns the schema does not declare: {undeclared:?}. \
             Either this is not the section it claims to be, or it was written \
             by a build with a different schema."
        ));
    }
    for field in batch.fields() {
        if let Ok(want) = declared.field_with_name(field.name()) {
            if want.data_type() != field.data_type() {
                return Err(format!(
                    "{section} column {:?} is {:?}, the schema declares {:?}",
                    field.name(),
                    field.data_type(),
                    want.data_type()
                ));
            }
        }
    }
    Ok(())
}

// One constructor per Arrow type this format uses, so a field declaration is a
// single readable line. Spelled out longhand, the 121 list fields alone carried
// about 100 characters each of DataType::List(Arc::new(Field::new("item", ..)))
// and buried the names they exist to declare.

fn utf8(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Utf8, nullable)
}

fn f64(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Float64, nullable)
}

fn i32(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Int32, nullable)
}

fn boolean(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Boolean, nullable)
}

/// Arrow's list item field is always named `item` and nullable here, matching
/// what pyarrow's `list_()` produces on the writer side.
fn list_of(inner: DataType) -> DataType {
    DataType::List(Arc::new(Field::new("item", inner, true)))
}

fn f64s(name: &str, nullable: bool) -> Field {
    Field::new(name, list_of(DataType::Float64), nullable)
}

fn i32s(name: &str, nullable: bool) -> Field {
    Field::new(name, list_of(DataType::Int32), nullable)
}

fn utf8s(name: &str, nullable: bool) -> Field {
    Field::new(name, list_of(DataType::Utf8), nullable)
}

fn f64ss(name: &str, nullable: bool) -> Field {
    Field::new(name, list_of(list_of(DataType::Float64)), nullable)
}

/// Schema metadata, which iterates in sorted key order.
///
/// Arrow writes schema metadata in the map's iteration order and does not sort
/// it (`arrow-ipc`, `metadata_to_fb`), so the order here is the order on disk.
/// That matters because a published library has to be checksummable: when the
/// order was a `HashMap`'s, the same conversion run twice wrote
/// `filetype,version` or `version,filetype` at random, and a converted
/// `element.arrow` came out as one of two byte patterns differing in 114 bytes,
/// all inside the two schema blocks. A rebuild then looked like it had changed
/// every file (issue #441).
///
/// Since arrow 60, `arrow_schema::Schema::metadata` is a `Metadata` newtype over
/// a `BTreeMap`, which is ordered by construction, so collecting into it is
/// enough (issue #28). Before that the field was a concrete
/// `HashMap<String, String>`, which could be neither swapped for a `BTreeMap`
/// nor given a fixed hasher, and this function held a build-and-check loop that
/// rebuilt the map until a fresh hasher seed happened to yield sorted order.
///
/// The determinism tests that pinned that loop are kept as they are: they assert
/// the property this format needs, not the mechanism that delivers it, so they
/// go on being the thing that would catch a regression here.
fn meta<const N: usize>(pairs: [(&str, &str); N]) -> Metadata {
    pairs
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// `branching/branching.arrow`
pub fn branching_branching() -> Schema {
    Schema::new(vec![
        utf8("nuclide", false),
        utf8("reaction", false),
        utf8("target", false),
        utf8("quantity", false),
        f64s("energy", false),
        f64s("values", false),
    ])
    .with_metadata(meta([
        ("filetype", "transmutation-branching"),
        ("version", "2.0"),
    ]))
}

/// `branching/branching_covariance.arrow`
///
/// MF=40, the covariance of the radionuclide production cross sections that
/// `branching.arrow` carries as MF=10 partials: the stated uncertainty of an
/// isomeric split. Optional, like `covariance.arrow` and for the same reasons:
/// a library without MF=40 writes no such file, a branching directory published
/// before it has none, and a reader treats absence as "no covariance", never as
/// an error.
///
/// Every covariance block of every MF=40 section of every evaluation, not a
/// selection. One row per block, which is one NC or NI block of one
/// sub-subsection of one product state. Every tape value is written as the tape
/// gives it: nothing is normalised, and a state the converter could not place
/// in the chain is still written, with a null `target`. A section with no
/// product state (NS 0), a product state with no sub-subsection (NL 0), or a
/// sub-subsection with no block (NC and NI 0), holds no covariance and so has
/// no row, and the counts NS, NL, NC and NI are not columns, being the row
/// counts. The converter lists each such part, with its tape values, under
/// `mf40_without_blocks` in `branching/provenance.json` instead. None of the
/// published libraries has one.
///
/// The first columns are the converter's key, the rest are the tape's:
///
/// * `nuclide` is the parent, named as `branching.arrow` names it.
///   `reaction` is the chain kind the section's MT maps to, null for an MT
///   with none (TENDL writes MF=40 for MT 18 with IZAP 0).
/// * `target` is the chain nuclide the product state (`izap`, `lfs`) is the
///   partial of, which is the `target` of the `branching.arrow` row for
///   (`nuclide`, `reaction`). Null when no MF=9 or MF=10 state of that MT
///   matched it, when several states of one chain nuclide sit within the
///   converter's tolerance of its excitation, since taking the nearest would
///   guess which partial weights the covariance, or when the state is excited
///   (`lfs` above 0) with `qm - qi` not positive, since then nothing on the
///   tape confirms which MF=9 or MF=10 level its `lfs` names; each such state
///   is listed under `mf40_unmatched_states` in `branching/provenance.json`.
///   A ground state (`lfs` 0) is taken at zero excitation whatever its
///   `qm - qi`, as MF=9 and MF=10 grounds are. A match by excitation never puts an excited state on a ground partial,
///   or the reverse, and one made because MF=9 and MF=10 give the state's
///   (`izap`, `lfs`) no state or one at another excitation is listed under
///   `mf40_states_placed_by_excitation` there. `target1` is
///   the same for the partner state the block correlates this one with, level
///   `xlfs1` of reaction `mt1`: the match of the one state of `mt1`'s MF=40
///   section at LFS `xlfs1`, as that state's own row's `target` is. Null when
///   the partner is in another material, `xmf1` is not 10, `mt1` is 0 (which
///   the manual gives no meaning in MF=40), or the tape does not pin the
///   partner to one state. MF=40 gives no IZAP for it, so a level that two
///   MF=40 states of `mt1` share, or an MF=9 or MF=10 state of another product,
///   is left null rather than read as this row's own product. So is an `mt1`
///   with no MF=40 section, or no MF=40 state at `xlfs1`, whose level XLFS1
///   would have to be read in MF=10's
///   numbering with no excitation to check it by. The manual numbers XLFS1 as
///   MF=10 does and MF=40 need not (ENDF/B-VIII.1 Pb204 MT 4, LFS 21 in MF=10
///   and 1 in MF=40), so `target1` is also null where MF=10 of `mt1` gives the
///   partner's product no state at LFS `xlfs1`, or one at another excitation
///   than the MF=40 state there. Each such null is listed under
///   `mf40_partner_unresolved` in `branching/provenance.json`. A state's block
///   with itself (`mt1` and `xlfs1` its own `mt` and `lfs`) names the state by
///   its own subsection's label, so its `target1` is its `target` unless a
///   second MF=40 state of `mt` carries that label; where that label is not
///   MF=10's number for the level, the state is one of those listed under
///   `mf40_states_placed_by_excitation`. Several levels can
///   resolve to one chain nuclide: JEFF-4.0 U235 MT 4 correlates its ground
///   (LFS 0) with its 77 eV isomer (XLFS1 1), both U235. So a consumer keys a
///   block on (`mt`, `lfs`, `mt1`, `xlfs1`), never on (`target`, `target1`).
/// * `energy` and `values` are this state's own curve, linearized by the
///   converter exactly as `branching.arrow` has it (the tape's own points
///   when every region is lin-lin), and are written only when several states
///   resolved to one target in one quantity, which `branching.arrow` then
///   carries as their sum. `quantity` says which of the target's curves is
///   this state's, as `branching.arrow`'s column does: `"cross_section"` for
///   an MF=10 partial, `"yield"` for an MF=9 yield, whose partial is that
///   yield times the MF=3 cross section of the MT. It is written for every
///   placed state (`target` set), merged or not, since one target can have
///   both a yield row and a cross-section row. A relative covariance of one
///   state has to be weighted by that state's own partial to fold exactly.
///   `energy` and `values` null with `quantity` set means the
///   `branching.arrow` row for (`nuclide`, `reaction`, `target`, `quantity`)
///   is that state's own curve.
/// * `mat` is the evaluation's MAT, so a reader can tell a `mat1` naming the
///   evaluation itself (JEFF-4.0 U235 MT 4 writes its own 9228 there) from a
///   correlation with another material.
/// * `za`, `awr` and `lis` are the section HEAD. `state_idx` is the product
///   state's position in the section and `qm`, `qi`, `izap` and `lfs` its CONT,
///   verbatim: JEFF-4.0 U235 MT 4 writes IZAP 0 for the target itself, and
///   ENDF/B-VIII.1 Pb204 MT 4 numbers its isomer LFS 1 here and 21 in MF=10.
///
/// Every column after `lfs` is [`covariance`]'s, verbatim and in order, so one
/// writer and one reader serve both files. `subsection_idx` there is the
/// sub-subsection's position within its product state, and `mtl` is always
/// null, since MF=40 has no lumped-reaction flag.
pub fn branching_branching_covariance() -> Schema {
    let mut fields = vec![
        utf8("nuclide", false),
        utf8("reaction", true),
        utf8("target", true),
        utf8("target1", true),
        f64s("energy", true),
        f64s("values", true),
        utf8("quantity", true),
        i32("mat", false),
        i32("za", false),
        f64("awr", false),
        i32("lis", false),
        i32("state_idx", false),
        f64("qm", false),
        f64("qi", false),
        i32("izap", false),
        i32("lfs", false),
    ];
    fields.extend(covariance().fields().iter().map(|f| f.as_ref().clone()));
    Schema::new(fields).with_metadata(meta([
        ("filetype", "transmutation-branching_covariance"),
        ("version", "2.0"),
    ]))
}

/// `bremsstrahlung.arrow`
pub fn bremsstrahlung() -> Schema {
    Schema::new(vec![
        f64("I", true),
        f64s("electron_energy", true),
        f64s("photon_energy", true),
        f64s("num_electrons", true),
        f64s("ionization_energy", true),
        f64s("dcs_data", true),
        i32s("dcs_shape", true),
    ])
}

/// `compton.arrow`
pub fn compton() -> Schema {
    Schema::new(vec![
        f64s("num_electrons", true),
        f64s("binding_energy", true),
        f64s("pz", true),
        f64s("J_data", true),
        i32s("J_shape", true),
        f64s("J_cdf_data", true),
        i32s("J_cdf_shape", true),
        i32s("subshell_map_offsets", true),
        i32s("subshell_map_indices", true),
        f64s("subshell_map_weights", true),
    ])
}

/// `covariance.arrow`
///
/// MF=33: the covariance of this nuclide's cross sections. Optional, and the
/// only section that is: the matrices are large, so a consumer that does not
/// want uncertainty should download nothing extra, and every already-published
/// `{nuclide}.arrow/` has no such file and must keep loading unchanged. Opt-in
/// is therefore by file presence, and a reader treats absence as "no
/// covariance", never as an error (issue #514).
///
/// One row per covariance block, which is one NC or NI sub-subsection of one
/// MF=33 subsection. That is the tape's own granularity, and it is what makes
/// the table both faithful and small:
///
/// - **Faithful.** No group structure is imposed. Each block keeps the
///   evaluation's own energy grid in `ek`, so `ne` varies from block to block
///   and no common grid has to exist. `fkk` stays in the format's packed order
///   with its own `ls` beside it, exactly as the parser leaves it, so the
///   writer never reshapes a matrix and cannot bake in a symmetry that the
///   evaluation did not state. `ls=1` is an upper triangle whose transpose is
///   implied; `ls=0` is a full asymmetric block, and folding one to a triangle
///   would lose numbers.
/// - **Sparse for free.** A block-keyed table IS the sparse form. Most (MT,
///   MT1) pairs have no cross terms, and those pairs simply have no row --
///   with no threshold, no dropped small values and nothing lossy about it. A
///   single dense matrix per nuclide would need the common grid that the
///   faithful half rules out anyway.
///
/// Within a subsection the NC blocks come first and the NI blocks after, with
/// `block_idx` running across both rather than restarting. Numbering the two
/// lists separately would give an NC block and an NI block in one subsection
/// the same `(mt, subsection_idx, block_idx)`, and that triple is the key.
///
/// `mt` is the reaction the section belongs to and `mt1`/`mat1` the reaction it
/// is correlated with, so the diagonal blocks are the rows with `mat1 == 0 &&
/// (mt1 == 0 || mt1 == mt)`. Those are the only rows for which the matrix is
/// symmetric in itself: an off-diagonal block's transpose is the (`mt1`, `mt`)
/// block, not the block itself.
///
/// The `kind` discriminator selects which columns are populated, the way
/// `distributions.arrow` uses `type`: `"ni"` for a covariance given explicitly,
/// where `lb` selects the layout again within it, and `"nc"` for one derived
/// from other reactions. Everything not belonging to a row's variant is null.
pub fn covariance() -> Schema {
    Schema::new(vec![
        // Which pair of reactions this block belongs to. Rows are written in
        // tape order; the two indices carry that order explicitly, since two
        // subsections of one section may name the same (MAT1, MT1).
        i32("mt", false),
        i32("subsection_idx", false),
        i32("block_idx", false),
        utf8("kind", false),
        i32("mat1", true),
        i32("mt1", true),
        f64("xmf1", true),
        f64("xlfs1", true),
        // MTL from the section HEAD: the reaction this one is lumped into, 0
        // when it is not lumped. Per section rather than per block, so it
        // repeats across a section's rows.
        i32("mtl", true),
        // kind = "ni". `lb` selects which of the rest are populated: 0-4 use
        // `lt`, `np` and both (E, F) tables; 5 uses `ls`, `ne`, `ek` and `fkk`;
        // 6 uses `ner`, `nec`, `er`, `ec` and `fkl`; 8 and 9 use `lt`, `np` and
        // the first table only.
        i32("lb", true),
        i32("ls", true),
        i32("lt", true),
        i32("nt", true),
        i32("np", true),
        i32("ne", true),
        i32("ner", true),
        i32("nec", true),
        f64s("ek", true),
        f64s("fk", true),
        f64s("el", true),
        f64s("fl", true),
        f64s("fkk", true),
        f64s("er", true),
        f64s("ec", true),
        f64s("fkl", true),
        // kind = "nc". `lty` selects the rest: 0 uses `nci`, `ci` and `xmti`,
        // anything else uses `mats`, `mts`, `nei`, `xmfs`, `xlfss`, `ei` and
        // `wei`.
        i32("lty", true),
        f64("e1", true),
        f64("e2", true),
        i32("nci", true),
        f64s("ci", true),
        f64s("xmti", true),
        i32("mats", true),
        i32("mts", true),
        i32("nei", true),
        f64("xmfs", true),
        f64("xlfss", true),
        f64s("ei", true),
        f64s("wei", true),
    ])
}

/// `decay/decay_modes.arrow`
pub fn decay_decay_modes() -> Schema {
    Schema::new(vec![
        utf8("nuclide", false),
        utf8("type", false),
        utf8("target", true),
        f64("branching_ratio", false),
        // The evaluation's dBR, as MT=457 writes it. Nullable and last, so a
        // file written without it still reads. A 0.0 is the format's "not
        // stated" and is stored as 0.0; readers take null and 0.0 alike as
        // not stated, never as an exact ratio (issue #140). It is the dBR on
        // the tape's BR, and `branching_ratio` in the same row may carry the
        // normalisation residual on the parent's largest mode.
        f64("branching_ratio_uncertainty", true),
    ])
}

/// `decay/nuclides.arrow`
pub fn decay_nuclides() -> Schema {
    Schema::new(vec![
        utf8("name", false),
        f64("half_life", true),
        f64("decay_energy", false),
        // Nullable and last, so a file written without it still reads.
        // Each sigma is stored as the evaluation writes it, and MT=457 writes
        // 0.0 for "not stated". Readers take null (no value, or a file that
        // predates the column) and 0.0 alike as not stated, which is not the
        // same as stated to be zero (issue #515).
        f64("half_life_uncertainty", true),
        f64("decay_energy_uncertainty", true),
        // The decay energy split into its recoverable-heat components (ENDF
        // MT=457 light particle, electromagnetic, heavy particle) with each
        // one's sigma. Nullable and last for the same reason: a file written
        // before them still reads, with no split. A null energy is a
        // component not given, not one given as zero; the sigmas follow the
        // rule above (issue #140).
        f64("decay_energy_beta", true),
        f64("decay_energy_beta_uncertainty", true),
        f64("decay_energy_gamma", true),
        f64("decay_energy_gamma_uncertainty", true),
        f64("decay_energy_alpha", true),
        f64("decay_energy_alpha_uncertainty", true),
    ])
    .with_metadata(meta([
        ("filetype", "transmutation-decay"),
        ("version", "2.0"),
    ]))
}

/// `decay/sources.arrow`
pub fn decay_sources() -> Schema {
    Schema::new(vec![
        utf8("nuclide", false),
        utf8("particle", false),
        utf8("type", false),
        f64s("energies", false),
        f64s("intensities", false),
    ])
}

/// `distributions.arrow`
pub fn distributions() -> Schema {
    Schema::new(vec![
        i32("reaction_mt", true),
        i32("product_idx", true),
        i32("dist_idx", true),
        utf8("type", true),
        f64s("applicability_data", true),
        i32s("applicability_shape", true),
        i32s("applicability_breakpoints", true),
        i32s("applicability_interpolation", true),
        f64s("angle_energies", true),
        f64s("angle_mu_data", true),
        i32s("angle_mu_offsets", true),
        i32s("angle_mu_interpolation", true),
        utf8("energy_dist_type", true),
        f64s("energy_dist_energies", true),
        i32s("energy_dist_interpolation", true),
        f64s("energy_dist_data", true),
        i32s("energy_dist_offsets", true),
        i32s("energy_dist_out_interp", true),
        i32s("energy_dist_n_discrete", true),
        f64s("energy_param_x", true),
        f64s("energy_param_y", true),
        f64s("energy_param2_x", true),
        f64s("energy_param2_y", true),
        f64("energy_restriction_u", true),
        f64("energy_threshold", true),
        f64("energy_mass_ratio", true),
        i32("energy_primary_flag", true),
        f64("energy_atomic_weight_ratio", true),
        f64("energy_discrete_energy", true),
        f64s("corr_energies", true),
        i32s("corr_breakpoints", true),
        i32s("corr_interpolation", true),
        f64s("corr_eout_data", true),
        i32s("corr_eout_offsets", true),
        i32s("corr_eout_interp", true),
        i32s("corr_eout_n_discrete", true),
        f64s("corr_mu_data", true),
        i32s("corr_mu_offsets", true),
        i32s("corr_mu_interp", true),
        f64s("km_energies", true),
        i32s("km_breakpoints", true),
        i32s("km_interpolation", true),
        f64s("km_data", true),
        i32s("km_offsets", true),
        i32s("km_interp", true),
        i32s("km_n_discrete", true),
        i32("nbody_n", true),
        f64("nbody_total_mass", true),
        f64("nbody_atomic_weight_ratio", true),
        f64("nbody_q_value", true),
    ])
}

/// `element.arrow`
pub fn element() -> Schema {
    Schema::new(vec![
        utf8("name", true),
        i32("Z", true),
        f64s("ln_energy", true),
        f64s("coherent_xs", true),
        f64s("incoherent_xs", true),
        f64s("photoelectric_xs", true),
        f64s("pair_production_nuclear_xs", true),
        f64s("pair_production_electron_xs", true),
        f64s("coherent_int_ff_x", true),
        f64s("coherent_int_ff_y", true),
        f64s("coherent_ff_x", true),
        f64s("coherent_ff_y", true),
        f64s("coherent_anomalous_real_x", true),
        f64s("coherent_anomalous_real_y", true),
        f64s("coherent_anomalous_imag_x", true),
        f64s("coherent_anomalous_imag_y", true),
        f64s("incoherent_ff_x", true),
        f64s("incoherent_ff_y", true),
    ])
    .with_metadata(meta([("filetype", "data_photon"), ("version", "5.0")]))
}

/// `fission_photon.arrow`
pub fn fission_photon() -> Schema {
    Schema::new(vec![
        utf8("role", true),
        utf8("kind", true),
        f64s("coefficients", true),
        f64s("x", true),
        f64s("y", true),
        i32s("interpolation", true),
        i32s("breakpoints", true),
    ])
}

/// `fission_yields/aliases.arrow`
pub fn fission_yields_aliases() -> Schema {
    Schema::new(vec![
        utf8("nuclide", false),
        utf8("fission_yield_parent", false),
    ])
}

/// `fission_yields/evaluated_yields.arrow`
///
/// The yield evaluations exactly as the tapes give them, one row per
/// (fissioning nuclide, incident energy, kind). `kind` is `"independent"`
/// (MT=454) or `"cumulative"` (MT=459). `products` are named from the tape's
/// ZAFP and FPS and kept in tape order, including products the decay library
/// has no data for; `yields` and `yield_uncertainties` are its Y and DY
/// verbatim, so an evaluator's 0.0 is 0.0 here. A reader treats a null or 0.0
/// DY as not stated, never as an exact yield. `interpolation` is the ENDF law
/// from the next lower energy to this one, null at the lowest energy, where the
/// tape puts LE in that field instead.
///
/// A separate file from [`fission_yields_fission_yields`] because that one is
/// derived and this is not: there a product the decay library lacks is mapped
/// onto a stand-in, products landing on the same name are summed, and every
/// energy is padded to one product list. A DY cannot sit beside such a sum
/// without the correlation of its parts, which no evaluation publishes. Being
/// a file of its own also means a consumer that never asks for it (the
/// browser) is not sent it, and an older build never reads it.
///
/// There is no correlation column because no evaluation publishes yield
/// correlations (ENDF/B-VIII.1, JEFF-4.0 and JENDL-5.0 carry only MF=8
/// MT=454/459). One would be appended when an evaluation has one.
pub fn fission_yields_evaluated_yields() -> Schema {
    Schema::new(vec![
        utf8("nuclide", false),
        f64("energy", false),
        utf8("kind", false),
        i32("interpolation", true),
        utf8s("products", false),
        f64s("yields", false),
        f64s("yield_uncertainties", true),
    ])
    .with_metadata(meta([
        ("filetype", "transmutation-fission_yields_evaluated"),
        ("version", "2.0"),
    ]))
}

/// `fission_yields/fission_yields.arrow`
pub fn fission_yields_fission_yields() -> Schema {
    Schema::new(vec![
        utf8("nuclide", false),
        f64("energy", false),
        utf8s("products", false),
        f64s("yields", false),
    ])
    .with_metadata(meta([
        ("filetype", "transmutation-fission_yields"),
        ("version", "2.0"),
    ]))
}

/// `nuclide.arrow`
///
/// Metadata only since the union energy grids moved to [`energy()`]
/// (fusion-neutronics/core#100). One row, on the order of a kilobyte, so a
/// client that wants to know what a nuclide IS no longer pays 6.33 MB of U238
/// grids to find out.
pub fn nuclide() -> Schema {
    Schema::new(vec![
        utf8("name", true),
        i32("Z", true),
        i32("A", true),
        f64("atomic_weight_ratio", true),
        utf8s("temperatures", true),
        f64s("kTs", true),
    ])
    .with_metadata(meta([("filetype", "data_neutron"), ("version", "5.0")]))
}

/// `energy.arrow`
///
/// The union energy grids, one row and one record batch per temperature, so a
/// client that plots at one temperature range-fetches one of them instead of
/// reading all six out of `nuclide.arrow`.
///
/// `temperature` is the label as the file spells it, `"294K"`. The rows are not
/// only the nuclide's `temperatures`: the NJOY route carries an extra 0 K grid
/// off the PENDF tape, which is published here under its own label and which no
/// temperature filter can name.
pub fn energy() -> Schema {
    Schema::new(vec![utf8("temperature", true), f64s("energy_values", true)]).with_metadata(meta([
        ("filetype", "data_neutron_energy"),
        ("version", "1.0"),
    ]))
}

/// `products.arrow`
pub fn products() -> Schema {
    Schema::new(vec![
        i32("reaction_mt", true),
        i32("product_idx", true),
        utf8("particle", true),
        utf8("emission_mode", true),
        f64("decay_rate", true),
        i32("n_distribution", true),
        utf8("yield_type", true),
        f64s("yield_data", true),
        i32s("yield_shape", true),
        i32s("yield_breakpoints", true),
        i32s("yield_interpolation", true),
    ])
}

/// `reactions.arrow`
pub fn reactions() -> Schema {
    Schema::new(vec![
        i32("mt", true),
        utf8("label", true),
        f64("Q_value", true),
        boolean("center_of_mass", true),
        boolean("redundant", true),
        utf8s("xs_temperatures", true),
        f64ss("xs_values", true),
        i32s("xs_threshold_idx", true),
    ])
}

/// `reactions/reactions.arrow`
pub fn reactions_reactions() -> Schema {
    Schema::new(vec![
        utf8("nuclide", false),
        utf8("type", false),
        utf8("target", true),
        f64("Q", false),
        f64("branching_ratio", false),
    ])
    .with_metadata(meta([
        ("filetype", "transmutation-reactions"),
        ("version", "2.0"),
    ]))
}

/// `subshells.arrow`
pub fn subshells() -> Schema {
    Schema::new(vec![
        utf8("designator", true),
        f64("binding_energy", true),
        f64("num_electrons", true),
        f64s("xs", true),
        f64s("ln_xs", true),
        i32("threshold_idx", true),
        f64s("transitions_data", true),
        i32s("transitions_shape", true),
    ])
}

/// `total_nu.arrow`
pub fn total_nu() -> Schema {
    Schema::new(vec![
        utf8("particle", true),
        utf8("emission_mode", true),
        f64("decay_rate", true),
        utf8("yield_type", true),
        f64s("yield_data", true),
        i32s("yield_shape", true),
        i32s("yield_breakpoints", true),
        i32s("yield_interpolation", true),
    ])
}

/// `urr.arrow`
pub fn urr() -> Schema {
    Schema::new(vec![
        utf8("temperature", true),
        f64s("energy", true),
        f64s("table_data", true),
        i32s("table_shape", true),
        i32("interpolation", true),
        i32("inelastic", true),
        i32("absorption", true),
        boolean("multiply_smooth", true),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_section_has_fields_and_a_unique_path() {
        let sections = all_sections();
        assert_eq!(
            sections.len(),
            22,
            "section count changed; update the manifest"
        );
        let mut paths: Vec<&str> = sections.iter().map(|(p, _)| *p).collect();
        paths.sort();
        let before = paths.len();
        paths.dedup();
        assert_eq!(paths.len(), before, "duplicate section path");
        for (path, schema) in &sections {
            assert!(!schema.fields().is_empty(), "{path} declares no fields");
        }
    }

    #[test]
    fn lookup_matches_the_listing() {
        for (path, schema) in all_sections() {
            let looked_up = section(path).unwrap_or_else(|| panic!("section({path}) is None"));
            assert_eq!(looked_up.fields(), schema.fields(), "{path}");
        }
        assert!(section("no/such.arrow").is_none());
    }
}

#[cfg(test)]
mod determinism {
    use super::*;

    /// Every declared section's metadata must iterate in sorted key order.
    ///
    /// This is what makes a written file byte-reproducible: arrow serialises
    /// the map in iteration order, so an unsorted one produces a different
    /// file from the same data (issue #441).
    #[test]
    fn every_section_metadata_iterates_sorted() {
        let mut checked = 0;
        for (path, schema) in all_sections() {
            let keys: Vec<&String> = schema.metadata().keys().collect();
            assert!(
                keys.windows(2).all(|w| w[0] <= w[1]),
                "{path}: schema metadata iterates as {keys:?}, which is not sorted, \
                 so two conversions of the same input would write different bytes"
            );
            if !keys.is_empty() {
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "no section carries metadata, so this test proved nothing"
        );
    }

    /// Built repeatedly, the order must not change.
    ///
    /// The check above could pass by luck on a single build. This one is what
    /// catches a return to a per-instance-seeded map: `HashMap` reseeds, so a
    /// rebuild would eventually come out the other way round and this would
    /// fail. Kept after arrow 60 made `Metadata` a `BTreeMap` (issue #28),
    /// because it asserts the property the published format needs rather than
    /// whichever mechanism currently provides it.
    #[test]
    fn metadata_order_is_stable_across_rebuilds() {
        let first: Vec<String> = section("nuclide.arrow")
            .expect("nuclide.arrow is declared")
            .metadata()
            .keys()
            .cloned()
            .collect();
        assert!(!first.is_empty(), "nuclide.arrow carries metadata");
        for _ in 0..200 {
            let again: Vec<String> = section("nuclide.arrow")
                .expect("nuclide.arrow is declared")
                .metadata()
                .keys()
                .cloned()
                .collect();
            assert_eq!(
                first, again,
                "the metadata order changed between two builds of the same schema"
            );
        }
    }
}
