//! Shared plumbing for the value-comparison tests, and nothing else.
//!
//! Every helper here decompresses a fixture, parses it, opens a written
//! section or reads one cell out of it. None of them assert anything, so a
//! failure always reports from a named test rather than from a helper three
//! frames down.
//!
//! # Why these tests exist
//!
//! `sections.rs` is thorough about STRUCTURE: which sections exist, which MTs
//! are present, which labels resolve. What it does not do is compare the
//! NUMBERS against the evaluation they were converted from, and it says so
//! outright: `reactions_section_matches_the_published_file` notes that "the
//! cross sections cannot be compared value for value". A retired Python
//! package used to make that comparison. These files make it in Rust, against
//! `crates/endf` rather than against a second implementation of the format.
//!
//! # Hermetic, by construction
//!
//! Every fixture is `include_bytes!` out of `crates/endf/fixtures/`, so these
//! run on a clean checkout with no NJOY, no nuclear-data cache and no network.
//! That is the point: the strongest tests in `sections.rs` need an ENDF tree
//! and an njoy binary that no CI job provides, so they report green by not
//! running. Nothing here can do that. A test that cannot reach its writer with
//! vendored bytes is not written at all, and is recorded in the pull request
//! instead of added as a silent skip.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use arrow_array::{cast::AsArray, Array, RecordBatch};
use endf::{IncidentNeutron, IncidentPhoton, Material};

// ---------------------------------------------------------------------------
// The vendored fixtures.
// ---------------------------------------------------------------------------

/// The Li6 ACE table. One temperature, a real 721-point grid, 18 parsed MTs.
/// The only vendored input that reaches `write_fast_xs`.
pub const LI6_ACE: &[u8] = include_bytes!("../../../endf/fixtures/Li6.ace.xz");
/// The only vendored source of a non-empty `IncidentNeutron::urr`.
pub const URR_ACE: &[u8] = include_bytes!("../../../endf/fixtures/synthetic-urr.ace.xz");
/// Denormal cross sections, for the writer's refusal paths.
pub const DENORMAL_ACE: &[u8] = include_bytes!("../../../endf/fixtures/synthetic-denormal.ace.xz");
/// The seven ACE energy laws, as blocks. This table has no MTR/SIG/ESZ, so
/// `IncidentNeutron::from_ace` cannot read it; the laws are decoded per DLW
/// locator and wrapped in a scaffold.
pub const LAWS_ACE: &[u8] = include_bytes!("../../../endf/fixtures/synthetic-laws.ace.xz");

/// Li6 as ENDF. Reaches the 0 K branch of `write_nuclide` and carries MF=33.
pub const LI6_ENDF: &[u8] = include_bytes!("../../../endf/fixtures/n-003_Li_006_trimmed.endf.xz");
/// Fe56 as ENDF. Carries MF=33 MT=103.
pub const FE56_ENDF: &[u8] = include_bytes!("../../../endf/fixtures/n-026_Fe_056_trimmed.endf.xz");
/// U235 as ENDF. MF=1 MT=452, 455, 456 and 458, so it reaches both
/// `write_total_nu` and `write_fission_photon`.
pub const U235_ENDF: &[u8] = include_bytes!("../../../endf/fixtures/n-092_U_235_trimmed.endf.xz");
/// Am244 as ENDF. MT=458 with LFC=0 and NPLY=2, so both energy-release terms
/// are polynomial where U235's prompt term is tabulated.
pub const AM244_ENDF: &[u8] = include_bytes!("../../../endf/fixtures/n-095_Am_244.endf.xz");

/// Hydrogen photoatomic, and the atomic relaxation that goes with it. The
/// photon route runs on these with no NJOY at all.
pub const H_PHOTOAT: &[u8] = include_bytes!("../../../endf/fixtures/photoat-001_H_000.endf.xz");
pub const H_ATOM: &[u8] = include_bytes!("../../../endf/fixtures/atom-001_H_000.endf.xz");

// ---------------------------------------------------------------------------
// Fixtures on disk, and parsed.
// ---------------------------------------------------------------------------

/// A compressed fixture as text.
pub fn text(compressed: &[u8]) -> String {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    String::from_utf8(out).expect("fixture is UTF-8")
}

/// A compressed fixture written out under `dir`, for the parsers that take a
/// path rather than a string.
pub fn write_fixture(compressed: &[u8], dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, text(compressed)).expect("fixture is written");
    path
}

/// A scratch directory that removes itself even when an assertion panics,
/// which the hand-rolled `temp_dir` join in `sections.rs` does not.
pub fn scratch() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

/// An ACE table parsed into the value the writers are handed.
pub fn ace_nuclide(compressed: &[u8]) -> IncidentNeutron {
    let tables = endf::ace::tables_from_str(&text(compressed), None).expect("ACE parses");
    IncidentNeutron::from_ace(&tables[0], endf::ace::MetastableScheme::Mcnp).expect("ACE reads")
}

/// The first ACE table itself, for the checks that go behind the parser to the
/// raw XSS array.
pub fn ace_table(compressed: &[u8]) -> endf::Table {
    let tables = endf::ace::tables_from_str(&text(compressed), None).expect("ACE parses");
    tables.into_iter().next().expect("one table")
}

pub fn li6_ace() -> IncidentNeutron {
    ace_nuclide(LI6_ACE)
}

/// The first material in a compressed ENDF fixture.
pub fn endf_material(compressed: &[u8]) -> Material {
    let materials = endf::materials_from_str(&text(compressed)).expect("ENDF parses");
    materials.into_iter().next().expect("one material")
}

/// An ENDF evaluation parsed into the value the writers are handed.
///
/// Note what this route does NOT carry: `IncidentNeutron::from_endf` sets no
/// `k_ts`, no `atomic_weight_ratio` and leaves `energy` empty, keying every
/// cross section at `"0K"`. That is not a defect in the fixture, it is what
/// the ENDF route is before NJOY runs, and it is the only hermetic way to
/// reach those branches of the writers.
pub fn endf_nuclide(compressed: &[u8]) -> IncidentNeutron {
    IncidentNeutron::from_endf(&endf_material(compressed)).expect("ENDF reads")
}

/// The hydrogen photoatomic evaluation, with its relaxation data attached.
///
/// `tabulations` decides whether `compton_profiles` and `bremsstrahlung` come
/// back `Some`. No evaluation carries either; they are read from the three
/// text files that ship in the wheel, and without them `write_compton` and
/// `write_bremsstrahlung` return early and write no file at all.
pub fn h_photon(tabulations: bool) -> IncidentPhoton {
    let photoatomic = endf_material(H_PHOTOAT);
    let relaxation = endf_material(H_ATOM);
    let mut data = IncidentPhoton::from_endf(&photoatomic, Some(&relaxation))
        .expect("the photoatomic evaluation reads");
    if tabulations {
        data.add_photon_data(&photon_tabulations());
    }
    data
}

/// The three auxiliary tabulations, parsed.
pub fn photon_tabulations() -> endf::PhotonData {
    let d = photon_tabulation_dir();
    endf::PhotonData::from_files(
        d.join("compton_profiles_biggs1975.txt"),
        d.join("density_effect_sternheimer1982.txt"),
        d.join("bremsstrahlung_seltzer_berger1986.txt"),
    )
    .expect("the auxiliary tabulations parse")
}

/// Where the three auxiliary photon tabulations live.
///
/// Resolved from `CARGO_MANIFEST_DIR`, not from a cache: these are git-tracked
/// text files that ship inside the wheel, so they are present on a clean
/// checkout and a test that reads them is still hermetic.
pub fn photon_tabulation_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/yamc-core/python/yamc/data")
}

// ---------------------------------------------------------------------------
// Reading a written section back.
// ---------------------------------------------------------------------------

/// One written section, read through the loader's own reader.
///
/// This is `yamc_nuclide::arrow_helpers::read_arrow_file`, and using it rather
/// than a local `FileReader` matters twice. It fuses `reactions.arrow`'s
/// batch-per-row framing into one row space, where taking `batches[0]` (which
/// is what `sections.rs:41` does) yields the first MT alone. And it runs
/// `nuclear_data_schema::check_batch` on the way in, so a column the
/// declaration does not know is an error here rather than a downcast panic in
/// a test.
pub fn section(dir: &Path, file: &str) -> RecordBatch {
    yamc_nuclide::arrow_helpers::read_arrow_file(&dir.join(file))
        .unwrap_or_else(|e| panic!("{file}: {e}"))
}

/// True when a section file was not written at all.
pub fn absent(dir: &Path, file: &str) -> bool {
    !dir.join(file).is_file()
}

pub fn f64_at(batch: &RecordBatch, col: &str, row: usize) -> f64 {
    yamc_nuclide::arrow_helpers::get_f64(batch, col, row).unwrap_or_else(|e| panic!("{col}: {e}"))
}

pub fn i32_at(batch: &RecordBatch, col: &str, row: usize) -> i32 {
    yamc_nuclide::arrow_helpers::get_i32(batch, col, row).unwrap_or_else(|e| panic!("{col}: {e}"))
}

pub fn str_at(batch: &RecordBatch, col: &str, row: usize) -> String {
    yamc_nuclide::arrow_helpers::get_str(batch, col, row).unwrap_or_else(|e| panic!("{col}: {e}"))
}

pub fn bool_at(batch: &RecordBatch, col: &str, row: usize) -> bool {
    yamc_nuclide::arrow_helpers::get_bool(batch, col, row).unwrap_or_else(|e| panic!("{col}: {e}"))
}

pub fn f64_list(batch: &RecordBatch, col: &str, row: usize) -> Vec<f64> {
    yamc_nuclide::arrow_helpers::get_f64_list(batch, col, row)
        .unwrap_or_else(|e| panic!("{col}: {e}"))
}

pub fn i32_list(batch: &RecordBatch, col: &str, row: usize) -> Vec<i32> {
    yamc_nuclide::arrow_helpers::get_i32_list(batch, col, row)
        .unwrap_or_else(|e| panic!("{col}: {e}"))
}

pub fn str_list(batch: &RecordBatch, col: &str, row: usize) -> Vec<String> {
    yamc_nuclide::arrow_helpers::get_str_list(batch, col, row)
        .unwrap_or_else(|e| panic!("{col}: {e}"))
}

/// A `list<list<double>>` cell as owned rows.
pub fn nested_f64_list(batch: &RecordBatch, col: &str, row: usize) -> Vec<Vec<f64>> {
    yamc_nuclide::arrow_helpers::borrow_nested_f64_list(batch, col, row)
        .unwrap_or_else(|e| panic!("{col}: {e}"))
        .into_iter()
        .map(|b| b.to_vec())
        .collect()
}

/// Whether a cell is null, as distinct from an empty list.
///
/// The two are different files and the loader can tell them apart, but
/// `try_get_f64_list` collapses both to an empty `Vec`, so a test that wants
/// the distinction has to go to the array. `subshells.arrow` writes null for a
/// shell with no relaxation cascade, and the published files do the same, so
/// an empty list there would load identically and still not be the same file.
pub fn is_null(batch: &RecordBatch, col: &str, row: usize) -> bool {
    batch
        .column_by_name(col)
        .unwrap_or_else(|| panic!("no column {col}"))
        .is_null(row)
}

/// The number of rows in a `list` cell's inner list, without materialising it.
pub fn list_len(batch: &RecordBatch, col: &str, row: usize) -> usize {
    let arr = batch
        .column_by_name(col)
        .unwrap_or_else(|| panic!("no column {col}"));
    arr.as_list::<i32>().value(row).len()
}

// ---------------------------------------------------------------------------
// Comparison.
// ---------------------------------------------------------------------------

/// Element-for-element `assert_eq!` on two f64 slices, with the index and both
/// values in the message.
///
/// Exact, deliberately. A column that is a verbatim clone of parsed data goes
/// through Arrow Float64 with LZ4, which is lossless, so any tolerance here
/// could only hide the mistakes the comparison exists to find: a unit
/// conversion applied twice, a column written into the wrong slot, a curve
/// shifted by one grid point.
#[track_caller]
pub fn assert_f64_slice_eq(what: &str, written: &[f64], expected: &[f64]) {
    assert_eq!(
        written.len(),
        expected.len(),
        "{what}: wrote {} values, the evaluation has {}",
        written.len(),
        expected.len()
    );
    for (i, (&w, &e)) in written.iter().zip(expected).enumerate() {
        assert_eq!(w, e, "{what}[{i}]: wrote {w:e}, the evaluation has {e:e}");
    }
}

#[track_caller]
pub fn assert_i32_slice_eq(what: &str, written: &[i32], expected: &[i32]) {
    assert_eq!(
        written, expected,
        "{what}: wrote {written:?}, the evaluation has {expected:?}"
    );
}

/// Element-for-element comparison to a relative tolerance, for the columns
/// that are recomputed rather than copied.
///
/// `rel` is a fraction, not a count of ulps. Zero on both sides passes; zero
/// on one side alone does not, since that is exactly the threshold-shift
/// mistake these tests are looking for.
#[track_caller]
pub fn assert_f64_slice_close(what: &str, written: &[f64], expected: &[f64], rel: f64) {
    assert_eq!(
        written.len(),
        expected.len(),
        "{what}: wrote {} values, expected {}",
        written.len(),
        expected.len()
    );
    for (i, (&w, &e)) in written.iter().zip(expected).enumerate() {
        let tol = rel * e.abs();
        assert!(
            (w - e).abs() <= tol,
            "{what}[{i}]: wrote {w:e}, expected {e:e}, off by {:e} which is more than {rel:e} relative",
            (w - e).abs()
        );
    }
}

#[track_caller]
pub fn assert_close(what: &str, written: f64, expected: f64, rel: f64) {
    let tol = rel * expected.abs();
    assert!(
        (written - expected).abs() <= tol,
        "{what}: wrote {written:e}, expected {expected:e}, off by {:e} which is more than {rel:e} relative",
        (written - expected).abs()
    );
}

/// The declared schema, field for field, in order.
///
/// Worth stating plainly what this does NOT do: `write_section` builds every
/// batch against the declared schema (`sections.rs:38-50`), so swapping two
/// same-typed argument expressions produces a batch that satisfies this and
/// carries the wrong numbers. It catches schema drift. Only a per-column value
/// comparison catches a transposition, which is why nothing in these files
/// substitutes this for one.
#[track_caller]
pub fn assert_schema_is_declared(batch: &RecordBatch, section_name: &str) {
    let declared = nuclear_data_schema::section(section_name)
        .unwrap_or_else(|| panic!("no declared schema for {section_name}"));
    let written: Vec<_> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    let expected: Vec<_> = declared
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    assert_eq!(
        written, expected,
        "{section_name} was written with a schema the declaration does not match"
    );
}
