//! Every discriminant value in the shipped data must resolve in this reader.
//!
//! The Arrow schema declares columns like `distributions.type` and
//! `products.yield_type` as plain `utf8`. It pins the name, the type and the
//! nullability, and it cannot pin the vocabulary: writer and reader agree only
//! by both spelling the same literals. That makes these columns the one part of
//! the format where a mismatch is invisible to `test_schema_manifest.py`, to
//! `check_declared`, and to every field-name check there is.
//!
//! It has bitten before. Issue #379: the transmutation writer spelled MT 18
//! "fission" while the Rust consumer's map held only "(n,fission)", so no
//! fission product was ever produced and every Python-side test stayed green.
//! The same shape sat unfired in the neutron reader, where `energy_dist_type`
//! "madland-nix" (written by `neutron_writer.py`) had no match arm and fell
//! through a catch-all to `None`, silently loading the distribution with no
//! energy component at all.
//!
//! Two tests, and both are needed. The first is total over the corpus rather
//! than sampled: it walks every committed fixture through the real loader, so a
//! value the reader does not know fails here rather than in someone's
//! simulation. The second proves the first is not vacuous by feeding the loader
//! a value nothing accepts and demanding it refuse: without it, restoring a
//! catch-all arm would leave the first test passing and the guard gone.

#![cfg(feature = "arrow")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, StringArray};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;

use yamc_nuclide::nuclide_arrow::read_nuclide_from_arrow;
use yamc_nuclide::LoadScope;

/// The committed nuclide fixtures live beside the `yamc` crate's tests.
fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("yamc")
        .join("tests")
}

/// Every `*.arrow` directory holding a `nuclide.arrow`. The photon element
/// directories (`Fe.arrow`, `Li.arrow`, ...) sit in the same place and are read
/// by a different loader, so they are skipped here by that test rather than by
/// name.
fn nuclide_fixtures() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(fixture_root())
        .expect("fixture directory is readable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir() && p.extension().is_some_and(|e| e == "arrow"))
        .filter(|p| p.join("nuclide.arrow").is_file())
        .collect();
    dirs.sort();
    dirs
}

/// Load every committed nuclide fixture through the real reader.
///
/// Total over the corpus, not sampled: the point is that no discriminant value
/// present in any shipped file is one this build cannot resolve. The count
/// assertion is not decoration. An empty or mislocated fixture directory would
/// otherwise make this pass in 0.00s having checked nothing, which is the
/// failure mode that let earlier data bugs through.
#[test]
fn every_fixture_nuclide_loads() {
    let fixtures = nuclide_fixtures();
    assert!(
        fixtures.len() >= 16,
        "expected the committed nuclide fixtures, found {} in {}",
        fixtures.len(),
        fixture_root().display()
    );

    for dir in &fixtures {
        if let Err(e) = read_nuclide_from_arrow(dir, &LoadScope::full()) {
            panic!("{} failed to load: {e}", dir.display());
        }
    }
}

/// Rewrite one string column of an Arrow IPC file so every row holds `value`,
/// and write the result to `dst`. Used to inject a discriminant no build knows.
fn rewrite_string_column(src: &Path, dst: &Path, column: &str, value: &str) {
    let reader = FileReader::try_new(std::fs::File::open(src).expect("open fixture"), None)
        .expect("fixture is a valid Arrow IPC file");
    let schema = reader.schema();
    let idx = schema
        .index_of(column)
        .unwrap_or_else(|_| panic!("{} has no column {column}", src.display()));
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().expect("read batches");

    let out = std::fs::File::create(dst).expect("create rewritten file");
    let mut writer = FileWriter::try_new(out, &schema).expect("open writer");
    for batch in &batches {
        let mut columns = batch.columns().to_vec();
        let replacement = StringArray::from(vec![value; batch.num_rows()]);
        columns[idx] = Arc::new(replacement) as Arc<dyn Array>;
        let rewritten =
            RecordBatch::try_new(schema.clone(), columns).expect("rebuild batch with new column");
        writer.write(&rewritten).expect("write batch");
    }
    writer.finish().expect("finish writer");
}

/// Copy a fixture directory, replacing one section file with a rewritten one.
fn fixture_with_column_replaced(
    nuclide: &str,
    section: &str,
    column: &str,
    value: &str,
    into: &Path,
) -> PathBuf {
    let src = fixture_root().join(nuclide);
    let dst = into.join(nuclide);
    std::fs::create_dir_all(&dst).expect("create temp fixture dir");
    for entry in std::fs::read_dir(&src).expect("read fixture dir") {
        let path = entry.expect("dir entry").path();
        let name = path.file_name().expect("file name").to_owned();
        if path.is_file() {
            std::fs::copy(&path, dst.join(&name)).expect("copy section");
        }
    }
    rewrite_string_column(&src.join(section), &dst.join(section), column, value);
    dst
}

/// A `distributions.type` nothing accepts must be refused, not dropped.
///
/// Before this, an unknown type fell through `_ => None` and the row's
/// distribution vanished: the nuclide loaded, the simulation ran, and the
/// answer was wrong rather than absent.
#[test]
fn unknown_distribution_type_is_refused() {
    let tmp = std::env::temp_dir().join("yamc-discriminant-dist-type");
    let _ = std::fs::remove_dir_all(&tmp);
    let dir = fixture_with_column_replaced(
        "Fe56.arrow",
        "distributions.arrow",
        "type",
        "not-a-distribution-shape",
        &tmp,
    );

    let err = read_nuclide_from_arrow(&dir, &LoadScope::full())
        .expect_err("an unknown distribution type must not load");
    let msg = err.to_string();
    assert!(
        msg.contains("distributions.arrow") && msg.contains("not-a-distribution-shape"),
        "the error must name the section and the offending value, got: {msg}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// The same for `energy_dist_type`, which is the arm "madland-nix" fell through.
///
/// `neutron_writer.py` writes "madland-nix" for an LF=12 fission spectrum and
/// this reader has no arm for it, so it is used here as the injected value: it
/// is not hypothetical, it is what the converter emits today for a Madland-Nix
/// evaluation. No such nuclide is in the published data yet, which is why this
/// never fired in the field.
#[test]
fn unknown_energy_distribution_type_is_refused() {
    let tmp = std::env::temp_dir().join("yamc-discriminant-energy-dist-type");
    let _ = std::fs::remove_dir_all(&tmp);
    let dir = fixture_with_column_replaced(
        "Fe56.arrow",
        "distributions.arrow",
        "energy_dist_type",
        "madland-nix",
        &tmp,
    );

    let err = read_nuclide_from_arrow(&dir, &LoadScope::full())
        .expect_err("an unknown energy distribution type must not load");
    let msg = err.to_string();
    assert!(
        msg.contains("energy_dist_type") && msg.contains("madland-nix"),
        "the error must name the column and the offending value, got: {msg}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// An unknown `products.particle` must be refused rather than silently becoming
/// a neutron, which is what the old fallback did: a proton product would have
/// been transported as a neutron with the proton's yield.
#[test]
fn unknown_product_particle_is_refused() {
    let tmp = std::env::temp_dir().join("yamc-discriminant-particle");
    let _ = std::fs::remove_dir_all(&tmp);
    let dir =
        fixture_with_column_replaced("Fe56.arrow", "products.arrow", "particle", "proton", &tmp);

    let err = read_nuclide_from_arrow(&dir, &LoadScope::full())
        .expect_err("an unknown product particle must not load");
    let msg = err.to_string();
    assert!(
        msg.contains("products.arrow") && msg.contains("proton"),
        "the error must name the section and the offending value, got: {msg}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// An unknown `products.yield_type` must be refused rather than leaving the
/// product with no multiplicity at all.
#[test]
fn unknown_yield_type_is_refused() {
    let tmp = std::env::temp_dir().join("yamc-discriminant-yield-type");
    let _ = std::fs::remove_dir_all(&tmp);
    let dir = fixture_with_column_replaced(
        "Fe56.arrow",
        "products.arrow",
        "yield_type",
        "Interpolated",
        &tmp,
    );

    let err = read_nuclide_from_arrow(&dir, &LoadScope::full())
        .expect_err("an unknown yield type must not load");
    let msg = err.to_string();
    assert!(
        msg.contains("yield_type") && msg.contains("Interpolated"),
        "the error must name the column and the offending value, got: {msg}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
