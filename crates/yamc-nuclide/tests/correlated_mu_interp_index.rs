//! Regression for issue #484: `corr_mu_interp` indexed by code, not by point.
//!
//! `corr_mu_interp` holds one interpolation code per outgoing-energy point of a
//! correlated distribution, values in `{0, 1, 2}`. `parse_correlated` used the
//! code as the index:
//!
//! ```text
//! match mu_interp[mu_interp_indices[j]]      // mu_interp_indices[j] IS a code
//! ```
//!
//! so every mu table in a row took its interpolation from one of the row's
//! first three points instead of from its own. It coincides with the right
//! answer whenever a row's codes are all equal, which is every correlated row
//! in the shipped ENDF/B-VIII.1 data (2435 of them, all code 2), and that is
//! why nothing caught it. It goes wrong for a row that mixes histogram and
//! lin-lin mu tables.
//!
//! No shipped file mixes them, so the mix is injected here: a copy of the Fe56
//! fixture with `corr_mu_interp` rewritten to lin-lin for a row's first three
//! points and histogram for the rest. Under the old indexing every table read
//! `mu_interp[2]`, lin-lin, and not one histogram table came out. That is the
//! difference this test measures.
//!
//! Note the rewrite deliberately leaves row 3 of `corr_eout_data` (the f64 copy
//! of the same codes) alone, so the two disagree in a way no real file does.
//! That is the lever: the reader is supposed to read the int32 column.
#![cfg(feature = "arrow")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder};
use arrow_array::{Array, ListArray, RecordBatch};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;

use yamc_nuclide::nuclide_arrow::read_nuclide_from_arrow;
use yamc_nuclide::reaction_product::AngleEnergyDistribution;
use yamc_nuclide::secondary_correlated::Interpolation;
use yamc_nuclide::LoadScope;

/// Points per row given lin-lin, before the histogram tail starts.
const N_LINLIN: usize = 3;

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("yamc")
        .join("tests")
}

/// Rewrite the `corr_mu_interp` list column: `2` for a row's first
/// [`N_LINLIN`] points, `1` for the rest. Null rows (no correlated
/// distribution) stay null.
fn rewrite_mu_interp(src: &Path, dst: &Path) -> usize {
    let reader = FileReader::try_new(std::fs::File::open(src).expect("open fixture"), None)
        .expect("fixture is a valid Arrow IPC file");
    let schema = reader.schema();
    let idx = schema
        .index_of("corr_mu_interp")
        .expect("distributions.arrow has a corr_mu_interp column");
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().expect("read batches");

    let mut patched_rows = 0usize;
    let out = std::fs::File::create(dst).expect("create rewritten file");
    let mut writer = FileWriter::try_new(out, &schema).expect("open writer");
    for batch in &batches {
        let list = batch
            .column(idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("corr_mu_interp is a ListArray");

        // `ListBuilder<Int32Builder>` names its child field "item", nullable,
        // which is what the schema declares (`nuclear_data_schema::list_of`),
        // so the rebuilt column keeps the type `RecordBatch::try_new` checks.
        let mut builder = ListBuilder::new(Int32Builder::new());
        for row in 0..batch.num_rows() {
            if list.is_null(row) {
                builder.append_null();
                continue;
            }
            let n = list.value(row).len();
            for k in 0..n {
                builder
                    .values()
                    .append_value(if k < N_LINLIN { 2 } else { 1 });
            }
            if n > N_LINLIN {
                patched_rows += 1;
            }
            builder.append(true);
        }
        let replacement = builder.finish();
        let mut columns = batch.columns().to_vec();
        columns[idx] = Arc::new(replacement) as Arc<dyn Array>;
        writer
            .write(&RecordBatch::try_new(schema.clone(), columns).expect("rebuild batch"))
            .expect("write batch");
    }
    writer.finish().expect("finish writer");
    patched_rows
}

/// Copy the fixture, rewriting `corr_mu_interp` in `distributions.arrow`.
fn fixture_with_mixed_mu_interp(nuclide: &str, into: &Path) -> (PathBuf, usize) {
    let src = fixture_root().join(nuclide);
    let dst = into.join(nuclide);
    std::fs::create_dir_all(&dst).expect("create temp fixture dir");
    for entry in std::fs::read_dir(&src).expect("read fixture dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file() {
            let name = path.file_name().expect("file name").to_owned();
            std::fs::copy(&path, dst.join(&name)).expect("copy section");
        }
    }
    let patched = rewrite_mu_interp(
        &src.join("distributions.arrow"),
        &dst.join("distributions.arrow"),
    );
    (dst, patched)
}

#[test]
fn each_mu_table_takes_its_own_interpolation_code() {
    let tmp = std::env::temp_dir().join("yamc-corr-mu-interp-index");
    let _ = std::fs::remove_dir_all(&tmp);
    let (dir, patched_rows) = fixture_with_mixed_mu_interp("Fe56.arrow", &tmp);
    assert!(
        patched_rows > 0,
        "no correlated row in the fixture is long enough to mix codes"
    );

    let nd = read_nuclide_from_arrow(&dir, &LoadScope::full()).expect("patched fixture loads");

    let mut linlin = 0usize;
    let mut histogram = 0usize;
    let mut rows = 0usize;
    for temp_map in &nd.reactions {
        for rxn in temp_map.values() {
            for prod in &rxn.products {
                for ae in &prod.distribution {
                    let AngleEnergyDistribution::CorrelatedAngleEnergy { correlated } = ae else {
                        continue;
                    };
                    // Corrtables hold consecutive runs of the row's points, in
                    // order, one mu table per point, so flattening them walks
                    // the same points `corr_mu_interp` is indexed by.
                    let flat: Vec<Interpolation> = correlated
                        .distributions
                        .iter()
                        .flat_map(|ct| ct.angle.iter().map(|a| a.interpolation))
                        .collect();
                    if flat.len() <= N_LINLIN {
                        continue;
                    }
                    rows += 1;
                    for (k, interp) in flat.iter().enumerate() {
                        let want = if k < N_LINLIN {
                            Interpolation::LinLin
                        } else {
                            Interpolation::Histogram
                        };
                        assert_eq!(
                            *interp, want,
                            "mu table at point {k} took {interp:?}, not its own code's {want:?}"
                        );
                        match want {
                            Interpolation::LinLin => linlin += 1,
                            Interpolation::Histogram => histogram += 1,
                        }
                    }
                }
            }
        }
    }

    assert!(rows > 0, "no correlated distribution was checked");
    // The old indexing read `mu_interp[2]` for every table, so it produced
    // exactly zero of these. The count is what separates the fix from a
    // coincidence.
    assert!(
        histogram > 0,
        "not one histogram mu table: the interpolation is still being looked up by code"
    );
    eprintln!("checked {rows} correlated rows: {histogram} histogram + {linlin} lin-lin mu tables");

    let _ = std::fs::remove_dir_all(&tmp);
}
