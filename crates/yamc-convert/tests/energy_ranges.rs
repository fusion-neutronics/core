//! The union energy grids as their own section, one batch per temperature.
//!
//! The energy half of fusion-neutronics/core#100. Three claims, in order of how
//! much they would cost to get wrong:
//!
//! 1. A version 1 folder migrates without changing a single number. The grids
//!    are copied, so this is checkable exactly rather than to a tolerance.
//! 2. One temperature's batch can be spliced out on its own and read back,
//!    which is the whole point of the index.
//! 3. The real fixture loads through the reader a run uses, and the grid it
//!    hands out at each temperature is the one the section holds.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, ListBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};
use nuclear_data_schema::reaction_ranges::splice_spans;
use yamc_convert::energy_ranges::index_energy;
use yamc_convert::nuclide::migrate_energy_out_of_nuclide;

fn fixture(nuclide: &str) -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../yamc/tests")
        .join(format!("{nuclide}.arrow"));
    dir.join("energy.arrow").exists().then_some(dir)
}

fn range(bytes: &[u8], (off, len): (u64, u64)) -> Vec<u8> {
    bytes[off as usize..(off + len) as usize].to_vec()
}

/// The grids a folder's `energy.arrow` holds, by label.
fn grids_of(dir: &Path) -> Vec<(String, Vec<f64>)> {
    let batch = yamc_nuclide::arrow_helpers::read_arrow_file(&dir.join("energy.arrow")).unwrap();
    (0..batch.num_rows())
        .map(|row| {
            let label = batch
                .column_by_name("temperature")
                .unwrap()
                .as_string::<i32>()
                .value(row)
                .to_string();
            let cell = batch
                .column_by_name("energy_values")
                .unwrap()
                .as_list::<i32>()
                .value(row);
            let grid = cell
                .as_primitive::<Float64Type>()
                .iter()
                .flatten()
                .collect();
            (label, grid)
        })
        .collect()
}

/// A `list<double>` column of one row.
fn f64_list_col(values: &[f64]) -> ArrayRef {
    let mut b = ListBuilder::new(Float64Builder::new());
    b.values().append_slice(values);
    b.append(true);
    Arc::new(b.finish())
}

/// A `list<utf8>` column of one row.
fn str_list_col(values: &[&str]) -> ArrayRef {
    let mut b = ListBuilder::new(StringBuilder::new());
    for v in values {
        b.values().append_value(v);
    }
    b.append(true);
    Arc::new(b.finish())
}

/// A `list<list<double>>` column of one row: the shape version 1 used.
fn nested_f64_list_col(grids: &[Vec<f64>]) -> ArrayRef {
    let mut b = ListBuilder::new(ListBuilder::new(Float64Builder::new()));
    for g in grids {
        b.values().values().append_slice(g);
        b.values().append(true);
    }
    b.append(true);
    Arc::new(b.finish())
}

/// A version 1 `{Name}.arrow` folder: the grids inside `nuclide.arrow`.
///
/// Built here rather than taken from the cache so this runs in CI whatever the
/// published library is, and keeps running once every library is version 2 and
/// no version 1 folder exists to borrow.
fn v1_folder(labels: &[&str], grids: &[Vec<f64>]) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let list = |name: &str, item: DataType| {
        Field::new(
            name,
            DataType::List(Arc::new(Field::new("item", item, true))),
            true,
        )
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, true),
        Field::new("Z", DataType::Int32, true),
        Field::new("A", DataType::Int32, true),
        Field::new("atomic_weight_ratio", DataType::Float64, true),
        list("temperatures", DataType::Utf8),
        list("kTs", DataType::Float64),
        list("energy_temperatures", DataType::Utf8),
        list(
            "energy_values",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
        ),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Xx1"])),
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(arrow_array::Float64Array::from(vec![0.999])),
            str_list_col(labels),
            f64_list_col(&vec![0.0253; labels.len()]),
            str_list_col(labels),
            nested_f64_list_col(grids),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(tmp.path().join("nuclide.arrow")).unwrap();
    let mut writer = FileWriter::try_new(file, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    std::fs::write(
        tmp.path().join("version.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "format_version": 1,
            "library": "test",
            "data_version": "test",
        }))
        .unwrap(),
    )
    .unwrap();
    tmp
}

#[test]
fn a_migration_moves_every_grid_across_unchanged() {
    let labels = ["294K", "600K", "0K"];
    let grids = vec![
        vec![1.0e-5, 1.0, 2.0e7],
        vec![1.0e-5, 1.5, 2.5, 2.0e7],
        vec![1.0e-5, 2.0e7],
    ];
    let dir = v1_folder(&labels, &grids);

    assert!(migrate_energy_out_of_nuclide(dir.path()).unwrap());
    let moved = grids_of(dir.path());
    assert_eq!(
        moved.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
        labels,
        "the rows keep the order nuclide.arrow listed them in, 0 K last"
    );
    for ((label, got), want) in moved.iter().zip(&grids) {
        assert_eq!(got, want, "the {label} grid changed in the move");
    }

    // nuclide.arrow keeps its identity and loses the grids, which is the whole
    // saving: a client that only wants to know what a nuclide IS no longer
    // reads every grid to find out.
    let slim =
        yamc_nuclide::arrow_helpers::read_arrow_file(&dir.path().join("nuclide.arrow")).unwrap();
    assert_eq!(
        slim.column_by_name("name")
            .unwrap()
            .as_string::<i32>()
            .value(0),
        "Xx1"
    );
    assert!(slim.column_by_name("energy_values").is_none());
    assert!(slim.column_by_name("energy_temperatures").is_none());

    // Idempotent: a rerun over a folder already moved finds nothing to do,
    // which is what makes an interrupted migration safe to repeat.
    assert!(!migrate_energy_out_of_nuclide(dir.path()).unwrap());
}

#[test]
fn a_migrated_folder_indexes_one_batch_per_temperature() {
    let labels = ["294K", "600K"];
    let grids = vec![vec![1.0e-5, 1.0, 2.0e7], vec![1.0e-5, 2.0, 3.0, 2.0e7]];
    let dir = v1_folder(&labels, &grids);
    migrate_energy_out_of_nuclide(dir.path()).unwrap();

    let path = dir.path().join("energy.arrow");
    let index = index_energy(&path).unwrap();
    assert_eq!(
        index.temperatures.keys().cloned().collect::<Vec<_>>(),
        vec!["294K".to_string(), "600K".to_string()]
    );
    // Distinct batches, or the index would name the same bytes twice and a
    // single-temperature fetch would be a whole-file fetch wearing a hat.
    assert_ne!(index.temperatures["294K"], index.temperatures["600K"]);

    // And the splice reads back that one grid, which is the claim the whole
    // index exists to support.
    let bytes = std::fs::read(&path).unwrap();
    let spans = index.spans_for(|t| t == "600K");
    let stream = splice_spans(&spans.iter().map(|s| range(&bytes, *s)).collect::<Vec<_>>());
    let batches: Vec<RecordBatch> = StreamReader::try_new(std::io::Cursor::new(stream), None)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(batches.len(), 1, "one temperature is one batch");
    let only = &batches[0];
    assert_eq!(
        only.column_by_name("temperature")
            .unwrap()
            .as_string::<i32>()
            .value(0),
        "600K"
    );
    let cell = only
        .column_by_name("energy_values")
        .unwrap()
        .as_list::<i32>()
        .value(0);
    let got: Vec<f64> = cell
        .as_primitive::<Float64Type>()
        .iter()
        .flatten()
        .collect();
    assert_eq!(got, grids[1]);
}

/// The real fixture, end to end through the loader, and the saving measured.
///
/// Self-skips without the downloaded fixtures, like every other test here that
/// needs them.
#[test]
fn the_fe56_fixture_serves_one_temperature_from_its_own_batch() {
    let Some(dir) = fixture("Fe56") else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    let path = dir.join("energy.arrow");
    let index = index_energy(&path).unwrap();
    let whole = std::fs::metadata(&path).unwrap().len();
    let one: u64 = index
        .spans_for(|t| t == "294K")
        .iter()
        .map(|(_, l)| l)
        .sum();
    eprintln!(
        "Fe56 energy.arrow: {whole} bytes for {} grids, {one} bytes for 294K alone ({:.1}x less)",
        index.temperatures.len(),
        whole as f64 / one as f64
    );
    assert!(
        one * 3 < whole,
        "one of {} grids should be a small fraction of the file, got {one} of {whole}",
        index.temperatures.len()
    );

    // nuclide.arrow is metadata now, so the identity of a nuclide is cheap to
    // ask for. Before #100 this file carried every grid.
    let meta = std::fs::metadata(dir.join("nuclide.arrow")).unwrap().len();
    eprintln!("Fe56 nuclide.arrow: {meta} bytes");
    assert!(
        meta < 100_000,
        "nuclide.arrow is metadata only and should be small, got {meta} bytes"
    );

    // Through the reader a real run uses: every temperature the file publishes
    // resolves to a grid, at the same lengths the section holds.
    let scope = yamc_nuclide::LoadScope::activation((1..1000).collect());
    let nuclide = yamc_nuclide::nuclide_loader::load_nuclide(&dir, &scope).expect("loads");
    let energy = nuclide.energy.as_ref().expect("energy grids");
    for (label, grid) in grids_of(&dir) {
        let key = label.trim_end_matches('K');
        let loaded = energy
            .get(key)
            .unwrap_or_else(|| panic!("{label} is in energy.arrow and not in the loaded map"));
        assert_eq!(loaded.len(), grid.len(), "{label}: grid length");
        assert_eq!(loaded.first(), grid.first(), "{label}: first point");
        assert_eq!(loaded.last(), grid.last(), "{label}: last point");
    }
}
