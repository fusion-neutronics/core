//! The claim the byte-range index rests on: fetching only some MTs' ranges and
//! splicing them into a stream gives batches identical to reading the whole
//! file. If that fails, an activation run silently gets different cross
//! sections from a transport run over the same data.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use arrow_array::RecordBatch;
use arrow_ipc::reader::{FileReader, StreamReader};
use yamc_convert::reaction_ranges::{index_reactions, splice_stream, ReactionRanges};

/// A converted nuclide from the downloaded test fixtures.
///
/// `python3 scripts/fetch_test_fixtures.py` populates these; skip rather than
/// fail when they are absent, matching how the rest of the suite treats them.
fn fixture(nuclide: &str) -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../yamc/tests")
        .join(format!("{nuclide}.arrow"));
    dir.join("reactions.arrow").exists().then_some(dir)
}

fn batches_by_mt(path: &Path) -> BTreeMap<i32, RecordBatch> {
    let reader = FileReader::try_new(std::fs::File::open(path).unwrap(), None).unwrap();
    reader
        .map(|b| {
            let b = b.unwrap();
            let mt = b
                .column_by_name("mt")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .unwrap()
                .value(0);
            (mt, b)
        })
        .collect()
}

/// Read the exact bytes an HTTP range request would return.
fn range(bytes: &[u8], (offset, len): (u64, u64)) -> Vec<u8> {
    let at = offset as usize;
    bytes[at..at + len as usize].to_vec()
}

#[test]
fn spliced_ranges_reproduce_the_whole_file() {
    let Some(dir) = fixture("Fe56") else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    let path = dir.join("reactions.arrow");
    let bytes = std::fs::read(&path).unwrap();
    let index = index_reactions(&path).unwrap();
    let whole = batches_by_mt(&path);

    assert_eq!(
        index.mts.keys().copied().collect::<Vec<_>>(),
        whole.keys().copied().collect::<Vec<_>>(),
        "the index must name exactly the MTs the file holds",
    );

    // Every MT, one at a time: this is the isotope-plotter case, and the
    // strictest form of the claim.
    let schema = range(&bytes, index.schema);
    for (&mt, &r) in &index.mts {
        let stream = splice_stream(&schema, &[range(&bytes, r)]);
        let mut reader = StreamReader::try_new(stream.as_slice(), None).unwrap();
        let got = reader.next().expect("one batch").unwrap();
        assert!(
            reader.next().is_none(),
            "MT {mt} spliced to more than one batch"
        );
        assert_eq!(
            &got, &whole[&mt],
            "MT {mt} differs from the whole-file read"
        );
    }
}

#[test]
fn an_activation_subset_splices_into_one_stream() {
    let Some(dir) = fixture("Fe56") else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    let path = dir.join("reactions.arrow");
    let bytes = std::fs::read(&path).unwrap();
    let index = index_reactions(&path).unwrap();
    let whole = batches_by_mt(&path);

    // The channels an activation chain actually names, as opposed to the
    // full-grid transport MTs that dominate the file.
    let wanted: Vec<i32> = [16, 22, 28, 102, 103, 104, 105, 106, 107]
        .into_iter()
        .filter(|mt| index.mts.contains_key(mt))
        .collect();
    assert!(
        wanted.len() > 3,
        "fixture should carry several activation MTs"
    );

    let schema = range(&bytes, index.schema);
    let fetched: Vec<Vec<u8>> = wanted
        .iter()
        .map(|mt| range(&bytes, index.mts[mt]))
        .collect();
    let fetched_bytes: usize = fetched.iter().map(Vec::len).sum();

    let stream = splice_stream(&schema, &fetched);
    let got: Vec<RecordBatch> = StreamReader::try_new(stream.as_slice(), None)
        .unwrap()
        .map(Result::unwrap)
        .collect();

    assert_eq!(got.len(), wanted.len());
    for (batch, mt) in got.iter().zip(&wanted) {
        assert_eq!(
            batch, &whole[mt],
            "MT {mt} differs from the whole-file read"
        );
    }

    // The point of the exercise. Not asserted as a hard threshold because it is
    // evaluation-dependent, but a subset that is not markedly smaller means the
    // index is not buying anything and something has gone wrong.
    assert!(
        fetched_bytes * 2 < bytes.len(),
        "activation subset was {fetched_bytes} of {} bytes; expected far less",
        bytes.len(),
    );
    eprintln!(
        "activation subset: {fetched_bytes} of {} bytes ({:.1}x less)",
        bytes.len(),
        bytes.len() as f64 / fetched_bytes as f64,
    );
}

#[test]
fn json_round_trips() {
    let ranges = ReactionRanges {
        schema: (8, 1176),
        mts: [(16, (2451328, 7770)), (102, (19923456, 1363869))]
            .into_iter()
            .collect(),
    };
    assert_eq!(ReactionRanges::from_json(&ranges.to_json()), Some(ranges));
}

#[test]
fn absent_index_is_not_an_error() {
    // What every already-published library looks like until it is reindexed.
    let version = serde_json::json!({"format_version": 1, "library": "endf-b8.1"});
    assert_eq!(ReactionRanges::from_json(&version), None);
}

/// The index has to work on files this crate writes, not just on the published
/// ones the fixture happens to be.
///
/// The fixture is a downloaded object, written by the Python converter, whose
/// schema message begins immediately after the file magic. arrow-rs, which is
/// what writes a conversion here, pads it to a 64-byte boundary. Taking the
/// schema to start at the end of the magic reads that padding as a legacy frame
/// of length zero, records a 4-byte schema, and yields a stream no reader can
/// decode, since the LZ4 codec is declared in the schema message.
#[test]
fn a_file_this_crate_wrote_indexes_and_splices() {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array};
    use arrow_ipc::writer::{FileWriter, IpcWriteOptions};
    use arrow_ipc::CompressionType;
    use arrow_schema::{DataType, Field, Schema};

    let schema = Arc::new(Schema::new(vec![Field::new("mt", DataType::Int32, false)]));
    let dir = std::env::temp_dir().join("yamc_convert_rust_written_index");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("reactions.arrow");

    // One batch per MT, LZ4, exactly as a conversion writes them.
    let options = IpcWriteOptions::default()
        .try_with_compression(Some(CompressionType::LZ4_FRAME))
        .unwrap();
    let mut writer =
        FileWriter::try_new_with_options(std::fs::File::create(&path).unwrap(), &schema, options)
            .unwrap();
    for mt in [1, 2, 102] {
        let column: ArrayRef = Arc::new(Int32Array::from(vec![mt; 4]));
        writer
            .write(&RecordBatch::try_new(schema.clone(), vec![column]).unwrap())
            .unwrap();
    }
    writer.finish().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let ranges = index_reactions(&path).unwrap();

    // The schema is where the writer actually put it, and long enough to be one.
    let (offset, len) = ranges.schema;
    assert!(
        offset >= 8 && len > 8,
        "schema range {ranges:?} is not a message: a 4-byte length means the \
         padding was read as the frame",
    );

    let whole = batches_by_mt(&path);
    for (mt, range) in &ranges.mts {
        let spliced = splice_stream(
            &range_of(&bytes, ranges.schema),
            &[range_of(&bytes, *range)],
        );
        let mut reader = StreamReader::try_new(spliced.as_slice(), None)
            .unwrap_or_else(|e| panic!("MT {mt} spliced into an unreadable stream: {e}"));
        let got = reader.next().expect("one batch").unwrap();
        assert_eq!(&got, &whole[mt], "MT {mt} differs from the whole-file read");
    }

    std::fs::remove_dir_all(&dir).ok();
}

fn range_of(bytes: &[u8], (offset, len): (u64, u64)) -> Vec<u8> {
    bytes[offset as usize..(offset + len) as usize].to_vec()
}

/// The reader half of the claim. The two tests above prove the spliced bytes
/// are right; this proves the loader accepts them. Without it the index buys
/// nothing, because the bytes arrive correct and `read_arrow_file` refuses
/// them: a stream has no `ARROW1` magic and no footer, and `FileReader` wants
/// both.
///
/// Reading the whole file through the same function in the same test is the
/// regression guard on the other branch of the sniff: a change that made
/// streams work by breaking files would pass the assertions above and fail
/// here.
#[test]
fn read_arrow_file_accepts_a_spliced_stream() {
    let Some(dir) = fixture("Fe56") else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    let path = dir.join("reactions.arrow");
    let bytes = std::fs::read(&path).unwrap();
    let index = index_reactions(&path).unwrap();

    let wanted: Vec<i32> = [16, 102, 103, 107]
        .into_iter()
        .filter(|mt| index.mts.contains_key(mt))
        .collect();
    assert!(
        wanted.len() > 2,
        "fixture should carry several activation MTs"
    );

    let schema = range(&bytes, index.schema);
    let fetched: Vec<Vec<u8>> = wanted
        .iter()
        .map(|mt| range(&bytes, index.mts[mt]))
        .collect();
    let stream = splice_stream(&schema, &fetched);

    // Named `reactions.arrow` so the reader's declared-schema check
    // (`nuclear_data_schema::flat_section_for_path`) applies to the spliced
    // stream exactly as it does to the published file. A stream that decoded
    // but no longer matched the section's schema would pass unnoticed under any
    // other name.
    let tmp = tempfile::tempdir().unwrap();
    let spliced_path = tmp.path().join("reactions.arrow");
    std::fs::write(&spliced_path, &stream).unwrap();

    let got = yamc_nuclide::arrow_helpers::read_arrow_file(&spliced_path).unwrap();
    let whole = yamc_nuclide::arrow_helpers::read_arrow_file(&path).unwrap();

    assert_eq!(
        whole.num_rows(),
        index.mts.len(),
        "fixture is one row per MT, which is what indexing by row below assumes",
    );
    assert_eq!(got.num_rows(), wanted.len());

    let mts = |batch: &RecordBatch| -> Vec<i32> {
        batch
            .column_by_name("mt")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap()
            .values()
            .to_vec()
    };
    let whole_mts = mts(&whole);
    assert_eq!(mts(&got), wanted, "spliced batch holds the MTs asked for");

    // Row for row against the whole-file read, which is the only comparison
    // that catches a stream that decodes into plausible but different numbers.
    for (row, mt) in wanted.iter().enumerate() {
        let at = whole_mts.iter().position(|m| m == mt).unwrap();
        assert_eq!(
            got.slice(row, 1),
            whole.slice(at, 1),
            "MT {mt} differs from the whole-file read",
        );
    }
}

/// A nuclide that publishes none of the wanted MTs splices to a schema and
/// nothing after it. That is an empty subset, not a corrupt one, and it must
/// read as zero reactions rather than fail.
///
/// He4 is the real case, in both endf-b8.1 and jendl-5.0: it carries elastic and
/// the transport lookups and no activation channel at all, so an activation run
/// that loads it wants none of what it has. Reading the whole file produces the
/// same zero reactions once the MT filter runs, so erroring here would break a
/// nuclide that loads fine today.
#[test]
fn a_subset_with_none_of_the_wanted_mts_reads_as_no_reactions() {
    let Some(dir) = fixture("Fe56") else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    let path = dir.join("reactions.arrow");
    let bytes = std::fs::read(&path).unwrap();
    let index = index_reactions(&path).unwrap();

    let stream = splice_stream(&range(&bytes, index.schema), &[]);
    let tmp = tempfile::tempdir().unwrap();
    let spliced = tmp.path().join("reactions.arrow");
    std::fs::write(&spliced, &stream).unwrap();

    let batch = yamc_nuclide::arrow_helpers::read_arrow_file(&spliced)
        .expect("an empty subset is not an error");
    assert_eq!(batch.num_rows(), 0);
    // The schema still has to be the section's, or the loader's declared-schema
    // check would be passing something it never inspected.
    assert!(batch.schema().field_with_name("mt").is_ok());
}

/// An empty *file*, by contrast, is damage and stays an error: it has a footer
/// promising batches that are not there.
#[test]
fn an_empty_file_is_still_an_error() {
    let Some(dir) = fixture("Fe56") else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    // Written as a file (magic + footer) with no batches at all.
    let src = FileReader::try_new(
        std::fs::File::open(dir.join("reactions.arrow")).unwrap(),
        None,
    )
    .unwrap();
    let schema = src.schema();
    let tmp = tempfile::tempdir().unwrap();
    let empty = tmp.path().join("reactions.arrow");
    {
        let file = std::fs::File::create(&empty).unwrap();
        let mut writer = arrow_ipc::writer::FileWriter::try_new(file, &schema).unwrap();
        writer.finish().unwrap();
    }
    let err = yamc_nuclide::arrow_helpers::read_arrow_file(&empty).unwrap_err();
    assert!(
        err.to_string().contains("No record batches"),
        "unexpected error: {err}",
    );
}

/// A source that is neither framing must still fail as a truncated file rather
/// than as a stream, so the message keeps naming what the caller expected.
#[test]
fn a_file_too_short_to_sniff_still_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("reactions.arrow");
    std::fs::write(&path, b"AR").unwrap();
    let err = yamc_nuclide::arrow_helpers::read_arrow_file(&path).unwrap_err();
    assert!(
        err.to_string().contains("reactions.arrow"),
        "error should name the section, got: {err}",
    );
}
