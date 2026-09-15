//! The claim the byte-range index rests on: fetching only some batches' ranges
//! and splicing them into a stream gives batches identical to reading the whole
//! file. If that fails, an activation run silently gets different cross
//! sections from a transport run over the same data.
//!
//! And the claim the temperature split rests on (fusion-neutronics/core#100): a
//! file rewritten one batch per (MT, temperature) loads to the same reactions
//! as the one-batch-per-MT file it came from, and one temperature of one MT
//! splices to a batch carrying that cross section alone.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::{Array, RecordBatch};
use arrow_ipc::reader::{FileReader, StreamReader};
use yamc_convert::reaction_ranges::{
    index_reactions, splice_stream, write_reaction_ranges, Range, ReactionRanges,
};
use yamc_convert::reactions::rewrite_per_temperature;

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

fn mt_of(batch: &RecordBatch) -> i32 {
    batch
        .column_by_name("mt")
        .unwrap()
        .as_primitive::<Int32Type>()
        .value(0)
}

/// The temperature labels a one-row batch carries, as the file spells them.
fn temperatures_of(batch: &RecordBatch) -> Vec<String> {
    let list = batch
        .column_by_name("xs_temperatures")
        .unwrap()
        .as_list::<i32>()
        .value(0);
    let list = list.as_string::<i32>();
    (0..list.len()).map(|i| list.value(i).to_string()).collect()
}

/// The cross section a one-row batch carries at its `i`th temperature.
fn xs_of(batch: &RecordBatch, i: usize) -> Vec<f64> {
    let outer = batch
        .column_by_name("xs_values")
        .unwrap()
        .as_list::<i32>()
        .value(0);
    let inner = outer.as_list::<i32>().value(i);
    inner.as_primitive::<Float64Type>().values().to_vec()
}

/// Every batch of the file keyed by (MT, temperature). A batch carrying several
/// temperatures appears under each, which is how the index lists it too.
fn batches_by_key(path: &Path) -> BTreeMap<(i32, String), RecordBatch> {
    let reader = FileReader::try_new(std::fs::File::open(path).unwrap(), None).unwrap();
    let mut out = BTreeMap::new();
    for batch in reader {
        let batch = batch.unwrap();
        let mt = mt_of(&batch);
        for t in temperatures_of(&batch) {
            out.insert((mt, t), batch.clone());
        }
    }
    out
}

/// The distinct ranges of one MT in file order: one per temperature for a file
/// written per (MT, temperature), one in all for a file written per MT.
fn ranges_of(index: &ReactionRanges, mt: i32) -> Vec<Range> {
    let mut r: Vec<Range> = index.mts[&mt].values().copied().collect();
    r.sort_unstable();
    r.dedup();
    r
}

/// Read the exact bytes an HTTP range request would return.
fn range(bytes: &[u8], (offset, len): Range) -> Vec<u8> {
    let at = offset as usize;
    bytes[at..at + len as usize].to_vec()
}

fn read_stream(stream: &[u8]) -> Vec<RecordBatch> {
    StreamReader::try_new(stream, None)
        .unwrap()
        .map(Result::unwrap)
        .collect()
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
    let whole = batches_by_key(&path);

    assert_eq!(
        index
            .mts
            .iter()
            .flat_map(|(mt, by_t)| by_t.keys().map(move |t| (*mt, t.clone())))
            .collect::<Vec<_>>(),
        whole.keys().cloned().collect::<Vec<_>>(),
        "the index must name exactly the (MT, temperature) pairs the file holds",
    );

    // Every cross section, one at a time: this is the isotope-plotter case, and
    // the strictest form of the claim.
    let schema = range(&bytes, index.schema);
    for (mt, by_temperature) in &index.mts {
        for (t, r) in by_temperature {
            let stream = splice_stream(&schema, &[range(&bytes, *r)]);
            let got = read_stream(&stream);
            assert_eq!(
                got.len(),
                1,
                "MT {mt} at {t} spliced to more than one batch"
            );
            assert_eq!(
                &got[0],
                &whole[&(*mt, t.clone())],
                "MT {mt} at {t} differs from the whole-file read"
            );
        }
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
    let whole = batches_by_key(&path);

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

    // Every temperature of every wanted MT, in file order, paired with the batch
    // the whole-file read holds for it.
    let mut expected: Vec<(Range, RecordBatch)> = Vec::new();
    for mt in &wanted {
        for (t, r) in &index.mts[mt] {
            if !expected.iter().any(|(seen, _)| seen == r) {
                expected.push((*r, whole[&(*mt, t.clone())].clone()));
            }
        }
    }
    expected.sort_by_key(|(r, _)| *r);

    let schema = range(&bytes, index.schema);
    let fetched: Vec<Vec<u8>> = expected.iter().map(|(r, _)| range(&bytes, *r)).collect();
    let fetched_bytes: usize = fetched.iter().map(Vec::len).sum();

    let got = read_stream(&splice_stream(&schema, &fetched));
    assert_eq!(got.len(), expected.len());
    for (batch, (_, want)) in got.iter().zip(&expected) {
        assert_eq!(
            batch,
            want,
            "MT {} differs from the whole-file read",
            mt_of(want)
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
    let mut mts: BTreeMap<i32, BTreeMap<String, Range>> = BTreeMap::new();
    mts.entry(16)
        .or_default()
        .insert("294K".to_string(), (2451328, 7770));
    mts.entry(16)
        .or_default()
        .insert("600K".to_string(), (2459098, 7770));
    mts.entry(102)
        .or_default()
        .insert("294K".to_string(), (19923456, 1363869));
    let ranges = ReactionRanges {
        schema: (8, 1176),
        mts,
    };
    assert_eq!(ReactionRanges::from_json(&ranges.to_json()), Some(ranges));
}

#[test]
fn absent_index_is_not_an_error() {
    // What every already-published library looks like until it is reindexed.
    let version = serde_json::json!({"format_version": 1, "library": "endf-b8.1"});
    assert_eq!(ReactionRanges::from_json(&version), None);
}

/// A `reactions.arrow` row: one MT at one temperature, the columns the section
/// schema declares.
fn reaction_row(mt: i32, temperature: &str, xs: &[f64]) -> Vec<arrow_array::ArrayRef> {
    use yamc_convert::sections::{
        bools, float_list_list, floats, int_list, ints, string_list, strings,
    };
    vec![
        ints(&[mt]),
        strings(&[format!("MT{mt}")]),
        floats(&[0.0]),
        bools(&[false]),
        bools(&[false]),
        string_list(&[temperature.to_string()]),
        float_list_list(&[xs.to_vec()]),
        int_list(&[0]),
    ]
}

/// The index has to work on files this crate writes, not just on the published
/// ones the fixture happens to be.
///
/// The fixture is a downloaded object, written one batch per MT with every
/// temperature inside, whose schema message begins immediately after the file
/// magic. This crate writes one batch per (MT, temperature) with arrow-rs,
/// which pads the schema to a 64-byte boundary. Taking the schema to start at
/// the end of the magic reads that padding as a legacy frame of length zero,
/// records a 4-byte schema, and yields a stream no reader can decode, since the
/// LZ4 codec is declared in the schema message.
#[test]
fn a_file_this_crate_wrote_indexes_and_splices() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reactions.arrow");
    yamc_convert::sections::write_section_per_row(
        &path,
        "reactions.arrow",
        vec![
            reaction_row(1, "294K", &[1.0, 2.0, 3.0, 4.0]),
            reaction_row(2, "294K", &[1.5, 2.5, 3.5, 4.5]),
            reaction_row(2, "600K", &[1.6, 2.6, 3.6, 4.6]),
            reaction_row(102, "294K", &[0.1, 0.2, 0.3, 0.4]),
        ],
    )
    .unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let index = index_reactions(&path).unwrap();

    // The schema is where the writer actually put it, and long enough to be one.
    let (offset, len) = index.schema;
    assert!(
        offset >= 8 && len > 8,
        "schema range {index:?} is not a message: a 4-byte length means the \
         padding was read as the frame",
    );
    assert_eq!(
        index.mts.keys().copied().collect::<Vec<_>>(),
        vec![1, 2, 102]
    );
    assert_eq!(
        index.mts[&2].keys().collect::<Vec<_>>(),
        vec!["294K", "600K"],
        "each temperature of MT 2 is its own batch"
    );
    assert_ne!(index.mts[&2]["294K"], index.mts[&2]["600K"]);

    let whole = batches_by_key(&path);
    for (mt, by_temperature) in &index.mts {
        for (t, r) in by_temperature {
            let spliced = splice_stream(&range(&bytes, index.schema), &[range(&bytes, *r)]);
            let got = read_stream(&spliced);
            assert_eq!(got.len(), 1);
            assert_eq!(
                &got[0],
                &whole[&(*mt, t.clone())],
                "MT {mt} at {t} differs from the whole-file read"
            );
            assert_eq!(temperatures_of(&got[0]), vec![t.clone()]);
        }
    }
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
    let mut ranges: Vec<Range> = wanted
        .iter()
        .flat_map(|mt| ranges_of(&index, *mt))
        .collect();
    ranges.sort_unstable();
    let fetched: Vec<Vec<u8>> = ranges.iter().map(|r| range(&bytes, *r)).collect();
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

    let batches: BTreeSet<Range> = index
        .mts
        .values()
        .flat_map(|by_t| by_t.values().copied())
        .collect();
    assert_eq!(
        whole.num_rows(),
        batches.len(),
        "one row per batch is what indexing by row below assumes",
    );

    let mts = |batch: &RecordBatch| -> Vec<i32> {
        batch
            .column_by_name("mt")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .to_vec()
    };
    let whole_mts = mts(&whole);
    // The rows of the whole file that carry a wanted MT, in file order, are
    // exactly the rows the splice must produce, in that order.
    let whole_rows: Vec<usize> = whole_mts
        .iter()
        .enumerate()
        .filter(|(_, mt)| wanted.contains(mt))
        .map(|(row, _)| row)
        .collect();
    assert_eq!(got.num_rows(), whole_rows.len());
    assert_eq!(
        mts(&got).into_iter().collect::<BTreeSet<_>>(),
        wanted.iter().copied().collect::<BTreeSet<_>>(),
        "spliced batch holds the MTs asked for"
    );

    // Row for row against the whole-file read, which is the only comparison
    // that catches a stream that decodes into plausible but different numbers.
    for (row, at) in whole_rows.iter().enumerate() {
        assert_eq!(
            got.slice(row, 1),
            whole.slice(*at, 1),
            "MT {} differs from the whole-file read",
            whole_mts[*at],
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

/// A copy of the fixture with its `reactions.arrow` rewritten one batch per
/// (MT, temperature) and reindexed: what `split_reactions` does to a published
/// folder. Only the two sections an activation load reads are copied.
fn split_copy_of(dir: &Path) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    for section in ["nuclide.arrow", "version.json"] {
        std::fs::copy(dir.join(section), tmp.path().join(section)).unwrap();
    }
    rewrite_per_temperature(
        &dir.join("reactions.arrow"),
        &tmp.path().join("reactions.arrow"),
    )
    .unwrap();
    assert!(write_reaction_ranges(tmp.path()).unwrap());
    tmp
}

/// The reader claim of fusion-neutronics/core#100: the loader selects its
/// temperature by searching each row's list and skipping rows that do not
/// carry it, so a file rewritten one row per (MT, temperature) loads to the
/// same reactions, at every temperature, as the one-row-per-MT file it came
/// from. No reader change is involved, which is what makes the rewrite safe
/// to publish.
#[test]
fn a_per_temperature_rewrite_loads_the_same_reactions() {
    let Some(dir) = fixture("Fe56") else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    let split = split_copy_of(&dir);

    let original = index_reactions(&dir.join("reactions.arrow")).unwrap();
    let rewritten = index_reactions(&split.path().join("reactions.arrow")).unwrap();
    assert_eq!(
        original.mts.keys().collect::<Vec<_>>(),
        rewritten.mts.keys().collect::<Vec<_>>(),
        "the same MTs"
    );
    for (mt, by_temperature) in &rewritten.mts {
        assert_eq!(
            by_temperature.keys().collect::<Vec<_>>(),
            original.mts[mt].keys().collect::<Vec<_>>(),
            "MT {mt}: the same temperatures"
        );
        let distinct: BTreeSet<Range> = by_temperature.values().copied().collect();
        assert_eq!(
            distinct.len(),
            by_temperature.len(),
            "MT {mt}: every temperature is its own batch after the rewrite"
        );
    }
    let before = std::fs::metadata(dir.join("reactions.arrow"))
        .unwrap()
        .len();
    let after = std::fs::metadata(split.path().join("reactions.arrow"))
        .unwrap()
        .len();
    eprintln!(
        "Fe56 reactions.arrow: {before} bytes in {} batches before, {after} bytes in {} batches \
         after ({:+.1}%)",
        original.mts.len(),
        rewritten.mts.values().map(|by_t| by_t.len()).sum::<usize>(),
        100.0 * (after as f64 - before as f64) / before as f64,
    );

    // Every MT at every temperature, through the real loader.
    let every_mt: HashSet<i32> = (1..1000).collect();
    let scope = yamc_nuclide::LoadScope::activation(every_mt);
    let load =
        |path: &Path| yamc_nuclide::nuclide_loader::load_nuclide(path, &scope).expect("loads");
    let from_original = load(&dir);
    let from_split = load(split.path());
    let temperatures = from_original
        .temperatures()
        .expect("the fixture names its temperatures");
    assert_eq!(Some(temperatures.clone()), from_split.temperatures());
    assert!(
        temperatures.len() > 1,
        "the fixture carries several temperatures"
    );
    let mut compared = 0usize;
    let mut compared_temperatures = 0usize;
    for t in &temperatures {
        // `temperatures()` also names grids that carry no reactions (the 0 K
        // grid the NJOY route leaves behind); only the loaded ones have rows.
        let (Some(ia), Some(ib)) = (from_original.get_temp_idx(t), from_split.get_temp_idx(t))
        else {
            continue;
        };
        compared_temperatures += 1;
        let a = &from_original.reactions[ia];
        let b = &from_split.reactions[ib];
        assert_eq!(
            a.keys().collect::<BTreeSet<_>>(),
            b.keys().collect::<BTreeSet<_>>(),
            "{t} K: the same MTs load"
        );
        for (mt, rx) in a {
            let other = &b[mt];
            assert_eq!(rx.threshold_idx, other.threshold_idx, "MT {mt} at {t} K");
            assert_eq!(
                rx.cross_section.to_vec(),
                other.cross_section.to_vec(),
                "MT {mt} at {t} K differs between the two layouts"
            );
            compared += 1;
        }
    }
    assert!(
        compared_temperatures > 1,
        "compared only {compared_temperatures} temperatures"
    );
    assert!(compared > 100, "compared only {compared} cross sections");
}

/// The client claim of fusion-neutronics/core#100: one temperature of one MT
/// is one batch, and it carries that cross section and nothing else.
#[test]
fn one_temperature_of_one_mt_splices_to_that_cross_section_alone() {
    let Some(dir) = fixture("Fe56") else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };
    let split = split_copy_of(&dir);
    let split_path = split.path().join("reactions.arrow");
    let bytes = std::fs::read(&split_path).unwrap();
    let index = index_reactions(&split_path).unwrap();

    // What the rewrite was copied from, read whole: the reference values.
    let original = batches_by_key(&dir.join("reactions.arrow"));

    let spans = index.spans_where(|mt, t| mt == 102 && t == "294K");
    let fetched: usize = spans.iter().map(|(_, len)| *len as usize).sum();
    let bodies: Vec<Vec<u8>> = spans.iter().map(|s| range(&bytes, *s)).collect();
    let got = read_stream(&nuclear_data_schema::reaction_ranges::splice_spans(&bodies));
    assert_eq!(got.len(), 1, "one batch");
    assert_eq!(mt_of(&got[0]), 102);
    assert_eq!(temperatures_of(&got[0]), vec!["294K".to_string()]);

    let reference = &original[&(102, "294K".to_string())];
    let at = temperatures_of(reference)
        .iter()
        .position(|t| t == "294K")
        .unwrap();
    assert_eq!(
        xs_of(&got[0], 0),
        xs_of(reference, at),
        "MT 102 at 294K differs from the whole-file value"
    );
    eprintln!(
        "MT 102 at 294K: {fetched} of {} bytes ({:.0}x less)",
        bytes.len(),
        bytes.len() as f64 / fetched as f64
    );
    assert!(
        fetched * 20 < bytes.len(),
        "one cross section should be a small fraction of the file"
    );
}
