//! `covariance.arrow` must carry MF=33 out of the parser without losing any of
//! it.
//!
//! The writer is a faithful dump, so the test that matters is a round trip:
//! parse an evaluation, write the section, read the file back, and rebuild the
//! parser's own structures from the columns. Anything the writer drops,
//! reorders or reshapes shows up as a mismatched field rather than as a plausible
//! matrix that happens to be wrong.
//!
//! This is deliberately not a comparison against a hand-written expected table.
//! The point of the section is that it is the tape's own numbers in the tape's
//! own order, and the parser is what defines that; a second transcription of
//! the same values by hand would only test the transcription.

use std::collections::BTreeMap;
use std::path::Path;

use arrow_array::{Array, Float64Array, Int32Array, ListArray, RecordBatch, StringArray};
use endf::mf::covariance::{NcSubsection, NiSubsection};
use endf::Material;

/// Li6 and Fe56 both carry MF=33; Li6 is the smaller of the two.
const LI6_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-003_Li_006_trimmed.endf.xz");
const FE56_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-026_Fe_056_trimmed.endf.xz");
/// In115 has no MF=33 at all, which is what "absent" has to be tested against.
const IN115_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-049_In-115_trimmed.endf.xz");

fn material(compressed: &[u8], dir: &Path, name: &str) -> Material {
    let path = dir.join(format!("{name}.endf"));
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut raw).expect("fixture decompresses");
    std::fs::write(&path, raw).expect("fixture writes");
    Material::from_file(&path).expect("evaluation parses")
}

/// Read the whole section back as one batch.
///
/// The writer emits a single batch, so concatenating would be a no-op; this
/// asserts that rather than quietly tolerating a split, since a reader that
/// takes only the first batch of a multi-batch file would silently lose rows.
fn read_back(path: &Path) -> RecordBatch {
    let file = std::fs::File::open(path).expect("section opens");
    let reader = arrow_ipc::reader::FileReader::try_new(file, None).expect("section reads");
    let batches: Vec<RecordBatch> = reader.map(|b| b.expect("batch decodes")).collect();
    assert_eq!(batches.len(), 1, "covariance.arrow is written as one batch");
    batches.into_iter().next().unwrap()
}

fn ints(batch: &RecordBatch, name: &str) -> Int32Array {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("no column {name}"))
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap_or_else(|| panic!("{name} is not int32"))
        .clone()
}

fn floats(batch: &RecordBatch, name: &str) -> Float64Array {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("no column {name}"))
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap_or_else(|| panic!("{name} is not double"))
        .clone()
}

/// One row of a `list<double>` column, with null read as the empty list.
///
/// The writer stores an empty list as null so the file matches the "everything
/// not belonging to a row's variant is null" rule the schema states, and the
/// parser's own empty `Vec` is what that means on the way back in.
fn list_row(batch: &RecordBatch, name: &str, row: usize) -> Vec<f64> {
    let col = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("no column {name}"))
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap_or_else(|| panic!("{name} is not list<double>"));
    if col.is_null(row) {
        return Vec::new();
    }
    let values = col.value(row);
    values
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("list of double")
        .values()
        .to_vec()
}

fn int_row(batch: &RecordBatch, name: &str, row: usize) -> i64 {
    let col = ints(batch, name);
    assert!(!col.is_null(row), "{name} is null on a row that uses it");
    col.value(row) as i64
}

fn float_row(batch: &RecordBatch, name: &str, row: usize) -> f64 {
    let col = floats(batch, name);
    assert!(!col.is_null(row), "{name} is null on a row that uses it");
    col.value(row)
}

/// Rebuild an `NiSubsection` from the row that was written for it.
fn ni_from_row(batch: &RecordBatch, row: usize) -> NiSubsection {
    NiSubsection {
        lt: int_row(batch, "lt", row),
        ls: int_row(batch, "ls", row),
        lb: int_row(batch, "lb", row),
        nt: int_row(batch, "nt", row),
        np: int_row(batch, "np", row),
        ne: int_row(batch, "ne", row),
        ner: int_row(batch, "ner", row),
        nec: int_row(batch, "nec", row),
        ek: list_row(batch, "ek", row),
        fk: list_row(batch, "fk", row),
        el: list_row(batch, "el", row),
        fl: list_row(batch, "fl", row),
        fkk: list_row(batch, "fkk", row),
        er: list_row(batch, "er", row),
        ec: list_row(batch, "ec", row),
        fkl: list_row(batch, "fkl", row),
    }
}

/// Rebuild an `NcSubsection` from the row that was written for it.
fn nc_from_row(batch: &RecordBatch, row: usize) -> NcSubsection {
    NcSubsection {
        lty: int_row(batch, "lty", row),
        e1: float_row(batch, "e1", row),
        e2: float_row(batch, "e2", row),
        nci: int_row(batch, "nci", row),
        ci: list_row(batch, "ci", row),
        xmti: list_row(batch, "xmti", row),
        mats: int_row(batch, "mats", row),
        mts: int_row(batch, "mts", row),
        nei: int_row(batch, "nei", row),
        xmfs: float_row(batch, "xmfs", row),
        xlfss: float_row(batch, "xlfss", row),
        ei: list_row(batch, "ei", row),
        wei: list_row(batch, "wei", row),
    }
}

fn kinds(batch: &RecordBatch) -> StringArray {
    batch
        .column_by_name("kind")
        .expect("no column kind")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("kind is not string")
        .clone()
}

/// Parse, write, read back, and compare every block against the parser.
fn round_trip(compressed: &[u8], name: &str, expect_mts: &[i32]) {
    let tmp = tempfile::tempdir().expect("temp dir");
    let material = material(compressed, tmp.path(), name);

    let written = yamc_convert::covariance::write_covariance(&material, tmp.path())
        .expect("covariance writes");
    assert!(written, "{name} carries MF=33, so a file must be written");

    let path = tmp.path().join("covariance.arrow");
    let batch = read_back(&path);

    let mt = ints(&batch, "mt");
    let subsection_idx = ints(&batch, "subsection_idx");
    let block_idx = ints(&batch, "block_idx");
    let kind = kinds(&batch);
    let mat1 = ints(&batch, "mat1");
    let mt1 = ints(&batch, "mt1");
    let xmf1 = floats(&batch, "xmf1");
    let xlfs1 = floats(&batch, "xlfs1");
    let mtl = ints(&batch, "mtl");

    // Every row, keyed the way the schema says it is keyed. A duplicate key
    // would mean two blocks collapsed onto one identity, which is the failure
    // the two indices exist to prevent.
    let mut seen: BTreeMap<(i32, i32, i32), usize> = BTreeMap::new();
    for row in 0..batch.num_rows() {
        let key = (
            mt.value(row),
            subsection_idx.value(row),
            block_idx.value(row),
        );
        assert!(
            seen.insert(key, row).is_none(),
            "two rows share (mt, subsection_idx, block_idx) = {key:?}"
        );
    }

    let mut rows_checked = 0;
    let mut mts_seen = Vec::new();
    for mt_value in material
        .sections()
        .into_iter()
        .filter(|(mf, _)| *mf == 33)
        .map(|(_, mt)| mt)
    {
        let mf33 = material.mf33(mt_value).expect("the section parses");
        mts_seen.push(mt_value);
        for (si, sub) in mf33.subsections.iter().enumerate() {
            // Tape order: NC blocks first, then NI, with one running index.
            let expected: Vec<(&str, usize)> = (0..sub.nc_subsections.len())
                .map(|i| ("nc", i))
                .chain((0..sub.ni_subsections.len()).map(|i| ("ni", i)))
                .collect();

            for (bi, (want_kind, within)) in expected.into_iter().enumerate() {
                let key = (mt_value, si as i32, bi as i32);
                let row = *seen
                    .get(&key)
                    .unwrap_or_else(|| panic!("no row written for {key:?}"));

                assert_eq!(kind.value(row), want_kind, "kind at {key:?}");
                assert_eq!(mat1.value(row) as i64, sub.mat1, "mat1 at {key:?}");
                assert_eq!(mt1.value(row) as i64, sub.mt1, "mt1 at {key:?}");
                assert_eq!(xmf1.value(row), sub.xmf1, "xmf1 at {key:?}");
                assert_eq!(xlfs1.value(row), sub.xlfs1, "xlfs1 at {key:?}");
                assert_eq!(mtl.value(row) as i64, mf33.mtl, "mtl at {key:?}");

                match want_kind {
                    "nc" => {
                        assert_eq!(
                            nc_from_row(&batch, row),
                            sub.nc_subsections[within],
                            "NC block at {key:?}"
                        );
                        // The other variant's columns must be null, not zero:
                        // a zero `lb` is a real LB code.
                        for c in ["lb", "ls", "lt", "nt", "np", "ne", "ner", "nec"] {
                            assert!(ints(&batch, c).is_null(row), "{c} is not null at {key:?}");
                        }
                    }
                    _ => {
                        assert_eq!(
                            ni_from_row(&batch, row),
                            sub.ni_subsections[within],
                            "NI block at {key:?}"
                        );
                        for c in ["lty", "nci", "mats", "mts", "nei"] {
                            assert!(ints(&batch, c).is_null(row), "{c} is not null at {key:?}");
                        }
                    }
                }
                rows_checked += 1;
            }
        }
    }

    assert_eq!(
        rows_checked,
        batch.num_rows(),
        "the file has rows the evaluation does not account for"
    );
    assert!(rows_checked > 0, "{name} produced no blocks to compare");
    assert_eq!(mts_seen, expect_mts, "{name} MF=33 MTs");
}

#[test]
fn li6_covariance_round_trips_through_the_section() {
    // MT=105 is (n,t), the channel this trimmed fixture keeps covariance for.
    round_trip(LI6_ENDF, "li6", &[105]);
}

#[test]
fn fe56_covariance_round_trips_through_the_section() {
    // MT=103 is (n,p).
    round_trip(FE56_ENDF, "fe56", &[103]);
}

/// An evaluation with no MF=33 writes no file at all.
///
/// Absence is how this section says "no covariance", and the reader is required
/// to treat a missing file that way rather than as an error. Writing an empty
/// table instead would be a different thing: a positive claim that the
/// evaluation was examined and found to have zero covariance.
#[test]
fn an_evaluation_without_mf33_writes_nothing() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let material = material(IN115_ENDF, tmp.path(), "in115");
    assert!(
        material.mf33(1).is_none(),
        "the fixture must have no MF=33 for this test to mean anything"
    );

    let written =
        yamc_convert::covariance::write_covariance(&material, tmp.path()).expect("no error");
    assert!(!written, "nothing to write");
    assert!(
        !tmp.path().join("covariance.arrow").exists(),
        "no file, not an empty one"
    );
}

/// The section the writer produces is the section the loader reads.
///
/// This lives here rather than in `yamc-nuclide` because the dependency points
/// this way: the converter dev-depends on the loader precisely so a conversion
/// can be judged by its real consumer. Nothing published carries this section
/// yet, so the file under test has to be written first.
#[test]
fn the_loader_reads_back_what_the_converter_writes() {
    use yamc_nuclide::covariance::expand::{expand_ni, Scale};
    use yamc_nuclide::covariance::CovarianceData;

    let tmp = tempfile::tempdir().expect("temp dir");
    let material = material(FE56_ENDF, tmp.path(), "fe56");
    assert!(yamc_convert::covariance::write_covariance(&material, tmp.path()).expect("writes"));

    let blocks = yamc_nuclide::arrow::covariance_arrow::read_covariance(tmp.path(), "Fe56")
        .expect("reads")
        .expect("the file is there");
    assert!(!blocks.is_empty());

    let mut expanded = 0;
    for block in &blocks {
        assert!(
            !block.is_cross_material(),
            "every block in this evaluation is its own material's"
        );
        assert!(
            block.is_diagonal(),
            "the fixture keeps only MT=103 with itself"
        );
        let CovarianceData::Ni(ni) = &block.data else {
            continue;
        };
        assert_eq!(ni.ls, 1, "the fixture is the symmetric-triangle case");

        let matrix = expand_ni(ni).expect("LB=5 expands");
        assert_eq!(matrix.scale, Scale::Relative);
        assert_eq!(matrix.n_rows(), matrix.n_cols(), "LB=5 is square");
        assert_eq!(
            matrix.n_rows(),
            ni.ek.len() - 1,
            "one interval per pair of grid boundaries"
        );

        // LS=1 stores an upper triangle whose transpose is implied, so what
        // comes out must be symmetric even though what went in was not.
        for i in 0..matrix.n_rows() {
            assert!(
                matrix.get(i, i) >= 0.0,
                "a relative variance cannot be negative, at {i}"
            );
            for j in 0..matrix.n_cols() {
                assert_eq!(matrix.get(i, j), matrix.get(j, i), "asymmetric at {i},{j}");
            }
        }
        expanded += 1;
    }
    assert!(expanded > 0, "no NI blocks were expanded");
}
