//! Read `covariance.arrow` back into the blocks the parser produced.
//!
//! The inverse of `yamc-convert`'s writer, and deliberately as dumb as it is:
//! every column maps to one field, nothing is combined, and no matrix is built.
//! Interpreting a block is [`crate::covariance::expand`]'s job.
//!
//! # Absence is normal
//!
//! Every `{Nuclide}.arrow/` published before this section existed has no such
//! file, and most evaluations have no MF=33 at all, so a missing file reads as
//! "no covariance" and never as an error. The `.absent` markers a downloaded
//! directory may carry play no part: those are a download-cache record of a
//! settled 404 and this reader does not consult them, exactly as the rest of
//! the loader does not.

use std::error::Error;
use std::path::Path;

use arrow_array::Array;
use endf::mf::covariance::{NcSubsection, NiSubsection};

use crate::arrow::arrow_helpers::{
    get_str, read_arrow_file, try_get_f64, try_get_f64_list, try_get_i32, try_get_str,
};
use crate::covariance::{BranchingCovarianceBlock, CovarianceBlock, CovarianceData};

/// A nullable `int32` column read as the parser's `i64`, with null as zero.
///
/// Null and zero mean the same thing here by construction: the writer nulls a
/// column only when the row's variant does not use it, and the parser leaves
/// exactly those fields at their zero default. Reading null as zero is
/// therefore what makes the round trip exact rather than a tolerance.
fn int_or_zero(batch: &arrow_array::RecordBatch, col: &str, row: usize) -> i64 {
    try_get_i32(batch, col, row).unwrap_or(0) as i64
}

/// A nullable `double` column, with null as zero. Same reasoning as above.
fn float_or_zero(batch: &arrow_array::RecordBatch, col: &str, row: usize) -> f64 {
    try_get_f64(batch, col, row).unwrap_or(0.0)
}

fn ni_from_row(batch: &arrow_array::RecordBatch, row: usize) -> NiSubsection {
    NiSubsection {
        lt: int_or_zero(batch, "lt", row),
        ls: int_or_zero(batch, "ls", row),
        lb: int_or_zero(batch, "lb", row),
        nt: int_or_zero(batch, "nt", row),
        np: int_or_zero(batch, "np", row),
        ne: int_or_zero(batch, "ne", row),
        ner: int_or_zero(batch, "ner", row),
        nec: int_or_zero(batch, "nec", row),
        ek: try_get_f64_list(batch, "ek", row),
        fk: try_get_f64_list(batch, "fk", row),
        el: try_get_f64_list(batch, "el", row),
        fl: try_get_f64_list(batch, "fl", row),
        fkk: try_get_f64_list(batch, "fkk", row),
        er: try_get_f64_list(batch, "er", row),
        ec: try_get_f64_list(batch, "ec", row),
        fkl: try_get_f64_list(batch, "fkl", row),
    }
}

fn nc_from_row(batch: &arrow_array::RecordBatch, row: usize) -> NcSubsection {
    NcSubsection {
        lty: int_or_zero(batch, "lty", row),
        e1: float_or_zero(batch, "e1", row),
        e2: float_or_zero(batch, "e2", row),
        nci: int_or_zero(batch, "nci", row),
        ci: try_get_f64_list(batch, "ci", row),
        xmti: try_get_f64_list(batch, "xmti", row),
        mats: int_or_zero(batch, "mats", row),
        mts: int_or_zero(batch, "mts", row),
        nei: int_or_zero(batch, "nei", row),
        xmfs: float_or_zero(batch, "xmfs", row),
        xlfss: float_or_zero(batch, "xlfss", row),
        ei: try_get_f64_list(batch, "ei", row),
        wei: try_get_f64_list(batch, "wei", row),
    }
}

/// Read `covariance.arrow` from a `{Nuclide}.arrow/` directory.
///
/// `Ok(None)` when the file is not there. A file that IS there and cannot be
/// read is an error: a caller who asked for uncertainty and would otherwise get
/// silence needs to know the difference between "this evaluation has no
/// covariance" and "the covariance could not be read".
pub fn read_covariance(
    dir: &Path,
    nuclide: &str,
) -> Result<Option<Vec<CovarianceBlock>>, Box<dyn Error>> {
    let path = dir.join("covariance.arrow");
    if !crate::storage::exists(&path) {
        return Ok(None);
    }
    let batch = read_arrow_file(&path)?;

    let mut blocks = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        blocks.push(block_from_row(
            &batch,
            row,
            &format!("{nuclide} covariance.arrow"),
        )?);
    }

    Ok(Some(blocks))
}

/// One row's block, in the columns `covariance.arrow` and
/// `branching_covariance.arrow` share. `what` names the file in errors.
fn block_from_row(
    batch: &arrow_array::RecordBatch,
    row: usize,
    what: &str,
) -> Result<CovarianceBlock, Box<dyn Error>> {
    let kind = get_str(batch, "kind", row)?;
    let data = match kind.as_str() {
        "ni" => CovarianceData::Ni(ni_from_row(batch, row)),
        "nc" => CovarianceData::Nc(nc_from_row(batch, row)),
        other => {
            return Err(format!(
                "{what} row {row}: unknown kind {other:?}; expected \"ni\" or \"nc\""
            )
            .into())
        }
    };
    Ok(CovarianceBlock {
        mt: try_get_i32(batch, "mt", row)
            .ok_or_else(|| format!("{what} row {row}: mt is null, and it is the key"))?,
        subsection_idx: int_or_zero(batch, "subsection_idx", row) as i32,
        block_idx: int_or_zero(batch, "block_idx", row) as i32,
        mat1: int_or_zero(batch, "mat1", row) as i32,
        mt1: int_or_zero(batch, "mt1", row) as i32,
        xmf1: float_or_zero(batch, "xmf1", row),
        xlfs1: float_or_zero(batch, "xlfs1", row),
        mtl: int_or_zero(batch, "mtl", row) as i32,
        // Both files carry the evaluation's own MAT under this name; null, or
        // absent in a file written before the column, reads as 0 (unknown).
        mat: int_or_zero(batch, "mat", row) as i32,
        data,
    })
}

/// A nullable `list<double>` column, with null as `None` rather than empty.
fn optional_list(batch: &arrow_array::RecordBatch, col: &str, row: usize) -> Option<Vec<f64>> {
    let present = batch.column_by_name(col).is_some_and(|c| !c.is_null(row));
    present.then(|| try_get_f64_list(batch, col, row))
}

/// The blocks of `branching/branching_covariance.arrow`, the MF=40 covariance
/// of the isomeric branching partials, from the batches the chain loader holds
/// (`yani::BranchTable::covariance`).
///
/// There is no absent case here: the loader holds batches only when the file
/// was there, and has already checked them against the declared schema. A row
/// whose key is null is an error rather than a skipped row, since a dropped
/// block would read as a covariance the evaluator never stated.
pub fn branching_covariance_blocks(
    batches: &[arrow_array::RecordBatch],
) -> Result<Vec<BranchingCovarianceBlock>, Box<dyn Error>> {
    const WHAT: &str = "branching/branching_covariance.arrow";
    let mut blocks = Vec::with_capacity(batches.iter().map(|b| b.num_rows()).sum());
    for (batch_idx, batch) in batches.iter().enumerate() {
        // Row numbers in errors are within the batch, so the batch is named.
        let what = format!("{WHAT} batch {batch_idx}");
        for row in 0..batch.num_rows() {
            let int = |col: &str| -> Result<i32, Box<dyn Error>> {
                try_get_i32(batch, col, row)
                    .ok_or_else(|| format!("{what} row {row}: {col} is null").into())
            };
            let float = |col: &str| -> Result<f64, Box<dyn Error>> {
                try_get_f64(batch, col, row)
                    .ok_or_else(|| format!("{what} row {row}: {col} is null").into())
            };
            blocks.push(BranchingCovarianceBlock {
                nuclide: try_get_str(batch, "nuclide", row).ok_or_else(|| {
                    format!("{what} row {row}: nuclide is null, and it is the key")
                })?,
                reaction: try_get_str(batch, "reaction", row),
                target: try_get_str(batch, "target", row),
                target1: try_get_str(batch, "target1", row),
                energy: optional_list(batch, "energy", row),
                values: optional_list(batch, "values", row),
                quantity: try_get_str(batch, "quantity", row),
                mat: int("mat")?,
                za: int("za")?,
                awr: float("awr")?,
                lis: int("lis")?,
                state_idx: int("state_idx")?,
                qm: float("qm")?,
                qi: float("qi")?,
                izap: int("izap")?,
                lfs: int("lfs")?,
                block: block_from_row(batch, row, &what)?,
            });
        }
    }
    Ok(blocks)
}
