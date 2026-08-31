//! Shared helper functions for reading Apache Arrow IPC columns.
//!
//! Used by both `nuclide_arrow` and `photon_arrow` modules to avoid duplication.

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::{Array, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow_buffer::ScalarBuffer;
use arrow_ipc::reader::{FileReader, StreamReader};
use arrow_select::concat::concat_batches;

use std::error::Error;
use std::io::{self, Read, Seek};
use std::path::Path;

use crate::storage;

/// Prefix an error with the section it came from.
///
/// Neither the io errors nor the Arrow ones carry a path, so a section that was
/// absent or malformed surfaced as a bare "No such file or directory (os error
/// 2)" or "Io error: Invalid argument (os error 22)" naming nothing, which is
/// what made issue #506 take a bisection to find. Every failure
/// [`read_arrow_file`] can return goes through here, so all of them name the
/// section: opening it, sniffing it, decoding it and fusing it alike.
fn at(path: &Path, e: impl std::fmt::Display) -> String {
    format!("{}: {e}", path.display())
}

/// `"ARROW1"`, the magic an Arrow IPC *file* opens with.
///
/// A *stream* carries neither this nor the footer, and opens straight into a
/// message frame: the v4+ continuation marker `0xffffffff`, or a legacy
/// metadata length. Neither can be read as this, so six bytes tell the two
/// framings apart.
const IPC_FILE_MAGIC: [u8; 6] = *b"ARROW1";

/// Whether `reader` holds an Arrow IPC *file* rather than a *stream*.
///
/// Rewinds either way, so whichever reader the caller then builds starts at the
/// beginning.
///
/// A source too short to hold the magic is called a file, which hands the error
/// to `FileReader` below. That keeps a truncated section reported as the
/// truncated file it is, rather than as a stream decoder complaining about a
/// frame that was never there.
fn is_ipc_file<R: Read + Seek + ?Sized>(reader: &mut R) -> io::Result<bool> {
    let mut magic = [0u8; 6];
    let is_file = match reader.read_exact(&mut magic) {
        Ok(()) => magic == IPC_FILE_MAGIC,
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => true,
        Err(e) => return Err(e),
    };
    reader.rewind()?;
    Ok(is_file)
}

/// Read a single RecordBatch from an Arrow IPC file or stream. Routes through
/// the configured [`storage`] backend so browser hosts can swap in OPFS-backed
/// reads without touching the parser.
///
/// Both framings are accepted. Every published section is a file; a host that
/// fetched only the byte ranges of the MTs it needs and spliced them (the
/// `reaction_ranges` index in `version.json`) has a stream instead, and it must
/// read back as the batches the whole file would have given. The browser is
/// where that path is taken: it fetches in JS and hands the bytes over, so the
/// splice happens before this crate ever sees them.
pub fn read_arrow_file(path: &Path) -> Result<RecordBatch, Box<dyn Error>> {
    let mut file = storage::open_read(path).map_err(|e| at(path, e))?;
    let is_file = is_ipc_file(&mut file).map_err(|e| at(path, e))?;
    let (schema, batches): (_, Vec<RecordBatch>) = if is_file {
        let reader = FileReader::try_new(file, None).map_err(|e| at(path, e))?;
        (
            reader.schema(),
            reader
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| at(path, e))?,
        )
    } else {
        let reader = StreamReader::try_new(file, None).map_err(|e| at(path, e))?;
        (
            reader.schema(),
            reader
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| at(path, e))?,
        )
    };
    if batches.is_empty() {
        // A published file with no batches is damage, and stays an error.
        if is_file {
            return Err(format!("No record batches in {}", path.display()).into());
        }
        // A stream with none is a legitimately empty subset: the caller fetched
        // the byte ranges of the MTs it wanted and this nuclide publishes none
        // of them, so the splice is the schema message and nothing after it.
        // He4 is the real case, in both endf-b8.1 and jendl-5.0: it carries only
        // elastic and the transport lookups, and no activation channel at all.
        // Reading the whole file would have produced these same zero reactions
        // once the MT filter ran, so an empty batch is what keeps the two paths
        // agreeing rather than a special case.
        return Ok(RecordBatch::new_empty(schema));
    }
    if batches.len() == 1 {
        let batch = batches.into_iter().next().unwrap();
        check_declared(path, &batch)?;
        return Ok(batch);
    }
    // Multi-batch sources (the option-D batch-per-MT reactions.arrow, and every
    // spliced stream) fuse into one batch so the loader keeps its single
    // unified row space.
    let fused = concat_batches(&schema, &batches).map_err(|e| at(path, e))?;
    check_declared(path, &fused)?;
    Ok(fused)
}

/// Check a batch against the schema declared for the section at `path`.
///
/// One call here covers every section this crate reads, and turns a
/// writer/reader disagreement into an error naming the file and the column
/// rather than a downcast panic somewhere further in (issue #126).
fn check_declared(path: &Path, batch: &RecordBatch) -> Result<(), Box<dyn Error>> {
    if let Some(section) = nuclear_data_schema::flat_section_for_path(path) {
        nuclear_data_schema::check_batch(&section, batch.schema().as_ref())
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

fn column<'a>(
    batch: &'a RecordBatch,
    col: &str,
) -> Result<&'a std::sync::Arc<dyn Array>, Box<dyn Error>> {
    batch
        .column_by_name(col)
        .ok_or_else(|| format!("Missing column: {col}").into())
}

fn check_row(col: &str, len: usize, row: usize) -> Result<(), Box<dyn Error>> {
    if row >= len {
        return Err(format!("Row {row} out of bounds for column {col} (len {len})").into());
    }
    Ok(())
}

pub fn get_f64(batch: &RecordBatch, col: &str, row: usize) -> Result<f64, Box<dyn Error>> {
    let arr = column(batch, col)?;
    let typed = arr
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| format!("Column {col} is not Float64 (got {:?})", arr.data_type()))?;
    check_row(col, typed.len(), row)?;
    Ok(typed.value(row))
}

pub fn get_i32(batch: &RecordBatch, col: &str, row: usize) -> Result<i32, Box<dyn Error>> {
    let arr = column(batch, col)?;
    let typed = arr
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| format!("Column {col} is not Int32 (got {:?})", arr.data_type()))?;
    check_row(col, typed.len(), row)?;
    Ok(typed.value(row))
}

pub fn get_str(batch: &RecordBatch, col: &str, row: usize) -> Result<String, Box<dyn Error>> {
    let arr = column(batch, col)?;
    let typed = arr.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        format!(
            "Column {col} is not Utf8/String (got {:?})",
            arr.data_type()
        )
    })?;
    check_row(col, typed.len(), row)?;
    Ok(typed.value(row).to_string())
}

pub fn get_bool(batch: &RecordBatch, col: &str, row: usize) -> Result<bool, Box<dyn Error>> {
    let arr = column(batch, col)?;
    let typed = arr
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| format!("Column {col} is not Boolean (got {:?})", arr.data_type()))?;
    check_row(col, typed.len(), row)?;
    Ok(typed.value(row))
}

pub fn get_f64_list(
    batch: &RecordBatch,
    col: &str,
    row: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let arr = batch
        .column_by_name(col)
        .ok_or_else(|| format!("Missing column: {col}"))?;
    let list_arr = arr.as_list::<i32>();
    if list_arr.is_null(row) {
        return Ok(Vec::new());
    }
    let values = list_arr.value(row);
    let f64_arr = values.as_primitive::<Float64Type>();
    Ok(f64_arr.values().iter().copied().collect())
}

/// Zero-copy view of one `List<Float64>` cell (issue #476, task 1).
///
/// Where [`get_f64_list`] allocates a `Vec` and copies the cell into it, this
/// hands back the Arrow values buffer itself, offset and trimmed to the cell.
/// Cloning or slicing the result is O(1).
///
/// The view holds an `Arc` on the column's *entire* values buffer, not just the
/// cell, so a caller that keeps one cell of a wide column and drops the batch
/// pins the rest of the column with it. Callers that only want a few cells
/// should copy instead; see `LoadScope::is_unfiltered`.
pub fn borrow_f64_list(
    batch: &RecordBatch,
    col: &str,
    row: usize,
) -> Result<ScalarBuffer<f64>, Box<dyn Error>> {
    let arr = batch
        .column_by_name(col)
        .ok_or_else(|| format!("Missing column: {col}"))?;
    let list_arr = arr.as_list::<i32>();
    if list_arr.is_null(row) {
        return Ok(Vec::new().into());
    }
    let values = list_arr.value(row);
    Ok(values.as_primitive::<Float64Type>().values().clone())
}

/// Zero-copy view of one `List<Int32>` cell, the `int32` twin of
/// [`borrow_f64_list`] (issue #482).
///
/// Same sharing caveat: the view keeps the column's entire values buffer alive.
/// Used where a cell is converted on the way in rather than kept as `i32`, so
/// the intermediate `Vec<i32>` that [`get_i32_list`] would allocate is skipped.
pub fn borrow_i32_list(
    batch: &RecordBatch,
    col: &str,
    row: usize,
) -> Result<ScalarBuffer<i32>, Box<dyn Error>> {
    let arr = batch
        .column_by_name(col)
        .ok_or_else(|| format!("Missing column: {col}"))?;
    let list_arr = arr.as_list::<i32>();
    if list_arr.is_null(row) {
        return Ok(Vec::new().into());
    }
    let values = list_arr.value(row);
    Ok(values.as_primitive::<Int32Type>().values().clone())
}

/// Zero-copy views of the inner lists of one `List<List<Float64>>` cell.
///
/// Same sharing caveat as [`borrow_f64_list`]: every returned view keeps the
/// whole column alive.
pub fn borrow_nested_f64_list(
    batch: &RecordBatch,
    col: &str,
    row: usize,
) -> Result<Vec<ScalarBuffer<f64>>, Box<dyn Error>> {
    let arr = batch
        .column_by_name(col)
        .ok_or_else(|| format!("Missing column: {col}"))?;
    let outer_list = arr.as_list::<i32>();
    if outer_list.is_null(row) {
        return Ok(Vec::new());
    }
    let outer_values = outer_list.value(row);
    let inner_list = outer_values.as_list::<i32>();
    Ok((0..inner_list.len())
        .map(|i| {
            let inner_values = inner_list.value(i);
            inner_values.as_primitive::<Float64Type>().values().clone()
        })
        .collect())
}

pub fn get_i32_list(
    batch: &RecordBatch,
    col: &str,
    row: usize,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let arr = batch
        .column_by_name(col)
        .ok_or_else(|| format!("Missing column: {col}"))?;
    let list_arr = arr.as_list::<i32>();
    if list_arr.is_null(row) {
        return Ok(Vec::new());
    }
    let values = list_arr.value(row);
    let i32_arr = values.as_primitive::<Int32Type>();
    Ok(i32_arr.values().iter().copied().collect())
}

pub fn get_str_list(
    batch: &RecordBatch,
    col: &str,
    row: usize,
) -> Result<Vec<String>, Box<dyn Error>> {
    let arr = batch
        .column_by_name(col)
        .ok_or_else(|| format!("Missing column: {col}"))?;
    let list_arr = arr.as_list::<i32>();
    if list_arr.is_null(row) {
        return Ok(Vec::new());
    }
    let values = list_arr.value(row);
    let str_arr = values.as_string::<i32>();
    Ok((0..str_arr.len())
        .map(|i| str_arr.value(i).to_string())
        .collect())
}

pub fn try_get_f64(batch: &RecordBatch, col: &str, row: usize) -> Option<f64> {
    let arr = batch.column_by_name(col)?;
    if arr.is_null(row) {
        return None;
    }
    Some(arr.as_primitive::<Float64Type>().value(row))
}

pub fn try_get_i32(batch: &RecordBatch, col: &str, row: usize) -> Option<i32> {
    let arr = batch.column_by_name(col)?;
    if arr.is_null(row) {
        return None;
    }
    Some(arr.as_primitive::<Int32Type>().value(row))
}

pub fn try_get_str(batch: &RecordBatch, col: &str, row: usize) -> Option<String> {
    let arr = batch.column_by_name(col)?;
    if arr.is_null(row) {
        return None;
    }
    Some(arr.as_string::<i32>().value(row).to_string())
}

pub fn try_get_f64_list(batch: &RecordBatch, col: &str, row: usize) -> Vec<f64> {
    get_f64_list(batch, col, row).unwrap_or_default()
}

pub fn try_get_i32_list(batch: &RecordBatch, col: &str, row: usize) -> Vec<i32> {
    get_i32_list(batch, col, row).unwrap_or_default()
}
