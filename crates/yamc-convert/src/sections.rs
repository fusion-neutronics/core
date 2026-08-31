//! Arrow section writing, shared by every part of the transport converter.

use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int32Builder, ListBuilder, StringBuilder,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_ipc::writer::{FileWriter, IpcWriteOptions};
use arrow_ipc::CompressionType;
use arrow_schema::{ArrowError, Schema};

/// The declared schema for a section, by its path in the format.
///
/// Panics on an unknown path, which can only be a typo: the schema crate is the
/// whole set of sections this format has.
pub fn section_schema(path: &str) -> Schema {
    nuclear_data_schema::section(path)
        .unwrap_or_else(|| panic!("no declared schema for section {path}"))
}

/// LZ4 for every section this format writes.
///
/// The published libraries have been LZ4 throughout since the Python converter
/// learned to compress, and the reader decompresses transparently, so an
/// uncompressed file is not a different format: it is the same data at roughly
/// 2.4x the size, which is what every client then downloads. The generation
/// scripts' `validate_arrow.py` fails a tree carrying one, which is how this
/// was found after the conversion moved to Rust.
fn section_write_options() -> Result<IpcWriteOptions, ArrowError> {
    IpcWriteOptions::default().try_with_compression(Some(CompressionType::LZ4_FRAME))
}

/// Write one batch to a section file.
pub fn write_section(
    path: &Path,
    section: &str,
    columns: Vec<ArrayRef>,
) -> Result<(), Box<dyn Error>> {
    let schema = Arc::new(section_schema(section));
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let mut writer =
        FileWriter::try_new_with_options(File::create(path)?, &schema, section_write_options()?)?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(())
}

/// Write one record batch PER ROW.
///
/// Only `reactions.arrow` is written this way, so a consumer can range-read a
/// single MT out of the middle of the file without decoding the rest. A single
/// fused batch would read the same through the loader and defeat that.
pub fn write_section_per_row(
    path: &Path,
    section: &str,
    rows: Vec<Vec<ArrayRef>>,
) -> Result<(), Box<dyn Error>> {
    let schema = Arc::new(section_schema(section));
    let mut writer =
        FileWriter::try_new_with_options(File::create(path)?, &schema, section_write_options()?)?;
    for columns in rows {
        writer.write(&RecordBatch::try_new(schema.clone(), columns)?)?;
    }
    writer.finish()?;
    Ok(())
}

pub fn strings(values: &[String]) -> ArrayRef {
    let mut b = StringBuilder::new();
    for v in values {
        b.append_value(v);
    }
    Arc::new(b.finish())
}

pub fn floats(values: &[f64]) -> ArrayRef {
    let mut b = Float64Builder::new();
    b.append_slice(values);
    Arc::new(b.finish())
}

pub fn ints(values: &[i32]) -> ArrayRef {
    let mut b = Int32Builder::new();
    b.append_slice(values);
    Arc::new(b.finish())
}

pub fn bools(values: &[bool]) -> ArrayRef {
    let mut b = BooleanBuilder::new();
    for &v in values {
        b.append_value(v);
    }
    Arc::new(b.finish())
}

/// A single-row `list<utf8>` column.
pub fn string_list(values: &[String]) -> ArrayRef {
    let mut b = ListBuilder::new(StringBuilder::new());
    for v in values {
        b.values().append_value(v);
    }
    b.append(true);
    Arc::new(b.finish())
}

/// A single-row `list<double>` column.
pub fn float_list(values: &[f64]) -> ArrayRef {
    let mut b = ListBuilder::new(Float64Builder::new());
    b.values().append_slice(values);
    b.append(true);
    Arc::new(b.finish())
}

/// A single-row `list<int32>` column.
pub fn int_list(values: &[i32]) -> ArrayRef {
    let mut b = ListBuilder::new(Int32Builder::new());
    b.values().append_slice(values);
    b.append(true);
    Arc::new(b.finish())
}

/// A single-row `list<list<double>>` column.
pub fn float_list_list(rows: &[Vec<f64>]) -> ArrayRef {
    let mut b = ListBuilder::new(ListBuilder::new(Float64Builder::new()));
    for row in rows {
        b.values().values().append_slice(row);
        b.values().append(true);
    }
    b.append(true);
    Arc::new(b.finish())
}

/// A multi-row `list<double>` column, one list per row.
pub fn float_lists(rows: &[Vec<f64>]) -> ArrayRef {
    let mut b = ListBuilder::new(Float64Builder::new());
    for row in rows {
        b.values().append_slice(row);
        b.append(true);
    }
    Arc::new(b.finish())
}

/// A multi-row `list<int32>` column, one list per row.
pub fn int_lists(rows: &[Vec<i32>]) -> ArrayRef {
    let mut b = ListBuilder::new(Int32Builder::new());
    for row in rows {
        b.values().append_slice(row);
        b.append(true);
    }
    Arc::new(b.finish())
}

/// A multi-row `list<utf8>` column, one list per row.
pub fn string_lists(rows: &[Vec<String>]) -> ArrayRef {
    let mut b = ListBuilder::new(StringBuilder::new());
    for row in rows {
        for v in row {
            b.values().append_value(v);
        }
        b.append(true);
    }
    Arc::new(b.finish())
}

/// A multi-row `list<list<double>>` column, one outer list per row.
pub fn float_list_lists(rows: &[Vec<Vec<f64>>]) -> ArrayRef {
    let mut b = ListBuilder::new(ListBuilder::new(Float64Builder::new()));
    for row in rows {
        for inner in row {
            b.values().values().append_slice(inner);
            b.values().append(true);
        }
        b.append(true);
    }
    Arc::new(b.finish())
}

/// A nullable `utf8` column. `None` writes a null, which is what a
/// discriminant-driven schema means by "this row does not use this column".
pub fn opt_strings(values: &[Option<String>]) -> ArrayRef {
    let mut b = StringBuilder::new();
    for v in values {
        match v {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
    }
    Arc::new(b.finish())
}

/// A nullable `double` column.
pub fn opt_floats(values: &[Option<f64>]) -> ArrayRef {
    let mut b = Float64Builder::new();
    for v in values {
        b.append_option(*v);
    }
    Arc::new(b.finish())
}

/// A nullable `int32` column.
pub fn opt_ints(values: &[Option<i32>]) -> ArrayRef {
    let mut b = Int32Builder::new();
    for v in values {
        b.append_option(*v);
    }
    Arc::new(b.finish())
}

/// A `list<double>` column where an EMPTY list is written as null.
///
/// `distributions.arrow` never holds an empty list: a column its row's type
/// does not use is null. Writing `[]` instead loads identically, since the
/// reader treats both as no data, but it is not the same file, which makes a
/// byte comparison against the published data impossible.
pub fn float_lists_or_null(rows: &[Vec<f64>]) -> ArrayRef {
    let mut b = ListBuilder::new(Float64Builder::new());
    for row in rows {
        if row.is_empty() {
            b.append_null();
        } else {
            b.values().append_slice(row);
            b.append(true);
        }
    }
    Arc::new(b.finish())
}

/// A `list<int32>` column where an EMPTY list is written as null.
pub fn int_lists_or_null(rows: &[Vec<i32>]) -> ArrayRef {
    let mut b = ListBuilder::new(Int32Builder::new());
    for row in rows {
        if row.is_empty() {
            b.append_null();
        } else {
            b.values().append_slice(row);
            b.append(true);
        }
    }
    Arc::new(b.finish())
}
