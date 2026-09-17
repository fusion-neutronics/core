//! Byte ranges of each temperature's record batch within `energy.arrow`.
//!
//! The union energy grids are the floor under every other saving
//! (fusion-neutronics/core#100). On U238 they are 6.33 MB, against 8.98 MB for
//! the whole single-temperature JSON bundle the Arrow layout replaced, so no
//! amount of slicing `reactions.arrow` helps a plotter until the grids move
//! too. They used to be two columns of one row of `nuclide.arrow`, which is all
//! or nothing: read the cell and you have read all six temperatures.
//!
//! They are their own section now, written one record batch per temperature, so
//! each grid is already a contiguous byte range in the published object. This
//! records those ranges into `version.json` beside `reaction_ranges`, the same
//! way and for the same reason: no new object is published, and a reader that
//! wants every temperature still issues one plain GET.
//!
//! The Arrow-reading half (walking the footer for the batch offsets) is what
//! lives here; the JSON shape and the span planning are in
//! `nuclear_data_schema::energy_ranges`, so the loader can read an index this
//! crate wrote without depending on the converter.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::Array;
use arrow_ipc::reader::FileReader;

pub use nuclear_data_schema::energy_ranges::EnergyRanges;
use nuclear_data_schema::reaction_ranges::Range;

use crate::reaction_ranges::{message_len, schema_offset};

/// Index the per-temperature record batches of an `energy.arrow`.
///
/// One row per batch, for the same reason `reactions.arrow` needs it: a batch
/// holding two grids would be indexed by its first and fetched whole, so the
/// index would lie about what a range holds.
pub fn index_energy(energy: &Path) -> Result<EnergyRanges, Box<dyn Error>> {
    let bytes = std::fs::read(energy)?;

    // The footer's Block array is the authority on where each batch starts;
    // deriving offsets by re-serializing would only reproduce them if this
    // build wrote the file, which for already-published data it did not.
    let footer_len_at = bytes
        .len()
        .checked_sub(10)
        .ok_or("energy.arrow is too short to hold an Arrow footer")?;
    let footer_len = u32::from_le_bytes(bytes[footer_len_at..footer_len_at + 4].try_into()?);
    let footer_at = footer_len_at
        .checked_sub(usize::try_from(footer_len)?)
        .ok_or("Arrow footer length runs past the start of the file")?;
    let footer = arrow_ipc::root_as_footer(&bytes[footer_at..footer_len_at])
        .map_err(|e| format!("unreadable Arrow footer: {e}"))?;
    let blocks = footer
        .recordBatches()
        .ok_or("Arrow footer names no record batches")?;

    let reader = FileReader::try_new(File::open(energy)?, None)?;
    let mut temperatures: BTreeMap<String, Range> = BTreeMap::new();
    for (i, batch) in reader.enumerate() {
        let batch = batch?;
        let block = blocks.get(i);
        let labels = batch
            .column_by_name("temperature")
            .ok_or("an energy batch has no `temperature` column")?
            .as_string_opt::<i32>()
            .ok_or("`temperature` is not a StringArray")?;
        if labels.len() != 1 {
            return Err(format!(
                "an energy batch carries {} rows; the index needs one row per batch",
                labels.len()
            )
            .into());
        }
        let label = labels.value(0).to_string();
        let len = u64::try_from(block.metaDataLength())? + u64::try_from(block.bodyLength())?;
        if temperatures
            .insert(label.clone(), (u64::try_from(block.offset())?, len))
            .is_some()
        {
            // Two batches for one temperature would make the index lossy: a
            // reader asking for it would silently get half the grid.
            return Err(format!("{label} appears in more than one energy batch").into());
        }
    }
    if temperatures.is_empty() {
        return Err("energy.arrow carries no temperature".into());
    }

    let schema_at = schema_offset(&bytes)?;
    Ok(EnergyRanges {
        schema: (schema_at, message_len(&bytes, schema_at)?),
        temperatures,
    })
}

/// Add (or refresh) `energy_ranges` in a converted nuclide's `version.json`.
///
/// A post-process, like [`crate::reaction_ranges::write_reaction_ranges`]: only
/// `version.json` is rewritten and `energy.arrow` is never opened for writing.
/// A folder with no `energy.arrow` (a photon element) is left alone.
///
/// Returns whether an index was written.
pub fn write_energy_ranges(dir: &Path) -> Result<bool, Box<dyn Error>> {
    let energy = dir.join("energy.arrow");
    if !energy.exists() {
        return Ok(false);
    }
    let version_path = dir.join("version.json");
    let mut version: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&version_path)?)?;
    let object = version
        .as_object_mut()
        .ok_or("version.json is not a JSON object")?;
    object.insert(
        "energy_ranges".to_string(),
        index_energy(&energy)?.to_json(),
    );

    let tmp = dir.join("version.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&version)?)?;
    std::fs::rename(tmp, version_path)?;
    Ok(true)
}
