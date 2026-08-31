//! Byte ranges of each MT's record batch within `reactions.arrow`.
//!
//! A transmutation run reads only the MTs its chain names, but `reactions.arrow`
//! is fetched whole. On Fe56 (TENDL-2025) ten full-grid transport MTs -- total,
//! elastic, nonelastic, inelastic, absorption, disappearance, heating, damage --
//! are 13.6 MB of a 14.7 MB file, and an activation run reads none of them.
//! [`LoadScope::activation`] already drops them, but only after the download:
//! `yamc-nuclide/src/load_scope.rs` says outright that filtering there "saves
//! parse time and retained memory, **not bytes read**".
//!
//! The file is written one record batch per MT, so every MT is *already* a
//! contiguous byte range in the published object. Recording those ranges lets a
//! reader fetch just the MTs it wants over HTTP range requests, which cuts a
//! full ENDF/B-8.1 activation closure from 2.9 GB to about 508 MB. The origin
//! serves `accept-ranges: bytes` behind Cloudflare with a one-year TTL, so
//! ranged reads are answered from the edge cache.
//!
//! Nothing is duplicated and no new object is published: the index rides in
//! `version.json`, which every consumer already fetches. A reader that wants
//! every MT still issues one plain GET and gets a byte-identical file, so the
//! transport path is untouched.
//!
//! # Where the pieces live
//!
//! The JSON shape, the span planning and the splice moved to
//! `nuclear_data_schema::reaction_ranges`, so the loader can read an index this
//! crate wrote without depending on the converter. They are re-exported here,
//! because the indexing below is what produces one and this is where a caller
//! looks. What stays is the part that needs an Arrow reader: walking the footer
//! to find the batches.
//!
//! # Reassembling a stream
//!
//! Record batch messages are self-contained but a decoder needs the schema
//! first, so the index carries the schema message's range too. Splicing
//!
//! ```text
//! schema_bytes ++ selected_batch_bytes ++ EOS
//! ```
//!
//! yields a valid Arrow IPC *stream* that reads with no special casing and
//! produces batches identical to the whole-file path. [`splice_stream`] does
//! this, and the round-trip test asserts the equality.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;

use arrow_ipc::reader::FileReader;

pub use nuclear_data_schema::reaction_ranges::{splice_stream, Range, ReactionRanges, EOS};

/// `"ARROW1\0\0"`, the file magic and padding the first message sits after.
const FILE_HEADER_LEN: u64 = 8;

/// Length of the IPC message starting at `offset`.
///
/// Two framings exist and both appear in the wild. The v4+ encapsulated format
/// is a 4-byte continuation marker, a 4-byte little-endian metadata length,
/// that much flatbuffer, then the body. The legacy format omits the marker and
/// leads with the length. Distinguishing them is exactly the continuation
/// marker's purpose: a legacy length of 0xffffffff would be an absurd metadata
/// size, so the value is unambiguous.
///
/// The metadata length already includes padding to an 8-byte boundary, so
/// header plus length is the whole message when there is no body -- the case
/// for a schema.
fn message_len(bytes: &[u8], offset: u64) -> Result<u64, Box<dyn Error>> {
    let at = usize::try_from(offset)?;
    let header: [u8; 8] = bytes
        .get(at..at + 8)
        .ok_or("reactions.arrow is truncated before its first message")?
        .try_into()?;
    Ok(if header[..4] == [0xff, 0xff, 0xff, 0xff] {
        8 + u64::from(u32::from_le_bytes(header[4..8].try_into()?))
    } else {
        4 + u64::from(u32::from_le_bytes(header[..4].try_into()?))
    })
}

/// Offset of the schema, which is the first message after the file magic.
///
/// Not `FILE_HEADER_LEN`: that is where the magic ends, not where the message
/// begins. A writer may align the first message, and arrow-rs does, padding to
/// a 64-byte boundary with zeros. Assuming offset 8 there reads the padding as
/// a legacy frame whose length is zero and calls the schema 4 bytes long, which
/// is not a range any consumer can splice: the LZ4 codec is declared in the
/// schema, so a stream missing it does not know to decompress.
///
/// Both writers this format is produced by therefore have to be handled. The
/// padding is zeros and no message starts with eight zero bytes, since the
/// encapsulated framing opens with the continuation marker and the legacy one
/// with a non-zero metadata length, so skipping zeroed 8-byte blocks finds the
/// message under either.
fn schema_offset(bytes: &[u8]) -> Result<u64, Box<dyn Error>> {
    let mut at = usize::try_from(FILE_HEADER_LEN)?;
    while bytes
        .get(at..at + 8)
        .ok_or("reactions.arrow is truncated before its schema")?
        == [0u8; 8]
    {
        at += 8;
    }
    Ok(u64::try_from(at)?)
}

/// Index the record batches of a `{Nuclide}.arrow/reactions.arrow`.
///
/// Reads the footer for the batch offsets and the batches themselves for their
/// MTs; footer block *i* is reader batch *i*, which is what ties the two
/// together.
pub fn index_reactions(reactions: &Path) -> Result<ReactionRanges, Box<dyn Error>> {
    let bytes = std::fs::read(reactions)?;

    // The footer's Block array is the authority on where each batch starts;
    // deriving offsets by re-serializing would only reproduce them if this
    // build wrote the file, which for already-published data it did not.
    let footer_len_at = bytes
        .len()
        .checked_sub(10)
        .ok_or("reactions.arrow is too short to hold an Arrow footer")?;
    let footer_len = u32::from_le_bytes(bytes[footer_len_at..footer_len_at + 4].try_into()?);
    let footer_at = footer_len_at
        .checked_sub(usize::try_from(footer_len)?)
        .ok_or("Arrow footer length runs past the start of the file")?;
    let footer = arrow_ipc::root_as_footer(&bytes[footer_at..footer_len_at])
        .map_err(|e| format!("unreadable Arrow footer: {e}"))?;
    let blocks = footer
        .recordBatches()
        .ok_or("Arrow footer names no record batches")?;

    // MTs come from the batches, so this pass is what makes the index
    // addressable by reaction rather than by position.
    let reader = FileReader::try_new(File::open(reactions)?, None)?;
    let mut mts = BTreeMap::new();
    for (i, batch) in reader.enumerate() {
        let batch = batch?;
        let block = blocks.get(i);
        let mt = batch
            .column_by_name("mt")
            .ok_or("a reactions batch has no `mt` column")?
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or("`mt` is not an Int32Array")?;
        if mt.is_empty() {
            return Err("a reactions batch carries no mt value".into());
        }
        let len = u64::try_from(block.metaDataLength())? + u64::try_from(block.bodyLength())?;
        if mts
            .insert(mt.value(0), (u64::try_from(block.offset())?, len))
            .is_some()
        {
            // Two batches for one MT would make the index lossy: a reader
            // asking for that MT would silently get half its cross section.
            return Err(format!("MT {} appears in more than one batch", mt.value(0)).into());
        }
    }

    let schema_at = schema_offset(&bytes)?;
    Ok(ReactionRanges {
        schema: (schema_at, message_len(&bytes, schema_at)?),
        mts,
    })
}

/// Add (or refresh) `reaction_ranges` in a converted nuclide's `version.json`.
///
/// A post-process, so an already-built tree gains the index without being
/// reconverted: only `version.json` is rewritten and the multi-GB cross-section
/// files are never touched. A folder with no `reactions.arrow` -- a photon
/// element, or a conversion that produced no reactions table -- is left alone.
///
/// Returns whether an index was written.
pub fn write_reaction_ranges(dir: &Path) -> Result<bool, Box<dyn Error>> {
    let reactions = dir.join("reactions.arrow");
    if !reactions.exists() {
        return Ok(false);
    }
    let version_path = dir.join("version.json");
    let mut version: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&version_path)?)?;
    let object = version
        .as_object_mut()
        .ok_or("version.json is not a JSON object")?;
    object.insert(
        "reaction_ranges".to_string(),
        index_reactions(&reactions)?.to_json(),
    );

    // Same write-then-rename as `entry::write_version`: version.json is what a
    // resume takes as proof the conversion finished, so it must never be
    // observed half-written.
    let tmp = dir.join("version.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&version)?)?;
    std::fs::rename(tmp, &version_path)?;
    Ok(true)
}
