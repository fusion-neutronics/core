//! Where each MT lives inside a published `reactions.arrow`.
//!
//! The file is written one Arrow record batch per MT, so every MT is already a
//! contiguous byte range in the object. Recording those ranges lets a reader
//! fetch just the channels it needs over HTTP range requests: an activation run
//! reads the handful its chain names and none of the full-grid transport MTs,
//! which are most of every file.
//!
//! The index rides in `version.json`, which every consumer already fetches, so
//! nothing extra is published and a reader that wants every MT still issues one
//! plain GET for a byte-identical file.
//!
//! # Why this crate
//!
//! Both halves need it and neither can depend on the other: `yamc-convert`
//! writes the index and `yamc-nuclide` reads it, and the dependency runs
//! convert to nuclide, not back. This crate is the leaf they already share.
//! Only the arithmetic and the JSON shape live here; `index_reactions`, which
//! has to walk an Arrow footer to produce them, stays in the converter.

use std::collections::BTreeMap;

/// The 8-byte end-of-stream marker: a continuation marker followed by a zero
/// metadata length. A stream without it is truncated rather than merely short.
pub const EOS: [u8; 8] = [0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0];

/// Where a message lives in `reactions.arrow`: `(offset, length)` in bytes.
pub type Range = (u64, u64);

/// Byte ranges for the schema message and each MT's record batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReactionRanges {
    /// The schema message, which any spliced stream must start with.
    pub schema: Range,
    /// MT number to the range of its record batch message.
    pub mts: BTreeMap<i32, Range>,
}

impl ReactionRanges {
    /// The `version.json` representation.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": [self.schema.0, self.schema.1],
            "mts": self.mts.iter()
                .map(|(mt, (off, len))| (mt.to_string(), serde_json::json!([off, len])))
                .collect::<serde_json::Map<_, _>>(),
        })
    }

    /// Parse back from `version.json`. `None` when the key is absent, which is
    /// what a library published before this existed looks like.
    pub fn from_json(value: &serde_json::Value) -> Option<Self> {
        let pair = |v: &serde_json::Value| -> Option<Range> {
            let a = v.as_array()?;
            Some((a.first()?.as_u64()?, a.get(1)?.as_u64()?))
        };
        Some(Self {
            schema: pair(value.get("schema")?)?,
            mts: value
                .get("mts")?
                .as_object()?
                .iter()
                .map(|(mt, v)| Some((mt.parse().ok()?, pair(v)?)))
                .collect::<Option<_>>()?,
        })
    }

    /// Pull the index out of a whole parsed `version.json`.
    ///
    /// `None` for a marker that carries no index, which is what a library
    /// converted before this existed looks like and is a normal answer: the
    /// caller fetches the whole file instead.
    pub fn from_version_json(version: &serde_json::Value) -> Option<Self> {
        Self::from_json(version.get("reaction_ranges")?)
    }

    /// The byte spans to request in order to read `wants`: the schema message,
    /// then the batch of every wanted MT this nuclide actually publishes, with
    /// spans that touch merged into one.
    ///
    /// Merged only where they are strictly adjacent, so every byte fetched is a
    /// byte wanted and the responses concatenate to exactly
    /// `schema ++ batches`. Nothing has to be cut back out, which is the whole
    /// reason not to merge across gaps.
    ///
    /// That costs little: the writer follows ENDF order, so the activation
    /// channels sit together and the published libraries come out at about
    /// three spans per nuclide rather than one per MT.
    ///
    /// An MT `wants` names that this nuclide does not publish is simply absent.
    /// The nuclide has no such channel, which is not the same as a fetch having
    /// failed, and the chain-wide MT union routinely names more than any one
    /// nuclide carries.
    pub fn spans_for(&self, wants: impl Fn(i32) -> bool) -> Vec<Range> {
        let mut spans: Vec<Range> = Vec::new();
        // The schema is the first message in the file, so leading with it also
        // keeps this list in ascending offset order, which is what lets the
        // merge below be a single backward look.
        for (off, len) in std::iter::once(self.schema).chain(
            self.mts
                .iter()
                .filter(|(mt, _)| wants(**mt))
                .map(|(_, r)| *r),
        ) {
            match spans.last_mut() {
                Some(last) if last.0 + last.1 == off => last.1 += len,
                _ => spans.push((off, len)),
            }
        }
        spans
    }

    /// Which of `wants` this nuclide actually publishes a batch for.
    ///
    /// What a cache has to record after a ranged fetch. The set asked for is the
    /// wrong thing to remember: a later request for an MT this nuclide does not
    /// carry would look uncovered forever and refetch on every load.
    pub fn present<'a>(&'a self, wants: impl Fn(i32) -> bool + 'a) -> Vec<i32> {
        self.mts.keys().copied().filter(|mt| wants(*mt)).collect()
    }
}

/// Splice fetched ranges into a readable Arrow IPC stream.
///
/// `schema` and each element of `batches` are the exact bytes at the ranges the
/// index gives, in the order the caller wants them back.
pub fn splice_stream(schema: &[u8], batches: &[Vec<u8>]) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(schema.len() + batches.iter().map(Vec::len).sum::<usize>() + EOS.len());
    out.extend_from_slice(schema);
    for batch in batches {
        out.extend_from_slice(batch);
    }
    out.extend_from_slice(&EOS);
    out
}

/// Splice the bodies of [`ReactionRanges::spans_for`] into a readable stream.
///
/// The spans already start with the schema and carry no unwanted bytes, so this
/// is a concatenation and the end-of-stream marker. Separate from
/// [`splice_stream`] only because that one takes the schema apart from the
/// batches, which a span list has already fused.
pub fn splice_spans(bodies: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bodies.iter().map(Vec::len).sum::<usize>() + EOS.len());
    for body in bodies {
        out.extend_from_slice(body);
    }
    out.extend_from_slice(&EOS);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges(schema: Range, mts: &[(i32, Range)]) -> ReactionRanges {
        ReactionRanges {
            schema,
            mts: mts.iter().copied().collect(),
        }
    }

    #[test]
    fn json_round_trips() {
        let r = ranges((64, 704), &[(16, (768, 100)), (102, (868, 200))]);
        assert_eq!(ReactionRanges::from_json(&r.to_json()), Some(r));
    }

    #[test]
    fn a_marker_with_no_index_is_none_rather_than_an_error() {
        let version = serde_json::json!({"data_version": "1"});
        assert_eq!(ReactionRanges::from_version_json(&version), None);
    }

    /// Touching spans fuse; a gap keeps them apart. This is what makes the
    /// concatenation of the responses a valid stream with nothing to cut out.
    #[test]
    fn adjacent_spans_merge_and_gaps_do_not() {
        let r = ranges(
            (64, 704),
            &[
                // 768 continues the schema, and 868 continues that.
                (16, (768, 100)),
                (102, (868, 200)),
                // A gap: 1068 is where the run above ends, this starts past it.
                (103, (5000, 50)),
            ],
        );
        assert_eq!(
            r.spans_for(|_| true),
            vec![(64, 1004), (5000, 50)],
            "one run to 1068, then the outlier",
        );
        assert_eq!(
            r.spans_for(|mt| mt == 103),
            vec![(64, 704), (5000, 50)],
            "the schema always leads, even when it abuts nothing wanted",
        );
    }

    /// An MT the caller names but the nuclide does not carry is not an error.
    /// The chain-wide union names far more than any one nuclide publishes.
    #[test]
    fn an_unpublished_mt_is_skipped_not_reported() {
        let r = ranges((0, 8), &[(102, (8, 10))]);
        assert_eq!(r.spans_for(|mt| mt == 16 || mt == 102), vec![(0, 18)]);
        assert_eq!(r.present(|mt| mt == 16 || mt == 102), vec![102]);
    }

    /// The schema alone is still a valid stream, and is what a nuclide carrying
    /// none of the wanted MTs splices to.
    #[test]
    fn no_wanted_mts_leaves_the_schema_and_nothing_else() {
        let r = ranges((0, 8), &[(102, (8, 10))]);
        assert_eq!(r.spans_for(|_| false), vec![(0, 8)]);
        assert_eq!(splice_spans(&[vec![0u8; 8]]).len(), 8 + EOS.len());
    }

    #[test]
    fn both_splices_agree() {
        let schema = vec![1u8, 2, 3];
        let batches = vec![vec![4u8, 5], vec![6u8]];
        assert_eq!(
            splice_stream(&schema, &batches),
            splice_spans(&[vec![1u8, 2, 3, 4, 5, 6]]),
            "one fused span is the same stream as schema plus batches",
        );
    }
}
