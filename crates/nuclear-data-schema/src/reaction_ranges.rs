//! Where each (MT, temperature) lives inside a published `reactions.arrow`.
//!
//! The file is written one Arrow record batch per (MT, temperature), so every
//! cross section is already a contiguous byte range in the object. Recording
//! those ranges lets a reader fetch just what it needs over HTTP range requests:
//! an activation run reads the handful of channels its chain names at every
//! temperature and none of the full-grid transport MTs, which are most of every
//! file, and a plotter reads one channel at the one temperature it draws
//! (fusion-neutronics/core#100).
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
//!
//! # Files written one batch per MT
//!
//! A `reactions.arrow` from before the temperature split carries one batch per
//! MT with every temperature inside it. Indexing such a file lists that one
//! batch under each temperature it holds, so the index says the truth about
//! where each (MT, temperature) can be read from, and a reader that fetches one
//! temperature of it gets a batch carrying the others as well.

use std::collections::BTreeMap;

/// The 8-byte end-of-stream marker: a continuation marker followed by a zero
/// metadata length. A stream without it is truncated rather than merely short.
pub const EOS: [u8; 8] = [0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0];

/// Where a message lives in `reactions.arrow`: `(offset, length)` in bytes.
pub type Range = (u64, u64);

/// Byte ranges for the schema message and each (MT, temperature) record batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReactionRanges {
    /// The schema message, which any spliced stream must start with.
    pub schema: Range,
    /// MT number to temperature label (as the file spells it, `"294K"`) to the
    /// range of the record batch carrying that cross section.
    pub mts: BTreeMap<i32, BTreeMap<String, Range>>,
}

impl ReactionRanges {
    /// The `version.json` representation:
    /// `{"schema": [off, len], "mts": {"102": {"294K": [off, len], ...}, ...}}`.
    pub fn to_json(&self) -> serde_json::Value {
        let pair = |(off, len): &Range| serde_json::json!([off, len]);
        serde_json::json!({
            "schema": pair(&self.schema),
            "mts": self.mts.iter()
                .map(|(mt, by_temperature)| {
                    let temps = by_temperature
                        .iter()
                        .map(|(t, r)| (t.clone(), pair(r)))
                        .collect::<serde_json::Map<_, _>>();
                    (mt.to_string(), serde_json::Value::Object(temps))
                })
                .collect::<serde_json::Map<_, _>>(),
        })
    }

    /// Parse back from `version.json`. `None` when the key is absent or the
    /// shape is not this one, which is what a library indexed before the
    /// temperature split looks like (its `mts` values are bare ranges); such a
    /// library is fetched whole until it is reindexed.
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
                .map(|(mt, by_temperature)| {
                    let temps = by_temperature
                        .as_object()?
                        .iter()
                        .map(|(t, r)| Some((t.clone(), pair(r)?)))
                        .collect::<Option<BTreeMap<_, _>>>()?;
                    Some((mt.parse().ok()?, temps))
                })
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

    /// The byte spans to request in order to read every temperature of the MTs
    /// `wants` names: the schema message, then the batches of every wanted MT
    /// this nuclide actually publishes, with spans that touch merged into one.
    ///
    /// What a transport or activation load asks for, since it wants every
    /// temperature a cross section is published at.
    pub fn spans_for(&self, wants: impl Fn(i32) -> bool) -> Vec<Range> {
        self.spans_where(|mt, _| wants(mt))
    }

    /// The byte spans to request in order to read the (MT, temperature) pairs
    /// `wants` names: the schema message, then the batch of every wanted pair
    /// this nuclide actually publishes, with spans that touch merged into one.
    ///
    /// Merged only where they are strictly adjacent, so every byte fetched is a
    /// byte wanted and the responses concatenate to exactly
    /// `schema ++ batches`. Nothing has to be cut back out, which is the whole
    /// reason not to merge across gaps.
    ///
    /// That costs little: the writer follows ENDF order with the temperatures
    /// of one MT back to back, so the activation channels sit together and the
    /// published libraries come out at a few spans per nuclide rather than one
    /// per batch.
    ///
    /// A batch listed under several temperatures (a file written one batch per
    /// MT) is requested once however many of them are wanted. A pair `wants`
    /// names that this nuclide does not publish is simply absent: the nuclide
    /// has no such channel, which is not the same as a fetch having failed.
    pub fn spans_where(&self, wants: impl Fn(i32, &str) -> bool) -> Vec<Range> {
        let mut ranges: Vec<Range> = self
            .mts
            .iter()
            .flat_map(|(mt, by_temperature)| {
                by_temperature
                    .iter()
                    .filter(|(t, _)| wants(*mt, t))
                    .map(|(_, r)| *r)
            })
            .collect();
        // Ascending offset order is what lets the merge below be a single
        // backward look, and the schema is the first message in the file, so
        // leading with it keeps the list sorted.
        ranges.sort_unstable();
        ranges.dedup();
        let mut spans: Vec<Range> = Vec::new();
        for (off, len) in std::iter::once(self.schema).chain(ranges) {
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

/// Splice the bodies of [`ReactionRanges::spans_for`] or
/// [`ReactionRanges::spans_where`] into a readable stream.
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

    /// An index over batches written one per (MT, temperature).
    fn ranges(schema: Range, batches: &[(i32, &str, Range)]) -> ReactionRanges {
        let mut mts: BTreeMap<i32, BTreeMap<String, Range>> = BTreeMap::new();
        for (mt, t, r) in batches {
            mts.entry(*mt).or_default().insert((*t).to_string(), *r);
        }
        ReactionRanges { schema, mts }
    }

    #[test]
    fn json_round_trips() {
        let r = ranges(
            (64, 704),
            &[
                (16, "294K", (768, 100)),
                (16, "600K", (868, 100)),
                (102, "294K", (968, 200)),
            ],
        );
        let json = r.to_json();
        assert_eq!(json["mts"]["16"]["600K"], serde_json::json!([868, 100]));
        assert_eq!(ReactionRanges::from_json(&json), Some(r));
    }

    #[test]
    fn a_marker_with_no_index_is_none_rather_than_an_error() {
        let version = serde_json::json!({"data_version": "1"});
        assert_eq!(ReactionRanges::from_version_json(&version), None);
    }

    /// The shape from before the temperature split, `mt -> [off, len]`, is not
    /// an index this reader can plan temperatures from, so it reads as none and
    /// the caller fetches the whole file. Reindexing the published
    /// `version.json` is what turns the ranged path back on.
    #[test]
    fn a_flat_index_from_before_the_split_reads_as_no_index() {
        let version = serde_json::json!({
            "reaction_ranges": {"schema": [64, 704], "mts": {"16": [768, 100]}}
        });
        assert_eq!(ReactionRanges::from_version_json(&version), None);
    }

    /// Touching spans fuse; a gap keeps them apart. This is what makes the
    /// concatenation of the responses a valid stream with nothing to cut out.
    #[test]
    fn adjacent_spans_merge_and_gaps_do_not() {
        let r = ranges(
            (64, 704),
            &[
                // 768 continues the schema, and each temperature continues the
                // last, as the writer lays one MT's temperatures back to back.
                (16, "294K", (768, 100)),
                (16, "600K", (868, 100)),
                (102, "294K", (968, 200)),
                // A gap: 1168 is where the run above ends, this starts past it.
                (103, "294K", (5000, 50)),
            ],
        );
        assert_eq!(
            r.spans_for(|_| true),
            vec![(64, 1104), (5000, 50)],
            "one run to 1168, then the outlier",
        );
        assert_eq!(
            r.spans_for(|mt| mt == 103),
            vec![(64, 704), (5000, 50)],
            "the schema always leads, even when it abuts nothing wanted",
        );
    }

    /// One temperature of one MT is the plotter's request, and the answer is
    /// that batch alone: the other temperature of the same MT is a separate
    /// batch and is not fetched.
    #[test]
    fn one_temperature_of_one_mt_is_one_batch() {
        let r = ranges(
            (0, 8),
            &[
                (16, "294K", (8, 10)),
                (16, "600K", (18, 10)),
                (102, "294K", (28, 10)),
            ],
        );
        assert_eq!(
            r.spans_where(|mt, t| mt == 16 && t == "600K"),
            vec![(0, 8), (18, 10)]
        );
        assert_eq!(
            r.spans_where(|_, t| t == "294K"),
            vec![(0, 18), (28, 10)],
            "every MT at one temperature skips the other temperature's batch",
        );
        assert_eq!(
            r.mts[&16].keys().collect::<Vec<_>>(),
            vec!["294K", "600K"],
            "both temperatures of MT 16 are indexed"
        );
        assert!(!r.mts.contains_key(&999));
    }

    /// A file written one batch per MT lists that batch under every temperature
    /// it carries. Wanting several of those temperatures must fetch it once.
    #[test]
    fn a_batch_shared_by_temperatures_is_requested_once() {
        let r = ranges(
            (0, 8),
            &[
                (16, "294K", (8, 10)),
                (16, "600K", (8, 10)),
                (102, "294K", (18, 10)),
                (102, "600K", (18, 10)),
            ],
        );
        assert_eq!(r.spans_for(|_| true), vec![(0, 28)]);
        assert_eq!(r.spans_where(|_, t| t == "600K"), vec![(0, 28)]);
    }

    /// An MT the caller names but the nuclide does not carry is not an error.
    /// The chain-wide union names far more than any one nuclide publishes.
    #[test]
    fn an_unpublished_mt_is_skipped_not_reported() {
        let r = ranges((0, 8), &[(102, "294K", (8, 10))]);
        assert_eq!(r.spans_for(|mt| mt == 16 || mt == 102), vec![(0, 18)]);
        assert_eq!(r.present(|mt| mt == 16 || mt == 102), vec![102]);
    }

    /// The schema alone is still a valid stream, and is what a nuclide carrying
    /// none of the wanted MTs splices to.
    #[test]
    fn no_wanted_mts_leaves_the_schema_and_nothing_else() {
        let r = ranges((0, 8), &[(102, "294K", (8, 10))]);
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
