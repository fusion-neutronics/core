//! Byte ranges for the per-temperature batches of `energy.arrow`.
//!
//! The union energy grids are their own section, one record batch per
//! temperature, so a client that wants one temperature does not read every one
//! of them (6.33 MB on U238, against 8.98 MB for the entire single-temperature
//! JSON bundle). This is what says where each batch is.
//!
//! The same shape as [`crate::reaction_ranges::ReactionRanges`] with one axis
//! instead of two, and the same adjacency merge, so a load that wants several
//! temperatures asks for one span when they sit together (which the writer
//! makes them do).
//!
//! Each entry may carry the sha256 of its bytes as a third element, exactly as
//! in the reaction index (see [`crate::reaction_ranges`]): one hash per
//! temperature's grid, and one for the schema message.

use std::collections::BTreeMap;

use crate::reaction_ranges::{coalesce, entry_json, parse_entry, Hashes, Range};

/// Byte ranges for the schema message and each temperature's record batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnergyRanges {
    /// The schema message, which any spliced stream must start with.
    pub schema: Range,
    /// Temperature label, as the file spells it (`"294K"`), to the range of the
    /// record batch carrying that grid.
    ///
    /// Carries every label the section holds, including a grid whose label is
    /// not one of the nuclide's temperatures: the NJOY route publishes a 0 K
    /// grid that no temperature filter can ever name, and dropping it here
    /// would make a ranged load quietly provide less than a whole-file one.
    pub temperatures: BTreeMap<String, Range>,
    /// The sha256 of the schema message and of each grid, where the index
    /// carries them. Empty for an index written before hashes existed.
    pub sha256: Hashes,
}

impl EnergyRanges {
    /// The `version.json` representation:
    /// `{"schema": [off, len, sha], "temperatures": {"294K": [off, len, sha], ...}}`,
    /// each entry losing its third element when [`Self::sha256`] has no hash
    /// for it.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": entry_json(&self.schema, &self.sha256),
            "temperatures": self.temperatures.iter()
                .map(|(t, r)| (t.clone(), entry_json(r, &self.sha256)))
                .collect::<serde_json::Map<_, _>>(),
        })
    }

    /// Parse back from `version.json`, with or without hashes. `None` when the
    /// key is absent or the shape is not this one, which is what a library
    /// published before the grids moved looks like; such a library is fetched
    /// whole.
    pub fn from_json(value: &serde_json::Value) -> Option<Self> {
        let mut sha256 = Hashes::new();
        let schema = parse_entry(value.get("schema")?, &mut sha256)?;
        let mut temperatures = BTreeMap::new();
        for (t, r) in value.get("temperatures")?.as_object()? {
            temperatures.insert(t.clone(), parse_entry(r, &mut sha256)?);
        }
        Some(Self {
            schema,
            temperatures,
            sha256,
        })
    }

    /// Pull the index out of a whole parsed `version.json`.
    pub fn from_version_json(version: &serde_json::Value) -> Option<Self> {
        Self::from_json(version.get("energy_ranges")?)
    }

    /// The byte spans to request in order to read the temperatures `wants`
    /// names: the schema message, then the batch of each, spans that touch
    /// merged into one.
    ///
    /// A label `wants` names that this nuclide does not publish is simply
    /// absent, the same as an MT it does not carry.
    pub fn spans_for(&self, wants: impl Fn(&str) -> bool) -> Vec<Range> {
        coalesce(
            self.schema,
            self.temperatures
                .iter()
                .filter(|(t, _)| wants(t))
                .map(|(_, r)| *r)
                .collect(),
        )
    }

    /// The messages [`Self::spans_for`] covers, each with a name to report it
    /// by: the schema message first, then every wanted grid in offset order.
    /// What a reader checks a ranged fetch against, one hash per entry.
    pub fn pieces_for(&self, wants: impl Fn(&str) -> bool) -> Vec<(Range, String)> {
        let mut grids: Vec<(Range, String)> = self
            .temperatures
            .iter()
            .filter(|(t, _)| wants(t))
            .map(|(t, r)| (*r, format!("the {t} grid")))
            .collect();
        grids.sort();
        std::iter::once((self.schema, "the schema message".to_string()))
            .chain(grids)
            .collect()
    }

    /// Which of `wants` this nuclide actually publishes a grid for.
    ///
    /// What a cache records after a ranged fetch, for the same reason the
    /// reaction index does: the set asked for would leave a temperature this
    /// nuclide does not carry looking uncovered forever.
    pub fn present<'a>(&'a self, wants: impl Fn(&str) -> bool + 'a) -> Vec<String> {
        self.temperatures
            .keys()
            .filter(|t| wants(t))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> EnergyRanges {
        EnergyRanges {
            schema: (0, 100),
            temperatures: BTreeMap::from([
                ("0K".to_string(), (100, 50)),
                ("294K".to_string(), (150, 60)),
                ("600K".to_string(), (210, 70)),
                ("900K".to_string(), (400, 80)),
            ]),
            sha256: Hashes::new(),
        }
    }

    #[test]
    fn json_round_trips() {
        let e = index();
        assert_eq!(EnergyRanges::from_json(&e.to_json()), Some(e));
    }

    #[test]
    fn json_round_trips_with_hashes() {
        let mut e = index();
        e.sha256 = Hashes::from([((0, 100), "a".repeat(64)), ((150, 60), "b".repeat(64))]);
        let json = e.to_json();
        assert_eq!(json["schema"][2], "a".repeat(64));
        assert_eq!(json["temperatures"]["294K"][2], "b".repeat(64));
        assert_eq!(
            json["temperatures"]["600K"],
            serde_json::json!([210, 70]),
            "an entry with no hash keeps the two-element form"
        );
        assert_eq!(EnergyRanges::from_json(&json), Some(e));
    }

    #[test]
    fn pieces_are_the_schema_then_each_wanted_grid() {
        assert_eq!(
            index().pieces_for(|t| t == "900K" || t == "294K"),
            vec![
                ((0, 100), "the schema message".to_string()),
                ((150, 60), "the 294K grid".to_string()),
                ((400, 80), "the 900K grid".to_string()),
            ]
        );
    }

    #[test]
    fn the_json_shape_is_temperature_to_range() {
        let json = index().to_json();
        assert_eq!(json["temperatures"]["294K"][0], 150);
        assert_eq!(json["temperatures"]["294K"][1], 60);
    }

    #[test]
    fn a_marker_without_the_key_has_no_index() {
        let version = serde_json::json!({"format_version": 2});
        assert_eq!(EnergyRanges::from_version_json(&version), None);
    }

    #[test]
    fn adjacent_temperatures_merge_into_one_span() {
        // 294K (150..210) and 600K (210..280) sit back to back, so they are one
        // request of 130 bytes rather than two. The schema stays its own span:
        // the 0 K grid lies between it and 294K and was not asked for.
        let spans = index().spans_for(|t| t == "294K" || t == "600K");
        assert_eq!(spans, vec![(0, 100), (150, 130)]);
    }

    #[test]
    fn a_run_from_the_schema_is_one_span() {
        // Everything up to 600K, with nothing skipped, is a single range
        // request covering the file from byte 0 to byte 280.
        let spans = index().spans_for(|t| t != "900K");
        assert_eq!(spans, vec![(0, 280)]);
    }

    #[test]
    fn a_gap_is_not_merged_across() {
        // 900K starts at 400, with 190 unasked-for bytes in front of it, so
        // those bytes are not fetched just to make the request contiguous.
        let spans = index().spans_for(|t| t == "294K" || t == "900K");
        assert_eq!(spans, vec![(0, 100), (150, 60), (400, 80)]);
    }

    #[test]
    fn one_temperature_is_the_schema_and_its_batch() {
        assert_eq!(
            index().spans_for(|t| t == "900K"),
            vec![(0, 100), (400, 80)]
        );
    }

    #[test]
    fn present_reports_what_is_published_not_what_was_asked() {
        let held = index().present(|t| t == "294K" || t == "1200K");
        assert_eq!(held, vec!["294K".to_string()]);
    }
}
