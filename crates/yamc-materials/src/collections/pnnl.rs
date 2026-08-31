//! The PNNL compendium collection.
//!
//! Compositions from the *Compendium of Material Composition Data for
//! Radiation Transport Modeling*, PNNL-15870 Rev. 2 (R.J. McConn Jr et al.,
//! Pacific Northwest National Laboratory, 2021):
//! <https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-15870Rev2.pdf>
//!
//! 410 of the report's 411 entries are bundled. Entry 173 "Iron Boride
//! (Fe2B)" is omitted because the report contradicts itself there: the
//! stated formula and molecular weight are Fe2B's, while every composition
//! number in the row describes FeB2.
//!
//! Names are the report's own, verbatim, so an entry can be quoted straight
//! back to it: `"Steel, Stainless 304"` is entry 331.

use super::{parse_table, CollectionEntry};
use crate::material::Material;
use std::sync::OnceLock;

/// Every bundled entry, in report order. Parsed once, on first use.
pub fn entries() -> &'static [CollectionEntry] {
    static ENTRIES: OnceLock<Vec<CollectionEntry>> = OnceLock::new();
    ENTRIES.get_or_init(|| parse_table(include_str!("pnnl_15870_rev2.txt")))
}

/// Look up an entry by its exact report name.
pub fn entry(name: &str) -> Option<&'static CollectionEntry> {
    entries().iter().find(|e| e.name == name)
}

/// Every material name, in report order.
pub fn names() -> Vec<&'static str> {
    entries().iter().map(|e| e.name).collect()
}

/// Names containing `query`, compared case-insensitively.
pub fn search(query: &str) -> Vec<&'static str> {
    let needle = query.to_lowercase();
    entries()
        .iter()
        .filter(|e| e.name.to_lowercase().contains(&needle))
        .map(|e| e.name)
        .collect()
}

/// Build a [`Material`] for the named entry.
///
/// # Errors
/// Returns a message listing near matches when `name` is not in the
/// collection, so a misremembered name fails with something actionable.
pub fn material(name: &str) -> Result<Material, String> {
    match entry(name) {
        Some(e) => e.to_material(),
        None => Err(unknown_name_error(name)),
    }
}

/// "no such material" with suggestions drawn from a word-overlap search.
fn unknown_name_error(name: &str) -> String {
    let mut hits = search(name);
    if hits.is_empty() {
        // Retry on the longest word, so "stainless 304" still suggests
        // something when the full string matches nothing.
        if let Some(word) = name
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .max_by_key(|w| w.len())
        {
            hits = search(word);
        }
    }
    hits.truncate(5);
    if hits.is_empty() {
        format!(
            "no material named '{name}' in the pnnl collection ({} entries); \
             use search() to find one",
            entries().len()
        )
    } else {
        format!(
            "no material named '{name}' in the pnnl collection; did you mean: {}",
            hits.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundles_the_expected_number_of_entries() {
        // 411 in the report, less the self-contradictory Fe2B entry.
        assert_eq!(entries().len(), 410);
    }

    #[test]
    fn names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for e in entries().iter() {
            assert!(seen.insert(e.name), "duplicate entry name {:?}", e.name);
        }
    }

    #[test]
    fn every_entry_has_a_sane_density_and_composition() {
        for e in entries().iter() {
            assert!(
                e.density > 0.0 && e.density < 25.0,
                "{}: implausible density {}",
                e.name,
                e.density
            );
            assert!(!e.composition.is_empty(), "{}: empty composition", e.name);
            let total: f64 = e.composition.iter().map(|(_, v)| v).sum();
            assert!(
                (total - 1.0).abs() < 5e-3,
                "{}: atom fractions sum to {total}",
                e.name
            );
            for (_, v) in &e.composition {
                assert!(*v > 0.0, "{}: non-positive fraction", e.name);
            }
        }
    }

    #[test]
    fn every_entry_builds_a_material() {
        for e in entries().iter() {
            let m = e
                .to_material()
                .unwrap_or_else(|err| panic!("{}: {err}", e.name));
            assert_eq!(m.name.as_deref(), Some(e.name));
            assert!(!m.nuclides.is_empty(), "{}: no nuclides", e.name);
        }
    }

    #[test]
    fn stainless_304_matches_the_report() {
        // Entry 331, the worked example in the issue. Atom fractions are the
        // report's Elemental column verbatim.
        let e = entry("Steel, Stainless 304").expect("entry 331");
        assert_eq!(e.number, 331);
        assert_eq!(e.density, 8.03);
        let fe = e
            .composition
            .iter()
            .find(|(k, _)| *k == "Fe")
            .expect("Fe present");
        assert_eq!(fe.1, 0.667971);
        let cr = e
            .composition
            .iter()
            .find(|(k, _)| *k == "Cr")
            .expect("Cr present");
        assert_eq!(cr.1, 0.199443);
    }

    #[test]
    fn elements_expand_but_enriched_nuclides_do_not() {
        // Natural iron in stainless expands to yamc's abundances...
        let steel = material("Steel, Stainless 304").unwrap();
        assert!(steel.nuclides.contains_key("Fe56"));
        assert!(steel.nuclides.contains_key("Fe54"));
        // ...while an enriched entry keeps the report's own isotopics.
        let lgb = entry("Lithium Gadolinium Borate (LGB)").unwrap();
        let keys: Vec<&str> = lgb.composition.iter().map(|(k, _)| *k).collect();
        assert!(
            keys.contains(&"Li6"),
            "LGB should be Li6-enriched: {keys:?}"
        );
        assert!(
            keys.contains(&"B10"),
            "LGB should be B10-enriched: {keys:?}"
        );
        assert!(!keys.contains(&"Li"), "LGB must not use natural Li");
    }

    #[test]
    fn heavy_water_is_deuterated() {
        let e = entry("Water, Heavy").unwrap();
        let keys: Vec<&str> = e.composition.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&"H2"), "expected D2O, got {keys:?}");
        assert!(!keys.contains(&"H"));
    }

    #[test]
    fn the_self_contradictory_iron_boride_entry_is_absent() {
        // Entry 173 is omitted; entry 174 (FeB) is not.
        assert!(entry("Iron Boride (Fe2B)").is_none());
        let feb = entry("Iron Boride (FeB)").expect("entry 174 kept");
        assert_eq!(feb.number, 174);
    }

    #[test]
    fn search_is_case_insensitive_and_substring() {
        let hits = search("stainless");
        assert!(hits.contains(&"Steel, Stainless 304"));
        assert!(
            hits.len() > 5,
            "expected several stainless steels: {hits:?}"
        );
        assert_eq!(search("STAINLESS"), hits);
        assert!(search("no such material anywhere").is_empty());
    }

    #[test]
    fn unknown_names_suggest_alternatives() {
        let err = material("Steel, Stainless 999").unwrap_err();
        assert!(err.contains("did you mean"), "unhelpful error: {err}");
        assert!(err.contains("Stainless"), "unhelpful error: {err}");
    }

    #[test]
    fn entry_numbers_are_the_reports_own() {
        // A spot check that numbering was not renumbered by the omission.
        assert_eq!(entry("A-150 Tissue-Equivalent Plastic").unwrap().number, 1);
        assert_eq!(entry("Iron Boride (FeB)").unwrap().number, 174);
    }
}
