//! Data module for nuclear data and dose coefficients
//!
//! This module provides access to nuclear data including dose conversion coefficients
//! from ICRP publications.
#![cfg_attr(rustfmt, rustfmt_skip)]

pub mod effective_dose;
pub mod photon_attenuation;

pub use effective_dose::{dose_coefficients, DoseDataSource, DoseGeometry, DoseParticle};
pub use photon_attenuation::{
    mass_attenuation_coefficient, mass_energy_absorption_air, CoefficientTable,
};
use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Parse a whitespace-delimited `"<key> <value>"` table (one entry per
/// line, e.g. `Fe56 55.934936`) embedded via `include_str!` into a
/// `HashMap<&'static str, f64>`. Blank lines are skipped; on duplicate keys
/// the last line wins, matching the previous explicit-`insert` ordering.
/// `.parse::<f64>()` of the exact decimal literal reproduces the same f64
/// the inline literal did (both round to nearest), so values are unchanged.
fn parse_f64_table(data: &'static str) -> HashMap<&'static str, f64> {
    data.lines()
        .filter_map(|line| {
            let mut it = line.split_whitespace();
            let key = it.next()?;
            let value = it.next()?;
            Some((key, value.parse::<f64>().expect("malformed f64 in embedded data table")))
        })
        .collect()
}

/// Map from element symbol to sorted vector of nuclide (isotope) identifiers.
///
/// Keys are element symbols (e.g. `"Li"`) and values are the sorted list of
/// nuclide names for naturally occurring isotopes of that element (e.g.
/// `["Li6", "Li7"]`). The mapping is derived automatically from
/// [`NATURAL_ABUNDANCE`] so it stays consistent with the set of isotopes for
/// which natural abundances are defined.
pub static ELEMENT_NUCLIDES: Lazy<HashMap<&'static str, Vec<&'static str>>> = Lazy::new(|| {
    let mut map: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
    for &nuclide in NATURAL_ABUNDANCE.keys() {
        // Find the index where the first digit occurs
        let idx = nuclide
            .find(|c: char| c.is_ascii_digit())
            .unwrap_or(nuclide.len());
        let element = &nuclide[..idx]; // This is a &'static str slice
        map.entry(element).or_default().push(nuclide);
    }
    // Sort nuclides for each element
    for nuclides in map.values_mut() {
        nuclides.sort();
    }
    map
});
// src/data.rs
// This module contains large static data tables for the materials library.
// The volume of data is significant; doc comments summarize the intent of
// each table while the literals provide the canonical numeric values.

/// Natural terrestrial isotopic abundances (fractional, summing to ~1.0 per
/// element) for stable isotopes.
///
/// Each key is a nuclide name (e.g. `"Fe56"`) and the value is its natural
/// abundance by atom fraction. Values are sourced from standard reference
/// compilations (rounded as needed). Elements with a single stable isotope are
/// assigned 1.0.
pub static NATURAL_ABUNDANCE: Lazy<HashMap<&'static str, f64>> =
    Lazy::new(|| parse_f64_table(include_str!("natural_abundance.txt")));

/// Atomic masses in unified atomic mass units (u) from the AME2020
/// evaluation (Huang et al., Chinese Physics C45, 2021).
/// Parsed from mass_1.mas20.txt.
///
/// 3558 isotopes.
///
/// Regenerate with: python scripts/gen_atomic_mass_table.py
pub static ATOMIC_MASS: Lazy<HashMap<&'static str, f64>> =
    Lazy::new(|| parse_f64_table(include_str!("atomic_mass.txt")));

/// Mapping from element symbol to its lowercase English name.
///
/// Provided for convenience when presenting user‑facing descriptions and for
/// validating element inputs (case sensitive symbol keys matching the raw
/// nuclear data tables).
pub static ELEMENT_NAMES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut names = HashMap::new();
    names.insert("H", "hydrogen");
    names.insert("He", "helium");
    names.insert("Li", "lithium");
    names.insert("Be", "beryllium");
    names.insert("B", "boron");
    names.insert("C", "carbon");
    names.insert("N", "nitrogen");
    names.insert("O", "oxygen");
    names.insert("F", "fluorine");
    names.insert("Ne", "neon");
    names.insert("Na", "sodium");
    names.insert("Mg", "magnesium");
    names.insert("Al", "aluminum");
    names.insert("Si", "silicon");
    names.insert("P", "phosphorus");
    names.insert("S", "sulfur");
    names.insert("Cl", "chlorine");
    names.insert("Ar", "argon");
    names.insert("K", "potassium");
    names.insert("Ca", "calcium");
    names.insert("Sc", "scandium");
    names.insert("Ti", "titanium");
    names.insert("V", "vanadium");
    names.insert("Cr", "chromium");
    names.insert("Mn", "manganese");
    names.insert("Fe", "iron");
    names.insert("Co", "cobalt");
    names.insert("Ni", "nickel");
    names.insert("Cu", "copper");
    names.insert("Zn", "zinc");
    names.insert("Ga", "gallium");
    names.insert("Ge", "germanium");
    names.insert("As", "arsenic");
    names.insert("Se", "selenium");
    names.insert("Br", "bromine");
    names.insert("Kr", "krypton");
    names.insert("Rb", "rubidium");
    names.insert("Sr", "strontium");
    names.insert("Y", "yttrium");
    names.insert("Zr", "zirconium");
    names.insert("Nb", "niobium");
    names.insert("Mo", "molybdenum");
    names.insert("Tc", "technetium");
    names.insert("Ru", "ruthenium");
    names.insert("Rh", "rhodium");
    names.insert("Pd", "palladium");
    names.insert("Ag", "silver");
    names.insert("Cd", "cadmium");
    names.insert("In", "indium");
    names.insert("Sn", "tin");
    names.insert("Sb", "antimony");
    names.insert("Te", "tellurium");
    names.insert("I", "iodine");
    names.insert("Xe", "xenon");
    names.insert("Cs", "cesium");
    names.insert("Ba", "barium");
    names.insert("La", "lanthanum");
    names.insert("Ce", "cerium");
    names.insert("Pr", "praseodymium");
    names.insert("Nd", "neodymium");
    names.insert("Pm", "promethium");
    names.insert("Sm", "samarium");
    names.insert("Eu", "europium");
    names.insert("Gd", "gadolinium");
    names.insert("Tb", "terbium");
    names.insert("Dy", "dysprosium");
    names.insert("Ho", "holmium");
    names.insert("Er", "erbium");
    names.insert("Tm", "thulium");
    names.insert("Yb", "ytterbium");
    names.insert("Lu", "lutetium");
    names.insert("Hf", "hafnium");
    names.insert("Ta", "tantalum");
    names.insert("W", "tungsten");
    names.insert("Re", "rhenium");
    names.insert("Os", "osmium");
    names.insert("Ir", "iridium");
    names.insert("Pt", "platinum");
    names.insert("Au", "gold");
    names.insert("Hg", "mercury");
    names.insert("Tl", "thallium");
    names.insert("Pb", "lead");
    names.insert("Bi", "bismuth");
    names.insert("Po", "polonium");
    names.insert("At", "astatine");
    names.insert("Rn", "radon");
    names.insert("Fr", "francium");
    names.insert("Ra", "radium");
    names.insert("Ac", "actinium");
    names.insert("Th", "thorium");
    names.insert("Pa", "protactinium");
    names.insert("U", "uranium");
    names.insert("Np", "neptunium");
    names.insert("Pu", "plutonium");
    names.insert("Am", "americium");
    names.insert("Cm", "curium");
    names.insert("Bk", "berkelium");
    names.insert("Cf", "californium");
    names.insert("Es", "einsteinium");
    names.insert("Fm", "fermium");
    names.insert("Md", "mendelevium");
    names.insert("No", "nobelium");
    names.insert("Lr", "lawrencium");
    names
});

/// Reaction name to MT number, for every name a reaction goes by.
///
/// Derived from [`endf::reaction_name`] rather than listed here. The list used
/// to be written out, 414 entries of it, and the two copies could disagree
/// without anything noticing: the names in this map are the names written into
/// `reactions.arrow`, so a drift between them would make a reaction tally
/// silently fail to resolve rather than produce a wrong number.
///
/// The aliases are the part that cannot be derived. `"fission"` is the one
/// worth naming, because deriving this map naively drops it: MT 18's own name
/// is `"(n,fission)"`, and a tally asking for `"fission"` would stop resolving.
///
/// This map is deliberately NOT widened with aliases. The synthesized sums used
/// to be written into `reactions.arrow` as `(n,non-elastic)`, `(n,inelastic)`
/// and `(n,disappearance)`, none of which resolve here, so a label copied out
/// of a data file was rejected. That was fixed at the source rather than
/// papered over: the converter has written `(n,nonelastic)`, `(n,level)` and
/// `(n,disappear)` since #438, and the 2026-08-21 republish reissued every
/// library with them (#439). Adding aliases now would only re-admit the
/// spellings nothing produces any more.
pub static REACTION_MT: Lazy<HashMap<&'static str, i32>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Every MT the format names. The upper bound covers the level families
    // (up to 891) and the derived quantities (301, 444, 901).
    for mt in 1..1000 {
        if let Some(name) = endf::reaction_name(mt) {
            // Leaked deliberately: the map is a process-lifetime static, the
            // level-family names are formatted rather than literals, and the
            // key type stays `&'static str` so no caller has to change.
            map.insert(&*Box::leak(name.into_boxed_str()), mt);
        }
    }

    // The two bare forms, which are not any reaction's own name. These are the
    // whole alias set, and `tests/data_tests.rs` holds it to exactly that: an
    // MT with two names has no single inverse, so every addition here has to
    // be a deliberate one.
    //
    // `endf::reaction_mt` accepts four more (`total`, `elastic`, `absorption`,
    // `capture`) and this does not, because widening the set of names a tally
    // accepts is a separate decision from removing a duplicated table.
    map.insert("nonelastic", 3);
    map.insert("fission", 18);

    map
});

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::{ATOMIC_MASS, NATURAL_ABUNDANCE, REACTION_MT};

    #[test]
    fn atomic_mass_table_loads_from_embedded_data() {
        assert_eq!(ATOMIC_MASS.len(), 3558);
        // Exact literals preserved through the .txt round-trip.
        assert_eq!(ATOMIC_MASS["Fe56"], 55.934936);
        assert_eq!(ATOMIC_MASS["Ac205"], 205.015144);
        assert_eq!(ATOMIC_MASS["U238"], 238.050787);
    }

    #[test]
    fn natural_abundance_table_loads_from_embedded_data() {
        // The source had Li6/Li7 inserted twice; the HashMap keeps the last
        // (more precise) value, so 291 lines collapse to 289 unique keys.
        assert_eq!(NATURAL_ABUNDANCE.len(), 289);
        assert_eq!(NATURAL_ABUNDANCE["Fe56"], 0.91754);
        assert_eq!(NATURAL_ABUNDANCE["Li6"], 0.07589);
        assert_eq!(NATURAL_ABUNDANCE["Li7"], 0.92411);
    }

    /// The aliases a derived table drops.
    ///
    /// `REACTION_MT` is built from `endf::reaction_name`, which gives each MT
    /// one name, so anything else has to be added back. `"fission"` is the
    /// trap: MT 18's own name is `"(n,fission)"`, and a naive rebuild loses
    /// the short form that every fission tally is written with.
    #[test]
    fn the_aliases_survive_the_derivation() {
        for (name, mt) in [("nonelastic", 3), ("fission", 18)] {
            assert_eq!(
                REACTION_MT.get(name).copied(),
                Some(mt),
                "{name:?} must resolve to MT {mt}"
            );
        }

        // And the canonical names the derivation supplies, including a level
        // family member, which is formatted rather than listed.
        for (name, mt) in [
            ("(n,elastic)", 2),
            ("(n,2n)", 16),
            ("(n,fission)", 18),
            ("(n,gamma)", 102),
            ("(n,n3)", 53),
            ("(n,p0)", 600),
            ("heating", 301),
            ("heating-local", 901),
        ] {
            assert_eq!(
                REACTION_MT.get(name).copied(),
                Some(mt),
                "{name:?} must resolve to MT {mt}"
            );
        }

        // The table used to be written out by hand with 414 entries. Fewer
        // than that means the derivation lost something.
        assert!(
            REACTION_MT.len() >= 414,
            "REACTION_MT has {} entries, fewer than the 414 the hand-written \
             table had; the derivation dropped something",
            REACTION_MT.len()
        );
    }

    /// The table had `Li6` and `Li7` twice, once at the top out of alphabetical
    /// order and once in place. `parse_f64_table` collects into a `HashMap`, so
    /// the later line won and the surviving 0.07589 was the right one, but that
    /// is luck: a first-wins parser or a reordered file would have shifted Li6
    /// by +0.013% silently.
    #[test]
    fn the_natural_abundance_table_has_no_duplicate_nuclides() {
        let mut seen = std::collections::HashSet::new();
        let mut duplicated: Vec<&str> = include_str!("natural_abundance.txt")
            .lines()
            .filter_map(|line| line.split_whitespace().next())
            .filter(|name| !seen.insert(*name))
            .collect();
        duplicated.sort_unstable();
        assert!(
            duplicated.is_empty(),
            "duplicate entries would make the value depend on parser order: {duplicated:?}"
        );
    }

    #[test]
    fn every_element_abundance_sums_to_one() {
        let mut by_element: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        for (nuclide, abundance) in NATURAL_ABUNDANCE.iter() {
            // `Ta180_m1` is tantalum, and its abundance counts towards it.
            let stem = nuclide.split('_').next().unwrap_or(nuclide);
            let symbol: String = stem.chars().take_while(|c| c.is_alphabetic()).collect();
            *by_element.entry(symbol).or_insert(0.0) += abundance;
        }
        assert!(!by_element.is_empty());
        for (symbol, sum) in &by_element {
            assert!(
                (sum - 1.0).abs() < 1.0e-3,
                "{symbol} abundances sum to {sum}, not 1"
            );
        }
    }

}
