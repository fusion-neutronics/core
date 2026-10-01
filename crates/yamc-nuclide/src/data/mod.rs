//! Data module for nuclear data and dose coefficients
//!
//! This module provides access to nuclear data including dose conversion coefficients
//! from ICRP publications.
#![cfg_attr(rustfmt, rustfmt_skip)]

pub mod effective_dose;
pub mod photon_attenuation;

pub use effective_dose::{ambient_dose_coefficients, dose_coefficients, DoseDataSource, DoseGeometry, DoseParticle};
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
/// Columns after the value are ignored: `natural_abundance.txt` carries the
/// rest of its TICE row there, read by [`NATURAL_ABUNDANCE_RECORDS`].
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
/// abundance by atom fraction, from Table 1 of Meija et al., "Isotopic
/// compositions of the elements 2013 (IUPAC Technical Report)", Pure Appl.
/// Chem. 88(3), 293-306 (2016), doi:10.1515/pac-2015-0503 (© IUPAC, De Gruyter
/// 2016). The value is column 9, the representative abundance, except for the
/// 12 elements where column 9 is an interval (H, Li, B, C, N, O, Mg, Si, S, Cl,
/// Br, Tl), which have no single value there and take the column 6 best
/// measurement instead. Mononuclidic elements are 1.0. The rest of each row,
/// uncertainties included, is in [`NATURAL_ABUNDANCE_RECORDS`].
///
/// Regenerate with: python scripts/gen_natural_abundance.py
pub static NATURAL_ABUNDANCE: Lazy<HashMap<&'static str, f64>> =
    Lazy::new(|| parse_f64_table(include_str!("natural_abundance.txt")));

/// Column 9 of TICE 2013: the representative isotopic abundance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RepresentativeAbundance {
    /// A value, with the uncertainty printed after it. TICE says the
    /// uncertainty covers the probable variation between materials as well as
    /// measurement error, and gives no coverage factor for it. `None` for a
    /// mononuclidic element, which the table prints as a bare 1.
    Value { value: f64, uncertainty: Option<f64> },
    /// The interval `[low, high]` that the 12 elements with interval atomic
    /// weights are given in place of a value. It is the observed range in
    /// normal materials, not a probability distribution.
    Interval { low: f64, high: f64 },
}

/// Columns 4, 5, 6 and 9 of one isotope's row of TICE 2013 Table 1. Columns 7
/// (reference) and 8 (material) are not transcribed.
///
/// Every field is published data; nothing is derived, except that TICE prints
/// the column 5 annotations and the column 6 coverage and calibration once per
/// element and they are repeated here onto each of its isotopes. `None` means
/// the table leaves the field empty, which is "not stated", never zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NaturalAbundanceRecord {
    /// Column 9.
    pub representative: RepresentativeAbundance,
    /// Column 4: the observed interval `(low, high)` of natural variation,
    /// given only where one has been reliably established.
    pub observed_interval: Option<(f64, f64)>,
    /// Column 6: the best measurement of the abundance, made on a single
    /// terrestrial material.
    pub best_measurement: f64,
    /// Column 6 uncertainty, at the coverage in `best_measurement_coverage`.
    pub best_measurement_uncertainty: Option<f64>,
    /// Column 6 coverage: the factor k, then `s` (standard
    /// deviation), `se` (standard error) or `uc` (combined uncertainty), e.g.
    /// `"2s"`. The table prints `"n/a"` for C and N, and `"9uc"` for Mg.
    /// Printed on the element's first row only; it applies to every isotope
    /// of that element's best measurement.
    pub best_measurement_coverage: Option<&'static str>,
    /// Column 6 calibration flag: `'C'` fully, `'F'` partially, `'N'` not
    /// calibrated. Printed on the element's first row only, like the coverage.
    pub best_measurement_calibration: Option<char>,
    /// Column 5 annotations for the element, comma separated from `g`
    /// (geologically exceptional specimens), `m` (modified commercial
    /// material) and `r` (range prevents a more precise value).
    pub annotations: Option<&'static str>,
}

/// Parse `natural_abundance.txt` into its TICE 2013 records. The columns are
/// the ones `scripts/gen_natural_abundance.py` writes, `-` for an empty field.
fn parse_abundance_records(data: &'static str) -> HashMap<&'static str, NaturalAbundanceRecord> {
    fn field(cell: &'static str) -> Option<&'static str> {
        (cell != "-").then_some(cell)
    }
    fn number(cell: &'static str) -> Option<f64> {
        field(cell).map(|c| c.parse::<f64>().expect("malformed f64 in natural_abundance.txt"))
    }
    fn pair(low: &'static str, high: &'static str) -> Option<(f64, f64)> {
        match (number(low), number(high)) {
            (Some(low), Some(high)) => Some((low, high)),
            (None, None) => None,
            _ => panic!("natural_abundance.txt has an interval with one bound"),
        }
    }
    data.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let cells: Vec<&'static str> = line.split_whitespace().collect();
            let [
                nuclide, _abundance,
                rep_value, rep_uncertainty, rep_low, rep_high,
                interval_low, interval_high,
                best, best_uncertainty, coverage, calibration,
                annotations,
            ] = cells[..] else {
                panic!("natural_abundance.txt line {line:?} does not have 13 columns");
            };
            let representative = match (number(rep_value), pair(rep_low, rep_high)) {
                (Some(value), None) => RepresentativeAbundance::Value {
                    value,
                    uncertainty: number(rep_uncertainty),
                },
                (None, Some((low, high))) => RepresentativeAbundance::Interval { low, high },
                _ => panic!("{nuclide}: column 9 must be exactly one of a value or an interval"),
            };
            let record = NaturalAbundanceRecord {
                representative,
                observed_interval: pair(interval_low, interval_high),
                best_measurement: number(best).expect("every row has a column 6 value"),
                best_measurement_uncertainty: number(best_uncertainty),
                best_measurement_coverage: field(coverage),
                best_measurement_calibration: field(calibration).map(|c| {
                    let mut chars = c.chars();
                    let flag = chars.next().expect("non-empty calibration flag");
                    assert!(chars.next().is_none(), "{nuclide}: calibration flag {c:?}");
                    flag
                }),
                annotations: field(annotations),
            };
            (nuclide, record)
        })
        .collect()
}

/// The TICE 2013 record behind each entry of [`NATURAL_ABUNDANCE`], keyed
/// the same way: the representative value or interval, the observed interval,
/// and the best measurement with its uncertainty, coverage and calibration.
///
/// Read-only reference data. Nothing in yamc samples or propagates these
/// uncertainties; turning them into a distribution needs choices TICE does not
/// make (a coverage factor for column 9, correlations between isotopes, how to
/// treat an interval).
pub static NATURAL_ABUNDANCE_RECORDS: Lazy<HashMap<&'static str, NaturalAbundanceRecord>> =
    Lazy::new(|| parse_abundance_records(include_str!("natural_abundance.txt")));

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
/// This map is deliberately NOT widened with aliases. The converter writes the
/// synthesized sums into `reactions.arrow` as `(n,nonelastic)`, `(n,level)` and
/// `(n,disappear)`, and the 2026-08-21 republish reissued every library with
/// them. Aliases for the older `(n,non-elastic)`, `(n,inelastic)` and
/// `(n,disappearance)` spellings would only re-admit labels nothing produces
/// any more.
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
    use super::{
        RepresentativeAbundance, ATOMIC_MASS, NATURAL_ABUNDANCE, NATURAL_ABUNDANCE_RECORDS,
        REACTION_MT,
    };

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
        // Every isotope TICE 2013 lists for a normal material, one row each.
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

    /// The value every lookup uses is the one the TICE row says it is:
    /// column 9 where that is a value, the column 6 best measurement where
    /// column 9 is an interval. Both sides come from the same file, so this is
    /// internal consistency only; `the_lookup_values_are_the_ones_shipped_before_tice`
    /// is the check against the values that were there before.
    #[test]
    fn the_lookup_value_is_the_tice_value_its_row_names() {
        assert_eq!(NATURAL_ABUNDANCE_RECORDS.len(), NATURAL_ABUNDANCE.len());
        let (mut representative, mut interval, mut mononuclidic) = (0, 0, 0);
        for (nuclide, &abundance) in NATURAL_ABUNDANCE.iter() {
            let record = NATURAL_ABUNDANCE_RECORDS[nuclide];
            let expected = match record.representative {
                RepresentativeAbundance::Value { value, uncertainty: None } => {
                    assert_eq!(value, 1.0, "{nuclide}: only a mononuclidic 1 has no uncertainty");
                    assert_eq!(record.best_measurement, 1.0, "{nuclide}");
                    mononuclidic += 1;
                    value
                }
                RepresentativeAbundance::Value { value, uncertainty: Some(_) } => {
                    representative += 1;
                    value
                }
                RepresentativeAbundance::Interval { low, high } => {
                    // TICE: for these elements columns 4 and 9 are identical.
                    assert_eq!(record.observed_interval, Some((low, high)), "{nuclide}");
                    interval += 1;
                    record.best_measurement
                }
            };
            assert_eq!(abundance.to_bits(), expected.to_bits(), "{nuclide}");
        }
        assert_eq!((representative, interval, mononuclidic), (239, 29, 21));
    }

    /// Every central value was already the TICE 2013 number before the
    /// uncertainty columns were added, and adding them changed none. This is
    /// an FNV-1a hash of the sorted (name, f64 bits) pairs taken from the
    /// table as it stood then, so a regeneration that moves any value, adds
    /// or drops a nuclide fails here rather than passing silently. Change the
    /// hash only for a deliberate change of source data.
    #[test]
    fn the_lookup_values_are_the_ones_shipped_before_tice() {
        let mut pairs: Vec<(&str, f64)> = NATURAL_ABUNDANCE.iter().map(|(k, v)| (*k, *v)).collect();
        pairs.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for (name, value) in &pairs {
            for byte in name.bytes().chain(value.to_bits().to_le_bytes()) {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        assert_eq!(pairs.len(), 289);
        assert_eq!(hash, 0x453c_7b51_e24c_2dc2, "a central abundance value changed");
    }

    /// The shipped table is what `scripts/gen_natural_abundance.py` makes of
    /// the checked-in transcription. The two files hold the same data, so a
    /// hand edit to either one, or a transcription fix without regenerating,
    /// fails here. This repeats the generator's mapping line by line.
    #[test]
    fn the_shipped_table_is_generated_from_the_tice_transcription() {
        // The only quoting in the file is around comma-separated annotations
        // such as "g,r", with no escaped quotes inside.
        fn split(line: &str) -> Vec<&str> {
            let mut cells = Vec::new();
            let (mut start, mut quoted) = (0, false);
            for (i, c) in line.char_indices() {
                match c {
                    '"' => quoted = !quoted,
                    ',' if !quoted => {
                        cells.push(line[start..i].trim_matches('"'));
                        start = i + 1;
                    }
                    _ => {}
                }
            }
            cells.push(line[start..].trim_matches('"'));
            cells
        }
        let csv = include_str!("tice_2013_table1.csv");
        let mut rows = csv.lines().filter(|line| !line.starts_with('#'));
        let header = split(rows.next().expect("CSV header"));
        let column = |name: &str| {
            header
                .iter()
                .position(|h| *h == name)
                .unwrap_or_else(|| panic!("CSV has no {name} column"))
        };
        let generated: Vec<String> = rows
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let cells = split(line);
                assert_eq!(cells.len(), header.len(), "{line:?}");
                let get = |name: &str| cells[column(name)];
                let (element, a) = (get("element"), get("a"));
                let name = if (element, a) == ("Ta", "180") {
                    "Ta180_m1".to_string()
                } else {
                    format!("{element}{a}")
                };
                let abundance = match get("representative_value") {
                    "" => get("best_value"),
                    value => value,
                };
                let mut out = vec![name, abundance.to_owned()];
                for field in [
                    "representative_value",
                    "representative_uncertainty",
                    "representative_low",
                    "representative_high",
                    "interval_low",
                    "interval_high",
                    "best_value",
                    "best_uncertainty",
                    "best_coverage",
                    "best_calibration",
                    "annotations",
                ] {
                    out.push(match get(field) {
                        "" => "-".to_owned(),
                        value => value.to_owned(),
                    });
                }
                out.join(" ")
            })
            .collect();
        let shipped: Vec<&str> = include_str!("natural_abundance.txt").lines().collect();
        assert_eq!(generated.len(), shipped.len());
        for (generated, shipped) in generated.iter().zip(&shipped) {
            assert_eq!(generated, shipped, "rerun python scripts/gen_natural_abundance.py");
        }
    }

    #[test]
    fn the_best_measurements_of_each_element_sum_to_one() {
        let mut by_element: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        for (nuclide, record) in NATURAL_ABUNDANCE_RECORDS.iter() {
            let symbol: String = nuclide.chars().take_while(|c| c.is_alphabetic()).collect();
            *by_element.entry(symbol).or_insert(0.0) += record.best_measurement;
        }
        for (symbol, sum) in &by_element {
            // Printed to 4 to 9 decimals; Sn's rounds to 1.00002.
            assert!((sum - 1.0).abs() < 5.0e-5, "{symbol} column 6 sums to {sum}");
        }
    }

    /// Rows read off the printed table.
    #[test]
    fn tice_rows_match_the_printed_table() {
        let fe58 = NATURAL_ABUNDANCE_RECORDS["Fe58"];
        assert_eq!(
            fe58.representative,
            RepresentativeAbundance::Value { value: 0.00282, uncertainty: Some(0.00012) }
        );
        assert_eq!(fe58.observed_interval, Some((0.00281, 0.00282)));
        assert_eq!(fe58.best_measurement, 0.002819);
        assert_eq!(fe58.best_measurement_uncertainty, Some(0.000027));
        assert_eq!(fe58.best_measurement_coverage, Some("2s"));
        assert_eq!(fe58.best_measurement_calibration, Some('C'));
        assert_eq!(fe58.annotations, None);

        let w186 = NATURAL_ABUNDANCE_RECORDS["W186"];
        assert_eq!(
            w186.representative,
            RepresentativeAbundance::Value { value: 0.2843, uncertainty: Some(0.0019) }
        );
        assert_eq!(w186.observed_interval, None);
        assert_eq!(w186.best_measurement, 0.284259);
        assert_eq!(w186.best_measurement_uncertainty, Some(0.000062));
        assert_eq!(w186.best_measurement_coverage, Some("1s"));
        assert_eq!(w186.best_measurement_calibration, Some('N'));

        let li6 = NATURAL_ABUNDANCE_RECORDS["Li6"];
        assert_eq!(
            li6.representative,
            RepresentativeAbundance::Interval { low: 0.019, high: 0.078 }
        );
        assert_eq!(li6.best_measurement, 0.07589);
        assert_eq!(li6.best_measurement_uncertainty, Some(0.00024));
        assert_eq!(li6.annotations, Some("m"));

        // Printed as 4.6 x 10^-10, the one bound not in fixed notation.
        assert_eq!(NATURAL_ABUNDANCE_RECORDS["He3"].observed_interval, Some((4.6e-10, 0.000041)));
        assert_eq!(NATURAL_ABUNDANCE_RECORDS["He3"].annotations, Some("g,r"));
        assert_eq!(NATURAL_ABUNDANCE_RECORDS["Mg24"].best_measurement_coverage, Some("9uc"));
        assert_eq!(NATURAL_ABUNDANCE_RECORDS["N14"].best_measurement_coverage, Some("n/a"));

        let co59 = NATURAL_ABUNDANCE_RECORDS["Co59"];
        assert_eq!(
            co59.representative,
            RepresentativeAbundance::Value {
                value: 1.0,
                uncertainty: None
            }
        );
        assert_eq!(co59.best_measurement_uncertainty, None);
        assert_eq!(co59.best_measurement_coverage, None);
        assert_eq!(co59.best_measurement_calibration, None);

        // TICE's "Ta 180" row, under the isomer name the nuclear data uses.
        assert_eq!(
            NATURAL_ABUNDANCE_RECORDS["Ta180_m1"].representative,
            RepresentativeAbundance::Value { value: 0.0001201, uncertainty: Some(0.0000032) }
        );
    }

}
