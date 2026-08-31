//! Free functions for building nuclide compositions from elements, enriched
//! elements, and chemical formulas.
//!
//! These were previously methods on [`Material`] but are now standalone so they
//! can be composed freely before constructing a material.

use crate::data::{ELEMENT_NAMES, ELEMENT_NUCLIDES, NATURAL_ABUNDANCE};
use std::collections::HashMap;

/// Validate that a fraction type string is `"atom"` or `"mass"`.
///
/// Exactly one accepted spelling per type; anything else (including the old
/// `"weight"`) fails hard.
pub fn validate_fraction_type(input: &str) -> Result<(), String> {
    match input {
        "atom" | "mass" => Ok(()),
        other => Err(format!("fraction must be 'atom' or 'mass', got '{other}'")),
    }
}

/// Validate that a nuclide name contains a mass number (at least one digit).
///
/// Valid formats: Fe56, U235, Am241m, H3, etc.
/// Invalid formats: Fe, U, Li (element symbols without mass numbers).
pub fn validate_nuclide_name(name: &str) -> Result<(), String> {
    if !name.chars().any(|c| c.is_ascii_digit()) {
        return Err(format!(
            "Invalid nuclide name '{name}'. Nuclide names must include mass number \
             (e.g., 'Fe56', 'U235m'). Use expand_element() for natural element composition."
        ));
    }
    Ok(())
}

/// Resolve an element input string (symbol like `"Li"` or name like `"lithium"`)
/// to its canonical symbol. Case-sensitive exact match.
pub fn resolve_element_symbol(element: impl AsRef<str>) -> Result<String, String> {
    let input = element.as_ref().trim();

    // Try to match as symbol (case-sensitive, exact match)
    for symbol in ELEMENT_NAMES.keys() {
        if *symbol == input {
            return Ok(symbol.to_string());
        }
    }
    // If not found as symbol, try to match as name (case-sensitive, exact match)
    for (symbol, name) in ELEMENT_NAMES.iter() {
        if *name == input {
            return Ok(symbol.to_string());
        }
    }
    Err(format!(
        "Element '{}' is not a recognized element symbol or name \
         (case-sensitive, must match exactly)",
        element.as_ref()
    ))
}

/// Look up the atomic mass (in u) for a nuclide from the AME2020 table.
///
/// AME2020 indexes ground states only. For metastables (e.g. `Ta180_m1`,
/// `Ag110_m1`) we fall back to the ground-state mass -- the isomeric
/// excitation (typically tens of keV to a few MeV) is negligible against
/// the atomic mass (≈ A × 931 MeV/u), and the reference convention is the same
/// (the metastable state takes the ground-state atomic mass).
pub fn atomic_mass(nuclide: &str) -> Result<f64, String> {
    if let Some(&mass) = crate::data::ATOMIC_MASS.get(nuclide) {
        return Ok(mass);
    }
    // Metastable fallback: strip the `_m<n>` suffix and retry on the ground state.
    if let Some((base, suffix)) = nuclide.rsplit_once('_') {
        if suffix.starts_with('m') && suffix[1..].chars().all(|c| c.is_ascii_digit()) {
            if let Some(&mass) = crate::data::ATOMIC_MASS.get(base) {
                return Ok(mass);
            }
        }
    }
    Err(format!(
        "Nuclide '{nuclide}' not found in AME2020 atomic mass table"
    ))
}

/// The atomic number of a nuclide, e.g. `Fe56` gives 26 and `Ta180_m1` gives
/// 73.
///
/// Metastable states carry the atomic number of their ground state, which is
/// what makes this usable for anything indexed by element -- attenuation
/// coefficients, photon interaction data -- where the isomeric state is not a
/// distinct entry.
pub fn atomic_number(nuclide: &str) -> Result<u32, String> {
    let (z, _a, _m) = endf::zam(nuclide).map_err(|_| {
        format!("Nuclide '{nuclide}' is not a recognized nuclide name (e.g. 'Fe56', 'Ta180_m1')")
    })?;
    Ok(z)
}

/// Expand a natural element into its isotopes with natural abundances.
///
/// Returns a map of nuclide name → fraction, using the given `fraction_type`:
/// - `"atom"`: natural abundances (atom fractions) are used directly.
/// - `"mass"`: abundances are converted to mass fractions.
pub fn expand_element(
    element: &str,
    fraction: f64,
    fraction_type: &str,
) -> Result<HashMap<String, f64>, String> {
    validate_fraction_type(fraction_type)?;
    if fraction <= 0.0 {
        return Err("Fraction must be positive".into());
    }

    let element_sym = resolve_element_symbol(element)?;

    let isotopes_vec = ELEMENT_NUCLIDES.get(element_sym.as_str()).ok_or_else(|| {
        format!("Element '{element_sym}' not found in the natural abundance database")
    })?;

    let mut nuclides = HashMap::new();

    if fraction_type == "mass" {
        // Natural abundances are atom fractions -- convert to mass fractions.
        let mut weighted: Vec<(&str, f64)> = Vec::new();
        let mut total_mass = 0.0;
        for &isotope in isotopes_vec.iter() {
            if let Some(&abundance) = NATURAL_ABUNDANCE.get(isotope) {
                let mass = atomic_mass(isotope).map_err(|e| {
                    format!("Cannot convert to weight fractions for element '{element_sym}': {e}")
                })?;
                let w = abundance * mass;
                total_mass += w;
                weighted.push((isotope, w));
            }
        }
        for (isotope, w) in weighted {
            let weight_frac = w / total_mass;
            let isotope_fraction = fraction * weight_frac;
            if isotope_fraction > 0.0 {
                validate_nuclide_name(isotope)?;
                nuclides.insert(isotope.to_string(), isotope_fraction);
            }
        }
    } else {
        // Atom fractions -- use natural abundances directly
        for &isotope in isotopes_vec.iter() {
            if let Some(&abundance) = NATURAL_ABUNDANCE.get(isotope) {
                let isotope_fraction = fraction * abundance;
                if isotope_fraction > 0.0 {
                    validate_nuclide_name(isotope)?;
                    nuclides.insert(isotope.to_string(), isotope_fraction);
                }
            }
        }
    }

    Ok(nuclides)
}

/// Expand a natural element with enrichment of a specific isotope.
///
/// Only supported for elements with exactly 2 naturally-occurring isotopes
/// (e.g. Li, B, Cl, Cu, etc.).
///
/// # Arguments
/// * `element` - Element symbol (e.g. `"Li"`) or name (e.g. `"lithium"`)
/// * `fraction` - Overall fraction for this element
/// * `enrichment` - Enrichment of the target isotope in percent (0, 100]
/// * `enrichment_target` - Nuclide name to enrich (e.g. `"Li6"`)
/// * `enrichment_type` - `"atom"` for atom percent or `"mass"` for mass percent
/// * `fraction_type` - Material's fraction type (`"atom"` or `"mass"`)
pub fn expand_element_enriched(
    element: &str,
    fraction: f64,
    enrichment: f64,
    enrichment_target: &str,
    enrichment_type: &str,
    fraction_type: &str,
) -> Result<HashMap<String, f64>, String> {
    validate_fraction_type(enrichment_type)?;
    validate_fraction_type(fraction_type)?;
    if fraction <= 0.0 {
        return Err("Fraction must be positive".into());
    }
    if enrichment <= 0.0 || enrichment > 100.0 {
        return Err(format!(
            "Enrichment must be between 0 (exclusive) and 100 (inclusive), got {enrichment}"
        ));
    }

    let element_sym = resolve_element_symbol(element)?;

    let isotopes_vec = ELEMENT_NUCLIDES.get(element_sym.as_str()).ok_or_else(|| {
        format!("Element '{element_sym}' not found in the natural abundance database")
    })?;

    // Must have exactly 2 naturally-occurring isotopes
    if isotopes_vec.len() != 2 {
        return Err(format!(
            "Element '{}' has {} naturally-occurring isotopes. \
             Enrichment is only supported for elements with exactly 2. \
             Please enter isotopic composition manually.",
            element_sym,
            isotopes_vec.len()
        ));
    }

    // enrichment_target must be one of the two isotopes
    if !isotopes_vec.contains(&enrichment_target) {
        return Err(format!(
            "enrichment_target '{}' is not a naturally-occurring isotope of {} ({:?})",
            enrichment_target, element_sym, isotopes_vec
        ));
    }

    let target = enrichment_target;
    let other = if isotopes_vec[0] == target {
        isotopes_vec[1]
    } else {
        isotopes_vec[0]
    };

    let target_mass = atomic_mass(target)?;
    let other_mass = atomic_mass(other)?;

    // Step 1: Get atom fractions from the enrichment value
    let (a_target, a_other) = if enrichment_type == "atom" {
        (enrichment / 100.0, 1.0 - enrichment / 100.0)
    } else {
        // enrichment_type == "mass": convert mass percent → atom fractions
        let w_t = enrichment / 100.0;
        let w_o = 1.0 - w_t;
        let a_t = w_t / target_mass;
        let a_o = w_o / other_mass;
        let total = a_t + a_o;
        (a_t / total, a_o / total)
    };

    // Step 2: Convert to the material's fraction_type for storage
    let (frac_target, frac_other) = if fraction_type == "mass" {
        // Convert atom fractions → mass fractions
        let w_t = a_target * target_mass;
        let w_o = a_other * other_mass;
        let total = w_t + w_o;
        (w_t / total, w_o / total)
    } else {
        // Already atom fractions
        (a_target, a_other)
    };

    let mut nuclides = HashMap::new();
    validate_nuclide_name(target)?;
    validate_nuclide_name(other)?;
    nuclides.insert(target.to_string(), fraction * frac_target);
    nuclides.insert(other.to_string(), fraction * frac_other);
    Ok(nuclides)
}

/// Parse a chemical formula and return element fractions.
///
/// Supports formulas like `"H2O"`, `"Li4SiO4"`, `"(NH4)2SO4"`.
/// Nested parentheses are supported.
///
/// Enrichment parameters, if provided, are forwarded to the matching
/// element only (the element whose symbol matches `enrichment_target`'s
/// element prefix).
pub fn expand_formula(
    formula: &str,
    fraction_type: &str,
    enrichment: Option<f64>,
    enrichment_target: Option<&str>,
    enrichment_type: Option<&str>,
) -> Result<HashMap<String, f64>, String> {
    if formula.contains('.') {
        return Err("Formula cannot contain '.' (non-integer multipliers)".into());
    }
    if formula.is_empty() {
        return Err("Formula cannot be empty".into());
    }

    // Tokenize: element symbols [A-Z][a-z]*, digits, parentheses
    let tokens = tokenize_formula(formula)?;

    // Parse element counts using a stack
    let counts = parse_formula_tokens(&tokens)?;

    // Normalize to fractions summing to 1.0
    let total: f64 = counts.values().sum();
    if total <= 0.0 {
        return Err("Formula produced no elements".into());
    }

    // Determine which element gets enrichment (if any)
    let enrichment_element: Option<String> = enrichment_target.map(|target| {
        target
            .chars()
            .take_while(|c| c.is_alphabetic())
            .collect::<String>()
    });

    let mut nuclides = HashMap::new();

    for (element, &count) in &counts {
        let frac = count / total;
        let is_enrichment_target = enrichment_element.as_ref() == Some(element);

        let element_nuclides = if is_enrichment_target {
            if let (Some(enr), Some(target)) = (enrichment, enrichment_target) {
                let et = enrichment_type.unwrap_or("atom");
                expand_element_enriched(element, frac, enr, target, et, fraction_type)?
            } else {
                expand_element(element, frac, fraction_type)?
            }
        } else {
            expand_element(element, frac, fraction_type)?
        };

        for (nuc, frac_val) in element_nuclides {
            *nuclides.entry(nuc).or_insert(0.0) += frac_val;
        }
    }

    Ok(nuclides)
}

/// Merge a map of nuclides into an existing nuclide map (additive).
pub fn merge_nuclides(target: &mut HashMap<String, f64>, source: &HashMap<String, f64>) {
    for (nuc, &frac) in source {
        *target.entry(nuc.clone()).or_insert(0.0) += frac;
    }
}

/// Tokenize a chemical formula into element symbols, digits, and brackets.
fn tokenize_formula(formula: &str) -> Result<Vec<String>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = formula.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_uppercase() {
            // Element symbol: uppercase letter followed by optional lowercase
            let mut sym = String::new();
            sym.push(c);
            i += 1;
            while i < chars.len() && chars[i].is_ascii_lowercase() {
                sym.push(chars[i]);
                i += 1;
            }
            // Validate this is a known element
            if !ELEMENT_NAMES.contains_key(sym.as_str()) {
                return Err(format!("Unknown element symbol '{sym}' in formula"));
            }
            tokens.push(sym);
        } else if c.is_ascii_digit() {
            // Number
            let mut num = String::new();
            while i < chars.len() && chars[i].is_ascii_digit() {
                num.push(chars[i]);
                i += 1;
            }
            tokens.push(num);
        } else if c == '(' || c == ')' {
            tokens.push(c.to_string());
            i += 1;
        } else {
            return Err(format!("Invalid character '{c}' in formula '{formula}'"));
        }
    }

    Ok(tokens)
}

/// Parse tokenized formula into element counts using a stack.
fn parse_formula_tokens(tokens: &[String]) -> Result<HashMap<String, f64>, String> {
    let mut stack: Vec<HashMap<String, f64>> = vec![HashMap::new()];

    let mut i = 0;
    while i < tokens.len() {
        let tok = &tokens[i];
        if tok == "(" {
            stack.push(HashMap::new());
            i += 1;
        } else if tok == ")" {
            if stack.len() < 2 {
                return Err("Unbalanced parentheses in formula".into());
            }
            let top = stack.pop().unwrap();
            // Check for a multiplier after ')'
            let multiplier: f64 = if i + 1 < tokens.len() {
                if let Ok(n) = tokens[i + 1].parse::<f64>() {
                    i += 1; // consume the number
                    n
                } else {
                    1.0
                }
            } else {
                1.0
            };
            let parent = stack.last_mut().unwrap();
            for (elem, count) in top {
                *parent.entry(elem).or_insert(0.0) += count * multiplier;
            }
            i += 1;
        } else if tok.chars().next().unwrap().is_ascii_uppercase() {
            // Element symbol -- check for following number
            let count: f64 = if i + 1 < tokens.len() {
                if let Ok(n) = tokens[i + 1].parse::<f64>() {
                    i += 1; // consume the number
                    n
                } else {
                    1.0
                }
            } else {
                1.0
            };
            let current = stack.last_mut().unwrap();
            *current.entry(tok.clone()).or_insert(0.0) += count;
            i += 1;
        } else {
            // Standalone number not preceded by element or ')'
            return Err(format!("Unexpected token '{tok}' in formula"));
        }
    }

    if stack.len() != 1 {
        return Err("Unbalanced parentheses in formula".into());
    }

    Ok(stack.into_iter().next().unwrap())
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_mass_ground_state() {
        // Sanity: a stable, common nuclide must be present.
        let m_fe56 = atomic_mass("Fe56").expect("Fe56 should be in AME2020");
        assert!((m_fe56 - 55.93493633).abs() < 1e-6);
    }

    /// Regression: metastable nuclides (`<base>_m<n>`) must fall back to
    /// the ground-state mass. AME2020 indexes ground states only; without
    /// this fallback yamc panics when natural-abundance expansion produces
    /// `Ta180_m1` (Ta has ~0.012 % natural Ta180_m1) -- see the broomstick
    /// photon-transport sweep in yamc-verification. The reference convention is the same
    /// convention (the metastable state takes the ground-state atomic mass).
    #[test]
    fn atomic_mass_falls_back_for_metastables() {
        let ta180 = atomic_mass("Ta180").expect("Ta180 should be in AME2020");
        let ta180_m1 = atomic_mass("Ta180_m1").expect("Ta180_m1 should fall back to Ta180's mass");
        assert_eq!(ta180, ta180_m1);

        // Multi-digit metastable suffix (e.g. _m12) must also fall back.
        let ag110 = atomic_mass("Ag110").expect("Ag110 should be in AME2020");
        let ag110_m1 = atomic_mass("Ag110_m1").expect("Ag110_m1 should fall back to Ag110's mass");
        assert_eq!(ag110, ag110_m1);
    }

    #[test]
    fn atomic_mass_unknown_nuclide_errors() {
        let err = atomic_mass("Xx999").unwrap_err();
        assert!(err.contains("Xx999"));
        assert!(err.contains("AME2020"));
    }

    #[test]
    fn atomic_mass_unknown_metastable_still_errors() {
        // Suffix that isn't `_m<digits>` must NOT fall back -- keep failures loud
        // for genuine typos like `Fe56_foo`.
        assert!(atomic_mass("Fe56_foo").is_err());
        // Base nuclide truly absent → error even with metastable suffix.
        assert!(atomic_mass("Xx999_m1").is_err());
    }
}
