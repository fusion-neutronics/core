//! Temperature labels, and the one place they become numbers.
//!
//! A temperature is carried through the format and the runtime as a label, not
//! a number: `"294"` in memory, `"294K"` on disk, and it is the key that
//! selects a cross-section set. Physics kernels need the value in Kelvin, so
//! somewhere the label has to be parsed, and that somewhere is here.
//!
//! It exists because it was in two places and they disagreed. `Material` cached
//! a `temperature_k` that `set_temperature` never updated, so the CPU free-gas
//! kernel ran every material at 294 K, while the GPU parsed the label itself
//! and got it right (issue #478). Two parsers is one more than can be kept in
//! step, and the failure is silent: 294 K is the common case, so every test
//! passed.

/// The label with its `K` suffix and surrounding whitespace removed.
///
/// The on-disk form is `"294K"` and the in-memory form is `"294"`. Both appear
/// in comparisons against a label the caller supplied, so normalise rather
/// than assuming which side is which.
pub fn strip_k(label: &str) -> &str {
    label.trim().trim_end_matches('K').trim()
}

/// Temperature in Kelvin, or `None` if the label is not a number.
///
/// Accepts `"294"`, `"294K"` and `"294.0"`. Returns `None` for the empty
/// label, which is not an error: an unset temperature means "resolve it from
/// whatever the nuclide data offers" (see `Material::resolve_temperature`).
pub fn label_to_kelvin(label: &str) -> Option<f64> {
    strip_k(label).parse::<f64>().ok()
}

/// The temperature assumed when no label has been resolved yet.
///
/// Room temperature, and the value `Material::new` has always started from.
/// Reachable only before a temperature is resolved: `resolve_temperature`
/// rejects a label the nuclide data does not offer, and every label the data
/// offers parses.
pub const DEFAULT_TEMPERATURE_K: f64 = 294.0;

/// Kelvin from a label, falling back to [`DEFAULT_TEMPERATURE_K`].
///
/// The infallible form, for callers on a path with no way to report an error.
pub fn label_to_kelvin_or_default(label: &str) -> f64 {
    label_to_kelvin(label).unwrap_or(DEFAULT_TEMPERATURE_K)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_both_written_forms() {
        assert_eq!(label_to_kelvin("294"), Some(294.0));
        assert_eq!(label_to_kelvin("294K"), Some(294.0));
        assert_eq!(label_to_kelvin("294.0"), Some(294.0));
        assert_eq!(label_to_kelvin(" 900K "), Some(900.0));
    }

    #[test]
    fn empty_and_nonsense_labels_have_no_value() {
        assert_eq!(label_to_kelvin(""), None);
        assert_eq!(label_to_kelvin("hot"), None);
        assert_eq!(label_to_kelvin_or_default(""), DEFAULT_TEMPERATURE_K);
    }

    #[test]
    fn strip_k_is_idempotent_across_both_forms() {
        assert_eq!(strip_k("294K"), "294");
        assert_eq!(strip_k("294"), "294");
        assert_eq!(strip_k(strip_k("294K")), "294");
    }
}
