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

/// How far off an available temperature a request may be and still be treated
/// as that temperature rather than blended.
///
/// Not a user-facing tolerance and deliberately far too small to be one. It
/// exists because a request of `"293.99999"` against a library carrying `294`
/// would otherwise synthesise a whole temperature at a blend weight of
/// essentially zero, duplicating data already in memory to reproduce it. There
/// is no `nearest` behaviour hiding here: a request a hundredth of a Kelvin
/// away still blends.
pub const SNAP_TOLERANCE_K: f64 = 1e-6;

/// The interpolation variable, and the only place it is chosen.
///
/// Linear in T, which is what OpenMC uses. The argument for
/// `sqrt(T)` is that the Doppler width scales that way, so it might track
/// resonance broadening better across a gap as wide as 294 K to 600 K; the
/// argument against is that a pointwise comparison against NJOY-broadened data
/// favours linear in T, and the resonance integral cannot separate the two at
/// all. It is not settled, which is exactly why it lives in one function:
/// switching is a one-line change here and nothing else in the tree computes a
/// second weight.
///
/// Returns the fraction of the UPPER temperature. Clamped, so a caller that
/// has already established the bracket cannot produce a weight outside it
/// through rounding.
pub fn blend_weight(t_lo_k: f64, t_hi_k: f64, t_k: f64) -> f64 {
    if t_hi_k <= t_lo_k {
        return 0.0;
    }
    ((t_k - t_lo_k) / (t_hi_k - t_lo_k)).clamp(0.0, 1.0)
}

/// Where a requested temperature's cross sections come from.
///
/// Indices, not labels, and into the slice that was passed to [`resolve`]. The
/// caller's parallel `reactions` / `fast_xs` / `urr_data` vectors are indexed
/// the same way, so returning a label would only make the caller look it up
/// again.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemperatureSource {
    /// The data carries this temperature. Use it directly.
    Exact { idx: usize },
    /// The request falls strictly between two available temperatures.
    Blend {
        lo_idx: usize,
        hi_idx: usize,
        /// Fraction of the upper temperature, from [`blend_weight`].
        weight: f64,
    },
}

/// Why a temperature could not be resolved.
///
/// Out of range is an error and never a silent clamp. OpenMC snaps a request
/// outside the loaded range to the nearest bound; doing that here would answer
/// a question about 1500 K with data at 2500 K and say nothing, and the
/// difference would show up as an unexplained discrepancy rather than as a
/// message.
#[derive(Debug, Clone, PartialEq)]
pub enum TemperatureError {
    /// The label is not a number.
    Unparseable { label: String },
    /// The data carries no temperatures at all to bracket against.
    NoTemperatures { label: String },
    /// The request is below the lowest or above the highest available.
    OutOfRange {
        label: String,
        requested_k: f64,
        lowest_k: f64,
        highest_k: f64,
        available: Vec<String>,
    },
}

impl std::fmt::Display for TemperatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemperatureError::Unparseable { label } => {
                write!(f, "Temperature '{label}' is not a number")
            }
            TemperatureError::NoTemperatures { label } => write!(
                f,
                "Temperature '{label}' cannot be resolved: this nuclear data \
                 carries no temperatures"
            ),
            TemperatureError::OutOfRange {
                label,
                requested_k,
                lowest_k,
                highest_k,
                available,
            } => write!(
                f,
                "Temperature '{label}' ({requested_k} K) is outside the range \
                 this nuclear data covers, {lowest_k} K to {highest_k} K. \
                 Available temperatures: [{}]",
                available.join(", ")
            ),
        }
    }
}

impl std::error::Error for TemperatureError {}

/// Resolve a requested temperature against the set the data carries.
///
/// Three rules, no parameters:
///
/// - the request matches an available temperature: use it,
/// - the request falls strictly between two: blend them,
/// - anything else: error, listing what is available.
///
/// A single-temperature library is the degenerate third case and needs no
/// special path.
///
/// Sorts `available` by parsed Kelvin internally rather than trusting the
/// caller's order. `available_temperatures` happens to be numerically sorted by
/// the Arrow loader, but a caller holding a set, or a list in the published
/// files' lexicographic order (`"1200"` before `"250"`), must still get the
/// right bracket. Entries that do not parse are dropped rather than compared as
/// strings, so a stray label can never become a bracket endpoint.
pub fn resolve(
    requested: &str,
    available: &[String],
) -> Result<TemperatureSource, TemperatureError> {
    let label = strip_k(requested);
    let Some(requested_k) = label_to_kelvin(label) else {
        return Err(TemperatureError::Unparseable {
            label: label.to_string(),
        });
    };

    // (kelvin, index into `available`). Sorted by kelvin, so the bracket search
    // is a walk, while the indices still point back at the caller's order.
    let mut ladder: Vec<(f64, usize)> = available
        .iter()
        .enumerate()
        .filter_map(|(i, t)| label_to_kelvin(t).map(|k| (k, i)))
        .collect();
    ladder.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    if ladder.is_empty() {
        return Err(TemperatureError::NoTemperatures {
            label: label.to_string(),
        });
    }

    // Exact first, and before the range check, so the endpoints resolve to
    // themselves rather than to a zero-width blend.
    if let Some(&(_, idx)) = ladder
        .iter()
        .find(|(k, _)| (k - requested_k).abs() <= SNAP_TOLERANCE_K)
    {
        return Ok(TemperatureSource::Exact { idx });
    }

    let (lowest_k, _) = ladder[0];
    let (highest_k, _) = ladder[ladder.len() - 1];
    if requested_k < lowest_k || requested_k > highest_k {
        let mut available: Vec<String> = ladder
            .iter()
            .map(|&(_, i)| available[i].clone())
            .collect::<Vec<_>>();
        available.dedup();
        return Err(TemperatureError::OutOfRange {
            label: label.to_string(),
            requested_k,
            lowest_k,
            highest_k,
            available,
        });
    }

    // Strictly between two rungs, since the exact case is already handled.
    let hi = ladder
        .iter()
        .position(|&(k, _)| k > requested_k)
        .expect("a request at or below the highest rung has an upper neighbour");
    let (lo_k, lo_idx) = ladder[hi - 1];
    let (hi_k, hi_idx) = ladder[hi];
    Ok(TemperatureSource::Blend {
        lo_idx,
        hi_idx,
        weight: blend_weight(lo_k, hi_k, requested_k),
    })
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

    fn ladder() -> Vec<String> {
        ["250", "294", "600", "900", "1200", "2500"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn an_intermediate_temperature_brackets_its_two_neighbours() {
        let got = resolve("450", &ladder()).expect("450 is inside the ladder");
        let TemperatureSource::Blend {
            lo_idx,
            hi_idx,
            weight,
        } = got
        else {
            panic!("450 sits between 294 and 600, so it must blend: {got:?}");
        };
        assert_eq!((lo_idx, hi_idx), (1, 2));
        // Against `blend_weight` itself rather than against a second copy of
        // the arithmetic: if anything else in the tree ever computes its own
        // weight, this stops agreeing by accident.
        assert_eq!(weight, blend_weight(294.0, 600.0, 450.0));
        assert!((weight - (450.0 - 294.0) / (600.0 - 294.0)).abs() < 1e-12);
    }

    #[test]
    fn a_listed_temperature_resolves_to_itself_in_either_spelling() {
        assert_eq!(
            resolve("600", &ladder()).unwrap(),
            TemperatureSource::Exact { idx: 2 }
        );
        assert_eq!(
            resolve("600K", &ladder()).unwrap(),
            TemperatureSource::Exact { idx: 2 }
        );
    }

    #[test]
    fn the_snap_tolerance_is_a_tolerance_and_not_a_rounding_rule() {
        // Inside SNAP_TOLERANCE_K: the same temperature, not a blend of width
        // 1e-9 that would duplicate the whole data set to reproduce it.
        assert_eq!(
            resolve("599.9999999", &ladder()).unwrap(),
            TemperatureSource::Exact { idx: 2 }
        );
        // A hundredth of a Kelvin away is a real request and blends. This is
        // the assertion that says there is no `nearest` behaviour here.
        assert!(matches!(
            resolve("599.99", &ladder()).unwrap(),
            TemperatureSource::Blend { .. }
        ));
    }

    #[test]
    fn out_of_range_names_every_available_temperature_and_never_clamps() {
        for probe in ["3000", "100"] {
            let err = resolve(probe, &ladder()).expect_err("outside the ladder");
            let TemperatureError::OutOfRange { .. } = err else {
                panic!("{probe} should be out of range, got {err:?}");
            };
            let message = err.to_string();
            assert!(message.contains(probe), "{message} does not name {probe}");
            assert!(message.contains("Available temperatures:"));
            for t in ladder() {
                assert!(message.contains(&t), "{message} does not list {t}");
            }
        }
    }

    #[test]
    fn resolve_does_not_depend_on_the_order_of_the_available_list() {
        // The order the published files list temperatures in, which is
        // lexicographic and puts 1200 before 250.
        let lexicographic: Vec<String> = ["1200", "250", "294", "2500", "600", "900"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let TemperatureSource::Blend {
            lo_idx,
            hi_idx,
            weight,
        } = resolve("450", &lexicographic).unwrap()
        else {
            panic!("450 must still blend");
        };
        // Indices into the slice as given, so they differ from the sorted
        // case; the temperatures they name must not.
        assert_eq!(lexicographic[lo_idx], "294");
        assert_eq!(lexicographic[hi_idx], "600");
        assert_eq!(weight, blend_weight(294.0, 600.0, 450.0));
    }

    #[test]
    fn a_single_temperature_library_can_only_be_hit_exactly() {
        let one = vec!["294".to_string()];
        assert_eq!(
            resolve("294", &one).unwrap(),
            TemperatureSource::Exact { idx: 0 }
        );
        // The degenerate case of the out-of-range rule, which is why it needs
        // no path of its own.
        assert!(matches!(
            resolve("300", &one),
            Err(TemperatureError::OutOfRange { .. })
        ));
    }

    #[test]
    fn nothing_resolves_against_no_temperatures_or_a_nonsense_label() {
        assert!(matches!(
            resolve("294", &[]),
            Err(TemperatureError::NoTemperatures { .. })
        ));
        assert!(matches!(
            resolve("hot", &ladder()),
            Err(TemperatureError::Unparseable { .. })
        ));
    }

    #[test]
    fn a_label_that_is_not_a_number_cannot_become_a_bracket_endpoint() {
        // A stray non-numeric entry is dropped rather than string-compared, so
        // the bracket is the same as it would be without it.
        let with_junk: Vec<String> = ["294", "hot", "600"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let TemperatureSource::Blend { lo_idx, hi_idx, .. } = resolve("450", &with_junk).unwrap()
        else {
            panic!("450 must blend");
        };
        assert_eq!(
            (with_junk[lo_idx].as_str(), with_junk[hi_idx].as_str()),
            ("294", "600")
        );
    }

    #[test]
    fn the_blend_weight_is_zero_and_one_at_its_own_endpoints() {
        assert_eq!(blend_weight(294.0, 600.0, 294.0), 0.0);
        assert_eq!(blend_weight(294.0, 600.0, 600.0), 1.0);
        // A degenerate bracket cannot divide by zero.
        assert_eq!(blend_weight(294.0, 294.0, 294.0), 0.0);
    }
}
