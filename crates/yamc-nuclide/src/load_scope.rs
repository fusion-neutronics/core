//! Which subset of a nuclide's Arrow data a caller needs (issue #389).
//!
//! `Material.transmute()` runs no transport, so it reads only the union energy
//! grid and the per-MT cross sections of the reactions its chain names. On the
//! TENDL-2025 conversion that is about 21% of the per-nuclide data by size:
//! `fast_xs.arrow` alone is 61% of the library and is a transport lookup
//! accelerator the activation path never touches.
//!
//! A `LoadScope` says what to fetch and parse, and is recorded on the resulting
//! [`Nuclide`](crate::nuclide::Nuclide) so the global cache can tell whether an
//! entry it already holds is wide enough for the next request. That mirrors the
//! `available_temperatures` / `loaded_temperatures` pair the type has always
//! carried; the temperature filter is folded in here so "what subset did we
//! parse" is one concept rather than three parallel arguments.

use std::collections::HashSet;

/// Which sections of a `{Nuclide}.arrow/` directory to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SectionScope {
    /// Every section. What transport needs, and the historic behaviour.
    #[default]
    Full,
    /// `nuclide.arrow` (the union energy grid) and `reactions.arrow` only.
    ///
    /// Skips the secondary distributions, the product tables, the `fast_xs`
    /// accelerator, URR tables, fission nu and fission-photon release. The
    /// reactions that survive carry no products, which is why a nuclide loaded
    /// this way must never reach transport.
    XsOnly,
}

/// The subset of a nuclide's data a caller needs.
///
/// Build with [`LoadScope::full`] for transport, or [`LoadScope::activation`]
/// for a transport-free reaction-rate collapse.
#[derive(Debug, Clone, Default)]
pub struct LoadScope {
    /// Which Arrow sections to read.
    pub sections: SectionScope,
    /// MT numbers to materialize from `reactions.arrow`. `None` means every MT.
    ///
    /// Filtering here saves parse time and retained memory, not bytes read: the
    /// Arrow batch is mapped whole either way, what shrinks is the set of
    /// `Vec<f64>` that outlives it.
    pub mts: Option<HashSet<i32>>,
    /// Temperatures to materialize. `None` means every temperature present.
    pub temperatures: Option<HashSet<String>>,
    /// Whether to read `covariance.arrow`, the MF=33 cross-section covariance.
    ///
    /// Its own axis rather than part of [`SectionScope`], because it is
    /// orthogonal to the transport/activation split: an uncertainty calculation
    /// wants covariance with an `XsOnly` load, and transport wants `Full`
    /// without it. Off by default, so nothing on the ordinary path reads or
    /// allocates it.
    pub covariance: bool,
}

impl LoadScope {
    /// Everything, at every temperature. Transport's scope.
    pub fn full() -> Self {
        Self::default()
    }

    /// Cross sections for `mts` only, with no transport-side sections.
    pub fn activation(mts: HashSet<i32>) -> Self {
        Self {
            sections: SectionScope::XsOnly,
            mts: Some(mts),
            temperatures: None,
            covariance: false,
        }
    }

    /// Also read `covariance.arrow`.
    pub fn with_covariance(mut self, covariance: bool) -> Self {
        self.covariance = covariance;
        self
    }

    /// Restrict to a set of temperatures.
    pub fn with_temperatures(mut self, temperatures: Option<HashSet<String>>) -> Self {
        self.temperatures = temperatures;
        self
    }

    /// Whether an MT should be materialized under this scope.
    pub fn wants_mt(&self, mt: i32) -> bool {
        self.mts.as_ref().is_none_or(|set| set.contains(&mt))
    }

    /// Whether this load materializes every row of every column it reads.
    ///
    /// The Arrow loader uses this to decide whether to share a column's values
    /// buffer or copy out of it. A shared view keeps its whole parent allocation
    /// alive, so it only pays when the load keeps essentially all of the column;
    /// under an MT or temperature filter it would pin exactly the bytes the
    /// filter exists to drop.
    pub fn is_unfiltered(&self) -> bool {
        self.mts.is_none() && self.temperatures.is_none()
    }

    /// Whether the transport-only sections should be read.
    pub fn wants_transport_sections(&self) -> bool {
        self.sections == SectionScope::Full
    }

    /// Whether data loaded under `self` satisfies a request for `other`.
    ///
    /// Every axis widens the same way: `None` means "all of it" and so covers
    /// any concrete set, while a concrete set covers only its own subsets.
    pub fn covers(&self, other: &LoadScope) -> bool {
        let sections_ok = match (self.sections, other.sections) {
            (SectionScope::Full, _) => true,
            (SectionScope::XsOnly, SectionScope::XsOnly) => true,
            (SectionScope::XsOnly, SectionScope::Full) => false,
        };
        // Covariance widens the same way every other axis does: having it
        // covers a request that does not want it, and not having it covers only
        // a request that does not either.
        sections_ok
            && (self.covariance || !other.covariance)
            && covers_set(self.mts.as_ref(), other.mts.as_ref())
            && covers_set(self.temperatures.as_ref(), other.temperatures.as_ref())
    }

    /// The narrowest scope covering both `self` and `other`.
    ///
    /// Used when a cached entry does not cover an incoming request: reloading
    /// at the union rather than at the request keeps two callers wanting, say,
    /// MT {102, 16} and MT {102, 103} from evicting each other on every call.
    pub fn union(&self, other: &LoadScope) -> LoadScope {
        let sections =
            if self.sections == SectionScope::Full || other.sections == SectionScope::Full {
                SectionScope::Full
            } else {
                SectionScope::XsOnly
            };
        LoadScope {
            sections,
            mts: union_set(self.mts.as_ref(), other.mts.as_ref()),
            temperatures: union_set(self.temperatures.as_ref(), other.temperatures.as_ref()),
            covariance: self.covariance || other.covariance,
        }
    }
}

/// `None` is the universal set, so it covers anything and nothing but another
/// `None` covers it.
fn covers_set<T: std::hash::Hash + Eq>(
    have: Option<&HashSet<T>>,
    want: Option<&HashSet<T>>,
) -> bool {
    match (have, want) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(have), Some(want)) => want.is_subset(have),
    }
}

/// Union under the same convention: anything unioned with the universal set is
/// the universal set.
fn union_set<T: std::hash::Hash + Eq + Clone>(
    a: Option<&HashSet<T>>,
    b: Option<&HashSet<T>>,
) -> Option<HashSet<T>> {
    match (a, b) {
        (None, _) | (_, None) => None,
        (Some(a), Some(b)) => Some(a.union(b).cloned().collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mts(values: &[i32]) -> HashSet<i32> {
        values.iter().copied().collect()
    }

    fn temps(values: &[&str]) -> HashSet<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn full_covers_everything() {
        let full = LoadScope::full();
        assert!(full.covers(&LoadScope::full()));
        assert!(full.covers(&LoadScope::activation(mts(&[102, 16]))));
    }

    #[test]
    fn activation_does_not_cover_transport() {
        let activation = LoadScope::activation(mts(&[102]));
        assert!(!activation.covers(&LoadScope::full()));
    }

    #[test]
    fn a_wider_mt_set_covers_a_narrower_one() {
        let wide = LoadScope::activation(mts(&[102, 16, 103]));
        assert!(wide.covers(&LoadScope::activation(mts(&[102, 16]))));
        assert!(!LoadScope::activation(mts(&[102])).covers(&LoadScope::activation(mts(&[102, 16]))));
    }

    #[test]
    fn temperature_filtering_narrows_coverage() {
        let one = LoadScope::full().with_temperatures(Some(temps(&["294"])));
        let two = LoadScope::full().with_temperatures(Some(temps(&["294", "600"])));
        assert!(two.covers(&one));
        assert!(!one.covers(&two));
        // An unfiltered load holds every temperature, so it covers a filtered one.
        assert!(LoadScope::full().covers(&two));
        assert!(!two.covers(&LoadScope::full()));
    }

    #[test]
    fn a_synthesised_load_covers_both_the_request_and_the_bracket_it_was_built_from() {
        // What the loader records after serving a 450 K request from a file
        // carrying 294 K and 600 K: the two rungs it read plus the label it
        // synthesised. The union is the only choice that works in both
        // directions, and this pins it.
        let served = LoadScope::full().with_temperatures(Some(temps(&["294", "450", "600"])));

        // Recording only the two rungs would miss here, and the cache would
        // rebuild the blend on every call.
        assert!(served.covers(&LoadScope::full().with_temperatures(Some(temps(&["450"])))));
        // Recording only the request would miss here, and a later query at a
        // temperature already in memory would reload the file.
        assert!(served.covers(&LoadScope::full().with_temperatures(Some(temps(&["294"])))));
        // It must not claim coverage it does not have.
        assert!(!served.covers(&LoadScope::full().with_temperatures(Some(temps(&["900"])))));
    }

    #[test]
    fn union_keeps_both_callers_satisfied() {
        let a = LoadScope::activation(mts(&[102, 16]));
        let b = LoadScope::activation(mts(&[102, 103]));
        let merged = a.union(&b);
        assert!(merged.covers(&a));
        assert!(merged.covers(&b));
        assert_eq!(merged.sections, SectionScope::XsOnly);
    }

    #[test]
    fn union_with_transport_widens_the_sections() {
        let merged = LoadScope::activation(mts(&[102])).union(&LoadScope::full());
        assert_eq!(merged.sections, SectionScope::Full);
        assert!(merged.mts.is_none());
        assert!(merged.covers(&LoadScope::full()));
    }

    #[test]
    fn wants_mt_defaults_to_everything() {
        assert!(LoadScope::full().wants_mt(51));
        let activation = LoadScope::activation(mts(&[102]));
        assert!(activation.wants_mt(102));
        assert!(!activation.wants_mt(51));
    }
}
