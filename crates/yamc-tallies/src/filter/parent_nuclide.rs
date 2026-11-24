use std::sync::OnceLock;
use yamc_nuclide::nuclide_registry::{NuclideId, NuclideRegistry};

/// Filter that bins tally scores by the parent nuclide of D1S decay photons.
///
/// Each nuclide name in the filter is a separate bin. Particles whose
/// `parent_nuclide` matches one of the filter's nuclide ids are scored
/// into the corresponding bin. Particles with no `parent_nuclide` or
/// whose parent is not in the filter list are not scored.
///
/// The user-facing `nuclides: Vec<String>` is the source of truth (Python API).
/// During setup, `resolve(&mut registry)` populates `ids` so that scoring on the
/// hot path uses integer comparisons rather than string compares.
/// Serializes only the user-supplied `nuclides` names; resolved
/// `NuclideId`s are rebuilt via `resolve(&mut registry)` after load.
/// `PartialEq` is intentionally compared only on the user-supplied
/// names so a freshly-loaded filter compares equal to one with a
/// populated ID cache.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ParentNuclideFilter {
    /// List of nuclide names (e.g., ["Co60", "Mn56", "Fe59"])
    pub nuclides: Vec<String>,
    /// Resolved ids parallel to `nuclides`; populated once at simulation setup.
    #[serde(skip)]
    ids: OnceLock<Vec<NuclideId>>,
}

impl ParentNuclideFilter {
    /// Create a new ParentNuclideFilter with the given nuclide names.
    pub fn new(nuclides: Vec<String>) -> Self {
        Self {
            nuclides,
            ids: OnceLock::new(),
        }
    }

    /// Get the number of bins (one per nuclide).
    pub fn num_bins(&self) -> usize {
        self.nuclides.len()
    }

    /// Resolve each filter name into a `NuclideId`, interning any unseen names.
    ///
    /// Call once during simulation setup, after D1S target names have been
    /// interned. Names that are not produced by any D1S channel still get ids
    /// here -- they simply won't match any particle, which is the correct
    /// behavior (the user typed a nuclide that isn't actually a D1S parent).
    pub fn resolve(&self, registry: &mut NuclideRegistry) {
        if self.ids.get().is_some() {
            return;
        }
        let ids: Vec<NuclideId> = self
            .nuclides
            .iter()
            .map(|name| registry.intern(name))
            .collect();
        let _ = self.ids.set(ids);
    }

    /// Get the bin index for a given parent nuclide id.
    /// Returns `None` if the id is not in this filter, or if the filter has
    /// not been resolved yet (in which case no scoring happens -- correct
    /// fallback if something invokes the hot path before setup completes).
    pub fn get_bin(&self, id: NuclideId) -> Option<usize> {
        let ids = self.ids.get()?;
        ids.iter().position(|&i| i == id)
    }

    /// The resolved parent-nuclide ids in filter (= bin) order, or an empty
    /// slice if the filter has not been resolved yet. Used by the GPU dispatch
    /// to build the per-tally parent-bin map the photon kernel scans.
    pub fn resolved_ids(&self) -> &[NuclideId] {
        self.ids.get().map(Vec::as_slice).unwrap_or(&[])
    }
}

impl Clone for ParentNuclideFilter {
    fn clone(&self) -> Self {
        Self {
            nuclides: self.nuclides.clone(),
            ids: OnceLock::new(),
        }
    }
}

impl PartialEq for ParentNuclideFilter {
    fn eq(&self, other: &Self) -> bool {
        self.nuclides == other.nuclides
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_bin_found() {
        let filter = ParentNuclideFilter::new(vec![
            "Mn56".to_string(),
            "Co60".to_string(),
            "Fe59".to_string(),
        ]);
        let mut registry = NuclideRegistry::new();
        filter.resolve(&mut registry);
        let mn56 = registry.lookup("Mn56").unwrap();
        let co60 = registry.lookup("Co60").unwrap();
        let fe59 = registry.lookup("Fe59").unwrap();
        assert_eq!(filter.get_bin(mn56), Some(0));
        assert_eq!(filter.get_bin(co60), Some(1));
        assert_eq!(filter.get_bin(fe59), Some(2));
    }

    #[test]
    fn test_get_bin_not_found() {
        let filter = ParentNuclideFilter::new(vec![
            "Mn56".to_string(),
            "Co60".to_string(),
            "Fe59".to_string(),
        ]);
        let mut registry = NuclideRegistry::new();
        filter.resolve(&mut registry);
        let na24 = registry.intern("Na24");
        let u235 = registry.intern("U235");
        assert_eq!(filter.get_bin(na24), None);
        assert_eq!(filter.get_bin(u235), None);
    }

    #[test]
    fn test_num_bins() {
        let filter = ParentNuclideFilter::new(vec!["Co60".to_string(), "Mn56".to_string()]);
        assert_eq!(filter.num_bins(), 2);

        let empty = ParentNuclideFilter::new(vec![]);
        assert_eq!(empty.num_bins(), 0);
    }

    #[test]
    fn test_single_nuclide() {
        let filter = ParentNuclideFilter::new(vec!["Co60".to_string()]);
        let mut registry = NuclideRegistry::new();
        filter.resolve(&mut registry);
        let co60 = registry.lookup("Co60").unwrap();
        let mn56 = registry.intern("Mn56");
        assert_eq!(filter.num_bins(), 1);
        assert_eq!(filter.get_bin(co60), Some(0));
        assert_eq!(filter.get_bin(mn56), None);
    }

    #[test]
    fn test_resolve_is_idempotent() {
        let filter = ParentNuclideFilter::new(vec!["Co60".to_string()]);
        let mut registry = NuclideRegistry::new();
        filter.resolve(&mut registry);
        let first = registry.lookup("Co60").unwrap();
        filter.resolve(&mut registry);
        let second = registry.lookup("Co60").unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn test_clone_drops_resolved_cache() {
        // Clone is used when a Tally is cloned; the resolved-id cache is
        // bound to the registry of the original simulation, so the clone
        // starts empty and must be re-resolved.
        let filter = ParentNuclideFilter::new(vec!["Co60".to_string()]);
        let mut registry = NuclideRegistry::new();
        filter.resolve(&mut registry);
        let cloned = filter.clone();
        let co60 = registry.lookup("Co60").unwrap();
        assert_eq!(cloned.get_bin(co60), None); // not resolved in clone
        cloned.resolve(&mut registry);
        assert_eq!(cloned.get_bin(co60), Some(0));
    }
}
