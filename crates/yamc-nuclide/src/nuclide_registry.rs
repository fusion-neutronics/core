use std::collections::HashMap;
use std::num::NonZeroU16;

/// Compact integer identifier for a nuclide name.
///
/// Stored on `Particle.parent_nuclide` instead of a `String` so the hot path
/// carries no heap pointer. `NonZeroU16` gives `Option<NuclideId>` a 2-byte
/// niche-packed layout. IDs are assigned by `NuclideRegistry` starting at 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NuclideId(NonZeroU16);

impl NuclideId {
    #[inline]
    pub fn get(self) -> u16 {
        self.0.get()
    }
}

/// Interner mapping nuclide names <-> `NuclideId`.
///
/// Built once during simulation setup (D1S precompute and tally filter
/// resolution). Not touched during transport: the hot path only reads the
/// already-assigned `NuclideId` from `Particle` and compares integers in
/// filters. Names remain available for diagnostics and I/O via `name()`.
#[derive(Debug, Default, Clone)]
pub struct NuclideRegistry {
    by_name: HashMap<String, NuclideId>,
    by_id: Vec<String>,
}

impl NuclideRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up an existing id without inserting.
    pub fn lookup(&self, name: &str) -> Option<NuclideId> {
        self.by_name.get(name).copied()
    }

    /// Intern a name, assigning a fresh id if unseen.
    pub fn intern(&mut self, name: &str) -> NuclideId {
        if let Some(&id) = self.by_name.get(name) {
            return id;
        }
        let next = self.by_id.len() + 1;
        let raw = u16::try_from(next).expect("NuclideRegistry exceeded u16 capacity (65535 names)");
        let id = NuclideId(NonZeroU16::new(raw).expect("next >= 1"));
        self.by_id.push(name.to_string());
        self.by_name.insert(name.to_string(), id);
        id
    }

    /// Resolve an id back to the interned name. Panics for ids from a different registry.
    pub fn name(&self, id: NuclideId) -> &str {
        let idx = id.get() as usize - 1;
        &self.by_id[idx]
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_same_name_returns_same_id() {
        let mut r = NuclideRegistry::new();
        let a = r.intern("Co60");
        let b = r.intern("Co60");
        assert_eq!(a, b);
    }

    #[test]
    fn distinct_names_get_distinct_ids() {
        let mut r = NuclideRegistry::new();
        let a = r.intern("Co60");
        let b = r.intern("Mn56");
        assert_ne!(a, b);
    }

    #[test]
    fn name_roundtrip() {
        let mut r = NuclideRegistry::new();
        let id = r.intern("Fe59");
        assert_eq!(r.name(id), "Fe59");
    }

    #[test]
    fn lookup_returns_none_for_unseen() {
        let r = NuclideRegistry::new();
        assert!(r.lookup("Co60").is_none());
    }

    #[test]
    fn option_nuclide_id_is_two_bytes() {
        assert_eq!(std::mem::size_of::<Option<NuclideId>>(), 2);
    }
}
