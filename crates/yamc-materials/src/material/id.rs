//! Material ids, and the process-wide allocator that hands out the automatic
//! ones.
//!
//! The user-facing constructors (the Python `Material`, `from_atom_densities`,
//! `mix_materials` and the material collections) give each material an id
//! when it is built, so a tally can filter on a material before
//! any geometry exists. The ids come from one allocator per process, which also
//! records every id set explicitly, so an id it hands out never repeats one in
//! use already: an earlier automatic id, or an explicit one given earlier to a
//! constructor, to [`Material::set_material_id`] or by deserialization.
//!
//! An id is a label, and two materials can still share one: an explicit id
//! set later can repeat an automatic one, and a copy keeps the id of the
//! material it copies. A Python geometry refuses two different `Material`
//! objects with one id wherever they meet, in its cells or their mesh fills,
//! and a material filter refuses one id twice. No computed result goes by an
//! id's value, since an automatic id depends on what else the process built:
//! the model fingerprint, `combine_results` and the statistical uncertainty
//! draw go by the slot a material holds in the geometry.
//!
//! Internal constructors do not allocate. [`Material::new`], clones and the
//! materials a transmutation step produces keep the id they were given (an
//! evolved material keeps its template's, which its results are keyed by), so
//! code building materials on several threads cannot make ids depend on the
//! thread schedule.

use super::Material;
use std::collections::BTreeSet;
use std::sync::{Mutex, MutexGuard};

/// Hands out the lowest material id nothing in the process has used.
#[derive(Debug)]
struct IdAllocator {
    /// The next candidate. Every id below it has been handed out or passed
    /// over and is never offered again. Wider than `u32` so that running past
    /// `u32::MAX` is an error rather than a wrap back to 0.
    next: u64,
    /// Explicit ids at or above `next`, which `allocate` must skip. One below
    /// `next` needs no record, since the allocator never goes back.
    reserved: BTreeSet<u32>,
}

impl IdAllocator {
    const fn new() -> Self {
        Self {
            next: 1,
            reserved: BTreeSet::new(),
        }
    }

    fn reserve(&mut self, id: u32) {
        if u64::from(id) >= self.next {
            self.reserved.insert(id);
        }
    }

    fn allocate(&mut self) -> Result<u32, String> {
        loop {
            let Ok(id) = u32::try_from(self.next) else {
                return Err(format!(
                    "no material id is left: every id from 1 to {} is already in use in this \
                     process",
                    u32::MAX
                ));
            };
            self.next += 1;
            if !self.reserved.remove(&id) {
                return Ok(id);
            }
        }
    }
}

static IDS: Mutex<IdAllocator> = Mutex::new(IdAllocator::new());

fn ids() -> MutexGuard<'static, IdAllocator> {
    // Each update is a single step, so a panic elsewhere while the lock was
    // held cannot have left the allocator half-updated.
    IDS.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Material {
    /// Set the material's id, reserving it so no automatic id repeats it.
    pub fn set_material_id(&mut self, material_id: u32) {
        ids().reserve(material_id);
        self.material_id = Some(material_id);
    }

    /// Give the material `material_id`, or a fresh id when it is `None`, and
    /// return the id. The user-facing constructors call this, and so does a
    /// geometry for a material that still has none.
    ///
    /// # Errors
    /// When a fresh id is wanted and every id is in use.
    pub fn assign_material_id(&mut self, material_id: Option<u32>) -> Result<u32, String> {
        let id = match material_id {
            Some(id) => {
                ids().reserve(id);
                id
            }
            None => ids().allocate()?,
        };
        self.material_id = Some(id);
        Ok(id)
    }

    /// The material's id, if it has one.
    #[inline]
    pub fn get_material_id(&self) -> Option<u32> {
        self.material_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::MaterialSerde;
    use std::collections::{HashMap, HashSet};

    fn material() -> Material {
        Material::new(HashMap::new(), "atom", "sum", None).unwrap()
    }

    #[test]
    fn allocator_counts_up_from_one_skipping_reserved_ids() {
        let mut ids = IdAllocator::new();
        ids.reserve(2);
        ids.reserve(3);
        assert_eq!(ids.allocate(), Ok(1));
        assert_eq!(ids.allocate(), Ok(4));
        // Below the counter: already passed over, so nothing to record.
        ids.reserve(1);
        ids.reserve(3);
        assert!(ids.reserved.is_empty());
        assert_eq!(ids.allocate(), Ok(5));
    }

    #[test]
    fn allocator_reports_exhaustion_instead_of_wrapping() {
        let mut ids = IdAllocator::new();
        ids.next = u64::from(u32::MAX);
        assert_eq!(ids.allocate(), Ok(u32::MAX));
        for _ in 0..2 {
            let err = ids.allocate().unwrap_err();
            assert!(err.contains("no material id is left"), "{err}");
        }

        let mut ids = IdAllocator::new();
        ids.next = u64::from(u32::MAX);
        ids.reserve(u32::MAX);
        assert!(ids.allocate().is_err());
    }

    #[test]
    fn an_explicit_maximum_id_does_not_use_up_the_allocator() {
        let mut ids = IdAllocator::new();
        ids.reserve(u32::MAX);
        ids.reserve(u32::MAX - 1);
        assert_eq!(ids.allocate(), Ok(1));
        assert_eq!(ids.allocate(), Ok(2));
    }

    #[test]
    fn automatic_ids_are_unique() {
        let ids: Vec<u32> = (0..1000)
            .map(|_| material().assign_material_id(None).unwrap())
            .collect();
        assert_eq!(ids.iter().collect::<HashSet<_>>().len(), ids.len());
        assert!(
            !ids.contains(&0),
            "0 means 'no id' to the transmutation results"
        );
    }

    #[test]
    fn automatic_ids_skip_ids_set_explicitly_earlier() {
        // Absolute values depend on what else ran in this process, so the ids
        // to reserve are taken relative to a fresh one.
        let probe = material().assign_material_id(None).unwrap();
        let mut by_setter = material();
        by_setter.set_material_id(probe + 1);
        let by_constructor = material().assign_material_id(Some(probe + 2)).unwrap();
        let spec = MaterialSerde::from(material());
        let deserialized = Material::try_from(MaterialSerde {
            material_id: Some(probe + 3),
            ..spec
        })
        .unwrap();
        let explicit = [
            by_setter.get_material_id().unwrap(),
            by_constructor,
            deserialized.get_material_id().unwrap(),
        ];
        assert_eq!(explicit, [probe + 1, probe + 2, probe + 3]);

        for _ in 0..10 {
            let id = material().assign_material_id(None).unwrap();
            assert!(
                id > probe && !explicit.contains(&id),
                "{id} repeats {explicit:?}"
            );
        }
    }

    #[test]
    fn clones_and_evolved_materials_keep_the_id() {
        let mut template = material();
        let id = template.assign_material_id(None).unwrap();
        assert_eq!(template.clone().get_material_id(), Some(id));
        let evolved =
            Material::from_nuclide_densities(HashMap::from([("Li6".into(), 1.0)]), &template);
        assert_eq!(evolved.get_material_id(), Some(id));
        assert_eq!(material().get_material_id(), None);
    }
}
