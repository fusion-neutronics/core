//! Hash containers with a fixed seed, for reproducible iteration order.
//!
//! `std::collections::HashSet`/`HashMap` default to `RandomState`, which seeds
//! itself per process. Iterating one therefore yields a different order on
//! every run, and anywhere that order reaches the output the mesher stops
//! being reproducible. That is not hypothetical here: the cavity
//! re-triangulation in `boundary_recovery` fed its `HashSet` order into the
//! vertex, face and tet lists it re-meshes, so Paraboloid produced a different
//! tetrahedralisation on every run -- sometimes filling its boundary exactly,
//! sometimes leaking 1.6e-7.
//!
//! Two places had already been patched by hand (the segment queue in
//! `recover_segments_with_steiner` sorts for exactly this reason), which is the
//! argument for fixing the class rather than the instances: the next hash
//! container someone adds is a silent regression otherwise.
//!
//! `BuildHasherDefault<DefaultHasher>` is SipHash keyed with zeros -- the same
//! hash quality, minus the per-process seed. HashDoS resistance is what is
//! given up, and it buys nothing here: the keys are vertex and tet indices the
//! mesher generates itself, never adversarial input.

use std::collections::hash_map::DefaultHasher;
use std::hash::BuildHasherDefault;

/// Fixed-seed `BuildHasher`. It is `Default`, so the collections' `Default`,
/// `FromIterator` and `with_capacity_and_hasher` all work unchanged.
pub(crate) type DetState = BuildHasherDefault<DefaultHasher>;

/// Drop-in `HashSet` with reproducible iteration order. Construct with
/// `::default()` -- `::new()` exists only for `RandomState`.
pub(crate) type HashSet<T> = std::collections::HashSet<T, DetState>;

/// Drop-in `HashMap` with reproducible iteration order. Construct with
/// `::default()` -- `::new()` exists only for `RandomState`.
pub(crate) type HashMap<K, V> = std::collections::HashMap<K, V, DetState>;
