//! Collection of finalized `TallyResult`s returned from `Model::simulate()`.
//!
//! This module provides the Rust-side `SimulationResults` type; the Python
//! bindings live in `crates/yamc-python/src/simulation_results.rs`.
//!
//! ## Lookup flavors
//!
//! Results can be retrieved four ways:
//!
//! - by position (`get(i)`) -- always works, preserves input order
//! - by `Tally` object identity (`get_by_tally(&arc)`) -- always works if the
//!   same `Arc<Tally>` instance was used to build the results
//! - by name (`get_by_name("flux")`) -- only works if the user set `Tally.name`
//! - by id (`get_by_id(2)`) -- only works if the user set `Tally.tally_id`
//!
//! Duplicate names or ids are rejected at construction with a clear error.
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::result::TallyResult;
use crate::tally::Tally;

/// Provenance of one simulation run contributing to a
/// [`SimulationResults`]. Recorded at `simulate_transport` time and
/// carried (and concatenated) through [`crate::combine::combine_results`]
/// so combined results stay attributable and further combinable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunProvenance {
    /// Base RNG seed of the run. Two runs with the same base seed share
    /// per-particle RNG streams and must never be combined.
    pub seed: u64,
    /// Exact finished source-history count of the run.
    pub n_histories: u64,
    /// Wall-clock seconds of the run.
    pub elapsed_secs: f64,
    /// Stable hash of the model's physics identity: geometry, materials,
    /// sources and physics settings plus the nuclear-data-library map;
    /// excludes seed, total_particles, verbose, diagnostics and the
    /// tallies subtree (tallies are observation-only).
    pub fingerprint: String,
    /// Per-isotope / per-element nuclear-data provenance, e.g.
    /// `"n:Co58" -> "endf-b8.1"`, `"p:Fe" -> "endf-b8.1"`. Also folded
    /// into `fingerprint`; kept readable here so mismatch errors can name
    /// the nuclide and libraries.
    pub data_libraries: BTreeMap<String, String>,
    /// Compute path that produced the run: `"cpu"` or `"gpu"`. GPU runs
    /// carry no Welford merge state and are refused by `combine_results`.
    pub compute: String,
    /// yamc version that produced the run.
    pub yamc_version: String,
    /// MPI world size of the run (1 for non-MPI).
    pub mpi_size: i32,
    /// MPI rank that produced this result (0 for non-MPI). Only rank-0
    /// results are complete; others are refused by `combine_results`.
    pub mpi_rank: i32,
}

/// Ordered collection of `TallyResult`s plus run-level metadata.
///
/// Returned from `Model::simulate()`. Immutable after construction.
#[derive(Debug)]
pub struct SimulationResults {
    /// Results in the same order the input tallies were supplied.
    results: Vec<Arc<TallyResult>>,

    /// Lookup: `Tally.name` -> index into `results`. Populated only for tallies
    /// that had a name set.
    by_name: HashMap<String, usize>,

    /// Lookup: `Tally.tally_id` -> index into `results`. Populated only for
    /// tallies that had an id set.
    by_id: HashMap<u32, usize>,

    /// Lookup: raw `Arc<Tally>` pointer -> index into `results`. The pointer
    /// value is stable for the lifetime of the `Arc`, so this is a valid key
    /// for identity-based lookup (`results[tally_obj]`). Stored as `usize` so
    /// the struct remains `Send + Sync`.
    by_ptr: HashMap<usize, usize>,

    /// Number of batches accumulated. Mirrored from the first tally's
    /// `TallyResult::n_batches`.
    pub n_batches: u32,

    /// Source particles per batch. Mirrored from the first tally.
    pub particles_per_chunk: u32,

    /// Wall-clock time elapsed during the simulate call (seconds). For a
    /// combined result this is the sum over all contributing runs.
    pub elapsed_secs: f64,

    /// Provenance of the run(s) whose statistics these results carry.
    /// One entry for a plain `simulate_transport` result; concatenated
    /// by `combine_results`. Empty for results built without provenance
    /// (legacy constructors, unit tests) -- such results are refused by
    /// `combine_results`.
    pub runs: Vec<RunProvenance>,
}

impl SimulationResults {
    /// Build a `SimulationResults` by finalizing each tally in order.
    ///
    /// Returns `Err` if two tallies share the same name or the same id.
    /// No auto-generation of names or ids -- if the user did not set one,
    /// that lookup flavor is unavailable for that tally.
    pub fn from_tallies(tallies: &[Arc<Tally>], elapsed_secs: f64) -> Result<Self, String> {
        Self::build(tallies, elapsed_secs, Vec::new())
    }

    /// Build a `SimulationResults` with run provenance attached. This is
    /// the constructor the real `simulate_transport` path uses; results
    /// built this way are combinable via `combine_results`.
    pub fn from_tallies_with_run(
        tallies: &[Arc<Tally>],
        elapsed_secs: f64,
        run: RunProvenance,
    ) -> Result<Self, String> {
        Self::build(tallies, elapsed_secs, vec![run])
    }

    fn build(
        tallies: &[Arc<Tally>],
        elapsed_secs: f64,
        runs: Vec<RunProvenance>,
    ) -> Result<Self, String> {
        // Validate duplicates before touching any accumulator.
        let mut seen_names: HashMap<&str, usize> = HashMap::new();
        let mut seen_ids: HashMap<u32, usize> = HashMap::new();
        for (i, t) in tallies.iter().enumerate() {
            if let Some(name) = t.name.as_deref() {
                if let Some(&prev) = seen_names.get(name) {
                    return Err(format!(
                        "duplicate tally name {name:?} at indices {prev} and {i}"
                    ));
                }
                seen_names.insert(name, i);
            }
            if let Some(id) = t.tally_id {
                if let Some(&prev) = seen_ids.get(&id) {
                    return Err(format!("duplicate tally id {id} at indices {prev} and {i}"));
                }
                seen_ids.insert(id, i);
            }
        }

        // Finalize each tally into an Arc<TallyResult>, stamping the run's
        // wall-clock time and provenance onto it. The elapsed-derived stats
        // (figure of merit etc.) are populated centrally by `from_parts`
        // below, so every constructor path fills them the same way. Every
        // tally of a single run is attributed to run 0 when provenance is
        // present.
        let run_indices: Vec<usize> = if runs.is_empty() { Vec::new() } else { vec![0] };
        let results: Vec<Arc<TallyResult>> = tallies
            .iter()
            .map(|t| {
                let mut r = t.finalize();
                r.run_indices = run_indices.clone();
                r.elapsed_secs = elapsed_secs;
                Arc::new(r)
            })
            .collect();

        // Build owned-key lookup tables (can't share refs with `seen_*` because
        // those borrow from `tallies`).
        let mut by_name = HashMap::new();
        let mut by_id = HashMap::new();
        let mut by_ptr = HashMap::new();
        for (i, t) in tallies.iter().enumerate() {
            if let Some(name) = &t.name {
                by_name.insert(name.clone(), i);
            }
            if let Some(id) = t.tally_id {
                by_id.insert(id, i);
            }
            by_ptr.insert(Arc::as_ptr(t) as usize, i);
        }

        // Run-level metadata: mirror from the first tally if present, else
        // fall back to zero (empty-tallies case).
        let (n_batches, particles_per_chunk) = results
            .first()
            .map(|r| (r.n_batches, r.particles_per_chunk))
            .unwrap_or((0, 0));

        Ok(Self::from_parts(
            results,
            by_name,
            by_id,
            by_ptr,
            n_batches,
            particles_per_chunk,
            elapsed_secs,
            runs,
        ))
    }

    /// Low-level constructor used by the Arrow reader and
    /// `combine_results`. Not part of the public API surface users
    /// construct directly -- prefer `from_tallies_with_run`.
    ///
    /// This is the single place elapsed-derived stats (figure of merit, and
    /// any other stat that needs the run's wall-clock time) are populated:
    /// every incoming result is run through [`TallyResult::with_fom`] using
    /// its own `elapsed_secs`. `with_fom` is idempotent, so callers that
    /// already applied it (or only set `elapsed_secs`) both end up correct --
    /// no constructor path can ship a result with empty figure-of-merit.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        results: Vec<Arc<TallyResult>>,
        by_name: HashMap<String, usize>,
        by_id: HashMap<u32, usize>,
        by_ptr: HashMap<usize, usize>,
        n_batches: u32,
        particles_per_chunk: u32,
        elapsed_secs: f64,
        runs: Vec<RunProvenance>,
    ) -> Self {
        let results: Vec<Arc<TallyResult>> = results
            .into_iter()
            .map(|arc| {
                let elapsed = arc.elapsed_secs;
                let r = Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone());
                Arc::new(r.with_fom(elapsed))
            })
            .collect();
        Self {
            results,
            by_name,
            by_id,
            by_ptr,
            n_batches,
            particles_per_chunk,
            elapsed_secs,
            runs,
        }
    }

    /// Number of tallies finalized.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// True if no tallies were supplied.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Result at position `i`, or `None` if out of range.
    pub fn get(&self, i: usize) -> Option<&Arc<TallyResult>> {
        self.results.get(i)
    }

    /// Result for the given tally object (pointer-identity lookup).
    pub fn get_by_tally(&self, tally: &Arc<Tally>) -> Option<&Arc<TallyResult>> {
        let ptr = Arc::as_ptr(tally) as usize;
        self.by_ptr.get(&ptr).and_then(|&i| self.results.get(i))
    }

    /// Result for a tally with the given name, if any.
    pub fn get_by_name(&self, name: &str) -> Option<&Arc<TallyResult>> {
        self.by_name.get(name).and_then(|&i| self.results.get(i))
    }

    /// Result for a tally with the given `tally_id`, if any.
    pub fn get_by_id(&self, id: u32) -> Option<&Arc<TallyResult>> {
        self.by_id.get(&id).and_then(|&i| self.results.get(i))
    }

    /// All names available for name-based lookup (sorted for deterministic
    /// error messages).
    pub fn available_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.by_name.keys().cloned().collect();
        names.sort();
        names
    }

    /// All ids available for id-based lookup (sorted).
    pub fn available_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.by_id.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Iterate over `(Arc<TallyResult>,)` in input order.
    pub fn iter(&self) -> std::slice::Iter<'_, Arc<TallyResult>> {
        self.results.iter()
    }

    /// Write the results to an Arrow IPC file at `path`. See
    /// [`crate::arrow_io`] for the on-disk format.
    #[cfg(feature = "arrow")]
    pub fn to_arrow(&self, path: &std::path::Path) -> Result<(), String> {
        crate::arrow_io::write_simulation_results_arrow(self, path)
    }

    /// Read a previously-saved `SimulationResults` from an Arrow IPC file.
    ///
    /// The reconstructed tallies carry only name and id (scores/filters are
    /// not yet roundtripped -- see `arrow_io` module docs).
    #[cfg(feature = "arrow")]
    pub fn from_arrow(path: &std::path::Path) -> Result<Self, String> {
        crate::arrow_io::read_simulation_results_arrow(path)
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{FluxScore, Score};

    fn mk_tally(name: Option<&str>, id: Option<u32>) -> Arc<Tally> {
        let mut t = Tally::new();
        t.scores = vec![Score::Flux(FluxScore)];
        t.name = name.map(|s| s.to_string());
        t.tally_id = id;
        t.initialize_batches(1);
        Arc::new(t)
    }

    #[test]
    fn duplicate_name_rejected() {
        let a = mk_tally(Some("flux"), None);
        let b = mk_tally(Some("flux"), None);
        let err = SimulationResults::from_tallies(&[a, b], 0.0).unwrap_err();
        assert!(err.contains("duplicate tally name"), "got: {err}");
        assert!(err.contains("\"flux\""), "got: {err}");
    }

    #[test]
    fn duplicate_id_rejected() {
        let a = mk_tally(None, Some(7));
        let b = mk_tally(None, Some(7));
        let err = SimulationResults::from_tallies(&[a, b], 0.0).unwrap_err();
        assert!(err.contains("duplicate tally id"), "got: {err}");
        assert!(err.contains("7"), "got: {err}");
    }

    #[test]
    fn lookup_by_all_flavors() {
        let a = mk_tally(Some("flux"), Some(1));
        let b = mk_tally(None, Some(2));
        let c = mk_tally(Some("heat"), None);
        let tallies = vec![a.clone(), b.clone(), c.clone()];
        let results = SimulationResults::from_tallies(&tallies, 1.5).unwrap();

        assert_eq!(results.len(), 3);
        assert!(!results.is_empty());
        assert_eq!(results.elapsed_secs, 1.5);

        // by position
        assert!(results.get(0).is_some());
        assert!(results.get(2).is_some());
        assert!(results.get(3).is_none());

        // by name -- only for tallies that set one
        assert!(results.get_by_name("flux").is_some());
        assert!(results.get_by_name("heat").is_some());
        assert!(results.get_by_name("missing").is_none());

        // by id
        assert!(results.get_by_id(1).is_some());
        assert!(results.get_by_id(2).is_some());
        assert!(results.get_by_id(99).is_none());

        // by object identity
        assert!(results.get_by_tally(&a).is_some());
        assert!(results.get_by_tally(&b).is_some());
        assert!(results.get_by_tally(&c).is_some());

        // Insertion order preserved
        let names: Vec<&Option<String>> = results.iter().map(|r| &r.tally.name).collect();
        assert_eq!(
            names,
            vec![&Some("flux".into()), &None, &Some("heat".into())]
        );
    }

    #[test]
    fn empty_input_ok() {
        let results = SimulationResults::from_tallies(&[], 0.0).unwrap();
        assert!(results.is_empty());
        assert_eq!(results.n_batches, 0);
        assert_eq!(results.particles_per_chunk, 0);
    }

    #[test]
    fn available_names_and_ids_sorted() {
        let a = mk_tally(Some("zeta"), Some(3));
        let b = mk_tally(Some("alpha"), Some(1));
        let c = mk_tally(Some("mu"), Some(2));
        let results = SimulationResults::from_tallies(&[a, b, c], 0.0).unwrap();
        assert_eq!(results.available_names(), vec!["alpha", "mu", "zeta"]);
        assert_eq!(results.available_ids(), vec![1, 2, 3]);
    }

    /// Regression guard: the public constructor must leave every result with
    /// its figure-of-merit populated (sized to the bins) and the run's
    /// elapsed time recorded -- i.e. `with_fom` is applied. A bare
    /// `Tally::finalize` leaves `figure_of_merit` empty and `elapsed_secs`
    /// at 0, so this pins the "forgot with_fom" regression.
    #[test]
    fn from_tallies_populates_fom() {
        let a = mk_tally(Some("flux"), Some(1));
        let results = SimulationResults::from_tallies(&[a], 2.0).unwrap();
        let r = results.get(0).unwrap();
        assert_eq!(r.figure_of_merit.len(), r.mean.len());
        assert_eq!(r.elapsed_secs, 2.0);
    }

    /// `from_parts` is the central place that applies `with_fom`: a result
    /// handed in with an empty figure-of-merit but a non-zero `elapsed_secs`
    /// comes back with the FOM filled from that elapsed time.
    #[test]
    fn from_parts_applies_fom() {
        let t = mk_tally(Some("flux"), None);
        let mut r = t.finalize();
        r.mean = vec![1.0];
        r.standard_deviation = vec![0.1];
        r.relative_error = vec![0.1];
        r.elapsed_secs = 4.0;
        assert!(
            r.figure_of_merit.is_empty(),
            "bare finalize leaves FOM empty"
        );
        let arc = Arc::new(r);
        let mut by_ptr = std::collections::HashMap::new();
        by_ptr.insert(Arc::as_ptr(&arc.tally) as usize, 0usize);
        let sim = SimulationResults::from_parts(
            vec![arc],
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            by_ptr,
            1,
            1,
            4.0,
            Vec::new(),
        );
        let r = sim.get(0).unwrap();
        assert_eq!(r.figure_of_merit.len(), 1);
        // FOM = 1 / (0.1² × 4) = 25.
        assert!((r.figure_of_merit[0] - 25.0).abs() < 1e-9);
    }
}
