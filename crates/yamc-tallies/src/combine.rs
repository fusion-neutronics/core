//! Exact pooling of results from independent simulation runs.
//!
//! [`combine_results`] merges the finalized tallies of two or more
//! `SimulationResults` produced by *independent same-model runs*
//! (different seeds) into a single result that is statistically
//! identical to one longer run: per-bin Chen/Welford merge of the raw
//! `(mean, m2, n_histories)` state, exactly the operation the transport
//! loop already uses to combine per-thread (and per-MPI-rank) partials.
//!
//! ## Guards
//!
//! Combining is refused (hard error) when it cannot be proven sound:
//!
//! - **Missing provenance**: results built without run metadata (legacy
//!   constructors) carry no seed/fingerprint to validate.
//! - **Shared base seed**: yamc derives per-particle RNG streams from
//!   `base_seed + global_particle_index x stride`, so two runs with the
//!   same base seed share histories and are not independent.
//! - **Fingerprint mismatch**: the runs came from different models
//!   (geometry, materials, sources, physics settings, or nuclear-data
//!   libraries -- library differences are named per nuclide).
//! - **GPU results / non-root MPI results**: carry no (complete)
//!   Welford merge state.
//! - **Same-name tallies with different configuration**: a structural
//!   mismatch (scores, filters, bins, estimator, ...) means the merge
//!   would pool different physical quantities.
//!
//! Tallies present in only some of the inputs are carried through with
//! the statistics of the runs that scored them, and reported as
//! warnings (not errors) so partial tally overlap stays usable.
//!
//! ## Memory
//!
//! The merge is an in-place fold: the first input seeds an accumulator
//! and each further input is merged into it, so peak memory is one
//! result-set plus the largest single input, independent of how many
//! inputs are combined.
use std::collections::HashMap;
use std::sync::Arc;

use crate::result::TallyResult;
use crate::simulation_results::{RunProvenance, SimulationResults};
use crate::tally::Tally;
use crate::welford::WelfordTallyStats;

/// Spacing between consecutive per-particle RNG seeds in the transport
/// loop (`yamc::model` derives particle seeds as
/// `base_seed + global_particle_index * PARTICLE_SEED_STRIDE`). Defined
/// here, and imported by the transport loop, so the stream-overlap
/// validation below can never drift from the actual seed derivation.
pub const PARTICLE_SEED_STRIDE: u64 = 152917;

/// Multiplicative inverse of [`PARTICLE_SEED_STRIDE`] modulo 2^64 (the
/// stride is odd, hence invertible). Recovers the index offset between
/// two seed progressions from their base-seed difference.
const PARTICLE_SEED_STRIDE_INV: u64 = 18157105691312717821;

/// Whether two runs' per-particle seed progressions overlap.
///
/// Run A's particle seeds are `{seed_a + g * S : g in [0, n_a)}` and
/// likewise for B. They collide iff `seed_b - seed_a == k * S (mod 2^64)`
/// for some integer `k` with `-n_b < k < n_a` (particle `g` of A is born
/// exactly where particle `g - k` of B is). Since #315 the collision
/// stream is keyed on the base seed too, so an overlap of this kind no
/// longer makes the two histories identical; it still makes them
/// correlated (same source births), which is enough to invalidate a
/// pooled error bar, so the refusal stands. Multiplying the seed delta
/// by the stride's modular inverse recovers `k`. This subsumes the
/// equal-seed case (`k == 0`) and also catches seeds a small
/// stride-multiple apart, which would silently share source births
/// despite being "different seeds".
/// `n_histories` is the finished-history count; with lost particles the
/// true index range can be marginally larger, which this check does not
/// attempt to cover.
fn seed_streams_overlap(a: &RunProvenance, b: &RunProvenance) -> bool {
    // Equal base seeds are always refused, independent of the history
    // counts: whatever histories did run were identical.
    if a.seed == b.seed {
        return true;
    }
    let k = b
        .seed
        .wrapping_sub(a.seed)
        .wrapping_mul(PARTICLE_SEED_STRIDE_INV);
    k < a.n_histories || k.wrapping_neg() < b.n_histories
}

/// Matching key for a tally across result sets: name preferred, id as
/// the fallback. Tallies with neither are never matched (pass-through).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TallyKey {
    Name(String),
    Id(u32),
}

fn key_of(tally: &Tally) -> Option<TallyKey> {
    if let Some(name) = &tally.name {
        return Some(TallyKey::Name(name.clone()));
    }
    tally.tally_id.map(TallyKey::Id)
}

fn key_display(key: &TallyKey) -> String {
    match key {
        TallyKey::Name(n) => format!("'{n}'"),
        TallyKey::Id(i) => format!("id {i}"),
    }
}

/// One tally's merge state inside the accumulator.
struct Entry {
    tally: Arc<Tally>,
    key: Option<TallyKey>,
    stats: WelfordTallyStats,
    total_count: Vec<u64>,
    shape: Vec<usize>,
    dim_labels: Vec<String>,
    particles_per_chunk: u32,
    elapsed_secs: f64,
    run_indices: Vec<usize>,
}

/// Combine two or more independent same-model runs into one pooled
/// result. Returns the combined results plus any pass-through warnings
/// (tallies that were not present in every input).
///
/// See the module docs for the validation rules. The merge is exact in
/// real arithmetic and deterministic for a given input order; it is not
/// bit-invariant under reordering (f64 pairwise merging is not
/// bit-associative).
pub fn combine_results(
    inputs: &[&SimulationResults],
) -> Result<(SimulationResults, Vec<String>), String> {
    if inputs.len() < 2 {
        return Err(format!(
            "combine_results needs at least two results (got {})",
            inputs.len()
        ));
    }
    let mut warnings: Vec<String> = Vec::new();
    // Each pass-through key is warned about once, however many inputs it
    // is missing from.
    let mut warned_keys: std::collections::HashSet<TallyKey> = std::collections::HashSet::new();

    // Seed the accumulator from the first input, then fold the rest in.
    validate_provenance(inputs[0])?;
    let mut runs: Vec<RunProvenance> = inputs[0].runs.clone();
    let mut entries: Vec<Entry> = Vec::with_capacity(inputs[0].len());
    let mut key_index: HashMap<TallyKey, usize> = HashMap::new();
    for (i, r) in inputs[0].iter().enumerate() {
        let entry = entry_from_result(r, 0)?;
        if let Some(key) = &entry.key {
            key_index.insert(key.clone(), entries.len());
        } else {
            warnings.push(format!(
                "tally at position {i} has neither a name nor a tally_id; it is carried \
                 through unchanged (set Tally.name to enable combining)"
            ));
        }
        entries.push(entry);
    }

    for input in &inputs[1..] {
        validate_provenance(input)?;
        validate_compatible(&runs, input)?;
        let run_offset = runs.len();
        runs.extend(input.runs.iter().cloned());

        let mut matched_existing = vec![false; entries.len()];
        for (i, r) in input.iter().enumerate() {
            let entry = entry_from_result(r, run_offset)?;
            match entry.key.clone() {
                Some(key) => {
                    if let Some(&idx) = key_index.get(&key) {
                        merge_into(&mut entries[idx], entry, &key)?;
                        matched_existing[idx] = true;
                    } else {
                        if warned_keys.insert(key.clone()) {
                            warnings.push(format!(
                                "tally {} is not present in all combined results; it is \
                                 carried with the statistics of the runs that scored it",
                                key_display(&key)
                            ));
                        }
                        key_index.insert(key, entries.len());
                        entries.push(entry);
                    }
                }
                None => {
                    warnings.push(format!(
                        "tally at position {i} has neither a name nor a tally_id; it is \
                         carried through unchanged (set Tally.name to enable combining)"
                    ));
                    entries.push(entry);
                }
            }
        }
        for (idx, matched) in matched_existing.iter().enumerate() {
            if !matched {
                if let Some(key) = &entries[idx].key {
                    if warned_keys.insert(key.clone()) {
                        warnings.push(format!(
                            "tally {} is not present in all combined results; it is carried \
                             with the statistics of the runs that scored it",
                            key_display(key)
                        ));
                    }
                }
            }
        }
    }

    // Assemble the combined SimulationResults.
    let mut results: Vec<Arc<TallyResult>> = Vec::with_capacity(entries.len());
    let mut by_name: HashMap<String, usize> = HashMap::new();
    let mut by_id: HashMap<u32, usize> = HashMap::new();
    let mut by_ptr: HashMap<usize, usize> = HashMap::new();
    for (i, entry) in entries.into_iter().enumerate() {
        let result = finish_entry(entry, &mut by_name, &mut by_id, i);
        by_ptr.insert(Arc::as_ptr(&result.tally) as usize, i);
        results.push(Arc::new(result));
    }

    let (n_batches, particles_per_chunk) = results
        .first()
        .map(|r| (r.n_batches, r.particles_per_chunk))
        .unwrap_or((0, 0));
    let elapsed_total: f64 = runs.iter().map(|r| r.elapsed_secs).sum();

    Ok((
        SimulationResults::from_parts(
            results,
            by_name,
            by_id,
            by_ptr,
            n_batches,
            particles_per_chunk,
            elapsed_total,
            runs,
        ),
        warnings,
    ))
}

/// A result must carry run provenance and complete (CPU, MPI-root)
/// statistics to be combinable.
fn validate_provenance(input: &SimulationResults) -> Result<(), String> {
    if input.runs.is_empty() {
        return Err(
            "result carries no run provenance (it was built without simulate_transport \
             metadata, e.g. by an older yamc version) and cannot be combined"
                .into(),
        );
    }
    for run in &input.runs {
        if run.compute != "cpu" {
            return Err(format!(
                "cannot combine {}-produced results: only the CPU path records the per-bin \
                 Welford state an exact merge requires",
                run.compute
            ));
        }
        if run.mpi_size > 1 && run.mpi_rank != 0 {
            return Err(format!(
                "cannot combine a non-root MPI result (rank {} of {}): only rank 0 holds the \
                 complete reduced statistics",
                run.mpi_rank, run.mpi_size
            ));
        }
    }
    Ok(())
}

/// Cross-input checks: same model fingerprint everywhere, all base
/// seeds distinct.
fn validate_compatible(
    accumulated: &[RunProvenance],
    next: &SimulationResults,
) -> Result<(), String> {
    let reference = &accumulated[0];
    for run in &next.runs {
        if run.fingerprint != reference.fingerprint {
            // Try to give a precise reason: differing nuclear data is the
            // named case, otherwise a generic identity mismatch.
            for (k, lib) in &run.data_libraries {
                if let Some(ref_lib) = reference.data_libraries.get(k) {
                    if lib != ref_lib {
                        return Err(format!(
                            "cannot combine: nuclear data differs for {k}: \
                             \"{ref_lib}\" vs \"{lib}\""
                        ));
                    }
                } else {
                    return Err(format!(
                        "cannot combine: nuclear data for {k} is present in one model only"
                    ));
                }
            }
            return Err(
                "cannot combine: the results come from different models (geometry, \
                 materials, source or physics settings differ)"
                    .into(),
            );
        }
    }
    for run in &next.runs {
        if let Some(prev) = accumulated.iter().find(|r| seed_streams_overlap(r, run)) {
            if prev.seed == run.seed {
                return Err(format!(
                    "cannot combine: two runs share base seed {}; yamc derives per-particle \
                     RNG streams from the base seed, so their histories overlap and are not \
                     independent. Run each simulation with a distinct seed (e.g. \
                     simulate_transport(seed={}))",
                    run.seed,
                    prev.seed.wrapping_add(1)
                ));
            }
            return Err(format!(
                "cannot combine: runs with base seeds {} and {} have overlapping per-particle \
                 RNG streams (the seeds differ by a small multiple of the particle stride \
                 {PARTICLE_SEED_STRIDE}, so they re-sample the same source births). Choose \
                 seeds that are not separated by small stride multiples, e.g. consecutive \
                 integers",
                prev.seed, run.seed
            ));
        }
    }
    Ok(())
}

/// Extract a result's merge state, remapping its run indices by
/// `run_offset` (its position in the concatenated run list).
fn entry_from_result(r: &Arc<TallyResult>, run_offset: usize) -> Result<Entry, String> {
    let label = r
        .tally
        .name
        .clone()
        .or_else(|| r.tally.tally_id.map(|i| format!("id {i}")))
        .unwrap_or_else(|| "<unnamed>".into());
    if r.m2.len() != r.mean.len() || r.n_histories == 0 {
        return Err(format!(
            "tally {label} carries no Welford merge state (m2/history count missing); it \
             was produced by a path that cannot be combined exactly"
        ));
    }
    Ok(Entry {
        key: key_of(&r.tally),
        tally: r.tally.clone(),
        stats: WelfordTallyStats {
            mean: r.mean.clone(),
            m2: r.m2.clone(),
            n_histories: r.n_histories,
            agg: r.agg,
            score_pdf: r.score_pdf.clone(),
        },
        total_count: r.total_count.clone(),
        shape: r.shape.clone(),
        dim_labels: r.dim_labels.clone(),
        particles_per_chunk: r.particles_per_chunk,
        elapsed_secs: r.elapsed_secs,
        run_indices: r.run_indices.iter().map(|&i| i + run_offset).collect(),
    })
}

/// Merge `incoming` into the accumulator entry for the same key.
fn merge_into(acc: &mut Entry, incoming: Entry, key: &TallyKey) -> Result<(), String> {
    // Configuration equality (`Tally::eq` goes through the TallySerde
    // projection: scores incl. order, filters and their bin edges,
    // estimator, multiply_density, nuclides, ...) plus the result shape
    // must agree, otherwise the merge would pool different physical
    // quantities.
    if *acc.tally != *incoming.tally {
        return Err(format!(
            "cannot combine: tallies named {} have different configurations (scores, \
             filters, estimator or bin structure differ)",
            key_display(key)
        ));
    }
    if acc.shape != incoming.shape || acc.stats.mean.len() != incoming.stats.mean.len() {
        return Err(format!(
            "cannot combine: tallies named {} have different shapes ({:?} vs {:?})",
            key_display(key),
            acc.shape,
            incoming.shape
        ));
    }
    acc.stats.combine(&incoming.stats)?;
    for (a, b) in acc.total_count.iter_mut().zip(incoming.total_count.iter()) {
        *a = a.saturating_add(*b);
    }
    acc.elapsed_secs += incoming.elapsed_secs;
    acc.run_indices.extend(incoming.run_indices);
    Ok(())
}

/// Convert a finished entry into a `TallyResult`, recomputing the
/// derived quantities from the merged Welford state. The FOM uses the
/// summed elapsed time of exactly the runs that contributed to this
/// tally (so pass-through tallies keep their own run's time).
fn finish_entry(
    entry: Entry,
    by_name: &mut HashMap<String, usize>,
    by_id: &mut HashMap<u32, usize>,
    index: usize,
) -> TallyResult {
    if let Some(name) = &entry.tally.name {
        by_name.insert(name.clone(), index);
    }
    if let Some(id) = entry.tally.tally_id {
        by_id.insert(id, index);
    }
    let standard_deviation = entry.stats.std_err();
    // Mirrors `Tally::get_rel_error`: zero when the mean is non-positive.
    let relative_error: Vec<f64> = entry
        .stats
        .mean
        .iter()
        .zip(standard_deviation.iter())
        .map(|(&m, &s)| if m > 0.0 { s / m } else { 0.0 })
        .collect();
    let n_batches = u32::try_from(entry.stats.n_histories).unwrap_or(u32::MAX);
    // Figure of merit is filled centrally by `SimulationResults::from_parts`
    // from each result's `elapsed_secs` (set below).
    TallyResult {
        tally: entry.tally,
        mean: entry.stats.mean,
        standard_deviation,
        relative_error,
        m2: entry.stats.m2,
        n_histories: entry.stats.n_histories,
        total_count: entry.total_count,
        figure_of_merit: Vec::new(),
        aggregate_figure_of_merit: 0.0,
        agg: entry.stats.agg,
        score_pdf: entry.stats.score_pdf,
        // Per-run series is not merged across combine_results.
        convergence_history: Vec::new(),
        shape: entry.shape,
        dim_labels: entry.dim_labels,
        n_batches,
        particles_per_chunk: entry.particles_per_chunk,
        elapsed_secs: entry.elapsed_secs,
        run_indices: entry.run_indices,
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{FluxScore, HeatingScore, Score};
    use std::collections::BTreeMap;

    fn run(seed: u64, fingerprint: &str) -> RunProvenance {
        RunProvenance {
            seed,
            n_histories: 0,
            elapsed_secs: 1.0,
            fingerprint: fingerprint.into(),
            data_libraries: BTreeMap::new(),
            compute: "cpu".into(),
            yamc_version: "test".into(),
            mpi_size: 1,
            mpi_rank: 0,
        }
    }

    fn tally(name: &str) -> Arc<Tally> {
        let mut t = Tally::new();
        t.scores = vec![Score::Flux(FluxScore)];
        t.name = Some(name.into());
        t.initialize_batches(1);
        Arc::new(t)
    }

    /// TallyResult holding the Welford state of explicit per-history
    /// samples: mean and m2 computed two-pass, n = samples.len().
    fn result_from_samples(tally: Arc<Tally>, samples: &[f64], elapsed: f64) -> Arc<TallyResult> {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let m2: f64 = samples.iter().map(|&x| (x - mean) * (x - mean)).sum();
        Arc::new(TallyResult {
            tally,
            mean: vec![mean],
            standard_deviation: vec![0.0],
            relative_error: vec![0.0],
            m2: vec![m2],
            n_histories: samples.len() as u64,
            total_count: vec![samples.len() as u64],
            figure_of_merit: Vec::new(),
            aggregate_figure_of_merit: 0.0,
            agg: crate::welford::AggMoments::ZERO,
            score_pdf: crate::welford::ScorePdf::default(),
            convergence_history: Vec::new(),
            shape: vec![1],
            dim_labels: vec!["score".into()],
            n_batches: samples.len() as u32,
            particles_per_chunk: 1,
            elapsed_secs: elapsed,
            run_indices: vec![0],
        })
    }

    fn sim(results: Vec<Arc<TallyResult>>, runs: Vec<RunProvenance>) -> SimulationResults {
        let mut by_name = HashMap::new();
        let mut by_id = HashMap::new();
        let mut by_ptr = HashMap::new();
        for (i, r) in results.iter().enumerate() {
            if let Some(name) = &r.tally.name {
                by_name.insert(name.clone(), i);
            }
            if let Some(id) = r.tally.tally_id {
                by_id.insert(id, i);
            }
            by_ptr.insert(Arc::as_ptr(&r.tally) as usize, i);
        }
        let elapsed: f64 = runs.iter().map(|r| r.elapsed_secs).sum();
        SimulationResults::from_parts(results, by_name, by_id, by_ptr, 0, 0, elapsed, runs)
    }

    #[test]
    fn merge_matches_two_pass_pooled_statistics() {
        // A: samples [1,2,3]; B: samples [4,5]. Pooled over [1..5]:
        // mean = 3, m2 = 4+1+0+1+4 = 10, n = 5.
        let a = sim(
            vec![result_from_samples(tally("flux"), &[1.0, 2.0, 3.0], 2.0)],
            vec![run(1, "fp")],
        );
        let b = sim(
            vec![result_from_samples(tally("flux"), &[4.0, 5.0], 3.0)],
            vec![run(2, "fp")],
        );
        let (c, warnings) = combine_results(&[&a, &b]).unwrap();
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        let r = c.get_by_name("flux").unwrap();
        assert!((r.mean[0] - 3.0).abs() < 1e-12);
        assert!((r.m2[0] - 10.0).abs() < 1e-12);
        assert_eq!(r.n_histories, 5);
        // std error of pooled mean: sqrt(m2 / ((n-1) n)) = sqrt(10/20)
        assert!((r.standard_deviation[0] - (0.5_f64).sqrt()).abs() < 1e-12);
        // FOM uses the summed elapsed time of the contributing runs.
        assert!((r.elapsed_secs - 5.0).abs() < 1e-12);
        // total counts add.
        assert_eq!(r.total_count[0], 5);
        // run provenance concatenates and the tally points at both runs.
        assert_eq!(c.runs.len(), 2);
        assert_eq!(r.run_indices, vec![0, 1]);
    }

    #[test]
    fn variadic_three_way_merge_is_associative_within_fp() {
        let s1 = [1.0, 2.0];
        let s2 = [3.0, 4.0, 5.0];
        let s3 = [6.0];
        let all: Vec<f64> = s1.iter().chain(&s2).chain(&s3).copied().collect();
        let n = all.len() as f64;
        let mean_ref = all.iter().sum::<f64>() / n;
        let m2_ref: f64 = all.iter().map(|&x| (x - mean_ref) * (x - mean_ref)).sum();

        let a = sim(
            vec![result_from_samples(tally("flux"), &s1, 1.0)],
            vec![run(1, "fp")],
        );
        let b = sim(
            vec![result_from_samples(tally("flux"), &s2, 1.0)],
            vec![run(2, "fp")],
        );
        let c = sim(
            vec![result_from_samples(tally("flux"), &s3, 1.0)],
            vec![run(3, "fp")],
        );
        let (m, _) = combine_results(&[&a, &b, &c]).unwrap();
        let r = m.get_by_name("flux").unwrap();
        assert!((r.mean[0] - mean_ref).abs() < 1e-12);
        assert!((r.m2[0] - m2_ref).abs() < 1e-12);
        assert_eq!(r.n_histories, 6);
    }

    #[test]
    fn single_history_run_merges_exactly() {
        // n=1 has m2 = 0 and std_err forced to 0; only carrying raw m2
        // makes this mergeable at all.
        let a = sim(
            vec![result_from_samples(tally("flux"), &[10.0], 1.0)],
            vec![run(1, "fp")],
        );
        let b = sim(
            vec![result_from_samples(
                tally("flux"),
                &[2.0, 4.0, 6.0, 8.0],
                1.0,
            )],
            vec![run(2, "fp")],
        );
        let all = [10.0, 2.0, 4.0, 6.0, 8.0];
        let mean_ref = all.iter().sum::<f64>() / 5.0;
        let m2_ref: f64 = all.iter().map(|&x| (x - mean_ref) * (x - mean_ref)).sum();
        let (m, _) = combine_results(&[&a, &b]).unwrap();
        let r = m.get_by_name("flux").unwrap();
        assert!((r.mean[0] - mean_ref).abs() < 1e-12);
        assert!((r.m2[0] - m2_ref).abs() < 1e-12);
    }

    #[test]
    fn history_count_does_not_truncate_to_u32() {
        let mk = |n: u64, seed: u64| {
            let mut r = (*result_from_samples(tally("flux"), &[1.0, 2.0], 1.0)).clone();
            r.n_histories = n;
            sim(vec![Arc::new(r)], vec![run(seed, "fp")])
        };
        let a = mk(3_000_000_000, 1);
        let b = mk(3_000_000_000, 2);
        let (m, _) = combine_results(&[&a, &b]).unwrap();
        let r = m.get_by_name("flux").unwrap();
        assert_eq!(r.n_histories, 6_000_000_000);
        // The legacy u32 mirror saturates rather than wrapping.
        assert_eq!(r.n_batches, u32::MAX);
    }

    #[test]
    fn refuses_fewer_than_two_inputs() {
        let a = sim(
            vec![result_from_samples(tally("flux"), &[1.0, 2.0], 1.0)],
            vec![run(1, "fp")],
        );
        let err = combine_results(&[&a]).unwrap_err();
        assert!(err.contains("at least two"), "got: {err}");
    }

    #[test]
    fn refuses_missing_provenance() {
        let a = sim(
            vec![result_from_samples(tally("flux"), &[1.0, 2.0], 1.0)],
            vec![],
        );
        let b = sim(
            vec![result_from_samples(tally("flux"), &[3.0, 4.0], 1.0)],
            vec![run(2, "fp")],
        );
        let err = combine_results(&[&a, &b]).unwrap_err();
        assert!(err.contains("provenance"), "got: {err}");
    }

    #[test]
    fn refuses_shared_seed() {
        let a = sim(
            vec![result_from_samples(tally("flux"), &[1.0, 2.0], 1.0)],
            vec![run(7, "fp")],
        );
        let b = sim(
            vec![result_from_samples(tally("flux"), &[3.0, 4.0], 1.0)],
            vec![run(7, "fp")],
        );
        let err = combine_results(&[&a, &b]).unwrap_err();
        assert!(err.contains("share base seed 7"), "got: {err}");
        assert!(err.contains("simulate_transport(seed="), "got: {err}");
    }

    #[test]
    fn refuses_fingerprint_mismatch_naming_libraries() {
        let mut run_a = run(1, "fp-a");
        run_a
            .data_libraries
            .insert("n:Co58".into(), "endf-b8.1".into());
        let mut run_b = run(2, "fp-b");
        run_b
            .data_libraries
            .insert("n:Co58".into(), "fendl-3.2d".into());
        let a = sim(
            vec![result_from_samples(tally("flux"), &[1.0, 2.0], 1.0)],
            vec![run_a],
        );
        let b = sim(
            vec![result_from_samples(tally("flux"), &[3.0, 4.0], 1.0)],
            vec![run_b],
        );
        let err = combine_results(&[&a, &b]).unwrap_err();
        assert!(
            err.contains("n:Co58") && err.contains("endf-b8.1") && err.contains("fendl-3.2d"),
            "got: {err}"
        );
    }

    #[test]
    fn refuses_generic_fingerprint_mismatch() {
        let a = sim(
            vec![result_from_samples(tally("flux"), &[1.0, 2.0], 1.0)],
            vec![run(1, "fp-a")],
        );
        let b = sim(
            vec![result_from_samples(tally("flux"), &[3.0, 4.0], 1.0)],
            vec![run(2, "fp-b")],
        );
        let err = combine_results(&[&a, &b]).unwrap_err();
        assert!(err.contains("different models"), "got: {err}");
    }

    #[test]
    fn refuses_gpu_results() {
        let mut gpu_run = run(1, "fp");
        gpu_run.compute = "gpu".into();
        let a = sim(
            vec![result_from_samples(tally("flux"), &[1.0, 2.0], 1.0)],
            vec![gpu_run],
        );
        let b = sim(
            vec![result_from_samples(tally("flux"), &[3.0, 4.0], 1.0)],
            vec![run(2, "fp")],
        );
        let err = combine_results(&[&a, &b]).unwrap_err();
        assert!(err.contains("gpu"), "got: {err}");
    }

    #[test]
    fn refuses_non_root_mpi_results() {
        let mut rank1 = run(1, "fp");
        rank1.mpi_size = 2;
        rank1.mpi_rank = 1;
        let a = sim(
            vec![result_from_samples(tally("flux"), &[1.0, 2.0], 1.0)],
            vec![rank1],
        );
        let b = sim(
            vec![result_from_samples(tally("flux"), &[3.0, 4.0], 1.0)],
            vec![run(2, "fp")],
        );
        let err = combine_results(&[&a, &b]).unwrap_err();
        assert!(err.contains("non-root MPI"), "got: {err}");
    }

    #[test]
    fn refuses_same_name_different_config() {
        let mut t2 = Tally::new();
        t2.scores = vec![Score::Flux(FluxScore), Score::Heating(HeatingScore)];
        t2.name = Some("flux".into());
        t2.initialize_batches(1);
        let mut r2 = (*result_from_samples(Arc::new(t2), &[3.0, 4.0], 1.0)).clone();
        r2.mean = vec![3.5, 1.0];
        r2.m2 = vec![0.5, 0.1];
        r2.standard_deviation = vec![0.0, 0.0];
        r2.relative_error = vec![0.0, 0.0];
        r2.total_count = vec![2, 2];
        r2.shape = vec![2];

        let a = sim(
            vec![result_from_samples(tally("flux"), &[1.0, 2.0], 1.0)],
            vec![run(1, "fp")],
        );
        let b = sim(vec![Arc::new(r2)], vec![run(2, "fp")]);
        let err = combine_results(&[&a, &b]).unwrap_err();
        assert!(err.contains("different configurations"), "got: {err}");
    }

    #[test]
    fn partial_overlap_passes_through_with_warning() {
        let a = sim(
            vec![
                result_from_samples(tally("flux"), &[1.0, 2.0, 3.0], 2.0),
                result_from_samples(tally("dose"), &[7.0, 9.0], 2.0),
            ],
            vec![run(1, "fp")],
        );
        let b = sim(
            vec![result_from_samples(tally("flux"), &[4.0, 5.0], 3.0)],
            vec![run(2, "fp")],
        );
        let (c, warnings) = combine_results(&[&a, &b]).unwrap();
        assert!(
            warnings.iter().any(|w| w.contains("'dose'")),
            "expected a pass-through warning for dose, got: {warnings:?}"
        );
        // dose carried unchanged: original mean/m2/n and its own elapsed.
        let dose = c.get_by_name("dose").unwrap();
        assert!((dose.mean[0] - 8.0).abs() < 1e-12);
        assert_eq!(dose.n_histories, 2);
        assert!((dose.elapsed_secs - 2.0).abs() < 1e-12);
        assert_eq!(dose.run_indices, vec![0]);
        // flux merged across both runs.
        let flux = c.get_by_name("flux").unwrap();
        assert_eq!(flux.n_histories, 5);
        assert_eq!(flux.run_indices, vec![0, 1]);
        // combined results remain combinable: provenance concatenated.
        assert_eq!(c.runs.len(), 2);
    }

    #[test]
    fn combined_result_recombines_with_another_run() {
        // (a + b) + c must validate seeds across ALL prior runs.
        let mk = |seed: u64, samples: &[f64]| {
            sim(
                vec![result_from_samples(tally("flux"), samples, 1.0)],
                vec![run(seed, "fp")],
            )
        };
        let a = mk(1, &[1.0, 2.0]);
        let b = mk(2, &[3.0, 4.0]);
        let (ab, _) = combine_results(&[&a, &b]).unwrap();

        // Recombining with a seed already inside ab is refused.
        let c_dup = mk(2, &[5.0, 6.0]);
        let err = combine_results(&[&ab, &c_dup]).unwrap_err();
        assert!(err.contains("share base seed 2"), "got: {err}");

        // A fresh seed extends the pool.
        let c = mk(3, &[5.0, 6.0]);
        let (abc, _) = combine_results(&[&ab, &c]).unwrap();
        let r = abc.get_by_name("flux").unwrap();
        assert_eq!(r.n_histories, 6);
        assert_eq!(abc.runs.len(), 3);
    }

    #[test]
    fn zero_bin_merges_with_nonzero_bin() {
        // bin layout: one tally with 2 bins; run A scored only bin 0,
        // run B scored only bin 1. Pooled statistics must match the
        // two-pass result over the union (zeros included).
        let mk = |seed: u64, samples0: &[f64], samples1: &[f64], name: &str| {
            let n = samples0.len();
            assert_eq!(n, samples1.len());
            let two_pass = |s: &[f64]| {
                let nf = s.len() as f64;
                let mean = s.iter().sum::<f64>() / nf;
                let m2 = s.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>();
                (mean, m2)
            };
            let (mean0, m20) = two_pass(samples0);
            let (mean1, m21) = two_pass(samples1);
            let mut r = (*result_from_samples(tally(name), &[0.0], 1.0)).clone();
            r.mean = vec![mean0, mean1];
            r.m2 = vec![m20, m21];
            r.standard_deviation = vec![0.0, 0.0];
            r.relative_error = vec![0.0, 0.0];
            r.total_count = vec![n as u64, n as u64];
            r.shape = vec![2];
            r.n_histories = n as u64;
            r.n_batches = n as u32;
            sim(vec![Arc::new(r)], vec![run(seed, "fp")])
        };
        let a = mk(1, &[5.0, 7.0], &[0.0, 0.0], "flux");
        let b = mk(2, &[0.0, 0.0, 0.0], &[2.0, 4.0, 6.0], "flux");
        let (c, _) = combine_results(&[&a, &b]).unwrap();
        let r = c.get_by_name("flux").unwrap();

        let bin0: Vec<f64> = vec![5.0, 7.0, 0.0, 0.0, 0.0];
        let bin1: Vec<f64> = vec![0.0, 0.0, 2.0, 4.0, 6.0];
        for (bin, samples) in [(0usize, bin0), (1usize, bin1)] {
            let nf = samples.len() as f64;
            let mean_ref = samples.iter().sum::<f64>() / nf;
            let m2_ref: f64 = samples
                .iter()
                .map(|&x| (x - mean_ref) * (x - mean_ref))
                .sum();
            assert!(
                (r.mean[bin] - mean_ref).abs() < 1e-12,
                "bin {bin} mean {} vs {}",
                r.mean[bin],
                mean_ref
            );
            assert!(
                (r.m2[bin] - m2_ref).abs() < 1e-12,
                "bin {bin} m2 {} vs {}",
                r.m2[bin],
                m2_ref
            );
        }
    }

    #[test]
    fn refuses_stride_overlapping_seeds() {
        // seed_b = seed_a + 5 * stride: run B's particle 0 re-traces run
        // A's particle 5. Must be refused even though the seeds differ.
        let mk = |seed: u64, n: u64, run_seed: u64| {
            let mut r = (*result_from_samples(tally("flux"), &[1.0, 2.0], 1.0)).clone();
            r.n_histories = n;
            let mut rp = run(run_seed, "fp");
            rp.n_histories = n;
            let _ = seed;
            sim(vec![Arc::new(r)], vec![rp])
        };
        let a = mk(0, 1_000, 100);
        let b = mk(0, 1_000, 100 + 5 * PARTICLE_SEED_STRIDE);
        let err = combine_results(&[&a, &b]).unwrap_err();
        assert!(
            err.contains("overlapping per-particle RNG streams"),
            "got: {err}"
        );

        // Consecutive-integer seeds never overlap (the stride is huge in
        // index space).
        let c = mk(0, 1_000, 100);
        let d = mk(0, 1_000, 101);
        assert!(combine_results(&[&c, &d]).is_ok());

        // A stride multiple beyond either run's history count does not
        // overlap either.
        let e = mk(0, 1_000, 100);
        let f = mk(0, 1_000, 100 + 2_000 * PARTICLE_SEED_STRIDE);
        assert!(combine_results(&[&e, &f]).is_ok());
    }

    #[test]
    fn pass_through_warning_emitted_once_for_many_inputs() {
        let a = sim(
            vec![
                result_from_samples(tally("flux"), &[1.0, 2.0], 1.0),
                result_from_samples(tally("dose"), &[7.0, 9.0], 1.0),
            ],
            vec![run(1, "fp")],
        );
        let b = sim(
            vec![result_from_samples(tally("flux"), &[3.0], 1.0)],
            vec![run(2, "fp")],
        );
        let c = sim(
            vec![result_from_samples(tally("flux"), &[4.0], 1.0)],
            vec![run(3, "fp")],
        );
        let (_, warnings) = combine_results(&[&a, &b, &c]).unwrap();
        let dose_warnings = warnings.iter().filter(|w| w.contains("'dose'")).count();
        assert_eq!(dose_warnings, 1, "got warnings: {warnings:?}");
    }
}
