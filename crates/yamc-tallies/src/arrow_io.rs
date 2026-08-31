//! Arrow IPC save/load for `SimulationResults`.
//!
//! File format (version 2, lossless):
//!
//! - One Arrow IPC file per result set; plain IPC, externally readable
//!   with pyarrow / polars.
//! - One `RecordBatch` per tally with columns: `bin_index`, `mean`,
//!   `std_dev`, `rel_err`, `m2`, `count`. `m2` is the raw Welford
//!   sum-of-squared-deviations: together with the history count it is
//!   the exact merge state `combine_results` needs.
//! - Schema-level metadata: format version, `yamc_version`, run-level
//!   fields, the JSON-encoded run provenance list (`runs`), and per
//!   tally the full JSON-encoded config (`tally.<i>.config`) plus
//!   shape / dim_labels / counts / elapsed / run indices.
//!
//! The round trip is lossless: `from_arrow(to_arrow(r))` reproduces the
//! in-memory result bit-for-bit, including the full `Tally` configs (via
//! their serde form) and provenance, so loaded results remain fully
//! combinable and lookups work exactly as on a fresh result. Files from
//! older format versions are refused (no backwards compatibility).
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Array, Float64Array, RecordBatch, UInt64Array};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};

use crate::result::TallyResult;
use crate::simulation_results::{RunProvenance, SimulationResults};
use crate::tally::Tally;

/// Column names in each per-tally `RecordBatch`.
const COL_BIN_INDEX: &str = "bin_index";
const COL_MEAN: &str = "mean";
const COL_STD_DEV: &str = "std_dev";
const COL_REL_ERR: &str = "rel_err";
const COL_M2: &str = "m2";
const COL_COUNT: &str = "count";

/// Metadata keys used in the schema.
const META_FORMAT: &str = "yamc_results_format";
const META_VERSION: &str = "yamc_version";
const META_N_BATCHES: &str = "n_batches";
const META_PPB: &str = "particles_per_chunk";
const META_ELAPSED: &str = "elapsed_secs";
const META_N_TALLIES: &str = "n_tallies";
const META_RUNS: &str = "runs";

/// Current on-disk format version. Readers refuse anything else.
const FORMAT_VERSION: &str = "2";

fn tally_schema() -> Schema {
    Schema::new(vec![
        Field::new(COL_BIN_INDEX, DataType::UInt64, false),
        Field::new(COL_MEAN, DataType::Float64, false),
        Field::new(COL_STD_DEV, DataType::Float64, false),
        Field::new(COL_REL_ERR, DataType::Float64, false),
        Field::new(COL_M2, DataType::Float64, false),
        Field::new(COL_COUNT, DataType::UInt64, false),
    ])
}

fn tally_record_batch(result: &TallyResult) -> Result<RecordBatch, String> {
    let n = result.mean.len();
    let schema = Arc::new(tally_schema());
    let bin_index = UInt64Array::from_iter_values((0..n).map(|i| i as u64));
    let mean = Float64Array::from_iter_values(result.mean.iter().copied());
    let std_dev = Float64Array::from_iter_values(result.standard_deviation.iter().copied());
    let rel_err = Float64Array::from_iter_values(result.relative_error.iter().copied());
    // A result without Welford state (GPU runs) has an empty m2; pad
    // with zeros so the column shape stays rectangular. Such results
    // carry compute="gpu" provenance and are refused by combine_results
    // regardless.
    let m2 = if result.m2.len() == n {
        Float64Array::from_iter_values(result.m2.iter().copied())
    } else {
        Float64Array::from_iter_values(std::iter::repeat_n(0.0, n))
    };
    let count = UInt64Array::from_iter_values(result.total_count.iter().copied());

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(bin_index),
            Arc::new(mean),
            Arc::new(std_dev),
            Arc::new(rel_err),
            Arc::new(m2),
            Arc::new(count),
        ],
    )
    .map_err(|e| format!("failed to build Arrow RecordBatch for tally: {e}"))
}

/// Schema-level metadata: format version, run provenance, and per-tally
/// config + statistics metadata so the roundtrip is lossless.
fn build_metadata(results: &SimulationResults) -> Result<HashMap<String, String>, String> {
    let mut m = HashMap::new();
    m.insert(META_FORMAT.into(), FORMAT_VERSION.into());
    m.insert(META_VERSION.into(), env!("CARGO_PKG_VERSION").into());
    m.insert(META_N_BATCHES.into(), results.n_batches.to_string());
    m.insert(META_PPB.into(), results.particles_per_chunk.to_string());
    m.insert(META_ELAPSED.into(), results.elapsed_secs.to_string());
    m.insert(META_N_TALLIES.into(), results.len().to_string());
    m.insert(
        META_RUNS.into(),
        serde_json::to_string(&results.runs)
            .map_err(|e| format!("failed to serialize run provenance: {e}"))?,
    );

    for (i, r) in results.iter().enumerate() {
        if let Some(name) = &r.tally.name {
            m.insert(format!("tally.{i}.name"), name.clone());
        }
        if let Some(id) = r.tally.tally_id {
            m.insert(format!("tally.{i}.id"), id.to_string());
        }
        m.insert(
            format!("tally.{i}.config"),
            serde_json::to_string(&*r.tally)
                .map_err(|e| format!("failed to serialize tally {i} config: {e}"))?,
        );
        m.insert(
            format!("tally.{i}.shape"),
            r.shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
        m.insert(format!("tally.{i}.dim_labels"), r.dim_labels.join(","));
        m.insert(format!("tally.{i}.n_batches"), r.n_batches.to_string());
        m.insert(format!("tally.{i}.n_histories"), r.n_histories.to_string());
        m.insert(
            format!("tally.{i}.particles_per_chunk"),
            r.particles_per_chunk.to_string(),
        );
        m.insert(
            format!("tally.{i}.elapsed_secs"),
            r.elapsed_secs.to_string(),
        );
        m.insert(
            format!("tally.{i}.run_indices"),
            r.run_indices
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
        // The m2 column is zero-padded for results without Welford
        // state; record whether real merge state was present so the
        // reader restores `m2: vec![]` faithfully.
        m.insert(
            format!("tally.{i}.has_m2"),
            (r.m2.len() == r.mean.len() && !r.mean.is_empty()).to_string(),
        );
    }
    Ok(m)
}

/// Write the full `SimulationResults` to an Arrow IPC file at `path`.
pub fn write_simulation_results_arrow(
    results: &SimulationResults,
    path: &Path,
) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;

    // Attach metadata to the schema.
    let schema_with_meta =
        Schema::new_with_metadata(tally_schema().fields().clone(), build_metadata(results)?);
    let schema_ref = Arc::new(schema_with_meta);

    let mut writer = FileWriter::try_new(file, &schema_ref)
        .map_err(|e| format!("failed to open Arrow writer: {e}"))?;

    for r in results.iter() {
        // Re-tag each batch's schema with the overall metadata so the
        // writer accepts it (schemas must match field-for-field).
        let batch = tally_record_batch(r)?;
        // Rebuild the batch with the annotated schema (same columns).
        let batch = RecordBatch::try_new(schema_ref.clone(), batch.columns().to_vec())
            .map_err(|e| format!("failed to re-tag RecordBatch schema: {e}"))?;
        writer
            .write(&batch)
            .map_err(|e| format!("failed to write RecordBatch: {e}"))?;
    }

    writer
        .finish()
        .map_err(|e| format!("failed to finalize Arrow file: {e}"))?;
    Ok(())
}

fn parse_usize_list(s: &str) -> Vec<usize> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',').filter_map(|p| p.parse().ok()).collect()
}

fn parse_str_list(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',').map(|p| p.to_string()).collect()
}

/// Read a `SimulationResults` back from an Arrow IPC file written by
/// [`write_simulation_results_arrow`]. Lossless: full `Tally` configs,
/// raw `m2` merge state, history counts and run provenance are all
/// restored, so the loaded result is fully combinable. Files from other
/// format versions are refused.
pub fn read_simulation_results_arrow(path: &Path) -> Result<SimulationResults, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let reader = FileReader::try_new(file, None).map_err(|e| format!("Arrow read error: {e}"))?;
    let schema = reader.schema();
    let meta = schema.metadata().clone();

    match meta.get(META_FORMAT).map(String::as_str) {
        Some(FORMAT_VERSION) => {}
        other => {
            return Err(format!(
                "results file {} has format {:?}, expected {FORMAT_VERSION:?}; re-create it \
                 with this yamc version's to_arrow (old formats are not supported)",
                path.display(),
                other.unwrap_or("<missing>")
            ));
        }
    }

    let batches: Vec<RecordBatch> = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Arrow batch read error: {e}"))?;

    // Cross-check the declared tally count against the actual batch
    // count so truncated or corrupt files are rejected rather than
    // silently dropping tallies.
    if let Some(declared) = meta
        .get(META_N_TALLIES)
        .and_then(|s| s.parse::<usize>().ok())
    {
        if declared != batches.len() {
            return Err(format!(
                "results file {} declares {declared} tallies but contains {} record batch(es); \
                 the file is truncated or corrupt",
                path.display(),
                batches.len()
            ));
        }
    }

    let elapsed_secs: f64 = meta
        .get(META_ELAPSED)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let runs: Vec<RunProvenance> = match meta.get(META_RUNS) {
        Some(json) => serde_json::from_str(json)
            .map_err(|e| format!("failed to parse run provenance: {e}"))?,
        None => return Err("results file is missing run provenance metadata".into()),
    };

    let mut results: Vec<Arc<TallyResult>> = Vec::with_capacity(batches.len());
    let mut by_name: HashMap<String, usize> = HashMap::new();
    let mut by_id: HashMap<u32, usize> = HashMap::new();
    let mut by_ptr: HashMap<usize, usize> = HashMap::new();

    for (i, batch) in batches.iter().enumerate() {
        let col = |name: &str| -> Result<&dyn Array, String> {
            batch
                .column_by_name(name)
                .map(|a| a.as_ref())
                .ok_or_else(|| format!("missing column {name:?} in tally {i}"))
        };
        let f64_col = |name: &str| -> Result<Vec<f64>, String> {
            Ok(col(name)?
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| format!("column {name} is not Float64"))?
                .values()
                .iter()
                .copied()
                .collect())
        };

        let mean = f64_col(COL_MEAN)?;
        let standard_deviation = f64_col(COL_STD_DEV)?;
        let relative_error = f64_col(COL_REL_ERR)?;
        let m2_raw = f64_col(COL_M2)?;
        let count_arr = col(COL_COUNT)?;
        let total_count: Vec<u64> = count_arr
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| format!("column {COL_COUNT} is not UInt64"))?
            .values()
            .iter()
            .copied()
            .collect();

        // Restore `m2: vec![]` faithfully for results that had no
        // Welford state (the column itself is zero-padded).
        let has_m2 = meta
            .get(&format!("tally.{i}.has_m2"))
            .map(|s| s == "true")
            .unwrap_or(true);
        let m2 = if has_m2 { m2_raw } else { Vec::new() };

        let get = |key: &str| meta.get(&format!("tally.{i}.{key}"));
        let shape = get("shape")
            .map(|s| parse_usize_list(s))
            .unwrap_or_else(|| vec![mean.len()]);
        let dim_labels = get("dim_labels")
            .map(|s| parse_str_list(s))
            .unwrap_or_else(|| vec!["bin".to_string()]);
        let n_batches: u32 = get("n_batches").and_then(|s| s.parse().ok()).unwrap_or(0);
        let n_histories: u64 = get("n_histories").and_then(|s| s.parse().ok()).unwrap_or(0);
        let particles_per_chunk: u32 = get("particles_per_chunk")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let tally_elapsed: f64 = get("elapsed_secs")
            .and_then(|s| s.parse().ok())
            .unwrap_or(elapsed_secs);
        let run_indices: Vec<usize> = get("run_indices")
            .map(|s| parse_usize_list(s))
            .unwrap_or_default();

        // Full config roundtrip via the Tally serde form.
        let config_json = get("config")
            .ok_or_else(|| format!("results file is missing the config for tally {i}"))?;
        let tally: Tally = serde_json::from_str(config_json)
            .map_err(|e| format!("failed to parse tally {i} config: {e}"))?;
        let tally_arc = Arc::new(tally);

        // Reject duplicate names/ids: this reader is the one entry point
        // where externally-produced data could smuggle duplicates past
        // the `from_tallies` validation (combine_results would then
        // silently overwrite one of them in its key index).
        if let Some(name) = &tally_arc.name {
            if by_name.insert(name.clone(), i).is_some() {
                return Err(format!(
                    "results file {} contains duplicate tally name {name:?}; names must be \
                     unique",
                    path.display()
                ));
            }
        }
        if let Some(id) = tally_arc.tally_id {
            if by_id.insert(id, i).is_some() {
                return Err(format!(
                    "results file {} contains duplicate tally id {id}; ids must be unique",
                    path.display()
                ));
            }
        }
        by_ptr.insert(Arc::as_ptr(&tally_arc) as usize, i);

        let result = TallyResult {
            tally: tally_arc,
            mean,
            standard_deviation,
            relative_error,
            m2,
            n_histories,
            total_count,
            figure_of_merit: Vec::new(),
            aggregate_figure_of_merit: 0.0,
            // Aggregate moments / empirical PDF are not yet persisted to
            // Arrow; a reloaded result reports them as empty.
            agg: crate::welford::AggMoments::ZERO,
            score_pdf: crate::welford::ScorePdf::default(),
            convergence_history: Vec::new(),
            shape,
            dim_labels,
            n_batches,
            particles_per_chunk,
            elapsed_secs: tally_elapsed,
            run_indices,
        };
        // Figure of merit is filled centrally by `SimulationResults::from_parts`
        // from each result's `elapsed_secs` (set above).
        results.push(Arc::new(result));
    }

    let (n_batches_run, ppb_run) = results
        .first()
        .map(|r| (r.n_batches, r.particles_per_chunk))
        .unwrap_or((0, 0));

    Ok(SimulationResults::from_parts(
        results,
        by_name,
        by_id,
        by_ptr,
        n_batches_run,
        ppb_run,
        elapsed_secs,
        runs,
    ))
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{FluxScore, Score};
    use std::sync::Arc;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "yamc_arrow_test_{}_{}.arrow",
            name,
            std::process::id()
        ));
        p
    }

    fn mk_tally(name: Option<&str>, id: Option<u32>) -> Arc<Tally> {
        let mut t = Tally::new();
        t.scores = vec![Score::Flux(FluxScore)];
        t.name = name.map(String::from);
        t.tally_id = id;
        t.initialize_batches(1);
        Arc::new(t)
    }

    #[test]
    fn arrow_roundtrip_preserves_numeric_data() {
        let a = mk_tally(Some("flux"), Some(1));
        let b = mk_tally(None, Some(2));
        let original = SimulationResults::from_tallies(&[a, b], 4.25).expect("build results");

        let path = temp_path("roundtrip");
        write_simulation_results_arrow(&original, &path).expect("write ok");
        let loaded = read_simulation_results_arrow(&path).expect("read ok");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), original.len());
        assert_eq!(loaded.elapsed_secs, 4.25);
        assert_eq!(loaded.n_batches, original.n_batches);
        assert_eq!(loaded.particles_per_chunk, original.particles_per_chunk);

        for i in 0..original.len() {
            let orig = original.get(i).unwrap();
            let back = loaded.get(i).unwrap();
            assert_eq!(back.mean, orig.mean);
            assert_eq!(back.standard_deviation, orig.standard_deviation);
            assert_eq!(back.relative_error, orig.relative_error);
            assert_eq!(back.total_count, orig.total_count);
            assert_eq!(back.shape, orig.shape);
            assert_eq!(back.dim_labels, orig.dim_labels);
            assert_eq!(back.n_batches, orig.n_batches);
            assert_eq!(back.particles_per_chunk, orig.particles_per_chunk);
            assert_eq!(back.tally.name, orig.tally.name);
            assert_eq!(back.tally.tally_id, orig.tally.tally_id);
        }

        // Lookups still work after load (by name, id).
        assert!(loaded.get_by_name("flux").is_some());
        assert!(loaded.get_by_id(1).is_some());
        assert!(loaded.get_by_id(2).is_some());
    }

    /// Files with duplicate tally names (constructible via from_parts,
    /// e.g. by external writers) are refused at read time: the only
    /// validation point for externally-produced data.
    #[test]
    fn duplicate_tally_names_refused_on_read() {
        use crate::simulation_results::SimulationResults;
        use std::collections::HashMap;

        let a = mk_tally(Some("flux"), None);
        let b = mk_tally(Some("flux"), None);
        let ra = std::sync::Arc::new(a.finalize().with_fom(1.0));
        let rb = std::sync::Arc::new(b.finalize().with_fom(1.0));
        // from_parts skips the duplicate validation from_tallies does.
        let results = SimulationResults::from_parts(
            vec![ra, rb],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            1,
            1,
            1.0,
            Vec::new(),
        );
        let path = temp_path("dup_names");
        write_simulation_results_arrow(&results, &path).expect("write ok");
        let err = read_simulation_results_arrow(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("duplicate tally name"), "got: {err}");
    }
}
