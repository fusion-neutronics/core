//! `reactions.arrow`: the per-MT cross sections, plus the redundant sums.
//!
//! With `nuclide.arrow` this is everything a transport-free reaction-rate
//! collapse reads (see `NEUTRON_XS_ONLY_SECTIONS` in `yamc-nuclide`), so the two
//! together are a complete conversion for an activation calculation. The
//! products, distributions and lookup accelerators a transport run needs are a
//! separate concern.
//!
//! Written one record batch per (MT, temperature), which no other section does,
//! so a consumer can range-read a single cross section out of the middle
//! without decoding the rest (fusion-neutronics/core#100). Each row's
//! `xs_temperatures`, `xs_values` and `xs_threshold_idx` therefore carry one
//! element. The schema is the one the reader has always taken, and the reader
//! selects its temperature by searching each row's list and skipping the rows
//! that do not carry it, so a file in this shape and one written one batch per
//! MT load to the same reactions.
//!
//! # The label is `endf::reaction_name`, for every row
//!
//! Synthesized reactions used to get their own spellings here: `(n,non-elastic)`
//! for MT 3, `(n,inelastic)` for MT 4, `(n,disappearance)` for MT 101. That was
//! kept to match the published files, and it meant the format had two
//! vocabularies for the same MTs, only one of which `yamc_nuclide::REACTION_MT`
//! could resolve. Reading a label out of a file and tallying it did not work
//! (issue #438).
//!
//! Nothing reads this column back: the reader keys everything by `mt`, and no
//! Rust, Python or verification code touches `label`. It is there for a person
//! looking at a file, which is exactly why it should say what the format says.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::Array;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_ipc::reader::FileReader;
use endf::IncidentNeutron;

use crate::sections::*;
use crate::synthesis;

/// Every reaction's cross section on the nuclide's full energy grid, by MT,
/// for one temperature. Excludes the redundant MTs, which are rebuilt.
fn partials_on_grid(
    data: &IncidentNeutron,
    temperature: &str,
    n_energy: usize,
) -> BTreeMap<i32, Vec<f64>> {
    let mut out = BTreeMap::new();
    for (&mt, rx) in &data.reactions {
        if synthesis::SYNTHETIC_MTS.contains(&mt) {
            continue;
        }
        let Some(xs) = rx.xs.get(temperature) else {
            continue;
        };
        out.insert(
            mt,
            synthesis::on_grid(&xs.y, xs.threshold_idx.unwrap_or(0), n_energy),
        );
    }
    out
}

/// One row of `reactions.arrow`: one MT at one temperature.
#[allow(clippy::too_many_arguments)]
fn row(
    mt: i32,
    label: &str,
    q_value: f64,
    center_of_mass: bool,
    redundant: bool,
    temperature: &str,
    xs: Vec<f64>,
    threshold_idx: i32,
) -> Vec<ArrayRef> {
    vec![
        ints(&[mt]),
        strings(&[label.to_string()]),
        floats(&[q_value]),
        bools(&[center_of_mass]),
        bools(&[redundant]),
        string_list(&[temperature.to_string()]),
        float_list_list(&[xs]),
        int_list(&[threshold_idx]),
    ]
}

/// Write `reactions.arrow`.
pub fn write_reactions(data: &IncidentNeutron, dir: &Path) -> Result<(), Box<dyn Error>> {
    // A summation set that overlaps would double count into MT 3 and MT 1, and
    // the result stays plausible. Checked before anything is written.
    synthesis::groups_are_disjoint()?;

    let temperatures = data.temperatures();

    // The redundant MTs, per temperature. Built once here rather than per row.
    let mut synthesized: BTreeMap<String, BTreeMap<i32, Vec<f64>>> = BTreeMap::new();
    for t in &temperatures {
        let n_energy = data.energy.get(t).map(Vec::len).unwrap_or(0);
        if n_energy == 0 {
            continue;
        }
        let partials = partials_on_grid(data, t, n_energy);
        synthesized.insert(t.clone(), synthesis::synthesize(&partials, n_energy));
    }

    let mut rows: Vec<Vec<ArrayRef>> = Vec::new();

    // The evaluation's own reactions first, each keeping its canonical name,
    // with its temperatures back to back so one MT's batches are one run of
    // bytes and a reader wanting every temperature of it asks for one span.
    for (&mt, rx) in &data.reactions {
        // In processed order, not the BTreeMap's lexicographic walk over the
        // temperature names, which would put 1200K before 250K.
        let mut xs_temperatures: Vec<String> = temperatures
            .iter()
            .filter(|t| rx.xs.contains_key(*t))
            .cloned()
            .collect();
        xs_temperatures.extend(rx.xs.keys().filter(|t| !temperatures.contains(t)).cloned());
        let label = endf::reaction_name(mt).unwrap_or_else(|| mt.to_string());
        for t in &xs_temperatures {
            let xs = rx.xs.get(t).map(|x| x.y.clone()).unwrap_or_default();
            let threshold_idx = rx.xs.get(t).and_then(|x| x.threshold_idx).unwrap_or(0) as i32;
            rows.push(row(
                mt,
                &label,
                rx.q_reaction,
                rx.center_of_mass,
                rx.redundant,
                t,
                xs,
                threshold_idx,
            ));
        }
    }

    // Then the redundant sums the evaluation did not carry. A synthesized MT is
    // defined on the whole grid, so its threshold index is zero everywhere.
    //
    // SYNTHETIC_MTS is maintained in ascending order, so the rows come out
    // ordered without sorting here.
    for mt in synthesis::SYNTHETIC_MTS {
        if data.reactions.contains_key(&mt) {
            continue;
        }
        // Nothing synthesized this MT, which is what an evaluation with no
        // nuclide grid looks like: the loop above skipped every temperature, so
        // a row written here would carry an empty cross section at every one of
        // them. The loader skips its grid length check in exactly that
        // situation (nuclide_arrow.rs:496), so the row would load as a reaction
        // that resolves to a name and carries nothing (issue #12).
        if !synthesized.values().any(|per_mt| per_mt.contains_key(&mt)) {
            continue;
        }
        let label = endf::reaction_name(mt).unwrap_or_else(|| mt.to_string());
        for t in &temperatures {
            let xs = synthesized
                .get(t)
                .and_then(|m| m.get(&mt))
                .cloned()
                .unwrap_or_default();
            rows.push(row(mt, &label, 0.0, false, true, t, xs, 0));
        }
    }

    write_section_per_row(&dir.join("reactions.arrow"), "reactions.arrow", rows)
}

/// A column of `batch` by name, or an error naming it.
fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, Box<dyn Error>> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("reactions.arrow has no `{name}` column").into())
}

/// Rewrite an existing `reactions.arrow` one record batch per (MT, temperature).
///
/// For a library published one batch per MT: the cross sections are copied, not
/// recomputed, so NJOY does not run again and every value is what it was; only
/// the framing changes. `version.json` has to be reindexed afterwards
/// ([`crate::reaction_ranges::write_reaction_ranges`]), since every offset
/// moves. A file already in this shape rewrites to the same rows.
///
/// `dst` must not be `src`: the rows are read into memory first, so writing
/// over the source would destroy the input the moment the write fails.
pub fn rewrite_per_temperature(src: &Path, dst: &Path) -> Result<(), Box<dyn Error>> {
    if src == dst {
        return Err("rewrite_per_temperature: the destination must differ from the source".into());
    }
    let reader = FileReader::try_new(File::open(src)?, None)?;
    let mut rows: Vec<Vec<ArrayRef>> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let mts = column(&batch, "mt")?
            .as_primitive_opt::<Int32Type>()
            .ok_or("`mt` is not an Int32Array")?;
        let labels = column(&batch, "label")?
            .as_string_opt::<i32>()
            .ok_or("`label` is not a StringArray")?;
        let q_values = column(&batch, "Q_value")?
            .as_primitive_opt::<Float64Type>()
            .ok_or("`Q_value` is not a Float64Array")?;
        let center_of_mass = column(&batch, "center_of_mass")?
            .as_boolean_opt()
            .ok_or("`center_of_mass` is not a BooleanArray")?;
        let redundant = column(&batch, "redundant")?
            .as_boolean_opt()
            .ok_or("`redundant` is not a BooleanArray")?;
        let temperatures = column(&batch, "xs_temperatures")?
            .as_list_opt::<i32>()
            .ok_or("`xs_temperatures` is not a ListArray")?;
        let values = column(&batch, "xs_values")?
            .as_list_opt::<i32>()
            .ok_or("`xs_values` is not a ListArray")?;
        let thresholds = column(&batch, "xs_threshold_idx")?
            .as_list_opt::<i32>()
            .ok_or("`xs_threshold_idx` is not a ListArray")?;

        for r in 0..batch.num_rows() {
            let mt = mts.value(r);
            let row_temperatures = temperatures.value(r);
            let row_temperatures = row_temperatures
                .as_string_opt::<i32>()
                .ok_or("`xs_temperatures` items are not strings")?;
            let row_values = values.value(r);
            let row_values = row_values
                .as_list_opt::<i32>()
                .ok_or("`xs_values` items are not lists")?;
            let row_thresholds = thresholds.value(r);
            let row_thresholds = row_thresholds
                .as_primitive_opt::<Int32Type>()
                .ok_or("`xs_threshold_idx` items are not Int32")?;
            if row_values.len() != row_temperatures.len()
                || row_thresholds.len() != row_temperatures.len()
            {
                return Err(format!(
                    "MT {mt}: {} temperatures against {} cross sections and {} thresholds",
                    row_temperatures.len(),
                    row_values.len(),
                    row_thresholds.len()
                )
                .into());
            }
            for i in 0..row_temperatures.len() {
                let xs = row_values.value(i);
                let xs = xs
                    .as_primitive_opt::<Float64Type>()
                    .ok_or("`xs_values` inner items are not Float64")?
                    .values()
                    .to_vec();
                rows.push(row(
                    mt,
                    labels.value(r),
                    q_values.value(r),
                    center_of_mass.value(r),
                    redundant.value(r),
                    row_temperatures.value(i),
                    xs,
                    row_thresholds.value(i),
                ));
            }
        }
    }
    write_section_per_row(dst, "reactions.arrow", rows)
}
