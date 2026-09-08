//! `reactions.arrow`: the per-MT cross sections, plus the redundant sums.
//!
//! With `nuclide.arrow` this is everything a transport-free reaction-rate
//! collapse reads (see `NEUTRON_XS_ONLY_SECTIONS` in `yamc-nuclide`), so the two
//! together are a complete conversion for an activation calculation. The
//! products, distributions and lookup accelerators a transport run needs are a
//! separate concern.
//!
//! Written one record batch per row, which no other section does, so a consumer
//! can range-read a single MT out of the middle without decoding the rest.
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
use std::path::Path;

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

    let mut rows: Vec<Vec<arrow_array::ArrayRef>> = Vec::new();

    // The evaluation's own reactions first, each keeping its canonical name.
    for (&mt, rx) in &data.reactions {
        // In processed order, not the BTreeMap's lexicographic walk over the
        // temperature names, which would put 1200K before 250K.
        let mut xs_temperatures: Vec<String> = temperatures
            .iter()
            .filter(|t| rx.xs.contains_key(*t))
            .cloned()
            .collect();
        xs_temperatures.extend(rx.xs.keys().filter(|t| !temperatures.contains(t)).cloned());
        let xs_values: Vec<Vec<f64>> = xs_temperatures
            .iter()
            .map(|t| rx.xs.get(t).map(|x| x.y.clone()).unwrap_or_default())
            .collect();
        let xs_threshold_idx: Vec<i32> = xs_temperatures
            .iter()
            .map(|t| rx.xs.get(t).and_then(|x| x.threshold_idx).unwrap_or(0) as i32)
            .collect();

        rows.push(vec![
            ints(&[mt]),
            strings(&[endf::reaction_name(mt).unwrap_or_else(|| mt.to_string())]),
            floats(&[rx.q_reaction]),
            bools(&[rx.center_of_mass]),
            bools(&[rx.redundant]),
            string_list(&xs_temperatures),
            float_list_list(&xs_values),
            int_list(&xs_threshold_idx),
        ]);
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
        let xs_values: Vec<Vec<f64>> = temperatures
            .iter()
            .map(|t| {
                synthesized
                    .get(t)
                    .and_then(|m| m.get(&mt))
                    .cloned()
                    .unwrap_or_default()
            })
            .collect();
        rows.push(vec![
            ints(&[mt]),
            strings(&[endf::reaction_name(mt).unwrap_or_else(|| mt.to_string())]),
            floats(&[0.0]),
            bools(&[false]),
            bools(&[true]),
            string_list(&temperatures),
            float_list_list(&xs_values),
            int_list(&vec![0; temperatures.len()]),
        ]);
    }

    write_section_per_row(&dir.join("reactions.arrow"), "reactions.arrow", rows)
}
