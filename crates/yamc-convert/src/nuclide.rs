//! `nuclide.arrow` and `urr.arrow`: the nuclide's identity and its unresolved
//! resonance probability tables.
//!
//! The two most direct sections. Almost every value comes straight off the
//! parsed table, and the URR data is already stored flat with an explicit
//! shape because that is the form the format wants.

use std::error::Error;
use std::fs::File;
use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_ipc::reader::FileReader;

use endf::IncidentNeutron;

use crate::sections::*;

/// Whether the nuclide has a fission channel.
///
/// Recorded at conversion time so the loader does not rescan every reaction to
/// find out. MT 18 is total fission; 19, 20, 21 and 38 are its partials, and an
/// evaluation may carry the partials without the total.
pub fn is_fissionable(data: &IncidentNeutron) -> bool {
    [18, 19, 20, 21, 38].iter().any(|mt| data.contains(*mt))
}

/// Write `nuclide.arrow`: one row, the nuclide's identity and energy grids.
pub fn write_nuclide(data: &IncidentNeutron, dir: &Path) -> Result<(), Box<dyn Error>> {
    write_section(
        &dir.join("nuclide.arrow"),
        "nuclide.arrow",
        vec![
            strings(&[data.name()]),
            ints(&[data.atomic_number as i32]),
            ints(&[data.mass_number as i32]),
            floats(&[data.atomic_weight_ratio.unwrap_or(0.0)]),
            string_list(&data.temperatures()),
            float_list(&data.k_ts),
        ],
    )
}

/// The labels of the grids `energy.arrow` carries, in the order they are
/// written.
///
/// Not the same list as `temperatures`, and deliberately so. The NJOY route
/// carries an extra 0 K grid read off the PENDF tape, so a converted file has
/// one more energy grid than it has temperatures.
///
/// Ordered by the temperatures as processed, NOT as the energy map iterates
/// them. That map is keyed by the temperature's NAME, so a BTreeMap walks it
/// lexicographically and puts 1200K and 2500K before 250K. Every consumer now
/// looks a grid up by its own label rather than zipping two columns
/// positionally, so the order is no longer load-bearing; it is kept because it
/// is the order the published files are in, and because it puts the processed
/// temperatures in one run for the range index to merge.
///
/// Anything the map holds that is not a processed temperature follows, in the
/// map's own order, which is where the 0 K grid ends up. Filtering to the
/// temperature list instead would drop it silently.
fn energy_labels(data: &IncidentNeutron) -> Vec<String> {
    let temperatures = data.temperatures();
    let mut labels: Vec<String> = temperatures
        .iter()
        .filter(|t| data.energy.contains_key(*t))
        .cloned()
        .collect();
    labels.extend(
        data.energy
            .keys()
            .filter(|t| !temperatures.contains(t))
            .cloned(),
    );
    labels
}

/// Write `energy.arrow`, one row and one record batch per temperature.
///
/// One batch each so a client that wants a single temperature range-fetches a
/// single grid. Stored as columns of `nuclide.arrow` they would be all or
/// none: 6.33 MB on U238, against 8.98 MB for the whole single-temperature
/// JSON bundle.
///
/// Returns whether a file was written. Nothing is written when the nuclide has
/// no grid at all, which is what the ENDF route produces: it carries no
/// processed temperature, so there is no union grid to publish and an empty
/// file would only be one the loader cannot read (an Arrow FILE with no record
/// batches is damage, not emptiness). The loader ties the two facts together
/// and requires this section exactly when `temperatures` is non-empty.
pub fn write_energy(data: &IncidentNeutron, dir: &Path) -> Result<bool, Box<dyn Error>> {
    let rows: Vec<Vec<ArrayRef>> = energy_labels(data)
        .into_iter()
        .map(|label| {
            let grid = data.energy.get(&label).cloned().unwrap_or_default();
            vec![strings(&[label]), float_list(&grid)]
        })
        .collect();
    if rows.is_empty() {
        return Ok(false);
    }
    write_section_per_row(&dir.join("energy.arrow"), "energy.arrow", rows)?;
    Ok(true)
}

/// Move the union energy grids of an already-converted folder out of
/// `nuclide.arrow` and into `energy.arrow`.
///
/// For a library published with the grids inside `nuclide.arrow`: the grids are
/// copied, not recomputed, so NJOY does not run again and every value is what
/// it was. A folder already migrated is left alone and reported as such.
///
/// Both files are staged beside their targets and renamed into place, and
/// `energy.arrow` is put down BEFORE `nuclide.arrow` loses its columns, so an
/// interrupted run leaves a folder that still has its grids somewhere. The
/// caller writes the version marker last, which is what makes it the record of
/// a finished migration: a folder whose marker still says 1 is re-migrated
/// correctly on a rerun, since the grids are read from whichever file has them.
pub fn migrate_energy_out_of_nuclide(dir: &Path) -> Result<bool, Box<dyn Error>> {
    let nuclide_path = dir.join("nuclide.arrow");
    if !nuclide_path.exists() {
        return Ok(false);
    }
    let batch = read_one_batch(&nuclide_path)?;
    let Some(labels) = batch.column_by_name("energy_temperatures") else {
        // No such column: already migrated. Nothing here to move.
        return Ok(false);
    };
    let labels = labels
        .as_list_opt::<i32>()
        .ok_or("`energy_temperatures` is not a ListArray")?
        .value(0);
    let labels = labels
        .as_string_opt::<i32>()
        .ok_or("`energy_temperatures` items are not strings")?;
    let grids = batch
        .column_by_name("energy_values")
        .ok_or("nuclide.arrow has `energy_temperatures` and no `energy_values`")?
        .as_list_opt::<i32>()
        .ok_or("`energy_values` is not a ListArray")?
        .value(0);
    let grids = grids
        .as_list_opt::<i32>()
        .ok_or("`energy_values` items are not lists")?;
    if labels.len() != grids.len() {
        return Err(format!(
            "nuclide.arrow lists {} energy temperatures and {} grids; they are zipped \
             positionally, so a migration cannot tell which grid belongs to which label",
            labels.len(),
            grids.len()
        )
        .into());
    }

    let rows: Vec<Vec<ArrayRef>> = (0..labels.len())
        .map(|i| {
            let grid = grids.value(i);
            let grid = grid
                .as_primitive_opt::<arrow_array::types::Float64Type>()
                .ok_or("an energy grid is not Float64")?;
            Ok(vec![
                strings(&[labels.value(i).to_string()]),
                float_list(grid.values()),
            ])
        })
        .collect::<Result<_, Box<dyn Error>>>()?;

    let staged_energy = dir.join("energy.arrow.tmp");
    write_section_per_row(&staged_energy, "energy.arrow", rows)?;
    std::fs::rename(&staged_energy, dir.join("energy.arrow"))?;
    // A runtime cache that asked an origin for `energy.arrow` before the
    // library was republished holds a zero-byte `.absent` marker, which the
    // loader reads as "this nuclide has no such section". The folder now HAS
    // one, so the marker is a lie and goes.
    let _ = std::fs::remove_file(dir.join("energy.arrow.absent"));

    // The metadata columns, carried across by name rather than by position so
    // a file written with a different column order still migrates.
    let staged_nuclide = dir.join("nuclide.arrow.tmp");
    let column = |name: &str| -> Result<ArrayRef, Box<dyn Error>> {
        batch
            .column_by_name(name)
            .cloned()
            .ok_or_else(|| format!("nuclide.arrow has no `{name}`").into())
    };
    write_section(
        &staged_nuclide,
        "nuclide.arrow",
        vec![
            column("name")?,
            column("Z")?,
            column("A")?,
            column("atomic_weight_ratio")?,
            column("temperatures")?,
            column("kTs")?,
        ],
    )?;
    std::fs::rename(&staged_nuclide, &nuclide_path)?;
    Ok(true)
}

/// Read a single-batch section without checking it against this build's
/// declaration.
///
/// The migration's whole job is to read a file whose columns this build no
/// longer declares, so the usual validating reader would refuse it.
fn read_one_batch(path: &Path) -> Result<RecordBatch, Box<dyn Error>> {
    let mut reader = FileReader::try_new(File::open(path)?, None)?;
    reader
        .next()
        .ok_or_else(|| format!("{}: no record batches", path.display()))?
        .map_err(Into::into)
}

/// Write `urr.arrow`, one row per temperature that has tables.
///
/// Written only when the nuclide has unresolved resonance data at all, which
/// most do not: the loader treats an absent file as "no unresolved range"
/// rather than as an error.
pub fn write_urr(data: &IncidentNeutron, dir: &Path) -> Result<bool, Box<dyn Error>> {
    if data.urr.is_empty() {
        return Ok(false);
    }

    let mut temperature = Vec::new();
    let mut energy = Vec::new();
    let mut table_data = Vec::new();
    let mut table_shape = Vec::new();
    let mut interpolation = Vec::new();
    let mut inelastic = Vec::new();
    let mut absorption = Vec::new();
    let mut multiply_smooth = Vec::new();

    // One row per temperature, in the order they were processed. The map is
    // keyed by the temperature's NAME, so iterating it directly puts 1200K and
    // 2500K before 250K: the same rows, permuted, which loads identically and
    // matches no published file.
    let mut ordered: Vec<&String> = data
        .temperatures()
        .iter()
        .filter(|t| data.urr.contains_key(*t))
        .filter_map(|t| data.urr.get_key_value(t).map(|(k, _)| k))
        .collect();
    ordered.extend(data.urr.keys().filter(|t| !data.temperatures().contains(t)));

    for (t, tables) in ordered.into_iter().map(|t| (t, &data.urr[t])) {
        temperature.push(t.clone());
        energy.push(tables.energy.clone());
        table_data.push(tables.table.clone());
        // (n_energy, 6, n_band) in C order, kept flat with the shape beside it
        // rather than reshaped, because that is how the reader wants it.
        table_shape.push(tables.shape.iter().map(|&n| n as i32).collect::<Vec<i32>>());
        interpolation.push(tables.interpolation as i32);
        inelastic.push(tables.inelastic_flag as i32);
        absorption.push(tables.absorption_flag as i32);
        multiply_smooth.push(tables.multiply_smooth);
    }

    write_section(
        &dir.join("urr.arrow"),
        "urr.arrow",
        vec![
            strings(&temperature),
            float_lists(&energy),
            float_lists(&table_data),
            int_lists(&table_shape),
            ints(&interpolation),
            ints(&inelastic),
            ints(&absorption),
            bools(&multiply_smooth),
        ],
    )?;
    Ok(true)
}
