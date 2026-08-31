//! `nuclide.arrow` and `urr.arrow`: the nuclide's identity and its unresolved
//! resonance probability tables.
//!
//! The two most direct sections. Almost every value comes straight off the
//! parsed table, and the URR data is already stored flat with an explicit
//! shape because that is the form the format wants.

use std::error::Error;
use std::path::Path;

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
    let temperatures = data.temperatures();

    // Not the same list as `temperatures`, and deliberately so. The NJOY route
    // carries an extra 0 K grid read off the PENDF tape, so a converted file
    // has one more energy grid than it has temperatures. The reader zips these
    // two positionally, so they must be written from the same iteration.
    // Ordered by the temperatures as processed, NOT as the energy map iterates
    // them. That map is keyed by the temperature's NAME, so a BTreeMap walks
    // it lexicographically and puts 1200K and 2500K before 250K. Consumers
    // zip this against `energy_values` positionally, so either order loads the
    // same, but only this one matches the published files.
    //
    // Anything the map holds that is not a processed temperature follows, in
    // the map's own order, which is where the 0 K grid ends up. Filtering to
    // the temperature list instead would drop it silently.
    let mut energy_temperatures: Vec<String> = temperatures
        .iter()
        .filter(|t| data.energy.contains_key(*t))
        .cloned()
        .collect();
    energy_temperatures.extend(
        data.energy
            .keys()
            .filter(|t| !temperatures.contains(t))
            .cloned(),
    );
    let energy_values: Vec<Vec<f64>> = energy_temperatures
        .iter()
        .map(|t| data.energy.get(t).cloned().unwrap_or_default())
        .collect();

    write_section(
        &dir.join("nuclide.arrow"),
        "nuclide.arrow",
        vec![
            strings(&[data.name()]),
            ints(&[data.atomic_number as i32]),
            ints(&[data.mass_number as i32]),
            floats(&[data.atomic_weight_ratio.unwrap_or(0.0)]),
            string_list(&temperatures),
            float_list(&data.k_ts),
            string_list(&energy_temperatures),
            float_list_list(&energy_values),
        ],
    )
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
