//! The lookup built from the loaded reactions equals the one read from the
//! published `fast_xs.arrow`, on every fixture, at every temperature.
//!
//! This is the correctness statement of dropping the file: the converter wrote
//! `fast_xs.arrow` from the parsed evaluation, `FastXSGrid::build` derives the
//! same table from the reactions the loader read back from `reactions.arrow`,
//! and the two must agree to the bit. The fields the file carried are compared
//! exactly (the four summed columns, the two matrices, the MT lists, the fission
//! flag, the capture column, the log index and its two scalars); the fields the
//! Arrow-fed builder already derived from reactions are compared too, so a
//! change in that half would not hide behind the file's half.
//!
//! Self-skips a fixture that is absent. Deleted once the writer is gone, since
//! there is then nothing to compare against; the builder's own tests stay.

use std::path::PathBuf;

use yamc_nuclide::load_scope::LoadScope;
use yamc_nuclide::nuclide::{load_nuclide, FastXSGrid};

fn fixture(name: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../yamc/tests/{name}.arrow"));
    if dir.join("fast_xs.arrow").is_file() {
        Some(dir)
    } else {
        eprintln!("skip: {name} fixture has no fast_xs.arrow");
        None
    }
}

fn assert_same_f64(what: &str, name: &str, temp: &str, a: &[f64], b: &[f64]) {
    assert_eq!(
        a.len(),
        b.len(),
        "{name} at {temp}: {what} length {} (arrow) vs {} (built)",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.to_bits() == y.to_bits(),
            "{name} at {temp}: {what}[{i}] {x:e} (arrow) vs {y:e} (built), \
             relative difference {:e}",
            (x - y).abs() / x.abs().max(y.abs()).max(f64::MIN_POSITIVE)
        );
    }
}

fn assert_same_grid(name: &str, temp: &str, arrow: &FastXSGrid, built: &FastXSGrid) {
    let flat = |xs: &[[f64; 4]]| -> Vec<f64> { xs.iter().flatten().copied().collect() };
    assert_same_f64("xs", name, temp, &flat(&arrow.xs), &flat(&built.xs));
    assert_eq!(
        arrow.scatter_mt_numbers, built.scatter_mt_numbers,
        "{name} at {temp}: scatter MTs"
    );
    assert_same_f64(
        "scatter_mt_xs",
        name,
        temp,
        arrow.scatter_mt_xs.as_slice(),
        built.scatter_mt_xs.as_slice(),
    );
    assert_eq!(
        arrow.fission_mt_numbers, built.fission_mt_numbers,
        "{name} at {temp}: fission MTs"
    );
    assert_same_f64(
        "fission_mt_xs",
        name,
        temp,
        arrow.fission_mt_xs.as_slice(),
        built.fission_mt_xs.as_slice(),
    );
    assert_eq!(
        arrow.has_partial_fission, built.has_partial_fission,
        "{name} at {temp}: has_partial_fission"
    );
    assert_same_f64(
        "xs_ngamma",
        name,
        temp,
        arrow.xs_ngamma.as_slice(),
        built.xs_ngamma.as_slice(),
    );
    assert_eq!(
        arrow.log_grid_index, built.log_grid_index,
        "{name} at {temp}: log_grid_index"
    );
    assert_eq!(
        arrow.log_e_min.to_bits(),
        built.log_e_min.to_bits(),
        "{name} at {temp}: log_e_min"
    );
    assert_eq!(
        arrow.inv_log_delta.to_bits(),
        built.inv_log_delta.to_bits(),
        "{name} at {temp}: inv_log_delta"
    );
    assert_same_f64(
        "energy",
        name,
        temp,
        arrow.energy.as_slice(),
        built.energy.as_slice(),
    );
    // The half the Arrow-fed builder already derived from reactions.
    assert_eq!(
        arrow.elastic_idx, built.elastic_idx,
        "{name} at {temp}: elastic_idx"
    );
    assert_eq!(
        arrow.inelastic_walk_order, built.inelastic_walk_order,
        "{name} at {temp}: walk order"
    );
    assert_eq!(
        arrow.photon_rxn_mt_numbers, built.photon_rxn_mt_numbers,
        "{name} at {temp}: photon MTs"
    );
    assert_same_f64(
        "photon_rxn_xs",
        name,
        temp,
        arrow.photon_rxn_xs.as_slice(),
        built.photon_rxn_xs.as_slice(),
    );
    assert_same_f64(
        "photon_prod",
        name,
        temp,
        arrow.photon_prod.as_slice(),
        built.photon_prod.as_slice(),
    );
    assert_eq!(
        arrow.absorption_mt_numbers, built.absorption_mt_numbers,
        "{name} at {temp}: absorption MTs"
    );
    assert_same_f64(
        "absorption_mt_xs",
        name,
        temp,
        arrow.absorption_mt_xs.as_slice(),
        built.absorption_mt_xs.as_slice(),
    );
    assert_same_f64(
        "delayed_photon_scaling",
        name,
        temp,
        arrow.delayed_photon_scaling.as_slice(),
        built.delayed_photon_scaling.as_slice(),
    );
    assert_eq!(
        arrow.scatter_mt_reactions.len(),
        built.scatter_mt_reactions.len(),
        "{name} at {temp}: scatter reaction pointers"
    );
    assert_eq!(
        arrow.reaction_absorption.is_some(),
        built.reaction_absorption.is_some(),
        "{name} at {temp}: MT 101 pointer"
    );
}

#[test]
fn the_built_lookup_equals_the_published_one_on_every_fixture() {
    // Light, structural, heavy, fissile with and without chance partials, and
    // the URR nuclide. Th232 and U240 are the fissionables; U240 is the only
    // ENDF/B-VIII.1 nuclide with non-redundant partial fission channels.
    let names = [
        "Li6", "Li7", "Be9", "B10", "C12", "O16", "Al27", "Si28", "Cr52", "Fe54", "Fe56", "Fe57",
        "Fe58", "Co58", "W184", "Pb208", "Th232", "U240",
    ];
    let mut checked = 0;
    for name in names {
        let Some(dir) = fixture(name) else { continue };
        let nuclide =
            load_nuclide(&dir, &LoadScope::full()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            nuclide.fast_xs.len(),
            nuclide.loaded_temperatures.len(),
            "{name}: one grid per loaded temperature"
        );
        for (idx, temp) in nuclide.loaded_temperatures.iter().enumerate() {
            let arrow = &nuclide.fast_xs[idx];
            let grid = nuclide
                .energy_grid(temp)
                .unwrap_or_else(|| panic!("{name}: no grid at {temp}"));
            let built = FastXSGrid::build(
                grid,
                &nuclide.reactions[idx],
                nuclide.fission_photon_release.as_ref(),
                &format!("{name} at {temp} K"),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            assert_same_grid(name, temp, arrow, &built);
            checked += 1;
        }
        eprintln!(
            "{name}: {} temperature(s) identical",
            nuclide.loaded_temperatures.len()
        );
    }
    assert!(
        checked > 0,
        "no fixture with fast_xs.arrow found; nothing was compared"
    );
}
