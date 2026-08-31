//! Issue #476, task 1: the loader shares its large numeric buffers instead of
//! copying them into a fresh `Vec` per consumer.
//!
//! Sharing is not observable through the data, which is the point: every field
//! still derefs to the same `&[f64]` it did before. What these tests assert is
//! the allocation identity behind it, via [`F64Buffer::shares_with`], plus the
//! invariant that makes sharing safe to do at all: the values are unchanged.
//!
//! Uses the shared Fe56 / U240 Arrow fixtures; self-skips when absent so the
//! suite is inert without the data.

use std::collections::HashSet;

use yamc_nuclide::nuclide::{load_nuclide, Nuclide};
use yamc_nuclide::LoadScope;

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../yamc/tests/{name}.arrow"))
}

fn load(name: &str, scope: &LoadScope) -> Option<Nuclide> {
    load_nuclide(fixture(name), scope).ok()
}

fn temps(values: &[&str]) -> HashSet<String> {
    values.iter().map(|s| s.to_string()).collect()
}

/// Every reaction's energy grid is a view of the nuclide's grid for its
/// temperature, not a copy of it.
///
/// This is the largest single saving in the change: on the Fe56 fixture the
/// loop that used to copy the grid ran 540 times over ~42k points apiece.
#[test]
fn reaction_grids_are_views_of_the_nuclide_grid() {
    let Some(nuclide) = load("Fe56", &LoadScope::full()) else {
        eprintln!("Skipping: Fe56.arrow fixture not found");
        return;
    };
    let energy_map = nuclide.energy.as_ref().expect("energy grids");

    let mut checked = 0;
    for (temp_idx, temp_key) in nuclide.loaded_temperatures.iter().enumerate() {
        let grid = energy_map
            .get(temp_key)
            .expect("grid for loaded temperature");
        for reaction in nuclide.reactions[temp_idx].values() {
            if reaction.energy.is_empty() {
                continue;
            }
            assert!(
                reaction.energy.shares_with(grid),
                "MT {} at {temp_key} K copied the energy grid instead of viewing it",
                reaction.mt_number
            );
            // A view of the grid's tail, so it still lines up with the cross
            // section it indexes.
            assert_eq!(
                reaction.energy.as_slice(),
                &grid[grid.len() - reaction.energy.len()..],
                "MT {} at {temp_key} K views the wrong span of the grid",
                reaction.mt_number
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "fixture carried no reactions to check");
}

/// The `fast_xs` accelerator points at the nuclide's grid rather than keeping
/// the second copy `fast_xs.arrow` ships.
#[test]
fn fast_xs_grids_share_the_nuclide_grid() {
    let Some(nuclide) = load("Fe56", &LoadScope::full()) else {
        eprintln!("Skipping: Fe56.arrow fixture not found");
        return;
    };
    let energy_map = nuclide.energy.as_ref().expect("energy grids");
    assert!(!nuclide.fast_xs.is_empty(), "fixture carried no fast_xs");

    for (temp_idx, temp_key) in nuclide.loaded_temperatures.iter().enumerate() {
        let grid = energy_map
            .get(temp_key)
            .expect("grid for loaded temperature");
        let fast = &nuclide.fast_xs[temp_idx];
        assert_eq!(
            fast.energy.as_slice(),
            grid.as_slice(),
            "fast_xs energy at {temp_key} K disagrees with the nuclide grid"
        );
        assert!(
            fast.energy.shares_with(grid),
            "fast_xs at {temp_key} K kept its own copy of the grid"
        );
    }
}

/// A filtered load copies rather than shares, so it does not pin the rows it
/// exists to drop.
///
/// Asserted through the grid the reaction views: under a single-temperature
/// scope the nuclide's own grid is a private allocation, so the reaction views
/// something whose parent is exactly one grid wide rather than the whole
/// `energy_values` column.
#[test]
fn a_filtered_load_does_not_pin_the_dropped_temperatures() {
    let full = LoadScope::full();
    let Some(unfiltered) = load("Fe56", &full) else {
        eprintln!("Skipping: Fe56.arrow fixture not found");
        return;
    };
    if unfiltered.available_temperatures.len() < 2 {
        eprintln!("Skipping: fixture carries a single temperature");
        return;
    }
    let one = unfiltered.available_temperatures[0].clone();
    let filtered = load(
        "Fe56",
        &full.clone().with_temperatures(Some(temps(&[&one]))),
    )
    .expect("single-temperature load");

    let grid = filtered
        .energy
        .as_ref()
        .expect("energy grids")
        .get(&one)
        .expect("the requested temperature");
    let full_grid = unfiltered
        .energy
        .as_ref()
        .expect("energy grids")
        .get(&one)
        .expect("the same temperature");

    // The unfiltered load shares, so its view keeps the whole `energy_values`
    // column alive -- every temperature's grid, not just this one.
    let width = std::mem::size_of::<f64>() * grid.len();
    if full_grid.allocation_bytes() <= width {
        eprintln!("Skipping: fixture ships a single energy grid, nothing to pin");
        return;
    }
    assert!(
        grid.allocation_bytes() < full_grid.allocation_bytes(),
        "a single-temperature load pinned as much as an unfiltered one \
         ({} vs {} bytes for a {width}-byte grid)",
        grid.allocation_bytes(),
        full_grid.allocation_bytes(),
    );

    // Filtering changed the ownership, not the numbers.
    assert_eq!(grid.as_slice(), full_grid.as_slice());
}

/// Cross sections survive the round trip byte for byte, whether they were
/// shared out of the batch or copied out of it.
#[test]
fn shared_and_copied_cross_sections_agree() {
    let full = LoadScope::full();
    let Some(unfiltered) = load("U240", &full) else {
        eprintln!("Skipping: U240.arrow fixture not found");
        return;
    };
    let temp = unfiltered.loaded_temperatures[0].clone();
    let filtered = load(
        "U240",
        &full.clone().with_temperatures(Some(temps(&[&temp]))),
    )
    .expect("single-temperature load");

    let shared = &unfiltered.reactions[unfiltered
        .loaded_temperatures
        .iter()
        .position(|t| *t == temp)
        .expect("temperature index")];
    let copied = &filtered.reactions[0];

    assert!(!copied.is_empty(), "fixture carried no reactions");
    for (mt, reaction) in copied {
        let counterpart = shared.get(mt).expect("MT present in the unfiltered load");
        assert_eq!(
            reaction.cross_section.as_slice(),
            counterpart.cross_section.as_slice(),
            "MT {mt} cross section differs between a shared and a copied read"
        );
        assert_eq!(
            reaction.energy.as_slice(),
            counterpart.energy.as_slice(),
            "MT {mt} energy grid differs between a shared and a copied read"
        );
        assert_eq!(reaction.threshold_idx, counterpart.threshold_idx);
    }
}
