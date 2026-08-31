//! Issue #389: a transmutation-scoped load must be a strict subset of a full
//! load, never a different one.
//!
//! `Material.transmute()` reads only the union energy grid and the cross
//! sections of the MTs its chain names, skipping the products, the secondary
//! distributions and the `fast_xs` accelerator. That is worth roughly 5x in
//! memory and load time, but only if the sigma(E) it does read is the same
//! sigma(E) the full load would have produced. These tests are that guard.
//!
//! Uses the shared Fe56 Arrow fixture; self-skips when absent so the suite is
//! inert without the data.

use std::collections::HashSet;
use yamc_nuclide::nuclide::{load_nuclide, Nuclide};
use yamc_nuclide::LoadScope;

/// MTs a typical activation chain asks Fe56 for: radiative capture and the
/// threshold channels that make the FNS decay-heat carriers.
const ACTIVATION_MTS: &[i32] = &[16, 102, 103, 107];

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc/tests/Fe56.arrow")
}

fn load(scope: &LoadScope) -> Option<Nuclide> {
    load_nuclide(fixture(), scope).ok()
}

fn activation_scope() -> LoadScope {
    LoadScope::activation(ACTIVATION_MTS.iter().copied().collect())
}

#[test]
fn xs_only_cross_sections_are_bit_identical_to_a_full_load() {
    let (Some(full), Some(narrow)) = (load(&LoadScope::full()), load(&activation_scope())) else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };

    assert_eq!(
        full.loaded_temperatures, narrow.loaded_temperatures,
        "narrowing the sections must not change which temperatures load"
    );

    let temp = full
        .loaded_temperatures
        .first()
        .expect("Fe56 has a loaded temperature")
        .clone();
    let t_full = full.get_temp_idx(&temp).expect("full temp idx");
    let t_narrow = narrow.get_temp_idx(&temp).expect("narrow temp idx");

    let mut compared = 0;
    for &mt in ACTIVATION_MTS {
        let Some(rf) = full.reactions[t_full].get(&mt) else {
            continue;
        };
        let rn = narrow.reactions[t_narrow]
            .get(&mt)
            .unwrap_or_else(|| panic!("MT {mt} present in the full load but missing in XsOnly"));

        assert_eq!(rf.energy, rn.energy, "MT {mt}: energy grid differs");
        assert_eq!(
            rf.cross_section, rn.cross_section,
            "MT {mt}: cross section differs"
        );
        assert_eq!(rf.threshold_idx, rn.threshold_idx, "MT {mt}: threshold");
        assert_eq!(rf.q_value, rn.q_value, "MT {mt}: Q value");

        // Interpolated lookups are what the rate collapse actually calls.
        for &e in &[1.0e-2_f64, 1.0e3, 1.0e6, 1.4e7] {
            assert_eq!(
                rf.cross_section_at(e),
                rn.cross_section_at(e),
                "MT {mt}: cross_section_at({e}) differs"
            );
        }
        compared += 1;
    }
    assert!(
        compared >= 2,
        "expected Fe56 to carry several activation MTs, compared only {compared}"
    );
}

#[test]
fn xs_only_skips_the_transport_sections_and_the_unwanted_mts() {
    let Some(narrow) = load(&activation_scope()) else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };

    let wanted: HashSet<i32> = ACTIVATION_MTS.iter().copied().collect();
    for reactions in &narrow.reactions {
        for (&mt, rxn) in reactions {
            assert!(
                wanted.contains(&mt),
                "MT {mt} materialized outside the scope"
            );
            // Products describe what comes OUT of a reaction, which only
            // transport samples.
            assert!(rxn.products.is_empty(), "MT {mt} carried products");
        }
    }

    // The single largest section on disk, and pure transport machinery. Left
    // empty rather than defaulted so nothing can read zeros out of it.
    assert!(
        narrow.fast_xs.is_empty(),
        "XsOnly must not build the fast_xs accelerator"
    );
    assert!(!narrow.urr_present, "XsOnly must not read URR tables");
    assert!(narrow.fission_nu.is_none(), "XsOnly must not read nu-bar");
}

#[test]
fn a_full_load_still_carries_everything() {
    let Some(full) = load(&LoadScope::full()) else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };
    assert_eq!(full.fast_xs.len(), full.loaded_temperatures.len());
    assert!(
        full.reactions[0].len() > ACTIVATION_MTS.len(),
        "a full load should carry far more MTs than an activation scope"
    );
    assert!(
        full.reactions[0]
            .get(&2)
            .is_some_and(|r| !r.products.is_empty()),
        "elastic should carry its products under a full load"
    );
}

#[test]
fn the_recorded_scope_reflects_how_the_nuclide_was_loaded() {
    let (Some(full), Some(narrow)) = (load(&LoadScope::full()), load(&activation_scope())) else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };
    // This is what stops the cache handing a transmutation load to transport.
    assert!(full.load_scope.covers(&activation_scope()));
    assert!(!narrow.load_scope.covers(&LoadScope::full()));
}
