//! Issue #481: a material could only ever be queried at the one temperature its
//! nuclear data was loaded at.
//!
//! `read_nuclear_data` narrows the load scope to `self.temperature`, so a
//! material built at 294 K holds only 294 K reactions even though its Arrow
//! files carry every temperature. `available_temperatures` then advertised a
//! temperature that `loaded_temperatures` could not serve, and the request died
//! downstream as `No cross section data found for MT 1` rather than as anything
//! resembling a temperature problem.
//!
//! These need a multi-temperature nuclide, and no such fixture is committed, so
//! they self-skip. The parts that need no data are unit tests in `material.rs`.

use std::collections::HashMap;
use yamc_materials::Material;

/// A cached H1 with the full ENDF/B-VIII.1 temperature ladder, or `None`.
fn h1_cache() -> Option<String> {
    yamc_test_cache::nuclide("H1")
}

fn h1_at(temperature: &str, cache: &str) -> Material {
    let mut m = Material::new(
        HashMap::from([("H1".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("H1 material");
    m.density = Some(1.0);
    m.set_temperature(temperature);
    m.read_nuclear_data(
        &HashMap::from([("H1".to_string(), cache.to_string())]),
        None,
    )
    .expect("read H1");
    m
}

/// Every temperature the data advertises must be servable, not just the loaded one.
#[test]
fn every_advertised_temperature_can_be_queried() {
    let Some(cache) = h1_cache() else {
        eprintln!("every_advertised_temperature_can_be_queried: skip (data missing)");
        return;
    };
    let mut m = h1_at("294", &cache);

    // Every temperature the file advertises, not a hardcoded list, so the test
    // cannot quietly stop covering one.
    let mut temperatures = m.nuclide_data["H1"].available_temperatures.clone();
    temperatures.retain(|t| t != "0"); // the 0 K grid carries no reaction set
    temperatures.sort_by_key(|t| t.parse::<i64>().unwrap_or(i64::MAX));
    assert!(
        temperatures.len() >= 3,
        "expected a multi-temperature fixture, got {temperatures:?}"
    );

    let mut seen: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
    for temperature in &temperatures {
        let (xs, grid) = m.macroscopic_cross_section(1, Some(temperature));
        assert!(!xs.is_empty(), "MT 1 missing at {temperature} K");
        assert_eq!(
            xs.len(),
            grid.len(),
            "at {temperature} K the cross section and its grid disagree in length, \
             which is what a stale microscopic cache used to produce"
        );
        seen.push((temperature.clone(), xs, grid));
    }

    // Each temperature builds its own unified grid, so comparing values by index
    // is only meaningful once the grids are known to match. Assert that rather
    // than assume it: on differing grids, index 100 would be a different energy
    // at each temperature and the comparison below would be meaningless.
    let (_, _, reference_grid) = &seen[0];
    for (temperature, _, grid) in &seen {
        assert_eq!(
            grid, reference_grid,
            "the unified grid at {temperature} K differs from the first, so \
             comparing cross sections by index would compare different energies"
        );
    }

    // Doppler broadening is monotone in temperature for H1 capture at this
    // energy, so identical values would mean the temperature never reached the
    // data even though the call succeeded.
    for pair in seen.windows(2) {
        let (lo_t, lo_xs, _) = &pair[0];
        let (hi_t, hi_xs, _) = &pair[1];
        assert!(
            hi_xs[100] > lo_xs[100],
            "cross section did not rise from {lo_t} K ({}) to {hi_t} K ({})",
            lo_xs[100],
            hi_xs[100]
        );
    }
}

/// The on-disk spelling and the bare number must name one temperature.
#[test]
fn the_k_suffix_is_the_same_temperature() {
    let Some(cache) = h1_cache() else {
        eprintln!("the_k_suffix_is_the_same_temperature: skip (data missing)");
        return;
    };
    let mut m = h1_at("294", &cache);

    let (bare, _) = m.macroscopic_cross_section(1, Some("900"));
    let (suffixed, _) = m.macroscopic_cross_section(1, Some("900K"));
    assert_eq!(
        bare, suffixed,
        "'900' and '900K' are the same temperature; the Arrow files use the \
         suffixed spelling and nuclide data is keyed by the bare one"
    );
}

/// Querying another temperature must not change what the material itself reports.
///
/// This is the #481 coherence regression, and it could not be written before the
/// widening landed: the 900 K query panicked long before it could poison
/// anything.
#[test]
fn a_query_at_another_temperature_does_not_poison_the_material() {
    let Some(cache) = h1_cache() else {
        eprintln!(
            "a_query_at_another_temperature_does_not_poison_the_material: skip (data missing)"
        );
        return;
    };

    let expected = {
        let mut clean = h1_at("294", &cache);
        clean.macroscopic_cross_section(1, None).0
    };

    let mut m = h1_at("294", &cache);
    let (hot, _) = m.macroscopic_cross_section(1, Some("900"));
    let (own, _) = m.macroscopic_cross_section(1, None);

    assert_ne!(hot, expected, "the 900 K query returned the 294 K numbers");
    assert_eq!(
        own, expected,
        "after a 900 K query the material returned something other than its own \
         294 K cross section"
    );
    assert_eq!(m.temperature(), "294", "and its label must be unchanged");
}

/// The gap that let three defects through review: the data-free unit test for
/// this invariant never runs the widening, because a material with no nuclides
/// has nothing stale to widen. With real data loaded, the widening runs, and it
/// used to invalidate every cache on the way past.
#[test]
fn a_foreign_temperature_query_leaves_a_loaded_material_alone() {
    let Some(cache) = h1_cache() else {
        eprintln!(
            "a_foreign_temperature_query_leaves_a_loaded_material_alone: skip (data missing)"
        );
        return;
    };
    let mut m = h1_at("294", &cache);
    m.calculate_macroscopic_xs(&vec![1], true);

    // Precondition: warm at its own temperature, and narrow, so the query below
    // really does trigger a widen.
    assert!(!m.unified_energy_grid_neutron.is_empty(), "grid warm");
    assert!(m.fast_xs.is_some(), "fast_xs warm");
    assert!(m.cached_microscopic_xs.is_some(), "micro cache warm");
    assert!(
        m.macroscopic_xs_neutron_total_by_nuclide.is_some(),
        "per-nuclide totals warm"
    );
    assert_eq!(
        m.nuclide_data["H1"].loaded_temperatures,
        vec!["294".to_string()],
        "narrow, so the 900 query must widen"
    );
    let grid_before = m.unified_energy_grid_neutron.clone();

    let _ = m.macroscopic_cross_section(1, Some("900"));

    assert_eq!(
        m.unified_energy_grid_neutron, grid_before,
        "the material's own grid must survive a query at another temperature"
    );
    assert!(m.fast_xs.is_some(), "fast_xs must survive");
    assert!(
        m.cached_microscopic_xs.is_some(),
        "micro cache must survive"
    );
    assert!(
        m.macroscopic_xs_neutron_total_by_nuclide.is_some(),
        "per-nuclide totals must survive: sample_interacting_nuclide expects them"
    );
    assert_eq!(m.temperature(), "294", "and the label must be unchanged");
}

/// An unlabelled material queried at an explicit temperature must still end up
/// with the tables the transport hot path expects.
#[test]
fn an_unlabelled_material_still_caches_what_transport_needs() {
    let Some(cache) = h1_cache() else {
        eprintln!("an_unlabelled_material_still_caches_what_transport_needs: skip (data missing)");
        return;
    };
    let mut m = Material::new(
        HashMap::from([("H1".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("H1 material");
    m.density = Some(1.0);
    m.read_nuclear_data(&HashMap::from([("H1".to_string(), cache)]), None)
        .expect("read H1");
    assert_eq!(m.temperature(), "", "precondition: no label");

    let (xs, _) = m.macroscopic_cross_section(1, Some("294"));
    assert!(!xs.is_empty());

    assert!(
        m.macroscopic_xs_neutron_total_by_nuclide.is_some(),
        "sample_interacting_nuclide unwraps this and would panic"
    );
    assert!(m.fast_xs.is_some(), "the transport hot path needs fast_xs");
    assert_eq!(
        m.temperature(),
        "294",
        "the queried temperature is adopted, so the label and the caches agree"
    );
}

/// Photon state must survive a temperature change: it does not depend on
/// temperature, and `cached_elements` / `cached_element_atom_densities` are
/// indexed against each other with only a `debug_assert_eq!` between them.
#[test]
fn set_temperature_leaves_photon_caches_consistent() {
    let mut m = Material::new(
        HashMap::from([("H1".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("H1 material");
    // Fabricate the post-`init_photon_data` shape: the two vectors parallel.
    m.cached_element_atom_densities = vec![1.0];
    m.element_indices = vec![("H1".to_string(), 0)];

    m.set_temperature("900");

    assert_eq!(
        m.cached_element_atom_densities.len(),
        1,
        "photon atom densities do not depend on temperature, and only \
         calculate_macroscopic_xs rebuilds them"
    );
    assert_eq!(
        m.element_indices.len(),
        1,
        "element_indices is rebuilt only by init_photon_data, so clearing it \
         here leaves calculate_photon_xs indexing past the end in release"
    );
}
