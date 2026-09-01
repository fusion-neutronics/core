//! Building a temperature the library does not carry, checked without one.
//!
//! Every test here constructs its input in memory. That is not a shortcut, it
//! is the only honest option: nothing in `crates/endf/fixtures` is a
//! multi-temperature Arrow directory, the published fixtures the rest of the
//! suite uses are fetched from a cache no clean checkout has, and a test that
//! reaches for that cache self-skips and reports green. The blend is
//! arithmetic over two `FastXSGrid` values, so it can be checked exactly, on
//! numbers chosen to make a mistake visible.
//!
//! What this cannot check is the physics question: whether a linear blend of
//! 294 K and 600 K is close to the same nuclide broadened at 450 K by NJOY.
//! That needs a raw ENDF tree and an njoy binary no job provides, so it is
//! deliberately not attempted here rather than attempted and skipped. See the
//! pull request for what was measured offline.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_nuclide::blend::{blend_fast_xs, blend_reactions, build_log_grid_index, union_grid};
use yamc_nuclide::buffer::F64Buffer;
use yamc_nuclide::nuclide::{FastXSGrid, Nuclide};
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::temperature;
use yamc_nuclide::urr::UrrData;

/// A cross section that is linear in energy and in temperature, so the blend
/// has a closed form at every point of the union grid, including the points
/// only one source carries.
fn linear_xs(e: f64, t: f64) -> f64 {
    1.0 + 3.0 * e + 0.5 * t + 0.002 * e * t
}

/// A grid carrying one scattering MT and one fission MT, on its own energies.
fn grid_at(t: f64, energies: &[f64]) -> FastXSGrid {
    let (log_e_min, inv_log_delta, log_grid_index) = build_log_grid_index(energies);
    let xs: Vec<[f64; 4]> = energies
        .iter()
        .map(|&e| {
            [
                linear_xs(e, t),
                linear_xs(e, t) * 0.25,
                linear_xs(e, t) * 0.5,
                linear_xs(e, t) * 0.125,
            ]
        })
        .collect();
    // Row-major [n_energies, n_mts], two scattering columns with DIFFERENT
    // values so a transposed ravel or a swapped column is visible.
    let mut scatter = Vec::with_capacity(energies.len() * 2);
    for &e in energies {
        scatter.push(linear_xs(e, t));
        scatter.push(linear_xs(e, t) * 10.0);
    }
    FastXSGrid {
        log_grid_index,
        log_e_min,
        inv_log_delta,
        xs,
        energy: F64Buffer::from_slice(energies),
        scatter_mt_numbers: vec![2, 51],
        scatter_mt_xs: F64Buffer::from_slice(&scatter),
        elastic_idx: Some(0),
        inelastic_walk_order: FastXSGrid::build_inelastic_walk_order(&[2, 51], Some(0)),
        xs_ngamma: F64Buffer::from_slice(
            &energies
                .iter()
                .map(|&e| linear_xs(e, t) * 0.05)
                .collect::<Vec<_>>(),
        ),
        ..Default::default()
    }
}

/// A reaction on `energies`, with a threshold at `threshold_idx`.
fn reaction_at(mt: i32, energies: &[f64], threshold_idx: usize, t: f64) -> Arc<Reaction> {
    let grid = F64Buffer::from_slice(energies);
    let values: Vec<f64> = energies[threshold_idx..]
        .iter()
        .map(|&e| linear_xs(e, t))
        .collect();
    Arc::new(Reaction {
        cross_section: F64Buffer::from_slice(&values),
        threshold_idx,
        energy: grid.tail(threshold_idx),
        mt_number: mt,
        q_value: -1.5e6,
        products: Vec::new(),
        scatter_in_cm: true,
        redundant: false,
    })
}

/// A two-temperature nuclide with everything the synthesiser touches.
fn two_temperature_nuclide() -> Nuclide {
    let lo_energies = vec![1.0, 10.0, 100.0, 1000.0];
    let hi_energies = vec![1.0, 20.0, 100.0, 500.0, 1000.0];

    let mut energy = HashMap::new();
    energy.insert("294".to_string(), F64Buffer::from_slice(&lo_energies));
    energy.insert("600".to_string(), F64Buffer::from_slice(&hi_energies));

    let mut lo_reactions = HashMap::new();
    lo_reactions.insert(2, reaction_at(2, &lo_energies, 0, 294.0));
    lo_reactions.insert(51, reaction_at(51, &lo_energies, 2, 294.0));
    let mut hi_reactions = HashMap::new();
    hi_reactions.insert(2, reaction_at(2, &hi_energies, 0, 600.0));
    hi_reactions.insert(51, reaction_at(51, &hi_energies, 2, 600.0));

    Nuclide {
        name: Some("Xx1".to_string()),
        element: None,
        atomic_symbol: Some("Xx".to_string()),
        atomic_number: Some(1),
        neutron_number: Some(0),
        mass_number: Some(1),
        atomic_weight_ratio: Some(1.0),
        library: None,
        energy: Some(energy),
        reactions: vec![lo_reactions, hi_reactions],
        fissionable: false,
        available_temperatures: vec!["294".to_string(), "600".to_string()],
        loaded_temperatures: vec!["294".to_string(), "600".to_string()],
        data_path: None,
        fission_nu: None,
        fast_xs: vec![grid_at(294.0, &lo_energies), grid_at(600.0, &hi_energies)],
        urr_data: vec![Some(urr_marked(1.0)), Some(urr_marked(2.0))],
        urr_present: true,
        fission_photon_release: None,
        covariance: None,
        elastic_flat_cache: Default::default(),
        fission_chi_flat_cache: Default::default(),
        delayed_neutron_cache: Default::default(),
        inelastic_angle_flat_cache: Default::default(),
        load_scope: Default::default(),
    }
}

/// A URR table whose energy grid identifies which temperature it came from.
fn urr_marked(marker: f64) -> UrrData {
    UrrData {
        energy: vec![marker, marker * 10.0],
        ..Default::default()
    }
}

/// A failure means the blend is defined on a grid that is not the union of its
/// two sources, so a point one temperature resolves and the other does not
/// would be dropped or duplicated.
#[test]
fn the_union_grid_keeps_every_point_and_both_endpoints_exactly() {
    let a = [1.0, 10.0, 100.0, 1000.0];
    let b = [1.0, 20.0, 100.0, 500.0, 1000.0];
    assert_eq!(
        union_grid(&a, &b),
        vec![1.0, 10.0, 20.0, 100.0, 500.0, 1000.0]
    );

    // Endpoints bit-identical to the merged extremes, because log_e_min and the
    // top of the lookup index are derived from them.
    let u = union_grid(&a, &b);
    assert_eq!(u[0], 1.0);
    assert_eq!(u[u.len() - 1], 1000.0);

    // Two points a relative 1e-13 apart are one point: that is the same energy
    // written by two NJOY runs, not two energies.
    let near = [1.0, 1.0 + 1e-13, 2.0];
    assert_eq!(union_grid(&near, &[]), vec![1.0, 2.0]);

    // Two points a relative 1e-7 apart are two points. The closest genuine
    // neighbours in a published library are about that far apart, so collapsing
    // them would thin a real grid.
    let distinct = [1.0, 1.0 + 1e-7, 2.0];
    assert_eq!(union_grid(&distinct, &[]).len(), 3);
}

/// A failure means the blend is not the weighted average it claims to be at the
/// points that need re-interpolation, which is every point one source carries
/// and the other does not.
#[test]
fn a_cross_section_linear_in_energy_and_temperature_blends_exactly() {
    let lo_energies = [1.0, 10.0, 100.0, 1000.0];
    let hi_energies = [1.0, 20.0, 100.0, 500.0, 1000.0];
    let lo = grid_at(294.0, &lo_energies);
    let hi = grid_at(600.0, &hi_energies);
    let w = temperature::blend_weight(294.0, 600.0, 450.0);

    let blended = blend_fast_xs(&lo, &hi, w).expect("two full grids blend");
    let union = union_grid(&lo_energies, &hi_energies);
    assert_eq!(blended.energy.as_slice(), union.as_slice());

    for (i, &e) in union.iter().enumerate() {
        // Linear in both variables, so the blend of the two endpoints IS the
        // value at the intermediate temperature, at every union point.
        let want = linear_xs(e, 450.0);
        assert!(
            (blended.xs[i][0] - want).abs() <= 1e-9 * want.abs(),
            "total at E={e}: got {}, want {want}",
            blended.xs[i][0]
        );
        // Both scattering columns, so a swap between them shows: the second is
        // ten times the first at every point.
        let n = blended.scatter_mt_numbers.len();
        let got_2 = blended.scatter_mt_xs.as_slice()[i * n];
        let got_51 = blended.scatter_mt_xs.as_slice()[i * n + 1];
        assert!((got_2 - want).abs() <= 1e-9 * want.abs(), "MT 2 at E={e}");
        assert!(
            (got_51 - want * 10.0).abs() <= 1e-9 * (want * 10.0).abs(),
            "MT 51 at E={e}: got {got_51}, want {}",
            want * 10.0
        );
    }
}

/// A failure means a blend at an endpoint is not the endpoint, so taking the
/// synthesised path unconditionally would perturb a temperature the data
/// already carries.
#[test]
fn a_blend_at_weight_zero_or_one_reproduces_its_own_source() {
    let lo_energies = [1.0, 10.0, 100.0, 1000.0];
    let hi_energies = [1.0, 20.0, 100.0, 500.0, 1000.0];
    let lo = grid_at(294.0, &lo_energies);
    let hi = grid_at(600.0, &hi_energies);

    let at_lo = blend_fast_xs(&lo, &hi, 0.0).expect("blends");
    for (i, &e) in lo_energies.iter().enumerate() {
        let j = at_lo
            .energy
            .as_slice()
            .iter()
            .position(|&x| x == e)
            .expect("the union contains every source point");
        assert_eq!(at_lo.xs[j][0], lo.xs[i][0], "weight 0 at E={e}");
    }

    let at_hi = blend_fast_xs(&lo, &hi, 1.0).expect("blends");
    for (i, &e) in hi_energies.iter().enumerate() {
        let j = at_hi
            .energy
            .as_slice()
            .iter()
            .position(|&x| x == e)
            .expect("the union contains every source point");
        assert_eq!(at_hi.xs[j][0], hi.xs[i][0], "weight 1 at E={e}");
    }
}

/// A failure means the lookup accelerator built for the union grid does not
/// satisfy what `FastXSGrid::lookup` assumes of it, so a search would start
/// past the answer and silently return the wrong cross section.
#[test]
fn the_rebuilt_lookup_index_brackets_every_energy_it_indexes() {
    let lo = grid_at(294.0, &[1.0, 10.0, 100.0, 1000.0]);
    let hi = grid_at(600.0, &[1.0, 20.0, 100.0, 500.0, 1000.0]);
    let blended = blend_fast_xs(&lo, &hi, 0.5).expect("blends");

    let index = &blended.log_grid_index;
    let n = blended.energy.len();
    assert!(index.len() >= 2);
    assert!(
        index.windows(2).all(|w| w[0] <= w[1]),
        "index must not go backwards"
    );
    assert!(
        index.iter().all(|&i| (i as usize) < n),
        "index must stay in the grid"
    );
    // The top entry is the top of the grid, set rather than derived, because
    // exp(ln(e_max)) need not return e_max and the reader uses this entry as
    // the upper bracket of a search.
    assert_eq!(*index.last().unwrap() as usize, n - 1);

    // The property a uniformly-too-high table violates: for every real grid
    // point, the bin the index sends a search to must not already be past it.
    for (k, &e) in blended.energy.as_slice().iter().enumerate() {
        let bin =
            (((e.ln() - blended.log_e_min) * blended.inv_log_delta) as usize).min(index.len() - 1);
        assert!(
            (index[bin] as usize) <= k,
            "grid point {k} at E={e} falls in bin {bin}, whose start index {} is already past it",
            index[bin]
        );
    }
}

/// A failure means the blend would invent a cross section for a channel one of
/// its two sources does not have. Zero-filling the absent side scales that
/// channel by the blend weight at every energy, not just near a threshold, and
/// the result loads and samples without complaint.
#[test]
fn a_channel_present_at_one_temperature_and_absent_at_the_other_is_refused() {
    let energies = [1.0, 10.0, 100.0];
    let lo = grid_at(294.0, &energies);
    let mut hi = grid_at(600.0, &energies);
    hi.scatter_mt_numbers = vec![2];

    let err = blend_fast_xs(&lo, &hi, 0.5).expect_err("differing channels must not blend");
    let message = err.to_string();
    assert!(
        message.contains("51"),
        "{message} does not name the missing MT"
    );
    assert!(
        message.contains("scattering"),
        "{message} does not name the group"
    );
}

/// A failure means an intermediate temperature is half-built: one of the five
/// parallel structures the nuclide indexes by temperature position would be out
/// of step with the others, and `get_temp_idx` would return an index that is
/// right for one and wrong for the rest.
#[test]
fn synthesising_a_temperature_keeps_every_indexed_structure_in_step() {
    let mut n = two_temperature_nuclide();
    yamc_nuclide::blend::synthesise_temperature(&mut n, "450").expect("450 is bracketed");

    assert_eq!(n.loaded_temperatures, vec!["294", "450", "600"]);
    assert_eq!(n.reactions.len(), 3);
    assert_eq!(n.fast_xs.len(), 3);
    assert_eq!(n.urr_data.len(), 3);

    let idx = n
        .get_temp_idx("450")
        .expect("the synthesised temperature is loaded");
    assert_eq!(idx, 1, "inserted in numeric order, not appended");
    assert!(!n.reactions[idx].is_empty());
    assert!(!n.fast_xs[idx].energy.is_empty());

    // The file's own ladder is unchanged. Adding "450" to it would make a later
    // 500 K request bracket 450 to 600 rather than 294 to 600, so the answer
    // would depend on the order the queries arrived in.
    assert_eq!(n.available_temperatures, vec!["294", "600"]);

    // The energy map gained the same grid the accelerator holds.
    let grid = n
        .energy
        .as_ref()
        .and_then(|m| m.get("450"))
        .expect("the synthesised temperature has an energy grid");
    assert_eq!(grid.as_slice(), n.fast_xs[idx].energy.as_slice());
}

/// A failure means calling for a temperature that is already loaded rebuilds
/// it, which would double the memory for no change in the numbers.
#[test]
fn synthesising_a_temperature_that_is_already_loaded_does_nothing() {
    let mut n = two_temperature_nuclide();
    yamc_nuclide::blend::synthesise_temperature(&mut n, "294").expect("294 is loaded");
    yamc_nuclide::blend::synthesise_temperature(&mut n, "600K").expect("600 is loaded");
    assert_eq!(n.loaded_temperatures, vec!["294", "600"]);
    assert_eq!(n.fast_xs.len(), 2);
}

/// A failure means a request outside the ladder is answered rather than
/// refused, which is the silent clamp this design exists to avoid.
#[test]
fn a_temperature_outside_the_loaded_range_is_refused_with_the_list() {
    let mut n = two_temperature_nuclide();
    let err = yamc_nuclide::blend::synthesise_temperature(&mut n, "3000")
        .expect_err("3000 is above the ladder");
    let message = err.to_string();
    assert!(message.contains("3000"));
    assert!(message.contains("Available temperatures:"));
    assert!(message.contains("600"));
    // Nothing was inserted on the way to the error.
    assert_eq!(n.loaded_temperatures, vec!["294", "600"]);
}

/// A failure means the synthesised temperature has no URR table, which is
/// entirely silent: the material path reads a missing table as "no unresolved
/// range" and falls back to the unshielded smooth total, so the nuclide loses
/// all its self-shielding while every existing test stays green.
#[test]
fn the_synthesised_temperature_borrows_the_nearer_urr_table_and_never_none() {
    let mut below = two_temperature_nuclide();
    yamc_nuclide::blend::synthesise_temperature(&mut below, "400").expect("400 is bracketed");
    let idx = below.get_temp_idx("400").unwrap();
    let table = below.urr_data[idx]
        .as_ref()
        .expect("a synthesised temperature must carry a URR table, not None");
    assert_eq!(
        table.energy,
        urr_marked(1.0).energy,
        "400 K is nearer 294 K"
    );

    let mut above = two_temperature_nuclide();
    yamc_nuclide::blend::synthesise_temperature(&mut above, "500").expect("500 is bracketed");
    let idx = above.get_temp_idx("500").unwrap();
    let table = above.urr_data[idx]
        .as_ref()
        .expect("a synthesised temperature must carry a URR table, not None");
    assert_eq!(
        table.energy,
        urr_marked(2.0).energy,
        "500 K is nearer 600 K"
    );
}

/// A failure means the synthesised temperature duplicates its energy grid once
/// per reaction, which is the 38 MiB per nuclide that issue #476 removed,
/// coming back through a path that test did not cover.
#[test]
fn the_synthesised_reactions_are_views_of_one_grid_not_copies_of_it() {
    let mut n = two_temperature_nuclide();
    yamc_nuclide::blend::synthesise_temperature(&mut n, "450").expect("450 is bracketed");
    let idx = n.get_temp_idx("450").unwrap();

    let grid = &n.fast_xs[idx].energy;
    let map_grid = n.energy.as_ref().and_then(|m| m.get("450")).unwrap();
    assert!(
        map_grid.shares_with(grid),
        "the energy map holds a second copy of the synthesised grid"
    );
    for (mt, reaction) in &n.reactions[idx] {
        assert!(
            reaction.energy.shares_with(grid),
            "MT {mt} holds its own copy of the synthesised grid"
        );
    }
}

/// A failure means a reaction whose threshold moves between the two
/// temperatures is blended wrongly at the energies between the two thresholds,
/// where one side contributes and the other does not.
#[test]
fn a_threshold_that_moves_between_temperatures_blends_without_a_special_case() {
    let energies = [1.0, 10.0, 100.0, 1000.0];
    let grid = F64Buffer::from_slice(&energies);

    // Same MT, thresholds at different grid points.
    let mut lo = HashMap::new();
    lo.insert(51, reaction_at(51, &energies, 1, 294.0));
    let mut hi = HashMap::new();
    hi.insert(51, reaction_at(51, &energies, 3, 600.0));

    let blended = blend_reactions(&lo, &hi, &grid, 0.5);
    let r = &blended[&51];

    // Below both thresholds: still zero, so the blend has not invented a
    // cross section where neither source has one.
    assert_eq!(r.cross_section_at(1.0), Some(0.0));

    // Between the two thresholds only the lower side contributes, weighted.
    // Asserting the exact half rather than "greater than zero" is what makes a
    // dropped weight visible.
    let want = 0.5 * linear_xs(10.0, 294.0);
    let got = r.cross_section_at(10.0).unwrap();
    assert!(
        (got - want).abs() <= 1e-9 * want,
        "at E=10 only the 294 K side is above threshold: got {got}, want {want}"
    );

    // Above both, it is the full weighted average.
    let want = 0.5 * (linear_xs(1000.0, 294.0) + linear_xs(1000.0, 600.0));
    let got = r.cross_section_at(1000.0).unwrap();
    assert!(
        (got - want).abs() <= 1e-9 * want,
        "at E=1000: got {got}, want {want}"
    );

    // The recovered threshold is the first energy where the blend is non-zero,
    // which is the lower of the two.
    assert_eq!(r.threshold_idx, 1);
}

/// A failure means a channel one temperature carries and the other does not is
/// dropped from the blended reaction set, so a sampler would never select it.
#[test]
fn a_reaction_only_one_temperature_carries_survives_the_blend() {
    let energies = [1.0, 10.0, 100.0];
    let grid = F64Buffer::from_slice(&energies);
    let mut lo = HashMap::new();
    lo.insert(2, reaction_at(2, &energies, 0, 294.0));
    let mut hi = HashMap::new();
    hi.insert(2, reaction_at(2, &energies, 0, 600.0));
    hi.insert(102, reaction_at(102, &energies, 0, 600.0));

    let blended = blend_reactions(&lo, &hi, &grid, 0.25);
    assert!(blended.contains_key(&102), "MT 102 was dropped");
    // Only the upper side has it, so it is weighted by w and not by 1.
    let want = 0.25 * linear_xs(10.0, 600.0);
    let got = blended[&102].cross_section_at(10.0).unwrap();
    assert!((got - want).abs() <= 1e-9 * want, "got {got}, want {want}");

    // Metadata comes from one side rather than being mixed.
    assert_eq!(blended[&2].q_value, -1.5e6);
    assert!(blended[&2].scatter_in_cm);
}
