//! Building a temperature the library does not carry, checked without one.
//!
//! Every test here constructs its input in memory. That is not a shortcut, it
//! is the only honest option: nothing in `crates/endf/fixtures` is a
//! multi-temperature Arrow directory, the published fixtures the rest of the
//! suite uses are fetched from a cache no clean checkout has, and a test that
//! reaches for that cache self-skips and reports green. The blend is
//! arithmetic over two temperatures' reactions, and the synthesised lookup is
//! [`FastXSGrid::build`] on the blended ones, so both can be checked exactly,
//! on numbers chosen to make a mistake visible.
//!
//! What this cannot check is the physics question: whether a linear blend of
//! 294 K and 600 K is close to the same nuclide broadened at 450 K by NJOY.
//! That needs a raw ENDF tree and an njoy binary no job provides, so it is
//! deliberately not attempted here rather than attempted and skipped. See the
//! pull request for what was measured offline.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_nuclide::blend::{blend_reactions, union_grid};
use yamc_nuclide::buffer::F64Buffer;
use yamc_nuclide::nuclide::{FastXSGrid, Nuclide};
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::reaction_product::ReactionProduct;
use yamc_nuclide::urr::UrrData;
use yamc_nuclide::ParticleType;

/// A cross section that is linear in energy and in temperature, so the blend
/// has a closed form at every point of the union grid, including the points
/// only one source carries.
fn linear_xs(e: f64, t: f64) -> f64 {
    1.0 + 3.0 * e + 0.5 * t + 0.002 * e * t
}

/// One product of the given particle, with no distributions.
fn emits(particle: ParticleType) -> ReactionProduct {
    ReactionProduct {
        particle,
        emission_mode: "prompt".to_string(),
        decay_rate: 0.0,
        applicability: Vec::new(),
        distribution: Vec::new(),
        product_yield: None,
    }
}

/// A reaction on `energies`, with a threshold at `threshold_idx`.
fn reaction_at(mt: i32, energies: &[f64], threshold_idx: usize, t: f64) -> Arc<Reaction> {
    scaled_reaction_at(mt, energies, threshold_idx, t, 1.0, Vec::new())
}

/// As [`reaction_at`], with the cross section `scale` times [`linear_xs`] and
/// the given products.
fn scaled_reaction_at(
    mt: i32,
    energies: &[f64],
    threshold_idx: usize,
    t: f64,
    scale: f64,
    products: Vec<ReactionProduct>,
) -> Arc<Reaction> {
    let grid = F64Buffer::from_slice(energies);
    let values: Vec<f64> = energies[threshold_idx..]
        .iter()
        .map(|&e| scale * linear_xs(e, t))
        .collect();
    Arc::new(Reaction {
        cross_section: F64Buffer::from_slice(&values),
        threshold_idx,
        energy: grid.tail(threshold_idx),
        mt_number: mt,
        q_value: -1.5e6,
        products,
        scatter_in_cm: true,
        redundant: false,
    })
}

/// Every reaction one temperature carries.
///
/// Elastic, (n,2n) and one inelastic level, so the scattering columns are
/// stored in ascending MT (2, 16, 51) while the inelastic walk visits them in
/// slot order (51 before 16): a lookup that got its column order from anywhere
/// but the builder would show here. Capture with a photon product, so the
/// photon-producing matrix is not empty, and (n,alpha), which emits no neutron
/// and so lands in the absorption-only matrix. Each MT has a different scale
/// so a swapped column is visible.
///
/// Thresholds are given as energies both fixture grids carry, so they do not
/// move between temperatures and the blend has a closed form everywhere.
fn reactions_at(t: f64, energies: &[f64]) -> HashMap<i32, Arc<Reaction>> {
    let neutron = || vec![emits(ParticleType::Neutron)];
    let from = |threshold: f64| {
        energies
            .iter()
            .position(|&e| e == threshold)
            .expect("the threshold is a grid point")
    };
    let mut r = HashMap::new();
    r.insert(2, scaled_reaction_at(2, energies, 0, t, 1.0, neutron()));
    r.insert(
        16,
        scaled_reaction_at(16, energies, from(1000.0), t, 0.5, neutron()),
    );
    r.insert(
        51,
        scaled_reaction_at(51, energies, from(100.0), t, 10.0, neutron()),
    );
    r.insert(
        102,
        scaled_reaction_at(102, energies, 0, t, 0.05, vec![emits(ParticleType::Photon)]),
    );
    r.insert(
        107,
        scaled_reaction_at(107, energies, from(100.0), t, 0.2, Vec::new()),
    );
    r
}

/// The lookup for one loaded temperature, built the way the loader builds it.
fn lookup_for(energy: &F64Buffer, reactions: &HashMap<i32, Arc<Reaction>>) -> FastXSGrid {
    FastXSGrid::build(energy, reactions, None, "Xx1").expect("the fixture builds")
}

/// A two-temperature nuclide with everything the synthesiser touches.
fn two_temperature_nuclide() -> Nuclide {
    let lo_energies = vec![1.0, 10.0, 100.0, 1000.0];
    let hi_energies = vec![1.0, 20.0, 100.0, 500.0, 1000.0];
    let lo_grid = F64Buffer::from_slice(&lo_energies);
    let hi_grid = F64Buffer::from_slice(&hi_energies);
    let lo_reactions = reactions_at(294.0, &lo_energies);
    let hi_reactions = reactions_at(600.0, &hi_energies);
    two_temperature_nuclide_from(lo_grid, hi_grid, lo_reactions, hi_reactions)
}

fn two_temperature_nuclide_from(
    lo_grid: F64Buffer,
    hi_grid: F64Buffer,
    lo_reactions: HashMap<i32, Arc<Reaction>>,
    hi_reactions: HashMap<i32, Arc<Reaction>>,
) -> Nuclide {
    let fast_xs = vec![
        lookup_for(&lo_grid, &lo_reactions),
        lookup_for(&hi_grid, &hi_reactions),
    ];
    let mut energy = HashMap::new();
    energy.insert("294".to_string(), lo_grid);
    energy.insert("600".to_string(), hi_grid);

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
        data_source: None,
        fission_nu: None,
        fast_xs,
        urr_data: vec![Some(urr_marked(1.0)), Some(urr_marked(2.0))],
        urr_present: true,
        fission_photon_release: None,
        covariance: None,
        angular_covariance: None,
        elastic_flat_cache: Default::default(),
        fission_chi_flat_cache: Default::default(),
        delayed_neutron_cache: Default::default(),
        inelastic_angle_flat_cache: Default::default(),
        load_scope: Default::default(),
    }
}

/// The reaction pointers of two lookups name the same `Arc`s.
fn same_arcs(a: &[Arc<Reaction>], b: &[Arc<Reaction>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| Arc::ptr_eq(x, y))
}

/// Every field of two lookups, bit for bit, reaction pointers by identity.
fn assert_same_lookup(got: &FastXSGrid, want: &FastXSGrid) {
    assert_eq!(got.energy.as_slice(), want.energy.as_slice(), "energy");
    assert_eq!(got.log_grid_index, want.log_grid_index, "log_grid_index");
    assert_eq!(
        got.log_e_min.to_bits(),
        want.log_e_min.to_bits(),
        "log_e_min"
    );
    assert_eq!(
        got.inv_log_delta.to_bits(),
        want.inv_log_delta.to_bits(),
        "inv_log_delta"
    );
    assert_eq!(got.xs, want.xs, "summed columns");
    assert_eq!(got.scatter_mt_numbers, want.scatter_mt_numbers);
    assert_eq!(got.scatter_mt_xs.as_slice(), want.scatter_mt_xs.as_slice());
    assert!(same_arcs(
        &got.scatter_mt_reactions,
        &want.scatter_mt_reactions
    ));
    assert_eq!(got.elastic_idx, want.elastic_idx);
    assert_eq!(got.inelastic_walk_order, want.inelastic_walk_order);
    match (&got.reaction_absorption, &want.reaction_absorption) {
        (Some(a), Some(b)) => assert!(Arc::ptr_eq(a, b), "reaction_absorption"),
        (None, None) => {}
        _ => panic!("reaction_absorption differs"),
    }
    assert_eq!(got.fission_mt_numbers, want.fission_mt_numbers);
    assert_eq!(got.fission_mt_xs.as_slice(), want.fission_mt_xs.as_slice());
    assert!(same_arcs(
        &got.fission_mt_reactions,
        &want.fission_mt_reactions
    ));
    assert_eq!(got.has_partial_fission, want.has_partial_fission);
    assert_eq!(got.xs_ngamma.as_slice(), want.xs_ngamma.as_slice());
    assert_eq!(got.photon_prod.as_slice(), want.photon_prod.as_slice());
    assert_eq!(got.photon_rxn_mt_numbers, want.photon_rxn_mt_numbers);
    assert_eq!(got.photon_rxn_xs.as_slice(), want.photon_rxn_xs.as_slice());
    assert!(same_arcs(
        &got.photon_rxn_reactions,
        &want.photon_rxn_reactions
    ));
    assert_eq!(got.absorption_mt_numbers, want.absorption_mt_numbers);
    assert_eq!(
        got.absorption_mt_xs.as_slice(),
        want.absorption_mt_xs.as_slice()
    );
    assert_eq!(
        got.delayed_photon_scaling.as_slice(),
        want.delayed_photon_scaling.as_slice()
    );
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

/// A failure means a synthesised temperature's lookup is not what the loader
/// would build from the same reactions, so there are two routes to a lookup and
/// a blended temperature can disagree with the reactions its samplers read.
#[test]
fn the_synthesised_lookup_is_the_builder_run_on_the_blended_reactions() {
    let mut n = two_temperature_nuclide();
    yamc_nuclide::blend::synthesise_temperature(&mut n, "450").expect("450 is bracketed");
    let idx = n.get_temp_idx("450").unwrap();

    let grid = n.energy.as_ref().and_then(|m| m.get("450")).unwrap();
    let want = FastXSGrid::build(grid, &n.reactions[idx], None, "Xx1").expect("builds");
    assert_same_lookup(&n.fast_xs[idx], &want);
}

/// A failure means the synthesised temperature lays its MT columns out in a
/// different order from a loaded one. `build_inelastic_walk_order` depends on
/// storage order, so a different order changes which channel a given draw
/// selects and desynchronises the CPU from the GPU.
#[test]
fn the_synthesised_lookup_keeps_the_column_and_walk_order_of_a_loaded_one() {
    let mut n = two_temperature_nuclide();
    yamc_nuclide::blend::synthesise_temperature(&mut n, "450").expect("450 is bracketed");
    let (lo, mid, hi) = (&n.fast_xs[0], &n.fast_xs[1], &n.fast_xs[2]);

    assert_eq!(mid.scatter_mt_numbers, vec![2, 16, 51]);
    assert_eq!(mid.inelastic_walk_order, vec![2, 1], "51 walks before 16");
    for neighbour in [lo, hi] {
        assert_eq!(mid.scatter_mt_numbers, neighbour.scatter_mt_numbers);
        assert_eq!(mid.elastic_idx, neighbour.elastic_idx);
        assert_eq!(mid.inelastic_walk_order, neighbour.inelastic_walk_order);
        assert_eq!(mid.fission_mt_numbers, neighbour.fission_mt_numbers);
        assert_eq!(mid.photon_rxn_mt_numbers, neighbour.photon_rxn_mt_numbers);
        assert_eq!(mid.absorption_mt_numbers, neighbour.absorption_mt_numbers);
    }
    assert_eq!(mid.photon_rxn_mt_numbers, vec![102]);
    assert_eq!(mid.absorption_mt_numbers, vec![107]);

    // The lookup's reaction pointers are the blended reactions, so a sampler
    // going through `fast_xs` and one going through `reactions` read the same
    // cross sections.
    for (mt, r) in mid.scatter_mt_numbers.iter().zip(&mid.scatter_mt_reactions) {
        assert!(Arc::ptr_eq(r, &n.reactions[1][mt]), "MT {mt}");
    }
}

/// A failure means the synthesised lookup is not the weighted average it claims
/// to be at the points that need re-interpolation, which is every point one
/// source carries and the other does not.
#[test]
fn a_cross_section_linear_in_energy_and_temperature_blends_exactly() {
    let mut n = two_temperature_nuclide();
    yamc_nuclide::blend::synthesise_temperature(&mut n, "450").expect("450 is bracketed");
    let blended = &n.fast_xs[n.get_temp_idx("450").unwrap()];

    let union = union_grid(
        &[1.0, 10.0, 100.0, 1000.0],
        &[1.0, 20.0, 100.0, 500.0, 1000.0],
    );
    assert_eq!(blended.energy.as_slice(), union.as_slice());

    let close = |got: f64, want: f64| (got - want).abs() <= 1e-9 * want.abs();
    let cols = blended.scatter_mt_numbers.len();
    for (i, &e) in union.iter().enumerate() {
        // Linear in both variables, so the blend of the two endpoints IS the
        // value at the intermediate temperature, at every union point.
        let v = linear_xs(e, 450.0);
        let open = |threshold: f64| if e >= threshold { 1.0 } else { 0.0 };
        let elastic = v;
        let n2n = 0.5 * v * open(1000.0);
        let level = 10.0 * v * open(100.0);
        let capture = 0.05 * v;
        let alpha = 0.2 * v * open(100.0);

        let row = blended.xs[i];
        let total = elastic + n2n + level + capture + alpha;
        assert!(
            close(row[0], total),
            "total at E={e}: got {}, want {total}",
            row[0]
        );
        assert!(close(row[1], capture + alpha), "absorption at E={e}");
        assert!(close(row[2], elastic + n2n + level), "scattering at E={e}");
        assert_eq!(row[3], 0.0, "fission at E={e}");

        // Every scattering column, so a swap between them shows.
        let m = &blended.scatter_mt_xs.as_slice()[i * cols..(i + 1) * cols];
        assert!(close(m[0], elastic), "MT 2 at E={e}");
        assert!(
            close(m[1], n2n) || n2n == 0.0 && m[1] == 0.0,
            "MT 16 at E={e}"
        );
        assert!(
            close(m[2], level) || level == 0.0 && m[2] == 0.0,
            "MT 51 at E={e}"
        );
        assert!(
            close(blended.xs_ngamma.as_slice()[i], capture),
            "MT 102 at E={e}"
        );
    }
}

/// A failure means a blend at an endpoint is not the endpoint, so taking the
/// synthesised path unconditionally would perturb a temperature the data
/// already carries.
#[test]
fn a_blend_at_weight_zero_or_one_reproduces_its_own_source() {
    let n = two_temperature_nuclide();
    let union = F64Buffer::from_slice(&union_grid(
        n.fast_xs[0].energy.as_slice(),
        n.fast_xs[1].energy.as_slice(),
    ));

    for (w, source) in [(0.0, 0usize), (1.0, 1usize)] {
        let blended = blend_reactions(&n.reactions[0], &n.reactions[1], &union, w);
        let lookup = FastXSGrid::build(&union, &blended, None, "Xx1").expect("builds");
        let own = &n.fast_xs[source];
        for (i, &e) in own.energy.as_slice().iter().enumerate() {
            let j = lookup
                .energy
                .as_slice()
                .iter()
                .position(|&x| x == e)
                .expect("the union contains every source point");
            assert_eq!(lookup.xs[j], own.xs[i], "weight {w} at E={e}");
        }
    }
}

/// A failure means the lookup index built for the union grid does not satisfy
/// what `FastXSGrid::lookup` assumes of it, so a search would start past the
/// answer and silently return the wrong cross section.
#[test]
fn the_rebuilt_lookup_index_brackets_every_energy_it_indexes() {
    let mut n = two_temperature_nuclide();
    yamc_nuclide::blend::synthesise_temperature(&mut n, "450").expect("450 is bracketed");
    let blended = &n.fast_xs[n.get_temp_idx("450").unwrap()];

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
    let energies = [1.0, 10.0, 100.0, 1000.0];
    let lo_reactions = reactions_at(294.0, &energies);
    let mut hi_reactions = reactions_at(600.0, &energies);
    hi_reactions.remove(&51);
    let mut n = two_temperature_nuclide_from(
        F64Buffer::from_slice(&energies),
        F64Buffer::from_slice(&energies),
        lo_reactions,
        hi_reactions,
    );

    let err = yamc_nuclide::blend::synthesise_temperature(&mut n, "450")
        .expect_err("differing channels must not blend");
    let message = err.to_string();
    assert!(
        message.contains("51"),
        "{message} does not name the missing MT"
    );
    assert!(
        message.contains("scattering"),
        "{message} does not name the group"
    );
    // Nothing was inserted on the way to the error.
    assert_eq!(n.loaded_temperatures, vec!["294", "600"]);
    assert_eq!(n.fast_xs.len(), 2);
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

    // The energy map gained the same grid the lookup holds.
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
/// per reaction, 38 MiB per nuclide, through a path the loader-side tests do
/// not cover.
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
