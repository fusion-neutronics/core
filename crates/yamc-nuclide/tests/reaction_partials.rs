//! Regression for the #111 reaction-type split: `Nuclide::reaction_partials`
//! is the analog (survival-off, single-nuclide) replacement for
//! `sample_reaction_type` + `sample_scattering_constituent`, splitting the
//! collision into `(sigma_e, sigma_a, sigma_i, sigma_f)` so the transport loop
//! can partition elastic/inelastic/fission/absorption from a single shared-PCG
//! `xi2` draw, mirroring the GPU twin.
//!
//! These guard the partition's accounting against drift:
//!   * the four partials sum to the total and match the `collision_xs`
//!     scatter/fission breakdown (the survival-biasing path's XS source), so
//!     the analog `xi2` boundaries carry the same marginal channel
//!     probabilities the legacy two-step selection did (OpenMC parity);
//!   * `sigma_e` equals the elastic (MT 2) column;
//!   * the inelastic constituent selector never returns elastic;
//!   * the selector's cumulative walk visits candidates in the canonical
//!     `INELASTIC_MT_SLOTS` order, which is what makes a shared `xi_mt` pick
//!     the same MT on the CPU and on the GPU (whose per-MT slot sweep uses the
//!     same table, re-exported as yamc-gpu's `MT_SLOTS`).
//!
//! Uses the shared Fe56 Arrow fixture (elastic + inelastic, non-fissile);
//! self-skips when absent so it is inert without the data.

use rand::SeedableRng;
use yamc_nuclide::nuclide::{is_scattering_mt, load_nuclide, FastXSGrid, INELASTIC_MT_SLOTS};

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-18)
}

#[test]
fn reaction_partials_match_collision_xs_breakdown() {
    let fe56 = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc/tests/Fe56.arrow");
    let Ok(nuclide) = load_nuclide(&fe56, &yamc_nuclide::LoadScope::full()) else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };
    let temp = nuclide
        .loaded_temperatures
        .first()
        .expect("Fe56 has a loaded temperature")
        .clone();
    let temp_idx = nuclide.get_temp_idx(&temp).expect("temp idx");
    let fast_grid = &nuclide.fast_xs[temp_idx];
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);

    // Fast region, comfortably above any URR window, so the partials reduce to
    // the smooth grid columns and the comparison is exact.
    for &energy in &[1.0e6, 5.0e6, 1.4e7] {
        let p = nuclide
            .reaction_partials(energy, &temp, Some(0.5), &mut rng)
            .expect("Fe56 has a valid reaction at fast energies");
        let cx = nuclide
            .collision_xs(energy, &temp, Some(0.5), &mut rng)
            .expect("collision_xs at fast energies");

        let sigma_t = p.sigma_e + p.sigma_a + p.sigma_i + p.sigma_f;

        // Channel accounting matches the established collision_xs breakdown.
        assert!(
            rel(sigma_t, cx.total) < 1e-12,
            "E={energy:e}: sigma sum {sigma_t:e} vs collision_xs total {:e}",
            cx.total
        );
        // scatter = elastic + inelastic.
        assert!(
            rel(p.sigma_e + p.sigma_i, cx.scatter) < 1e-12,
            "E={energy:e}: sigma_e+sigma_i {:e} vs collision_xs scatter {:e}",
            p.sigma_e + p.sigma_i,
            cx.scatter
        );
        assert!(
            rel(p.sigma_f, cx.fission) < 1e-12 || (p.sigma_f == 0.0 && cx.fission == 0.0),
            "E={energy:e}: sigma_f {:e} vs collision_xs fission {:e}",
            p.sigma_f,
            cx.fission
        );

        // sigma_e is exactly the elastic (MT 2) column.
        let (i_grid, f) = fast_grid.lookup_grid_index(energy);
        let elastic = fast_grid
            .elastic_idx
            .map(|idx| fast_grid.scatter_xs_interp(i_grid, f, idx))
            .expect("Fe56 has an elastic (MT 2) column");
        assert!(
            rel(p.sigma_e, elastic) < 1e-12,
            "E={energy:e}: sigma_e {:e} vs elastic column {elastic:e}",
            p.sigma_e
        );
        assert!(p.sigma_e > 0.0, "E={energy:e}: elastic must be positive");
        assert!(
            p.sigma_i >= 0.0,
            "E={energy:e}: inelastic must be non-negative"
        );
    }
}

#[test]
fn inelastic_constituent_selector_excludes_elastic() {
    let fe56 = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc/tests/Fe56.arrow");
    let Ok(nuclide) = load_nuclide(&fe56, &yamc_nuclide::LoadScope::full()) else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };
    let temp = nuclide.loaded_temperatures.first().unwrap().clone();

    // 14 MeV: Fe56 has open inelastic + (n,2n) channels, so a non-elastic
    // constituent always exists.
    for &xi_mt in &[0.01_f64, 0.25, 0.5, 0.75, 0.999] {
        let rxn = nuclide
            .sample_inelastic_scatter_reaction(1.4e7, &temp, xi_mt)
            .expect("non-elastic scatter constituent exists at 14 MeV");
        assert_ne!(
            rxn.mt_number, 2,
            "inelastic constituent selector must never return elastic (xi_mt={xi_mt})"
        );
        assert!(
            is_scattering_mt(rxn.mt_number),
            "selected MT {} must be a scattering reaction (xi_mt={xi_mt})",
            rxn.mt_number
        );
    }
}

/// The walk order is the CPU half of the CPU/GPU reaction-selection contract
/// (issue #111): both backends accumulate the non-elastic partials in
/// `INELASTIC_MT_SLOTS` order, so one shared `xi_mt` lands on one MT. Pinned on
/// a synthetic column list so the invariants are visible without the fixture:
/// elastic dropped, tabled MTs in TABLE order (not storage order), untabled MTs
/// appended last, every non-elastic column present exactly once.
#[test]
fn walk_order_follows_the_canonical_slot_table() {
    // Storage order is ascending MT (the on-disk Arrow layout). MT 875 is a
    // level-specific (n,2n) that the slot table does not carry.
    let columns = [2, 5, 16, 51, 52, 91, 875];
    let elastic_idx = Some(0);
    let order = FastXSGrid::build_inelastic_walk_order(&columns, elastic_idx);
    let walked: Vec<i32> = order.iter().map(|&j| columns[j]).collect();
    assert_eq!(
        walked,
        vec![51, 52, 91, 16, 5, 875],
        "walk must visit tabled MTs in INELASTIC_MT_SLOTS order, then untabled ones"
    );
    assert!(
        !order.contains(&0),
        "the elastic column must be excluded (xi2 already made that split)"
    );

    // Tabled entries come out in strictly increasing slot position.
    let slot_of = |mt: i32| INELASTIC_MT_SLOTS.iter().position(|&s| s == mt);
    let slots: Vec<usize> = walked.iter().filter_map(|&mt| slot_of(mt)).collect();
    assert!(
        slots.windows(2).all(|w| w[0] < w[1]),
        "tabled MTs must be visited in ascending slot order, got {slots:?}"
    );

    // No column is dropped or duplicated.
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![1, 2, 3, 4, 5, 6],
        "every non-elastic column must appear exactly once"
    );
}

/// Same contract on the real fixture: Fe56's scatter columns are 2, 5, 16,
/// 51..=89, 91, all of which the slot table carries, so the walk is a pure
/// permutation of the non-elastic columns with nothing appended.
#[test]
fn fe56_walk_order_covers_every_non_elastic_column() {
    let fe56 = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc/tests/Fe56.arrow");
    let Ok(nuclide) = load_nuclide(&fe56, &yamc_nuclide::LoadScope::full()) else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };
    let temp_idx = nuclide
        .get_temp_idx(nuclide.loaded_temperatures.first().unwrap())
        .expect("temp idx");
    let grid = &nuclide.fast_xs[temp_idx];

    assert_eq!(
        grid.inelastic_walk_order.len(),
        grid.scatter_mt_numbers.len() - 1,
        "walk covers every scatter column except elastic"
    );
    let mut seen = grid.inelastic_walk_order.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        grid.inelastic_walk_order.len(),
        "no column visited twice"
    );
    assert!(
        !grid
            .inelastic_walk_order
            .iter()
            .any(|&j| Some(j) == grid.elastic_idx),
        "elastic column must not be in the walk"
    );

    let slots: Vec<usize> = grid
        .inelastic_walk_order
        .iter()
        .map(|&j| {
            INELASTIC_MT_SLOTS
                .iter()
                .position(|&s| s == grid.scatter_mt_numbers[j])
                .unwrap_or_else(|| {
                    panic!(
                        "Fe56 scatter MT {} is missing from INELASTIC_MT_SLOTS",
                        grid.scatter_mt_numbers[j]
                    )
                })
        })
        .collect();
    assert!(
        slots.windows(2).all(|w| w[0] < w[1]),
        "Fe56 walk must be in ascending slot order, got {slots:?}"
    );
}
