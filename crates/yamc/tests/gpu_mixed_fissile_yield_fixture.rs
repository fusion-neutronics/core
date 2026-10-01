//! The per-nuclide fission-yield rows of the GPU kernel, on a fissile pair that is
//! in the CI fixture set.
//!
//! `gpu_mixed_fissile_yield.rs` is the original guard for that fix: in a
//! multi-nuclide material the kernel selected the struck nuclide but took
//! `nu_bar` and the delayed fraction `beta` from the material's
//! fission-weighted averages, so a U238 fission was given U235's yield
//! whenever U235 dominated `sigma_f`. It reaches U235 and U238 through
//! `yamc_test_cache::nuclide`, neither of which is a fixture (191 MB and
//! 179 MB), so it has self-skipped on every CI run since it was written and
//! the fix has been unguarded there.
//!
//! What the extractor half of that test actually asserts is nuclide-agnostic:
//! `extract_per_nuclide_inelastic` must give each row of the pool exactly the
//! same fold as `extract_xs_from_nuclide` gives that nuclide on its own, so no
//! row can be carrying a material average. This runs that assertion on
//! Th232 + U240, the two fissionables in the fixture set, so it runs in CI.
//!
//! The pair discriminates better than the original, as it happens: at
//! 14.06 MeV Th232 reads nu_bar 3.9251 / beta 0.00764 against U240's 4.4705 /
//! 0.00582, a 14% spread in the yield where U235 against U238 is 1%.
//!
//! The U235 / U238 file stays as it is: it is the case the issue was filed on,
//! it carries the GPU-against-CPU lockstep run this one does not, and on a box
//! that has those two in the cache it still runs.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

const SOURCE_E: f64 = 14.06e6;

fn load(path: &str) -> yamc_nuclide::nuclide::Nuclide {
    yamc_nuclide::nuclide_loader::load_nuclide(
        std::path::PathBuf::from(path),
        &yamc_nuclide::LoadScope::full(),
    )
    .expect("load")
}

fn cache_pair() -> Option<(String, String)> {
    match (
        yamc_test_cache::nuclide("Th232"),
        yamc_test_cache::nuclide("U240"),
    ) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => {
            eprintln!("skipping: Th232 / U240 not in the cache");
            None
        }
    }
}

#[test]
fn fixture_pair_per_nuclide_yield_matches_each_nuclides_own_fold() {
    let Some(paths) = cache_pair() else { return };
    let th232 = load(&paths.0);
    let u240 = load(&paths.1);
    let mix = [(&th232, 0.5), (&u240, 0.5)];

    let grid: Vec<f64> = yamc_gpu::neutron::xs::union_energy_grid(&mix, "294").expect("union grid");
    let pool = yamc_gpu::extract_per_nuclide_inelastic(&mix, "294", &grid, &grid).expect("pool");
    let n = grid.len();
    assert_eq!(pool.nu_bar.len(), 2 * n);
    assert_eq!(pool.beta_delayed.len(), 2 * n);

    for (row, nuc, name) in [(0usize, &th232, "Th232"), (1usize, &u240, "U240")] {
        let own = yamc_gpu::extract_xs_from_nuclide(nuc, "294").expect("single");
        // The single-nuclide extraction folds on the nuclide's own grid, every
        // point of which is in the union grid: walk the nuclide's grid and
        // binary-search the union for each point.
        let mut checked = 0usize;
        for (j, le) in own.log_energy_grid.iter().enumerate() {
            let e = le.exp();
            let i = grid.partition_point(|&g| g < e * (1.0 - 1e-12));
            if i >= n || (grid[i] - e).abs() > 1e-9 * e {
                continue;
            }
            let (nu_pool, nu_own) = (pool.nu_bar[row * n + i], own.nu_bar[j]);
            let (b_pool, b_own) = (pool.beta_delayed[row * n + i], own.beta_delayed[j]);
            if nu_own > 0.0 {
                checked += 1;
                assert!(
                    (nu_pool - nu_own).abs() <= 1e-12 * nu_own,
                    "{name} at {e:e} eV: pool nu_bar {nu_pool} vs own {nu_own}"
                );
                assert!(
                    (b_pool - b_own).abs() <= 1e-12 * b_own.max(1e-300),
                    "{name} at {e:e} eV: pool beta {b_pool} vs own {b_own}"
                );
            }
        }
        // Th232 compares two orders of magnitude fewer points than U240, and
        // that is the data rather than a fold going missing: its fission is a
        // threshold reaction, so `nu_bar` is zero over most of the union grid
        // and the loop above only counts the points that carry a yield. 291
        // for Th232, 16596 for U240 on this library.
        eprintln!("{name}: {checked} grid points with a yield compared");
        assert!(checked > 100, "{name}: only {checked} grid points compared");
    }

    // Exact equality above cannot tell a correct pool from one where both rows
    // happen to hold the same thing, which is precisely the bug: a shared
    // material average would still equal each nuclide's own fold if the two
    // nuclides agreed. They must not.
    let at = |row: usize, e: f64| {
        let i = grid.iter().position(|&g| g >= e).unwrap();
        (pool.nu_bar[row * n + i], pool.beta_delayed[row * n + i])
    };
    let (nu_th, b_th) = at(0, SOURCE_E);
    let (nu_u, b_u) = at(1, SOURCE_E);
    eprintln!(
        "at 14.06 MeV: Th232 nu_bar {nu_th:.4} beta {b_th:.5}; U240 nu_bar {nu_u:.4} beta {b_u:.5}"
    );
    assert!(nu_th > 0.0 && nu_u > 0.0, "both rows should carry a yield");
    assert!(
        (nu_th - nu_u).abs() > 0.01,
        "Th232 and U240 nu_bar should differ at 14 MeV (got {nu_th} and {nu_u})"
    );
    assert!(
        (b_th - b_u).abs() > 1e-4,
        "Th232 and U240 delayed fractions should differ (got {b_th} and {b_u})"
    );
}
