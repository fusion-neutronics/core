//! Integration test for the Arrow IPC nuclide reader.

#[cfg(feature = "arrow")]
mod arrow_tests {
    use std::path::Path;

    #[test]
    fn test_load_h1_from_arrow() {
        let h1_path = Path::new("/home/jon/nuclear_data/endf-b8.0-arrow/neutron/H1.arrow");
        if !h1_path.exists() {
            eprintln!("Skipping: H1.arrow not found");
            return;
        }

        let h1 = yamc_nuclide::nuclide_arrow::read_nuclide_from_arrow(
            h1_path,
            &yamc_nuclide::LoadScope::full(),
        )
        .unwrap();

        assert_eq!(h1.name.as_deref(), Some("H1"));
        assert_eq!(h1.atomic_number, Some(1));
        assert_eq!(h1.mass_number, Some(1));
        assert!(h1.atomic_weight_ratio.unwrap() > 0.99);
        assert!(!h1.loaded_temperatures.is_empty());
        assert!(!h1.reactions.is_empty());
        assert!(!h1.fast_xs.is_empty());
        assert!(!h1.fissionable);
        assert!(!h1.urr_present);

        let grid = &h1.fast_xs[0];
        assert!(!grid.energy.is_empty());
        assert_eq!(grid.log_grid_index.len(), 8001);
        assert!(!grid.scatter_mt_xs.is_empty());

        // H1 should have elastic scattering (MT 2)
        assert!(grid.scatter_mt_numbers.contains(&2));

        // XS at thermal should be reasonable
        let (total, abs, scat, _fis) = grid.lookup(0.0253);
        assert!(
            total > 10.0 && total < 100.0,
            "H1 thermal total XS: {total}"
        );
        assert!(scat > 10.0, "H1 thermal scatter XS: {scat}");
        assert!(abs < 1.0, "H1 thermal absorption XS: {abs}");

        println!(
            "H1 Arrow test passed: {} reactions, {} energy points",
            h1.reactions[0].len(),
            grid.energy.len()
        );
    }

    #[test]
    fn test_load_u235_from_arrow() {
        let u235_path = Path::new("/home/jon/nuclear_data/endf-b8.0-arrow/neutron/U235.arrow");
        if !u235_path.exists() {
            eprintln!("Skipping: U235.arrow not found");
            return;
        }

        let u235 = yamc_nuclide::nuclide_arrow::read_nuclide_from_arrow(
            u235_path,
            &yamc_nuclide::LoadScope::full(),
        )
        .unwrap();

        assert_eq!(u235.name.as_deref(), Some("U235"));
        assert_eq!(u235.atomic_number, Some(92));
        assert_eq!(u235.mass_number, Some(235));
        assert!(u235.fissionable);
        assert!(u235.urr_present);
        assert!(u235.fission_nu.is_some());

        let grid = &u235.fast_xs[0];
        assert!(!grid.energy.is_empty());

        // U235 should have both scatter and fission MTs
        assert!(!grid.scatter_mt_xs.is_empty());
        assert!(!grid.fission_mt_xs.is_empty());

        // XS at thermal should be reasonable for U235
        let (total, _abs, _scat, fis) = grid.lookup(0.0253);
        assert!(
            total > 500.0 && total < 1000.0,
            "U235 thermal total XS: {total}"
        );
        assert!(fis > 400.0 && fis < 700.0, "U235 thermal fission XS: {fis}");

        // URR data should be populated
        assert!(u235.urr_data[0].is_some());
        let urr = u235.urr_data[0].as_ref().unwrap();
        assert!(!urr.energy.is_empty());
        assert!(!urr.cdf_values.is_empty());

        // Fission nu
        let nu = u235.fission_nu.as_ref().unwrap();
        let nu_thermal = nu.evaluate(0.0253);
        assert!(
            nu_thermal > 2.0 && nu_thermal < 3.0,
            "U235 thermal nu: {nu_thermal}"
        );

        println!(
            "U235 Arrow test passed: {} reactions, fission_nu thermal={:.4}",
            u235.reactions[0].len(),
            nu_thermal
        );
    }

    /// The log-grid accelerator must agree with the search it replaces.
    ///
    /// `lookup_grid_index` brackets the energy with two entries of
    /// `log_grid_index` and searches only between them, so a table that is
    /// narrowed (issue #482) or out of order can return an index a full search
    /// would not, and the interpolation then runs between the wrong pair of grid
    /// points without panicking. This walks the committed fixtures' own grids
    /// and demands the accelerated answer equal the brute-force one.
    #[test]
    fn log_grid_lookup_agrees_with_a_full_search() {
        let mut checked = 0usize;
        for name in ["Fe56", "U240"] {
            let dir =
                Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../yamc/tests/{name}.arrow"));
            if !dir.exists() {
                eprintln!("Skipping: {name}.arrow fixture not found");
                continue;
            }
            let Ok(nuclide) = yamc_nuclide::nuclide_arrow::read_nuclide_from_arrow(
                &dir,
                &yamc_nuclide::LoadScope::full(),
            ) else {
                eprintln!("Skipping: {name}.arrow is narrower than full scope");
                continue;
            };

            for grid in &nuclide.fast_xs {
                let energies = grid.energy.as_slice();
                if energies.len() < 3 {
                    continue;
                }
                assert!(
                    grid.log_grid_index.len() >= 2,
                    "{name}: log-grid table too short to bracket anything"
                );

                // Every grid point, each midpoint, and both ends, so the sweep
                // covers exact hits, interior gaps and the clamped boundaries.
                let mut probes: Vec<f64> = Vec::with_capacity(energies.len() * 2);
                for w in energies.windows(2) {
                    probes.push(w[0]);
                    probes.push(0.5 * (w[0] + w[1]));
                }
                probes.push(energies[energies.len() - 1]);
                probes.push(energies[0] * 0.5);
                probes.push(energies[energies.len() - 1] * 2.0);

                for e in probes {
                    let (i_fast, _) = grid.lookup_grid_index(e);
                    // What the accelerator is an accelerator for.
                    let want = if e <= energies[0] {
                        0
                    } else if e >= energies[energies.len() - 1] {
                        energies.len() - 1
                    } else {
                        energies
                            .partition_point(|&g| g <= e)
                            .saturating_sub(1)
                            .min(energies.len() - 2)
                    };
                    assert_eq!(
                        i_fast, want,
                        "{name}: log-grid lookup of {e:e} gave index {i_fast}, \
                         a full search gives {want}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 0,
            "no fixture grid was probed (both fixtures missing?)"
        );
        eprintln!("log-grid lookup agreed with a full search on {checked} probes");
    }
}
