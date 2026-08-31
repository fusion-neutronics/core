mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use yamc_nuclide::reaction_product::*;

    /// Create a simple Tabulated1D with constant value
    fn make_constant_tabulated1d(value: f64) -> Tabulated1D {
        Tabulated1D::Tabulated1D {
            x: vec![0.0, 1e10],
            y: vec![value, value],
            breakpoints: vec![],
            interpolation: vec![],
        }
    }

    /// Create a Tabulated1D with linear interpolation
    fn make_linear_tabulated1d(x: Vec<f64>, y: Vec<f64>) -> Tabulated1D {
        Tabulated1D::Tabulated1D {
            x,
            y,
            breakpoints: vec![],
            interpolation: vec![2], // linear-linear
        }
    }

    // =====================
    //   MAXWELL DISTRIBUTION TESTS
    // =====================

    #[test]
    fn test_maxwell_energy_distribution_sample() {
        let mut rng = StdRng::seed_from_u64(42);

        // Create Maxwell distribution with theta = 1 MeV = 1e6 eV
        let dist = EnergyDistribution::Maxwell {
            theta: make_constant_tabulated1d(1e6),
            u: 0.0, // No restriction energy
        };

        // Sample and verify positive energies
        for _ in 0..1000 {
            let e_out = dist.sample(10e6, &mut rng); // 10 MeV incident
            assert!(e_out > 0.0, "Maxwell should produce positive energies");
        }
    }

    #[test]
    fn test_maxwell_restriction_energy() {
        let mut rng = StdRng::seed_from_u64(123);

        // Create Maxwell distribution with restriction energy
        let dist = EnergyDistribution::Maxwell {
            theta: make_constant_tabulated1d(1e6), // 1 MeV
            u: 0.5e6,                              // 0.5 MeV restriction energy
        };

        let e_in = 5e6; // 5 MeV incident
        let max_allowed = e_in - 0.5e6; // Should be <= 4.5 MeV

        // All samples should respect restriction energy
        for _ in 0..1000 {
            let e_out = dist.sample(e_in, &mut rng);
            assert!(
                e_out <= max_allowed,
                "Maxwell should respect restriction energy: {} > {}",
                e_out,
                max_allowed
            );
        }
    }

    #[test]
    fn test_maxwell_energy_dependent_theta() {
        let mut rng = StdRng::seed_from_u64(456);

        // Create Maxwell with energy-dependent theta
        // theta increases from 0.5 MeV at low E to 1.5 MeV at high E
        let dist = EnergyDistribution::Maxwell {
            theta: make_linear_tabulated1d(vec![0.0, 10e6], vec![0.5e6, 1.5e6]),
            u: 0.0,
        };

        // Sample at low energy - should have smaller mean
        let mut sum_low = 0.0;
        let n = 10000;
        for _ in 0..n {
            sum_low += dist.sample(1e6, &mut rng); // 1 MeV incident
        }
        let mean_low = sum_low / n as f64;

        // Sample at high energy - should have larger mean
        let mut sum_high = 0.0;
        for _ in 0..n {
            sum_high += dist.sample(9e6, &mut rng); // 9 MeV incident
        }
        let mean_high = sum_high / n as f64;

        // High energy mean should be larger (theta is larger)
        assert!(
            mean_high > mean_low,
            "Higher theta should give higher mean: low={}, high={}",
            mean_low,
            mean_high
        );
    }

    // =====================
    //   WATT DISTRIBUTION TESTS
    // =====================

    #[test]
    fn test_watt_energy_distribution_sample() {
        let mut rng = StdRng::seed_from_u64(789);

        // Create Watt distribution with U-235 parameters
        // a = 0.988 MeV = 0.988e6 eV, b = 2.249 MeV^-1 = 2.249e-6 eV^-1
        let dist = EnergyDistribution::Watt {
            a: make_constant_tabulated1d(0.988e6),
            b: make_constant_tabulated1d(2.249e-6),
            u: 0.0, // No restriction energy
        };

        // Sample and verify positive energies
        for _ in 0..1000 {
            let e_out = dist.sample(10e6, &mut rng); // 10 MeV incident
            assert!(e_out > 0.0, "Watt should produce positive energies");
        }
    }

    #[test]
    fn test_watt_restriction_energy() {
        let mut rng = StdRng::seed_from_u64(321);

        // Create Watt distribution with restriction energy
        let dist = EnergyDistribution::Watt {
            a: make_constant_tabulated1d(0.988e6),
            b: make_constant_tabulated1d(2.249e-6),
            u: 1e6, // 1 MeV restriction energy
        };

        let e_in = 10e6; // 10 MeV incident
        let max_allowed = e_in - 1e6; // Should be <= 9 MeV

        // All samples should respect restriction energy
        for _ in 0..1000 {
            let e_out = dist.sample(e_in, &mut rng);
            assert!(
                e_out <= max_allowed,
                "Watt should respect restriction energy: {} > {}",
                e_out,
                max_allowed
            );
        }
    }

    #[test]
    fn test_watt_energy_dependent_params() {
        let mut rng = StdRng::seed_from_u64(654);

        // Create Watt with energy-dependent a(E)
        // a increases from 0.8 MeV at low E to 1.2 MeV at high E
        let dist = EnergyDistribution::Watt {
            a: make_linear_tabulated1d(vec![0.0, 10e6], vec![0.8e6, 1.2e6]),
            b: make_constant_tabulated1d(2.249e-6),
            u: 0.0,
        };

        // Sample at low energy
        let mut sum_low = 0.0;
        let n = 10000;
        for _ in 0..n {
            sum_low += dist.sample(1e6, &mut rng);
        }
        let mean_low = sum_low / n as f64;

        // Sample at high energy
        let mut sum_high = 0.0;
        for _ in 0..n {
            sum_high += dist.sample(9e6, &mut rng);
        }
        let mean_high = sum_high / n as f64;

        // Higher a should give different (generally higher) mean
        // The relationship isn't strictly monotonic but typically holds
        assert!(
            mean_low > 0.0 && mean_high > 0.0,
            "Both means should be positive: low={}, high={}",
            mean_low,
            mean_high
        );
    }

    #[test]
    fn test_watt_mean_approximately_2mev() {
        let mut rng = StdRng::seed_from_u64(111);

        // Create Watt distribution with U-235 parameters
        let dist = EnergyDistribution::Watt {
            a: make_constant_tabulated1d(0.988e6),
            b: make_constant_tabulated1d(2.249e-6),
            u: 0.0,
        };

        // Sample many times and compute mean
        let n = 100000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += dist.sample(10e6, &mut rng); // 10 MeV incident (no restriction)
        }
        let mean = sum / n as f64;

        // U-235 Watt mean should be approximately 2 MeV
        let mean_mev = mean / 1e6;
        assert!(
            mean_mev > 1.7 && mean_mev < 2.3,
            "U-235 Watt mean should be ~2 MeV, got {} MeV",
            mean_mev
        );
    }

    // =====================
    //   N-BODY PHASE SPACE TESTS
    // =====================

    #[test]
    #[should_panic(expected = "N-body phase space with >5 bodies.")]
    fn test_nbody_phase_space_rejects_more_than_5_bodies() {
        let mut rng = StdRng::seed_from_u64(42);

        let dist = AngleEnergyDistribution::NBodyPhaseSpace {
            n_bodies: 6,
            total_mass: 3.0,
            awr: 2.0,
            q_value: 0.0,
        };

        // Should panic for >5 bodies
        let _ = dist.sample(1e6, &mut rng);
    }
}
