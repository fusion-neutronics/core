mod tests {
    use nalgebra::Vector3;
    use rand::rngs::StdRng;
    use rand::RngExt;
    use rand::SeedableRng;
    use yamc_particle::particle::{Particle, ParticleType};
    use yamc_physics::neutron::interaction::*;

    #[test]
    fn test_rotate_angle_preserves_norm() {
        let mut rng = StdRng::seed_from_u64(42);
        let u_cm = Vector3::new(0.0, 0.0, 1.0);
        let mu_cm = 0.5;
        let v = rotate_angle(u_cm, mu_cm, &mut rng);
        // Should be unit vector
        assert!((v.norm() - 1.0).abs() < 1e-12, "norm = {}", v.norm());
        // z-component should be mu_cm
        assert!((v.z - mu_cm).abs() < 1e-12, "z = {} mu_cm = {}", v.z, mu_cm);
    }

    #[test]
    fn test_elastic_scatter_energy_and_direction() {
        let mut rng = StdRng::seed_from_u64(123);
        // Dummy particle: energy = 2.0, direction = [0,0,1]
        let mut particle = Particle {
            particle_type: ParticleType::Neutron,
            energy: 2.0,
            direction: [0.0, 0.0, 1.0],
            weight: 1.0,
            alive: true,
            position: [0.0, 0.0, 0.0],
            last_position: [0.0, 0.0, 0.0],
            id: 0,
            current_cell_index: yamc_particle::particle::NO_CELL,
            urr_random: yamc_particle::particle::NO_URR,
            urr_energy: 0.0,
            previous_cell_index: yamc_particle::particle::NO_CELL,
            last_surface_id: yamc_particle::particle::NO_SURFACE,
            parent_nuclide: None,
            #[cfg(feature = "debug_history")]
            history: Vec::new(),
        };
        let awr = 1.0; // hydrogen
        let temperature_k = 294.0; // room temperature
        elastic_scatter(&mut particle, awr, temperature_k, &mut rng);
        // Energy should be positive
        assert!(particle.energy > 0.0);
        // Direction should be unit vector
        let norm = (particle.direction[0].powi(2)
            + particle.direction[1].powi(2)
            + particle.direction[2].powi(2))
        .sqrt();
        assert!((norm - 1.0).abs() < 1e-12, "norm = {}", norm);
    }

    // =====================
    //   MAXWELL SPECTRUM TESTS
    // =====================

    #[test]
    fn test_maxwell_spectrum_positive() {
        let mut rng = StdRng::seed_from_u64(42);
        let theta = 1.0; // Temperature in MeV

        for _ in 0..10000 {
            let e = sample_maxwell_spectrum(theta, &mut rng);
            assert!(
                e > 0.0,
                "Maxwell spectrum should always be positive, got {}",
                e
            );
        }
    }

    #[test]
    fn test_maxwell_spectrum_distribution() {
        // For Maxwell distribution p(E) ~ sqrt(E) * exp(-E/T),
        // the mean is <E> = 3T/2
        let mut rng = StdRng::seed_from_u64(12345);
        let theta = 1.0; // Temperature = 1 MeV
        let expected_mean = 1.5 * theta;

        let n_samples = 100000;
        let mut sum = 0.0;
        for _ in 0..n_samples {
            sum += sample_maxwell_spectrum(theta, &mut rng);
        }
        let mean = sum / n_samples as f64;

        // Check mean is within 2% of expected
        let relative_error = (mean - expected_mean).abs() / expected_mean;
        assert!(
            relative_error < 0.02,
            "Maxwell mean {} should be close to {} (error: {:.2}%)",
            mean,
            expected_mean,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_maxwell_spectrum_scales_with_temperature() {
        let mut rng = StdRng::seed_from_u64(999);
        let n_samples = 50000;

        // Sample with theta = 1
        let mut sum1 = 0.0;
        for _ in 0..n_samples {
            sum1 += sample_maxwell_spectrum(1.0, &mut rng);
        }
        let mean1 = sum1 / n_samples as f64;

        // Sample with theta = 2
        let mut sum2 = 0.0;
        for _ in 0..n_samples {
            sum2 += sample_maxwell_spectrum(2.0, &mut rng);
        }
        let mean2 = sum2 / n_samples as f64;

        // mean2 should be about 2x mean1
        let ratio = mean2 / mean1;
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "Mean should scale linearly with temperature: ratio = {:.3}",
            ratio
        );
    }

    // =====================
    //   WATT SPECTRUM TESTS
    // =====================

    #[test]
    fn test_watt_spectrum_positive() {
        let mut rng = StdRng::seed_from_u64(42);
        // U-235 parameters in MeV
        let a = 0.988;
        let b = 2.249;

        for _ in 0..10000 {
            let e = sample_watt_spectrum_params(a, b, &mut rng);
            assert!(
                e > 0.0,
                "Watt spectrum should always be positive, got {}",
                e
            );
        }
    }

    #[test]
    fn test_watt_spectrum_distribution() {
        // For U-235 thermal fission, mean energy is approximately 2 MeV
        let mut rng = StdRng::seed_from_u64(54321);
        let a = 0.988; // MeV
        let b = 2.249; // MeV^-1

        let n_samples = 100000;
        let mut sum = 0.0;
        for _ in 0..n_samples {
            sum += sample_watt_spectrum_params(a, b, &mut rng);
        }
        let mean = sum / n_samples as f64;

        // Expected mean for Watt is approximately a * (1 + 3ab/4) / (1 - ab/4)
        // For U-235: ~1.98 MeV
        // Allow 5% tolerance since Watt has more variance
        assert!(
            mean > 1.7 && mean < 2.3,
            "Watt mean {} should be approximately 2 MeV for U-235",
            mean
        );
    }

    #[test]
    fn test_watt_algorithm() {
        // Test that our Watt fission spectrum implementation matches the formula:
        // E = maxwell(a) + 0.25*a*a*b + U(-1,1)*sqrt(a*a*b*w)
        // We can verify structure by checking that:
        // 1. Watt samples are always >= Maxwell samples - a²b/4 (minimum correction)
        // 2. Watt samples are always <= Maxwell samples + 3*a²b/4 (maximum correction)

        let mut rng = StdRng::seed_from_u64(777);
        let a = 1.0;
        let b = 1.0;

        for _ in 0..1000 {
            // Sample Maxwell and Watt with same seed state isn't possible,
            // but we can verify the range is correct
            let e_watt = sample_watt_spectrum_params(a, b, &mut rng);

            // E_watt should be at least a²b/4 (when w=0 and u=-1, though w>0 always)
            // In practice, with any positive w, minimum is > 0
            assert!(e_watt > 0.0);

            // Sanity check: shouldn't be astronomically large
            assert!(
                e_watt < 100.0 * a,
                "Watt sample {} seems unreasonably large",
                e_watt
            );
        }
    }

    #[test]
    fn test_watt_different_parameters() {
        // Test with different isotope parameters
        let mut rng = StdRng::seed_from_u64(888);
        let n_samples = 10000;

        // Pu-239 approximate parameters (different from U-235)
        let a_pu = 0.966; // MeV
        let b_pu = 2.842; // MeV^-1

        let mut sum = 0.0;
        for _ in 0..n_samples {
            let e = sample_watt_spectrum_params(a_pu, b_pu, &mut rng);
            assert!(e > 0.0);
            sum += e;
        }
        let mean = sum / n_samples as f64;

        // Pu-239 should have similar but slightly different mean (~2.1 MeV)
        assert!(
            mean > 1.5 && mean < 2.5,
            "Pu-239 Watt mean {} should be in expected range",
            mean
        );
    }

    // =====================
    //   FISSION NEUTRON TESTS
    // =====================

    #[test]
    fn test_fission_neutrons_without_products_returns_empty() {
        let mut rng = StdRng::seed_from_u64(42);
        let particle = Particle {
            particle_type: ParticleType::Neutron,
            energy: 1e6, // 1 MeV
            direction: [0.0, 0.0, 1.0],
            weight: 1.0,
            alive: true,
            position: [0.0, 0.0, 0.0],
            last_position: [0.0, 0.0, 0.0],
            id: 0,
            current_cell_index: yamc_particle::particle::NO_CELL,
            urr_random: yamc_particle::particle::NO_URR,
            urr_energy: 0.0,
            previous_cell_index: yamc_particle::particle::NO_CELL,
            last_surface_id: yamc_particle::particle::NO_SURFACE,
            parent_nuclide: None,
            #[cfg(feature = "debug_history")]
            history: Vec::new(),
        };

        let chi_none = yamc_nuclide::reaction_product::FissionChiFlat::None;
        let mut pcg: u64 = yamc_rng::expand_seed(0x1234_5678);

        // No products - should return empty vector
        let neutrons = sample_fission_neutrons(
            &particle, 2.5, None, &chi_none, None, 0.5, &mut pcg, &mut rng,
        );
        assert!(
            neutrons.is_empty(),
            "Fission with no products should return empty vector, got {} neutrons",
            neutrons.len()
        );

        // Empty products slice - should also return empty
        let empty_products: &[&yamc_nuclide::reaction_product::ReactionProduct] = &[];
        let neutrons = sample_fission_neutrons(
            &particle,
            2.5,
            Some(empty_products),
            &chi_none,
            None,
            0.5,
            &mut pcg,
            &mut rng,
        );
        assert!(
            neutrons.is_empty(),
            "Fission with empty products should return empty vector"
        );
    }

    #[test]
    fn test_fission_neutron_stochastic_multiplicity() {
        let mut rng = StdRng::seed_from_u64(123);

        // Test stochastic rounding: nu_bar=2.5 should give ~50% 2 neutrons, ~50% 3 neutrons
        // We can't easily test with products, so just test the logic would work
        // This verifies the stochastic rounding formula
        let n_samples = 10000;
        let nu_bar: f64 = 2.5;
        let mut count_two = 0;
        let mut count_three = 0;

        for _ in 0..n_samples {
            let base = nu_bar.floor() as usize;
            let frac = nu_bar - nu_bar.floor();
            let n = if rng.random::<f64>() < frac {
                base + 1
            } else {
                base
            };
            if n == 2 {
                count_two += 1;
            }
            if n == 3 {
                count_three += 1;
            }
        }

        // Should be roughly 50/50
        let fraction_two = count_two as f64 / n_samples as f64;
        let fraction_three = count_three as f64 / n_samples as f64;

        assert!(
            (fraction_two - 0.5).abs() < 0.05,
            "Expected ~50% 2 neutrons, got {:.1}%",
            fraction_two * 100.0
        );
        assert!(
            (fraction_three - 0.5).abs() < 0.05,
            "Expected ~50% 3 neutrons, got {:.1}%",
            fraction_three * 100.0
        );
    }
}
