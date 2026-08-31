mod tests {
    use yamc_source::distribution::energy::Discrete;
    use yamc_source::distribution::spatial::Point;
    use yamc_source::source::*;

    /// Helper: wrap a Source in ParticleSource::Neutron for sampling.
    fn neutron(s: Source) -> ParticleSource {
        ParticleSource::Neutron(s)
    }

    #[test]
    fn test_source_construction() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(1);
        let mut s = Source::new();
        s.space = SourceSpatialDistribution::Point(Point::new([1.0, 2.0, 3.0]));
        s.angle = yamc_source::distribution::angular::AngularDistribution::new_monodirectional(
            0.0, 0.0, 1.0,
        );
        s.energy = SourceEnergyDistribution::Discrete(Discrete::new(vec![2e6], vec![1.0]).unwrap());

        let p = neutron(s).sample(&mut rng);
        assert_eq!(p.position, [1.0, 2.0, 3.0]);
        assert_eq!(p.direction, [0.0, 0.0, 1.0]);
        assert_eq!(p.energy, 2e6);
        assert!(p.alive);
    }

    #[test]
    fn test_default_source() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(1);
        let s = neutron(Source::new());
        let p = s.sample(&mut rng);

        // Check that we get valid values
        assert_eq!(p.position, [0.0, 0.0, 0.0]);
        assert_eq!(p.energy, 14.06e6);
        assert!(p.alive);

        // Check that direction is normalized (isotropic sampling)
        let mag = (p.direction[0] * p.direction[0]
            + p.direction[1] * p.direction[1]
            + p.direction[2] * p.direction[2])
            .sqrt();
        assert!((mag - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_source_space_modification() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(1);
        let mut s = Source::new();

        // Test different space values
        let test_positions = [[1.0, 2.0, 3.0], [-5.0, 0.0, 10.0], [0.5, -0.5, 0.0]];

        for &position in &test_positions {
            s.space = SourceSpatialDistribution::Point(Point::new(position));
            let p = neutron(s.clone()).sample(&mut rng);
            assert_eq!(p.position, position);
        }
    }

    #[test]
    fn test_source_energy_modification() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(1);
        let mut s = Source::new();

        // Test different energy values
        let test_energies = [1e6, 2.5e6, 14.1e6, 20e6];

        for &energy in &test_energies {
            s.energy =
                SourceEnergyDistribution::Discrete(Discrete::new(vec![energy], vec![1.0]).unwrap());
            let p = neutron(s.clone()).sample(&mut rng);
            assert_eq!(p.energy, energy);
        }
    }

    #[test]
    fn test_source_angle_switching() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(1);
        let mut s = Source::new();

        // Start with monodirectional
        s.angle = yamc_source::distribution::angular::AngularDistribution::new_monodirectional(
            1.0, 0.0, 0.0,
        );
        let p1 = neutron(s.clone()).sample(&mut rng);
        assert_eq!(p1.direction, [1.0, 0.0, 0.0]);

        // Switch to different monodirectional
        s.angle = yamc_source::distribution::angular::AngularDistribution::new_monodirectional(
            0.0, 1.0, 0.0,
        );
        let p2 = neutron(s.clone()).sample(&mut rng);
        assert_eq!(p2.direction, [0.0, 1.0, 0.0]);

        // Switch to isotropic
        s.angle = yamc_source::distribution::angular::AngularDistribution::Isotropic;
        let ns = neutron(s);
        let p3 = ns.sample(&mut rng);
        let p4 = ns.sample(&mut rng);

        // Both should be normalized
        let p3_mag = (p3.direction[0] * p3.direction[0]
            + p3.direction[1] * p3.direction[1]
            + p3.direction[2] * p3.direction[2])
            .sqrt();
        let p4_mag = (p4.direction[0] * p4.direction[0]
            + p4.direction[1] * p4.direction[1]
            + p4.direction[2] * p4.direction[2])
            .sqrt();
        assert!((p3_mag - 1.0).abs() < 1e-10);
        assert!((p4_mag - 1.0).abs() < 1e-10);

        // Very unlikely to be identical with isotropic sampling
        assert_ne!(p3.direction, p4.direction);
    }

    #[test]
    fn test_source_consistency() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(1);
        let mut s = Source::new();
        s.space = SourceSpatialDistribution::Point(Point::new([1.0, 2.0, 3.0]));
        s.energy = SourceEnergyDistribution::Discrete(Discrete::new(vec![5e6], vec![1.0]).unwrap());
        s.angle = yamc_source::distribution::angular::AngularDistribution::new_monodirectional(
            0.0, 0.0, 1.0,
        );

        let ns = neutron(s);
        // Multiple samples should be consistent for monodirectional
        for _ in 0..10 {
            let p = ns.sample(&mut rng);
            assert_eq!(p.position, [1.0, 2.0, 3.0]);
            assert_eq!(p.energy, 5e6);
            assert_eq!(p.direction, [0.0, 0.0, 1.0]);
            assert!(p.alive);
        }
    }

    #[test]
    fn test_source_isotropic_variation() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(1);
        let s = neutron(Source::new()); // Uses isotropic by default

        // Sample many particles and check for variation
        let mut directions = Vec::new();
        for _ in 0..100 {
            let p = s.sample(&mut rng);

            // All should have same position and energy (deterministic)
            assert_eq!(p.position, [0.0, 0.0, 0.0]);
            assert_eq!(p.energy, 14.06e6);
            assert!(p.alive);

            // Direction should be normalized
            let mag = (p.direction[0] * p.direction[0]
                + p.direction[1] * p.direction[1]
                + p.direction[2] * p.direction[2])
                .sqrt();
            assert!((mag - 1.0).abs() < 1e-10);

            directions.push(p.direction);
        }

        // Should have variation in directions (isotropic)
        let first_direction = directions[0];
        let all_same = directions.iter().all(|&d| d == first_direction);
        assert!(
            !all_same,
            "Isotropic source should produce varying directions"
        );
    }

    #[test]
    fn test_source_discrete_energy() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(1);
        let mut s = Source::new();

        let energy_dist = Discrete::new(vec![1e6, 2e6], vec![0.5, 0.5]).unwrap();
        s.energy = SourceEnergyDistribution::Discrete(energy_dist);

        let ns = neutron(s);
        // Sample and verify energies are either 1e6 or 2e6
        for _ in 0..100 {
            let p = ns.sample(&mut rng);
            assert!(p.energy == 1e6 || p.energy == 2e6);
        }
    }

    #[test]
    fn test_source_discrete_energy_distribution_correctness() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(42);
        let mut s = Source::new();

        // 90% at 14.06 MeV, 10% at 25 MeV (D-T and D-D fusion)
        let energy_dist = Discrete::new(vec![14.06e6, 25e6], vec![0.9, 0.1]).unwrap();
        s.energy = SourceEnergyDistribution::Discrete(energy_dist);

        let ns = neutron(s);
        let mut count_1406 = 0;
        let mut count_25 = 0;
        let n_samples = 10000;

        for _ in 0..n_samples {
            let p = ns.sample(&mut rng);
            if (p.energy - 14.06e6).abs() < 1e3 {
                count_1406 += 1;
            } else if (p.energy - 25e6).abs() < 1e3 {
                count_25 += 1;
            }
        }

        let ratio_1406 = count_1406 as f64 / n_samples as f64;
        let ratio_25 = count_25 as f64 / n_samples as f64;

        assert!(
            (ratio_1406 - 0.9).abs() < 0.02,
            "Expected ~0.9, got {}",
            ratio_1406
        );
        assert!(
            (ratio_25 - 0.1).abs() < 0.02,
            "Expected ~0.1, got {}",
            ratio_25
        );
    }

    #[test]
    fn test_source_multiple_discrete_energies() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(123);
        let mut s = Source::new();

        // Three energies with different probabilities
        let energy_dist = Discrete::new(vec![1e6, 5e6, 10e6], vec![0.5, 0.3, 0.2]).unwrap();
        s.energy = SourceEnergyDistribution::Discrete(energy_dist);

        let ns = neutron(s);
        let mut counts = std::collections::HashMap::new();
        let n_samples = 10000;

        for _ in 0..n_samples {
            let p = ns.sample(&mut rng);
            *counts.entry(p.energy.to_bits()).or_insert(0) += 1;
        }

        // Check approximate probabilities
        let ratio1 = counts[&1e6f64.to_bits()] as f64 / n_samples as f64;
        let ratio5 = counts[&5e6f64.to_bits()] as f64 / n_samples as f64;
        let ratio10 = counts[&10e6f64.to_bits()] as f64 / n_samples as f64;

        assert!(
            (ratio1 - 0.5).abs() < 0.02,
            "Expected ~0.5 for 1 MeV, got {}",
            ratio1
        );
        assert!(
            (ratio5 - 0.3).abs() < 0.02,
            "Expected ~0.3 for 5 MeV, got {}",
            ratio5
        );
        assert!(
            (ratio10 - 0.2).abs() < 0.02,
            "Expected ~0.2 for 10 MeV, got {}",
            ratio10
        );
    }
}
