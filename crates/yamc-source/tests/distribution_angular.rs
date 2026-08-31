mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use yamc_source::distribution::angular::*;

    #[test]
    fn test_monodirectional_distribution() {
        let mut rng = StdRng::seed_from_u64(1);
        let mono = AngularDistribution::new_monodirectional(0.0, 0.0, 1.0);
        let sample = mono.sample(&mut rng);
        assert_eq!(sample, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_monodirectional_different_directions() {
        let mut rng = StdRng::seed_from_u64(1);
        let test_directions = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.5773502691896257, 0.5773502691896257, 0.5773502691896257], // normalized (1,1,1)
        ];

        for &direction in &test_directions {
            let mono =
                AngularDistribution::new_monodirectional(direction[0], direction[1], direction[2]);
            let sample = mono.sample(&mut rng);
            assert_eq!(sample, direction);
        }
    }

    #[test]
    fn test_monodirectional_consistency() {
        let mut rng = StdRng::seed_from_u64(1);
        let mono = AngularDistribution::new_monodirectional(1.0, 0.0, 0.0);

        // Should return the same direction consistently
        for _ in 0..100 {
            let sample = mono.sample(&mut rng);
            assert_eq!(sample, [1.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_isotropic_distribution() {
        let mut rng = StdRng::seed_from_u64(1);
        let iso = AngularDistribution::Isotropic;
        let sample = iso.sample(&mut rng);

        // Check that the vector is normalized
        let mag = (sample[0] * sample[0] + sample[1] * sample[1] + sample[2] * sample[2]).sqrt();
        assert!((mag - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_isotropic_randomness() {
        let mut rng = StdRng::seed_from_u64(1);
        let iso = AngularDistribution::Isotropic;

        // Sample many directions and check they're all normalized and vary
        let mut samples = Vec::new();
        for _ in 0..1000 {
            let sample = iso.sample(&mut rng);

            // Check normalization
            let mag =
                (sample[0] * sample[0] + sample[1] * sample[1] + sample[2] * sample[2]).sqrt();
            assert!((mag - 1.0).abs() < 1e-10);

            samples.push(sample);
        }

        // Check that we have variation (not all the same)
        let first_sample = samples[0];
        let all_same = samples.iter().all(|&s| s == first_sample);
        assert!(!all_same, "Isotropic samples should vary");
    }

    #[test]
    fn test_enum_functionality() {
        let mut rng = StdRng::seed_from_u64(1);
        // Test that we can use the enum directly
        let iso = AngularDistribution::Isotropic;
        let mono = AngularDistribution::new_monodirectional(0.0, 0.0, 1.0);

        // Should be able to call sample on enum variants
        let iso_sample = iso.sample(&mut rng);
        let mono_sample = mono.sample(&mut rng);

        // Isotropic should be normalized
        let mag = (iso_sample[0] * iso_sample[0]
            + iso_sample[1] * iso_sample[1]
            + iso_sample[2] * iso_sample[2])
            .sqrt();
        assert!((mag - 1.0).abs() < 1e-10);

        // Monodirectional should match expected
        assert_eq!(mono_sample, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_send_sync_bounds() {
        // Test that our distribution enum implements Send + Sync for threading
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<AngularDistribution>();
        assert_sync::<AngularDistribution>();
    }
}
