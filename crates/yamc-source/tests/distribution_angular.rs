mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use yamc_source::distribution::angular::*;

    #[test]
    fn test_monodirectional_distribution() {
        let mut rng = StdRng::seed_from_u64(1);
        let mono = AngularDistribution::new_monodirectional(0.0, 0.0, 1.0).unwrap();
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
                AngularDistribution::new_monodirectional(direction[0], direction[1], direction[2])
                    .unwrap();
            let sample = mono.sample(&mut rng);
            assert_eq!(sample, direction);
        }
    }

    #[test]
    fn test_monodirectional_consistency() {
        let mut rng = StdRng::seed_from_u64(1);
        let mono = AngularDistribution::new_monodirectional(1.0, 0.0, 0.0).unwrap();

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
        let mono = AngularDistribution::new_monodirectional(0.0, 0.0, 1.0).unwrap();

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

    /// The direction a monodirectional distribution holds.
    fn direction_of(distribution: &AngularDistribution) -> [f64; 3] {
        match distribution {
            AngularDistribution::Monodirectional { direction } => *direction,
            AngularDistribution::Isotropic => panic!("not monodirectional"),
        }
    }

    fn monodirectional(direction: [f64; 3]) -> Result<AngularDistribution, String> {
        AngularDistribution::new_monodirectional(direction[0], direction[1], direction[2])
    }

    /// The zero vector used to panic, and a NaN or infinite component was
    /// kept, to surface mid-transport as lost particles.
    #[test]
    fn a_direction_that_points_nowhere_is_refused() {
        for (direction, says) in [
            ([0.0, 0.0, 0.0], "is the zero vector"),
            ([-0.0, 0.0, -0.0], "is the zero vector"),
            ([f64::NAN, 0.0, 1.0], "has a NaN component (u = NaN)"),
            (
                [0.0, f64::INFINITY, 0.0],
                "has an infinite component (v = inf)",
            ),
            (
                [0.0, 0.0, f64::NEG_INFINITY],
                "has an infinite component (w = -inf)",
            ),
        ] {
            let err = monodirectional(direction).unwrap_err();
            assert!(err.contains(says), "{direction:?}: {err}");
        }
    }

    /// Normalized by the plain sum of squares, `(1e200, 1e200, 0)` overflowed
    /// to the zero direction and `(1e-200, 0, 0)` underflowed and was called
    /// the zero vector.
    #[test]
    fn a_vector_too_large_or_too_small_to_square_is_still_normalized() {
        let half = std::f64::consts::FRAC_1_SQRT_2;
        let third = 1.0 / 3.0_f64.sqrt();
        for (given, expected) in [
            ([1e200, 1e200, 0.0], [half, half, 0.0]),
            ([f64::MAX, -f64::MAX, f64::MAX], [third, -third, third]),
            ([1e-200, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, -3e-320, 3e-320], [0.0, -half, half]),
        ] {
            let direction = direction_of(&monodirectional(given).unwrap());
            for (got, want) in direction.iter().zip(expected) {
                assert!(
                    (got - want).abs() <= 2.0 * f64::EPSILON,
                    "{given:?} -> {direction:?}"
                );
            }
        }
    }

    /// Ordinary vectors are normalized by exactly the arithmetic they always
    /// were, so no existing model's source changes.
    #[test]
    fn an_ordinary_vector_is_normalized_as_before() {
        let third = 1.0 / 3.0_f64.sqrt();
        assert_eq!(
            direction_of(&monodirectional([1.0, 1.0, 1.0]).unwrap()),
            [third; 3]
        );
        assert_eq!(
            direction_of(&monodirectional([3.0, 4.0, 0.0]).unwrap()),
            [0.6, 0.8, 0.0]
        );
        assert_eq!(
            direction_of(&monodirectional([0.0, 0.0, 2.0]).unwrap()),
            [0.0, 0.0, 1.0]
        );
    }

    /// A model file is read through the same check, under either field name,
    /// so a hand-edited zero direction fails the load rather than the run.
    #[test]
    fn a_model_file_direction_that_points_nowhere_fails_to_load() {
        for json in [
            r#"{"Monodirectional":{"direction":[0.0,0.0,0.0]}}"#,
            r#"{"Monodirectional":{"reference_uvw":[0.0,-0.0,0.0]}}"#,
        ] {
            let err = serde_json::from_str::<AngularDistribution>(json)
                .unwrap_err()
                .to_string();
            assert!(err.contains("is the zero vector"), "{json}: {err}");
        }
    }

    /// Any other direction is kept exactly as written, so a model saved and
    /// loaded again is bit for bit the same.
    #[test]
    fn a_saved_direction_loads_back_bit_for_bit() {
        for given in [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.3, -0.2, 0.9],
            [1e200, 1e200, 0.0],
        ] {
            let saved = monodirectional(given).unwrap();
            let json = serde_json::to_string(&saved).unwrap();
            let loaded: AngularDistribution = serde_json::from_str(&json).unwrap();
            assert_eq!(
                direction_of(&loaded).map(f64::to_bits),
                direction_of(&saved).map(f64::to_bits),
                "{json}"
            );
        }
        let legacy: AngularDistribution =
            serde_json::from_str(r#"{"Monodirectional":{"reference_uvw":[0.0,1.0,0.0]}}"#).unwrap();
        assert_eq!(direction_of(&legacy), [0.0, 1.0, 0.0]);
        let isotropic: AngularDistribution = serde_json::from_str(r#""Isotropic""#).unwrap();
        assert!(matches!(isotropic, AngularDistribution::Isotropic));
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
