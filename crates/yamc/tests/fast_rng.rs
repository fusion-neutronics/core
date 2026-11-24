mod tests {
    use yamc::util::fast_rng::*;

    #[test]
    fn test_fast_rng_deterministic() {
        let mut rng1 = FastRng::new(12345);
        let mut rng2 = FastRng::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.random(), rng2.random());
        }
    }

    #[test]
    fn test_fast_rng_range() {
        let mut rng = FastRng::new(42);

        for _ in 0..10000 {
            let val = rng.random();
            assert!(
                (0.0..1.0).contains(&val),
                "Value {} out of range [0, 1)",
                val
            );
        }
    }

    #[test]
    fn test_fast_rng_as_rand_rng() {
        // Test that FastRng works with rand's Rng/RngExt traits
        use rand::RngExt;
        let mut rng = FastRng::new(12345);

        let _: f64 = RngExt::random(&mut rng);
        let _: u32 = RngExt::random(&mut rng);
        let _: bool = RngExt::random(&mut rng);
    }

    #[test]
    fn test_fast_rng_reseed() {
        let mut rng = FastRng::new(12345);
        let first_val = rng.random();

        // Generate more values
        for _ in 0..100 {
            rng.random();
        }

        // Reseed and verify we get the same sequence
        rng.reseed(12345);
        assert_eq!(rng.random(), first_val);
    }
}
