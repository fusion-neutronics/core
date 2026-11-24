mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use yamc_nuclide::reaction_product::{AngleDistribution, Tabulated};
    use yamc_nuclide::secondary_uncorrelated::*;

    #[test]
    fn test_isotropic_angle_fallback() {
        let angle = AngleDistribution {
            energy: vec![],
            mu: vec![],
        };

        let mut rng = StdRng::seed_from_u64(42);
        let mu = angle.sample(1e6, &mut rng);

        // Should be isotropic: -1 <= mu <= 1
        assert!((-1.0..=1.0).contains(&mu));
    }

    #[test]
    #[should_panic(expected = "Missing energy distribution for uncorrelated angle-energy sampling")]
    fn test_uncorrelated_panics_without_energy_distribution() {
        // Panic if energy distribution is missing
        use yamc_nuclide::reaction_product::TabulatedInterp;
        let angle = AngleDistribution {
            energy: vec![1e5, 1e6],
            mu: vec![
                Tabulated {
                    x: vec![-1.0, 0.0, 1.0],
                    p: vec![0.0, 0.5, 1.0],
                    c: vec![0.0, 0.5, 1.0], // CDF values
                    interp: TabulatedInterp::LinLin,
                },
                Tabulated {
                    x: vec![-1.0, 0.0, 1.0],
                    p: vec![0.0, 0.5, 1.0],
                    c: vec![0.0, 0.5, 1.0], // CDF values
                    interp: TabulatedInterp::LinLin,
                },
            ],
        };

        let mut rng = StdRng::seed_from_u64(42);
        // This should panic because energy distribution is None
        let _ = sample_uncorrelated(5e5, &angle, &None, &mut rng);
    }
}
