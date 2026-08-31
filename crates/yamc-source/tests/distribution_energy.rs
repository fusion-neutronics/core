mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use yamc_source::distribution::energy::*;

    #[test]
    fn test_discrete_construction() {
        let dist = Discrete::new(vec![1.0, 2.0, 3.0], vec![0.2, 0.3, 0.5]).unwrap();
        assert_eq!(dist.energies().len(), 3);
        assert_eq!(dist.energies(), &[1.0, 2.0, 3.0]);
        assert_eq!(dist.probabilities(), &[0.2, 0.3, 0.5]);
    }

    #[test]
    fn test_discrete_validation_empty() {
        // Empty energies
        let result = Discrete::new(vec![], vec![]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Energies vector cannot be empty");
    }

    #[test]
    fn test_discrete_validation_mismatched() {
        // Mismatched lengths
        let result = Discrete::new(vec![1.0], vec![0.5, 0.5]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must have same length"));
    }

    #[test]
    fn test_discrete_validation_negative() {
        // Negative probabilities
        let result = Discrete::new(vec![1.0], vec![-0.5]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Probabilities cannot be negative");
    }

    #[test]
    fn test_discrete_validation_all_zero() {
        // All zero probabilities
        let result = Discrete::new(vec![1.0, 2.0], vec![0.0, 0.0]);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "At least one probability must be non-zero"
        );
    }

    #[test]
    fn test_discrete_sampling_single_energy() {
        let mut rng = StdRng::seed_from_u64(1);
        let dist = Discrete::new(vec![14.06e6], vec![1.0]).unwrap();

        for _ in 0..100 {
            assert_eq!(dist.sample(&mut rng), 14.06e6);
        }
    }

    #[test]
    fn test_discrete_sampling_distribution() {
        let mut rng = StdRng::seed_from_u64(1);
        let dist = Discrete::new(vec![1.0, 2.0, 3.0], vec![0.5, 0.3, 0.2]).unwrap();

        // Sample many times and check distribution
        let mut counts = std::collections::HashMap::new();
        let n_samples = 100000;

        for _ in 0..n_samples {
            let energy = dist.sample(&mut rng);
            *counts.entry(energy.to_bits()).or_insert(0) += 1;
        }

        // Check approximate probabilities (within 1%)
        let tol = 0.01;
        let ratio1 = counts[&1.0f64.to_bits()] as f64 / n_samples as f64;
        let ratio2 = counts[&2.0f64.to_bits()] as f64 / n_samples as f64;
        let ratio3 = counts[&3.0f64.to_bits()] as f64 / n_samples as f64;

        assert!((ratio1 - 0.5).abs() < tol, "Expected ~0.5, got {}", ratio1);
        assert!((ratio2 - 0.3).abs() < tol, "Expected ~0.3, got {}", ratio2);
        assert!((ratio3 - 0.2).abs() < tol, "Expected ~0.2, got {}", ratio3);
    }

    #[test]
    fn test_normalization() {
        let mut rng = StdRng::seed_from_u64(1);
        // Probabilities don't sum to 1 - should auto-normalize
        let dist = Discrete::new(vec![1.0, 2.0], vec![2.0, 8.0]).unwrap();

        let mut counts = std::collections::HashMap::new();
        let n_samples = 10000;

        for _ in 0..n_samples {
            let energy = dist.sample(&mut rng);
            *counts.entry(energy.to_bits()).or_insert(0) += 1;
        }

        // Should still be 20% / 80% split
        let ratio1 = counts[&1.0f64.to_bits()] as f64 / n_samples as f64;
        let ratio2 = counts[&2.0f64.to_bits()] as f64 / n_samples as f64;
        assert!((ratio1 - 0.2).abs() < 0.02, "Expected ~0.2, got {}", ratio1);
        assert!((ratio2 - 0.8).abs() < 0.02, "Expected ~0.8, got {}", ratio2);
    }

    #[test]
    fn test_two_energy_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        // Test D-D and D-T fusion neutrons
        let dist = Discrete::new(vec![2.5e6, 14.06e6], vec![0.1, 0.9]).unwrap();

        let mut count_25 = 0;
        let mut count_1406 = 0;
        let n_samples = 10000;

        for _ in 0..n_samples {
            let energy = dist.sample(&mut rng);
            if (energy - 2.5e6).abs() < 1e3 {
                count_25 += 1;
            } else if (energy - 14.06e6).abs() < 1e3 {
                count_1406 += 1;
            }
        }

        let ratio_25 = count_25 as f64 / n_samples as f64;
        let ratio_1406 = count_1406 as f64 / n_samples as f64;

        assert!(
            (ratio_25 - 0.1).abs() < 0.02,
            "Expected ~0.1, got {}",
            ratio_25
        );
        assert!(
            (ratio_1406 - 0.9).abs() < 0.02,
            "Expected ~0.9, got {}",
            ratio_1406
        );
    }

    #[test]
    fn test_uniform_construction() {
        let dist = Uniform::new(1e6, 20e6).unwrap();
        assert_eq!(dist.a(), 1e6);
        assert_eq!(dist.b(), 20e6);
    }

    #[test]
    fn test_uniform_validation_invalid_bounds() {
        // a >= b should fail
        let result = Uniform::new(10.0, 5.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Lower bound must be less than upper bound"));

        // a == b should also fail
        let result = Uniform::new(5.0, 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_uniform_sampling_range() {
        let mut rng = StdRng::seed_from_u64(1);
        let dist = Uniform::new(1e6, 20e6).unwrap();

        // Sample many times and verify all samples are within bounds
        for _ in 0..1000 {
            let energy = dist.sample(&mut rng);
            assert!(energy >= 1e6, "Sample {} below lower bound", energy);
            assert!(energy <= 20e6, "Sample {} above upper bound", energy);
        }
    }

    #[test]
    fn test_uniform_sampling_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(0.0, 10.0).unwrap();

        // Sample many times and check mean and distribution
        let n_samples = 100000;
        let mut sum = 0.0;
        let mut samples = Vec::new();

        for _ in 0..n_samples {
            let energy = dist.sample(&mut rng);
            sum += energy;
            samples.push(energy);
        }

        // Check mean (should be ~5.0 for uniform [0, 10])
        let mean = sum / n_samples as f64;
        assert!(
            (mean - 5.0).abs() < 0.05,
            "Expected mean ~5.0, got {}",
            mean
        );

        // Check that samples are well distributed across range
        let mut bins = [0; 10];
        for sample in samples {
            let bin = (sample as usize).min(9);
            bins[bin] += 1;
        }

        // Each bin should have roughly 10% of samples
        for (i, &count) in bins.iter().enumerate() {
            let ratio = count as f64 / n_samples as f64;
            assert!(
                (ratio - 0.1).abs() < 0.01,
                "Bin {} has ratio {}, expected ~0.1",
                i,
                ratio
            );
        }
    }

    #[test]
    fn test_uniform_example() {
        let mut rng = StdRng::seed_from_u64(123);
        // Example: Uniform(1 MeV, 20 MeV)
        let dist = Uniform::new(1e6, 20e6).unwrap();

        // Verify all samples are in range
        for _ in 0..1000 {
            let energy = dist.sample(&mut rng);
            assert!((1e6..=20e6).contains(&energy));
        }
    }

    // --- Normal distribution tests ---

    #[test]
    fn test_normal_construction() {
        let dist = Normal::new(14.06e6, 0.1e6).unwrap();
        assert_eq!(dist.mean_val(), 14.06e6);
        assert_eq!(dist.std_dev(), 0.1e6);
    }

    #[test]
    fn test_normal_validation_zero_std_dev() {
        let result = Normal::new(14.06e6, 0.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("positive"));
    }

    #[test]
    fn test_normal_validation_negative_std_dev() {
        let result = Normal::new(14.06e6, -1.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("positive"));
    }

    #[test]
    fn test_normal_sampling_mean() {
        let mut rng = StdRng::seed_from_u64(42);
        let mean = 14.06e6;
        let std_dev = 0.1e6;
        let dist = Normal::new(mean, std_dev).unwrap();

        let n_samples = 100_000;
        let mut sum = 0.0;
        for _ in 0..n_samples {
            sum += dist.sample(&mut rng);
        }
        let sample_mean = sum / n_samples as f64;

        // Mean should be close to 14.06 MeV (within 0.1%)
        let rel_err = (sample_mean - mean).abs() / mean;
        assert!(
            rel_err < 0.001,
            "Sample mean {} differs from expected {} by {:.4}%",
            sample_mean,
            mean,
            rel_err * 100.0
        );
    }

    #[test]
    fn test_normal_sampling_std_dev() {
        let mut rng = StdRng::seed_from_u64(42);
        let mean = 14.06e6;
        let std_dev = 0.1e6;
        let dist = Normal::new(mean, std_dev).unwrap();

        let n_samples = 100_000;
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            samples.push(dist.sample(&mut rng));
        }
        let sample_mean: f64 = samples.iter().sum::<f64>() / n_samples as f64;
        let variance: f64 = samples
            .iter()
            .map(|s| (s - sample_mean).powi(2))
            .sum::<f64>()
            / (n_samples - 1) as f64;
        let sample_std = variance.sqrt();

        // Std dev should be close to 0.1 MeV (within 2%)
        let rel_err = (sample_std - std_dev).abs() / std_dev;
        assert!(
            rel_err < 0.02,
            "Sample std_dev {} differs from expected {} by {:.4}%",
            sample_std,
            std_dev,
            rel_err * 100.0
        );
    }

    #[test]
    fn test_normal_deterministic_with_seed() {
        let dist = Normal::new(14.06e6, 0.1e6).unwrap();

        let mut rng1 = StdRng::seed_from_u64(123);
        let mut rng2 = StdRng::seed_from_u64(123);

        for _ in 0..100 {
            assert_eq!(dist.sample(&mut rng1), dist.sample(&mut rng2));
        }
    }

    // --- fusion_neutron_spectrum() tests ---

    #[test]
    fn test_fusion_spectrum_dt() {
        let dist = fusion_neutron_spectrum(10_000.0, FusionReactants::DT).unwrap();

        // E_0 for T(d,n)alpha is ~14.05 MeV; with thermal shift mean should be above E_0
        assert!(dist.mean_val() > 14.02e6);
        assert!(dist.mean_val() < 14.2e6);

        // Standard deviation should be on order of ~200-400 keV at 10 keV
        assert!(dist.std_dev() > 100e3);
        assert!(dist.std_dev() < 500e3);
    }

    #[test]
    fn test_fusion_spectrum_dd() {
        let dist = fusion_neutron_spectrum(10_000.0, FusionReactants::DD).unwrap();

        // E_0 for D(d,n)3He is ~2.45 MeV; with thermal shift
        assert!(dist.mean_val() > 2.45e6);
        assert!(dist.mean_val() < 2.6e6);

        // Standard deviation should be ~50-100 keV
        assert!(dist.std_dev() > 30e3);
        assert!(dist.std_dev() < 200e3);
    }

    #[test]
    fn test_fusion_spectrum_temp_continuity() {
        // Verify the low-T and high-T formulas produce nearly identical results
        // at the 40 keV switchover point
        let d_lo = fusion_neutron_spectrum(39_990.0, FusionReactants::DT).unwrap();
        let d_hi = fusion_neutron_spectrum(40_010.0, FusionReactants::DT).unwrap();

        let mean_rel = (d_lo.mean_val() - d_hi.mean_val()).abs() / d_lo.mean_val();
        let std_rel = (d_lo.std_dev() - d_hi.std_dev()).abs() / d_lo.std_dev();
        assert!(
            mean_rel < 1e-3,
            "Mean discontinuity at 40 keV: {}",
            mean_rel
        );
        assert!(
            std_rel < 1e-3,
            "Std dev discontinuity at 40 keV: {}",
            std_rel
        );

        // Same check for DD
        let d_lo = fusion_neutron_spectrum(39_990.0, FusionReactants::DD).unwrap();
        let d_hi = fusion_neutron_spectrum(40_010.0, FusionReactants::DD).unwrap();

        let mean_rel = (d_lo.mean_val() - d_hi.mean_val()).abs() / d_lo.mean_val();
        let std_rel = (d_lo.std_dev() - d_hi.std_dev()).abs() / d_lo.std_dev();
        assert!(
            mean_rel < 1e-3,
            "DD mean discontinuity at 40 keV: {}",
            mean_rel
        );
        assert!(
            std_rel < 1e-3,
            "DD std dev discontinuity at 40 keV: {}",
            std_rel
        );
    }

    #[test]
    fn test_fusion_spectrum_high_temp() {
        // At T_i = 80 keV (high-T regime), ensure reasonable results
        let d_dt = fusion_neutron_spectrum(80_000.0, FusionReactants::DT).unwrap();
        assert!(d_dt.mean_val() > 0.0);
        assert!(d_dt.std_dev() > 0.0);

        let d_dd = fusion_neutron_spectrum(80_000.0, FusionReactants::DD).unwrap();
        assert!(d_dd.mean_val() > 0.0);
        assert!(d_dd.std_dev() > 0.0);

        // DT mean at 80 keV should be higher than at 10 keV
        let d_10 = fusion_neutron_spectrum(10_000.0, FusionReactants::DT).unwrap();
        assert!(d_dt.mean_val() > d_10.mean_val());
        assert!(d_dt.std_dev() > d_10.std_dev());
    }

    #[test]
    fn test_fusion_spectrum_low_temp() {
        // At very low temperature, mean should approach E_0
        let dist = fusion_neutron_spectrum(1.0, FusionReactants::DT).unwrap();
        // E_0 for DT is ~14.049 MeV
        let rel_err = (dist.mean_val() - 14.049e6).abs() / 14.049e6;
        assert!(
            rel_err < 1e-3,
            "Expected ~14.049 MeV, got {}",
            dist.mean_val()
        );
        // Width approaches zero at low temperature
        assert!(dist.std_dev() < 5e3);
    }

    #[test]
    fn test_fusion_spectrum_invalid_temp() {
        // Negative temperature
        let result = fusion_neutron_spectrum(-10_000.0, FusionReactants::DT);
        assert!(result.is_err());

        // Temperature above 100 keV
        let result = fusion_neutron_spectrum(101_000.0, FusionReactants::DT);
        assert!(result.is_err());

        // Zero temperature
        let result = fusion_neutron_spectrum(0.0, FusionReactants::DT);
        assert!(result.is_err());
    }

    // --- Histogram -----------------------------------------------------------

    #[test]
    fn test_histogram_construction() {
        let dist = Histogram::new(vec![0.0, 1e6, 20e6], vec![0.3, 0.7]).unwrap();
        assert_eq!(dist.boundaries(), &[0.0, 1e6, 20e6]);
        assert_eq!(dist.probabilities(), &[0.3, 0.7]);
    }

    #[test]
    fn test_histogram_validation_boundary_count() {
        // boundaries must be exactly one longer than probabilities
        assert!(Histogram::new(vec![0.0, 1e6], vec![0.3, 0.7]).is_err());
        assert!(Histogram::new(vec![0.0, 1e6, 20e6], vec![1.0]).is_err());
    }

    #[test]
    fn test_histogram_validation_non_ascending() {
        assert!(Histogram::new(vec![0.0, 20e6, 1e6], vec![0.5, 0.5]).is_err());
        // equal edges (zero-width bin) are rejected too
        assert!(Histogram::new(vec![0.0, 1e6, 1e6], vec![0.5, 0.5]).is_err());
    }

    #[test]
    fn test_histogram_validation_negative_and_all_zero() {
        assert!(Histogram::new(vec![0.0, 1e6, 2e6], vec![-0.5, 1.0]).is_err());
        assert!(Histogram::new(vec![0.0, 1e6, 2e6], vec![0.0, 0.0]).is_err());
        assert!(Histogram::new(vec![], vec![]).is_err());
    }

    #[test]
    fn test_histogram_sampling_in_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Histogram::new(vec![0.0, 1e6, 20e6], vec![0.3, 0.7]).unwrap();
        for _ in 0..10000 {
            let e = dist.sample(&mut rng);
            assert!((0.0..=20e6).contains(&e), "sample {e} out of range");
        }
    }

    #[test]
    fn test_histogram_sampling_mass_per_bin() {
        // Per-bin probability MASS: ~30% of samples land in the first bin even
        // though it is far narrower than the second.
        let mut rng = StdRng::seed_from_u64(7);
        let dist = Histogram::new(vec![0.0, 1e6, 20e6], vec![0.3, 0.7]).unwrap();
        let n = 200_000;
        let low = (0..n).filter(|_| dist.sample(&mut rng) < 1e6).count();
        let frac_low = low as f64 / n as f64;
        assert!(
            (frac_low - 0.3).abs() < 0.01,
            "fraction in low bin {frac_low} not ~0.3"
        );
    }

    #[test]
    fn test_histogram_auto_normalizes() {
        // Unnormalized weights [3, 7] behave identically to [0.3, 0.7].
        let mut rng = StdRng::seed_from_u64(123);
        let dist = Histogram::new(vec![0.0, 1e6, 20e6], vec![3.0, 7.0]).unwrap();
        let n = 200_000;
        let low = (0..n).filter(|_| dist.sample(&mut rng) < 1e6).count();
        let frac_low = low as f64 / n as f64;
        assert!(
            (frac_low - 0.3).abs() < 0.01,
            "auto-normalize failed: low fraction {frac_low}"
        );
    }

    #[test]
    fn a_non_finite_histogram_edge_is_refused() {
        // `w[1] <= w[0]` is false for a NaN, so the ascending check said
        // nothing about one and a NaN edge reached the multigroup collapse,
        // where it produced a NaN group average and an inventory of NaNs with
        // no error anywhere along the way (issue #576).
        assert!(Histogram::new(vec![0.0, f64::NAN, 2e7], vec![0.5, 0.5]).is_err());
        assert!(Histogram::new(vec![0.0, 1e6, f64::INFINITY], vec![0.5, 0.5]).is_err());
        assert!(Histogram::new(vec![f64::NEG_INFINITY, 1e6, 2e7], vec![0.5, 0.5]).is_err());
        // Same hole on the other array: `p < 0.0` is false for a NaN too.
        assert!(Histogram::new(vec![0.0, 1e6, 2e7], vec![0.5, f64::NAN]).is_err());
        assert!(Histogram::new(vec![0.0, 1e6, 2e7], vec![0.5, f64::INFINITY]).is_err());
    }
}
