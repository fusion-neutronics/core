mod tests {
    use yamc_tallies::filter::energy::*;

    #[test]
    fn test_energy_filter_creation() {
        let bins = vec![0.0, 1e6, 10e6, 20e6];
        let filter = EnergyFilter::new(bins.clone());
        assert_eq!(filter.bins, bins);
        assert_eq!(filter.num_bins(), 3);
    }

    #[test]
    #[should_panic(expected = "EnergyFilter requires at least 2 bin boundaries")]
    fn test_energy_filter_too_few_bins() {
        EnergyFilter::new(vec![1e6]);
    }

    #[test]
    #[should_panic(expected = "Energy bins must be in strictly ascending order")]
    fn test_energy_filter_non_ascending() {
        EnergyFilter::new(vec![1e6, 10e6, 5e6]);
    }

    #[test]
    #[should_panic(expected = "Energy bins must be in strictly ascending order")]
    fn test_energy_filter_duplicate_bins() {
        EnergyFilter::new(vec![1e6, 10e6, 10e6, 20e6]);
    }

    #[test]
    fn test_energy_filter_get_bin() {
        // Convention:
        // - First bin: [E0, E1] (includes both boundaries)
        // - Other bins: (Ei-1, Ei] (excludes lower, includes upper)
        let filter = EnergyFilter::new(vec![0.0, 1e6, 10e6, 20e6]);

        // First bin [0, 1e6]
        assert_eq!(
            filter.get_bin(0.0),
            Some(0),
            "E=0 should be in first bin [0, 1e6]"
        );
        assert_eq!(filter.get_bin(5e5), Some(0));
        assert_eq!(
            filter.get_bin(1e6),
            Some(0),
            "E=1e6 should be in first bin [0, 1e6]"
        );

        // Second bin (1e6, 10e6]
        assert_eq!(
            filter.get_bin(1.0001e6),
            Some(1),
            "E=1.0001e6 should be in bin (1e6, 10e6]"
        );
        assert_eq!(filter.get_bin(5e6), Some(1));
        assert_eq!(
            filter.get_bin(10e6),
            Some(1),
            "E=10e6 should be in bin (1e6, 10e6]"
        );

        // Third bin (10e6, 20e6]
        assert_eq!(
            filter.get_bin(10.0001e6),
            Some(2),
            "E=10.0001e6 should be in bin (10e6, 20e6]"
        );
        assert_eq!(filter.get_bin(15e6), Some(2));
        assert_eq!(
            filter.get_bin(20e6),
            Some(2),
            "E=20e6 should be in last bin (10e6, 20e6]"
        );

        // Outside range
        assert_eq!(filter.get_bin(-1.0), None, "E=-1 should be outside");
        assert_eq!(filter.get_bin(20.0001e6), None, "E>20e6 should be outside");
        assert_eq!(filter.get_bin(25e6), None, "E=25e6 should be outside");
    }

    #[test]
    fn test_energy_filter_matches() {
        let filter = EnergyFilter::new(vec![1e6, 10e6, 20e6]);

        assert!(
            !filter.matches(0.5e6),
            "E=0.5e6 should be outside [1e6, 20e6]"
        );
        assert!(filter.matches(1e6), "E=1e6 should be in [1e6, 10e6]");
        assert!(filter.matches(5e6), "E=5e6 should be in [1e6, 10e6]");
        assert!(filter.matches(10e6), "E=10e6 should be in [1e6, 10e6]");
        assert!(filter.matches(15e6), "E=15e6 should be in (10e6, 20e6]");
        assert!(filter.matches(20e6), "E=20e6 should be in (10e6, 20e6]");
        assert!(!filter.matches(20.0001e6), "E>20e6 should be outside");
        assert!(!filter.matches(25e6), "E=25e6 should be outside");
    }

    #[test]
    fn test_energy_filter_typical_neutron_bins() {
        let filter = EnergyFilter::new(vec![0.0, 0.625, 100e3, 20e6]);

        assert_eq!(filter.num_bins(), 3);
        assert_eq!(filter.get_bin(0.025), Some(0));
        assert_eq!(filter.get_bin(1000.0), Some(1));
        assert_eq!(filter.get_bin(14.1e6), Some(2));
    }

    #[test]
    fn test_energy_filter_equality() {
        let filter1 = EnergyFilter::new(vec![0.0, 1e6, 10e6]);
        let filter2 = EnergyFilter::new(vec![0.0, 1e6, 10e6]);
        let filter3 = EnergyFilter::new(vec![0.0, 2e6, 10e6]);

        assert_eq!(filter1, filter2);
        assert_ne!(filter1, filter3);
    }

    #[test]
    fn test_energy_filter_logspace_bins() {
        let min_energy: f64 = 0.1;
        let max_energy: f64 = 20e6;
        let num_bins = 50;

        let log_min = min_energy.log10();
        let log_max = max_energy.log10();
        let mut bins = Vec::with_capacity(num_bins);
        for i in 0..num_bins {
            let log_value = log_min + (log_max - log_min) * (i as f64) / ((num_bins - 1) as f64);
            bins.push(10_f64.powf(log_value));
        }

        let filter = EnergyFilter::new(bins.clone());

        assert_eq!(filter.num_bins(), num_bins - 1);
        assert_eq!(filter.bins.len(), num_bins);
        assert!((filter.bins[0] - min_energy).abs() < 1e-10);
        assert!((filter.bins[num_bins - 1] - max_energy).abs() < 1e-3);

        assert_eq!(filter.get_bin(0.05), None);
        assert_eq!(filter.get_bin(0.1), Some(0));
        assert!(filter.get_bin(1e6).is_some());
        assert!(filter.get_bin(10e6).is_some());
        assert!(filter.get_bin(19.999e6).is_some());
        assert_eq!(filter.get_bin(25e6), None);

        for i in 1..bins.len() {
            assert!(filter.bins[i] > filter.bins[i - 1]);
        }

        for i in 0..filter.num_bins() {
            let energy = (filter.bins[i] + filter.bins[i + 1]) / 2.0;
            assert_eq!(filter.get_bin(energy), Some(i));
        }
    }

    #[test]
    fn test_from_group_structure_vitamin_j_175() {
        let filter = EnergyFilter::from_group_structure("VITAMIN-J-175").unwrap();

        // Verify correct number of bins
        assert_eq!(filter.num_bins(), 175);
        assert_eq!(filter.bins.len(), 176);

        // Verify boundary values
        assert_eq!(filter.bins[0], 1e-5);
        assert_eq!(filter.bins[175], 1.964e7);

        // Verify bins are usable for filtering
        assert!(filter.matches(1e6)); // 1 MeV should be in range
        assert!(filter.matches(10e6)); // 10 MeV should be in range
        assert!(!filter.matches(1e-6)); // Below range
        assert!(!filter.matches(20e6)); // Above range
    }

    #[test]
    fn test_from_group_structure_invalid() {
        let result = EnergyFilter::from_group_structure("INVALID-STRUCTURE");
        assert!(result.is_err());

        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("Unknown group structure"));
        assert!(error_msg.contains("INVALID-STRUCTURE"));
        assert!(error_msg.contains("VITAMIN-J-175"));
    }

    #[test]
    fn test_from_group_structure_case_sensitive() {
        // Verify that structure names are case-sensitive
        let result = EnergyFilter::from_group_structure("vitamin-j-175");
        assert!(result.is_err());
    }
}
