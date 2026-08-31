mod tests {
    use yamc_nuclide::group_structures::*;

    #[test]
    fn test_vitamin_j_175_structure() {
        let structure = get_group_structure("VITAMIN-J-175").unwrap();

        // Check boundary count (176 boundaries = 175 groups)
        assert_eq!(structure.len(), 176);

        // Check first value
        assert_eq!(structure[0], 1e-5);

        // Check last value
        assert_eq!(structure[175], 1.964e7);

        // Check boundary values to ensure data integrity
        assert_eq!(structure[1], 1.0000e-1); // Second value
        assert_eq!(structure[2], 4.1399e-1); // Third value
        assert_eq!(structure[173], 1.6905e7); // Third to last
        assert_eq!(structure[174], 1.7332e7); // Second to last

        // Verify monotonically increasing (critical for binary search in filters)
        for i in 1..structure.len() {
            assert!(
                structure[i] > structure[i - 1],
                "Boundaries must be strictly increasing at index {}: {} <= {}",
                i,
                structure[i - 1],
                structure[i]
            );
        }
    }

    #[test]
    fn test_unknown_structure() {
        let result = get_group_structure("UNKNOWN-123");
        assert!(result.is_err());

        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("Unknown group structure"));
        assert!(error_msg.contains("UNKNOWN-123"));
        assert!(error_msg.contains("VITAMIN-J-175"));
    }

    #[test]
    fn test_available_structures() {
        let available = available_group_structures();
        assert_eq!(available.len(), 12);
        for name in [
            "XMAS-172",
            "VITAMIN-J-175",
            "SCALE-252",
            "TRIPOLI-315",
            "SHEM-361",
            "LLNL-616",
            "CCFE-709",
            "SCALE-999",
            "UKAEA-1102",
            "ECCO-1968",
            "CCFE-24-PHOTON",
            "VITAMIN-J-42",
        ] {
            assert!(
                available.contains(&name),
                "{name} missing from the registry"
            );
        }
    }

    /// Every name in the registry resolves, and the boundaries behind it are
    /// usable: strictly ascending (the filters binary-search them) and as many
    /// groups as the name claims. A transcription slip in one of the tabulated
    /// arrays shows up here rather than as a quietly wrong spectrum.
    #[test]
    fn test_every_registered_structure_is_well_formed() {
        for name in available_group_structures() {
            let edges = get_group_structure(name)
                .unwrap_or_else(|e| panic!("{name} is listed but does not resolve: {e}"));

            let groups: usize = name
                .rsplit('-')
                .find_map(|part| part.parse().ok())
                .unwrap_or_else(|| panic!("{name} carries no group count"));
            assert_eq!(
                edges.len(),
                groups + 1,
                "{name}: {} boundaries is not {groups} groups",
                edges.len()
            );

            assert!(edges[0] >= 0.0, "{name}: negative lowest boundary");
            for i in 1..edges.len() {
                assert!(
                    edges[i] > edges[i - 1],
                    "{name}: boundaries must be strictly increasing at index {i}: {} <= {}",
                    edges[i - 1],
                    edges[i]
                );
            }
        }
    }

    /// The neutron structures all have to reach past the 14.1 MeV DT peak, or
    /// they cannot hold a fusion source spectrum at all.
    #[test]
    fn test_neutron_structures_span_the_dt_peak() {
        for name in available_group_structures() {
            if name.contains("PHOTON") || name == "VITAMIN-J-42" {
                continue;
            }
            let edges = get_group_structure(name).unwrap();
            assert!(
                *edges.last().unwrap() > 14.1e6,
                "{name} tops out at {} eV, below the DT peak",
                edges.last().unwrap()
            );
        }
    }

    #[test]
    fn test_ukaea_1102_structure() {
        let structure = get_group_structure("UKAEA-1102").unwrap();

        // 1103 boundaries = 1102 groups
        assert_eq!(structure.len(), 1103);

        // Endpoints: 1e-5 eV to 1 GeV, the same span as CCFE-709.
        assert_eq!(structure[0], 1e-5);
        assert_eq!(structure[1102], 1e9);

        // The DT peak sits in the same 14.0 to 14.2 MeV bin as in CCFE-709.
        let i = structure.partition_point(|&e| e <= 14.1e6) - 1;
        assert_eq!((structure[i], structure[i + 1]), (14.0e6, 14.2e6));
    }

    /// The eight structures added alongside UKAEA-1102 carry OpenMC's
    /// boundaries verbatim, so the endpoints are worth pinning: a shifted or
    /// truncated array would still be ascending and still be the right length.
    #[test]
    fn test_added_structure_endpoints() {
        let cases = [
            ("XMAS-172", 173, 1.00001e-5, 1.96403e7),
            ("SCALE-252", 253, 0.0, 2.0e7),
            ("TRIPOLI-315", 316, 1e-5, 1.964e7),
            ("SHEM-361", 362, 0.0, 1.96403e7),
            ("LLNL-616", 617, 1e-5, 2.0e7),
            ("SCALE-999", 1000, 1e-5, 2.0e7),
            ("UKAEA-1102", 1103, 1e-5, 1e9),
            ("ECCO-1968", 1969, 1.00001e-5, 1.964033e7),
        ];
        for (name, len, first, last) in cases {
            let edges = get_group_structure(name).unwrap();
            assert_eq!(edges.len(), len, "{name}: boundary count");
            assert_eq!(edges[0], first, "{name}: lowest boundary");
            assert_eq!(*edges.last().unwrap(), last, "{name}: highest boundary");
        }
    }

    #[test]
    fn test_ccfe_709_structure() {
        let structure = get_group_structure("CCFE-709").unwrap();

        // 710 boundaries = 709 groups
        assert_eq!(structure.len(), 710);

        // Endpoints: 1e-5 eV to 1 GeV
        assert_eq!(structure[0], 1e-5);
        assert_eq!(structure[709], 1e9);

        // Strictly increasing (required for binary search in filters)
        for i in 1..structure.len() {
            assert!(
                structure[i] > structure[i - 1],
                "Boundaries must be strictly increasing at index {}: {} <= {}",
                i,
                structure[i - 1],
                structure[i]
            );
        }
    }

    #[test]
    fn test_case_sensitive() {
        // Verify that lookup is case-sensitive
        let result = get_group_structure("vitamin-j-175");
        assert!(result.is_err());

        let result2 = get_group_structure("VITAMIN-j-175");
        assert!(result2.is_err());
    }
}
