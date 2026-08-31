mod tests {

    use yamc_nuclide::Reaction;

    #[test]
    fn test_cross_section_at() {
        let reaction = Reaction {
            cross_section: vec![1.0, 2.0, 3.0, 4.0].into(),
            threshold_idx: 0,
            energy: vec![0.5, 1.0, 2.0, 5.0].into(),
            mt_number: 102,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        };

        // Below grid
        assert_eq!(reaction.cross_section_at(0.1), Some(1.0));
        // Exact match
        assert_eq!(reaction.cross_section_at(1.0), Some(2.0));
        // Between grid points (linear interpolation: 2.0 + 0.5*(3.0-2.0) = 2.5)
        assert_eq!(reaction.cross_section_at(1.5), Some(2.5));
        // Above grid
        assert_eq!(reaction.cross_section_at(10.0), Some(4.0));
    }

    /// Q-values are parsed straight from the Arrow `reactions` batch (see
    /// `nuclide_arrow.rs`, the `Q_value` column). This asserts the parsed value
    /// against the known physics for the Li6 (n,t) reaction (MT=105), whose
    /// Q-value is +4.78 MeV. Catches regressions in the Arrow Q_value parsing.
    #[test]
    fn li6_nt_q_value_from_arrow() {
        use yamc_nuclide::nuclide::load_nuclide;

        // Shared Arrow fixtures live in `crates/yamc/tests/`; resolve relative
        // to this crate's manifest dir so the path works from any crate runner.
        let li6 = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc/tests/Li6.arrow");
        let nuclide = load_nuclide(&li6, &yamc_nuclide::LoadScope::full()).expect("load Li6.arrow");
        let temp = nuclide
            .loaded_temperatures
            .first()
            .expect("Li6 should have at least one loaded temperature")
            .clone();
        let reactions = nuclide
            .reactions_for_temp(&temp)
            .expect("Li6 should have reactions at the loaded temperature");

        // MT=105 is the (n,t) reaction: Li6 + n -> H3 + He4.
        let nt = reactions
            .get(&105)
            .expect("Li6 should have an MT=105 (n,t) reaction");

        // Known Q-value for Li6(n,t)He4 is +4.78 MeV. The parsed value is in eV;
        // assert within a few keV to allow for evaluation-library differences.
        let expected_ev = 4_783_649.0;
        let tolerance_ev = 5_000.0; // 5 keV
        assert!(
            (nt.q_value - expected_ev).abs() < tolerance_ev,
            "Li6 MT=105 Q-value parsed as {} eV, expected ~{} eV (within {} eV)",
            nt.q_value,
            expected_ev,
            tolerance_ev
        );
    }
}
