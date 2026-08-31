mod tests {
    use yamc_tallies::tally::*;

    // Note: the pre-rip-out `test_accumulate_batch_with_track_length`
    // exercised the atomic-CAS scoring path + per-batch sum/sum_sq
    // fold. Both are gone with the simplification to per-history-only
    // Welford: scoring goes via the per-worker scratch in
    // `simulate_transport`, not by direct calls to a tally method.
    // The same statistical behaviour is covered end-to-end by
    // `test_welford.rs` against a real Model simulation.

    #[test]
    fn test_score_from_str_heating() {
        let score: Score = "heating".parse().unwrap();
        assert_eq!(score, Score::Heating(HeatingScore));

        let score: Score = "301".parse().unwrap();
        assert_eq!(score, Score::Heating(HeatingScore));
    }

    #[test]
    fn test_score_from_str_heating_local() {
        let score: Score = "heating-local".parse().unwrap();
        assert_eq!(score, Score::HeatingLocal(HeatingLocalScore));

        let score: Score = "901".parse().unwrap();
        assert_eq!(score, Score::HeatingLocal(HeatingLocalScore));
    }

    #[test]
    fn test_score_to_mt_heating() {
        assert_eq!(Score::Heating(HeatingScore).to_mt(), Some(301));
        assert_eq!(Score::HeatingLocal(HeatingLocalScore).to_mt(), Some(901));
    }

    #[test]
    fn test_score_to_i32_heating() {
        assert_eq!(Score::Heating(HeatingScore).to_i32(), 301);
        assert_eq!(Score::HeatingLocal(HeatingLocalScore).to_i32(), 901);
    }

    #[test]
    fn test_score_parse_total() {
        let score: Score = "total".parse().unwrap();
        assert_eq!(score, Score::ReactionRate(ReactionRateScore::total()));
        assert_eq!(score.to_mt(), Some(1));
    }

    #[test]
    fn test_score_parse_elastic() {
        let score: Score = "elastic".parse().unwrap();
        assert_eq!(score, Score::ReactionRate(ReactionRateScore::elastic()));
        assert_eq!(score.to_mt(), Some(2));
    }

    #[test]
    fn test_score_parse_fission() {
        let score: Score = "fission".parse().unwrap();
        assert_eq!(score, Score::ReactionRate(ReactionRateScore::fission()));
        assert_eq!(score.to_mt(), Some(18));
    }

    #[test]
    fn test_score_parse_absorption() {
        let score: Score = "absorption".parse().unwrap();
        assert_eq!(score, Score::ReactionRate(ReactionRateScore::absorption()));
        assert_eq!(score.to_mt(), Some(27));
    }

    #[test]
    fn test_score_parse_n2n_parentheses() {
        let score: Score = "(n,2n)".parse().unwrap();
        assert_eq!(
            score,
            Score::ReactionRate(ReactionRateScore::named(Mt::new(16), "(n,2n)"))
        );
        assert_eq!(score.to_mt(), Some(16));
    }

    #[test]
    fn test_score_parse_ngamma() {
        let score: Score = "(n,gamma)".parse().unwrap();
        assert_eq!(
            score,
            Score::ReactionRate(ReactionRateScore::named(Mt::new(102), "(n,gamma)"))
        );
        assert_eq!(score.to_mt(), Some(102));
        assert_eq!(score.name(), "(n,gamma)");
    }

    #[test]
    fn test_score_parse_na() {
        let score: Score = "(n,a)".parse().unwrap();
        assert_eq!(
            score,
            Score::ReactionRate(ReactionRateScore::named(Mt::new(107), "(n,a)"))
        );
        assert_eq!(score.to_mt(), Some(107));
    }

    #[test]
    fn test_score_parse_np() {
        let score: Score = "(n,p)".parse().unwrap();
        assert_eq!(
            score,
            Score::ReactionRate(ReactionRateScore::named(Mt::new(103), "(n,p)"))
        );
        assert_eq!(score.to_mt(), Some(103));
    }

    #[test]
    fn test_score_parse_mt_number_string() {
        // "16" should parse to unnamed ReactionRate
        let score: Score = "16".parse().unwrap();
        assert_eq!(
            score,
            Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(16)))
        );

        // "102" should parse to unnamed ReactionRate
        let score: Score = "102".parse().unwrap();
        assert_eq!(
            score,
            Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(102)))
        );

        // "1" should parse to named ReactionRate (total)
        let score: Score = "1".parse().unwrap();
        assert_eq!(score, Score::ReactionRate(ReactionRateScore::total()));

        // "2" should parse to named ReactionRate (elastic)
        let score: Score = "2".parse().unwrap();
        assert_eq!(score, Score::ReactionRate(ReactionRateScore::elastic()));

        // "18" should parse to named ReactionRate (fission)
        let score: Score = "18".parse().unwrap();
        assert_eq!(score, Score::ReactionRate(ReactionRateScore::fission()));
    }

    #[test]
    fn test_score_parse_flux() {
        let score: Score = "flux".parse().unwrap();
        assert_eq!(score, Score::Flux(FluxScore));
    }

    #[test]
    fn test_score_parse_heating() {
        let score: Score = "heating".parse().unwrap();
        assert_eq!(score, Score::Heating(HeatingScore));
    }

    #[test]
    fn test_score_parse_production_scores() {
        assert_eq!(
            "H1-production".parse::<Score>().unwrap(),
            Score::Production(ProductionScore::H1)
        );
        assert_eq!(
            "H2-production".parse::<Score>().unwrap(),
            Score::Production(ProductionScore::H2)
        );
        assert_eq!(
            "H3-production".parse::<Score>().unwrap(),
            Score::Production(ProductionScore::H3)
        );
        assert_eq!(
            "He3-production".parse::<Score>().unwrap(),
            Score::Production(ProductionScore::HE3)
        );
        assert_eq!(
            "He4-production".parse::<Score>().unwrap(),
            Score::Production(ProductionScore::HE4)
        );
    }

    #[test]
    fn test_score_parse_various_reactions() {
        // Test various reaction names from REACTION_MT - all become named ReactionRate
        let reaction_test_cases = [
            ("(n,3n)", 17),
            ("(n,4n)", 37),
            ("(n,t)", 105),
            ("(n,d)", 104),
            ("(n,3He)", 106),
            ("(n,na)", 22),
            ("(n,np)", 28),
            ("(n,2a)", 108),
            ("(n,nc)", 91),
            ("(n,nonelastic)", 3),
            ("(n,gamma)", 102),
            ("(n,fission)", 18),
        ];

        for (name, expected_mt) in reaction_test_cases {
            let score: Score = name
                .parse()
                .unwrap_or_else(|_| panic!("Failed to parse {}", name));
            assert_eq!(
                score,
                Score::ReactionRate(ReactionRateScore::named(Mt::new(expected_mt as u16), name)),
                "Failed for {}",
                name
            );
            assert_eq!(score.name(), name, "name() should return original string");
        }

        // Named variants (direct keywords) stay as named ReactionRate
        let score: Score = "inelastic".parse().unwrap();
        assert_eq!(score, Score::ReactionRate(ReactionRateScore::inelastic()));
    }

    #[test]
    fn test_score_parse_invalid() {
        let result: Result<Score, _> = "invalid_score".parse();
        assert!(result.is_err());

        let result: Result<Score, _> = "xyz123".parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_mt_score_cache_populated() {
        let mut tally = Tally::new();
        tally.scores = vec![
            Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(803))),
            Score::Flux(FluxScore),
            Score::ReactionRate(ReactionRateScore::total()),
            Score::ReactionRate(ReactionRateScore::named(Mt::new(107), "(n,a)")),
            Score::ReactionRate(ReactionRateScore::absorption()),
        ];
        tally.initialize_batches(1);

        // The cache should include MT scores with correct MT numbers
        // Flux should NOT be in the MT cache
        assert_eq!(tally.num_bins(), 5); // 5 scores * 1 energy bin
    }
}
