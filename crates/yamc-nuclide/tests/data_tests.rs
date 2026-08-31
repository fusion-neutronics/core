// Integration tests for the data module

use yamc_nuclide::data::{ELEMENT_NUCLIDES, NATURAL_ABUNDANCE, REACTION_MT};

#[test]
fn test_lithium_natural_abundance() {
    let li6 = NATURAL_ABUNDANCE.get("Li6").copied().unwrap_or(0.0);
    let li7 = NATURAL_ABUNDANCE.get("Li7").copied().unwrap_or(0.0);
    let sum = li6 + li7;
    assert!(
        (li6 - 0.0759).abs() < 1e-4,
        "Li6 abundance incorrect: {li6}"
    );
    assert!(
        (li7 - 0.9241).abs() < 1e-4,
        "Li7 abundance incorrect: {li7}"
    );
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "Li6 + Li7 should sum to 1, got {sum}"
    );
}

#[test]
fn test_element_nuclides_li_and_be() {
    let li_nuclides = ELEMENT_NUCLIDES.get("Li").unwrap();
    assert_eq!(li_nuclides, &vec!["Li6", "Li7"]);
    let be_nuclides = ELEMENT_NUCLIDES.get("Be").unwrap();
    assert_eq!(be_nuclides, &vec!["Be9"]);
}

#[test]
fn test_reaction_mt_key_entries() {
    // Check that the authoritative REACTION_MT map contains the key
    // MT numbers used by reaction_names() Python binding.
    let cases = [
        ("(n,elastic)", 2),
        ("(n,fission)", 18),
        ("(n,gamma)", 102),
        ("(n,p)", 103),
        ("(n,a)", 107),
        ("(n,2n)", 16),
        ("(n,Xp)", 203),
        ("(n,Xa)", 207),
        ("heating", 301),
    ];
    for (name, expected_mt) in cases {
        let mt = REACTION_MT.get(name).copied();
        assert_eq!(
            mt,
            Some(expected_mt),
            "Expected reaction '{name}' -> MT {expected_mt}, got {mt:?}"
        );
    }
}

#[test]
fn test_reaction_mt_is_invertible() {
    // Most MT numbers should be unique; the only known intentional duplicates
    // are short-name aliases added alongside the ENDF notation:
    //   MT 3:   "(n,nonelastic)" + "nonelastic"
    //   MT 18:  "(n,fission)"    + "fission"
    let allowed_duplicates: std::collections::HashSet<i32> = [3, 18].iter().copied().collect();

    let mut seen_mts = std::collections::HashMap::new();
    let mut unexpected_duplicates: Vec<i32> = Vec::new();
    for (name, mt) in REACTION_MT.iter() {
        if let Some(prev) = seen_mts.insert(mt, name) {
            if !allowed_duplicates.contains(mt) {
                eprintln!("Unexpected duplicate MT {mt}: '{prev}' and '{name}'");
                unexpected_duplicates.push(*mt);
            }
        }
    }
    assert!(
        unexpected_duplicates.is_empty(),
        "REACTION_MT has unexpected duplicate MT numbers: {unexpected_duplicates:?}"
    );

    // Verify the known aliases are actually present
    assert!(REACTION_MT.get("(n,fission)").copied() == Some(18));
    assert!(REACTION_MT.get("fission").copied() == Some(18));
    assert!(REACTION_MT.get("(n,gamma)").copied() == Some(102));
    assert!(REACTION_MT.get("(n,nonelastic)").copied() == Some(3));
    assert!(REACTION_MT.get("nonelastic").copied() == Some(3));
}
