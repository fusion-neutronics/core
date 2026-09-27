//! The branching converter says how each excited production level was matched
//! to an isomer, so a rebuild can be checked for levels that changed route.

use endf::Material;

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("../../endf/fixtures/", $name))
    };
}

/// A fixture kept beside these tests rather than in the `endf` crate's set,
/// every one of which must have a golden dump from the Python reader. That
/// reader has the IZAP bug the Al27 test below is about, so a golden made
/// from it would pin the bug.
macro_rules! local_fixture {
    ($name:literal) => {
        include_bytes!(concat!("fixtures/", $name))
    };
}

fn material(compressed: &[u8]) -> Material {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    Material::from_str(&String::from_utf8(out).expect("fixture is UTF-8")).expect("fixture parses")
}

/// In115's evaluation gives isomer production for (n,n'), (n,2n) and (n,gamma).
/// With decay data for In116's two isomers only, the first two products have no
/// isomer table and the capture level is matched by energy: In116_m1 decays by
/// beta- alone, but its decay file's header states its 127.27 keV.
#[test]
fn routes_are_counted_and_nothing_here_is_flagged() {
    let neutron = vec![material(fixture!("n-049_In-115_trimmed.endf.xz"))];
    let decay = vec![
        material(fixture!("dec-049_In_116m1.endf.xz")),
        material(fixture!("dec-049_In_116m2.endf.xz")),
    ];
    let (rows, stats) = yani_convert::branching::extract_branching(
        &neutron,
        &decay,
        endf::radionuclide_production::ISOMER_ENERGY_TOLERANCE,
        yani_convert::branching::DEFAULT_LINEARIZE_TOL,
    )
    .expect("branching extracts");
    assert!(!rows.is_empty());
    assert_eq!(stats.metastable_targets, vec!["In116_m1".to_string()]);
    assert_eq!(stats.level_routes.get("energy"), Some(&1));
    assert_eq!(stats.level_routes.get("no_isomers"), Some(&2));
    assert_eq!(stats.level_routes.values().sum::<usize>(), 3);
    assert!(
        stats.flagged_levels.is_empty(),
        "{:?}",
        stats.flagged_levels
    );
}

/// FENDL-3.2d's Al27 is a JEFF-3.1.1 evaluation written to the layout before
/// IZAP joined MF=9, so its (n,2n) and (n,alpha) yields carry IZAP = 0 and
/// only MF=8 names the products, Al26 and Na24. Reading the zero as a ZA
/// used to send all four yields to a target called `n0`, where the two
/// (n,2n) states then merged into one curve summing to one. The fixture is
/// the tape's MF=1, the MF=3 sections the partial-sum check reads, and all of
/// MF=8 and MF=9.
#[test]
fn a_zero_izap_is_named_by_mf8() {
    let neutron = vec![material(local_fixture!(
        "n-013_Al_027_fendl-3.2d_trimmed.endf.xz"
    ))];
    let decay = vec![
        material(local_fixture!("dec-013_Al_026m1.endf.xz")),
        material(local_fixture!("dec-011_Na_024m1.endf.xz")),
    ];
    let (rows, stats) = yani_convert::branching::extract_branching(
        &neutron,
        &decay,
        endf::radionuclide_production::ISOMER_ENERGY_TOLERANCE,
        yani_convert::branching::DEFAULT_LINEARIZE_TOL,
    )
    .expect("branching extracts");

    let found: Vec<(&str, &str, &str)> = rows
        .iter()
        .map(|r| (r.reaction.as_str(), r.target.as_str(), r.quantity.as_str()))
        .collect();
    assert_eq!(
        found,
        [
            ("(n,2n)", "Al26", "yield"),
            ("(n,2n)", "Al26_m1", "yield"),
            ("(n,a)", "Na24", "yield"),
            ("(n,a)", "Na24_m1", "yield"),
        ]
    );
    assert!(rows.iter().all(|r| r.nuclide == "Al27"));

    // The yields are the tape's, point for point: the isomer's (n,2n) share
    // starts at zero at its own threshold and reaches 0.2511 at 20 MeV.
    let m1 = &rows[1];
    assert_eq!(m1.energy[..2], [1.37833e7, 1.4e7]);
    assert_eq!(m1.values[..2], [0.0, 2.265631e-3]);
    assert_eq!(m1.values[5], 2.511034e-1);

    // Both isomers are matched on MF=8's level energy, which a zero IZAP
    // used to keep from being joined, so the match fell back on QM - QI.
    assert_eq!(stats.level_routes.get("energy"), Some(&2));
    assert_eq!(stats.level_routes.values().sum::<usize>(), 2);
    assert!(
        stats.flagged_levels.is_empty(),
        "{:?}",
        stats.flagged_levels
    );
    assert!(
        stats.skipped_states.is_empty(),
        "{:?}",
        stats.skipped_states
    );
    assert_eq!(stats.merged_duplicate_groups, 0);
}
