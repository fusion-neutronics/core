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

fn text(compressed: &[u8]) -> String {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    String::from_utf8(out).expect("fixture is UTF-8")
}

fn material(compressed: &[u8]) -> Material {
    Material::from_str(&text(compressed)).expect("fixture parses")
}

/// In115's evaluation gives isomer production for (n,n'), (n,2n) and (n,gamma).
/// With decay data for In116's two isomers only, the first two products have no
/// isomer table and the capture level is matched by energy: In116_m1 decays by
/// beta- alone, but its decay file's header states its 127.27 keV.
///
/// The first two are excited levels written as ground, so they are flagged by
/// name: an excited level never reaches the ground state silently.
#[test]
fn routes_are_counted_and_excited_levels_taken_as_ground_are_flagged() {
    let neutron = vec![material(fixture!("n-049_In-115_trimmed.endf.xz"))];
    let decay = vec![
        material(fixture!("dec-049_In_116m1.endf.xz")),
        material(fixture!("dec-049_In_116m2.endf.xz")),
    ];
    let yani_convert::branching::Extracted { rows, stats, .. } =
        yani_convert::branching::extract_branching(
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
    assert_eq!(
        stats.flagged_levels,
        [
            "In115 MT4 -> In115: level 1 at 336.2 keV, no isomer in the decay data, taken as ground",
            "In115 MT16 -> In114: level 1 at 190.3 keV, no isomer in the decay data, taken as ground",
        ]
    );
}

/// A level booked to a nuclide's only isomer while far from it is flagged,
/// and still booked there. With In116_m2 the only In116 isomer in the decay
/// data, In115's capture level at 127.3 keV (In116_m1's energy) matches no
/// isomer by energy or by index and goes to In116_m2, 162.4 keV above it.
/// That is the shape of ENDF/B-VIII.1's Pt194 (n,d) level at 930 keV, booked
/// to Ir193_m1 850 keV below it, which used to be recorded only as a
/// difference.
#[test]
fn a_level_far_from_the_only_isomer_is_flagged_and_still_booked_there() {
    let neutron = vec![material(fixture!("n-049_In-115_trimmed.endf.xz"))];
    let decay = vec![material(fixture!("dec-049_In_116m2.endf.xz"))];
    let yani_convert::branching::Extracted { rows, stats, .. } =
        yani_convert::branching::extract_branching(
            &neutron,
            &decay,
            endf::radionuclide_production::ISOMER_ENERGY_TOLERANCE,
            yani_convert::branching::DEFAULT_LINEARIZE_TOL,
        )
        .expect("branching extracts");
    assert_eq!(stats.level_routes.get("single_isomer"), Some(&1));
    assert_eq!(
        stats.flagged_levels.last().map(String::as_str),
        Some(
            "In115 MT102 -> In116_m2: level 1 at 127.3 keV, taken as the only isomer, \
             -162.4 keV from it"
        ),
        "{:?}",
        stats.flagged_levels
    );

    // Reported, not rebooked: the row and its facts are what they were.
    let capture = rows
        .iter()
        .find(|r| r.reaction == "(n,gamma)" && r.target != "In116")
        .expect("a capture row to the isomer");
    assert_eq!(capture.target, "In116_m2");
    let state = &capture.states[0];
    assert_eq!(state.level_route.label(), "single_isomer");
    let difference = state.level_energy_difference.expect("both energies known");
    assert!((difference / 1.0e3 + 162.4).abs() < 0.05, "{difference}");
}

/// FENDL-3.2d's Al27, the JEFF-3.1.1 file of a 1997 LANL evaluation, is
/// written to the layout before IZAP joined MF=9, so its (n,2n) and (n,alpha)
/// yields carry IZAP = 0 and only MF=8 names the products, Al26 and Na24.
/// Reading the zero as a ZA used to send all four yields to a target called
/// `n0`, where the two (n,2n) states then merged into one curve summing to
/// one. The fixture is the tape's MF=1, the MF=3 sections the partial-sum
/// check reads, and all of MF=8 and MF=9.
#[test]
fn a_zero_izap_is_named_by_mf8() {
    let neutron = vec![material(local_fixture!(
        "n-013_Al_027_fendl-3.2d_trimmed.endf.xz"
    ))];
    let decay = vec![
        material(local_fixture!("dec-013_Al_026m1.endf.xz")),
        material(local_fixture!("dec-011_Na_024m1.endf.xz")),
    ];
    let yani_convert::branching::Extracted { rows, stats, .. } =
        yani_convert::branching::extract_branching(
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

    // Both isomers are matched by energy. On this tape QM - QI equals MF=8's
    // ELFS for both, so the energies are the same whether or not the zero
    // IZAP is joined to MF=8; what this checks is the product naming and the
    // isomer match. The join itself is checked in the `endf` crate, on a
    // subsection whose Q values say nothing about the level.
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

/// The Al27 tape with the MF=8 subsection for the (n,alpha) isomer moved to
/// level 2, so the MF=9 yield for level 1 has IZAP = 0 and nothing in MF=8 to
/// name it. The file then does not say which nuclide the level is, and the
/// state is reported and left out rather than given a product: no branching
/// row, and no reaction product named from a zero ZA.
#[test]
fn a_zero_izap_mf8_does_not_name_is_skipped_with_its_reason() {
    let tape = text(local_fixture!("n-013_Al_027_fendl-3.2d_trimmed.endf.xz"));
    let named = " 1.102400+4 4.722900+5          9          1          0       11291325 8107    3";
    assert!(
        tape.contains(named),
        "the fixture has the Na24 isomer's MF=8 subsection"
    );
    let tape = tape.replace(
        named,
        " 1.102400+4 4.722900+5          9          2          0       11291325 8107    3",
    );
    let al27 = Material::from_str(&tape).expect("the edited tape parses");

    let decay = vec![
        material(local_fixture!("dec-013_Al_026m1.endf.xz")),
        material(local_fixture!("dec-011_Na_024m1.endf.xz")),
    ];
    let yani_convert::branching::Extracted { rows, stats, .. } =
        yani_convert::branching::extract_branching(
            std::slice::from_ref(&al27),
            &decay,
            endf::radionuclide_production::ISOMER_ENERGY_TOLERANCE,
            yani_convert::branching::DEFAULT_LINEARIZE_TOL,
        )
        .expect("branching extracts");

    let found: Vec<(&str, &str)> = rows
        .iter()
        .map(|r| (r.reaction.as_str(), r.target.as_str()))
        .collect();
    assert_eq!(
        found,
        [("(n,2n)", "Al26"), ("(n,2n)", "Al26_m1"), ("(n,a)", "Na24")]
    );
    assert_eq!(
        stats.skipped_states,
        [
            "Al27 MT107 level 1: no product named: IZAP = 0 in MF=9/10, and MF=8 has no \
          single subsection for the level naming one (none, several, or ZAP = 0)"
        ]
    );

    let reaction = endf::Reaction::from_endf(107, &al27).expect("the reaction reads");
    let names: Vec<&str> = reaction.products.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(
        names,
        ["Na24"],
        "only the resolved ground state is a product"
    );
}
