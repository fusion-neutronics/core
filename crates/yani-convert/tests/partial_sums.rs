//! Whether a reaction's partials reconstruct its total is reported, not fixed:
//! yani shares the MF=3 total out in the partials' proportions either way.

use endf::Material;

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("../../endf/fixtures/", $name))
    };
}

fn material(compressed: &[u8]) -> Material {
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut out).expect("fixture decompresses");
    Material::from_str(&String::from_utf8(out).expect("fixture is UTF-8")).expect("fixture parses")
}

/// In115's ENDF/B-VIII.1 partials add up to their totals, and so do TENDL-2017's
/// Ir191 capture yields (MF=9, exactly one everywhere). TENDL-2017's Ir191
/// (n,2n) partials, to Ir190 and Ir190_m2, sum to 95% of its MF=3 cross
/// section at 14 MeV and to 84% at 20 MeV, which is why FISPACT-II, folding
/// the partials as they stand, makes less Ir190 from the same library than
/// yani does.
#[test]
fn partials_that_do_not_reconstruct_the_total_are_listed() {
    let neutron = vec![
        material(fixture!("n-049_In-115_trimmed.endf.xz")),
        material(fixture!("n-077_Ir_191_trimmed.endf.xz")),
    ];
    let decay = vec![
        material(fixture!("dec-049_In_116m1.endf.xz")),
        material(fixture!("dec-049_In_116m2.endf.xz")),
    ];
    let (_, stats) = yani_convert::branching::extract_branching(
        &neutron,
        &decay,
        endf::radionuclide_production::ISOMER_ENERGY_TOLERANCE,
        yani_convert::branching::DEFAULT_LINEARIZE_TOL,
    )
    .expect("branching extracts");
    let lines = &stats.partial_sum_mismatches;
    assert_eq!(lines.len(), 1, "{lines:?}");
    // The point reported is where the defect is largest against the nonelastic
    // cross section, not where the ratio is worst: the ratio keeps falling to
    // 0.84 at 20 MeV, but by then the reaction is a smaller share of what
    // happens.
    assert_eq!(
        lines[0],
        "Ir191 MT16: MF=10 partial cross sections sum to 0.901 of the MF=3 cross section at \
         1.7000e7 eV, a defect of 7.3% of the nonelastic cross section"
    );
}
